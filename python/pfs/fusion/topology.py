"""
F7 — Détection topologique de patterns, avec test de significativité.

Statut (audit du 14 août 2026) — bibliothèque, NON branchée
-----------------------------------------------------------
Importée par aucun code applicatif ni script livré : seul
``tests/test_app_and_data.py`` l'exerce. Pourquoi : ``detect_patterns``
attend des nuages de points (vecteurs d'action par adversaire) que personne
ne construit encore — la revue de session produit des stats agrégées, pas
des trajectoires par joueur, et aucune UI n'affiche de « pattern ».
Ce qui existe déjà : H₀ exact (Kruskal + union-find), test de permutation
avec Bonferroni, H₁ optionnel via ripser — testés. Accroche naturelle si on
branche : ``pfs/analysis/session_review.py:review_hands`` (construire les
nuages par adversaire depuis les mains parsées), exposé par la route
``review`` de ``pfs/app/server.py``. Règle du projet : aucun module
fusionné sans route + UI + test de bout en bout.

Sources
-------
- Edelsbrunner, Letscher & Zomorodian (2002), *Topological Persistence and
  Simplification*, Discrete Comput. Geom. 28
- Carlsson, G. (2009), *Topology and Data*, Bull. AMS 46(2)
- Chazal, F. & Michel, B. (2021), *An Introduction to Topological Data Analysis*
  — pour les garanties statistiques

⚠️ AVERTISSEMENT — LIRE AVANT D'UTILISER
────────────────────────────────────────
Le prompt v1 présentait la TDA comme « révélant une structure cachée ». C'est
enthousiaste et **statistiquement dangereux** : une boucle H₁ persistante
apparaît parfaitement dans du bruit pur. Sans test de significativité, la TDA
est une machine à fabriquer de faux exploits — et un faux exploit coûte de
l'argent réel.

D'où le protocole appliqué ici, non négociable :
  1. calculer les diagrammes de persistance sur les données réelles ;
  2. générer B jeux nuls par **permutation des labels d'action** (détruit la
     structure, préserve les marginales) ;
  3. p-value = proportion des nuls dont la persistance maximale égale ou
     dépasse la persistance réelle ;
  4. ne déclarer un pattern que si p < 0,01, **avec correction de Bonferroni**
     sur le nombre d'adversaires testés.

Implémentation : H₀ (composantes connexes) est calculé exactement par un
arbre couvrant minimal (algorithme de Kruskal + union-find) — c'est la
formulation standard de la persistance en dimension 0 sur un complexe de
Vietoris-Rips. H₁ (boucles) nécessite `ripser` ; il est détecté et utilisé
s'il est installé, sinon le module reste pleinement fonctionnel en H₀.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "PersistencePoint",
    "TopologyResult",
    "h0_persistence",
    "detect_patterns",
    "HAS_RIPSER",
]

F64 = npt.NDArray[np.float64]

try:  # pragma: no cover - dépend de l'environnement
    from ripser import ripser as _ripser  # type: ignore
    HAS_RIPSER = True
except Exception:  # pragma: no cover
    _ripser = None
    HAS_RIPSER = False


class TopologyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PersistencePoint:
    dimension: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return self.death - self.birth


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _standardise(x: F64) -> F64:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise TopologyError("attendu une matrice N×D.")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (x - mu) / sd


def h0_persistence(points: F64, max_points: int = 800, seed: int | None = 0) -> list[PersistencePoint]:
    r"""Persistance H₀ exacte, par arbre couvrant minimal.

    Sur un complexe de Vietoris-Rips, deux composantes fusionnent exactement à
    la longueur de l'arête qui les relie dans l'ACM. Les classes H₀ naissent
    donc toutes à 0 et meurent aux longueurs d'arêtes de l'ACM.

    Complexité : O(n² log n) en temps, O(n²) en mémoire. D'où le
    sous-échantillonnage à ``max_points`` — plafonner est indispensable, la
    littérature note un pire cas cubique pour les dimensions supérieures.
    """
    x = _standardise(points)
    n = x.shape[0]
    if n < 3:
        raise TopologyError("au moins 3 points requis.")
    if n > max_points:
        rng = np.random.default_rng(seed)
        x = x[rng.choice(n, size=max_points, replace=False)]
        n = max_points

    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))

    iu = np.triu_indices(n, k=1)
    edges = np.stack([iu[0], iu[1], dist[iu]], axis=1)
    edges = edges[np.argsort(edges[:, 2])]

    uf = _UnionFind(n)
    out: list[PersistencePoint] = []
    for a, b, d in edges:
        if uf.union(int(a), int(b)):
            out.append(PersistencePoint(0, 0.0, float(d)))
            if len(out) == n - 1:
                break
    out.append(PersistencePoint(0, 0.0, math.inf))   # composante infinie
    return out


def _max_finite_persistence(diag: Sequence[PersistencePoint], dim: int) -> float:
    vals = [p.persistence for p in diag if p.dimension == dim and math.isfinite(p.persistence)]
    return max(vals) if vals else 0.0


def _h1_persistence(points: F64) -> list[PersistencePoint]:  # pragma: no cover
    if not HAS_RIPSER:
        return []
    res = _ripser(_standardise(points), maxdim=1)
    out: list[PersistencePoint] = []
    for dim, dgm in enumerate(res["dgms"]):
        for birth, death in dgm:
            out.append(PersistencePoint(dim, float(birth), float(death)))
    return out


@dataclass(frozen=True, slots=True)
class TopologyResult:
    n_points: int
    n_permutations: int
    observed_h0: float
    p_value_h0: float
    observed_h1: float | None
    p_value_h1: float | None
    alpha: float
    significant: bool
    n_tests: int
    verdict: str
    diagram: tuple[PersistencePoint, ...] = field(default=())

    def explain(self) -> str:
        lines = [
            f"TDA sur {self.n_points} mains · {self.n_permutations} permutations "
            f"· seuil corrigé α = {self.alpha:.2e} (Bonferroni sur {self.n_tests} tests)",
            f"  H₀ persistance max observée : {self.observed_h0:.4f}  →  p = {self.p_value_h0:.4f}",
        ]
        if self.observed_h1 is not None:
            lines.append(
                f"  H₁ persistance max observée : {self.observed_h1:.4f}  "
                f"→  p = {self.p_value_h1:.4f}"
            )
        else:
            lines.append("  H₁ : non calculé (ripser absent — `pip install ripser`)")
        lines.append(f"  → {self.verdict}")
        return "\n".join(lines)


def detect_patterns(
    features: F64,
    labels: Sequence[int] | None = None,
    n_permutations: int = 1000,
    n_tests: int = 1,
    alpha: float = 0.01,
    max_points: int = 400,
    compute_h1: bool = True,
    seed: int | None = 0,
) -> TopologyResult:
    """Cherche une structure topologique **statistiquement significative**.

    Parameters
    ----------
    features
        Matrice N×D : une ligne par main, colonnes = position, action, sizing
        rapporté au pot, pot odds, texture du board, SPR, street, temps de
        décision… Standardisées automatiquement.
    labels
        Labels d'action. S'ils sont fournis, la permutation nulle les mélange
        (test conditionnel, plus puissant). Sinon on permute chaque colonne
        indépendamment, ce qui détruit la structure jointe.
    n_tests
        Nombre de tests effectués au total (typiquement le nombre
        d'adversaires analysés). Sert à la correction de Bonferroni.

    Examples
    --------
    Sur du bruit pur, le test ne doit rien déclarer :

    >>> rng = np.random.default_rng(0)
    >>> res = detect_patterns(rng.normal(size=(120, 5)), n_permutations=100)
    >>> res.significant
    False
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 20:
        raise TopologyError("au moins 20 mains et une matrice 2-D sont requises.")
    if n_permutations < 20:
        raise TopologyError("n_permutations trop faible pour une p-value utile.")

    rng = np.random.default_rng(seed)
    if x.shape[0] > max_points:
        x = x[rng.choice(x.shape[0], size=max_points, replace=False)]

    diag0 = h0_persistence(x, max_points=max_points, seed=seed)
    obs_h0 = _max_finite_persistence(diag0, 0)

    diag1 = _h1_persistence(x) if (compute_h1 and HAS_RIPSER) else []
    obs_h1 = _max_finite_persistence(diag1, 1) if diag1 else None

    ge0 = 0
    ge1 = 0
    for _ in range(n_permutations):
        if labels is not None:
            perm = np.asarray(labels)[rng.permutation(len(labels))]
            null = np.column_stack([x[:, :-1], perm[: x.shape[0]]])
        else:
            null = np.column_stack([rng.permutation(col) for col in x.T])
        d0 = h0_persistence(null, max_points=max_points, seed=None)
        if _max_finite_persistence(d0, 0) >= obs_h0:
            ge0 += 1
        if obs_h1 is not None:  # pragma: no cover
            d1 = _h1_persistence(null)
            if _max_finite_persistence(d1, 1) >= obs_h1:
                ge1 += 1

    # Estimateur non biaisé de la p-value : (1 + #extrêmes) / (1 + B).
    p0 = (1.0 + ge0) / (1.0 + n_permutations)
    p1 = ((1.0 + ge1) / (1.0 + n_permutations)) if obs_h1 is not None else None

    corrected = alpha / max(1, n_tests)
    sig = p0 < corrected or (p1 is not None and p1 < corrected)

    if sig:
        verdict = (
            "Structure significative après correction. Elle mérite une "
            "inspection manuelle — la TDA dit QU'il y a une structure, pas laquelle."
        )
    else:
        verdict = (
            "Aucune structure significative. C'est le résultat le plus fréquent, "
            "et c'est normal : sans ce test, on aurait « trouvé » un pattern."
        )

    return TopologyResult(
        n_points=int(x.shape[0]),
        n_permutations=n_permutations,
        observed_h0=obs_h0,
        p_value_h0=p0,
        observed_h1=obs_h1,
        p_value_h1=p1,
        alpha=corrected,
        significant=sig,
        n_tests=n_tests,
        verdict=verdict,
        diagram=tuple(diag0[:50]),
    )
