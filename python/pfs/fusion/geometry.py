"""
F5 — Géométrie de l'information sur les ranges.

Sources
-------
- Rao, C.R. (1945)
- Amari, S. (1985), *Differential-Geometrical Methods in Statistics*
- Amari, S. (1998), *Natural Gradient Works Efficiently in Learning*

Une range est un point du simplexe Δ^1325. La distance euclidienne entre deux
ranges n'a **aucune signification statistique** : passer de 0,001 à 0,002 est
un changement d'information bien plus grand que de 0,500 à 0,501. La métrique
correcte est celle de Fisher-Rao, qui admet une forme close sur le simplexe.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "normalise",
    "bhattacharyya_coefficient",
    "fisher_rao_distance",
    "hellinger_distance",
    "kl_divergence",
    "jensen_shannon_distance",
    "natural_gradient_step",
    "range_deviation_score",
    "kmeans_fisher",
]

F64 = npt.NDArray[np.float64]

# Régularisation contre le mauvais conditionnement aux bords du simplexe.
EPS: float = 1e-12
REG: float = 1e-9


class GeometryError(ValueError):
    pass


def normalise(p: Sequence[float] | F64, regularise: bool = True) -> F64:
    r"""Projette un vecteur de poids sur le simplexe, avec régularisation.

    Sur 1326 dimensions largement creuses, la métrique de Fisher est mal
    conditionnée aux bords. On mélange une pincée d'uniforme :
    :math:`p \leftarrow (1-\varepsilon)p + \varepsilon/n`.
    """
    a = np.asarray(p, dtype=np.float64).ravel()
    if a.size == 0:
        raise GeometryError("vecteur vide.")
    if np.any(a < -EPS):
        raise GeometryError("poids négatif.")
    a = np.maximum(a, 0.0)
    s = a.sum()
    if s <= 0.0:
        raise GeometryError("somme des poids nulle.")
    a = a / s
    if regularise:
        n = a.size
        a = (1.0 - REG) * a + REG / n
        a /= a.sum()
    return a


def bhattacharyya_coefficient(p: F64, q: F64) -> float:
    r""":math:`BC(p,q) = \sum_i \sqrt{p_i q_i} \in [0,1]`."""
    p, q = normalise(p), normalise(q)
    if p.shape != q.shape:
        raise GeometryError("dimensions incompatibles.")
    return float(np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0))


def fisher_rao_distance(p: F64, q: F64) -> float:
    r"""Distance de Fisher-Rao sur le simplexe.

    .. math::
        d_{FR}(p,q) = 2\arccos\Big(\sum_i \sqrt{p_i q_i}\Big) \in [0, \pi]

    Forme close obtenue par la transformation sphérique
    :math:`y_i = 2\sqrt{p_i}`, qui envoie le simplexe sur l'orthant positif
    de la sphère de rayon 2.

    Examples
    --------
    Exemple golden du Plan Directeur §4 F5 :

    >>> p = [0.40, 0.25, 0.20, 0.10, 0.05]
    >>> q = [0.55, 0.20, 0.15, 0.07, 0.03]
    >>> round(fisher_rao_distance(p, q), 6)
    0.30671
    """
    bc = bhattacharyya_coefficient(p, q)
    return float(2.0 * math.acos(min(1.0, max(-1.0, bc))))


def hellinger_distance(p: F64, q: F64) -> float:
    r""":math:`H(p,q) = \sqrt{1 - BC(p,q)} \in [0,1]`. Version bornée, utile en UI."""
    return float(math.sqrt(max(0.0, 1.0 - bhattacharyya_coefficient(p, q))))


def kl_divergence(p: F64, q: F64) -> float:
    r""":math:`D_{KL}(p\|q) = \sum p \log_2(p/q)`, en bits. Non symétrique."""
    p, q = normalise(p), normalise(q)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def jensen_shannon_distance(p: F64, q: F64) -> float:
    r"""Racine de la divergence de Jensen-Shannon : une vraie métrique, en bits."""
    p, q = normalise(p), normalise(q)
    m = 0.5 * (p + q)
    js = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(math.sqrt(max(0.0, js)))


def natural_gradient_step(
    p: F64, grad: F64, step: float = 0.05
) -> F64:
    r"""Un pas de gradient naturel sur le simplexe (Amari, 1998).

    Sur le simplexe, la métrique de Fisher est diagonale de terme
    :math:`1/p_i`, donc :math:`G^{-1} = \mathrm{diag}(p)` et la mise à jour
    naturelle prend la forme exponentielle (miroir/Hedge) :

    .. math::
        p_i \leftarrow \frac{p_i \exp(\eta\, g_i)}{\sum_j p_j \exp(\eta\, g_j)}

    Converge en nettement moins d'itérations que la descente euclidienne
    projetée, et reste dans le simplexe par construction.
    """
    p = normalise(p)
    g = np.asarray(grad, dtype=np.float64).ravel()
    if g.shape != p.shape:
        raise GeometryError("gradient de dimension incompatible.")
    z = step * (g - g.max())          # stabilisation numérique
    w = p * np.exp(z)
    return normalise(w, regularise=False)


def range_deviation_score(
    observed: F64, baseline: F64, threshold: float = 0.25
) -> tuple[float, bool, str]:
    """Mesure invariante de l'écart d'une range à sa référence GTO.

    Contrairement à un écart de VPIP, la distance de Fisher-Rao est comparable
    entre positions, formats et tailles de range.

    Returns
    -------
    (distance, est_notable, interprétation)
    """
    d = fisher_rao_distance(observed, baseline)
    notable = d > threshold
    if d < 0.10:
        label = "quasi identique au GTO"
    elif d < threshold:
        label = "écart mineur, non exploitable seul"
    elif d < 0.60:
        label = "déviation nette — chercher la stat responsable"
    else:
        label = "range très atypique — exploitation probable"
    return d, notable, label


def kmeans_fisher(
    points: Sequence[F64],
    k: int,
    n_iter: int = 50,
    seed: int | None = 0,
) -> tuple[list[F64], list[int]]:
    """k-means sur la variété statistique, avec la distance de Fisher-Rao.

    Utilisé pour construire une taxonomie d'adversaires à partir de leurs
    ranges observées. Les archétypes obtenus sont statistiquement fondés,
    contrairement à un k-means euclidien sur des stats brutes.

    Le barycentre sous la métrique de Fisher est calculé dans les coordonnées
    sphériques :math:`y = \\sqrt{p}` (moyenne puis re-normalisation), ce qui
    est l'approximation standard de la moyenne de Karcher sur la sphère.
    """
    if k < 1:
        raise GeometryError("k doit être >= 1.")
    pts = [normalise(p) for p in points]
    if len(pts) < k:
        raise GeometryError("moins de points que de clusters.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=k, replace=False)
    centroids = [pts[i].copy() for i in idx]
    labels = [0] * len(pts)

    for _ in range(n_iter):
        changed = False
        for i, p in enumerate(pts):
            dists = [fisher_rao_distance(p, c) for c in centroids]
            lab = int(np.argmin(dists))
            if lab != labels[i]:
                labels[i] = lab
                changed = True

        for c in range(k):
            members = [pts[i] for i in range(len(pts)) if labels[i] == c]
            if not members:
                continue
            y = np.mean([np.sqrt(m) for m in members], axis=0)
            centroids[c] = normalise(y**2)

        if not changed:
            break

    return centroids, labels
