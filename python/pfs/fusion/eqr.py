r"""
L7 — Equity realization (EQR) apprise depuis NOS PROPRES solves.

Statut (14 août 2026) — BRANCHÉE dans l'application
---------------------------------------------------
Branchée le 14 août 2026 (recommandation « S » de l'audit des orphelins) :
``/api/postflop`` accepte ``leaf_model="eqr"`` ; le serveur
(``pfs/app/server.py:_eqr_entraine``) entraîne ``train_eqr()`` UNE fois par
processus puis mémoïse (``lru_cache``), et republie dans chaque réponse le
R² et le n MESURÉS du modèle avec sa limite : valeur DIRECTIONNELLE — elle
relève l'EV du joueur en position par rapport au rollout, sans garantie
d'écart au solve complet (mesuré : elle peut sur-corriger) — et la somme
des EV dérive. Le registre (``pfs/bench/solver_registry.py``) porte
``wired=True`` sur l'entrée EQR ; test de bout en bout :
``tests/test_app_and_data.py::test_api_postflop_leaf_model_eqr_est_atteignable_et_honnete``.

EQR := EV réalisée / (équité brute × pot). C'est le facteur qui sépare
« avoir 55 % d'équité » de « gagner 55 % du pot » : position, initiative,
polarité de la range et SPR font qu'on réalise plus ou moins que sa part.

Les solveurs commerciaux l'affichent (Pio, GTO Wizard) mais leur modèle est
opaque. Ici, la boucle est fermée **en interne** : le solveur postflop (L1)
résout des spots river, chaque solve fournit un couple (équité brute,
EV Nash), et une régression ridge (Hoerl & Kennard, 1970 — forme close,
zéro dépendance d'apprentissage) apprend EQR en fonction de traits
interprétables :

    EQR ≈ β₀ + β₁·équité + β₂·IP + β₃·log(SPR) + β₄·polarité + β₅·part_nuts

Le solveur fabrique ses propres données d'entraînement — aucune donnée
externe, aucune main réelle nécessaire. Portée assumée : EQR *river* (une
street de mise) ; l'EQR multi-street suivra l'extension turn/flop du solveur.

Faits attendus, servant de tests de cohérence : le coefficient IP est
positif (l'avantage de position est un théorème folklorique mais
empiriquement universel), et l'EQR croît avec la part de nuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import Range, parse_range
from pfs.solver.postflop import IP, OOP, PostflopSolver

__all__ = ["EqrError", "EqrModel", "EqrSample", "train_eqr", "range_features"]

F64 = npt.NDArray[np.float64]

FEATURES = ("const", "equity", "is_ip", "log_spr", "polarity", "nut_share")

# Bibliothèque de paires de ranges aux profils contrastés (serré/large,
# polarisé/condensé) — la diversité des traits, pas le réalisme préflop.
_RANGE_LIB: tuple[tuple[str, str], ...] = (
    ("QQ+, AKs, AKo", "22+, A2s+, KTs+, QTs+, JTs, ATo+"),
    ("22+, A2s+, K9s+, QTs+, ATo+, KJo+", "TT+, AQs+, AQo+"),
    ("AA, KK, 44, 33", "QQ, JJ, TT, 99"),
    ("22+, ATs+, KQs, AJo+", "22+, ATs+, KQs, AJo+"),
    ("88+, AQs+, AKo", "55+, A8s+, KTs+, AJo+, KQo"),
)


class EqrError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EqrSample:
    equity: float
    is_ip: float
    log_spr: float
    polarity: float
    nut_share: float
    eqr: float                      # cible : EV / (équité × pot)


@dataclass(frozen=True, slots=True)
class EqrModel:
    """Régression ridge fermée sur les traits de FEATURES."""
    coef: tuple[float, ...]         # aligné sur FEATURES
    r2: float
    n: int

    def predict(self, equity: float, is_ip: bool, spr: float,
                polarity: float, nut_share: float) -> float:
        if not (0.0 < equity < 1.0):
            raise EqrError("équité dans (0,1) requise.")
        if spr <= 0:
            raise EqrError("SPR > 0 requis.")
        x = (1.0, equity, float(is_ip), float(np.log(spr)),
             polarity, nut_share)
        raw = float(np.dot(self.coef, x))
        return float(np.clip(raw, 0.2, 2.5))

    def explain(self) -> str:
        rows = [f"  {n:<10} {c:+.4f}" for n, c in zip(FEATURES, self.coef)]
        return (f"EQR ridge — R² {self.r2:.3f} sur {self.n} échantillons\n"
                + "\n".join(rows))


def range_features(solver: PostflopSolver, player: int) -> tuple[float, float, float]:
    """(équité brute, polarité, part de nuts) du joueur dans le spot.

    Équité par combo = part d'abattage contre la range adverse (card removal
    inclus), via le même contexte trié que le solveur — pot 1, mise 0.
    """
    ctx = solver._sd_ctx[(player, int(solver.rivers[0]))]
    q = solver.players[1 - player].weights
    share = ctx.cfv(q, pot=1.0, invested_p=0.0)
    compat = ctx.compat_mass(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        eq_combo = np.where(compat > 0, share / np.where(compat > 0, compat, 1), 0.0)
    w = solver.players[player].weights * (compat > 0)
    if w.sum() <= 0:
        raise EqrError("range vide après blocage.")
    w = w / w.sum()
    equity = float((w * eq_combo).sum())
    polarity = float(np.sqrt(((eq_combo - equity) ** 2 * w).sum()))
    nut_share = float(w[eq_combo > 0.9].sum())
    return equity, polarity, nut_share


def _random_river(rng: np.random.Generator) -> list[int]:
    return [int(x) for x in rng.choice(52, size=5, replace=False)]


def generate_samples(
    n_spots: int = 24,
    iterations: int = 150,
    seed: int = 0,
    pot: float = 100.0,
) -> list[EqrSample]:
    """Résout n_spots rivers aléatoires et extrait (traits → EQR) par joueur."""
    if n_spots < 2:
        raise EqrError("n_spots >= 2.")
    rng = np.random.default_rng(seed)
    out: list[EqrSample] = []
    made = 0
    guard = 0
    while made < n_spots and guard < n_spots * 4:
        guard += 1
        board = _random_river(rng)
        spec_o, spec_i = _RANGE_LIB[int(rng.integers(len(_RANGE_LIB)))]
        if rng.random() < 0.5:
            spec_o, spec_i = spec_i, spec_o
        spr = float(np.exp(rng.uniform(np.log(0.5), np.log(4.0))))
        try:
            s = PostflopSolver(board, parse_range(spec_o), parse_range(spec_i),
                               pot=pot, stack=spr * pot,
                               bet_fracs=(0.75,), max_bets=2)
        except Exception:
            continue                     # board bloque une range : re-tire
        s.solve(iterations)
        ev = s.values()
        for p in (OOP, IP):
            try:
                eq, pol, nuts = range_features(s, p)
            except EqrError:
                continue
            if not (0.02 < eq < 0.98):
                continue
            out.append(EqrSample(eq, float(p == IP), float(np.log(spr)),
                                 pol, nuts, ev[p] / (eq * pot)))
        made += 1
    if len(out) < 8:
        raise EqrError("trop peu d'échantillons exploitables.")
    return out


def fit(samples: Sequence[EqrSample], ridge: float = 0.05) -> EqrModel:
    """Ridge forme close : β = (XᵀX + λI)⁻¹ Xᵀy (constante non pénalisée)."""
    if len(samples) < len(FEATURES):
        raise EqrError("moins d'échantillons que de traits.")
    X = np.array([[1.0, s.equity, s.is_ip, s.log_spr, s.polarity, s.nut_share]
                  for s in samples])
    y = np.array([s.eqr for s in samples])
    lam = ridge * np.eye(X.shape[1])
    lam[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + lam, X.T @ y)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return EqrModel(tuple(float(b) for b in beta), r2, len(samples))


def train_eqr(n_spots: int = 24, iterations: int = 150,
              seed: int = 0) -> EqrModel:
    """Pipeline complet : solve → échantillons → ridge."""
    return fit(generate_samples(n_spots, iterations, seed))
