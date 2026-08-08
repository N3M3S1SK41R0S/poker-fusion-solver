"""
F4 — Bet sizing par gain d'information, sous contrainte d'EV.

Sources
-------
- Shannon, C.E. (1948)
- Berger, T. (1971), *Rate Distortion Theory*
- Frazier, Powell & Dayanik (2008), *A Knowledge-Gradient Policy for
  Sequential Information Collection*, SIAM J. Control Optim. 47(5)

⚠️ CORRECTION DU MODÈLE v1
──────────────────────────
Le prototype `optimal_bet_size` maximisait le gain d'information **sans
contrainte d'EV**. C'est une erreur de conception : le sizing qui maximise
I(X;Y) est presque toujours le plus gros possible — un all-in polarise
maximalement la réponse, donc révèle le maximum d'information. Un joueur qui
suit ce modèle part all-in en permanence.

La formulation correcte est un lagrangien :

    b* = argmax_b [ EV(b) + λ · I(b) ]

où λ est le **prix d'un bit d'information sur cet adversaire**, exprimé en bb.
Ce n'est pas une constante : λ doit croître avec le nombre de mains restantes
et avec l'incertitude actuelle sur le modèle adverse. C'est formellement le
*knowledge gradient* de Frazier-Powell-Dayanik.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

__all__ = [
    "entropy_bits",
    "CallModel",
    "LogisticCallModel",
    "SizingCandidate",
    "SizingAnalysis",
    "information_gain",
    "expected_value",
    "knowledge_price",
    "optimal_bet_size",
    "sizing_table",
]

F64 = npt.NDArray[np.float64]


class SizingError(ValueError):
    pass


def entropy_bits(weights: F64 | Sequence[float]) -> float:
    r""":math:`H = -\sum p\log_2 p` sur les poids normalisés, en bits."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if np.any(w < 0):
        raise SizingError("poids négatif.")
    total = w.sum()
    if total <= 0.0:
        return 0.0
    p = w[w > 0.0] / total
    return float(-np.sum(p * np.log2(p)))


class CallModel:
    """Interface : probabilité qu'un combo paie, en fonction de la mise."""

    def call_probs(self, bet: float, pot: float, equities: F64) -> F64:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LogisticCallModel(CallModel):
    r"""Modèle logistique : on paie si l'équité dépasse le seuil de pot odds.

    .. math::
        P(\text{call} \mid c) = \sigma\!\big(k\,(\mathrm{eq}(c) - \alpha(b) - \delta)\big),
        \qquad \alpha(b) = \frac{b}{P + 2b}

    Parameters
    ----------
    sharpness : float
        Pente k. Élevée ⇒ adversaire quasi-optimal (seuil net) ;
        faible ⇒ adversaire flou (call station ou hyper-nit).
    looseness : float
        Décalage δ du seuil. > 0 ⇒ paie plus large que les pot odds.

    Notes
    -----
    **À calibrer par régression logistique sur tes hand-histories réels**, par
    adversaire ou par archétype. Les valeurs par défaut sont un point de
    départ, pas une vérité.
    """

    sharpness: float = 14.0
    looseness: float = 0.0

    def call_probs(self, bet: float, pot: float, equities: F64) -> F64:
        if bet <= 0.0 or pot <= 0.0:
            raise SizingError("bet et pot doivent être > 0.")
        alpha = bet / (pot + 2.0 * bet)
        z = self.sharpness * (equities - alpha - self.looseness)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


@dataclass(frozen=True, slots=True)
class MDFCallModel(CallModel):
    r"""Adversaire compétent : défend la fraction MDF de sa range, par le haut.

    .. math::
        \mathrm{MDF}(b) = \frac{P}{P+b}

    Le villain paie avec ses ``MDF`` meilleures mains, ce qui rend les bluffs
    du hero **indifférents** — c'est la définition même de la MDF.

    C'est le modèle qui compte, parce qu'il produit la situation réelle
    observée par les solveurs : **l'EV est quasi plate entre les sizings**
    (sur T9s8s, 4 sizings vs 1 sizing donnent 427 vs 427,3). Quand l'EV
    n'arbitre plus, le gain d'information devient le critère rationnel de
    départage. C'est là que F4 gagne sa place.

    Parameters
    ----------
    slack : float
        Écart à la MDF théorique. > 0 ⇒ over-defend (call station),
        < 0 ⇒ over-fold (nit exploitable).
    softness : float
        Largeur de la transition autour du seuil. 0 ⇒ seuil net.
    """

    slack: float = 0.0
    softness: float = 0.04

    def call_probs(self, bet: float, pot: float, equities: F64) -> F64:
        if bet <= 0.0 or pot <= 0.0:
            raise SizingError("bet et pot doivent être > 0.")
        eq = np.asarray(equities, dtype=np.float64).ravel()
        mdf = float(np.clip(pot / (pot + bet) + self.slack, 0.0, 1.0))
        if mdf <= 0.0:
            return np.zeros_like(eq)
        if mdf >= 1.0:
            return np.ones_like(eq)
        # Seuil = quantile (1 - MDF) des équités : on défend par le haut.
        threshold = float(np.quantile(eq, 1.0 - mdf))
        if self.softness <= 0.0:
            return (eq >= threshold).astype(np.float64)
        z = (eq - threshold) / self.softness
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def information_gain(range_weights: F64, call_probs: F64) -> float:
    r"""Information mutuelle I(range ; réponse), en bits.

    .. math::
        I = H(R) - \big[P(\text{call})H(R\mid\text{call})
                        + P(\text{fold})H(R\mid\text{fold})\big]
    """
    w = np.asarray(range_weights, dtype=np.float64).ravel()
    c = np.asarray(call_probs, dtype=np.float64).ravel()
    if w.shape != c.shape:
        raise SizingError("dimensions incompatibles.")
    total = w.sum()
    if total <= 0.0:
        return 0.0
    r = w / total

    h_before = entropy_bits(r)
    p_call = float(np.sum(r * c))
    p_fold = 1.0 - p_call

    h_after = 0.0
    if p_call > 1e-12:
        h_after += p_call * entropy_bits(r * c)
    if p_fold > 1e-12:
        h_after += p_fold * entropy_bits(r * (1.0 - c))

    return float(max(0.0, h_before - h_after))


def expected_value(
    range_weights: F64,
    equities: F64,
    call_probs: F64,
    pot: float,
    bet: float,
) -> float:
    r"""EV d'une mise, en bb, contre la range adverse pondérée.

    .. math::
        EV(b) = \sum_c r(c)\Big[(1-p_c)\,P + p_c\big(\mathrm{eq}(c)(P+2b) - b\big)\Big]

    Convention : équité exprimée du point de vue **du villain**, donc le hero
    encaisse :math:`(1 - eq)` de la valeur du pot final.
    """
    w = np.asarray(range_weights, dtype=np.float64).ravel()
    eq = np.asarray(equities, dtype=np.float64).ravel()
    c = np.asarray(call_probs, dtype=np.float64).ravel()
    if not (w.shape == eq.shape == c.shape):
        raise SizingError("dimensions incompatibles.")
    total = w.sum()
    if total <= 0.0:
        return 0.0
    r = w / total

    fold_value = pot
    called_value = (1.0 - eq) * (pot + 2.0 * bet) - bet
    return float(np.sum(r * ((1.0 - c) * fold_value + c * called_value)))


def knowledge_price(
    hands_remaining: int,
    posterior_std: float,
    max_std: float = 0.25,
    base_price: float = 1.0,
    horizon: int = 100,
) -> float:
    r"""Prix λ d'un bit d'information, en bb.

    .. math::
        \lambda = \lambda_0 \cdot \frac{n_{\text{restantes}}}{H}
                  \cdot \frac{\sigma_\theta}{\sigma_{\max}}

    - Session longue **et** adversaire mal connu ⇒ λ élevé : payer pour apprendre.
    - Dernière main, ou adversaire déjà profilé ⇒ λ → 0 : pur EV.

    C'est un arbitrage exploration/exploitation, formellement identique au
    knowledge gradient (Frazier, Powell & Dayanik, 2008).
    """
    if hands_remaining < 0:
        raise SizingError("hands_remaining doit être >= 0.")
    if posterior_std < 0.0:
        raise SizingError("posterior_std doit être >= 0.")
    horizon_factor = min(3.0, hands_remaining / max(1, horizon))
    uncertainty_factor = min(1.0, posterior_std / max_std)
    return float(base_price * horizon_factor * uncertainty_factor)


@dataclass(frozen=True, slots=True)
class SizingCandidate:
    bet: float
    fraction_of_pot: float
    p_fold: float
    p_call: float
    ev: float
    entropy_after: float
    info_gain: float
    objective: float


@dataclass(frozen=True, slots=True)
class SizingAnalysis:
    pot: float
    lam: float
    entropy_before: float
    candidates: tuple[SizingCandidate, ...]
    best_ev: SizingCandidate
    best_info: SizingCandidate
    best_fused: SizingCandidate

    def explain(self) -> str:
        lines = [
            f"Pot {self.pot:.1f} bb · H(range) = {self.entropy_before:.2f} bits "
            f"· λ = {self.lam:.2f} bb/bit",
            f"{'bet':>8} {'b/pot':>7} {'P(fold)':>8} {'EV':>8} "
            f"{'H|.':>7} {'IG':>7} {'EV+λI':>8}",
        ]
        for c in self.candidates:
            mark = ""
            if c is self.best_fused:
                mark = "  ★ fusion"
            elif c is self.best_ev:
                mark = "  · EV pur"
            elif c is self.best_info:
                mark = "  · IG pur"
            lines.append(
                f"{c.bet:8.1f} {c.fraction_of_pot:7.2f} {c.p_fold:8.2f} "
                f"{c.ev:8.2f} {c.entropy_after:7.2f} {c.info_gain:7.2f} "
                f"{c.objective:8.2f}{mark}"
            )
        if self.best_fused is self.best_ev:
            lines.append(
                f"\nÀ λ = {self.lam:.2f} bb/bit, l'information ne renverse pas l'EV : "
                f"le sizing EV-optimal reste le bon."
            )
        else:
            d_ev = self.best_fused.ev - self.best_ev.ev
            d_ig = self.best_fused.info_gain - self.best_ev.info_gain
            lines.append(
                f"\nFusion vs EV pur : {d_ev:+.3f} bb concédés pour {d_ig:+.2f} bits "
                f"acquis — rentable dès que λ > {abs(d_ev / max(d_ig, 1e-9)):.2f} bb/bit."
            )
        return "\n".join(lines)


def _evaluate(
    bet: float,
    pot: float,
    range_weights: F64,
    equities: F64,
    model: CallModel,
    lam: float,
) -> SizingCandidate:
    c = model.call_probs(bet, pot, equities)
    w = np.asarray(range_weights, dtype=np.float64).ravel()
    r = w / w.sum()
    p_call = float(np.sum(r * c))
    ig = information_gain(w, c)
    ev = expected_value(w, equities, c, pot, bet)

    h_after = 0.0
    p_fold = 1.0 - p_call
    if p_call > 1e-12:
        h_after += p_call * entropy_bits(w * c)
    if p_fold > 1e-12:
        h_after += p_fold * entropy_bits(w * (1.0 - c))

    return SizingCandidate(
        bet=bet,
        fraction_of_pot=bet / pot,
        p_fold=p_fold,
        p_call=p_call,
        ev=ev,
        entropy_after=h_after,
        info_gain=ig,
        objective=ev + lam * ig,
    )


def optimal_bet_size(
    pot: float,
    range_weights: F64,
    equities: F64,
    lam: float = 0.5,
    model: CallModel | None = None,
    bounds: tuple[float, float] = (0.20, 2.00),
) -> SizingCandidate:
    r"""Optimise :math:`b^\ast = \arg\max_b\,[EV(b) + \lambda I(b)]` en continu.

    Parameters
    ----------
    bounds
        Bornes en fraction de pot. (0.20, 2.00) couvre du mini-bet à l'overbet
        double pot.
    """
    if pot <= 0.0:
        raise SizingError("pot doit être > 0.")
    m = model or LogisticCallModel()

    def neg(frac: float) -> float:
        return -_evaluate(frac * pot, pot, range_weights, equities, m, lam).objective

    res = minimize_scalar(neg, bounds=bounds, method="bounded",
                          options={"xatol": 1e-4})
    return _evaluate(float(res.x) * pot, pot, range_weights, equities, m, lam)


def sizing_table(
    pot: float,
    range_weights: F64,
    equities: F64,
    lam: float = 0.5,
    fractions: Sequence[float] = (0.33, 0.55, 0.75, 1.00, 1.50),
    model: CallModel | None = None,
) -> SizingAnalysis:
    """Compare une grille de sizings et isole les trois optima.

    C'est la vue destinée à l'écran : elle montre explicitement l'écart entre
    « le sizing qui gagne le plus » et « le sizing qui apprend le plus ».
    """
    if pot <= 0.0:
        raise SizingError("pot doit être > 0.")
    m = model or LogisticCallModel()
    cands = tuple(
        _evaluate(f * pot, pot, range_weights, equities, m, lam) for f in fractions
    )
    return SizingAnalysis(
        pot=pot,
        lam=lam,
        entropy_before=entropy_bits(range_weights),
        candidates=cands,
        best_ev=max(cands, key=lambda c: c.ev),
        best_info=max(cands, key=lambda c: c.info_gain),
        best_fused=max(cands, key=lambda c: c.objective),
    )
