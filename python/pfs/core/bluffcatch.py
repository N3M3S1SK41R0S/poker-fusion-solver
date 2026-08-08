"""
Poker Fusion Solver — F10 : seuil de bluff-catch par théorie de la détection
du signal.

Sources
-------
- Peterson, Birdsall & Fox (1954)
- Green, D.M. & Swets, J.A. (1966), *Signal Detection Theory and Psychophysics*

Un bluff-catch est formellement une tâche de détection : le "signal" est le
bluff, le "bruit" est la value. La SDT donne le seuil de décision optimal en
forme close, et — surtout — permet d'exprimer la **confiance** que l'appel est
+EV, quantité qu'aucun solveur commercial n'affiche.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

__all__ = [
    "BluffCatchAnalysis",
    "required_equity",
    "minimum_defence_frequency",
    "analyse_bluffcatch",
    "d_prime",
    "optimal_criterion",
]


class BluffCatchError(ValueError):
    """Paramètres hors domaine."""


def required_equity(pot: float, bet: float) -> float:
    r"""Équité minimale pour qu'un call soit +EV (pot odds).

    .. math::
        \alpha = \frac{b}{P + 2b}

    Examples
    --------
    >>> round(required_equity(100.0, 75.0), 4)
    0.3
    """
    if pot < 0.0 or bet <= 0.0:
        raise BluffCatchError("pot >= 0 et bet > 0 requis.")
    return bet / (pot + 2.0 * bet)


def minimum_defence_frequency(pot: float, bet: float) -> float:
    r"""MDF — fréquence de défense qui rend le bluff indifférent.

    .. math::
        \mathrm{MDF} = \frac{P}{P + b}

    Examples
    --------
    >>> round(minimum_defence_frequency(100.0, 75.0), 4)
    0.5714
    """
    if pot < 0.0 or bet <= 0.0:
        raise BluffCatchError("pot >= 0 et bet > 0 requis.")
    return pot / (pot + bet)


def d_prime(mu_bluff: float, mu_value: float, sigma: float) -> float:
    r"""Discriminabilité :math:`d' = (\mu_{\text{bluff}} - \mu_{\text{value}})/\sigma`.

    Mesure à quel point tes indices (sizing, timing, historique) séparent
    réellement bluff et value. ``d' = 0`` : aucune information ;
    ``d' > 2`` : séparation forte (rare au poker en ligne).
    """
    if sigma <= 0.0:
        raise BluffCatchError("sigma doit être strictement positif.")
    return (mu_bluff - mu_value) / sigma


def optimal_criterion(p_bluff: float, alpha: float, dp: float) -> float:
    r"""Critère de décision optimal :math:`x_c` (Bayes).

    .. math::
        \ln\beta^* = \ln\frac{P(\text{value})}{P(\text{bluff})}
                     + \ln\frac{\alpha}{1-\alpha}
        \qquad
        x_c = \frac{\ln\beta^*}{d'} + \frac{d'}{2}
    """
    if not (0.0 < p_bluff < 1.0):
        raise BluffCatchError("p_bluff doit être dans (0, 1).")
    if not (0.0 < alpha < 1.0):
        raise BluffCatchError("alpha doit être dans (0, 1).")
    if dp <= 0.0:
        raise BluffCatchError("d' doit être strictement positif.")

    log_beta = math.log((1.0 - p_bluff) / p_bluff) + math.log(alpha / (1.0 - alpha))
    return log_beta / dp + dp / 2.0


@dataclass(frozen=True, slots=True)
class BluffCatchAnalysis:
    """Analyse complète d'une décision de bluff-catch."""

    pot: float
    bet: float
    required_equity: float
    mdf: float
    estimated_bluff_freq: float
    bluff_freq_std: float
    margin: float
    z_margin: float
    prob_call_is_plus_ev: float
    ev_call: float
    ev_fold: float
    recommendation: str

    def explain(self) -> str:
        return (
            f"Pot {self.pot:.1f} bb, bet {self.bet:.1f} bb\n"
            f"  Équité requise (pot odds) : {self.required_equity * 100:.1f} %\n"
            f"  MDF                        : {self.mdf * 100:.1f} %\n"
            f"  Fréq. de bluff estimée     : {self.estimated_bluff_freq * 100:.1f} % "
            f"± {self.bluff_freq_std * 100:.1f}\n"
            f"  Marge                      : {self.margin * 100:+.1f} pts "
            f"(z = {self.z_margin:+.3f})\n"
            f"  P(call est +EV)            : {self.prob_call_is_plus_ev * 100:.1f} %\n"
            f"  EV(call) = {self.ev_call:+.3f} bb   EV(fold) = {self.ev_fold:+.3f} bb\n"
            f"  → {self.recommendation}"
        )


def analyse_bluffcatch(
    pot: float,
    bet: float,
    estimated_bluff_freq: float,
    bluff_freq_std: float,
    confidence_threshold: float = 0.60,
) -> BluffCatchAnalysis:
    r"""Analyse un spot de bluff-catch avec propagation d'incertitude.

    La différence essentielle avec un calcul de pot odds classique : la
    fréquence de bluff estimée est une **variable aléatoire** (issue du
    tracker Beta-Binomial F1), pas un point. On en déduit

    .. math::
        P(\text{call est +EV}) = \Phi\!\left(\frac{\hat p - \alpha}{\sigma_p}\right)

    Examples
    --------
    >>> a = analyse_bluffcatch(100.0, 75.0, 0.34, 0.09)
    >>> round(a.required_equity, 4)
    0.3
    >>> round(a.prob_call_is_plus_ev, 6)
    0.671639
    """
    if not (0.0 <= estimated_bluff_freq <= 1.0):
        raise BluffCatchError("estimated_bluff_freq doit être dans [0, 1].")
    if bluff_freq_std < 0.0:
        raise BluffCatchError("bluff_freq_std doit être >= 0.")

    alpha = required_equity(pot, bet)
    mdf = minimum_defence_frequency(pot, bet)
    margin = estimated_bluff_freq - alpha

    if bluff_freq_std > 0.0:
        z = margin / bluff_freq_std
        p_plus_ev = float(stats.norm.cdf(z))
    else:
        z = math.inf if margin > 0 else (-math.inf if margin < 0 else 0.0)
        p_plus_ev = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)

    # EV en bb, convention : fold = 0 (référence), call = gain espéré net.
    ev_call = estimated_bluff_freq * (pot + bet) - (1.0 - estimated_bluff_freq) * bet
    ev_fold = 0.0

    if p_plus_ev >= confidence_threshold:
        reco = f"CALL (confiance {p_plus_ev * 100:.0f} %)"
    elif p_plus_ev <= 1.0 - confidence_threshold:
        reco = f"FOLD (confiance {(1 - p_plus_ev) * 100:.0f} %)"
    else:
        reco = (
            f"MARGINAL — indifférent à {p_plus_ev * 100:.0f} %. "
            "Mixer, ou trancher sur les blockers et l'historique."
        )

    return BluffCatchAnalysis(
        pot=pot,
        bet=bet,
        required_equity=alpha,
        mdf=mdf,
        estimated_bluff_freq=estimated_bluff_freq,
        bluff_freq_std=bluff_freq_std,
        margin=margin,
        z_margin=z,
        prob_call_is_plus_ev=p_plus_ev,
        ev_call=ev_call,
        ev_fold=ev_fold,
        recommendation=reco,
    )
