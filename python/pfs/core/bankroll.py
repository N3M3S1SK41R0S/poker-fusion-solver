"""
Poker Fusion Solver — F9 : Bankroll sous contrainte ergodique.

Fusion de quatre cadres établis :
  - Kelly, J.L. (1956), "A New Interpretation of Information Rate"
  - Malmuth, M., forme exponentielle du risque de ruine
  - Peters, O. (2019), "The ergodicity problem in economics", Nature Physics 15
  - Doob (1953), théorème d'arrêt optimal

Toutes les quantités monétaires sont exprimées en **big blinds (bb)**.
Les winrates et écarts-types sont en **bb/100 mains**, convention standard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

__all__ = [
    "BankrollProfile",
    "risk_of_ruin",
    "bankroll_for_ror",
    "kelly_fraction",
    "growth_rate",
    "ergodic_penalty",
    "should_take_shot",
    "confidence_interval_winrate",
    "hands_for_significance",
]

# Seuil de risque de ruine considéré comme professionnellement acceptable.
DEFAULT_TARGET_ROR: Final[float] = 0.01


class BankrollError(ValueError):
    """Erreur de paramétrage bankroll (valeurs hors domaine de définition)."""


@dataclass(frozen=True, slots=True)
class BankrollProfile:
    """Profil de bankroll d'un joueur.

    Parameters
    ----------
    winrate_bb100 : float
        Winrate estimé, en bb/100 mains. Doit être > 0 pour que le RoR
        soit défini (un winrate <= 0 implique une ruine presque sûre).
    stddev_bb100 : float
        Écart-type des résultats, en bb/100 mains. Références empiriques :
        NLHE full-ring 60-80, 6-max 75-120, PLO 100-160.
    bankroll_bb : float
        Bankroll totale exprimée en big blinds.
    """

    winrate_bb100: float
    stddev_bb100: float
    bankroll_bb: float

    def __post_init__(self) -> None:
        if self.stddev_bb100 <= 0.0:
            raise BankrollError("stddev_bb100 doit être strictement positif.")
        if self.bankroll_bb <= 0.0:
            raise BankrollError("bankroll_bb doit être strictement positif.")

    @property
    def buyins(self) -> float:
        """Nombre de buy-ins de 100 bb."""
        return self.bankroll_bb / 100.0


def risk_of_ruin(winrate_bb100: float, stddev_bb100: float, bankroll_bb: float) -> float:
    r"""Risque de ruine par la forme exponentielle de Malmuth.

    .. math::
        \mathrm{RoR} = \exp\!\left(-\frac{2\,\mu\,B}{\sigma^2}\right)

    où :math:`\mu` est le winrate, :math:`\sigma` l'écart-type (mêmes unités
    de temps) et :math:`B` la bankroll.

    Returns
    -------
    float
        Probabilité de ruine dans [0, 1]. Vaut exactement 1.0 si
        ``winrate_bb100 <= 0`` (ruine presque sûre en temps infini).

    Examples
    --------
    >>> round(risk_of_ruin(5.0, 100.0, 3000.0), 6)
    0.049787
    """
    if stddev_bb100 <= 0.0:
        raise BankrollError("stddev_bb100 doit être strictement positif.")
    if bankroll_bb <= 0.0:
        return 1.0
    if winrate_bb100 <= 0.0:
        return 1.0

    exponent = -2.0 * winrate_bb100 * bankroll_bb / (stddev_bb100**2)
    # exp d'un grand négatif underflow proprement vers 0.0 — comportement voulu.
    return float(math.exp(exponent))


def bankroll_for_ror(
    winrate_bb100: float,
    stddev_bb100: float,
    target_ror: float = DEFAULT_TARGET_ROR,
) -> float:
    r"""Bankroll minimale (bb) pour atteindre un risque de ruine cible.

    Inversion analytique de :func:`risk_of_ruin` :

    .. math::
        B = -\frac{\sigma^2 \ln(\mathrm{RoR})}{2\mu}

    Examples
    --------
    >>> round(bankroll_for_ror(5.0, 100.0, 0.01), 1)
    4605.2
    """
    if not (0.0 < target_ror < 1.0):
        raise BankrollError("target_ror doit être dans (0, 1).")
    if winrate_bb100 <= 0.0:
        return math.inf
    if stddev_bb100 <= 0.0:
        raise BankrollError("stddev_bb100 doit être strictement positif.")

    return float(-(stddev_bb100**2) * math.log(target_ror) / (2.0 * winrate_bb100))


def kelly_fraction(winrate_bb100: float, stddev_bb100: float, fraction: float = 0.5) -> float:
    r"""Fraction de Kelly (full ou fractionnaire) pour un processus gaussien.

    Pour un processus à dérive :math:`\mu` et variance :math:`\sigma^2`, le
    critère de Kelly continu donne :math:`f^* = \mu / \sigma^2`.

    Parameters
    ----------
    fraction : float
        Multiplicateur appliqué au full-Kelly. ``0.5`` (half-Kelly) conserve
        exactement **75 %** du taux de croissance pour **50 %** de la variance —
        c'est le point d'exploitation standard.

    Notes
    -----
    Démonstration du 75 % : :math:`g(f) = f\mu - f^2\sigma^2/2`, donc
    :math:`g(f^*) = \mu^2/(2\sigma^2)` et
    :math:`g(f^*/2) = 3\mu^2/(8\sigma^2) = 0.75\,g(f^*)`.
    """
    if stddev_bb100 <= 0.0:
        raise BankrollError("stddev_bb100 doit être strictement positif.")
    if not (0.0 < fraction <= 1.0):
        raise BankrollError("fraction doit être dans (0, 1].")

    return float(fraction * winrate_bb100 / (stddev_bb100**2))


def growth_rate(f: float, winrate_bb100: float, stddev_bb100: float) -> float:
    r"""Taux de croissance logarithmique :math:`g(f) = f\mu - f^2\sigma^2/2`."""
    return float(f * winrate_bb100 - 0.5 * (f**2) * (stddev_bb100**2))


def ergodic_penalty(winrate_bb100: float, stddev_bb100: float, bankroll_bb: float) -> float:
    r"""Écart entre moyenne d'ensemble et moyenne temporelle (Peters, 2019).

    Pour un processus multiplicatif, le taux de croissance **temporel** que
    subit réellement une trajectoire unique est

    .. math::
        g = \frac{\mu}{B} - \frac{\sigma^2}{2B^2}

    alors que la moyenne d'ensemble vaut :math:`\mu / B`. La pénalité
    retournée est le terme :math:`\sigma^2 / (2B^2)`.

    C'est ce terme qui explique qu'un spot **à EV positive** puisse être
    à refuser : si la pénalité dépasse le gain d'EV relatif, prendre le
    spot réduit ton taux de croissance réel.
    """
    if bankroll_bb <= 0.0:
        raise BankrollError("bankroll_bb doit être strictement positif.")
    return float((stddev_bb100**2) / (2.0 * bankroll_bb**2))


def should_take_shot(
    profile: BankrollProfile,
    shot_buyin_bb: float,
    shot_winrate_bb100: float,
    shot_stddev_bb100: float,
    max_acceptable_ror: float = 0.05,
) -> tuple[bool, dict[str, float]]:
    """Décide d'un shot à un niveau supérieur, sous critère ergodique.

    Le shot est accepté si et seulement si les deux conditions sont vraies :

    1. Le risque de ruine au nouveau niveau reste sous ``max_acceptable_ror``.
    2. Le taux de croissance **temporel** au nouveau niveau dépasse celui
       du niveau actuel (critère de Peters, et non simple comparaison d'EV).

    Returns
    -------
    (bool, dict)
        Décision et métriques détaillées pour affichage.
    """
    if shot_buyin_bb <= 0.0:
        raise BankrollError("shot_buyin_bb doit être strictement positif.")

    # Le bankroll exprimé en bb du nouveau niveau : on renormalise.
    scale = shot_buyin_bb / 100.0
    bankroll_at_shot = profile.bankroll_bb / scale

    ror_shot = risk_of_ruin(shot_winrate_bb100, shot_stddev_bb100, bankroll_at_shot)

    g_current = growth_rate(
        kelly_fraction(profile.winrate_bb100, profile.stddev_bb100, 0.5),
        profile.winrate_bb100,
        profile.stddev_bb100,
    )
    g_shot = growth_rate(
        kelly_fraction(shot_winrate_bb100, shot_stddev_bb100, 0.5),
        shot_winrate_bb100,
        shot_stddev_bb100,
    )

    decision = bool(ror_shot <= max_acceptable_ror and g_shot > g_current)

    return decision, {
        "ror_current": risk_of_ruin(
            profile.winrate_bb100, profile.stddev_bb100, profile.bankroll_bb
        ),
        "ror_shot": ror_shot,
        "growth_current": g_current,
        "growth_shot": g_shot,
        "bankroll_at_shot_bb": bankroll_at_shot,
        "buyins_at_shot": bankroll_at_shot / 100.0,
        "ergodic_penalty": ergodic_penalty(
            shot_winrate_bb100, shot_stddev_bb100, bankroll_at_shot
        ),
    }


def confidence_interval_winrate(
    observed_bb100: float,
    stddev_bb100: float,
    n_hands: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    r"""Intervalle de confiance du winrate observé.

    L'erreur-type du winrate sur :math:`n` mains vaut
    :math:`\sigma / \sqrt{n/100}` puisque :math:`\sigma` est exprimé par
    100 mains.

    Examples
    --------
    Sur 10 000 mains avec sigma=100, la demi-largeur à 95 % vaut 19.6 bb/100 —
    ce qui montre qu'un "winrate" mesuré sur 10k mains est quasiment sans
    information.

    >>> lo, hi = confidence_interval_winrate(5.0, 100.0, 10_000)
    >>> round(hi - lo, 1)
    39.2
    """
    if n_hands <= 0:
        raise BankrollError("n_hands doit être strictement positif.")
    if not (0.0 < confidence < 1.0):
        raise BankrollError("confidence doit être dans (0, 1).")

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    se = stddev_bb100 / math.sqrt(n_hands / 100.0)
    return (observed_bb100 - z * se, observed_bb100 + z * se)


def hands_for_significance(
    true_winrate_bb100: float,
    stddev_bb100: float,
    confidence: float = 0.95,
    power: float = 0.80,
) -> int:
    r"""Nombre de mains pour distinguer un winrate de zéro (test bilatéral).

    .. math::
        n = 100 \left(\frac{(z_{1-\alpha/2} + z_{\text{power}})\,\sigma}{\mu}\right)^2

    Le facteur 100 vient de l'unité bb/100.

    Notes
    -----
    Avec mu=5 bb/100 et sigma=100, il faut environ **314 000 mains**. C'est le
    chiffre qui justifie mathématiquement pourquoi "je suis breakeven sur
    30k mains" ne veut rien dire.
    """
    if true_winrate_bb100 <= 0.0:
        raise BankrollError("true_winrate_bb100 doit être strictement positif.")

    z_alpha = float(stats.norm.ppf(0.5 + confidence / 2.0))
    z_power = float(stats.norm.ppf(power))
    n_blocks = ((z_alpha + z_power) * stddev_bb100 / true_winrate_bb100) ** 2
    return int(math.ceil(n_blocks * 100.0))


def drawdown_quantile(
    winrate_bb100: float,
    stddev_bb100: float,
    n_hands: int,
    quantile: float = 0.95,
    n_sims: int = 20_000,
    seed: int | None = 42,
) -> float:
    """Quantile du drawdown maximal par simulation Monte-Carlo.

    Le drawdown maximal d'une marche aléatoire à dérive n'a pas de forme
    close simple utilisable ; la simulation est la voie honnête.

    Returns
    -------
    float
        Drawdown (en bb, valeur positive) non dépassé avec la probabilité
        ``quantile``.
    """
    if n_hands <= 0:
        raise BankrollError("n_hands doit être strictement positif.")

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n_hands // 100)
    # Chaque bloc de 100 mains : gaussienne (mu, sigma).
    increments = rng.normal(winrate_bb100, stddev_bb100, size=(n_sims, n_blocks))
    equity = np.cumsum(increments, axis=1)
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = running_max - equity
    max_dd = drawdowns.max(axis=1)
    return float(np.quantile(max_dd, quantile))
