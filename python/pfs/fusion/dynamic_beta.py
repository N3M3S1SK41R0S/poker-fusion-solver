"""
Poker Fusion Solver — F1 : tracking adverse non-stationnaire.

**Correction du modèle v1.** Le prototype `OpponentKalmanTracker` appliquait un
filtre de Kalman linéaire-gaussien à des observations **binaires** (1 si VPIP
sinon 0). C'est invalide :

  1. La variance d'une Bernoulli dépend de l'état (p(1-p)) ; un R constant est
     structurellement faux.
  2. Rien ne contraint l'état à [0, 1] — VPIP peut devenir négatif.
  3. Le conjugué exact existe en forme close (Beta-Bernoulli) : recourir à une
     approximation gaussienne est une perte nette.

Le modèle correct est un **Beta-Binomial dynamique à facteur d'actualisation**
(West & Harrison, 1997, *Bayesian Forecasting and Dynamic Models*, ch. 14),
qui est l'analogue exact du bruit de processus Q du Kalman pour une famille
conjuguée.

Le filtre de Kalman reste correct — et est conservé ailleurs — pour les
quantités **continues** (bet-size moyen, agression en bb).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from scipy import stats

__all__ = ["DynamicBetaTracker", "StatBelief", "OpponentProfile", "GTO_BASELINES"]

# Prior de Jeffreys pour une Bernoulli : Beta(1/2, 1/2). Non informatif et
# invariant par reparamétrisation, contrairement à Beta(1,1).
JEFFREYS_A: Final[float] = 0.5
JEFFREYS_B: Final[float] = 0.5

# Baselines GTO 6-max 100bb (approximations de référence, à recalibrer sur
# ton pool réel — ce sont des points de comparaison, pas des vérités).
GTO_BASELINES: Final[dict[str, float]] = {
    "vpip": 0.24,
    "pfr": 0.19,
    "three_bet": 0.075,
    "fold_to_three_bet": 0.55,
    "cbet_flop": 0.62,
    "fold_to_cbet": 0.55,
    "wtsd": 0.27,
    "af": 0.50,
}


@dataclass(frozen=True, slots=True)
class StatBelief:
    """Croyance postérieure complète sur une statistique binaire."""

    name: str
    alpha: float
    beta: float
    n_observations: int

    @property
    def mean(self) -> float:
        r""":math:`\hat\theta = \alpha / (\alpha + \beta)`."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        r""":math:`\mathrm{Var}[\theta] = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`."""
        s = self.alpha + self.beta
        return (self.alpha * self.beta) / (s * s * (s + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Intervalle de crédibilité **exact** (quantiles Beta, pas normal).

        Sur petits échantillons l'approximation normale déborde de [0,1] et
        sous-estime l'asymétrie — d'où l'usage des quantiles exacts.
        """
        if not (0.0 < level < 1.0):
            raise ValueError("level doit être dans (0, 1).")
        tail = (1.0 - level) / 2.0
        lo = float(stats.beta.ppf(tail, self.alpha, self.beta))
        hi = float(stats.beta.ppf(1.0 - tail, self.alpha, self.beta))
        return (lo, hi)

    def deviation_z(self, baseline: float) -> float:
        """Écart à la baseline, en unités d'écart-type postérieur."""
        if self.std == 0.0:
            return 0.0
        return (self.mean - baseline) / self.std

    def is_exploitable(self, baseline: float, z_threshold: float = 1.96) -> bool:
        """Vrai si l'écart à la baseline est statistiquement significatif.

        C'est **le** garde-fou du système : un HUD classique affiche
        "VPIP 35 %" et t'incite à exploiter ; ce test répond "l'IC95 va de
        21 % à 51 %, il englobe le TAG standard — n'exploite pas".
        """
        return abs(self.deviation_z(baseline)) > z_threshold

    def exploitation_weight(self, baseline: float, z_c: float = 1.96) -> float:
        r"""Poids d'exploitation :math:`\lambda_{\text{brut}} \in [0, 1]`.

        .. math::
            \lambda = \Phi\!\left(\frac{|\hat\theta - \theta_0| - z_c\,\sigma}{\sigma}\right)

        Alimente directement la méta-fusion F13.
        """
        sigma = self.std
        if sigma <= 0.0:
            return 0.0
        excess = abs(self.mean - baseline) - z_c * sigma
        return float(stats.norm.cdf(excess / sigma))

    def hands_until_significant(self, baseline: float, z_threshold: float = 1.96) -> int:
        """Mains supplémentaires nécessaires pour que l'écart devienne significatif.

        Utilise l'approximation :math:`\\sigma \\approx \\sqrt{p(1-p)/n}`, donc
        :math:`n_{\\text{requis}} = p(1-p)\\,(z/\\delta)^2`.
        """
        p = self.mean
        delta = abs(p - baseline)
        if delta < 1e-12:
            return -1  # jamais, à cet écart
        n_required = p * (1.0 - p) * (z_threshold / delta) ** 2
        return max(0, int(math.ceil(n_required)) - self.n_observations)


class DynamicBetaTracker:
    """Tracker Beta-Binomial dynamique pour une statistique binaire.

    Parameters
    ----------
    name : str
        Identifiant de la statistique (``"vpip"``, ``"cbet_flop"``…).
    discount : float
        Facteur d'actualisation :math:`\\delta \\in (0, 1]`. La mémoire
        effective vaut :math:`n_{\\text{eff}} = 1/(1-\\delta)`.
        ``0.99`` → ~100 mains ; ``0.995`` → ~200 mains ; ``1.0`` → mémoire
        infinie (adversaire supposé stationnaire).
    prior_mean : float | None
        Si fourni, le prior est centré sur cette valeur avec un poids
        ``prior_strength`` — utile pour démarrer sur la baseline GTO plutôt
        que sur un prior non informatif.
    prior_strength : float
        Poids du prior, en "mains équivalentes".

    Examples
    --------
    >>> t = DynamicBetaTracker("vpip", discount=0.99)
    >>> for i in range(40):
    ...     t.update(i < 14)          # 14 VPIP sur 40 mains
    >>> b = t.belief
    >>> round(b.mean, 5)
    0.35366
    """

    __slots__ = ("_name", "_delta", "_a0", "_b0", "_a", "_b", "_n")

    def __init__(
        self,
        name: str,
        discount: float = 0.99,
        prior_mean: float | None = None,
        prior_strength: float = 1.0,
    ) -> None:
        if not (0.0 < discount <= 1.0):
            raise ValueError("discount doit être dans (0, 1].")
        if prior_strength <= 0.0:
            raise ValueError("prior_strength doit être strictement positif.")

        self._name = name
        self._delta = float(discount)

        if prior_mean is None:
            self._a0, self._b0 = JEFFREYS_A, JEFFREYS_B
        else:
            if not (0.0 < prior_mean < 1.0):
                raise ValueError("prior_mean doit être dans (0, 1).")
            self._a0 = prior_mean * prior_strength
            self._b0 = (1.0 - prior_mean) * prior_strength

        self._a = self._a0
        self._b = self._b0
        self._n = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def effective_sample_size(self) -> float:
        r""":math:`n_{\text{eff}} = 1/(1-\delta)`, ou +inf si delta == 1."""
        if self._delta >= 1.0:
            return math.inf
        return 1.0 / (1.0 - self._delta)

    def update(self, observed: bool) -> StatBelief:
        r"""Une observation binaire.

        Évolution (oubli exponentiel vers le prior) puis mise à jour de Bayes :

        .. math::
            a' = \delta a + (1-\delta) a_0, \quad
            b' = \delta b + (1-\delta) b_0

        .. math::
            a \leftarrow a' + x, \quad b \leftarrow b' + (1 - x)
        """
        # Étape d'évolution : le prior "revient" à mesure qu'on oublie.
        self._a = self._delta * self._a + (1.0 - self._delta) * self._a0
        self._b = self._delta * self._b + (1.0 - self._delta) * self._b0

        # Étape de mise à jour (conjuguée, exacte).
        if observed:
            self._a += 1.0
        else:
            self._b += 1.0

        self._n += 1
        return self.belief

    def update_batch(self, successes: int, trials: int) -> StatBelief:
        """Mise à jour groupée (import d'un hand-history, par exemple)."""
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("0 <= successes <= trials requis.")
        for i in range(trials):
            self.update(i < successes)
        return self.belief

    @property
    def belief(self) -> StatBelief:
        return StatBelief(
            name=self._name, alpha=self._a, beta=self._b, n_observations=self._n
        )

    def reset(self) -> None:
        self._a, self._b, self._n = self._a0, self._b0, 0


@dataclass(slots=True)
class OpponentProfile:
    """Profil complet d'un adversaire : un tracker par statistique.

    Le pseudo n'est **jamais** stocké en clair — seulement une clé de hash
    stable. C'est une donnée personnelle (RGPD), et le data-sharing est
    explicitement interdit par les CGU Winamax.
    """

    player_key: str
    discount: float = 0.99
    trackers: dict[str, DynamicBetaTracker] = field(default_factory=dict)

    def tracker(self, stat: str) -> DynamicBetaTracker:
        """Récupère (ou crée) le tracker d'une statistique, amorcé sur la baseline GTO."""
        if stat not in self.trackers:
            self.trackers[stat] = DynamicBetaTracker(
                stat,
                discount=self.discount,
                prior_mean=GTO_BASELINES.get(stat),
                prior_strength=2.0,
            )
        return self.trackers[stat]

    def observe(self, stat: str, occurred: bool) -> StatBelief:
        return self.tracker(stat).update(occurred)

    def exploitable_stats(self, z_threshold: float = 1.96) -> dict[str, StatBelief]:
        """Statistiques dont l'écart au GTO est statistiquement significatif.

        C'est la seule liste sur laquelle il est défendable de dévier.
        """
        out: dict[str, StatBelief] = {}
        for stat, tr in self.trackers.items():
            base = GTO_BASELINES.get(stat)
            if base is None:
                continue
            b = tr.belief
            if b.is_exploitable(base, z_threshold):
                out[stat] = b
        return out

    def summary(self) -> str:
        """Rendu texte pour le HUD (couche 2 de la progressive disclosure)."""
        lines = [f"Adversaire {self.player_key[:8]}"]
        for stat, tr in sorted(self.trackers.items()):
            b = tr.belief
            lo, hi = b.credible_interval()
            base = GTO_BASELINES.get(stat)
            flag = ""
            if base is not None and b.is_exploitable(base):
                flag = f"  ⚠ dévie de {(b.mean - base) * 100:+.1f} pts"
            lines.append(
                f"  {stat:<18} {b.mean * 100:5.1f}%  "
                f"IC95 [{lo * 100:4.1f} , {hi * 100:4.1f}]  n={b.n_observations}{flag}"
            )
        return "\n".join(lines)
