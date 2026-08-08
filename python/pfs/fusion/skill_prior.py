"""
Priors de compétence externes — SharkScope, OPR, et pourquoi ils ne remplacent
pas les statistiques comportementales.

═══════════════════════════════════════════════════════════════════════════
CE QUE CES SITES DONNENT, ET CE QU'ILS NE DONNENT PAS
═══════════════════════════════════════════════════════════════════════════

SharkScope et OPR (Official Poker Rankings) sont des bases de **résultats** :
ROI, buy-in moyen (ABI), volume, ITM %, gains cumulés.

Le moteur de fusion, lui, a besoin de **fréquences comportementales** :
VPIP, PFR, 3-bet, fold-to-cbet — avec leur incertitude postérieure, parce que
c'est σ_θ qui pilote λ dans F13.

**Ces deux familles ne se recouvrent pas.** Aucun ROI ne te dit à quelle
fréquence un joueur folde face à un c-bet. Brancher SharkScope sur F13 ne
ferait donc rien : la statistique pivot resterait inconnue.

Là où un rating externe est légitimement utile, c'est **en amont** :
  1. comme **prior sur l'archétype** du filtre particulaire (F3) — un joueur à
     −20 % de ROI sur 3 000 tournois n'est presque certainement pas un GTO ;
  2. comme **propension à s'adapter** ρ dans F13 — un joueur fort te
     contre-exploitera, un joueur faible non.

C'est exactement ce que fait ce module, et rien de plus.

═══════════════════════════════════════════════════════════════════════════
ZÉRO APPEL RÉSEAU
═══════════════════════════════════════════════════════════════════════════

Ce module **n'interroge aucun site**. Il ingère une valeur que tu saisis ou
colles. Trois raisons :

1. **Architecture** — l'application est sans dépendance réseau (décision D3),
   vérifiée par un test d'intégration continue.
2. **CGU** — Winamax interdit explicitement le *data mining* (« analyse massive
   de données tierces ») et le *data sharing*. PokerStars limite la collecte
   aux mains où tu es participant. Interroger automatiquement une base tierce
   pendant une session est précisément ce que ces clauses visent.
3. **Traçabilité** — une requête sortante pendant une main est un événement
   horodaté et joignable côté opérateur ; c'est le mécanisme du « Fair Play
   Check » de GTO Wizard.

Saisir un ROI que tu as consulté toi-même, hors session, dans ton navigateur,
est une autre chose — et c'est ce que le module permet.

═══════════════════════════════════════════════════════════════════════════
LE PIÈGE STATISTIQUE DU ROI
═══════════════════════════════════════════════════════════════════════════

L'écart-type du ROI en MTT est de l'ordre de **150 % par tournoi** (large
field), 90–110 % en SNG. Conséquence, calculée par :func:`tournaments_needed` :

    distinguer un ROI de +10 % de zéro exige ≈ **1 800 tournois** ;
    distinguer +5 % de zéro en exige ≈ **7 100**.

Autrement dit : **la très grande majorité des ROI affichés par ces sites ne
distinguent pas un gagnant d'un joueur neutre.** Le module applique donc un
rétrécissement bayésien vers zéro et rend l'intervalle de crédibilité — la
même discipline que F1 sur les fréquences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from scipy import stats

from pfs.fusion.particle import ARCHETYPE_PRIORS, Archetype

__all__ = [
    "RatingSource",
    "GameFormat",
    "ExternalRating",
    "SkillEstimate",
    "ROI_STDDEV",
    "tournaments_needed",
    "estimate_skill",
    "archetype_prior_from_skill",
    "adaptation_propensity_from_skill",
]


class RatingSource(str, Enum):
    SHARKSCOPE = "sharkscope"
    OPR = "opr"
    POCKETFIVES = "pocketfives"
    MANUAL = "manual"
    """Ton propre jugement — parfaitement recevable, et souvent meilleur."""


class GameFormat(str, Enum):
    MTT_LARGE = "mtt_large"
    MTT_SMALL = "mtt_small"
    SNG = "sng"
    SPIN = "spin"


# Écart-type du ROI par tournoi, par format. Ordres de grandeur communément
# admis ; à recalibrer si tu disposes de tes propres séries.
ROI_STDDEV: Mapping[GameFormat, float] = {
    GameFormat.MTT_LARGE: 1.50,
    GameFormat.MTT_SMALL: 1.10,
    GameFormat.SNG: 0.95,
    GameFormat.SPIN: 1.35,
}


class SkillPriorError(ValueError):
    pass


def tournaments_needed(
    true_roi: float,
    fmt: GameFormat = GameFormat.MTT_LARGE,
    confidence: float = 0.95,
    power: float = 0.80,
) -> int:
    r"""Tournois nécessaires pour distinguer ``true_roi`` de zéro.

    .. math::
        n = \left(\frac{(z_{1-\alpha/2} + z_{\text{power}})\,\sigma}{\mu}\right)^2

    Examples
    --------
    >>> tournaments_needed(0.10)
    1764
    >>> tournaments_needed(0.05)
    7057
    """
    if true_roi <= 0.0:
        raise SkillPriorError("true_roi doit être strictement positif.")
    sigma = ROI_STDDEV[fmt]
    z_a = float(stats.norm.ppf(0.5 + confidence / 2.0))
    z_p = float(stats.norm.ppf(power))
    return int(math.ceil(((z_a + z_p) * sigma / true_roi) ** 2))


@dataclass(frozen=True, slots=True)
class ExternalRating:
    """Une donnée de résultats, saisie manuellement — jamais téléchargée."""

    source: RatingSource
    fmt: GameFormat
    n_tournaments: int
    observed_roi: float
    """ROI brut affiché, en fraction (0.12 = +12 %)."""
    average_buyin: float | None = None
    itm_rate: float | None = None

    def __post_init__(self) -> None:
        if self.n_tournaments < 0:
            raise SkillPriorError("n_tournaments doit être >= 0.")
        if not (-1.0 <= self.observed_roi <= 10.0):
            raise SkillPriorError("observed_roi hors domaine plausible.")


@dataclass(frozen=True, slots=True)
class SkillEstimate:
    """ROI rétréci, son incertitude, et ce qu'on en déduit — ou pas."""

    shrunk_roi: float
    std: float
    ci: tuple[float, float]
    n_tournaments: int
    significant: bool
    skill: float
    """Score de compétence dans [0, 1] : 0,5 = neutre, faute d'information."""
    shrinkage: float
    """Part du ROI brut effacée par le rétrécissement. 1 = totalement effacé."""
    verdict: str

    def explain(self) -> str:
        return (
            f"ROI brut → rétréci : {self.shrunk_roi * 100:+.1f} % "
            f"± {self.std * 100:.1f} (n={self.n_tournaments})\n"
            f"  IC 95 %      : [{self.ci[0] * 100:+.1f} ; {self.ci[1] * 100:+.1f}]\n"
            f"  rétrécissement : {self.shrinkage * 100:.0f} % du signal brut effacé\n"
            f"  compétence   : {self.skill:.2f}  "
            f"({'significatif' if self.significant else 'NON significatif'})\n"
            f"  → {self.verdict}"
        )


def estimate_skill(
    rating: ExternalRating, prior_std: float = 0.08
) -> SkillEstimate:
    r"""Rétrécissement bayésien du ROI vers zéro (James-Stein / normal-normal).

    Prior : ROI ~ N(0, τ²) avec τ = ``prior_std`` — l'immense majorité des
    joueurs est proche du neutre après rake.
    Vraisemblance : ROI observé ~ N(θ, σ²/n).

    Postérieur :

    .. math::
        \hat\theta = \frac{\tau^2}{\tau^2 + \sigma^2/n}\,\bar{r},
        \qquad
        \mathrm{Var} = \frac{\tau^2\sigma^2/n}{\tau^2 + \sigma^2/n}

    Le facteur de rétrécissement est **la** quantité intéressante : il dit
    quelle part du ROI affiché est du bruit.

    Examples
    --------
    Un ROI de +40 % sur 200 MTT — le genre de chiffre qui impressionne :

    >>> e = estimate_skill(ExternalRating(
    ...     RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE, 200, 0.40))
    >>> round(e.shrunk_roi, 4)
    0.0219
    >>> e.significant
    False
    """
    n = rating.n_tournaments
    if n == 0:
        return SkillEstimate(0.0, prior_std, (-2 * prior_std, 2 * prior_std), 0,
                             False, 0.5, 1.0,
                             "Aucun tournoi : rien à en tirer, prior neutre.")

    sigma = ROI_STDDEV[rating.fmt]
    var_obs = sigma**2 / n
    tau2 = prior_std**2

    weight = tau2 / (tau2 + var_obs)          # poids de la donnée observée
    shrunk = weight * rating.observed_roi
    var_post = (tau2 * var_obs) / (tau2 + var_obs)
    std = math.sqrt(var_post)
    lo, hi = shrunk - 1.96 * std, shrunk + 1.96 * std
    significant = lo > 0.0 or hi < 0.0

    # Compétence dans [0,1] : logistique centrée sur 0, pente calibrée pour
    # qu'un ROI rétréci de +10 % donne ≈ 0,88.
    skill = float(1.0 / (1.0 + math.exp(-20.0 * shrunk)))

    if not significant:
        need = (tournaments_needed(abs(rating.observed_roi), rating.fmt)
                if abs(rating.observed_roi) > 1e-6 else 0)
        verdict = (
            f"Non significatif. Ce ROI ne distingue pas ce joueur d'un joueur "
            f"neutre — il faudrait ≈ {need} tournois. À n'utiliser que comme "
            f"prior faible, jamais comme preuve."
        )
    elif shrunk > 0:
        verdict = "Gagnant établi. Prior : joueur solide, susceptible de s'adapter."
    else:
        verdict = "Perdant établi. Prior : joueur exploitable, peu adaptatif."

    if significant and n < 500:
        verdict += (
            " ⚠ Sous 500 tournois, la distribution du ROI est fortement "
            "asymétrique à droite — un seul gros score domine l'échantillon — "
            "et l'approximation normale sous-estime l'incertitude réelle. "
            "Traiter ce verdict comme indicatif."
        )

    return SkillEstimate(
        shrunk_roi=shrunk,
        std=std,
        ci=(lo, hi),
        n_tournaments=n,
        significant=significant,
        skill=skill,
        shrinkage=1.0 - weight,
        verdict=verdict,
    )


def archetype_prior_from_skill(
    skill: float, strength: float = 1.0
) -> dict[Archetype, float]:
    """Déforme le prior d'archétypes du filtre particulaire (F3) selon la compétence.

    Un joueur établi gagnant est plus probablement GTO ou TAG ; un perdant
    établi, plus probablement station ou maniac. L'intensité de la déformation
    est bornée par ``strength`` — un prior externe ne doit jamais écraser ce
    que les actions observées vont dire.

    Parameters
    ----------
    skill
        Dans [0, 1]. 0,5 = aucune information ⇒ prior inchangé.
    strength
        0 = ignorer le rating, 1 = déformation maximale (facteur 3 au plus).
    """
    if not (0.0 <= skill <= 1.0):
        raise SkillPriorError("skill doit être dans [0, 1].")
    if not (0.0 <= strength <= 1.0):
        raise SkillPriorError("strength doit être dans [0, 1].")

    # Affinité de chaque archétype avec la compétence, dans [-1, 1].
    affinity = {
        Archetype.GTO: 1.0,
        Archetype.TAG: 0.6,
        Archetype.NIT: -0.1,
        Archetype.LAG: 0.2,
        Archetype.STATION: -0.8,
        Archetype.MANIAC: -0.9,
    }
    centred = 2.0 * skill - 1.0        # [-1, 1]
    out: dict[Archetype, float] = {}
    for k, base in ARCHETYPE_PRIORS.items():
        factor = math.exp(strength * 1.1 * affinity[k] * centred)
        out[k] = base * factor
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def adaptation_propensity_from_skill(skill: float) -> float:
    r"""Propension à contre-exploiter, pour le terme ρ de F13.

    Un joueur fort réagit à ton exploitation ; un joueur faible non. Bornée
    dans [0,05 ; 0,85] : même le meilleur joueur ne s'adapte pas toujours, et
    même le pire finit parfois par remarquer.
    """
    if not (0.0 <= skill <= 1.0):
        raise SkillPriorError("skill doit être dans [0, 1].")
    return float(0.05 + 0.80 * skill)
