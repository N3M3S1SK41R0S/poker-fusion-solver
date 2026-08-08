"""
Poker Fusion Solver — F13 : méta-fusion, arbitrage GTO ↔ exploitation.

C'est le cœur du système. Les 12 autres fusions n'existent que pour alimenter
celle-ci.

Le problème résolu : tous les solveurs existants donnent la stratégie GTO,
aucun ne dit **quand** ni **de combien** s'en écarter. C'est pourtant la
question centrale du poker appliqué — le GTO ne gagne pas d'argent contre les
faibles, et l'exploitation naïve perd contre les forts.

Sources
-------
- von Neumann (1928) ; Nash (1950) — socle minimax / équilibre
- Johanson, Zinkevich & Bowling (2008), "Computing Robust Counter-Strategies"
- **Ganzfried & Sandholm (2015), "Safe Opponent Exploitation", ACM TEAC 3(2)**
  → fournit la garantie de sûreté : on ne peut pas perdre plus que le "cadeau"
    déjà encaissé grâce aux erreurs adverses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from scipy import stats

__all__ = [
    "Action",
    "ActionDistribution",
    "MentalState",
    "FusionInput",
    "FusionResult",
    "arbitrate",
]


class Action(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALLIN = "allin"


class MentalState(str, Enum):
    """États cachés du HMM (F2)."""

    SOLID = "solid"
    LOOSE = "loose"
    TILT = "tilt"


# Propension à s'adapter à ton exploitation, par état mental.
# Un joueur SOLID te contre-exploitera ; un joueur en TILT non.
ADAPTATION_PROPENSITY: Mapping[MentalState, float] = {
    MentalState.SOLID: 0.60,
    MentalState.LOOSE: 0.20,
    MentalState.TILT: 0.05,
}


@dataclass(frozen=True, slots=True)
class ActionDistribution:
    """Distribution de probabilité sur les actions légales."""

    probs: Mapping[Action, float]

    def __post_init__(self) -> None:
        total = sum(self.probs.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Les probabilités doivent sommer à 1, obtenu {total:.9f}.")
        if any(p < -1e-12 for p in self.probs.values()):
            raise ValueError("Probabilité négative interdite.")

    def get(self, a: Action) -> float:
        return self.probs.get(a, 0.0)

    def blend(self, other: "ActionDistribution", weight: float) -> "ActionDistribution":
        r"""Interpolation convexe :math:`(1-w)\,\text{self} + w\,\text{other}`.

        L'interpolation linéaire est licite : le simplexe est convexe, donc
        le résultat est une distribution valide. C'est aussi ce qui garantit
        que l'exploitabilité du mélange est bornée par le mélange des
        exploitabilités (convexité de l'exploitabilité en la stratégie).
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError("weight doit être dans [0, 1].")
        keys = set(self.probs) | set(other.probs)
        blended = {k: (1.0 - weight) * self.get(k) + weight * other.get(k) for k in keys}
        # Renormalisation défensive contre l'accumulation d'erreurs flottantes.
        s = sum(blended.values())
        return ActionDistribution({k: v / s for k, v in blended.items()})

    def top(self) -> tuple[Action, float]:
        a = max(self.probs.items(), key=lambda kv: kv[1])
        return a[0], a[1]

    def entropy_bits(self) -> float:
        r""":math:`H = -\sum p \log_2 p`. Mesure la mixité de la stratégie."""
        return -sum(p * math.log2(p) for p in self.probs.values() if p > 0.0)


@dataclass(frozen=True, slots=True)
class FusionInput:
    """Tout ce dont l'arbitre a besoin, en provenance des fusions amont."""

    gto: ActionDistribution
    """σ_GTO — issu du blueprint (F8)."""

    best_response: ActionDistribution
    """σ_BR(m̂) — meilleure réponse au modèle adverse estimé (F1–F3)."""

    deviation: float
    """|θ̂ − θ_GTO| — écart observé sur la stat pertinente (F1)."""

    deviation_std: float
    """σ_θ — écart-type postérieur de cette stat (F1)."""

    mental_state_probs: Mapping[MentalState, float]
    """P(S | observations) — issu du forward du HMM (F2)."""

    ev_gto: float = 0.0
    """EV de σ_GTO contre le modèle adverse, en bb."""

    ev_best_response: float = 0.0
    """EV de σ_BR contre le modèle adverse, en bb."""

    exploitability_gto: float = 0.0
    """Exploitabilité de σ_GTO, en bb/100 (≈0 pour un vrai équilibre)."""

    exploitability_br: float = 0.0
    """Exploitabilité de σ_BR, en bb/100 — c'est ce que tu risques."""

    realized_gift: float = math.inf
    """
    Gain cumulé déjà encaissé grâce aux erreurs de cet adversaire, en bb/100.
    Borne de safe exploitation (Ganzfried & Sandholm 2015) : on n'accepte
    une exploitabilité supplémentaire qu'à hauteur de ce "cadeau".
    ``inf`` désactive la borne (mode étude).
    """

    z_critical: float = 1.96
    """Seuil de significativité. 1.96 → 95 %, 2.58 → 99 %."""


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Sortie de l'arbitrage, traçable de bout en bout."""

    strategy: ActionDistribution
    lambda_raw: float
    lambda_final: float
    adaptation_risk: float
    z_score: float
    significant: bool
    ev_gain_vs_gto: float
    exploitability: float
    exploitability_capped: bool
    rationale: str

    def explain(self) -> str:
        act, p = self.strategy.top()
        return (
            f"λ = {self.lambda_final:.3f} "
            f"(brut {self.lambda_raw:.3f}, risque d'adaptation {self.adaptation_risk:.3f})\n"
            f"z = {self.z_score:.3f} → "
            f"{'significatif' if self.significant else 'NON significatif'}\n"
            f"Action dominante : {act.value} à {p * 100:.1f} %\n"
            f"Gain EV vs GTO pur : {self.ev_gain_vs_gto:+.3f} bb\n"
            f"Exploitabilité : {self.exploitability:.3f} bb/100"
            f"{' (plafonnée par la borne de sûreté)' if self.exploitability_capped else ''}\n"
            f"{self.rationale}"
        )


def _adaptation_risk(mental: Mapping[MentalState, float]) -> float:
    r"""Risque de contre-exploitation :math:`\rho = \sum_s P(s)\,\text{prop}(s)`."""
    total = sum(mental.values())
    if total <= 0.0:
        return ADAPTATION_PROPENSITY[MentalState.SOLID]
    return sum(
        (p / total) * ADAPTATION_PROPENSITY.get(s, 0.5) for s, p in mental.items()
    )


def arbitrate(inp: FusionInput) -> FusionResult:
    r"""Calcule σ* = arbitrage entre GTO et meilleure réponse exploitatoire.

    Le poids d'exploitation n'est **pas** un curseur arbitraire : il se dérive
    de la confiance statistique.

    .. math::
        \lambda_{\text{brut}} =
        \Phi\!\left(\frac{|\hat\theta - \theta_0| - z_c\,\sigma_\theta}{\sigma_\theta}\right)

    .. math::
        \lambda = \lambda_{\text{brut}} \cdot (1 - \rho_{\text{adapt}})

    puis la garantie de sûreté (Ganzfried & Sandholm 2015) plafonne λ pour que

    .. math::
        \mathrm{Expl}(\sigma^*) \le \mathrm{Expl}(\sigma_{\text{GTO}}) + G_t

    Examples
    --------
    Villain : fold-to-cbet 71 % vs baseline GTO 55 % sur 84 mains,
    P(SOLID)=0.4728, P(LOOSE)=0.2864, P(TILT)=0.2409.

    >>> sigma = (0.71 * 0.29 / 84) ** 0.5
    >>> res = arbitrate(FusionInput(
    ...     gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
    ...     best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
    ...     deviation=0.16,
    ...     deviation_std=sigma,
    ...     mental_state_probs={MentalState.SOLID: 0.47,
    ...                         MentalState.LOOSE: 0.29,
    ...                         MentalState.TILT: 0.24},
    ... ))
    >>> round(res.lambda_final, 4)
    0.5812
    >>> round(res.strategy.get(Action.BET), 4)
    0.7711
    """
    sigma = inp.deviation_std
    if sigma <= 0.0:
        # Aucune incertitude déclarée : refuser d'exploiter plutôt que de
        # diviser par zéro. Échouer vers le GTO est toujours sûr.
        return FusionResult(
            strategy=inp.gto,
            lambda_raw=0.0,
            lambda_final=0.0,
            adaptation_risk=0.0,
            z_score=0.0,
            significant=False,
            ev_gain_vs_gto=0.0,
            exploitability=inp.exploitability_gto,
            exploitability_capped=False,
            rationale="σ_θ nul ou inconnu → repli sur le GTO (fail-safe).",
        )

    z = inp.deviation / sigma
    significant = abs(z) > inp.z_critical

    excess = abs(inp.deviation) - inp.z_critical * sigma
    lambda_raw = float(stats.norm.cdf(excess / sigma))

    rho = _adaptation_risk(inp.mental_state_probs)
    lambda_final = lambda_raw * (1.0 - rho)

    # ── Garantie de safe exploitation (Ganzfried & Sandholm 2015) ───────────
    capped = False
    extra_expl = inp.exploitability_br - inp.exploitability_gto
    if math.isfinite(inp.realized_gift) and extra_expl > 0.0:
        lambda_max = min(1.0, max(0.0, inp.realized_gift / extra_expl))
        if lambda_final > lambda_max:
            lambda_final = lambda_max
            capped = True

    strategy = inp.gto.blend(inp.best_response, lambda_final)
    ev_gain = lambda_final * (inp.ev_best_response - inp.ev_gto)
    expl = inp.exploitability_gto + lambda_final * extra_expl

    if not significant:
        rationale = (
            f"Écart non significatif (|z|={abs(z):.2f} ≤ {inp.z_critical}). "
            "L'échantillon ne justifie pas de dévier : jouer proche du GTO."
        )
    elif capped:
        rationale = (
            "Écart significatif, mais λ plafonné par la borne de sûreté : "
            f"le cadeau réalisé ({inp.realized_gift:.2f} bb/100) ne couvre pas "
            "une exploitation plus agressive."
        )
    elif rho > 0.45:
        rationale = (
            f"Écart significatif, mais l'adversaire est probablement capable de "
            f"s'adapter (ρ={rho:.2f}) : exploitation tempérée."
        )
    else:
        rationale = (
            f"Écart significatif (|z|={abs(z):.2f}) et faible risque d'adaptation "
            f"(ρ={rho:.2f}) : exploiter."
        )

    return FusionResult(
        strategy=strategy,
        lambda_raw=lambda_raw,
        lambda_final=lambda_final,
        adaptation_risk=rho,
        z_score=z,
        significant=significant,
        ev_gain_vs_gto=ev_gain,
        exploitability=expl,
        exploitability_capped=capped,
        rationale=rationale,
    )
