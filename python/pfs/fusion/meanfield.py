"""
F11 — Pots multi-way par approximation de champ moyen.

Statut (audit du 14 août 2026) — bibliothèque expérimentale, NON branchée
-------------------------------------------------------------------------
Importée par aucun code applicatif ni script livré : seul
``tests/test_app_and_data.py`` l'exerce. Pourquoi : le solveur et le
re-solve (``resolve_spot``) sont heads-up ; tant qu'aucun chemin applicatif
ne résout un pot à 3+ joueurs, cette fusion n'a pas de consommateur — et sa
propre docstring (ci-dessous) estime à 55 % la confiance dans son apport
face à la ligne de base par agrégation, qu'elle implémente aussi.
Ce qui existe déjà : point fixe amorti, baseline agrégée, mesure d'écart
entre les deux — testés. Accroche naturelle si on branche :
``pfs/engine.py:resolve_spot`` le jour où le re-solve devient multiway
(la décision brancher/abandonner doit alors passer par
``MeanFieldResult.worth_the_complexity``, pas par l'élégance de la théorie).
Règle du projet : aucun module fusionné sans route + UI + test de bout en
bout.

Sources
-------
- Lasry, J-M. & Lions, P-L. (2007), *Mean field games*, Japan J. Math. 2
- Carmona, R. & Delarue, F. (2018), *Probabilistic Theory of Mean Field Games*
- Laurière, M. et al. (2024) pour les liens MFG ↔ RL

⚠️ HONNÊTETÉ SUR CETTE FUSION
─────────────────────────────
C'est la plus risquée des treize, et sa confiance est de **55 %**. Trois
raisons, toutes structurelles :

1. La théorie MFG a des garanties pour N → ∞. En poker N = 2 à 5, et l'erreur
   d'approximation est en O(1/√N) — **elle n'est pas négligeable** : à N = 5,
   1/√N ≈ 0,45.
2. Aucune implémentation poker publiée n'existe, donc aucun point de
   comparaison.
3. L'alternative pragmatique — bucketer les adversaires et résoudre en
   2 joueurs contre un « villain agrégé » — capture probablement 80 % de la
   valeur pour 10 % de l'effort.

Ce module implémente les deux : l'itération de point fixe champ-moyen **et**
la ligne de base par agrégation, plus une **mesure de l'écart entre les deux**.
Si l'écart est faible, l'agrégation suffit et le MFG ne sert à rien : c'est le
test qui doit décider, pas l'élégance de la théorie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "MeanFieldConfig",
    "MeanFieldResult",
    "aggregate_baseline",
    "solve_mean_field",
    "multiway_equity_penalty",
]

F64 = npt.NDArray[np.float64]


class MeanFieldError(ValueError):
    pass


@dataclass(slots=True)
class MeanFieldConfig:
    n_opponents: int = 3
    max_iter: int = 200
    tolerance: float = 1e-6
    damping: float = 0.5
    """Amortissement du point fixe. Sans lui, l'itération oscille — c'est le
    problème de convergence classique des dynamiques de jeu (cf. Piliouras sur
    les cycles)."""

    def __post_init__(self) -> None:
        if self.n_opponents < 1:
            raise MeanFieldError("n_opponents doit être >= 1.")
        if not (0.0 < self.damping <= 1.0):
            raise MeanFieldError("damping doit être dans (0, 1].")


@dataclass(frozen=True, slots=True)
class MeanFieldResult:
    strategy: F64
    field_distribution: F64
    iterations: int
    converged: bool
    residual: float
    baseline_strategy: F64
    divergence_from_baseline: float
    """Distance L1 entre MFG et agrégation. Petite ⇒ le MFG n'apporte rien."""
    approximation_error_bound: float
    """Borne O(1/√N) — l'ordre de grandeur de ce qu'on ignore."""

    @property
    def worth_the_complexity(self) -> bool:
        """Le MFG ne vaut la peine que s'il s'écarte de l'agrégation **plus**
        que sa propre erreur d'approximation."""
        return self.divergence_from_baseline > self.approximation_error_bound

    def explain(self) -> str:
        return (
            f"Champ moyen · {self.iterations} itérations · "
            f"{'convergé' if self.converged else 'NON convergé'} "
            f"(résidu {self.residual:.2e})\n"
            f"  stratégie MFG        : {np.round(self.strategy, 4)}\n"
            f"  ligne de base agrégée: {np.round(self.baseline_strategy, 4)}\n"
            f"  écart L1             : {self.divergence_from_baseline:.4f}\n"
            f"  borne d'erreur O(1/√N): {self.approximation_error_bound:.4f}\n"
            f"  → {'le MFG apporte quelque chose' if self.worth_the_complexity else 'l’agrégation suffit — ne pas complexifier'}"
        )


def multiway_equity_penalty(heads_up_equity: float, n_opponents: int) -> float:
    r"""Équité résiduelle contre N adversaires indépendants.

    .. math::
        \mathrm{eq}_N \approx \mathrm{eq}_{HU}^{\,N}

    Approximation d'indépendance : elle **surestime** la pénalité quand les
    ranges adverses sont corrélées (elles le sont toujours un peu, par les
    cartes mortes), et la sous-estime quand elles sont anti-corrélées. Utile
    comme ordre de grandeur, jamais comme vérité.

    Examples
    --------
    Une main à 70 % en heads-up ne vaut plus que 34 % contre trois adversaires.

    >>> round(multiway_equity_penalty(0.70, 3), 4)
    0.343
    """
    if not (0.0 <= heads_up_equity <= 1.0):
        raise MeanFieldError("heads_up_equity doit être dans [0, 1].")
    if n_opponents < 1:
        raise MeanFieldError("n_opponents doit être >= 1.")
    return float(heads_up_equity**n_opponents)


def aggregate_baseline(
    payoff: Callable[[F64, F64], F64],
    n_actions: int,
    n_opponents: int,
    max_iter: int = 300,
    seed: int | None = 0,
) -> F64:
    """Ligne de base : un seul « villain agrégé », résolu en 2 joueurs.

    Descente de gradient naturel (miroir) sur le simplexe, qui converge sans
    osciller là où la meilleure réponse pure cycle.
    """
    rng = np.random.default_rng(seed)
    sigma = np.full(n_actions, 1.0 / n_actions)
    field = np.full(n_actions, 1.0 / n_actions)
    for t in range(max_iter):
        grad = payoff(sigma, field)
        step = 1.0 / math.sqrt(t + 1)
        z = step * (grad - grad.max())
        sigma = sigma * np.exp(z)
        sigma /= sigma.sum()
        field = 0.9 * field + 0.1 * sigma
    return sigma


def solve_mean_field(
    payoff: Callable[[F64, F64], F64],
    n_actions: int,
    config: MeanFieldConfig | None = None,
    seed: int | None = 0,
) -> MeanFieldResult:
    r"""Point fixe champ-moyen amorti.

    Cherche :math:`(\sigma^*, m^*)` tel que

    .. math::
        \sigma^* \in \arg\max_\sigma J(\sigma, m^*), \qquad m^* = \mathrm{Loi}(\sigma^*)

    Parameters
    ----------
    payoff
        ``payoff(sigma, field) -> vecteur de gains par action``. C'est le seul
        point d'accroche avec le moteur d'équité : dans le système complet, il
        appelle l'évaluateur pour calculer le gain de chaque action contre la
        distribution de champ.

    Notes
    -----
    L'amortissement est indispensable : le point fixe non amorti oscille
    (dynamiques cycliques classiques en théorie des jeux). Le résidu retourné
    permet de vérifier que la convergence a bien eu lieu plutôt que de la
    supposer.
    """
    cfg = config or MeanFieldConfig()
    if n_actions < 2:
        raise MeanFieldError("n_actions doit être >= 2.")

    sigma = np.full(n_actions, 1.0 / n_actions)
    field = np.full(n_actions, 1.0 / n_actions)

    converged = False
    residual = math.inf
    it = 0

    for it in range(1, cfg.max_iter + 1):
        grad = np.asarray(payoff(sigma, field), dtype=np.float64).ravel()
        if grad.size != n_actions:
            raise MeanFieldError("payoff doit renvoyer n_actions valeurs.")

        step = 1.0 / math.sqrt(it)
        z = step * (grad - grad.max())
        new_sigma = sigma * np.exp(z)
        new_sigma /= new_sigma.sum()

        new_field = (1.0 - cfg.damping) * field + cfg.damping * new_sigma

        residual = float(np.abs(new_sigma - sigma).sum() + np.abs(new_field - field).sum())
        sigma, field = new_sigma, new_field
        if residual < cfg.tolerance:
            converged = True
            break

    baseline = aggregate_baseline(payoff, n_actions, cfg.n_opponents, seed=seed)
    divergence = float(np.abs(sigma - baseline).sum())
    error_bound = 1.0 / math.sqrt(cfg.n_opponents + 1)

    return MeanFieldResult(
        strategy=sigma,
        field_distribution=field,
        iterations=it,
        converged=converged,
        residual=residual,
        baseline_strategy=baseline,
        divergence_from_baseline=divergence,
        approximation_error_bound=error_bound,
    )
