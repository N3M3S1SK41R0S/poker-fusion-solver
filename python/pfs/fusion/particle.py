"""
F3 — Inférence de stratégie adverse par filtre particulaire (SMC).

Sources
-------
- Gordon, Salmond & Smith (1993), *Novel approach to nonlinear/non-Gaussian
  Bayesian state estimation*, IEE Proc. F 140(2)
- Doucet, de Freitas & Gordon (2001), *Sequential Monte Carlo Methods in Practice*

L'objet réellement incertain n'est pas la range de l'adversaire — **c'est sa
stratégie**. Une mise à jour bayésienne classique de range suppose un modèle
adverse unique (typiquement GTO) ; si l'adversaire n'est pas GTO, le biais se
compose sur quatre streets.

Le filtre particulaire maintient une distribution **sur les modèles**, donc la
range postérieure marginalise l'incertitude de modèle. C'est la différence
entre « il a X sachant qu'il joue GTO » et « il a X, tous modèles plausibles
confondus ».
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import N_COMBOS, Range

__all__ = [
    "Archetype",
    "StrategyParticle",
    "ParticleFilter",
    "ARCHETYPE_PRIORS",
]

F64 = npt.NDArray[np.float64]


class ParticleError(ValueError):
    pass


class Archetype(str, Enum):
    """Familles de stratégie couvrant l'essentiel d'un pool."""

    NIT = "nit"
    TAG = "tag"
    LAG = "lag"
    STATION = "station"
    MANIAC = "maniac"
    GTO = "gto"


ARCHETYPE_PRIORS: dict[Archetype, float] = {
    Archetype.NIT: 0.15,
    Archetype.TAG: 0.30,
    Archetype.LAG: 0.15,
    Archetype.STATION: 0.20,
    Archetype.MANIAC: 0.05,
    Archetype.GTO: 0.15,
}

# (seuil d'équité pour continuer, agressivité, fréquence de bluff)
_ARCHETYPE_PARAMS: dict[Archetype, tuple[float, float, float]] = {
    Archetype.NIT:     (0.62, 0.35, 0.04),
    Archetype.TAG:     (0.52, 0.60, 0.12),
    Archetype.LAG:     (0.44, 0.85, 0.28),
    Archetype.STATION: (0.38, 0.20, 0.03),
    Archetype.MANIAC:  (0.30, 0.95, 0.45),
    Archetype.GTO:     (0.50, 0.62, 0.22),
}


@dataclass(slots=True)
class StrategyParticle:
    """Une hypothèse de stratégie adverse, avec sa range courante."""

    archetype: Archetype
    equity_threshold: float
    aggression: float
    bluff_freq: float
    range_weights: F64
    log_weight: float = 0.0

    def action_likelihood(self, equities: F64, action: str, bet_frac: float) -> F64:
        r"""P(action | combo) sous cette hypothèse de stratégie.

        Modèle logistique autour du seuil d'équité, décalé par la taille de
        mise affrontée et modulé par l'agressivité et la fréquence de bluff.
        """
        eq = np.asarray(equities, dtype=np.float64).ravel()
        shift = 0.16 * math.log1p(max(bet_frac, 0.0))
        z = 9.0 * (eq - (self.equity_threshold + shift))
        strong = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

        if action in ("bet", "raise"):
            # Value + bluffs : les mains très faibles bluffent aussi.
            weak = 1.0 / (1.0 + np.exp(np.clip(12.0 * (eq - 0.28), -60, 60)))
            return np.clip(
                self.aggression * strong + self.bluff_freq * weak, 1e-6, 1.0
            )
        if action == "call":
            return np.clip((1.0 - self.aggression) * strong + 0.05, 1e-6, 1.0)
        if action in ("check", "fold"):
            return np.clip(1.0 - strong, 1e-6, 1.0)
        raise ParticleError(f"action inconnue : {action!r}")


class ParticleFilter:
    """Filtre SMC sur les stratégies adverses.

    Parameters
    ----------
    n_particles
        100 particules suffisent en pratique ; le coût est
        ``n × 1326`` flottants par joueur (≈ 1 Mo à n=100).
    jitter
        Bruit de rééchantillonnage sur les paramètres — indispensable, sinon
        les particules dégénèrent vers un unique clone après quelques
        rééchantillonnages (appauvrissement de l'échantillon).
    """

    __slots__ = ("particles", "_rng", "_jitter", "_n_resamples", "_n_updates")

    def __init__(
        self,
        n_particles: int = 100,
        prior_range: Range | None = None,
        priors: dict[Archetype, float] | None = None,
        jitter: float = 0.02,
        seed: int | None = 0,
    ) -> None:
        if n_particles < 2:
            raise ParticleError("n_particles doit être >= 2.")
        self._rng = np.random.default_rng(seed)
        self._jitter = float(jitter)
        self._n_resamples = 0
        self._n_updates = 0

        base = (prior_range or Range.full()).weights
        pr = priors or ARCHETYPE_PRIORS
        kinds = list(pr)
        probs = np.array([pr[k] for k in kinds], dtype=np.float64)
        probs /= probs.sum()
        draw = self._rng.choice(len(kinds), size=n_particles, p=probs)

        self.particles: list[StrategyParticle] = []
        for i in draw:
            k = kinds[int(i)]
            th, ag, bl = _ARCHETYPE_PARAMS[k]
            self.particles.append(
                StrategyParticle(
                    archetype=k,
                    equity_threshold=float(np.clip(th + self._rng.normal(0, 0.04), 0.05, 0.95)),
                    aggression=float(np.clip(ag + self._rng.normal(0, 0.06), 0.02, 0.99)),
                    bluff_freq=float(np.clip(bl + self._rng.normal(0, 0.04), 0.0, 0.8)),
                    range_weights=base.copy(),
                    log_weight=-math.log(n_particles),
                )
            )

    # ── diagnostics ──────────────────────────────────────────────────────
    @property
    def weights(self) -> F64:
        lw = np.array([p.log_weight for p in self.particles])
        lw -= lw.max()
        w = np.exp(lw)
        return w / w.sum()

    @property
    def effective_sample_size(self) -> float:
        r""":math:`N_{eff} = 1/\sum w_i^2`. Chute = dégénérescence des poids."""
        w = self.weights
        return float(1.0 / np.sum(w**2))

    @property
    def n_resamples(self) -> int:
        return self._n_resamples

    def archetype_posterior(self) -> dict[Archetype, float]:
        w = self.weights
        out = {k: 0.0 for k in Archetype}
        for wi, p in zip(w, self.particles):
            out[p.archetype] += float(wi)
        return out

    # ── mise à jour ──────────────────────────────────────────────────────
    def observe(
        self, equities: F64, action: str, bet_frac: float = 0.0
    ) -> Range:
        """Une action observée : repondère les particules et met à jour les ranges.

        Returns
        -------
        Range
            Range marginale a posteriori, marginalisée sur les modèles :
            :math:`\\hat r(c) = \\sum_i w_i\\, r_i(c)`.
        """
        eq = np.asarray(equities, dtype=np.float64).ravel()
        if eq.size != N_COMBOS:
            raise ParticleError(f"equities doit couvrir les {N_COMBOS} combos.")

        for p in self.particles:
            lik = p.action_likelihood(eq, action, bet_frac)
            post = p.range_weights * lik
            mass = float(post.sum())
            if mass <= 0.0:
                p.log_weight = -math.inf
                continue
            p.log_weight += math.log(mass / max(p.range_weights.sum(), 1e-300))
            m = post.max()
            p.range_weights = post / m if m > 0 else post

        self._n_updates += 1
        if self.effective_sample_size < len(self.particles) / 2.0:
            self._resample()
        return self.marginal_range()

    def marginal_range(self) -> Range:
        w = self.weights
        acc = np.zeros(N_COMBOS, dtype=np.float64)
        for wi, p in zip(w, self.particles):
            acc += wi * p.range_weights
        m = acc.max()
        return Range(acc / m if m > 0 else acc)

    def _resample(self) -> None:
        """Rééchantillonnage systématique + jitter (anti-appauvrissement)."""
        n = len(self.particles)
        w = self.weights
        positions = (self._rng.random() + np.arange(n)) / n
        idx = np.searchsorted(np.cumsum(w), positions)
        idx = np.clip(idx, 0, n - 1)

        new: list[StrategyParticle] = []
        for i in idx:
            src = self.particles[int(i)]
            new.append(
                StrategyParticle(
                    archetype=src.archetype,
                    equity_threshold=float(
                        np.clip(src.equity_threshold + self._rng.normal(0, self._jitter), 0.05, 0.95)
                    ),
                    aggression=float(
                        np.clip(src.aggression + self._rng.normal(0, self._jitter), 0.02, 0.99)
                    ),
                    bluff_freq=float(
                        np.clip(src.bluff_freq + self._rng.normal(0, self._jitter), 0.0, 0.8)
                    ),
                    range_weights=src.range_weights.copy(),
                    log_weight=-math.log(n),
                )
            )
        self.particles = new
        self._n_resamples += 1

    def explain(self) -> str:
        post = self.archetype_posterior()
        top = sorted(post.items(), key=lambda kv: -kv[1])[:4]
        r = self.marginal_range()
        return (
            f"N_eff = {self.effective_sample_size:.1f}/{len(self.particles)} "
            f"({self._n_resamples} rééch.) · {r}\n  "
            + "  ".join(f"{k.value}={v * 100:4.1f}%" for k, v in top)
        )
