"""
F2 — État mental caché de l'adversaire par modèle de Markov caché.

Sources
-------
- Baum, L.E. & Petrie, T. (1966)
- Rabiner, L.R. (1989), *A Tutorial on Hidden Markov Models*, Proc. IEEE 77(2)

L'adversaire possède un état caché S ∈ {SOLID, LOOSE, TILT}. On n'observe que
ses actions. On veut P(S_t | o_1..t) **en ligne**, sans réestimer l'historique.

Intérêt central : une seule action improbable fait bondir P(TILT) d'un facteur
4 à 5, alors qu'aucune statistique brute (VPIP, 3bet%) ne bouge de façon
détectable après une main. On détecte la transition **avant** que les moyennes
ne bougent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "MentalState",
    "Observation",
    "HMMBelief",
    "OnlineHMM",
    "DEFAULT_TRANSITION",
    "DEFAULT_EMISSION",
    "fit_baum_welch",
]

F64 = npt.NDArray[np.float64]


class MentalState(IntEnum):
    SOLID = 0
    LOOSE = 1
    TILT = 2


class Observation(IntEnum):
    """Catégories d'action observables, discrétisées."""

    FOLD = 0
    PASSIVE = 1          # limp, check, call
    STANDARD_AGGRO = 2   # open, c-bet dans les fréquences normales
    LOOSE_AGGRO = 3      # open large, float, bluff-raise
    WILD = 4             # 3bet/4bet improbable, overbet, shove hors-spot


# Persistance forte : un état mental ne bascule pas en une main.
DEFAULT_TRANSITION: F64 = np.array(
    [
        [0.97, 0.02, 0.01],   # SOLID  →
        [0.05, 0.92, 0.03],   # LOOSE  →
        [0.08, 0.10, 0.82],   # TILT   →
    ],
    dtype=np.float64,
)

# P(observation | état). Lignes = états, colonnes = Observation.
DEFAULT_EMISSION: F64 = np.array(
    [
        [0.44, 0.30, 0.17, 0.05, 0.04],   # SOLID : folde beaucoup, agressivité mesurée
        [0.22, 0.30, 0.24, 0.12, 0.12],   # LOOSE
        [0.10, 0.18, 0.20, 0.22, 0.30],   # TILT  : actions improbables fréquentes
    ],
    dtype=np.float64,
)

DEFAULT_PRIOR: F64 = np.array([0.80, 0.15, 0.05], dtype=np.float64)


class HMMError(ValueError):
    pass


def _validate_row_stochastic(m: F64, name: str) -> None:
    if m.ndim != 2:
        raise HMMError(f"{name} doit être une matrice 2-D.")
    if np.any(m < 0.0):
        raise HMMError(f"{name} contient une probabilité négative.")
    if not np.allclose(m.sum(axis=1), 1.0, atol=1e-9):
        raise HMMError(f"{name} : chaque ligne doit sommer à 1.")


@dataclass(frozen=True, slots=True)
class HMMBelief:
    """Distribution postérieure sur l'état caché."""

    probs: F64
    n_observations: int
    log_likelihood: float

    def __getitem__(self, s: MentalState) -> float:
        return float(self.probs[int(s)])

    @property
    def most_likely(self) -> MentalState:
        return MentalState(int(np.argmax(self.probs)))

    @property
    def entropy_bits(self) -> float:
        p = self.probs[self.probs > 0.0]
        return float(-np.sum(p * np.log2(p)))

    @property
    def is_confident(self) -> bool:
        """Vrai si la croyance est nettement piquée (entropie < 1 bit sur 3 états)."""
        return self.entropy_bits < 1.0

    def as_mapping(self) -> dict[MentalState, float]:
        """Format attendu par la méta-fusion F13."""
        return {s: float(self.probs[int(s)]) for s in MentalState}

    def __str__(self) -> str:
        return "  ".join(
            f"{s.name}={self.probs[int(s)] * 100:5.1f}%" for s in MentalState
        )


@dataclass(slots=True)
class OnlineHMM:
    """Filtre forward normalisé, mis à jour action par action.

    Parameters
    ----------
    transition, emission, prior
        Paramètres du modèle. Les valeurs par défaut sont des priors
        plausibles — **à recalibrer sur ton pool** via :func:`fit_baum_welch`.
    floor
        Plancher appliqué à la croyance après chaque mise à jour. Empêche
        qu'un état tombe à exactement 0 et devienne inatteignable (un joueur
        peut toujours sortir du tilt).

    Examples
    --------
    Exemple golden du Plan Directeur §4 F2 — un 3-bet UTG improbable :

    >>> h = OnlineHMM()
    >>> b = h.update(Observation.WILD)
    >>> round(b[MentalState.TILT], 4)
    0.2409
    """

    transition: F64 = field(default_factory=lambda: DEFAULT_TRANSITION.copy())
    emission: F64 = field(default_factory=lambda: DEFAULT_EMISSION.copy())
    prior: F64 = field(default_factory=lambda: DEFAULT_PRIOR.copy())
    floor: float = 1e-4

    _alpha: F64 = field(init=False, repr=False)
    _n: int = field(init=False, default=0, repr=False)
    _loglik: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        _validate_row_stochastic(self.transition, "transition")
        _validate_row_stochastic(self.emission, "emission")
        if self.transition.shape[0] != len(MentalState):
            raise HMMError("transition doit être 3×3.")
        if self.emission.shape[0] != len(MentalState):
            raise HMMError("emission doit avoir 3 lignes.")
        if not math.isclose(float(self.prior.sum()), 1.0, abs_tol=1e-9):
            raise HMMError("prior doit sommer à 1.")
        self._alpha = self.prior.astype(np.float64).copy()

    # ── mise à jour ──────────────────────────────────────────────────────
    def update(self, obs: Observation | int) -> HMMBelief:
        r"""Une observation.

        .. math::
            \tilde\alpha_t(j) = \Big[\sum_i \alpha_{t-1}(i)\,A(i,j)\Big]\,B(j, o_t)
            \qquad
            \alpha_t = \tilde\alpha_t / \sum_j \tilde\alpha_t(j)
        """
        o = int(obs)
        if not (0 <= o < self.emission.shape[1]):
            raise HMMError(f"observation {o} hors domaine.")

        predicted = self._alpha @ self.transition          # prédiction
        unnormalised = predicted * self.emission[:, o]     # mise à jour
        c = float(unnormalised.sum())
        if c <= 0.0:
            # Observation impossible sous le modèle : on garde la prédiction
            # plutôt que de propager des NaN. Signalé par la log-vraisemblance.
            self._alpha = predicted / predicted.sum()
            self._loglik += -math.inf
        else:
            self._alpha = unnormalised / c
            self._loglik += math.log(c)

        if self.floor > 0.0:
            self._alpha = np.maximum(self._alpha, self.floor)
            self._alpha /= self._alpha.sum()

        self._n += 1
        return self.belief

    def update_many(self, observations: Iterable[Observation | int]) -> HMMBelief:
        b = self.belief
        for o in observations:
            b = self.update(o)
        return b

    @property
    def belief(self) -> HMMBelief:
        return HMMBelief(self._alpha.copy(), self._n, self._loglik)

    def reset(self) -> None:
        self._alpha = self.prior.copy()
        self._n = 0
        self._loglik = 0.0

    # ── diagnostics ──────────────────────────────────────────────────────
    def tilt_surge(self, obs: Observation | int) -> float:
        """Facteur multiplicatif de P(TILT) qu'appliquerait cette observation.

        Sans muter l'état — utile pour trier les actions par pouvoir
        diagnostique, ou pour expliquer une alerte à l'écran.
        """
        before = float(self._alpha[MentalState.TILT])
        predicted = self._alpha @ self.transition
        un = predicted * self.emission[:, int(obs)]
        s = float(un.sum())
        after = float(un[MentalState.TILT] / s) if s > 0 else before
        return after / before if before > 0 else math.inf

    def stationary_distribution(self) -> F64:
        """Distribution stationnaire de la chaîne (vecteur propre gauche, λ=1)."""
        vals, vecs = np.linalg.eig(self.transition.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, idx])
        v = np.abs(v)
        return v / v.sum()


def fit_baum_welch(
    sequences: Sequence[Sequence[int]],
    n_states: int = 3,
    n_obs: int = len(Observation),
    n_iter: int = 100,
    tol: float = 1e-6,
    seed: int | None = 0,
) -> tuple[F64, F64, F64, float]:
    """Estime (transition, emission, prior) par Baum-Welch sur des séquences.

    À utiliser sur ta base de mains une fois qu'elle est assez fournie
    (≥ 50 000 mains labellisées pour un ajustement honnête à 3 états).

    Returns
    -------
    (A, B, pi, log_likelihood)

    Notes
    -----
    Baum-Welch est un EM : il converge vers un **maximum local**. Relancer avec
    plusieurs graines et garder la meilleure log-vraisemblance. Pour choisir le
    nombre d'états, comparer les BIC — ne pas supposer que 3 est le bon nombre.
    """
    rng = np.random.default_rng(seed)
    A = rng.dirichlet(np.ones(n_states) * 8.0, size=n_states)
    B = rng.dirichlet(np.ones(n_obs) * 4.0, size=n_states)
    pi = rng.dirichlet(np.ones(n_states) * 8.0)

    prev_ll = -np.inf
    ll = -np.inf

    for _ in range(n_iter):
        A_num = np.zeros_like(A)
        B_num = np.zeros_like(B)
        pi_acc = np.zeros_like(pi)
        gamma_sum = np.zeros(n_states)
        ll = 0.0

        for seq in sequences:
            T = len(seq)
            if T == 0:
                continue
            alpha = np.zeros((T, n_states))
            scale = np.zeros(T)

            alpha[0] = pi * B[:, seq[0]]
            scale[0] = alpha[0].sum()
            if scale[0] <= 0:
                continue
            alpha[0] /= scale[0]
            for t in range(1, T):
                alpha[t] = (alpha[t - 1] @ A) * B[:, seq[t]]
                scale[t] = alpha[t].sum()
                if scale[t] <= 0:
                    scale[t] = 1e-300
                alpha[t] /= scale[t]

            beta = np.zeros((T, n_states))
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = A @ (B[:, seq[t + 1]] * beta[t + 1]) / scale[t + 1]

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            pi_acc += gamma[0]
            for t in range(T - 1):
                xi = (
                    alpha[t][:, None]
                    * A
                    * B[:, seq[t + 1]][None, :]
                    * beta[t + 1][None, :]
                    / scale[t + 1]
                )
                A_num += xi
            for t in range(T):
                B_num[:, seq[t]] += gamma[t]
            gamma_sum += gamma.sum(axis=0)
            ll += float(np.log(np.maximum(scale, 1e-300)).sum())

        A = A_num / np.maximum(A_num.sum(axis=1, keepdims=True), 1e-300)
        B = B_num / np.maximum(B_num.sum(axis=1, keepdims=True), 1e-300)
        pi = pi_acc / max(len(sequences), 1)
        pi /= pi.sum()

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return A, B, pi, ll
