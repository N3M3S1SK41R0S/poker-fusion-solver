"""
F8 — Solveur GTO : DCFR alterné + Hyperparameter Schedules.

Sources
-------
- Zinkevich, Johanson, Bowling & Piccione (2007), *Regret Minimization in Games
  with Incomplete Information*, NIPS
- Tammelin (2014), *Solving Large Imperfect Information Games Using CFR+*
- **Brown, N. & Sandholm, T. (2019), *Solving Imperfect-Information Games via
  Discounted Regret Minimization*, AAAI** — DCFR (α=1.5, β=0, γ=2)
- **Zhang, B.H., McAleer, S. & Sandholm, T. (2024), *Faster Game Solving via
  Hyperparameter Schedules*, arXiv:2404.09097** — HS-DCFR, < 15 lignes de plus
- Cai, Farina, Grand-Clément, Kroer, Lee, Luo & Zheng (2025), ICLR — RM+ et
  PRM+ **n'ont pas** de convergence last-iterate ; pertinent dès qu'on consomme
  l'itéré courant plutôt que la moyenne (re-solve à budget court)

Ce module est l'implémentation de référence, en Python. Elle sert à :
  1. valider les invariants mathématiques (convergence, exploitabilité) ;
  2. servir d'oracle aux tests du portage Rust de la Phase 2.

Elle n'est pas destinée à passer à l'échelle du NLHE — c'est le rôle du
portage Rust + blueprint pré-calculé.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "Game",
    "KuhnPoker",
    "DCFRConfig",
    "DCFRSolver",
    "SolveResult",
    "regret_matching",
    "hyperparameter_schedule",
]

F64 = npt.NDArray[np.float64]


class SolverError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# INTERFACE DE JEU
# ═══════════════════════════════════════════════════════════════════════════


class Game(Protocol):
    """Jeu à information imparfaite, forme extensive, somme nulle, 2 joueurs."""

    n_players: int

    def initial_states(self) -> Sequence[tuple[tuple, float]]:
        """(état, probabilité) des situations initiales tirées par le hasard."""

    def is_terminal(self, state: tuple) -> bool: ...

    def utility(self, state: tuple) -> float:
        """Gain du joueur 0, du point de vue de l'état terminal."""

    def current_player(self, state: tuple) -> int: ...

    def infoset(self, state: tuple) -> str:
        """Clé de l'ensemble d'information vu par le joueur courant."""

    def actions(self, state: tuple) -> Sequence[str]: ...

    def next_state(self, state: tuple, action: str) -> tuple: ...


@dataclass(frozen=True, slots=True)
class KuhnPoker:
    """Kuhn poker — 3 cartes, 1 mise. Le banc d'essai standard du CFR.

    Équilibre de Nash connu analytiquement : la valeur du jeu pour le joueur 0
    vaut exactement **−1/18 ≈ −0,0555…**. C'est l'oracle qui permet de
    valider une implémentation de CFR sans ambiguïté.
    """

    n_players: int = 2

    def initial_states(self) -> list[tuple[tuple, float]]:
        deals = [(a, b) for a in range(3) for b in range(3) if a != b]
        p = 1.0 / len(deals)
        return [(((a, b), ""), p) for a, b in deals]

    @staticmethod
    def _history(state: tuple) -> str:
        return state[1]

    def is_terminal(self, state: tuple) -> bool:
        h = self._history(state)
        return h in ("pp", "bp", "bb", "pbp", "pbb")

    def utility(self, state: tuple) -> float:
        (c0, c1), h = state
        win = 1 if c0 > c1 else -1
        if h == "pp":
            return float(win)
        if h == "bp":
            return 1.0
        if h == "pbp":
            return -1.0
        if h in ("bb", "pbb"):
            return float(2 * win)
        raise SolverError(f"état non terminal : {state!r}")

    def current_player(self, state: tuple) -> int:
        return len(self._history(state)) % 2

    def infoset(self, state: tuple) -> str:
        cards, h = state
        return f"{cards[self.current_player(state)]}|{h}"

    def actions(self, state: tuple) -> tuple[str, ...]:
        return ("p", "b")

    def next_state(self, state: tuple, action: str) -> tuple:
        cards, h = state
        return (cards, h + action)


# ═══════════════════════════════════════════════════════════════════════════
# REGRET MATCHING & SCHEDULES
# ═══════════════════════════════════════════════════════════════════════════


def regret_matching(regrets: F64) -> F64:
    r"""Stratégie par regret matching : :math:`\sigma(a) \propto [R(a)]^+`.

    Si tous les regrets sont ≤ 0, on joue uniformément — c'est la convention
    de Hart & Mas-Colell (2000) et elle est nécessaire à la convergence.
    """
    pos = np.maximum(regrets, 0.0)
    total = pos.sum()
    if total > 0.0:
        return pos / total
    return np.full(regrets.shape, 1.0 / regrets.size)


def hyperparameter_schedule(t: int, total: int) -> tuple[float, float, float]:
    r"""Schedule (α, β, γ) de HS-DCFR — Zhang, McAleer & Sandholm (2024).

    Principe : sous-pondérer agressivement les premières itérations, puis
    relâcher progressivement vers les valeurs DCFR standard. Les itérations
    initiales sont produites par des stratégies quasi aléatoires : les
    surpondérer ralentit la convergence.

    .. math::
        \alpha(t) = \alpha_\infty + (\alpha_0 - \alpha_\infty)\,e^{-3t/T}

    Coût d'implémentation : ces quelques lignes. Gain : état de l'art 2024.
    """
    if total <= 0:
        raise SolverError("total doit être > 0.")
    decay = math.exp(-3.0 * t / total)
    alpha = 1.5 + (4.0 - 1.5) * decay        # 4.0 → 1.5
    beta = 0.0 + (-1.0 - 0.0) * decay        # -1.0 → 0.0 (oubli initial fort)
    gamma = 2.0 + (5.0 - 2.0) * decay        # 5.0 → 2.0
    return alpha, beta, gamma


# ═══════════════════════════════════════════════════════════════════════════
# SOLVEUR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class DCFRConfig:
    alpha: float = 1.5
    """Pondération des regrets positifs : t^α / (t^α + 1)."""
    beta: float = 0.0
    """Pondération des regrets négatifs : t^β / (t^β + 1)."""
    gamma: float = 2.0
    """Pondération de la stratégie moyenne : (t/(t+1))^γ."""
    alternating: bool = True
    """Mise à jour alternée des joueurs — plus rapide en pratique."""
    use_schedule: bool = True
    """Active HS-DCFR (Zhang, McAleer & Sandholm 2024)."""
    prune_threshold: float | None = -1e5
    """Élagage : on n'explore pas une branche à regret très négatif. None = off."""


@dataclass(slots=True)
class SolveResult:
    strategy: dict[str, F64]
    actions: dict[str, tuple[str, ...]]
    iterations: int
    exploitability: float
    game_value: float
    history: list[tuple[int, float]] = field(default_factory=list)

    def freq(self, infoset: str) -> dict[str, float]:
        return dict(zip(self.actions[infoset], self.strategy[infoset]))

    def explain(self, top: int = 12) -> str:
        lines = [
            f"{self.iterations} itérations · valeur du jeu = {self.game_value:+.6f} "
            f"· exploitabilité = {self.exploitability:.3e}"
        ]
        for k in sorted(self.strategy)[:top]:
            fr = ", ".join(
                f"{a}={p * 100:5.1f}%" for a, p in self.freq(k).items()
            )
            lines.append(f"  {k:<10} {fr}")
        return "\n".join(lines)


class DCFRSolver:
    """DCFR alterné avec schedules, pruning et mesure d'exploitabilité."""

    __slots__ = ("game", "cfg", "_regrets", "_strategy_sum", "_actions")

    def __init__(self, game: Game, config: DCFRConfig | None = None) -> None:
        self.game = game
        self.cfg = config or DCFRConfig()
        self._regrets: dict[str, F64] = {}
        self._strategy_sum: dict[str, F64] = {}
        self._actions: dict[str, tuple[str, ...]] = {}

    # ── accès aux infosets ───────────────────────────────────────────────
    def _node(self, key: str, actions: Sequence[str]) -> tuple[F64, F64]:
        if key not in self._regrets:
            n = len(actions)
            self._regrets[key] = np.zeros(n, dtype=np.float64)
            self._strategy_sum[key] = np.zeros(n, dtype=np.float64)
            self._actions[key] = tuple(actions)
        return self._regrets[key], self._strategy_sum[key]

    def current_strategy(self, key: str) -> F64:
        return regret_matching(self._regrets[key])

    def average_strategy(self) -> dict[str, F64]:
        out: dict[str, F64] = {}
        for k, s in self._strategy_sum.items():
            total = s.sum()
            out[k] = s / total if total > 0 else np.full(s.size, 1.0 / s.size)
        return out

    # ── parcours CFR ─────────────────────────────────────────────────────
    def _walk(
        self, state: tuple, reach: list[float], updating: int | None, weight: float
    ) -> float:
        g = self.game
        if g.is_terminal(state):
            return g.utility(state)

        player = g.current_player(state)
        key = g.infoset(state)
        actions = g.actions(state)
        regrets, strat_sum = self._node(key, actions)
        sigma = regret_matching(regrets)

        # Élagage : branche dont le regret est très négatif et dont la
        # probabilité d'atteinte adverse est nulle.
        util = np.zeros(len(actions), dtype=np.float64)
        node_util = 0.0
        for i, a in enumerate(actions):
            if (
                self.cfg.prune_threshold is not None
                and regrets[i] < self.cfg.prune_threshold
                and sigma[i] == 0.0
            ):
                continue
            nxt_reach = list(reach)
            nxt_reach[player] *= sigma[i]
            util[i] = self._walk(g.next_state(state, a), nxt_reach, updating, weight)
            node_util += sigma[i] * util[i]

        if updating is None or updating == player:
            sign = 1.0 if player == 0 else -1.0
            counterfactual = reach[1 - player]
            regrets += counterfactual * sign * (util - node_util)
            strat_sum += weight * reach[player] * sigma

        return node_util

    # ── boucle principale ────────────────────────────────────────────────
    def solve(
        self,
        iterations: int = 5000,
        track_every: int = 0,
    ) -> SolveResult:
        g = self.game
        starts = g.initial_states()
        history: list[tuple[int, float]] = []

        for t in range(1, iterations + 1):
            if self.cfg.use_schedule:
                alpha, beta, gamma = hyperparameter_schedule(t, iterations)
            else:
                alpha, beta, gamma = self.cfg.alpha, self.cfg.beta, self.cfg.gamma

            players = (0, 1) if self.cfg.alternating else (None,)
            strat_weight = (t / (t + 1.0)) ** gamma

            for p in players:
                for state, prob in starts:
                    self._walk(state, [prob, prob], p, strat_weight)

            # Actualisation DCFR des regrets.
            pos_w = (t**alpha) / (t**alpha + 1.0)
            neg_w = (t**beta) / (t**beta + 1.0) if beta != 0.0 else 0.5
            for r in self._regrets.values():
                np.multiply(r, np.where(r > 0.0, pos_w, neg_w), out=r)

            if track_every and (t % track_every == 0 or t == iterations):
                history.append((t, self.exploitability()))

        avg = self.average_strategy()
        return SolveResult(
            strategy=avg,
            actions=dict(self._actions),
            iterations=iterations,
            exploitability=self.exploitability(),
            game_value=self.game_value(avg),
            history=history,
        )

    # ── évaluation ───────────────────────────────────────────────────────
    def game_value(self, strategy: dict[str, F64] | None = None) -> float:
        s = strategy if strategy is not None else self.average_strategy()

        def rec(state: tuple) -> float:
            g = self.game
            if g.is_terminal(state):
                return g.utility(state)
            key = g.infoset(state)
            actions = g.actions(state)
            probs = s.get(key, np.full(len(actions), 1.0 / len(actions)))
            return float(
                sum(
                    p * rec(g.next_state(state, a))
                    for a, p in zip(actions, probs)
                    if p > 0
                )
            )

        return float(sum(pr * rec(st) for st, pr in self.game.initial_states()))

    def best_response_value(
        self, strategy: dict[str, F64], br_player: int
    ) -> float:
        """Valeur de la meilleure réponse de ``br_player``, en **information
        imparfaite**.

        Point crucial, et erreur classique : le joueur BR doit choisir **une
        seule action par ensemble d'information**, pas par état. Laisser BR
        maximiser état par état revient à lui donner les cartes de l'adversaire
        — la « valeur » obtenue est alors celle d'un joueur clairvoyant, et
        l'exploitabilité mesurée ne converge jamais vers zéro, même à
        l'équilibre exact.

        Algorithme (exact sur les petits jeux) :
          1. énumérer les états atteignables, pondérés par
             (hasard × probabilité d'atteinte de l'adversaire) ;
          2. traiter les ensembles d'information par **historique décroissant**,
             de sorte que toute décision plus profonde soit déjà figée ;
          3. à chaque ensemble, choisir l'action maximisant la valeur moyenne
             pondérée sur les états qui le composent.
        """
        g = self.game

        # ── 1. états atteignables, pondérés par la reach adverse × hasard ──
        weights: dict[tuple, float] = {}
        by_infoset: dict[str, list[tuple]] = {}

        def collect(state: tuple, w: float) -> None:
            if w <= 0.0 or g.is_terminal(state):
                return
            player = g.current_player(state)
            actions = g.actions(state)
            key = g.infoset(state)
            if player == br_player:
                weights[state] = weights.get(state, 0.0) + w
                by_infoset.setdefault(key, [])
                if state not in by_infoset[key]:
                    by_infoset[key].append(state)
                for a in actions:
                    collect(g.next_state(state, a), w)
            else:
                probs = strategy.get(key, np.full(len(actions), 1.0 / len(actions)))
                for a, p in zip(actions, probs):
                    if p > 0.0:
                        collect(g.next_state(state, a), w * float(p))

        for st, pr in g.initial_states():
            collect(st, pr)

        # ── 2. décision figée par ensemble d'information ────────────────────
        br_choice: dict[str, str] = {}
        memo: dict[tuple, float] = {}

        def value(state: tuple) -> float:
            if state in memo:
                return memo[state]
            if g.is_terminal(state):
                u = g.utility(state)
                v = u if br_player == 0 else -u
                memo[state] = v
                return v
            player = g.current_player(state)
            actions = g.actions(state)
            key = g.infoset(state)
            if player == br_player:
                a = br_choice[key]            # déjà décidé (ordre décroissant)
                v = value(g.next_state(state, a))
            else:
                probs = strategy.get(key, np.full(len(actions), 1.0 / len(actions)))
                v = float(
                    sum(
                        float(p) * value(g.next_state(state, a))
                        for a, p in zip(actions, probs)
                        if p > 0.0
                    )
                )
            memo[state] = v
            return v

        # Historique le plus long d'abord : les sous-arbres sont alors résolus.
        for key in sorted(
            by_infoset, key=lambda k: len(k.split("|", 1)[1]), reverse=True
        ):
            states = by_infoset[key]
            actions = g.actions(states[0])
            best_a, best_v = actions[0], -math.inf
            for a in actions:
                v = 0.0
                for st in states:
                    w = weights[st]
                    if w > 0.0:
                        v += w * value(g.next_state(st, a))
                if v > best_v:
                    best_a, best_v = a, v
            br_choice[key] = best_a
            memo.clear()   # les valeurs dépendent des choix figés en aval

        return float(sum(pr * value(st) for st, pr in g.initial_states()))

    def exploitability(self, strategy: dict[str, F64] | None = None) -> float:
        r"""Exploitabilité :math:`\varepsilon = \frac{BR_0 + BR_1}{2}`.

        Vaut 0 à l'équilibre de Nash exact — c'est **la** métrique de qualité
        d'un solveur. Toute autre (vitesse, mémoire) n'a de sens qu'à
        exploitabilité comparable, et c'est pourquoi les comparaisons de
        solveurs qui l'omettent ne veulent rien dire.
        """
        s = strategy if strategy is not None else self.average_strategy()
        return float(
            (self.best_response_value(s, 0) + self.best_response_value(s, 1)) / 2.0
        )
