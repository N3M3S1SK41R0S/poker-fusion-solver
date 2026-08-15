r"""
L1 — Solveur postflop NLHE **réel** : CFR range-contre-range sur vraies cartes.

C'est le levier qui ferme l'écart principal du benchmark (« solving à
l'échelle Kuhn/Leduc »). Ici : river résolue EXACTEMENT (1326 combos, zéro
abstraction de cartes), turn résolu par nœud de chance sur les rivières,
flop résolu par nœuds de chance EMBOÎTÉS (turn puis river) — la même
architecture que PioSOLVER/GTO+, en NumPy pur.

Théorie
-------
- CFR : Zinkevich et al. (2007) — la régret-minimisation converge vers Nash
  en jeu à somme (quasi) constante.
- DCFR : Brown & Sandholm (2019) — poids t^α/(t^α+1) sur les regrets positifs,
  ½ sur les négatifs (β=0), (t/(t+1))^γ sur la stratégie moyenne. Mêmes
  constantes que ``pfs.solver.dcfr`` (α=1.5, β=0, γ=2).
- Vectorisation « range vs range » : les regrets vivent par (nœud, combo) ;
  chaque traversée propage des VECTEURS de reach, pas des mains une à une.

Le point dur — l'abattage avec card removal en O(n log n) :
    CFV_i = pot·[Q(s_j < s_i, j∤i) + ½·Q(s_j = s_i, j∤i)] − invested·Q(j∤i)
  se calcule par tri des combos adverses par force + sommes préfixes globales
  et PAR CARTE (52 lignes) : le terme bloqué = préfixe[carte a] + préfixe[b]
  − combo exactement identique (toujours dans la bande d'égalité, jamais
  au-dessous). Tout le tri et l'appartenance par carte sont précalculés une
  fois ; chaque itération ne fait que des cumsum sur le reach courant.

Comptabilité des valeurs : utilité = part brute du pot − mise investie dans
la branche. À rake nul, u_OOP + u_IP = pot initial à tout terminal (somme
constante) — c'est l'invariant testé à la précision machine, et ce qui rend
l'exploitabilité exacte : expl = (BR_OOP + BR_IP − pot)/2, en % du pot.
Avec rake (``pfs.core.rake``, convention won-pot), le gagnant encaisse
net(pot) = pot − min(cap, pct·pot) : la somme des utilités devient
pot − rake(pot terminal), branche par branche — voir la docstring de classe.

Le nœud de chance (turn → river, ou flop → turn) pondère chaque tirage c par
P(c | combo du traverseur) = 1/(cartes inconnues du traverseur), et masque
les reach adverses contenant c — l'actualisation bayésienne exacte du deck.
Le dénombrement (piège n°1 de la passation) : au turn, 44 rivières valides
par confrontation (52 − 4 board − 2×2 mains) ; au flop, 45 turns valides
(52 − 3 − 4) PUIS 44 rivières chacun, soit 45×44 runouts ordonnés et
45×44/2 = 990 confrontations turn+river par feuille flop. Le diviseur est
figé dans chaque nœud de chance à la construction — tout autre dénominateur
casse « somme des EV = pot », qui vaut pour TOUT profil de stratégies.

Limites assumées : arbre de mise configurable mais raisonnable (tailles en
fraction du pot, plafond de relances) ; le rake « % + cap » est branché aux
terminaux via ``rake=``. Le flop en profondeur complète est EXACT mais cher
(l'arbre porte 49×48 rounds river : compter ~10⁴–10⁵ nœuds selon les
tailles ; mesures dans ``banc_flop.py``) — l'échelle « toutes textures »
reste le blueprint Phase 2 (abstraction/isomorphisme, autre chantier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.equity import evaluate7
from pfs.core.rake import NO_RAKE, RakeModel
from pfs.core.range_model import N_CARDS, Range, RangeError, combo_cards

__all__ = [
    "PostflopError", "PostflopResult", "PostflopSolver", "RootAction",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

OOP, IP = 0, 1
_ALPHA, _GAMMA = 1.5, 2.0            # DCFR (Brown & Sandholm 2019)
# Feuilles rollout d'un solve FLOP tronqué : la part d'abattage est linéaire
# en q → matrice (n_OOP × n_IP) précalculée par turn, si elle tient sous ce
# seuil d'éléments (au-delà : boucle de CFV triés, exacte mais ~50× plus
# lente par itération — mesuré dans banc_flop.py).
_LEAF_W_MAX = 250_000


class PostflopError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _Player:
    cards: I64            # (n, 2)
    weights: F64          # (n,) — poids de la range (mesure de départ)
    n: int


@dataclass(slots=True)
class _Node:
    """Nœud de décision. amounts[a] = jetons AJOUTÉS par l'acteur pour l'action a."""
    player: int
    pot: float
    to_call: float
    invested: tuple[float, float]      # investi (OOP, IP) depuis la racine
    labels: list[str]
    amounts: list[float]
    children: list[int]                # indices dans self._nodes ; -1 = terminal
    terminal: list[object]             # _Fold | _Showdown | _Chance | None
    regrets: F64 = field(default=None)  # (n_actions, n_combos_acteur)
    strat_sum: F64 = field(default=None)
    locked_mask: F64 = field(default=None)    # (n_combos,) bool — nodelock
    locked_sigma: F64 = field(default=None)   # (n_actions, n_combos)


@dataclass(slots=True)
class _Fold:
    winner: int
    pot: float
    invested: tuple[float, float]


@dataclass(slots=True)
class _Showdown:
    pot: float
    invested: tuple[float, float]
    key: object                        # clé _sd_ctx : -1 (board complet),
                                       # carte river (turn), paire triée (flop)


@dataclass(slots=True)
class _Chance:
    """Distribution de la prochaine carte : subroots[k] = sous-arbre de deals[k].

    ``n_unknown`` = cartes inconnues PAR CONFRONTATION (52 − board effectif −
    2×2 mains) : 45 au tirage du turn depuis un flop, 44 au tirage de la
    river — le diviseur exact du piège n°1, figé à la construction.
    """
    deals: I64
    subroots: list[int]
    n_unknown: int


@dataclass(slots=True)
class _LeafRollout:
    """Feuille à profondeur limitée (P3) : la street river n'est PAS jouée.

    Valeur = part d'abattage « checked-down » moyennée sur les rivières,
    multipliée par le facteur de réalisation τ du joueur (EQR, L7) — le rôle
    du « value net » des solveurs à recherche limitée (DeepStack/ReBeL), tenu
    ici par une régression apprise de nos propres solves. ``dealt`` = cartes
    de chance déjà tirées au-dessus de la feuille : ``()`` pour un solve
    turn, ``(turn,)`` pour un solve FLOP tronqué au turn.
    """
    pot: float
    invested: tuple[float, float]
    dealt: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RootAction:
    label: str
    frequency: float                   # fréquence pondérée par la range
    per_combo: dict[str, float]        # combo lisible → fréquence


@dataclass(frozen=True, slots=True)
class PostflopResult:
    ev_oop: float                      # jetons, par confrontation moyenne
    ev_ip: float
    pot: float
    exploitability: float              # fraction du pot (0 = Nash exact)
    iterations: int
    n_nodes: int
    root_actions: tuple[RootAction, ...]

    def summary(self) -> str:
        lines = [
            f"EV OOP {self.ev_oop:.2f} + EV IP {self.ev_ip:.2f}"
            f" = {self.ev_oop + self.ev_ip:.2f} (pot {self.pot:g})",
            f"exploitabilité {self.exploitability * 100:.2f} % du pot"
            f" après {self.iterations} itérations · {self.n_nodes} nœuds",
        ]
        for a in self.root_actions:
            lines.append(f"  racine {a.label:<14} {a.frequency * 100:5.1f} %")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXTE D'ABATTAGE (précalculé une fois, par rivière)
# ═══════════════════════════════════════════════════════════════════════════


class _ShowdownCtx:
    """Pour le CFV du joueur p : l'adversaire trié par force + préfixes par carte."""

    __slots__ = ("order", "sorted_q_idx", "member", "posL", "posR",
                 "exact_idx", "card_a", "card_b", "valid_self", "valid_opp")

    def __init__(self, p_cards: I64, p_str: I64, o_cards: I64, o_str: I64,
                 dead: Sequence[int]) -> None:
        n_o = o_cards.shape[0]
        self.valid_opp = np.ones(n_o, dtype=bool)
        self.valid_self = np.ones(p_cards.shape[0], dtype=bool)
        for c in dead:                 # cartes de chance tirées (turn/river)
            self.valid_opp &= ~(o_cards == c).any(axis=1)
            self.valid_self &= ~(p_cards == c).any(axis=1)
        self.order = np.argsort(o_str, kind="stable")
        ss = o_str[self.order]
        # appartenance par carte, en ordre trié : (52, n_o) bool
        oc = o_cards[self.order]
        self.member = np.zeros((N_CARDS, n_o), dtype=bool)
        self.member[oc[:, 0], np.arange(n_o)] = True
        self.member[oc[:, 1], np.arange(n_o)] = True
        # positions de la bande d'égalité de chaque combo du joueur p
        self.posL = np.searchsorted(ss, p_str, side="left")
        self.posR = np.searchsorted(ss, p_str, side="right")
        self.card_a = p_cards[:, 0]
        self.card_b = p_cards[:, 1]
        # combo adverse strictement identique (mêmes 2 cartes) → index trié.
        # inv_order[j] = position de j dans l'ordre trié, en O(n) — le flop
        # construit 2×1176 contextes, l'ancienne recherche O(n²) y est rédhibitoire
        inv_order = np.empty(n_o, dtype=np.int64)
        inv_order[self.order] = np.arange(n_o)
        key_o = o_cards[:, 0] * N_CARDS + o_cards[:, 1]
        key_p = p_cards[:, 0] * N_CARDS + p_cards[:, 1]
        pos_of = {int(k): int(inv_order[j]) for j, k in enumerate(key_o)}
        self.exact_idx = np.array([pos_of.get(int(k), -1) for k in key_p],
                                  dtype=np.int64)

    def cfv(self, q: F64, pot: float, invested_p: float) -> F64:
        """CFV par combo de p, mesure = reach adverse compatible."""
        qs = q[self.order]
        cum = np.concatenate(([0.0], np.cumsum(qs)))
        C = np.zeros((N_CARDS, qs.shape[0] + 1))
        np.cumsum(self.member * qs[None, :], axis=1, out=C[:, 1:])
        q_exact = np.where(self.exact_idx >= 0, qs[self.exact_idx], 0.0)

        below = cum[self.posL] - C[self.card_a, self.posL] - C[self.card_b, self.posL]
        tie = ((cum[self.posR] - cum[self.posL])
               - (C[self.card_a, self.posR] - C[self.card_a, self.posL])
               - (C[self.card_b, self.posR] - C[self.card_b, self.posL])
               + q_exact)                 # le combo identique était retiré 2×
        total = cum[-1]
        compat = (total - C[self.card_a, -1] - C[self.card_b, -1] + q_exact)
        out = pot * (below + 0.5 * tie) - invested_p * compat
        return np.where(self.valid_self, out, 0.0)

    def compat_mass(self, q: F64) -> F64:
        """Reach adverse compatible avec chaque combo de p (pour les folds)."""
        qs = q[self.order]
        C_tot = self.member @ qs
        q_exact = np.where(self.exact_idx >= 0, qs[self.exact_idx], 0.0)
        out = qs.sum() - C_tot[self.card_a] - C_tot[self.card_b] + q_exact
        return np.where(self.valid_self, out, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# SOLVEUR
# ═══════════════════════════════════════════════════════════════════════════


class PostflopSolver:
    """CFR range-vs-range sur un board flop (3), turn (4) ou river (5 cartes).

    Parameters
    ----------
    board
        3 cartes (flop : turns PUIS rivières énumérés par nœuds de chance
        emboîtés — 45×44/2 = 990 confrontations turn+river par feuille),
        4 cartes (turn : les 44 rivières valides sont énumérées) ou
        5 (river exacte).
    oop_range, ip_range
        Ranges pondérées (fréquences mixtes respectées).
    pot, stack
        Pot au début de la street, tapis effectif restant.
    bet_fracs
        Tailles de mise en fraction du pot courant (ex. (0.5, 1.0)).
        Une mise qui dépasse le tapis devient all-in ; doublons fusionnés.
    max_bets
        Nombre maximal de mises/relances par street (2 = bet + raise).
    rake
        Modèle de rake « % + cap » (``pfs.core.rake``), convention won-pot :
        à chaque terminal, le GAGNANT reçoit ``rake.net(pot)`` au lieu du
        pot ; à l'abattage, c'est le pot net qui est partagé (égalité :
        moitié du pot net chacun). Les mises investies ne sont pas rakées —
        la part du perdant/du foldeur reste 0 − investi. Défaut : ``NO_RAKE``
        (identité bit à bit avec le solveur non raké).

        Avec ``rake > 0`` le jeu n'est PLUS à somme constante :
        u_OOP + u_IP = pot − rake(pot terminal) sur chaque branche. CFR reste
        appliqué tel quel (pratique standard des solveurs commerciaux : le
        rake est une perturbation des utilités terminales) ; voir
        ``exploitability`` pour la cible d'équilibre décalée et
        ``expected_rake`` pour le rake espéré payé au profil moyen.
    """

    def __init__(self, board: Sequence[int], oop_range: Range, ip_range: Range,
                 pot: float, stack: float,
                 bet_fracs: Sequence[float] = (0.75,),
                 max_bets: int = 2,
                 rake: RakeModel = NO_RAKE,
                 oop_bet_fracs: Sequence[float] | None = None,
                 ip_bet_fracs: Sequence[float] | None = None,
                 raise_fracs: Sequence[float] | None = None,
                 allin_threshold: float | None = None,
                 add_allin: bool = False,
                 leaf_model: str = "full",
                 eqr_model: object | None = None) -> None:
        board = [int(c) for c in board]
        if len(board) not in (3, 4, 5):
            raise PostflopError(
                "board de 3 (flop), 4 (turn) ou 5 (river) cartes.")
        if len(set(board)) != len(board):
            raise PostflopError("cartes de board dupliquées.")
        if any(not 0 <= c < N_CARDS for c in board):
            raise PostflopError("carte hors domaine.")
        if pot <= 0 or stack <= 0:
            raise PostflopError("pot > 0 et stack > 0 requis.")
        if bet_fracs is None or any(f <= 0 for f in bet_fracs):
            raise PostflopError("bet_fracs : fractions > 0 (tuple vide accepté "
                                "seulement via oop/ip_bet_fracs).")
        if max_bets < 1:
            raise PostflopError("max_bets >= 1.")
        if not isinstance(rake, RakeModel):
            raise PostflopError("rake doit être un RakeModel (pfs.core.rake).")
        self.board = board
        self.pot0 = float(pot)
        self.stack = float(stack)
        self.bet_fracs = tuple(sorted(set(float(f) for f in bet_fracs)))
        self.max_bets = int(max_bets)
        self.rake = rake

        # ── P1 : arbre de mise complet ───────────────────────────────────
        def _sizes(v, name):
            if v is None:
                return None
            v = tuple(sorted(set(float(f) for f in v)))
            if any(f <= 0 for f in v):
                raise PostflopError(f"{name} : fractions > 0 requises.")
            return v
        self.oop_bet_fracs = _sizes(oop_bet_fracs, "oop_bet_fracs")
        self.ip_bet_fracs = _sizes(ip_bet_fracs, "ip_bet_fracs")
        self.raise_fracs = _sizes(raise_fracs, "raise_fracs")
        if allin_threshold is not None and not (0.0 < allin_threshold <= 1.0):
            raise PostflopError("allin_threshold dans (0, 1].")
        self.allin_threshold = allin_threshold
        self.add_allin = bool(add_allin)
        eff_oop = self.oop_bet_fracs if self.oop_bet_fracs is not None else self.bet_fracs
        eff_ip = self.ip_bet_fracs if self.ip_bet_fracs is not None else self.bet_fracs
        if not eff_oop and not eff_ip and not self.add_allin:
            raise PostflopError("aucune taille de mise pour personne.")

        # ── P3 : profondeur limitée ──────────────────────────────────────
        if leaf_model not in ("full", "rollout", "eqr"):
            raise PostflopError("leaf_model ∈ {'full','rollout','eqr'}.")
        if leaf_model == "eqr" and eqr_model is None:
            raise PostflopError("leaf_model='eqr' exige un eqr_model (L7).")
        if leaf_model != "full" and len(board) not in (3, 4):
            raise PostflopError(
                "profondeur limitée : board flop (3 cartes) ou turn (4) — "
                "la feuille remplace le round river.")
        self.leaf_model = leaf_model
        self.eqr_model = eqr_model
        self._tau: list[float] = [1.0, 1.0]
        self._tau_ready = leaf_model != "eqr"

        self.players = [self._make_player(oop_range), self._make_player(ip_range)]
        if min(p.n for p in self.players) == 0:
            raise PostflopError("une range est vide après blocage du board.")

        # ``rivers`` (nom historique) = cartes distribuables au PROCHAIN
        # tirage : [-1] si le board est complet, les 48 rivières (turn) ou
        # les 49 turns (flop). ``_deal_keys`` = complétions du board jusqu'à
        # 5 cartes : (), (r,) ou les 1176 paires (t, r) triées du flop.
        restantes = [c for c in range(N_CARDS) if c not in set(board)]
        self.rivers = (np.array(restantes, dtype=np.int64)
                       if len(board) < 5 else np.array([-1], dtype=np.int64))
        if len(board) == 5:
            self._deal_keys: list[tuple[int, ...]] = [()]
        elif len(board) == 4:
            self._deal_keys = [(r,) for r in restantes]
        else:
            self._deal_keys = [(t, r) for i, t in enumerate(restantes)
                               for r in restantes[i + 1:]]
        self._strengths = {self._sd_key(d): self._strengths_for(d)
                           for d in self._deal_keys}
        self._sd_ctx: dict[tuple[int, object], _ShowdownCtx] = {}
        for d in self._deal_keys:
            k = self._sd_key(d)
            s0, s1 = self._strengths[k]
            self._sd_ctx[(OOP, k)] = _ShowdownCtx(
                self.players[OOP].cards, s0, self.players[IP].cards, s1, d)
            self._sd_ctx[(IP, k)] = _ShowdownCtx(
                self.players[IP].cards, s1, self.players[OOP].cards, s0, d)

        # appartenance par carte + combo identique croisé (folds, μ) —
        # indépendants de la rivière
        self._member: list[np.ndarray] = []
        for p in self.players:
            m = np.zeros((N_CARDS, p.n), dtype=bool)
            m[p.cards[:, 0], np.arange(p.n)] = True
            m[p.cards[:, 1], np.arange(p.n)] = True
            self._member.append(m)
        # masques « ne contient pas la carte c » précalculés : les nœuds de
        # chance du flop les consultent 49×48 fois par traversée
        self._not_member = [~m for m in self._member]
        # cache des matrices de feuille rollout (solve flop tronqué) :
        # t → (W_oop, C) avec W_oop[i,j] = part de gain moyenne d'OOP i
        # contre IP j sur les 44 rivières valides après le turn t, et
        # C[i,j] = fraction de rivières valides (compatibilité moyenne).
        # Côté IP par complémentarité : W_ip = (C − W_oop)ᵀ.
        self._leaf_W: dict[int, tuple[F64, F64]] = {}
        self._compat_pair: np.ndarray | None = None
        self._exact: list[I64] = []
        for trav in (OOP, IP):
            opp = self.players[1 - trav]
            key_o = {int(a * N_CARDS + b): j
                     for j, (a, b) in enumerate(opp.cards)}
            self._exact.append(np.array(
                [key_o.get(int(a * N_CARDS + b), -1)
                 for a, b in self.players[trav].cards], dtype=np.int64))

        self._nodes: list[_Node] = []
        self._root = self._build_decision(OOP, self.pot0, 0.0, (0.0, 0.0),
                                          bets_used=0, dealt=())
        for nd in self._nodes:
            n_act = self.players[nd.player].n
            nd.regrets = np.zeros((len(nd.labels), n_act))
            nd.strat_sum = np.zeros((len(nd.labels), n_act))
        self._iters_done = 0

    # ── construction ──────────────────────────────────────────────────────

    def _make_player(self, rng: Range) -> _Player:
        idx = np.nonzero(rng.weights > 0.0)[0]
        keep, w = [], []
        bset = set(self.board)
        for i in idx:
            a, b = combo_cards(int(i))
            if a in bset or b in bset:
                continue
            keep.append((a, b))
            w.append(float(rng.weights[i]))
        cards = np.array(keep, dtype=np.int64).reshape(-1, 2)
        weights = np.array(w, dtype=np.float64)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return _Player(cards, weights, cards.shape[0])

    @staticmethod
    def _sd_key(dealt: Sequence[int]) -> object:
        """Clé d'abattage d'une complétion du board.

        ``()`` → ``-1`` (board déjà complet), ``(r,)`` → ``r`` (les clés
        historiques des solves river/turn, que les tests consultent), paire
        → tuple trié (l'abattage est symétrique en turn/river).
        """
        if len(dealt) == 0:
            return -1
        if len(dealt) == 1:
            return int(dealt[0])
        return tuple(sorted(int(c) for c in dealt))

    def _strengths_for(self, dealt: Sequence[int]) -> tuple[I64, I64]:
        b = list(self.board) + [int(c) for c in dealt]
        barr = np.array(b, dtype=np.int64)
        out = []
        for p in self.players:
            hands = np.hstack([p.cards, np.tile(barr, (p.n, 1))])
            out.append(evaluate7(hands).astype(np.int64))
        return out[0], out[1]

    def _sizes_for(self, player: int, raising: bool) -> tuple[float, ...]:
        """Tailles applicables : relances dédiées > tailles par joueur > communes."""
        if raising and self.raise_fracs is not None:
            return self.raise_fracs
        per = self.oop_bet_fracs if player == OOP else self.ip_bet_fracs
        return per if per is not None else self.bet_fracs

    def _bet_amounts(self, player: int, pot: float, to_call: float,
                     already: float) -> list[tuple[str, float]]:
        """Montants ajoutés par l'acteur (call inclus pour une relance).

        P1 : tailles séparées OOP/IP et bet/raise ; ``allin_threshold`` t
        convertit toute mise ≥ t×cap en all-in (convention Pio) ;
        ``add_allin`` ajoute l'all-in comme action systématique.
        """
        cap = self.stack - already          # jetons restants de l'acteur
        if cap <= 1e-9:
            return []
        outs: list[tuple[str, float]] = []
        seen = set()

        def _push(amt: float, label: str) -> None:
            amt = min(amt, cap)
            if (self.allin_threshold is not None
                    and amt >= self.allin_threshold * cap):
                amt = cap
            key = round(amt, 6)
            if key in seen or amt <= to_call + 1e-9:
                return
            seen.add(key)
            if abs(amt - cap) < 1e-9:
                label = "all-in"
            outs.append((label, amt))

        for f in self._sizes_for(player, raising=to_call > 0):
            if to_call > 0:                 # relance : call + f × (pot après call)
                _push(to_call + f * (pot + to_call), f"raise {f:g}p")
            else:
                _push(f * pot, f"bet {f:g}p")
        if self.add_allin:
            _push(cap, "all-in")
        return outs

    def _build_decision(self, player: int, pot: float, to_call: float,
                        invested: tuple[float, float], bets_used: int,
                        dealt: tuple[int, ...]) -> int:
        """Construit récursivement ; retourne l'indice du nœud.

        ``dealt`` = cartes de chance déjà tirées au-dessus de ce nœud : le
        board effectif est ``board + dealt`` (3+0 = round flop, 3+1 ou 4+0
        = round turn, 3+2, 4+1 ou 5+0 = round river).
        """
        labels: list[str] = []
        amounts: list[float] = []
        children: list[int] = []
        terminal: list[object] = []
        me, opp = player, 1 - player
        total = len(self.board) + len(dealt)      # cartes de board effectives

        def _after_call() -> object:
            new_inv = list(invested)
            new_inv[me] += to_call
            new_pot = pot + to_call
            if total == 5:                  # board complet → abattage
                return _Showdown(new_pot, tuple(new_inv), self._sd_key(dealt))
            if self.leaf_model != "full" and total == 4:
                # P3 : la feuille remplace le round river (solve turn OU
                # solve flop tronqué au turn — dealt = () ou (turn,))
                return _LeafRollout(new_pot, tuple(new_inv), dealt)
            # chance : distribuer la carte suivante, puis round complet.
            # Diviseur du piège n°1 : 52 − board effectif − 2×2 mains
            # (45 pour flop→turn, 44 pour turn→river).
            bset = set(self.board)
            cards = [c for c in range(N_CARDS)
                     if c not in bset and c not in dealt]
            subroots = [self._build_decision(OOP, new_pot, 0.0,
                                             tuple(new_inv), 0,
                                             dealt + (int(c),))
                        for c in cards]
            return _Chance(np.array(cards, dtype=np.int64), subroots,
                           N_CARDS - total - 4)

        if to_call > 0:
            labels.append("fold")
            amounts.append(0.0)
            children.append(-1)
            terminal.append(_Fold(opp, pot, invested))
            labels.append("call")
            amounts.append(to_call)
            children.append(-1)
            terminal.append(_after_call())
        else:
            labels.append("check")
            amounts.append(0.0)
            children.append(-1)
            if player == IP:                # check derrière clôt la street
                terminal.append(_after_call())
            else:
                terminal.append(None)       # l'IP parle
        if bets_used < self.max_bets:
            for label, amt in self._bet_amounts(player, pot, to_call,
                                                invested[me]):
                labels.append(label)
                amounts.append(amt)
                children.append(-1)
                terminal.append(None)

        node = _Node(player, pot, to_call, invested, labels, amounts,
                     children, terminal)
        self._nodes.append(node)
        my_idx = len(self._nodes) - 1

        for a, lab in enumerate(labels):
            if terminal[a] is not None or children[a] >= 0:
                continue
            if lab == "check":              # OOP check → IP parle
                child = self._build_decision(IP, pot, 0.0, invested,
                                             bets_used, dealt)
            else:                           # bet/raise → l'adversaire répond
                amt = amounts[a]
                new_inv = list(invested)
                new_inv[me] += amt
                child = self._build_decision(opp, pot + amt, amt - to_call,
                                             tuple(new_inv), bets_used + 1,
                                             dealt)
            node.children[a] = child
        return my_idx

    # ── valeurs terminales ────────────────────────────────────────────────

    def _compat_mass(self, trav: int, q: F64,
                     dealt: "int | tuple[int, ...]") -> F64:
        """Reach adverse compatible avec chaque combo de trav (blockers).

        ``q`` doit déjà être masqué des combos adverses contenant les cartes
        de chance tirées ; ici on masque seulement les combos du traverseur
        qui les contiennent. ``dealt`` : tuple des cartes tirées — l'ancien
        scalaire (-1 = aucune, c = une rivière) reste accepté pour les
        appelants historiques (``spot_advisor``).
        """
        if isinstance(dealt, (int, np.integer)):
            dealt = () if int(dealt) < 0 else (int(dealt),)
        T = self._member[1 - trav] @ q
        ex = self._exact[trav]
        q_exact = np.where(ex >= 0, q[np.maximum(ex, 0)], 0.0)
        cards = self.players[trav].cards
        out = q.sum() - T[cards[:, 0]] - T[cards[:, 1]] + q_exact
        for c in dealt:
            out = np.where(self._member[trav][c], 0.0, out)
        return out

    def _leaf_share(self, trav: int, q: F64,
                    dealt: tuple[int, ...] = ()) -> F64:
        """Part d'abattage « checked-down » par combo, moyenne des runouts.

        Mesure = reach adverse compatible (comme les autres CFV) — même
        normalisation que ``_Chance`` (piège n°1) : s'il manque UNE carte au
        board (feuille turn, ou feuille flop tronquée sous le turn ``dealt``),
        44 rivières valides par confrontation ; s'il en manque DEUX (features
        τ à la racine d'un solve flop), 45×44/2 = 990 paires turn+river
        valides par confrontation — chaque paire comptée une fois, l'abattage
        étant symétrique en (turn, river).
        """
        n_trav = self.players[trav].n
        acc = np.zeros(n_trav)
        not_opp = self._not_member[1 - trav]
        bset = set(self.board)
        restantes = [c for c in range(N_CARDS)
                     if c not in bset and c not in dealt]
        manquantes = 5 - len(self.board) - len(dealt)
        if manquantes == 1:
            if (len(self.board) == 3
                    and self.players[OOP].n * self.players[IP].n
                    <= _LEAF_W_MAX):
                # solve flop tronqué : share = W @ q (linéaire en q), matrice
                # par turn précalculée — algébriquement identique à la boucle
                # de CFV triés, ~50× plus rapide par itération. Les solves
                # TURN gardent la boucle (identité bit à bit des goldens).
                W_oop, C = self._leaf_w(int(dealt[0]))
                if trav == OOP:
                    return W_oop @ q
                return (C - W_oop).T @ q
            for r in restantes:
                acc += self._sd_ctx[(trav, self._sd_key(dealt + (r,)))].cfv(
                    q * not_opp[r], 1.0, 0.0)
            return acc / (N_CARDS - len(self.board) - len(dealt) - 4)
        # deux cartes manquantes : double somme sur les paires triées
        for i, t in enumerate(restantes):
            for r in restantes[i + 1:]:
                acc += self._sd_ctx[(trav, self._sd_key((t, r)))].cfv(
                    q * not_opp[t] * not_opp[r], 1.0, 0.0)
        n = N_CARDS - len(self.board) - 4          # 45
        return acc / (n * (n - 1) / 2)             # 45×44/2 = 990

    def _leaf_w(self, t: int) -> tuple[F64, F64]:
        """Matrice de feuille rollout du turn ``t`` (solve flop tronqué).

        ``W_oop[i, j]`` = moyenne, sur les 44 rivières valides pour la
        confrontation (i, j), de [gain strict + ½ égalité] d'OOP i contre
        IP j — zéro si les combos sont incompatibles (carte partagée, turn
        compris). ``C[i, j]`` = fraction de rivières valides (44/44 pour une
        confrontation compatible, 0 sinon). Par symétrie de l'abattage :
        ``W_ip = (C − W_oop)ᵀ`` — gain IP = compatibilité − gain OOP, la
        demi-égalité se conservant d'elle-même.
        """
        cached = self._leaf_W.get(t)
        if cached is not None:
            return cached
        p0, p1 = self.players[OOP], self.players[IP]
        if self._compat_pair is None:
            partage = (p0.cards[:, None, :, None]
                       == p1.cards[None, :, None, :]).any(axis=(2, 3))
            self._compat_pair = ~partage
        base = (self._compat_pair
                & self._not_member[OOP][t][:, None]
                & self._not_member[IP][t][None, :])
        acc = np.zeros((p0.n, p1.n))
        cnt = np.zeros((p0.n, p1.n))
        bset = set(self.board)
        for r in range(N_CARDS):
            if r in bset or r == t:
                continue
            s0, s1 = self._strengths[self._sd_key((t, r))]
            valid = (base & self._not_member[OOP][r][:, None]
                     & self._not_member[IP][r][None, :])
            gt = np.where(s0[:, None] > s1[None, :], 1.0,
                          np.where(s0[:, None] == s1[None, :], 0.5, 0.0))
            acc += gt * valid
            cnt += valid
        n_riv = float(N_CARDS - len(self.board) - 1 - 4)   # 44
        out = (acc / n_riv, cnt / n_riv)
        self._leaf_W[t] = out
        return out

    def _ensure_tau(self) -> None:
        """τ (facteurs de réalisation EQR) — calculé une fois par solve.

        τ_p = EQR(équité, position, SPR, polarité, nuts) renormalisé pour que
        τ_o·ē_o + τ_i·ē_i = 1 : la somme des EV agrégés reste = pot.
        """
        if self._tau_ready:
            return
        feats = []
        for p in (OOP, IP):
            q = self.players[1 - p].weights
            share = self._leaf_share(p, q)
            compat = self._compat_mass(p, q, ())
            with np.errstate(invalid="ignore", divide="ignore"):
                eq_c = np.where(compat > 0, share / np.where(compat > 0, compat, 1), 0.0)
            w = self.players[p].weights * (compat > 0)
            w = w / w.sum() if w.sum() > 0 else w
            eq = float((w * eq_c).sum())
            pol = float(np.sqrt(((eq_c - eq) ** 2 * w).sum()))
            nuts = float(w[eq_c > 0.9].sum())
            feats.append((eq, pol, nuts))
        spr = self.stack / self.pot0
        tau = [self.eqr_model.predict(
            min(max(feats[p][0], 0.02), 0.98), p == IP, spr,
            feats[p][1], feats[p][2]) for p in (OOP, IP)]
        denom = tau[OOP] * feats[OOP][0] + tau[IP] * feats[IP][0]
        if denom <= 0:
            raise PostflopError("τ dégénéré : équités nulles.")
        self._tau = [t / denom for t in tau]
        self._tau_ready = True

    def _terminal_cfv(self, term: object, trav: int, q: F64,
                      dealt: tuple[int, ...]) -> F64:
        """CFV terminal — seul point où le rake agit : toute PART GAGNÉE du
        pot est remplacée par sa valeur nette ``rake.net(pot)`` (convention
        won-pot) ; l'investi, lui, n'est jamais raké."""
        if isinstance(term, _Fold):
            compat = self._compat_mass(trav, q, dealt)
            share = self.rake.net(term.pot) if term.winner == trav else 0.0
            return (share - term.invested[trav]) * compat
        if isinstance(term, _Showdown):
            ctx = self._sd_ctx[(trav, term.key)]
            return ctx.cfv(q, self.rake.net(term.pot), term.invested[trav])
        if isinstance(term, _LeafRollout):
            self._ensure_tau()
            share = self._leaf_share(trav, q, term.dealt)
            compat = self._compat_mass(trav, q, term.dealt)
            pot_net = self.rake.net(term.pot)
            return (self._tau[trav] * pot_net * share
                    - term.invested[trav] * compat)
        raise PostflopError("terminal inconnu.")   # pragma: no cover

    # ── traversée CFR ─────────────────────────────────────────────────────

    def _current_strategy(self, node: _Node) -> F64:
        pos = np.maximum(node.regrets, 0.0)
        tot = pos.sum(axis=0, keepdims=True)
        n_a = pos.shape[0]
        uniform = np.full_like(pos, 1.0 / n_a)
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), uniform)
        if node.locked_mask is not None:      # P2 : nodelock
            sigma = np.where(node.locked_mask[None, :], node.locked_sigma, sigma)
        return sigma

    # ── P2 : node-locking (signature Pio / nodelock 2.0 Wizard) ──────────

    def node_at(self, path: Sequence[str]) -> int:
        """Indice du nœud atteint en suivant les labels d'action depuis la racine.

        Exemple : ``("check",)`` = le nœud IP après check OOP ;
        ``("check", "bet 1p")`` = OOP face au bet IP.
        """
        idx = self._root
        for lab in path:
            node = self._nodes[idx]
            if lab not in node.labels:
                raise PostflopError(
                    f"action {lab!r} absente de {node.labels} à ce nœud.")
            a = node.labels.index(lab)
            if node.children[a] < 0:
                raise PostflopError(f"{lab!r} mène à un terminal, pas un nœud.")
            idx = node.children[a]
        return idx

    def lock_node(self, path: Sequence[str],
                  strategy: "dict[str, float] | F64",
                  combos: str | None = None) -> "PostflopSolver":
        """Verrouille la stratégie d'un nœud (totalement ou par combos).

        Parameters
        ----------
        path
            Chemin d'actions depuis la racine (voir ``node_at``).
        strategy
            ``{label: fréquence}`` (mêmes fréquences pour tous les combos)
            OU un tableau ``(n_actions, n_combos)`` par combo — le format de
            ``average_strategy``, ce qui permet de FIGER une stratégie issue
            d'un autre solve (même arbre, mêmes ranges héros).
        combos
            Spécification de range (ex. ``"22-88, A2s-A9s"``) : seuls ces
            combos sont verrouillés, le reste RE-SOLVE librement autour du
            lock (c'est le node-locking *partiel* de Pio et l'esprit du
            nodelock 2.0 : le solveur ré-équilibre le non-verrouillé).
        """
        idx = self.node_at(path)
        node = self._nodes[idx]
        n = self.players[node.player].n
        if isinstance(strategy, dict):
            freqs_v = np.array([max(0.0, float(strategy.get(lab, 0.0)))
                                for lab in node.labels])
            if freqs_v.sum() <= 0:
                raise PostflopError("stratégie verrouillée vide.")
            freqs = np.tile((freqs_v / freqs_v.sum())[:, None], (1, n))
        else:
            freqs = np.asarray(strategy, dtype=np.float64)
            if freqs.shape != (len(node.labels), n):
                raise PostflopError(
                    f"stratégie ({freqs.shape}) ≠ ({len(node.labels)}, {n}).")
            tot = freqs.sum(axis=0, keepdims=True)
            if (tot <= 0).any():
                raise PostflopError("stratégie verrouillée vide sur un combo.")
            freqs = freqs / tot
        if combos is None:
            mask = np.ones(n, dtype=bool)
        else:
            from pfs.core.range_model import parse_range
            w = parse_range(combos).weights
            key = {int(a * N_CARDS + b): True
                   for i in np.nonzero(w > 0)[0]
                   for a, b in [combo_cards(int(i))]}
            cards = self.players[node.player].cards
            mask = np.array([int(a * N_CARDS + b) in key for a, b in cards])
            if not mask.any():
                raise PostflopError("aucun combo du nœud dans la spécification.")
        if node.locked_mask is None:
            node.locked_mask = np.zeros(n, dtype=bool)
            node.locked_sigma = np.zeros((len(node.labels), n))
        node.locked_mask = node.locked_mask | mask
        node.locked_sigma[:, mask] = freqs[:, mask]
        return self

    def unlock_all(self) -> "PostflopSolver":
        for nd in self._nodes:
            nd.locked_mask = None
            nd.locked_sigma = None
        return self

    def _chance_value(self, term: _Chance, trav: int, q: F64,
                      dealt: tuple[int, ...], weight: float,
                      fixed: bool, best_response: bool) -> F64:
        """Valeur d'un nœud de chance : moyenne des tirages valides.

        Chaque tirage c masque les reach adverses contenant c et annule la
        valeur des combos du traverseur qui le contiennent ; la division par
        ``term.n_unknown`` (45 pour flop→turn, 44 pour →river — piège n°1)
        rend la moyenne exacte PAR CONFRONTATION compatible.
        """
        acc = np.zeros(self.players[trav].n)
        not_opp = self._not_member[1 - trav]
        holds = self._member[trav]
        for c, sub in zip(term.deals, term.subroots):
            c = int(c)
            if fixed:
                v = self._traverse_fixed(sub, trav, q * not_opp[c],
                                         dealt + (c,), best_response)
            else:
                v = self._traverse(sub, trav, q * not_opp[c],
                                   dealt + (c,), weight)
            acc += np.where(holds[c], 0.0, v)
        return acc / term.n_unknown

    def _traverse(self, idx: int, trav: int, reach_opp: F64,
                  dealt: tuple[int, ...], weight: float) -> F64:
        node = self._nodes[idx]
        me = node.player
        sigma = self._current_strategy(node)
        n_trav = self.players[trav].n

        def child_value(a: int) -> F64:
            term = node.terminal[a]
            if isinstance(term, (_Fold, _Showdown, _LeafRollout)):
                return self._terminal_cfv(term, trav, reach_opp, dealt)
            if isinstance(term, _Chance):
                return self._chance_value(term, trav, reach_opp, dealt,
                                          weight, False, False)
            return self._traverse(node.children[a], trav, reach_opp,
                                  dealt, weight)

        if me == trav:
            vals = [child_value(a) for a in range(len(node.labels))]
            V = np.zeros(n_trav)
            for a, v in enumerate(vals):
                V += sigma[a] * v
            for a, v in enumerate(vals):
                node.regrets[a] += v - V
            # (la stratégie moyenne de ce nœud s'accumule pendant la traversée
            #  de l'AUTRE joueur, où reach_opp est le reach de cet acteur)
            return V
        # nœud adverse : sa stratégie pondère son reach ; sa stratégie moyenne
        # s'accumule ici (reach = le sien)
        node.strat_sum += weight * sigma * reach_opp[None, :]
        V = np.zeros(n_trav)
        for a in range(len(node.labels)):
            term = node.terminal[a]
            q_a = reach_opp * sigma[a]
            if isinstance(term, (_Fold, _Showdown, _LeafRollout)):
                V += self._terminal_cfv(term, trav, q_a, dealt)
            elif isinstance(term, _Chance):
                V += self._chance_value(term, trav, q_a, dealt,
                                        weight, False, False)
            else:
                V += self._traverse(node.children[a], trav, q_a,
                                    dealt, weight)
        return V

    def solve(self, iterations: int = 400) -> "PostflopSolver":
        if iterations < 1:
            raise PostflopError("iterations >= 1.")
        for _ in range(iterations):
            t = self._iters_done + 1
            w_strat = (t / (t + 1.0)) ** _GAMMA
            for trav in (OOP, IP):
                self._traverse(self._root, trav,
                               self.players[1 - trav].weights.copy(),
                               (), w_strat)
            pos_w = (t ** _ALPHA) / (t ** _ALPHA + 1.0)
            for nd in self._nodes:
                nd.regrets = np.where(nd.regrets > 0,
                                      nd.regrets * pos_w,
                                      nd.regrets * 0.5)
            self._iters_done = t
        return self

    # ── stratégie moyenne, valeurs, exploitabilité ────────────────────────

    def average_strategy(self, idx: int) -> F64:
        node = self._nodes[idx]
        tot = node.strat_sum.sum(axis=0, keepdims=True)
        n_a = node.strat_sum.shape[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            sigma = np.where(tot > 0, node.strat_sum / np.where(tot > 0, tot, 1.0),
                             1.0 / n_a)
        if node.locked_mask is not None:      # les combos verrouillés rapportent
            sigma = np.where(node.locked_mask[None, :], node.locked_sigma, sigma)
        return sigma

    def _traverse_fixed(self, idx: int, trav: int, reach_opp: F64,
                        dealt: "int | tuple[int, ...]",
                        best_response: bool) -> F64:
        """Valeur pour `trav` : adversaire = stratégie moyenne ; trav = moyenne
        (best_response=False) ou meilleure réponse (True). La meilleure
        réponse maximise PAR ENSEMBLE D'INFORMATION — (nœud public, combo
        privé) — jamais par état de chance (piège n°3). ``dealt`` : tuple
        des cartes de chance tirées ; le scalaire historique (-1) reste
        accepté (``spot_advisor``)."""
        if isinstance(dealt, (int, np.integer)):
            dealt = () if int(dealt) < 0 else (int(dealt),)
        node = self._nodes[idx]
        me = node.player
        sigma = self.average_strategy(idx)
        n_trav = self.players[trav].n

        def child_value(a: int, q: F64) -> F64:
            term = node.terminal[a]
            if isinstance(term, (_Fold, _Showdown, _LeafRollout)):
                return self._terminal_cfv(term, trav, q, dealt)
            if isinstance(term, _Chance):
                return self._chance_value(term, trav, q, dealt,
                                          0.0, True, best_response)
            return self._traverse_fixed(node.children[a], trav, q,
                                        dealt, best_response)

        if me == trav:
            vals = np.stack([child_value(a, reach_opp)
                             for a in range(len(node.labels))])
            if best_response:
                return vals.max(axis=0)
            return (sigma * vals).sum(axis=0)
        V = np.zeros(n_trav)
        for a in range(len(node.labels)):
            V += child_value(a, reach_opp * sigma[a])
        return V

    def _mu(self) -> float:
        """Masse totale des confrontations compatibles (normalisation).

        Ne dépend PAS des tirages : la masse (combo OOP, combo IP)
        compatible est la même quels que soient les turns/rivières à venir.
        """
        compat = self._compat_mass(OOP, self.players[IP].weights, ())
        return float((self.players[OOP].weights * compat).sum())

    def values(self) -> tuple[float, float]:
        """(EV_OOP, EV_IP) en jetons par confrontation compatible moyenne."""
        mu = self._mu()
        out = []
        for p in (OOP, IP):
            v = self._traverse_fixed(self._root, p,
                                     self.players[1 - p].weights.copy(),
                                     (), best_response=False)
            out.append(float((self.players[p].weights * v).sum()) / mu)
        return out[0], out[1]

    def exploitability(self) -> float:
        """(BR_OOP + BR_IP − pot) / (2·pot) — 0 à l'équilibre exact SANS rake.

        Avec rake, le jeu n'est plus à somme constante : à chaque terminal,
        u_OOP + u_IP = pot − rake(pot terminal). À l'équilibre exact du jeu
        raké, BR_p = u_p(σ*) pour chaque joueur, donc

            expl* = (BR_OOP + BR_IP − pot)/(2·pot) = −E[rake]/(2·pot) < 0.

        La quantité reste calculable (et décroît en pratique au fil des
        itérations — la borne de Zinkevich n'est démontrée qu'à somme
        constante) ; seule sa cible d'équilibre est décalée de 0 vers
        −E[rake]/(2·pot), où E[rake] = ``expected_rake()``. La mesure de
        convergence corrigée est ``exploitability() + expected_rake()/(2·pot)``.
        """
        mu = self._mu()
        br = []
        for p in (OOP, IP):
            v = self._traverse_fixed(self._root, p,
                                     self.players[1 - p].weights.copy(),
                                     (), best_response=True)
            br.append(float((self.players[p].weights * v).sum()) / mu)
        return (br[0] + br[1] - self.pot0) / (2.0 * self.pot0)

    def expected_rake(self) -> float:
        """Rake espéré payé au profil moyen : pot initial − (EV_OOP + EV_IP).

        À chaque terminal, u_OOP + u_IP = pot − rake(pot terminal) ; en
        moyennant sur les branches du profil moyen (et les rivières du nœud
        de chance), la somme des EV vaut pot − E[rake]. La différence est
        donc EXACTEMENT le rake espéré payé par confrontation compatible
        moyenne — 0 à la précision machine pour ``NO_RAKE``, et borné par
        min(cap, pct · pot max de l'arbre) sinon.
        """
        ev_oop, ev_ip = self.values()
        return self.pot0 - (ev_oop + ev_ip)

    # ── sortie ────────────────────────────────────────────────────────────

    def root_report(self, top: int = 24) -> tuple[RootAction, ...]:
        from pfs.core.range_model import card_str
        node = self._nodes[self._root]
        sigma = self.average_strategy(self._root)
        w = self.players[node.player].weights
        out = []
        for a, lab in enumerate(node.labels):
            freq = float((sigma[a] * w).sum() / w.sum()) if w.sum() else 0.0
            per: dict[str, float] = {}
            order = np.argsort(-sigma[a] * w)[:top]
            for i in order:
                if w[i] <= 0:
                    continue
                ca, cb = self.players[node.player].cards[i]
                per[f"{card_str(int(ca))}{card_str(int(cb))}"] = float(sigma[a][i])
            out.append(RootAction(lab, freq, per))
        return tuple(out)

    def simplify_report(self, eps: float = 0.01) -> dict:
        """Merge des tailles (signature GTO Wizard) : à la racine, les actions
        de mise dont l'EV (pour l'acteur, contre la stratégie moyenne adverse)
        diffère de moins de ``eps × pot`` sont déclarées fusionnables — la
        simplification standard « ces sizings sont equivalents, n'en garde
        qu'un ».
        """
        if eps <= 0:
            raise PostflopError("eps > 0 requis.")
        node = self._nodes[self._root]
        p = node.player
        w = self.players[p].weights
        mu = self._mu()
        evs: dict[str, float] = {}
        for a, lab in enumerate(node.labels):
            term = node.terminal[a]
            if isinstance(term, (_Fold, _Showdown, _LeafRollout)):
                v = self._terminal_cfv(term, p, self.players[1 - p].weights, ())
            elif isinstance(term, _Chance):
                continue
            else:
                v = self._traverse_fixed(node.children[a], p,
                                         self.players[1 - p].weights.copy(),
                                         (), best_response=False)
            evs[lab] = float((w * v).sum()) / mu
        bets = [(lab, ev) for lab, ev in evs.items()
                if lab not in ("check", "fold", "call")]
        merged: list[list[str]] = []
        for lab, ev in sorted(bets, key=lambda x: x[1]):
            if merged and abs(ev - evs[merged[-1][-1]]) < eps * self.pot0:
                merged[-1].append(lab)
            else:
                merged.append([lab])
        return {
            "action_evs": evs,
            "mergeable": [g for g in merged if len(g) > 1],
            "keep": [g[-1] for g in merged],
            "eps_chips": eps * self.pot0,
        }

    def result(self) -> PostflopResult:
        ev0, ev1 = self.values()
        return PostflopResult(ev0, ev1, self.pot0, self.exploitability(),
                              self._iters_done, len(self._nodes),
                              self.root_report())
