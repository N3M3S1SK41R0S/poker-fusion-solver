r"""Moteur d'équité — évaluateur 7 cartes vectorisé + énumération exacte / MC.

C'est la brique la plus fondamentale de tout solveur : P(gagner l'abattage).
Tous les solveurs du registre de benchmark l'exposent ; ce module est notre
implémentation, sans dépendance exotique, NumPy pur.

Architecture
------------
1. ``evaluate7`` — évalue N mains de 7 cartes d'un coup (vectorisé).
   Encodage de force : entier uint32, ordre total strict — plus grand = plus
   fort. ``(catégorie << 20) | d1<<16 | d2<<12 | d3<<8 | d4<<4 | d5`` où les
   digits sont les *valeurs* de bris d'égalité (base 16, A=12 … 2=0).
2. ``equity_vs_range`` — équité exacte par énumération complète dès qu'un flop
   est connu (flop : 990 runouts × combos ≈ 10⁶ évaluations, ~1 s ; turn : 46 ;
   river : 1), Monte-Carlo semé préflop.

Convention de cartes : celle de ``range_model`` — ``carte = rang*4 + couleur``
avec RANKS = "AKQJT98765432" (rang 0 = As). L'évaluateur travaille en
*valeurs* ``v = 12 − rang`` (plus grand = plus fort) pour rester lisible.

Références : l'énumération exacte est la méthode standard (Pio, GTO+) ;
l'échantillonnage préflop suit Monte-Carlo classique (Metropolis inutile ici :
tirage uniforme direct sans rejet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import (
    N_CARDS,
    N_COMBOS,
    Range,
    RangeError,
    combo_cards,
)

__all__ = [
    "CATEGORIES", "EquityResult", "evaluate5", "evaluate7",
    "equity_vs_range", "equity_multiway", "hand_category",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

CATEGORIES: tuple[str, ...] = (
    "hauteur", "paire", "deux paires", "brelan", "suite",
    "couleur", "full", "carré", "quinte flush",
)

_CHUNK = 1 << 17  # borne mémoire : ~131k mains par lot


# ═══════════════════════════════════════════════════════════════════════════
# TABLE DES SUITES — précalculée sur les 8192 masques de 13 bits
# ═══════════════════════════════════════════════════════════════════════════

def _straight_top(mask: int) -> int:
    """Version scalaire de référence (pour la table et les tests)."""
    bits = [(mask >> v) & 1 for v in range(13)]
    seq = [bits[12]] + bits  # As en tête (roue), puis 2..A
    best, run = -1, 0
    for i, b in enumerate(seq):
        if b:
            run += 1
            if run >= 5:
                best = i - 1  # i indexe seq ; valeur = i−1 (seq[1] = v0)
        else:
            run = 0
    return best


_STRAIGHT: npt.NDArray[np.int8] = np.array(
    [_straight_top(m) for m in range(1 << 13)], dtype=np.int8
)


# ═══════════════════════════════════════════════════════════════════════════
# ÉVALUATEUR 7 CARTES
# ═══════════════════════════════════════════════════════════════════════════

def _top_k_packed(present: npt.NDArray[np.bool_], k: int) -> I64:
    """Emballe les k plus hautes valeurs présentes en digits base 16.

    ``present`` : (N,13) bool, colonne v. Retour (N,) int64 :
    d1*16^(k−1) + … + dk. S'il manque des valeurs, digits 0 (comparaison
    cohérente : mêmes catégories ⇒ mêmes nombres de digits réels).
    """
    n = present.shape[0]
    packed = np.zeros(n, dtype=np.int64)
    taken = np.zeros(n, dtype=np.int64)
    for v in range(12, -1, -1):
        hit = present[:, v] & (taken < k)
        packed = np.where(hit, packed * 16 + v, packed)
        taken = taken + hit
    # aligne : ceux qui ont pris moins de k digits décalent à gauche
    deficit = k - taken
    packed = packed * (16 ** deficit)
    return packed


def evaluate7(cards: npt.NDArray[np.integer]) -> npt.NDArray[np.uint32]:
    """Évalue N mains de 7 cartes. (N,7) → (N,) uint32, plus grand = plus fort.

    >>> import numpy as np
    >>> from pfs.core.range_model import RANKS, SUITS
    >>> def c(txt):  # 'As' → carte
    ...     return RANKS.index(txt[0]) * 4 + SUITS.index(txt[1])
    >>> royal = np.array([[c('As'),c('Ks'),c('Qs'),c('Js'),c('Ts'),c('2d'),c('3c')]])
    >>> int(evaluate7(royal)[0]) >> 20    # catégorie 8 = quinte flush
    8
    """
    cards = np.asarray(cards, dtype=np.int64)
    if cards.ndim != 2 or cards.shape[1] != 7:
        raise RangeError("evaluate7 attend un tableau (N, 7).")
    if cards.size and (cards.min() < 0 or cards.max() >= N_CARDS):
        raise RangeError("carte hors domaine [0, 52).")
    n = cards.shape[0]
    if n > _CHUNK:
        return np.concatenate(
            [evaluate7(cards[i:i + _CHUNK]) for i in range(0, n, _CHUNK)]
        )

    v = 12 - (cards >> 2)          # (N,7) valeurs, A=12
    s = cards & 3                  # (N,7) couleurs

    # One-hot (N,7,13) des valeurs : c'est le plus gros tableau de la
    # fonction — il servait deux fois (comptes, puis masque de couleur) et
    # était reconstruit à l'identique. Calculé une seule fois.
    onehot = v[:, :, None] == np.arange(13)             # (N,7,13)
    vc = onehot.sum(axis=1)                             # (N,13) comptes
    sc = (s[:, :, None] == np.arange(4)).sum(axis=1)    # (N,4)

    present = vc > 0
    mask = (present @ (1 << np.arange(13, dtype=np.int64))).astype(np.int64)

    # ── couleur (au plus une couleur peut avoir ≥5 cartes sur 7) ──────────
    flush_suit = np.argmax(sc, axis=1)
    has_flush = sc[np.arange(n), flush_suit] >= 5
    in_flush = s == flush_suit[:, None]                       # (N,7)
    # one-hot AND, réduit sur l'axe cartes — pas d'écriture indexée (les
    # indices dupliqués d'un |= fantaisie perdraient des bits)
    fpresent = (onehot & in_flush[:, :, None]).any(axis=1)
    fmask = (fpresent @ (1 << np.arange(13, dtype=np.int64))).astype(np.int64)

    sf_top = _STRAIGHT[fmask]                                 # (N,)
    has_sf = has_flush & (sf_top >= 0)

    st_top = _STRAIGHT[mask]
    has_st = st_top >= 0

    quad_v = np.where((vc == 4).any(axis=1), np.argmax(vc == 4, axis=1), -1)
    trips = vc == 3
    pairs = vc == 2

    # brelan le plus haut ; second brelan éventuel compte comme paire du full
    top_trip = np.where(trips.any(axis=1),
                        12 - np.argmax(trips[:, ::-1], axis=1), -1)
    pair_or_2nd_trip = pairs.copy()
    idx = np.arange(n)
    has_trip = top_trip >= 0
    trips_wo_top = trips.copy()
    trips_wo_top[idx[has_trip], top_trip[has_trip]] = False
    pair_or_2nd_trip |= trips_wo_top
    top_full_pair = np.where(pair_or_2nd_trip.any(axis=1),
                             12 - np.argmax(pair_or_2nd_trip[:, ::-1], axis=1), -1)

    n_pairs = pairs.sum(axis=1)
    top_pair = np.where(pairs.any(axis=1),
                        12 - np.argmax(pairs[:, ::-1], axis=1), -1)
    pairs_wo_top = pairs.copy()
    hasp = top_pair >= 0
    pairs_wo_top[idx[hasp], top_pair[hasp]] = False
    second_pair = np.where(pairs_wo_top.any(axis=1),
                           12 - np.argmax(pairs_wo_top[:, ::-1], axis=1), -1)

    # ── codes par catégorie (calculés partout, sélection à la fin) ────────
    code = np.zeros(n, dtype=np.int64)

    # 8 — quinte flush
    c_sf = (8 << 20) | (np.maximum(sf_top, 0).astype(np.int64) << 16)

    # 7 — carré : valeur du carré + meilleur kicker restant
    rest_q = present.copy()
    hq = quad_v >= 0
    rest_q[idx[hq], quad_v[hq]] = False
    kq = _top_k_packed(rest_q, 1)
    c_quad = (7 << 20) | (np.maximum(quad_v, 0).astype(np.int64) << 16) | (kq << 12)

    # 6 — full : brelan + paire
    c_full = ((6 << 20)
              | (np.maximum(top_trip, 0).astype(np.int64) << 16)
              | (np.maximum(top_full_pair, 0).astype(np.int64) << 12))

    # 5 — couleur : 5 meilleures cartes de la couleur
    c_fl = (5 << 20) | _top_k_packed(fpresent, 5)

    # 4 — suite
    c_st = (4 << 20) | (np.maximum(st_top, 0).astype(np.int64) << 16)

    # 3 — brelan : + 2 kickers
    rest_t = present.copy()
    rest_t[idx[has_trip], top_trip[has_trip]] = False
    c_tr = ((3 << 20) | (np.maximum(top_trip, 0).astype(np.int64) << 16)
            | (_top_k_packed(rest_t, 2) << 8))

    # 2 — deux paires : haute, basse, kicker
    rest_2p = present.copy()
    rest_2p[idx[hasp], top_pair[hasp]] = False
    h2 = second_pair >= 0
    rest_2p[idx[h2], second_pair[h2]] = False
    c_2p = ((2 << 20) | (np.maximum(top_pair, 0).astype(np.int64) << 16)
            | (np.maximum(second_pair, 0).astype(np.int64) << 12)
            | (_top_k_packed(rest_2p, 1) << 8))

    # 1 — paire : + 3 kickers
    rest_1p = present.copy()
    rest_1p[idx[hasp], top_pair[hasp]] = False
    c_1p = ((1 << 20) | (np.maximum(top_pair, 0).astype(np.int64) << 16)
            | (_top_k_packed(rest_1p, 3) << 4))

    # 0 — hauteur
    c_hi = _top_k_packed(present, 5)

    has_quad = quad_v >= 0
    has_full = has_trip & (top_full_pair >= 0)
    conds = [has_sf, has_quad, has_full, has_flush, has_st,
             has_trip, n_pairs >= 2, n_pairs == 1]
    codes = [c_sf, c_quad, c_full, c_fl, c_st, c_tr, c_2p, c_1p]
    code = np.select(conds, codes, default=c_hi)
    return code.astype(np.uint32)


def evaluate5(cards: npt.NDArray[np.integer]) -> npt.NDArray[np.uint32]:
    """Évalue des mains de 5 cartes — oracle scalaire lisible pour les tests.

    Sert de référence indépendante : ``evaluate7`` doit rendre, pour toute
    main de 7 cartes, le max de ``evaluate5`` sur ses 21 sous-mains.
    """
    cards = np.asarray(cards, dtype=np.int64)
    if cards.ndim != 2 or cards.shape[1] != 5:
        raise RangeError("evaluate5 attend un tableau (N, 5).")
    out = np.empty(cards.shape[0], dtype=np.uint32)
    for i, hand in enumerate(cards):
        out[i] = _eval5_scalar(tuple(int(c) for c in hand))
    return out


def _eval5_scalar(hand: tuple[int, ...]) -> int:
    """Évaluation de référence, scalaire, lisible — sert d'oracle aux tests."""
    vs = sorted((12 - (c >> 2) for c in hand), reverse=True)
    ss = [c & 3 for c in hand]
    flush = len(set(ss)) == 1
    mask = 0
    for v in vs:
        mask |= 1 << v
    st = _straight_top(mask) if len(set(vs)) == 5 else -1
    from collections import Counter
    cnt = Counter(vs)
    by = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    def pack(vals: list[int], k: int) -> int:
        vals = vals[:k] + [0] * (k - len(vals))
        p = 0
        for v in vals:
            p = p * 16 + v
        return p

    if flush and st >= 0:
        return (8 << 20) | (st << 16)
    if by[0][1] == 4:
        kick = [v for v in vs if v != by[0][0]]
        return (7 << 20) | (by[0][0] << 16) | (pack(kick, 1) << 12)
    if by[0][1] == 3 and by[1][1] >= 2:
        return (6 << 20) | (by[0][0] << 16) | (by[1][0] << 12)
    if flush:
        return (5 << 20) | pack(vs, 5)
    if st >= 0:
        return (4 << 20) | (st << 16)
    if by[0][1] == 3:
        kick = [v for v in vs if v != by[0][0]]
        return (3 << 20) | (by[0][0] << 16) | (pack(kick, 2) << 8)
    if by[0][1] == 2 and by[1][1] == 2:
        hi, lo = max(by[0][0], by[1][0]), min(by[0][0], by[1][0])
        kick = [v for v in vs if v not in (hi, lo)]
        return (2 << 20) | (hi << 16) | (lo << 12) | (pack(kick, 1) << 8)
    if by[0][1] == 2:
        kick = [v for v in vs if v != by[0][0]]
        return (1 << 20) | (by[0][0] << 16) | (pack(kick, 3) << 4)
    return pack(vs, 5)


def hand_category(code: int) -> str:
    """Nom français de la catégorie encodée."""
    return CATEGORIES[int(code) >> 20]


# ═══════════════════════════════════════════════════════════════════════════
# ÉQUITÉ CONTRE UNE RANGE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class EquityResult:
    """Équité pondérée de héros contre la range adverse."""
    equity: float          # (gains + ½·partages) / total
    win: float
    tie: float
    lose: float
    exact: bool            # énumération complète vs Monte-Carlo
    n_scenarios: int       # (runout, combo) évalués
    std_error: float       # ≈0 si exact ; erreur-type MC sinon

    def __str__(self) -> str:
        mode = "exacte" if self.exact else f"MC ±{self.std_error * 100:.2f} pts"
        return (f"équité {self.equity * 100:.2f} % ({mode}) — "
                f"W {self.win * 100:.1f} / T {self.tie * 100:.1f} / "
                f"L {self.lose * 100:.1f}")


def _villain_combos(rng: Range, dead: set[int]) -> tuple[I64, F64]:
    """Combos adverses de poids > 0 ne heurtant aucune carte morte."""
    idx = np.nonzero(rng.weights > 0.0)[0]
    keep, w = [], []
    for i in idx:
        a, b = combo_cards(int(i))
        if a in dead or b in dead:
            continue
        keep.append((a, b))
        w.append(rng.weights[i])
    if not keep:
        raise RangeError("aucun combo adverse compatible avec les cartes mortes.")
    return np.array(keep, dtype=np.int64), np.array(w, dtype=np.float64)


def equity_vs_range(
    hero: Sequence[int],
    villain: Range,
    board: Sequence[int] = (),
    n_sims: int = 100_000,
    seed: int | None = 0,
) -> EquityResult:
    """Équité de ``hero`` (2 cartes) contre la range adverse sur ``board``.

    Exact par énumération complète dès que le flop est connu ; Monte-Carlo
    semé préflop (0 carte de board). Le partage compte pour moitié.

    >>> from pfs.core.range_model import parse_range
    >>> from pfs.core.range_model import RANKS, SUITS
    >>> def c(t): return RANKS.index(t[0]) * 4 + SUITS.index(t[1])
    >>> # AA contre KK, river connu sans amélioration : équité 1.0
    >>> r = equity_vs_range([c('Ah'), c('Ad')], parse_range("KK"),
    ...                     [c('2s'), c('7d'), c('9c'), c('Jh'), c('3s')])
    >>> r.exact, round(r.equity, 6)
    (True, 1.0)
    """
    hero = [int(c) for c in hero]
    board = [int(c) for c in board]
    if len(hero) != 2 or len(set(hero)) != 2:
        raise RangeError("hero doit être 2 cartes distinctes.")
    if len(board) not in (0, 3, 4, 5):
        raise RangeError("board de 0, 3, 4 ou 5 cartes.")
    dead = set(hero) | set(board)
    if len(dead) != 2 + len(board):
        raise RangeError("collision de cartes hero/board.")
    combos, w = _villain_combos(villain, dead)

    if board:
        return _exact(hero, combos, w, board, dead)
    return _preflop_mc(hero, combos, w, dead, n_sims, seed)


def _compare(hc: npt.NDArray, vc: npt.NDArray, w: F64,
             exact: bool) -> EquityResult:
    """Confronte deux vecteurs de forces déjà évaluées, pondérés par ``w``."""
    total = float(w.sum())
    win = float(w[hc > vc].sum()) / total
    tie = float(w[hc == vc].sum()) / total
    lose = 1.0 - win - tie
    eq = win + 0.5 * tie
    n = int(hc.shape[0])
    se = 0.0 if exact else float(np.sqrt(max(eq * (1 - eq), 1e-12) / n))
    return EquityResult(eq, win, tie, lose, exact, n, se)


def _showdown(hero7: npt.NDArray, vill7: npt.NDArray,
              w: F64, exact: bool, extra_var: float = 0.0) -> EquityResult:
    return _compare(evaluate7(hero7).astype(np.int64),
                    evaluate7(vill7).astype(np.int64), w, exact)


def _exact(hero: list[int], combos: I64, w: F64,
           board: list[int], dead: set[int]) -> EquityResult:
    remaining = [c for c in range(N_CARDS) if c not in dead]
    need = 5 - len(board)

    if need == 0:
        runouts = np.empty((1, 0), dtype=np.int64)
    elif need == 1:
        runouts = np.array([[c] for c in remaining], dtype=np.int64)
    else:  # flop → C(45,2) = 990 runouts
        runouts = np.array(
            [[a, b] for i, a in enumerate(remaining) for b in remaining[i + 1:]],
            dtype=np.int64)

    n_r, n_c = runouts.shape[0], combos.shape[0]
    # produit cartésien (runout × combo), filtré des collisions runout∩combo
    r_idx = np.repeat(np.arange(n_r), n_c)
    c_idx = np.tile(np.arange(n_c), n_r)
    R, C = runouts[r_idx], combos[c_idx]           # (M,need), (M,2)
    if need:
        clash = (R[:, :, None] == C[:, None, :]).any(axis=(1, 2))
        keep = ~clash
        R, C, c_idx, r_idx = R[keep], C[keep], c_idx[keep], r_idx[keep]
    M = R.shape[0]
    b = np.array(board, dtype=np.int64)

    # La main du héros ne dépend QUE du runout (ses 2 cartes et le board sont
    # fixes) : on l'évalue une fois par runout — 990 évaluations au flop au
    # lieu de 169 290 — puis on ré-indexe. Le résultat est identique au bit
    # près ; seul le travail redondant disparaît.
    hero_per_runout = np.hstack([
        np.tile(hero, (n_r, 1)), np.tile(b, (n_r, 1)), runouts,
    ])
    hc = evaluate7(hero_per_runout).astype(np.int64)[r_idx]

    vill7 = np.hstack([C, np.tile(b, (M, 1)), R])
    vc = evaluate7(vill7).astype(np.int64)
    return _compare(hc, vc, w[c_idx], exact=True)


def _preflop_mc(hero: list[int], combos: I64, w: F64, dead: set[int],
                n_sims: int, seed: int | None) -> EquityResult:
    if n_sims <= 0:
        raise RangeError("n_sims doit être > 0.")
    g = np.random.default_rng(seed)
    # 1) tire un combo adverse par simulation, proportionnellement aux poids
    p = w / w.sum()
    pick = g.choice(combos.shape[0], size=n_sims, p=p)
    C = combos[pick]                                            # (n,2)
    # 2) tire 5 cartes de board hors {hero ∪ combo} — par permutation partielle
    deck = np.array([c for c in range(N_CARDS) if c not in dead],
                    dtype=np.int64)                             # 50 cartes
    keys = g.random((n_sims, deck.shape[0]), dtype=np.float32)
    # les cartes du combo adverse sont exclues en poussant leur clé à +∞
    excl = (deck[None, :] == C[:, 0:1]) | (deck[None, :] == C[:, 1:2])
    keys[excl] = np.inf
    order = np.argsort(keys, axis=1)[:, :5]
    B = deck[order]                                             # (n,5)
    hero7 = np.hstack([np.tile(hero, (n_sims, 1)), B])
    vill7 = np.hstack([C, B])
    # NB : le tirage du combo étant déjà pondéré, poids uniformes ici
    return _showdown(hero7, vill7, np.ones(n_sims), exact=False)


# ═══════════════════════════════════════════════════════════════════════════
# ÉQUITÉ MULTIWAY — héros contre N ranges (levier L3 du benchmark)
# ═══════════════════════════════════════════════════════════════════════════
#
# Le pot multiway se partage entre les mains les plus fortes ; l'équité de
# héros est sa part ESPÉRÉE du pot : E[1{héros ∈ max} / |argmax|].
# Exact par énumération sur la river avec ≤ 2 vilains (produit de combos,
# collisions filtrées, poids w₁·w₂) ; Monte-Carlo pondéré avec rejet de
# collisions au-delà. C'est le terrain de MonkerSolver — ici pour l'équité.


def equity_multiway(
    hero: Sequence[int],
    villains: Sequence[Range],
    board: Sequence[int] = (),
    n_sims: int = 100_000,
    seed: int | None = 0,
) -> EquityResult:
    """Part espérée du pot de ``hero`` contre plusieurs ranges.

    ``equity`` = E[part du pot] (les égalités partagent). ``win`` = P(gain
    intégral), ``tie`` = P(partage), ``lose`` = P(part nulle).

    Exact (énumération) si board complet et ≤ 2 vilains ; sinon Monte-Carlo
    pondéré, tirages de combos avec rejet des collisions, erreur-type fournie.

    >>> from pfs.core.range_model import parse_range
    >>> from pfs.core.range_model import RANKS, SUITS
    >>> def c(t): return RANKS.index(t[0]) * 4 + SUITS.index(t[1])
    >>> # River figée : AA tient le dessus sur KK ET QQ → équité 1.0
    >>> r = equity_multiway([c('Ah'), c('Ad')],
    ...                     [parse_range("KK"), parse_range("QQ")],
    ...                     [c('2s'), c('7d'), c('9c'), c('Jh'), c('3s')])
    >>> r.exact, round(r.equity, 6)
    (True, 1.0)
    """
    hero = [int(c) for c in hero]
    board = [int(c) for c in board]
    if len(hero) != 2 or len(set(hero)) != 2:
        raise RangeError("hero doit être 2 cartes distinctes.")
    if len(board) not in (0, 3, 4, 5):
        raise RangeError("board de 0, 3, 4 ou 5 cartes.")
    if not villains:
        raise RangeError("au moins une range adverse.")
    if len(villains) == 1:
        return equity_vs_range(hero, villains[0], board, n_sims, seed)
    dead = set(hero) | set(board)
    if len(dead) != 2 + len(board):
        raise RangeError("collision de cartes hero/board.")

    packs = [_villain_combos(v, dead) for v in villains]

    if len(board) == 5 and len(villains) == 2:
        return _multiway_exact_river(hero, board, packs)
    return _multiway_mc(hero, board, packs, dead, n_sims, seed)


def _share_stats(share: F64, w: F64) -> tuple[float, float, float, float]:
    """(équité, win, tie, lose) à partir des parts par scénario."""
    tot = float(w.sum())
    eq = float((share * w).sum()) / tot
    win = float(w[share >= 1.0 - 1e-12].sum()) / tot
    lose = float(w[share <= 1e-12].sum()) / tot
    return eq, win, 1.0 - win - lose, lose


def _multiway_exact_river(
    hero: list[int], board: list[int],
    packs: list[tuple[I64, F64]],
) -> EquityResult:
    (c1, w1), (c2, w2) = packs
    b = np.array(board, dtype=np.int64)

    # forces : héros (scalaire), et chaque combo de chaque vilain (une passe)
    s_h = int(evaluate7(np.hstack([np.array([hero]), b[None, :]]))[0])
    s1 = evaluate7(np.hstack([c1, np.tile(b, (c1.shape[0], 1))])).astype(np.int64)
    s2 = evaluate7(np.hstack([c2, np.tile(b, (c2.shape[0], 1))])).astype(np.int64)

    # produit cartésien vilain1 × vilain2, collisions v1∩v2 filtrées
    n1, n2 = c1.shape[0], c2.shape[0]
    i1 = np.repeat(np.arange(n1), n2)
    i2 = np.tile(np.arange(n2), n1)
    clash = (c1[i1][:, :, None] == c2[i2][:, None, :]).any(axis=(1, 2))
    i1, i2 = i1[~clash], i2[~clash]
    if i1.size == 0:
        raise RangeError("aucune paire de combos adverses compatible.")
    w = w1[i1] * w2[i2]
    m = np.maximum(s1[i1], s2[i2])                    # meilleur vilain
    # part du héros : 0 si battu ; sinon 1/(1 + nb de vilains à égalité)
    tie1 = (s1[i1] == s_h).astype(np.float64)
    tie2 = (s2[i2] == s_h).astype(np.float64)
    share = np.where(s_h > m, 1.0,
                     np.where(s_h < m, 0.0, 1.0 / (1.0 + tie1 + tie2)))
    eq, win, tie, lose = _share_stats(share, w)
    return EquityResult(eq, win, tie, lose, True, int(i1.size), 0.0)


def _multiway_mc(
    hero: list[int], board: list[int],
    packs: list[tuple[I64, F64]], dead: set[int],
    n_sims: int, seed: int | None,
) -> EquityResult:
    if n_sims <= 0:
        raise RangeError("n_sims doit être > 0.")
    g = np.random.default_rng(seed)
    nv = len(packs)
    probs = [w / w.sum() for _, w in packs]

    # 1) tirages de combos par vilain, rejet des lignes en collision
    picks = np.empty((n_sims, nv), dtype=np.int64)
    for k, ((_, w), p) in enumerate(zip(packs, probs)):
        picks[:, k] = g.choice(w.shape[0], size=n_sims, p=p)
    cards = np.stack([packs[k][0][picks[:, k]] for k in range(nv)], axis=1)
    # (n, nv, 2) → collision si une carte apparaît deux fois dans la ligne
    flat = cards.reshape(n_sims, nv * 2)
    bad = (flat[:, :, None] == flat[:, None, :]).sum(axis=(1, 2)) > nv * 2
    tries = 0
    while bad.any():
        tries += 1
        if tries > 200:
            raise RangeError("ranges trop bloquées : rejet de collisions sans fin.")
        idx = np.nonzero(bad)[0]
        for k, ((cc, w), p) in enumerate(zip(packs, probs)):
            picks[idx, k] = g.choice(w.shape[0], size=idx.size, p=p)
        cards = np.stack([packs[k][0][picks[:, k]] for k in range(nv)], axis=1)
        flat = cards.reshape(n_sims, nv * 2)
        bad = (flat[:, :, None] == flat[:, None, :]).sum(axis=(1, 2)) > nv * 2

    # 2) runout : permutation partielle du deck hors cartes mortes+vilains
    need = 5 - len(board)
    if need:
        deck = np.array([c for c in range(N_CARDS) if c not in dead],
                        dtype=np.int64)
        keys = g.random((n_sims, deck.shape[0]), dtype=np.float32)
        excl = (deck[None, None, :] == flat[:, :, None]).any(axis=1)
        keys[excl] = np.inf
        order = np.argsort(keys, axis=1)[:, :need]
        runout = deck[order]
    else:
        runout = np.empty((n_sims, 0), dtype=np.int64)
    full_board = np.hstack([np.tile(board, (n_sims, 1)).astype(np.int64), runout])

    # 3) forces et part du pot
    s_h = evaluate7(np.hstack([np.tile(hero, (n_sims, 1)), full_board])
                    ).astype(np.int64)
    s_v = np.stack([
        evaluate7(np.hstack([cards[:, k, :], full_board])).astype(np.int64)
        for k in range(nv)], axis=1)                       # (n, nv)
    best_v = s_v.max(axis=1)
    n_tied = (s_v == s_h[:, None]).sum(axis=1)
    share = np.where(s_h > best_v, 1.0,
                     np.where(s_h < best_v, 0.0, 1.0 / (1.0 + n_tied)))
    eq, win, tie, lose = _share_stats(share, np.ones(n_sims))
    se = float(np.sqrt(max(share.var(), 1e-12) / n_sims))
    return EquityResult(eq, win, tie, lose, False, n_sims, se)
