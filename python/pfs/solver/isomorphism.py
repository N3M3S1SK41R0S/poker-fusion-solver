"""
L8 — Isomorphisme de couleurs des boards : 22 100 flops → 1 755 classes.

Statut (audit du 14 août 2026) — brique Phase 2, NON branchée
-------------------------------------------------------------
Atteint uniquement par ``pfs.solver.abstraction`` (lui-même non branché) et
par ``tests/test_isomorphism.py``. Pourquoi : c'est le levier du pré-calcul
des flops de la Phase 2, et ce pré-calcul n'a pas commencé — river et turn
se résolvent aujourd'hui sans canonisation. Ce qui existe déjà : forme
canonique à la Waugh, comptes vérifiés par Burnside ET par énumération
(1 755 / 16 432 / 134 459), ``suit_mapping`` — testés. Accroche naturelle
si on branche : le futur blueprint flop de ``pfs/solver/postflop.py``
(mémoïser chaque solve par ``canonical_board``). Règle du projet : aucun
module fusionné sans route + UI + test de bout en bout.

Sources
-------
- Waugh, K. (2013), *A Fast and Optimal Hand Isomorphism Algorithm*, AAAI
  Workshop on Computer Poker and Imperfect Information — la référence : forme
  canonique par minimalité lexicographique sous permutation des couleurs.
- Gilpin, A. & Sandholm, T. (2007), *Lossless Abstraction of Imperfect
  Information Games*, JACM 54(5) — l'abstraction SANS PERTE (GameShrink) dont
  l'isomorphisme de couleurs est le cas particulier qu'exploitent tous les
  solveurs de référence (PioSOLVER, TexasSolver, GTO+).
- Lemme de Burnside (Cauchy–Frobenius) — la vérification analytique du compte
  des classes, détaillée ci-dessous.

Le principe en une phrase : **les couleurs n'ont pas de valeur intrinsèque.**
Échanger ♠ et ♥ partout (mains ET board) laisse équités, regrets et
stratégies rigoureusement invariants. Le groupe symétrique S₄ des 24
permutations de couleurs agit donc sur les boards, et il suffit de résoudre
UN représentant par orbite : c'est le levier n°1 de la Phase 2 (pré-calcul
des flops), qui divise par ~12,6 le coût du blueprint (22 100 → 1 755).

Forme canonique (définition, à la Waugh) : parmi les 24 images du board par
les permutations de couleurs, le tuple **trié croissant minimal** pour
l'ordre lexicographique. Deux boards sont stratégiquement équivalents à
couleurs près si et seulement si leurs formes canoniques coïncident.

Vérification analytique — lemme de Burnside : |classes| = (1/24)·Σ_σ |Fix(σ)|.
Les orbites de cartes sous σ sont {rang} × (cycles de couleurs) ; un board
fixé est une union d'orbites. Par type de cycle de σ, pour les flops (k=3) :

    identité             (×1) : C(52,3)          = 22 100
    transposition        (×6) : C(26,3) + 13·26  =  2 938
    double transposition (×3) :                        0
    3-cycle              (×8) : C(13,3) + 13     =    299
    4-cycle              (×6) :                        0

→ (22 100 + 6·2 938 + 8·299) / 24 = 42 120 / 24 = **1 755**. Le même lemme
donne 16 432 boards de turn (4 cartes) et 134 459 de river (5 cartes) —
valeurs que ``n_canonical`` retrouve par énumération exhaustive, jamais
codées en dur : c'est l'oracle du module.

Performance : énumération vectorisée NumPy (clé entière 6 bits/carte, minimum
sur les 24 permutations), mémoïsée par rue — flop < 0,1 s, turn ~ 1 s, river
quelques secondes, puis gratuit.

Limites assumées : isomorphisme du board comme ENSEMBLE de cartes — l'ordre
d'arrivée turn/river n'est pas distingué, exactement ce qu'exige le pré-calcul
par flop canonique (et ce que comptent les valeurs publiées ci-dessus).
L'indexation jointe main + board rue par rue (Waugh 2013, §4) viendra avec le
portage complet du solveur.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import N_CARDS, RANKS, SUITS

__all__ = [
    "IsoError",
    "board_str",
    "canonical_board",
    "suit_mapping",
    "enumerate_canonical_flops",
    "n_canonical",
]

I8 = npt.NDArray[np.int8]
I64 = npt.NDArray[np.int64]

N_SUITS: int = len(SUITS)   # 4
N_RANKS: int = len(RANKS)   # 13

_SUIT_PERMS: tuple[tuple[int, ...], ...] = tuple(itertools.permutations(range(N_SUITS)))
"""Le groupe S₄ complet : les 24 permutations de couleurs, ordre lexicographique."""

_STREET_SIZE: dict[str, int] = {"flop": 3, "turn": 4, "river": 5}
"""Nombre de cartes du board par rue."""


class IsoError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION & OUTILS
# ═══════════════════════════════════════════════════════════════════════════


def board_str(cards: Sequence[int]) -> str:
    """Libellé lisible d'une suite de cartes : ``[0, 4, 9]`` → ``'As Ks Qh'``.

    Outil de débogage et de messages — n'impose pas la taille d'un board.

    >>> board_str([0, 5, 10])
    'As Kh Qd'
    """
    labels: list[str] = []
    for c in cards:
        card = int(c)
        if not 0 <= card < N_CARDS:
            raise IsoError(f"carte {card} hors domaine [0, {N_CARDS}).")
        labels.append(f"{RANKS[card // N_SUITS]}{SUITS[card % N_SUITS]}")
    return " ".join(labels)


def _validate_board(cards: Sequence[int]) -> tuple[int, ...]:
    """Contrôle taille (3/4/5), domaine et unicité ; retourne le board en tuple."""
    try:
        board = tuple(int(c) for c in cards)
    except (TypeError, ValueError) as exc:
        raise IsoError(f"cartes illisibles : {cards!r}") from exc
    if len(board) not in (3, 4, 5):
        raise IsoError(
            f"board de {len(board)} cartes : attendu 3 (flop), 4 (turn) ou 5 (river)."
        )
    for card in board:
        if not 0 <= card < N_CARDS:
            raise IsoError(f"carte {card} hors domaine [0, {N_CARDS}).")
    if len(set(board)) != len(board):
        raise IsoError(f"carte dupliquée dans le board {board_str(board)}.")
    return board


def _apply_perm(board: tuple[int, ...], perm: Sequence[int]) -> tuple[int, ...]:
    """Image triée du board par une permutation de couleurs — un point de l'orbite."""
    return tuple(sorted((c - c % N_SUITS) + perm[c % N_SUITS] for c in board))


# ═══════════════════════════════════════════════════════════════════════════
# FORME CANONIQUE (Waugh, 2013)
# ═══════════════════════════════════════════════════════════════════════════


def canonical_board(cards: Sequence[int]) -> tuple[int, ...]:
    """Forme canonique d'un board sous permutation des couleurs.

    Parmi les 24 images du board par le groupe S₄ des couleurs, retourne le
    tuple trié croissant minimal pour l'ordre lexicographique (Waugh, 2013).
    Deux boards ont la même forme canonique si et seulement s'ils sont
    stratégiquement équivalents à couleurs près : mêmes équités, mêmes
    stratégies d'équilibre, mêmes EV. Idempotente par construction (le
    minimum d'une orbite est dans l'orbite).

    Parameters
    ----------
    cards : Sequence[int]
        3, 4 ou 5 cartes distinctes, encodage ``rang*4 + couleur``
        (rang 0 = As, couleurs « shdc »).

    Returns
    -------
    tuple[int, ...]
        Le représentant canonique de la classe, trié croissant.

    Raises
    ------
    IsoError
        Board de taille ≠ 3/4/5, carte hors domaine ou dupliquée.

    Examples
    --------
    >>> board_str(canonical_board([2, 7, 9]))    # Ad Kc Qh, arc-en-ciel
    'As Kh Qd'
    >>> board_str(canonical_board([0, 4, 10]))   # As Ks Qd, bicolore
    'As Ks Qh'
    >>> canonical_board([0, 4, 9]) == canonical_board([1, 5, 8])   # ♠ ↔ ♥
    True
    """
    board = _validate_board(cards)
    return min(_apply_perm(board, perm) for perm in _SUIT_PERMS)


def suit_mapping(cards: Sequence[int]) -> dict[int, int]:
    """Une permutation de couleurs qui envoie le board sur sa forme canonique.

    Retourne ``{ancienne_couleur: nouvelle_couleur}`` sur les 4 couleurs.
    Elle n'est pas unique en général (les couleurs absentes du board restent
    libres) ; le choix est déterministe : première permutation optimale dans
    l'ordre lexicographique de S₄ — en particulier, l'identité pour un board
    déjà canonique. C'est le pont de la Phase 2 : transporter ranges et
    stratégies d'un board brut vers sa classe résolue (et retour, par la
    permutation inverse).

    Parameters
    ----------
    cards : Sequence[int]
        3, 4 ou 5 cartes distinctes, encodage ``rang*4 + couleur``.

    Returns
    -------
    dict[int, int]
        Bijection de {0, 1, 2, 3} dans lui-même telle que l'image triée du
        board soit ``canonical_board(cards)``.

    Raises
    ------
    IsoError
        Mêmes conditions que ``canonical_board``.

    Examples
    --------
    >>> m = suit_mapping([0, 4, 10])            # As Ks Qd : ♦ → ♥
    >>> sorted(m.items())
    [(0, 0), (1, 2), (2, 1), (3, 3)]
    >>> b = [0, 4, 10]
    >>> tuple(sorted(4 * (c // 4) + m[c % 4] for c in b)) == canonical_board(b)
    True
    """
    board = _validate_board(cards)
    best = min(_SUIT_PERMS, key=lambda perm: _apply_perm(board, perm))
    return {suit: best[suit] for suit in range(N_SUITS)}


# ═══════════════════════════════════════════════════════════════════════════
# ÉNUMÉRATION DES CLASSES (l'oracle : 1 755 / 16 432 / 134 459)
# ═══════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def _perm_card_tables() -> I8:
    """Tables (24, 52) : image de chaque carte par chaque permutation de S₄."""
    tables = np.empty((len(_SUIT_PERMS), N_CARDS), dtype=np.int8)
    for i, perm in enumerate(_SUIT_PERMS):
        for card in range(N_CARDS):
            rank, suit = divmod(card, N_SUITS)
            tables[i, card] = rank * N_SUITS + perm[suit]
    return tables


def _pack_sorted(table: I8, boards: I8) -> I64:
    """Image des boards par une permutation, triée puis empaquetée en clé int64.

    6 bits par carte, poids fort = plus petite carte : l'ordre numérique des
    clés coïncide avec l'ordre lexicographique des tuples triés, donc le
    minimum des clés sur S₄ EST la forme canonique.
    """
    mapped = np.sort(table[boards], axis=1)
    keys: I64 = mapped[:, 0].astype(np.int64)
    for j in range(1, mapped.shape[1]):
        keys = (keys << 6) | mapped[:, j]
    return keys


@lru_cache(maxsize=None)
def _enumerate_canonical(k: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Classes canoniques des boards de ``k`` cartes, avec multiplicités.

    Énumération vectorisée : les C(52, k) boards traversent les 24
    permutations, on garde la clé minimale par board, ``np.unique`` regroupe
    les orbites. Mémoïsée sur un résultat IMMUABLE — le coût n'est payé
    qu'une fois par rue, et le cache est incorruptible.
    """
    n = math.comb(N_CARDS, k)
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(N_CARDS), k)),
        dtype=np.int8,
        count=n * k,
    )
    boards = flat.reshape(n, k)
    tables = _perm_card_tables()
    best = _pack_sorted(tables[0], boards)
    for table in tables[1:]:
        np.minimum(best, _pack_sorted(table, boards), out=best)
    uniq, counts = np.unique(best, return_counts=True)
    decoded = np.empty((uniq.size, k), dtype=np.int64)
    for j in range(k):
        decoded[:, k - 1 - j] = (uniq >> (6 * j)) & 0x3F
    return tuple(
        (tuple(int(card) for card in row), int(mult))
        for row, mult in zip(decoded, counts, strict=True)
    )


def enumerate_canonical_flops() -> dict[tuple[int, int, int], int]:
    """Les 1 755 classes canoniques de flops et leur multiplicité.

    Chaque clé est un flop canonique (représentant minimal, trié croissant),
    chaque valeur la taille de son orbite — le nombre de flops bruts
    stratégiquement équivalents. La somme des multiplicités vaut
    C(52, 3) = 22 100. C'est la table maîtresse du pré-calcul Phase 2 :
    résoudre les clés, pondérer les agrégats par les valeurs.

    Returns
    -------
    dict[tuple[int, int, int], int]
        Dictionnaire NEUF à chaque appel (le calcul, lui, est mémoïsé) :
        le modifier ne corrompt pas le cache.

    Examples
    --------
    >>> classes = enumerate_canonical_flops()
    >>> len(classes), sum(classes.values())
    (1755, 22100)
    >>> classes[canonical_board([0, 4, 8])]      # As Ks Qs monotone : 4 flops
    4
    """
    return {(b[0], b[1], b[2]): mult for b, mult in _enumerate_canonical(3)}


def n_canonical(street: str) -> int:
    """Nombre de classes canoniques de boards pour une rue donnée.

    Calculé par énumération exhaustive — jamais codé en dur — et recoupé
    analytiquement par le lemme de Burnside (voir l'en-tête du module) :
    « flop » → 1 755, « turn » → 16 432, « river » → 134 459.

    Parameters
    ----------
    street : str
        ``"flop"``, ``"turn"`` ou ``"river"`` (casse et espaces tolérés).

    Returns
    -------
    int
        Nombre d'orbites des boards de la rue sous S₄.

    Raises
    ------
    IsoError
        Rue inconnue.

    Examples
    --------
    >>> n_canonical("flop")
    1755
    """
    key = street.strip().lower() if isinstance(street, str) else ""
    size = _STREET_SIZE.get(key)
    if size is None:
        raise IsoError(f"rue inconnue : {street!r} — attendu 'flop', 'turn' ou 'river'.")
    return len(_enumerate_canonical(size))
