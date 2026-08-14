r"""
Abstraction de l'espace des mains par buckets de force — brique « cartes » du
blueprint flop (Phase 2).

Statut (audit du 14 août 2026) — brique Phase 2, NON branchée
-------------------------------------------------------------
Importée par aucun code applicatif ni script livré : seul
``tests/test_abstraction.py`` l'exerce (et ce module est l'unique
consommateur de ``pfs.solver.isomorphism``). Pourquoi : le blueprint flop
qui doit la consommer n'existe pas encore — ``PostflopSolver`` résout river
et turn sans abstraction, et c'est exact à ces rues-là ; bucketer n'a de
sens qu'au flop complet (Phase 2). Ce qui existe déjà : EHS exact mémoïsé
par classe canonique, buckets quantiles et k-means 1-D déterministe,
caractéristiques distribution-aware (Johanson 2013), validation contre
l'oracle ``equity_vs_range`` au turn — testés. Accroche naturelle si on
branche : l'extension flop de ``pfs/solver/postflop.py`` (CFR sur
``bucket_assignments`` par classe canonique de ``canonical_board``).
Règle du projet : aucun module fusionné sans route + UI + test de bout en
bout.

Un solveur blueprint ne peut pas traiter les 1 326 combos comme des états
distincts sur chacun des 1 755 flops canoniques : l'abstraction regroupe les
mains en K buckets stratégiquement similaires, et CFR (Zinkevich et al., 2007)
tourne sur les buckets. Ce module fournit la métrique (force espérée de la
main), les deux découpages classiques (quantiles en masse, k-means 1-D) et les
caractéristiques « distribution-aware » qui séparent mains faites et tirages.

Théorie
-------
- **Force de main (HS) et force espérée (EHS)** : Billings, Davidson,
  Schaeffer & Szafron (2002), *The Challenge of Poker*, Artificial
  Intelligence 134 — HS = percentile de force de la main parmi les mains
  possibles sur un board final ; EHS = moyenne de HS sur les runouts.
- **Buckets par force espérée** : Gilpin & Sandholm (2006, AAAI ; 2007,
  AAMAS) — regroupement des mains par espérance de force (quantiles,
  k-means) pour l'abstraction des solveurs de Texas Hold'em.
- **Abstraction distribution-aware** : Johanson, Burch, Valenzano & Bowling
  (2013), *Evaluating State-Space Abstractions in Extensive-Form Games*,
  AAMAS — E[HS] seul confond mains faites et tirages ; les abstractions
  fondées sur la DISTRIBUTION de la force (dont le couple (E[HS], E[HS²]))
  produisent des stratégies moins exploitables. Ici : (EHS, σ[HS]), qui est
  en bijection avec (E[HS], E[HS²]) puisque σ² = E[HS²] − E[HS]².
- **k-means 1-D** : Lloyd (1982), *Least Squares Quantization in PCM*, IEEE
  Trans. Information Theory 28(2) — algorithme des centroïdes, rendu
  déterministe (initialisation aux quantiles de masse, aucun aléa).
- **Isomorphisme de couleurs** : Waugh (2013) — la table de force est
  calculée UNE FOIS par classe canonique de board (``pfs.solver.isomorphism``)
  puis partagée par les 24 images du board sous S₄ : la mémoïsation suit
  exactement le levier ×12,6 du pré-calcul des flops de la Phase 2.

Définition exacte de la force utilisée
--------------------------------------
Pour un board final :math:`B` (5 cartes) et un combo héros :math:`c`, la main
adverse aléatoire est uniforme sur les :math:`\binom{45}{2} = 990` combos
évitant le board ET les deux cartes du héros (card removal exact) :

.. math::

   HS(c, B) = \frac{\#\{c' : s(c') < s(c)\} + \tfrac12\,\#\{c' :
              s(c') = s(c)\}}{990}

c'est le percentile de force de la main — P(battre) + ½·P(partager) contre
une main aléatoire, exactement l'équité d'abattage contre la range complète.
Puis :

.. math::

   EHS(c) = \operatorname{moy}_B\, HS(c, B), \qquad
   \sigma(c) = \operatorname{ec.type}_B\, HS(c, B)

sur les runouts :math:`B \supset` board (1 081 au flop, 46 au turn, une fois
retirés ceux qui heurtent le combo). Les nuts absolus valent HS = 1 exactement.

Le card removal en une passe : les 1 326 combos sont évalués et triés par
board final, puis des sommes préfixes PAR CARTE (52 lignes) donnent, pour
chaque combo (a, b), le nombre de combos plus faibles/égaux contenant a ou b
— le même schéma en O(52·n) que l'abattage de ``pfs.solver.postflop``.
Invariant vérifiable : au turn, EHS coïncide (à l'arrondi flottant près) avec
``equity_vs_range(combo, Range.full(), board).equity`` — les deux calculs
énumèrent le même espace (river × combo adverse) par des chemins distincts.

Coût : flop = 1 176 runouts × 1 326 combos ≈ 1,6 M évaluations 7 cartes
(≈ 5 s, une seule fois par classe canonique, et pour TOUS les combos à la
fois) ; turn = 48 runouts (< 0,3 s). ``n_rollouts`` sous-échantillonne les
runouts (tirage sans remise, graine fixée → reproductible) quand le budget
l'exige ; l'échantillon est tiré dans l'espace CANONIQUE, donc deux boards
isomorphes donnent des valeurs strictement identiques.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from typing import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.equity import evaluate7
from pfs.core.range_model import (
    N_CARDS,
    N_COMBOS,
    Range,
    card_str,
    combo_index,
)
from pfs.core.range_model import combo_cards as _cards_of_combo
from pfs.solver.isomorphism import canonical_board, suit_mapping

__all__ = [
    "AbstractionError",
    "BucketSummary",
    "abstract_range",
    "bucket_assignments",
    "distribution_aware_features",
    "expected_hand_strength",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

_RUNOUT_CHUNK: int = 64
"""Runouts par lot : 64 × 1326 = 84 864 mains ≤ la borne interne de ``evaluate7``."""

_METHODS: tuple[str, ...] = ("percentile", "kmeans")
_KMEANS_MAX_ITER: int = 200
_MIN_ROLLOUTS_FLOP: int = 96
"""Un combo bloque au plus 95 des 1 176 runouts d'un flop : 96 tirages sans
remise garantissent (principe des tiroirs) au moins un runout valide par combo."""


class AbstractionError(ValueError):
    """Entrée ou paramétrage invalide du module d'abstraction."""


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def _validate_board(board: Sequence[int]) -> tuple[int, ...]:
    """Contrôle un board de flop (3 cartes) ou de turn (4), distinctes, dans [0, 52)."""
    try:
        cards = tuple(int(c) for c in board)
    except (TypeError, ValueError) as exc:
        raise AbstractionError(f"board illisible : {board!r}") from exc
    if len(cards) not in (3, 4):
        raise AbstractionError(
            f"board de {len(cards)} cartes : l'abstraction moyenne la force sur les "
            "runouts, il faut un flop (3 cartes) ou un turn (4)."
        )
    for c in cards:
        if not 0 <= c < N_CARDS:
            raise AbstractionError(f"carte {c} hors domaine [0, {N_CARDS}).")
    if len(set(cards)) != len(cards):
        raise AbstractionError("carte dupliquée dans le board.")
    return cards


def _validate_combos(combos: npt.ArrayLike, board: tuple[int, ...]) -> I64:
    """Contrôle un tableau (n, 2) de combos : entiers, cartes distinctes, hors board."""
    raw = np.asarray(combos)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] != 2:
        raise AbstractionError("combo_cards doit être un tableau (n, 2) non vide.")
    if not np.issubdtype(raw.dtype, np.integer):
        raise AbstractionError("combo_cards doit contenir des entiers (cartes 0–51).")
    cards = raw.astype(np.int64)
    if cards.min() < 0 or cards.max() >= N_CARDS:
        raise AbstractionError(f"carte hors domaine [0, {N_CARDS}) dans combo_cards.")
    if np.any(cards[:, 0] == cards[:, 1]):
        raise AbstractionError("un combo exige deux cartes distinctes.")
    clash = np.isin(cards, np.array(board, dtype=np.int64)).any(axis=1)
    if clash.any():
        i = int(np.argmax(clash))
        raise AbstractionError(
            f"le combo {card_str(int(cards[i, 0]))}{card_str(int(cards[i, 1]))} "
            "heurte une carte du board."
        )
    return cards


def _validate_rollouts(n_rollouts: int | None) -> int | None:
    if n_rollouts is None:
        return None
    if isinstance(n_rollouts, bool) or not isinstance(n_rollouts, (int, np.integer)):
        raise AbstractionError("n_rollouts doit être un entier ≥ 1, ou None (exact).")
    if n_rollouts < 1:
        raise AbstractionError(f"n_rollouts = {n_rollouts} : il faut au moins 1 runout.")
    return int(n_rollouts)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise AbstractionError("seed doit être un entier.")
    return int(seed)


# ═══════════════════════════════════════════════════════════════════════════
# TABLE DE FORCE (mémoïsée par classe canonique de board — Waugh, 2013)
# ═══════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=1)
def _combo_pairs() -> I64:
    """Les 1 326 combos canoniques en tableau (1326, 2), lecture seule."""
    pairs = np.array(
        [_cards_of_combo(i) for i in range(N_COMBOS)], dtype=np.int64
    )
    pairs.flags.writeable = False
    return pairs


@lru_cache(maxsize=1)
def _card_membership() -> npt.NDArray[np.bool_]:
    """Table (52, 1326) : ``M[x, c]`` = la carte x appartient au combo c."""
    pairs = _combo_pairs()
    member = np.zeros((N_CARDS, N_COMBOS), dtype=np.bool_)
    member[pairs[:, 0], np.arange(N_COMBOS)] = True
    member[pairs[:, 1], np.arange(N_COMBOS)] = True
    member.flags.writeable = False
    return member


@lru_cache(maxsize=8)
def _hs_moments(
    board: tuple[int, ...], n_rollouts: int | None, seed: int
) -> tuple[F64, F64]:
    """(EHS, écart-type de HS) des 1 326 combos sur un board CANONIQUE.

    Cœur calculatoire du module : énumère les runouts (tous, ou un
    sous-échantillon sans remise de taille ``n_rollouts`` semé par ``seed``),
    évalue les 1 326 combos sur chaque board final en un seul appel vectorisé
    à ``evaluate7``, puis convertit chaque force en percentile AVEC card
    removal exact : tri des combos valides, recherche dichotomique pour les
    bandes plus-faible/égal, et sommes préfixes par carte (52 lignes) pour
    retrancher les combos adverses heurtant les deux cartes du héros. Les
    moments sont accumulés en une passe (Σ HS, Σ HS²), d'où EHS = Σ/n et
    σ = √(Σ²/n − EHS²).

    Parameters
    ----------
    board : tuple[int, ...]
        Board canonique (3 ou 4 cartes) — la clé de cache. Les appelants
        passent par ``canonical_board`` pour partager la table entre les
        24 images du board sous permutation des couleurs.
    n_rollouts : int | None
        ``None`` = énumération exacte ; sinon taille du sous-échantillon.
    seed : int
        Graine du tirage sans remise (ignorée si l'énumération est exacte).

    Returns
    -------
    tuple[ndarray, ndarray]
        Deux tableaux (1326,) float64 en lecture seule : EHS et écart-type.
        NaN pour les combos sans runout valide (collision avec le board, ou
        sous-échantillon trop court).
    """
    pairs = _combo_pairs()
    n_board = len(board)
    remaining = [c for c in range(N_CARDS) if c not in board]
    all_runouts = np.array(
        list(itertools.combinations(remaining, 5 - n_board)), dtype=np.int64
    )
    if n_rollouts is not None and n_rollouts < all_runouts.shape[0]:
        gen = np.random.default_rng(seed)
        pick = np.sort(
            gen.choice(all_runouts.shape[0], size=n_rollouts, replace=False)
        )
        runouts = all_runouts[pick]
    else:
        runouts = all_runouts

    b_arr = np.array(board, dtype=np.int64)
    member = _card_membership()
    board_clash = np.isin(pairs, b_arr).any(axis=1)          # (1326,)
    count = np.zeros(N_COMBOS, dtype=np.float64)
    sum_hs = np.zeros(N_COMBOS, dtype=np.float64)
    sum_hs2 = np.zeros(N_COMBOS, dtype=np.float64)

    for start in range(0, runouts.shape[0], _RUNOUT_CHUNK):
        chunk = runouts[start:start + _RUNOUT_CHUNK]
        r = chunk.shape[0]
        hands = np.empty((r, N_COMBOS, 7), dtype=np.int64)
        hands[:, :, 0:2] = pairs[None, :, :]
        hands[:, :, 2:2 + n_board] = b_arr[None, None, :]
        hands[:, :, 2 + n_board:] = chunk[:, None, :]
        strength = evaluate7(hands.reshape(-1, 7)).astype(np.int64)
        strength = strength.reshape(r, N_COMBOS)
        run_clash = (pairs[None, :, :, None] == chunk[:, None, None, :]).any(axis=(2, 3))
        valid = ~board_clash[None, :] & ~run_clash            # (r, 1326)
        for i in range(r):
            vidx = np.nonzero(valid[i])[0]                    # (m,) combos du board final
            m = vidx.size
            s_valid = strength[i, vidx]
            order = np.argsort(s_valid, kind="stable")
            sorted_s = s_valid[order]
            lo = np.searchsorted(sorted_s, s_valid, side="left")
            hi = np.searchsorted(sorted_s, s_valid, side="right")
            # Sommes préfixes par carte le long de l'ordre de force croissante :
            # prefix[x, k] = nb de combos parmi les k plus faibles contenant la carte x.
            prefix = np.zeros((N_CARDS, m + 1), dtype=np.int64)
            np.cumsum(member[:, vidx[order]], axis=1, out=prefix[:, 1:])
            card_a = pairs[vidx, 0]
            card_b = pairs[vidx, 1]
            weak_a = prefix[card_a, lo]
            weak_b = prefix[card_b, lo]
            eq_a = prefix[card_a, hi] - weak_a                # inclut le combo lui-même
            eq_b = prefix[card_b, hi] - weak_b                # idem
            tot_a = prefix[card_a, m]
            tot_b = prefix[card_b, m]
            # Card removal exact (inclusion–exclusion ; seul le combo lui-même
            # contient a ET b) : adversaires plus faibles, égaux, et dénominateur.
            n_weaker = (lo - weak_a - weak_b).astype(np.float64)
            n_equal = (hi - lo - 1 - (eq_a + eq_b - 2)).astype(np.float64)
            n_opp = (m - tot_a - tot_b + 1).astype(np.float64)   # 990 sur board réel
            hs = (n_weaker + 0.5 * n_equal) / n_opp
            sum_hs[vidx] += hs
            sum_hs2[vidx] += hs * hs
            count[vidx] += 1.0

    seen = count > 0
    mean = np.full(N_COMBOS, np.nan)
    std = np.full(N_COMBOS, np.nan)
    mean[seen] = sum_hs[seen] / count[seen]
    std[seen] = np.sqrt(
        np.maximum(sum_hs2[seen] / count[seen] - mean[seen] ** 2, 0.0)
    )
    mean.flags.writeable = False
    std.flags.writeable = False
    return mean, std


def _moments_for(
    cards: I64, board: tuple[int, ...], n_rollouts: int | None, seed: int
) -> tuple[F64, F64]:
    """(EHS, écart-type) des combos demandés, via la table du board canonique.

    Transport isomorphe (Waugh, 2013) : la permutation de couleurs qui envoie
    le board sur sa forme canonique est appliquée aux combos, l'invariance de
    la force sous S₄ garantissant EHS(c, B) = EHS(σ(c), σ(B)) exactement.
    ``n_rollouts`` couvrant tous les runouts est normalisé en mode exact
    (``None``, graine 0) pour partager l'entrée de cache correspondante.
    """
    n_total = comb(N_CARDS - len(board), 5 - len(board))
    if n_rollouts is not None and n_rollouts >= n_total:
        n_rollouts = None
    if n_rollouts is None:
        seed = 0
    canon = canonical_board(board)
    mapping = suit_mapping(board)
    perm = np.array([mapping[s] for s in range(4)], dtype=np.int64)
    mapped = (cards // 4) * 4 + perm[cards % 4]
    idx = np.array(
        [combo_index(int(a), int(b)) for a, b in mapped], dtype=np.int64
    )
    mean, std = _hs_moments(canon, n_rollouts, seed)
    if np.any(np.isnan(mean[idx])):
        raise AbstractionError(
            "n_rollouts trop faible : un combo demandé n'apparaît dans aucun runout "
            f"échantillonné — n_rollouts ≥ {_MIN_ROLLOUTS_FLOP} garantit la "
            "couverture de tout combo au flop (≥ 3 au turn)."
        )
    return mean[idx].copy(), std[idx].copy()


# ═══════════════════════════════════════════════════════════════════════════
# 1. FORCE ESPÉRÉE DE LA MAIN — E[HS] (Billings et al., 2002)
# ═══════════════════════════════════════════════════════════════════════════


def expected_hand_strength(
    combo_cards: npt.ArrayLike,
    board: Sequence[int],
    n_rollouts: int | None = None,
    seed: int = 0,
) -> F64:
    """Force espérée EHS de chaque combo : E sur les runouts du percentile de force.

    Pour chaque runout complétant le board, la main finale du combo est
    classée parmi les 990 combos adverses possibles sur ce board de 5 cartes
    — cartes du héros retirées (card removal exact), égalités pour moitié
    (Billings et al., 2002) : c'est P(gagner l'abattage contre une main
    aléatoire), et EHS en est la moyenne sur les runouts. Exact par
    énumération complète des runouts par défaut (1 176 au flop, 48 au turn) ;
    ``n_rollouts`` sous-échantillonne sans remise avec la graine ``seed``
    (reproductible, tiré dans l'espace du board canonique : deux boards
    isomorphes rendent des valeurs strictement identiques).

    Parameters
    ----------
    combo_cards : array_like
        Tableau (n, 2) d'entiers — cartes ``rang*4 + couleur`` de chaque
        combo. Cartes distinctes, sans collision avec le board.
    board : Sequence[int]
        3 cartes (flop) ou 4 (turn), distinctes.
    n_rollouts : int | None, optional
        ``None`` (défaut) = énumération exacte de tous les runouts ; sinon
        nombre de runouts tirés sans remise (≥ 96 au flop, ≥ 3 au turn pour
        garantir au moins un runout valide par combo).
    seed : int, optional
        Graine du sous-échantillonnage (défaut 0). Sans effet en mode exact.

    Returns
    -------
    ndarray
        (n,) float64 dans [0, 1] — EHS de chaque combo, dans l'ordre d'entrée.

    Raises
    ------
    AbstractionError
        Board invalide (taille ≠ 3/4, doublon, hors domaine), combos invalides
        (forme, non-entiers, carte dupliquée, collision board), ``n_rollouts``
        ou ``seed`` invalide, ou sous-échantillon trop court pour couvrir un
        combo demandé.

    Notes
    -----
    La table (1 326 combos × runouts) est mémoïsée par classe canonique de
    board (Waugh, 2013) : le premier appel sur un flop coûte ≈ 5 s, tous les
    appels suivants — y compris sur les 23 autres images du flop par
    permutation des couleurs — sont des lectures de table.

    Examples
    --------
    >>> from pfs.core.range_model import RANKS, SUITS
    >>> def c(t):
    ...     return RANKS.index(t[0]) * 4 + SUITS.index(t[1])
    >>> # Carré d'as au turn : nuts absolu sur les 46 rivières → EHS = 1.
    >>> b = [c('As'), c('Ah'), c('Ad'), c('Kc')]
    >>> float(expected_hand_strength([[c('Ac'), c('2h')]], b)[0])
    1.0
    """
    b = _validate_board(board)
    cards = _validate_combos(combo_cards, b)
    n_r = _validate_rollouts(n_rollouts)
    mean, _ = _moments_for(cards, b, n_r, _validate_seed(seed))
    return mean


# ═══════════════════════════════════════════════════════════════════════════
# 2. AFFECTATION EN BUCKETS (Gilpin & Sandholm 2006/2007 ; Lloyd 1982)
# ═══════════════════════════════════════════════════════════════════════════


def _percentile_buckets(strengths: F64, n_buckets: int, weights: F64) -> I64:
    """Quantiles égaux en MASSE : bucket k = ⌊K·(masse sous soi + moitié de soi)/W⌋."""
    order = np.argsort(strengths, kind="stable")
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    mid = cum - 0.5 * w_sorted
    labels_sorted = np.minimum(
        (n_buckets * mid / total).astype(np.int64), n_buckets - 1
    )
    out = np.empty(strengths.size, dtype=np.int64)
    out[order] = labels_sorted
    return out


def _kmeans_buckets(strengths: F64, n_buckets: int, weights: F64) -> I64:
    """k-means 1-D de Lloyd (1982), déterministe : init aux quantiles de masse.

    En dimension 1 avec centroïdes triés, les régions de Voronoï sont des
    intervalles bornés par les milieux de centroïdes consécutifs :
    l'affectation est une recherche dichotomique et chaque nouveau centroïde
    (moyenne pondérée de son intervalle) reste dans son intervalle, donc
    l'ordre des centroïdes est préservé — bucket 0 = le plus faible. Un
    cluster vidé conserve son centroïde (règle déterministe) et peut rester
    vide. Convergence : l'inertie décroît à chaque étape (Lloyd, 1982) ;
    arrêt au point fixe des affectations ou après ``_KMEANS_MAX_ITER``.
    """
    order = np.argsort(strengths, kind="stable")
    x = strengths[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    mid = cum - 0.5 * w_sorted
    targets = (np.arange(n_buckets, dtype=np.float64) + 0.5) * total / n_buckets
    init_idx = np.minimum(np.searchsorted(mid, targets), x.size - 1)
    centroids = x[init_idx].astype(np.float64)

    labels = np.full(x.size, -1, dtype=np.int64)
    for _ in range(_KMEANS_MAX_ITER):
        bounds = 0.5 * (centroids[1:] + centroids[:-1])
        new_labels = np.searchsorted(bounds, x, side="right").astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        mass = np.bincount(labels, weights=w_sorted, minlength=n_buckets)
        sums = np.bincount(labels, weights=w_sorted * x, minlength=n_buckets)
        filled = mass > 0
        centroids = np.where(filled, sums / np.maximum(mass, 1e-300), centroids)
        centroids = np.sort(centroids)  # défensif — déjà trié par construction

    out = np.empty(strengths.size, dtype=np.int64)
    out[order] = labels
    return out


def bucket_assignments(
    strengths: npt.ArrayLike,
    n_buckets: int = 8,
    method: str = "percentile",
    weights: npt.ArrayLike | None = None,
) -> I64:
    """Affecte chaque force à un bucket ∈ {0, …, K−1}, 0 = le plus faible.

    Deux découpages classiques de l'abstraction de cartes :

    - ``"percentile"`` — quantiles égaux en MASSE (Gilpin & Sandholm, 2006) :
      chaque bucket reçoit 1/K de la masse totale (les poids de range
      optionnels pondèrent la masse). L'affectation est ⌊K·(masse strictement
      sous l'élément + moitié de sa propre masse)/W⌋ : masses équilibrées à
      un élément près, ordre des buckets = ordre des forces.
    - ``"kmeans"`` — k-means 1-D de Lloyd (1982), déterministe
      (initialisation aux quantiles de masse, aucun tirage aléatoire) :
      minimise l'inertie intra-bucket, colle aux modes de la distribution.
      Contrairement à ``"percentile"``, des buckets peuvent rester vides.

    Parameters
    ----------
    strengths : array_like
        (n,) valeurs finies — typiquement les EHS de ``expected_hand_strength``.
    n_buckets : int, optional
        Nombre de buckets K ≥ 1 (défaut 8).
    method : str, optional
        ``"percentile"`` (défaut) ou ``"kmeans"``.
    weights : array_like | None, optional
        (n,) poids ≥ 0 de somme > 0 (poids de range) ; ``None`` = uniformes.

    Returns
    -------
    ndarray
        (n,) int64 — bucket de chaque élément, dans l'ordre d'entrée,
        croissant avec la force.

    Raises
    ------
    AbstractionError
        ``strengths`` non 1-D, vide ou non fini ; ``n_buckets`` non entier ou
        < 1 ; méthode inconnue ; poids invalides.

    Notes
    -----
    Égalités : le tri est stable, l'affectation déterministe ; en mode
    ``"percentile"``, des forces strictement égales peuvent chevaucher une
    frontière — l'équilibre des masses prime (à un élément près).

    Examples
    --------
    >>> import numpy as np
    >>> bucket_assignments(np.array([0.9, 0.1, 0.5, 0.3]), n_buckets=2)
    array([1, 0, 1, 0])
    >>> bucket_assignments(np.array([0.9, 0.1, 0.5, 0.3]), n_buckets=2,
    ...                    method="kmeans")
    array([1, 0, 0, 0])
    """
    arr = np.asarray(strengths)
    if arr.ndim != 1 or arr.size == 0:
        raise AbstractionError("strengths doit être un tableau 1-D non vide.")
    s = arr.astype(np.float64)
    if not np.all(np.isfinite(s)):
        raise AbstractionError("strengths contient des valeurs non finies.")
    if isinstance(n_buckets, bool) or not isinstance(n_buckets, (int, np.integer)):
        raise AbstractionError("n_buckets doit être un entier ≥ 1.")
    if n_buckets < 1:
        raise AbstractionError(f"n_buckets = {n_buckets} : il faut au moins 1 bucket.")
    if method not in _METHODS:
        raise AbstractionError(
            f"méthode inconnue : {method!r} — attendu {' ou '.join(map(repr, _METHODS))}."
        )
    if weights is None:
        w = np.ones(s.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != s.size:
            raise AbstractionError(
                f"weights de taille {w.size} pour {s.size} forces."
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise AbstractionError("weights doit être fini et ≥ 0.")
        if w.sum() <= 0.0:
            raise AbstractionError("weights de somme nulle : masse indéfinie.")

    if method == "percentile":
        return _percentile_buckets(s, int(n_buckets), w)
    return _kmeans_buckets(s, int(n_buckets), w)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ABSTRACTION D'UNE RANGE COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BucketSummary:
    """Résumé d'un bucket : masse, force moyenne, combos représentatifs.

    Attributes
    ----------
    weight : float
        Poids total de range dans le bucket (somme des poids des combos).
    mean_ehs : float
        EHS moyen du bucket, pondéré par les poids de range.
    combos : tuple[str, ...]
        Combos représentatifs (poids décroissant, puis EHS décroissant),
        au format ``"AsKh"``.
    """

    weight: float
    mean_ehs: float
    combos: tuple[str, ...]


def abstract_range(
    range_: Range,
    board: Sequence[int],
    n_buckets: int = 8,
    method: str = "percentile",
    n_rollouts: int | None = None,
    seed: int = 0,
    max_representatives: int = 5,
) -> dict[int, BucketSummary]:
    """Compresse une range en K buckets de force sur un board donné.

    Pipeline complet de l'abstraction de cartes : retrait des combos bloqués
    par le board, EHS de chaque combo survivant, affectation en buckets
    (masses pondérées par les poids de la range), puis résumé par bucket —
    exactement la représentation qu'un blueprint CFR consomme à la place des
    1 326 combos (Gilpin & Sandholm, 2007 ; Johanson et al., 2013).

    Parameters
    ----------
    range_ : Range
        Range adverse ou héros (poids sur les 1 326 combos).
    board : Sequence[int]
        3 cartes (flop) ou 4 (turn), distinctes.
    n_buckets : int, optional
        Nombre de buckets K ≥ 1 (défaut 8).
    method : str, optional
        ``"percentile"`` (défaut) ou ``"kmeans"`` — cf. ``bucket_assignments``.
    n_rollouts : int | None, optional
        Sous-échantillonnage des runouts — cf. ``expected_hand_strength``.
    seed : int, optional
        Graine du sous-échantillonnage (défaut 0).
    max_representatives : int, optional
        Nombre maximal de combos représentatifs par bucket (défaut 5).

    Returns
    -------
    dict[int, BucketSummary]
        Bucket → (poids total, EHS moyen, combos représentatifs), clés
        croissantes, buckets vides omis. La somme des poids vaut la masse de
        la range après retrait des blockers.

    Raises
    ------
    AbstractionError
        ``range_`` n'est pas une ``Range``, board invalide, paramètres de
        bucketing invalides, ou range vide une fois les blockers retirés.

    Examples
    --------
    >>> from pfs.core.range_model import RANKS, SUITS, parse_range
    >>> def c(t):
    ...     return RANKS.index(t[0]) * 4 + SUITS.index(t[1])
    >>> b = [c('As'), c('Ah'), c('Ad'), c('Kc')]
    >>> buckets = abstract_range(parse_range("KK, QQ"), b, n_buckets=2)
    >>> sorted(buckets)  # KK (full aux rois) domine QQ : deux buckets
    [0, 1]
    >>> buckets[1].combos[0][0], buckets[1].mean_ehs > buckets[0].mean_ehs
    ('K', True)
    """
    if not isinstance(range_, Range):
        raise AbstractionError("range_ doit être une pfs.core.range_model.Range.")
    b = _validate_board(board)
    if isinstance(max_representatives, bool) or not isinstance(
        max_representatives, (int, np.integer)
    ) or max_representatives < 1:
        raise AbstractionError("max_representatives doit être un entier ≥ 1.")
    live = range_.remove_blockers(b)
    idx = np.nonzero(live.weights > 0.0)[0]
    if idx.size == 0:
        raise AbstractionError(
            "range vide sur ce board : tous les combos sont à poids nul ou bloqués."
        )
    w = live.weights[idx]
    cards = _combo_pairs()[idx]
    n_r = _validate_rollouts(n_rollouts)
    ehs, _ = _moments_for(cards, b, n_r, _validate_seed(seed))
    labels = bucket_assignments(ehs, n_buckets=n_buckets, method=method, weights=w)

    out: dict[int, BucketSummary] = {}
    for k in range(int(n_buckets)):
        sel = labels == k
        if not sel.any():
            continue
        w_k, e_k, c_k = w[sel], ehs[sel], cards[sel]
        # représentants : poids décroissant, puis EHS décroissant, puis index combo
        order = np.lexsort((idx[sel], -e_k, -w_k))[: int(max_representatives)]
        reps = tuple(
            f"{card_str(int(a))}{card_str(int(b_))}" for a, b_ in c_k[order]
        )
        total = float(w_k.sum())
        out[k] = BucketSummary(
            weight=total,
            mean_ehs=float((w_k * e_k).sum() / total),
            combos=reps,
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. CARACTÉRISTIQUES DISTRIBUTION-AWARE (Johanson et al., 2013)
# ═══════════════════════════════════════════════════════════════════════════


def distribution_aware_features(
    combo_cards: npt.ArrayLike,
    board: Sequence[int],
    n_rollouts: int | None = None,
    seed: int = 0,
) -> F64:
    r"""Couple (EHS, écart-type de HS sur les runouts) — les deux premiers moments.

    Pourquoi deux moments ? E[HS] seul est AVEUGLE à la nature de la main :
    une paire faite et un gros tirage peuvent avoir exactement le même EHS
    alors que leurs distributions de force sont opposées — la main faite a un
    HS stable sur presque tous les runouts (σ faible), le tirage est bimodal
    (proche du nuts quand il rentre, air sinon → σ élevé). Or leurs
    stratégies d'équilibre diffèrent radicalement (value/protection contre
    semi-bluff) : les fusionner dans un même bucket coûte de l'exploitabilité.
    Johanson, Burch, Valenzano & Bowling (2013) montrent que les abstractions
    fondées sur la distribution de la force dominent celles fondées sur E[HS]
    seul ; le couple (E[HS], σ[HS]) en est la forme à deux moments, en
    bijection avec le classique (E[HS], E[HS²]) puisque
    :math:`\sigma^2 = E[HS^2] - E[HS]^2`. Un k-means sur ces deux colonnes
    (normalisées) sépare ce que le bucketing 1-D sur EHS confond.

    Parameters
    ----------
    combo_cards : array_like
        Tableau (n, 2) d'entiers — cf. ``expected_hand_strength``.
    board : Sequence[int]
        3 cartes (flop) ou 4 (turn), distinctes.
    n_rollouts : int | None, optional
        Sous-échantillonnage des runouts — cf. ``expected_hand_strength``.
    seed : int, optional
        Graine du sous-échantillonnage (défaut 0).

    Returns
    -------
    ndarray
        (n, 2) float64 : colonne 0 = EHS (identique à
        ``expected_hand_strength``), colonne 1 = écart-type de HS sur les
        runouts valides du combo (population, ddof = 0).

    Raises
    ------
    AbstractionError
        Mêmes conditions que ``expected_hand_strength``.

    Examples
    --------
    >>> from pfs.core.range_model import RANKS, SUITS
    >>> def c(t):
    ...     return RANKS.index(t[0]) * 4 + SUITS.index(t[1])
    >>> b = [c('As'), c('Ah'), c('Ad'), c('Kc')]
    >>> f = distribution_aware_features([[c('Ac'), c('2h')]], b)
    >>> f.shape, float(f[0, 0]), float(f[0, 1])  # nuts : EHS 1, variance nulle
    ((1, 2), 1.0, 0.0)
    """
    b = _validate_board(board)
    cards = _validate_combos(combo_cards, b)
    n_r = _validate_rollouts(n_rollouts)
    mean, std = _moments_for(cards, b, n_r, _validate_seed(seed))
    return np.column_stack([mean, std])
