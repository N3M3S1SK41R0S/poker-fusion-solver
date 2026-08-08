r"""Effet de bunching — repondération des ranges restantes après les folds.

Quand N joueurs ont foldé devant, les cartes muckées ne sont pas un
échantillon uniforme du paquet : une range de fold regorge de petites
cartes (les grosses mains sont jouées). Le paquet « résiduel » vu par les
joueurs restants est donc déformé — dans une range restante, les combos de
petites cartes deviennent relativement MOINS probables (leurs cartes sont
plus souvent parties dans les mucks) et les combos premium PLUS probables.
Cet effet, dit de *bunching*, n'est modélisé que par les solveurs multiway
les plus récents (GTO Wizard AI) ; ce module l'apporte au noyau.

Modèle mathématique
-------------------
Chaque folder :math:`f` tenait un combo :math:`c_f` tiré de sa range de
FOLD — le complémentaire pondéré de sa range de jeu, cf.
:func:`fold_range_from_play`. Les poids de fold sont lus comme
:math:`P(\text{fold} \mid \text{combo})` et les décisions de fold sont
supposées indépendantes entre folders (premier ordre de l'arbre préflop :
pas de dépendance aux sizings ni à la dynamique). Sachant les folds et les
cartes mortes, la loi jointe des mains foldées est

.. math::

   \pi(c_1,\dots,c_N) \;\propto\; \prod_f w_f(c_f)\,
   \mathbf 1[\text{combos deux à deux disjoints, hors cartes mortes}].

Pour un combo cible :math:`(a,b)` d'un joueur restant, le poids devient

.. math::

   w'(a,b) \;\propto\; w(a,b)\cdot
   P_\pi(\text{aucun folder ne tenait } a \text{ ni } b).

Ce facteur est *exactement* proportionnel à la vraisemblance bayésienne
:math:`P(\text{folds} \mid \text{cible}=(a,b))` : à nombre de folders fixé,
toute affectation disjointe de mains a la même probabilité de deal, que
l'on conditionne ou non sur la main cible — les deux quantités ne diffèrent
que par une constante indépendante de :math:`(a,b)`, absorbée par la
normalisation.

Le calcul exact de :math:`P_\pi` est une somme sur les affectations
disjointes (objet de type « permanent »), combinatoirement prohibitive dès
:math:`N \ge 2` folders sur 1326 combos. D'où deux estimateurs :

1. :func:`bunching_weights_mc` — Monte-Carlo séquentiel à pondération
   d'importance auto-normalisée (Kahn & Marshall, 1953) : converge vers
   :math:`\pi`, sans biais d'ordre des folders ;
2. :func:`bunching_weights_pairwise` — produit analytique au premier
   ordre : inclusion-exclusion exacte PAR folder, mais corrélations ENTRE
   folders négligées. Déterministe et quasi instantané.

Conventions communes
--------------------
- Les cartes mortes (board, main héros) sont retirées des ranges de fold
  avant normalisation : un folder ne peut pas les tenir. Un combo cible qui
  CONTIENT une carte morte garde ici un multiplicateur > 0 — le facteur ne
  mesure que la disponibilité côté folders ; c'est :func:`apply_bunching`
  qui annule ces combos via ``Range.remove_blockers``.
- Les multiplicateurs retournés sont NORMALISÉS AU MAXIMUM (max = 1) : le
  modèle ne définit :math:`w'` qu'à proportionnalité près, et la convention
  ``Range`` exige des poids par combo dans [0, 1] — même choix de
  normalisation que ``Range.bayes_update``.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import (
    N_CARDS,
    N_COMBOS,
    Range,
    RangeError,
    card_str,
    combo_cards,
)

__all__ = [
    "BunchingError",
    "apply_bunching",
    "bunching_weights_mc",
    "bunching_weights_pairwise",
    "fold_range_from_play",
]

F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]

_CHUNK_SIMS: int = 2048
"""Taille de lot du Monte-Carlo : deux tampons (2048 × 1326) float64 ≈ 22 Mo."""

_COMBO_CARDS: I32 = np.array(
    [combo_cards(i) for i in range(N_COMBOS)], dtype=np.int32
)
"""(1326, 2) — les deux cartes (haute, basse) de chaque combo canonique."""

_CA: I32 = _COMBO_CARDS[:, 0]
_CB: I32 = _COMBO_CARDS[:, 1]


class BunchingError(RangeError):
    """Entrées invalides ou modèle de folds incohérent (aucun scénario possible)."""


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def _checked_dead(dead_cards: Sequence[int]) -> list[int]:
    """Valide les cartes mortes : entiers de [0, 52), sans doublon.

    Parameters
    ----------
    dead_cards
        Indices de cartes (convention ``rang*4 + couleur``).

    Returns
    -------
    list[int]
        Les cartes validées, dans l'ordre d'entrée.

    Raises
    ------
    BunchingError
        Carte non entière, hors domaine, ou dupliquée.
    """
    out: list[int] = []
    seen: set[int] = set()
    for c in dead_cards:
        try:
            ci = operator.index(c)
        except TypeError as exc:
            raise BunchingError(f"carte morte non entière : {c!r}.") from exc
        if not 0 <= ci < N_CARDS:
            raise BunchingError(f"carte morte hors domaine [0, 52) : {ci}.")
        if ci in seen:
            raise BunchingError(f"carte morte dupliquée : {card_str(ci)}.")
        seen.add(ci)
        out.append(ci)
    return out


def _checked_range(r: Range, name: str) -> Range:
    """Exige une instance de ``Range`` (message d'erreur nommé)."""
    if not isinstance(r, Range):
        raise BunchingError(f"{name} doit être une Range, reçu {type(r).__name__}.")
    return r


def _checked_int(value: int, name: str, minimum: int) -> int:
    """Exige un entier ``>= minimum`` (message d'erreur nommé)."""
    try:
        v = operator.index(value)
    except TypeError as exc:
        raise BunchingError(f"{name} doit être un entier, reçu {value!r}.") from exc
    if v < minimum:
        raise BunchingError(f"{name} doit être >= {minimum}, reçu {v}.")
    return v


def _fold_distributions(folder_ranges: Sequence[Range], dead: list[int]) -> list[F64]:
    """Distributions de fold normalisées (somme 1), cartes mortes retirées.

    Parameters
    ----------
    folder_ranges
        Ranges de FOLD des joueurs ayant passé (pas leurs ranges de jeu !).
    dead
        Cartes mortes déjà validées par :func:`_checked_dead`.

    Returns
    -------
    list[numpy.ndarray]
        Une distribution (1326,) par folder, de somme 1.

    Raises
    ------
    BunchingError
        ``folder_ranges`` n'est pas une séquence de ``Range``, ou l'une des
        ranges de fold est vide après retrait des cartes mortes (le folder
        ne peut alors tenir aucun combo : modèle incohérent).
    """
    if isinstance(folder_ranges, Range):
        raise BunchingError(
            "folder_ranges doit être une séquence de Range, pas une Range seule."
        )
    out: list[F64] = []
    for k, fr in enumerate(folder_ranges):
        _checked_range(fr, f"folder_ranges[{k}]")
        w = fr.remove_blockers(dead).weights
        s = float(w.sum())
        if s <= 0.0:
            raise BunchingError(
                f"folder_ranges[{k}] : range de fold vide après retrait des "
                "cartes mortes — le folder ne peut tenir aucun combo."
            )
        out.append(w / s)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# RANGE DE FOLD
# ═══════════════════════════════════════════════════════════════════════════


def fold_range_from_play(play_range: Range) -> Range:
    r"""Range de fold complémentaire : :math:`w_{fold}(c) = 1 - w_{jeu}(c)`.

    Un joueur qui joue un combo avec probabilité :math:`w` le folde avec
    probabilité :math:`1 - w` : une main jouée à 50 % (fréquence mixte des
    presets GTO) pèse 0.5 dans les deux ranges. Le fold étant l'action par
    défaut, la range de fold d'un joueur serré regorge mécaniquement de
    petites cartes — c'est la source de l'effet de bunching.

    Parameters
    ----------
    play_range
        Range de JEU du joueur (open, call, …), poids par combo dans [0, 1].

    Returns
    -------
    Range
        La range de fold pondérée, ``1 − w`` combo par combo.

    Raises
    ------
    BunchingError
        ``play_range`` n'est pas une ``Range``.

    Examples
    --------
    >>> from pfs.core.range_model import parse_range
    >>> fold = fold_range_from_play(parse_range("AA"))
    >>> round(fold.fraction, 4)   # 1320 combos effectifs sur 1326
    0.9955
    """
    _checked_range(play_range, "play_range")
    return Range(1.0 - play_range.weights)


# ═══════════════════════════════════════════════════════════════════════════
# ESTIMATEUR 1 — MONTE-CARLO SÉQUENTIEL À PONDÉRATION D'IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════


def bunching_weights_mc(
    target_range: Range,
    folder_ranges: Sequence[Range],
    dead_cards: Sequence[int],
    n_sims: int = 20_000,
    seed: int = 0,
) -> F64:
    r"""Multiplicateurs de bunching par Monte-Carlo (fréquences JOINTES).

    Procédure, par simulation :

    1. les folders tirent séquentiellement un combo chacun dans leur
       distribution de fold, sans collision entre eux ni avec les cartes
       mortes (la distribution est renormalisée à chaque étape sur les
       combos encore disponibles, de masse restante :math:`z_f`) ;
    2. la simulation reçoit le poids d'importance :math:`u = \prod_f z_f` —
       la proposition séquentielle dépend de l'ordre des folders, mais ce
       poids la redresse exactement vers la loi jointe :math:`\pi \propto
       \prod_f q_f\,\mathbf 1[\text{disjoints}]`, qui n'en dépend pas
       (échantillonnage préférentiel auto-normalisé, Kahn & Marshall 1953) ;
    3. les cartes prises par les folders sont marquées indisponibles.

    Le facteur d'un combo cible :math:`(a,b)` est la fréquence pondérée
    JOINTE :math:`\widehat P(a \text{ libre ET } b \text{ libre})` —
    comptage des paires de cartes simultanément libres, PAS le produit des
    marginales : « a libre » et « b libre » sont corrélés (chaque folder
    prend deux cartes d'un coup, et les folds se concentrent sur les mêmes
    zones du paquet), l'hypothèse d'indépendance est fausse. La marginale
    :math:`\widehat P(c \text{ indisponible})` est disponible en diagonale
    du même comptage (``1 − p_free[c, c]``) mais ne sert pas au facteur.

    Parameters
    ----------
    target_range
        Range du joueur restant. Validée mais sans effet sur les
        multiplicateurs : ils ne dépendent que des folders et des cartes
        mortes (l'argument est conservé pour la symétrie d'API avec
        :func:`apply_bunching`, qui applique le produit).
    folder_ranges
        Ranges de FOLD des joueurs ayant passé — typiquement produites par
        :func:`fold_range_from_play`. Liste vide = aucun bunching.
    dead_cards
        Cartes connues indisponibles pour les folders (board, main héros).
    n_sims
        Nombre de simulations, ``>= 1``. Erreur type d'une fréquence
        :math:`p` : :math:`\sqrt{p(1-p)/n_{\text{eff}}}` — avec 20 000
        tirages, ~0.3 % au pire.
    seed
        Graine du générateur (``>= 0``) — résultat entièrement déterministe
        à arguments égaux.

    Returns
    -------
    numpy.ndarray
        (1326,) float64 — multiplicateurs NORMALISÉS AU MAX (max = 1) ;
        seuls les rapports entre combos portent l'information, cf. la
        convention du module.

    Raises
    ------
    BunchingError
        Entrées invalides ; ou aucune simulation cohérente (poids total
        nul : les ranges de fold se bloquent mutuellement — p. ex. deux
        folders dont les folds exigent les mêmes cartes).

    Notes
    -----
    Une simulation où un folder n'a plus aucun combo disponible est une
    impasse de la proposition séquentielle : son poids est mis à 0 (elle ne
    compte ni au numérateur ni au dénominateur). Coût :
    ``O(n_sims × n_folders × 1326)`` en temps, ``O(_CHUNK_SIMS × 1326)``
    en mémoire (traitement par lots).
    """
    dead = _checked_dead(dead_cards)
    _checked_range(target_range, "target_range")
    folders = _fold_distributions(folder_ranges, dead)
    n_sims = _checked_int(n_sims, "n_sims", minimum=1)
    seed = _checked_int(seed, "seed", minimum=0)
    if not folders:
        return np.ones(N_COMBOS, dtype=np.float64)

    rng = np.random.default_rng(seed)
    joint = np.zeros((N_CARDS, N_CARDS), dtype=np.float64)
    total_weight = 0.0

    remaining = n_sims
    while remaining > 0:
        m = min(_CHUNK_SIMS, remaining)
        remaining -= m
        # Disponibilité vue des folders : les cartes mortes restent True
        # (elles ne peuvent simplement pas être tirées — retirées des q).
        avail = np.ones((m, N_CARDS), dtype=bool)
        weight = np.ones(m, dtype=np.float64)
        alive = np.ones(m, dtype=bool)
        for q in folders:
            allowed = avail[:, _CA] & avail[:, _CB]            # (m, 1326)
            cum = np.cumsum(allowed * q[None, :], axis=1)      # (m, 1326)
            z = cum[:, -1]                                     # masse restante
            alive &= z > 0.0
            weight = np.where(alive, weight * z, 0.0)
            u = rng.random(m) * z                              # u ∈ [0, z)
            idx = np.argmax(cum > u[:, None], axis=1)          # CDF inverse
            rows = np.flatnonzero(alive)
            avail[rows, _CA[idx[rows]]] = False
            avail[rows, _CB[idx[rows]]] = False
        free = avail.astype(np.float64)
        joint += (free * weight[:, None]).T @ free             # (52, 52)
        total_weight += float(weight.sum())

    if total_weight <= 0.0:
        raise BunchingError(
            "aucun scénario de folds cohérent : les ranges de fold se "
            "bloquent mutuellement (poids Monte-Carlo total nul)."
        )
    p_free = joint / total_weight        # P(a libre ET b libre | folds)
    factors = p_free[_CA, _CB]
    top = float(factors.max())
    if top <= 0.0:
        raise BunchingError("tous les combos cibles sont bloqués par les folds simulés.")
    return factors / top


# ═══════════════════════════════════════════════════════════════════════════
# ESTIMATEUR 2 — PRODUIT ANALYTIQUE AU PREMIER ORDRE
# ═══════════════════════════════════════════════════════════════════════════


def bunching_weights_pairwise(
    target_range: Range,
    folder_ranges: Sequence[Range],
    dead_cards: Sequence[int],
) -> F64:
    r"""Multiplicateurs de bunching analytiques (premier ordre, déterministe).

    Pour chaque folder :math:`f`, de distribution de fold normalisée
    :math:`q_f` (cartes mortes retirées), on note :math:`m_f(c)` la masse
    des combos contenant la carte :math:`c` et :math:`m_f(a,b) = q_f(a,b)`
    la masse du combo exact. Par inclusion-exclusion — le seul combo
    contenant à la fois :math:`a` et :math:`b` étant :math:`(a,b)` — la
    probabilité que :math:`f` ne tienne ni :math:`a` ni :math:`b` vaut
    EXACTEMENT

    .. math::

       1 - m_f(a) - m_f(b) + m_f(a,b) \in [0, 1],

    et le facteur retenu est le produit sur les folders :

    .. math::

       \phi(a,b) = \prod_f \bigl(1 - m_f(a) - m_f(b) + m_f(a,b)\bigr).

    L'approximation est dans le PRODUIT : chaque folder est évalué sous sa
    distribution marginale, comme si les mains des folders étaient tirées
    indépendamment — les corrélations ENTRE folders (leurs combos sont en
    réalité disjoints : chaque main foldée retire deux cartes aux
    distributions des suivants) sont négligées. L'erreur relative par
    folder est de l'ordre de la masse par carte (~4/52 ≈ 8 % au pire,
    bien moins sur des ranges de fold larges) au CARRÉ des recouvrements ;
    la validation croisée avec :func:`bunching_weights_mc` (corrélation
    > 0.95, cf. tests) borne l'écart en pratique.

    Parameters
    ----------
    target_range
        Range du joueur restant — validée, sans effet sur les
        multiplicateurs (cf. :func:`bunching_weights_mc`).
    folder_ranges
        Ranges de FOLD des joueurs ayant passé. Liste vide = multiplicateurs
        unité.
    dead_cards
        Cartes connues indisponibles pour les folders.

    Returns
    -------
    numpy.ndarray
        (1326,) float64 — multiplicateurs NORMALISÉS AU MAX (max = 1).

    Raises
    ------
    BunchingError
        Entrées invalides, ou tous les combos de facteur nul (ranges de
        fold dégénérées bloquant tout le paquet).

    Notes
    -----
    Contrairement au Monte-Carlo, cette approximation ne détecte PAS les
    incohérences jointes (deux folders exigeant les mêmes cartes) : chaque
    folder étant traité isolément, elle rend simplement les combos
    concernés impossibles. Coût : ``O(n_folders × 1326)``.
    """
    dead = _checked_dead(dead_cards)
    _checked_range(target_range, "target_range")
    folders = _fold_distributions(folder_ranges, dead)
    if not folders:
        return np.ones(N_COMBOS, dtype=np.float64)

    factors = np.ones(N_COMBOS, dtype=np.float64)
    for q in folders:
        m_card = np.bincount(_CA, weights=q, minlength=N_CARDS) + np.bincount(
            _CB, weights=q, minlength=N_CARDS
        )
        per_folder = 1.0 - m_card[_CA] - m_card[_CB] + q
        # Théoriquement dans [0, 1] ; le clip absorbe la poussière flottante.
        factors *= np.clip(per_folder, 0.0, 1.0)

    top = float(factors.max())
    if top <= 0.0:
        raise BunchingError(
            "tous les combos cibles sont bloqués par le modèle de fold."
        )
    return factors / top


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION À UNE RANGE
# ═══════════════════════════════════════════════════════════════════════════


def apply_bunching(
    target_range: Range,
    folder_ranges: Sequence[Range],
    dead_cards: Sequence[int],
    method: str = "pairwise",
    n_sims: int = 20_000,
    seed: int = 0,
) -> Range:
    r"""Applique l'effet de bunching à la range d'un joueur restant.

    Repondération :math:`w'(c) = w(c) \cdot \phi(c)` où :math:`\phi` sont
    les multiplicateurs (max = 1) de la méthode choisie, suivie du retrait
    des combos utilisant une carte morte (``Range.remove_blockers``). La
    renormalisation est DOUCE : pas de renormalisation à somme fixe — les
    poids d'une ``Range`` sont des probabilités par combo dans [0, 1], pas
    une distribution ; on multiplie simplement et on clippe. Les rapports
    entre combos (seule information définie par le modèle) sont préservés.

    Parameters
    ----------
    target_range
        Range du joueur restant à repondérer.
    folder_ranges
        Ranges de FOLD des joueurs ayant passé — typiquement
        ``[fold_range_from_play(r) for r in ranges_de_jeu]``.
    dead_cards
        Cartes connues (board, main héros) : retirées des ranges de fold ET
        de la range résultat.
    method
        ``"pairwise"`` (défaut — analytique, déterministe, quasi instantané)
        ou ``"mc"`` (Monte-Carlo joint, plus fidèle en multiway serré).
    n_sims, seed
        Transmis à :func:`bunching_weights_mc` ; ignorés en ``"pairwise"``.

    Returns
    -------
    Range
        La range repondérée. Sans folder ni carte morte, copie à
        l'identique de ``target_range``.

    Raises
    ------
    BunchingError
        Méthode inconnue, ou toute erreur de validation des estimateurs.

    Examples
    --------
    >>> from pfs.core.range_model import GTO_PRESETS, Range, parse_range
    >>> folds = [fold_range_from_play(parse_range(GTO_PRESETS[p]))
    ...          for p in ("UTG", "MP", "CO")]
    >>> bb = apply_bunching(Range.full(), folds, [], method="pairwise")
    >>> bb.n_combos < Range.full().n_combos   # masse rognée, jamais accrue
    True
    """
    if method == "pairwise":
        factors = bunching_weights_pairwise(target_range, folder_ranges, dead_cards)
    elif method == "mc":
        factors = bunching_weights_mc(
            target_range, folder_ranges, dead_cards, n_sims=n_sims, seed=seed
        )
    else:
        raise BunchingError(
            f"méthode inconnue : {method!r} (attendu 'pairwise' ou 'mc')."
        )
    reweighted = Range(np.clip(target_range.weights * factors, 0.0, 1.0))
    return reweighted.remove_blockers(_checked_dead(dead_cards))
