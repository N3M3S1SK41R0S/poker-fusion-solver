"""
F14 — ICM : Independent Chip Model, bubble factor, $EV.

Sources
-------
- Malmuth, M. & Harville, D. (1973/1987) — le modèle standard de l'industrie,
  utilisé par PioSOLVER, HRC, ICMIZER, MonkerSolver, GTO Wizard.
- Harville, D.A. (1973), *Assigning Probabilities to the Outcomes of
  Multi-Entry Competitions*, JASA 68(342) — la récurrence d'origine (courses).
- Ganzfried & Sandholm (2015) pour l'articulation avec la safe exploitation.

Le principe en une phrase : **en tournoi, les jetons n'ont pas une valeur
linéaire.** Doubler son stack ne double pas son espérance de gain, parce que
les paiements sont plafonnés par place. L'ICM convertit une distribution de
stacks en valeur monétaire ($EV), et le **bubble factor** quantifie combien un
call devient plus cher qu'en cash game.

Pourquoi c'était notre lacune n°1 du benchmark : 5 solveurs de référence
l'exposent (Pio, HRC, ICMIZER, Monker, GTO Wizard), et sans lui toute
recommandation en tournoi est fausse près des paliers. Ce module la comble et
alimente directement F13 : l'équité requise d'un call devient
``alpha_icm = 1 - 1/(1 + bubble_factor · (P+b)/b)`` au lieu du pot odds brut.

Complexité : la récurrence de Harville est en O(n!) naïf ; l'implémentation
mémoïse sur les sous-ensembles (O(2ⁿ·n²)), exacte jusqu'à ~12 joueurs — le cas
d'usage réel (table finale). Au-delà, estimation Monte-Carlo par tirages
proportionnels sans remise, avec erreur-type rapportée.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

__all__ = [
    "icm_equities",
    "icm_dollar_ev",
    "bubble_factor",
    "risk_premium",
    "icm_required_equity",
    "IcmSpot",
    "analyse_icm_spot",
    "IcmError",
    "PayoutsTronques",
    "IcmMesure",
    "icm_equities_mesurees",
    "erreur_type_exacte",
    "stats_cache_harville",
    "vider_cache_harville",
]

EXACT_LIMIT = 12
"""Au-delà de 12 joueurs, bascule automatique en Monte-Carlo."""

CACHE_HARVILLE = 4096
"""Entrées du cache PARTAGÉ de `_finish_probs_exact`, clé = tapis triés.

Une entrée pèse au plus 12 × 12 flottants (≈ 1,2 ko) : le cache saturé tient
sous 5 Mo. À comparer aux 2¹² × 12 états que chaque appel non caché
reconstruit puis jette.
"""


class IcmError(ValueError):
    pass


class PayoutsTronques(UserWarning):
    """Des places payées ont été retirées faute de joueurs pour les occuper.

    La troncature elle-même est CORRECTE : à deux joueurs, la 3ᵉ place n'existe
    plus, son gain a déjà été versé à quelqu'un d'éliminé, et la dotation
    attribuable est ``sum(payouts[:len(stacks)])``. C'est le SILENCE qui était
    le défaut : une table finale saisie à cinq joueurs au lieu de six perdait
    une place sans que rien ne le dise, et toutes les équités sortaient fausses
    d'un montant plausible.

    Cet avertissement n'est émis que si la troncature retire un gain
    STRICTEMENT positif — une grille complétée par des zéros ne perd rien.
    """


# ═══════════════════════════════════════════════════════════════════════════
# NOYAU MALMUTH-HARVILLE
# ═══════════════════════════════════════════════════════════════════════════


def _validate(stacks: Sequence[float], payouts: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    s = tuple(float(x) for x in stacks)
    p = tuple(float(x) for x in payouts)
    if not s:
        raise IcmError("au moins un stack requis.")
    if any(x < 0 for x in s):
        raise IcmError("stack négatif.")
    if sum(s) <= 0:
        raise IcmError("somme des stacks nulle.")
    if any(x < 0 for x in p):
        raise IcmError("payout négatif.")
    if len(p) > len(s):
        # POURQUOI avertir plutôt que tronquer en silence : la troncature est
        # la bonne opération (une place qui n'existe plus n'est versée à
        # personne), mais elle est INDISCERNABLE d'une saisie incomplète des
        # tapis. Les deux causes produisent la même entrée ; seule l'une des
        # deux est voulue. L'avertissement rend le choix visible sans casser
        # le cas légitime — c'est le seul point où un `raise` aurait refusé des
        # tables parfaitement valides (deux joueurs restants sur une grille à
        # trois places, mesuré dans `test_icm_invariants.py`).
        perdus = p[len(s):]
        if any(x > 0.0 for x in perdus):
            warnings.warn(
                f"grille de {len(p)} places pour {len(s)} joueur(s) : "
                f"{len(perdus)} place(s) tronquée(s) "
                f"({', '.join(f'{x:g}' for x in perdus)}). Dotation "
                f"attribuable {sum(p[: len(s)]):g} au lieu de {sum(p):g}. "
                f"C'est correct si ces places ont déjà été payées ; c'est une "
                f"SAISIE INCOMPLÈTE des tapis si la table compte encore "
                f"{len(p)} joueurs.",
                PayoutsTronques, stacklevel=3)
        p = p[: len(s)]
    # Payouts décroissants : convention universelle (1re place d'abord).
    if any(p[i] < p[i + 1] for i in range(len(p) - 1)):
        raise IcmError("payouts doivent être décroissants (1re place d'abord).")
    return s, p


@lru_cache(maxsize=CACHE_HARVILLE)
def _finish_probs_tries(stacks: tuple[float, ...], n_places: int) -> np.ndarray:
    """Cœur de Harville sur des tapis DÉJÀ TRIÉS — la seule entrée cachée.

    POURQUOI un cache de module et pas les `lru_cache` d'origine : celles-ci
    étaient posées sur des closures RECRÉÉES à chaque appel. Elles servaient
    la récursion interne (2ⁿ états visités une fois) et ne survivaient pas à
    la sortie de fonction — aucun bénéfice entre deux appels, alors que
    `bubble_factor` appelle l'ICM TROIS fois de suite et `analyse_icm_spot`
    lui redemande en plus la ligne de base sur les MÊMES tapis.

    La clé est le tapis trié : les probabilités de Harville sont équivariantes
    par permutation des joueurs, donc deux tables aux mêmes tapis dans un
    ordre différent partagent une entrée. `_finish_probs_exact` remet la
    permutation à l'endroit.

    Le tableau rendu est marqué NON MODIFIABLE : il est partagé entre appels,
    et une écriture en place corromprait silencieusement toutes les tables
    suivantes.
    """
    n = len(stacks)
    probs = np.zeros((n, n_places), dtype=np.float64)

    @lru_cache(maxsize=None)
    def sub_total(mask: int) -> float:
        return sum(stacks[i] for i in range(n) if mask & (1 << i))

    @lru_cache(maxsize=None)
    def place_probs(mask: int, place: int) -> tuple[float, ...]:
        """P(i termine à `place` | ensemble `mask` encore en course)."""
        total = sub_total(mask)
        if total <= 0.0:
            # Plus aucun jeton dans l'ensemble considéré : la récursion a
            # épuisé les joueurs, ou l'un d'eux a un tapis nul.
            #
            # Ce cas n'est pas théorique : `bubble_factor` évalue précisément
            # ce que vaut le tapis du héros APRÈS avoir tout perdu, donc avec
            # un tapis à zéro. La division par le total plantait alors, et
            # c'est tout le calcul de pression de bulle — le nombre qui dit
            # de combien resserrer par rapport au cash — qui était
            # inutilisable. Un joueur sans jeton n'a aucune chance de prendre
            # une place devant : probabilité nulle, pas une exception.
            return tuple([0.0] * n)
        first = tuple(
            (stacks[i] / total if mask & (1 << i) else 0.0) for i in range(n)
        )
        if place == 0:
            return first
        out = [0.0] * n
        for w in range(n):                      # w prend la place au-dessus
            pw = first[w]
            if pw <= 0.0:
                continue
            rest = place_probs(mask & ~(1 << w), place - 1)
            for i in range(n):
                out[i] += pw * rest[i]
        return tuple(out)

    full = (1 << n) - 1
    for k in range(min(n_places, n)):
        probs[:, k] = place_probs(full, k)
    probs.flags.writeable = False
    return probs


def _finish_probs_exact(stacks: tuple[float, ...], n_places: int) -> np.ndarray:
    """P(joueur i termine à la place k), récurrence de Harville mémoïsée.

    La mémoïsation porte sur le **sous-ensemble de joueurs encore en course**
    (bitmask) : chaque état n'est calculé qu'une fois, d'où O(2ⁿ·n²) au lieu
    de O(n!). Le résultat est en outre mis en cache ENTRE APPELS sur les
    tapis triés (`_finish_probs_tries`).

    Le tri est une canonicalisation exacte, pas une approximation : Harville
    ne connaît les joueurs que par leur tapis, donc permuter les tapis permute
    les lignes du résultat et rien d'autre — l'invariant de permutation que
    `tests/test_icm_invariants.py` mesure déjà. Le tableau rendu est une copie
    fraîche : l'appelant peut l'écrire sans toucher au cache.
    """
    ordre = sorted(range(len(stacks)), key=lambda i: (-stacks[i], i))
    canonique = _finish_probs_tries(tuple(stacks[i] for i in ordre), n_places)
    probs = np.empty_like(canonique)
    probs[ordre] = canonique          # permutation inverse
    return probs


def stats_cache_harville() -> tuple[int, int, int]:
    """(succès, échecs, entrées) du cache exact — pour mesurer, pas pour décider."""
    info = _finish_probs_tries.cache_info()
    return info.hits, info.misses, info.currsize


def vider_cache_harville() -> None:
    """Vide le cache exact. Utile aux bancs de mesure et aux tests de coût."""
    _finish_probs_tries.cache_clear()


def _finish_probs_mc(
    stacks: tuple[float, ...], n_places: int, n_sims: int, seed: int | None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Estimation Monte-Carlo : tirages successifs proportionnels aux stacks.

    Returns
    -------
    probs : numpy.ndarray, shape (n, n_places)
        Fréquence estimée de « le joueur *i* finit à la place *k* ».
    erreur_places : numpy.ndarray, même forme
        Erreur-type MESURÉE de chacune de ces fréquences,
        :math:`\sqrt{\hat p(1-\hat p)/n}` — la variance d'une proportion
        binomiale, estimée sur l'échantillon lui-même.

    POURQUOI ce n'est plus :math:`\sqrt{0{,}25/n}` : cette valeur est le
    MAXIMUM de :math:`\sqrt{p(1-p)/n}`, atteint en p = 0,5 seulement. Sur une
    table à 9 joueurs, les fréquences par place tournent autour de 1/9 ≈ 0,11,
    où l'erreur-type réelle vaut 0,31 fois la borne — la borne surestime alors
    l'incertitude d'un facteur ~3,2, et le chiffre était de toute façon jeté
    par l'appelant (`probs, _ = ...`). Une largeur d'intervalle NATIVE, déjà
    disponible, était produite puis perdue.
    """
    rng = np.random.default_rng(seed)
    n = len(stacks)
    counts = np.zeros((n, n_places), dtype=np.float64)
    base = np.asarray(stacks, dtype=np.float64)
    idx = np.arange(n)
    for _ in range(n_sims):
        remaining = base.copy()
        alive = idx.copy()
        for k in range(min(n_places, n)):
            p = remaining / remaining.sum()
            j = rng.choice(alive.size, p=p)
            counts[alive[j], k] += 1.0
            remaining = np.delete(remaining, j)
            alive = np.delete(alive, j)
    probs = counts / n_sims
    erreur_places = np.sqrt(np.maximum(probs * (1.0 - probs), 0.0) / n_sims)
    return probs, erreur_places


def erreur_type_exacte(
    probs: np.ndarray, payouts: np.ndarray, n_sims: int
) -> np.ndarray:
    r"""Erreur-type du $EV estimé par Monte-Carlo, joueur par joueur.

    Le tirage répète ``n_sims`` fois la variable « gain encaissé par le
    joueur *i* ». Les places sont MUTUELLEMENT EXCLUSIVES — un joueur en
    occupe au plus une — donc cette variable vaut :math:`\pi_k` avec
    probabilité :math:`p_{ik}` et zéro sinon, d'où

    .. math:: \operatorname{Var}(X_i) = \sum_k p_{ik}\pi_k^2
              - \Bigl(\sum_k p_{ik}\pi_k\Bigr)^2

    et :math:`\sigma_i = \sqrt{\operatorname{Var}(X_i)/n}`. Ce n'est PAS la
    somme des erreurs-types par place : celles-ci sont corrélées négativement,
    et les additionner surestimerait la largeur.

    Le ``max(·, 0)`` n'est pas cosmétique : sur des probabilités estimées, la
    différence de deux quantités presque égales peut sortir à −1e-18.

    Parameters
    ----------
    probs : numpy.ndarray, shape (n, k)
        Probabilités de place — exactes ou estimées.
    payouts : numpy.ndarray, shape (k,)
        Gains par place.
    n_sims : int
        Nombre de tirages.

    Examples
    --------
    Un seul gain et un seul joueur payé : la variable est constante, donc
    d'erreur-type nulle quel que soit le nombre de tirages.

    >>> import numpy as np
    >>> float(erreur_type_exacte(np.array([[1.0]]), np.array([100.0]), 10)[0])
    0.0
    """
    if n_sims < 1:
        raise IcmError("n_sims >= 1 requis.")
    esp = probs @ payouts
    var = probs @ (payouts ** 2) - esp ** 2
    return np.sqrt(np.maximum(var, 0.0) / n_sims)


def _places_des_morts(
    morts: list[int],
    restes: list[float],
    ordre: Sequence[int] | None,
) -> dict[int, float]:
    """Répartit les places restantes entre les joueurs à tapis nul.

    Parameters
    ----------
    morts : list of int
        Indices des joueurs sans jeton, dans l'ordre de la table.
    restes : list of float
        Gains encore à distribuer, du MEILLEUR au pire (``payouts`` privé des
        places occupées par les vivants).
    ordre : sequence of int or None
        Éliminés dont l'ordre est CONNU, du premier sorti au dernier. Les
        éliminés absents de cette liste sont réputés sortis AVANT tous ceux
        qui y figurent, et se partagent à parts égales ce qui reste — la
        convention d'ignorance, seule compatible avec l'invariance par
        permutation.

    Returns
    -------
    dict
        Gain attribué à chaque mort. La somme égale ``sum(restes)`` : la
        conservation de la dotation tient quelle que soit la déclaration.
    """
    # Autant de places que de morts : les places manquantes valent zéro (grille
    # plus courte que la table), ce qui garde `sum(places) == sum(restes)`.
    places = list(restes) + [0.0] * max(0, len(morts) - len(restes))
    connus = list(ordre or ())
    ensemble = set(connus)
    inconnus = [i for i in morts if i not in ensemble]

    out: dict[int, float] = {}
    # Le DERNIER éliminé prend la meilleure place restante, l'avant-dernier la
    # suivante, etc. — c'est la définition même de l'ordre d'élimination.
    for rang, joueur in enumerate(reversed(connus)):
        out[joueur] = places[rang] if rang < len(places) else 0.0
    if inconnus:
        part = sum(places[len(connus):]) / len(inconnus)
        for joueur in inconnus:
            out[joueur] = part
    return out


def icm_equities(
    stacks: Sequence[float],
    payouts: Sequence[float],
    n_sims: int = 200_000,
    seed: int | None = 0,
    *,
    ordre_elimination: Sequence[int] | None = None,
) -> np.ndarray:
    r"""$EV de chaque joueur : :math:`\$EV_i = \sum_k P(i \text{ finit } k)\,\pi_k`.

    Exact (Harville mémoïsé) jusqu'à 12 joueurs, Monte-Carlo au-delà.

    Parameters
    ----------
    ordre_elimination : sequence of int, optional
        Joueurs à tapis nul dont l'ordre de sortie est CONNU, du premier
        éliminé au dernier. Par défaut ``None`` : plus aucun ordre n'est
        supposé et les morts se partagent les places restantes à parts égales.

        Ce paramètre existe parce que le partage égal est FAUX quand l'ordre
        est connu, et qu'il l'est justement là où ça compte. Dans
        `bubble_factor`, l'état « le héros perd » met le héros à zéro : il est
        PAR CONSTRUCTION le dernier éliminé, puisque tous les autres joueurs
        à zéro l'étaient déjà avant la main. Le partage égal lui donnait la
        moyenne des places restantes au lieu de la meilleure d'entre elles.

        Ce n'est pas une convention parmi d'autres : c'est la LIMITE de la
        récursion. Sur ``[100, 100, 100, 0]`` avec ``[50, 30, 20, 10]``, un
        héros à ε jeton (l'autre mort restant à zéro exact) vaut 20,000000183
        à ε = 1e-6 et 20,000000000 à ε = 1e-9 — la limite est 20, la place
        immédiatement inférieure aux vivants. Le partage égal rendait 15,
        soit 25 % de moins, et cette valeur-là est DISCONTINUE en zéro.

    Examples
    --------
    Le cas d'école : 3 joueurs à égalité, payouts 50/30/20 — chacun vaut
    exactement la moyenne :

    >>> icm_equities([1000, 1000, 1000], [50, 30, 20]).round(6).tolist()
    [33.333333, 33.333333, 33.333333]

    Et la non-linéarité, en une ligne : le gros stack vaut MOINS que sa part
    de jetons, le petit vaut PLUS :

    >>> eq = icm_equities([6000, 3000, 1000], [50, 30, 20])
    >>> float(eq[0]) < 60.0 and float(eq[2]) > 10.0
    True

    Deux morts, ordre inconnu : partage égal (25 = (30+20)/2 chacun).

    >>> icm_equities([100, 0, 0], [50, 30, 20]).round(6).tolist()
    [50.0, 25.0, 25.0]

    Le même spot en déclarant que le joueur 1 est sorti AVANT le joueur 2 :

    >>> icm_equities([100, 0, 0], [50, 30, 20],
    ...              ordre_elimination=[1, 2]).round(6).tolist()
    [50.0, 20.0, 30.0]
    """
    s, p = _validate(stacks, payouts)
    if ordre_elimination is not None:
        vus: set[int] = set()
        for i in ordre_elimination:
            if not (0 <= i < len(s)):
                raise IcmError(f"ordre_elimination : indice {i} hors table.")
            if s[i] > 0.0:
                raise IcmError(
                    f"ordre_elimination : le joueur {i} a encore {s[i]:g} "
                    f"jeton(s) — on ne peut pas ordonner l'élimination d'un "
                    f"joueur vivant.")
            if i in vus:
                raise IcmError(f"ordre_elimination : joueur {i} en double.")
            vus.add(i)
    if not p:
        return np.zeros(len(s))

    # Un joueur sans jeton est DÉJÀ éliminé : il n'entre pas dans la course
    # aux places, il occupe les dernières. Le traiter comme les autres fait
    # diviser par un total nul ; lui donner zéro fait disparaître son gain,
    # et la somme des équités cesse d'égaler la dotation.
    #
    # Ce n'est pas un cas d'école : `bubble_factor` évalue précisément ce que
    # vaut le tapis du héros APRÈS avoir tout perdu. Lui attribuer zéro au
    # lieu du dernier gain surestime ce qu'il perd, donc la pression de bulle
    # elle-même — mesuré sur une vraie table à neuf joueurs : facteur 2,42
    # face au gros tapis avec le zéro, 2,04 avec le dernier gain, soit une
    # équité exigée de 70,8 % au lieu de 67,1 %.
    #
    # Les joueurs à zéro se partagent à parts égales les dernières places
    # TANT QUE `ordre_elimination` ne dit rien : rien ne permet alors de les
    # départager, et l'ordre de leur élimination n'est pas dans les données.
    # Dès que l'appelant le connaît — et `bubble_factor` le connaît, puisque
    # son héros vient de mourir APRÈS tout le monde — le partage égal est
    # simplement faux, et discontinu par-dessus le marché (voir la docstring
    # du paramètre).
    #
    # Cette branche est-elle la LIMITE de la récursion, ou seulement une
    # valeur voisine ? La question n'est pas rhétorique : `bubble_factor`
    # divise $EV_perte — évaluée ICI, pour le joueur couvert — par $EV_gain,
    # évaluée par la récursion ordinaire. Le quotient met en rapport deux
    # régimes de code, et il AMPLIFIE leur écart (mesuré ×11,9 à ×76,5) là où
    # la conservation, l'échelle et la permutation l'absorbent toutes.
    #
    # Mesuré (banc_invariants_icm.py --continuite, section 10) : elle l'EST.
    # En remplaçant le tapis nul par un tapis résiduel ε décroissant, l'écart
    # décroît linéairement en ε, et l'extrapolation en zéro retombe sur cette
    # branche à 2,0e-15 près en relatif, sur huit spots et dans les deux sens
    # de couverture. La forme générale :
    #
    #     lim icm(survivants ⊕ ε·mourants) =
    #         icm(survivants, π[:r]) ⊕ icm(mourants, π[r:])
    #
    # Le second facteur, évalué à tapis égaux, est exactement le partage
    # ci-dessous. À UN seul tapis nul il n'y a plus de rapport entre mourants,
    # donc une seule limite : la continuité est vraie. À PLUSIEURS, la limite
    # dépend de ces rapports — [100, ε, ε/2] tend vers (26,667 ; 23,333) et
    # non vers (25 ; 25) — donc il n'existe pas de valeur en zéro, et le
    # partage égal est un CHOIX (le seul compatible avec la permutation), pas
    # une convergence. `tests/test_icm_invariants.py` épingle les deux.
    #
    # `vivants` ne peut pas être vide ici : les tapis sont validés positifs ou
    # nuls, donc « aucun vivant » implique une somme nulle, que `_validate` a
    # déjà refusée. Le code qui prétendait partager la dotation dans ce cas
    # était inatteignable — un lecteur pouvait croire le cas traité alors que
    # l'appel lève `IcmError`. Le test `test_icm_invariants.py` épingle ce
    # comportement réel.
    vivants = [i for i, v in enumerate(s) if v > 0.0]
    if len(vivants) < len(s):
        out = np.zeros(len(s), dtype=np.float64)
        restes = list(p[len(vivants):])
        if restes:
            morts = [i for i in range(len(s)) if i not in set(vivants)]
            for joueur, gain in _places_des_morts(
                    morts, restes, ordre_elimination).items():
                out[joueur] = gain
        haut = list(p[:len(vivants)])
        if haut:
            out[vivants] = icm_equities([s[i] for i in vivants], haut,
                                        n_sims=n_sims, seed=seed)
        return out

    if len(s) <= EXACT_LIMIT:
        probs = _finish_probs_exact(s, len(p))
    else:
        probs, _ = _finish_probs_mc(s, len(p), n_sims, seed)
    return probs @ np.asarray(p, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class IcmMesure:
    """$EV par joueur ASSORTIS de leur incertitude — pas seulement le chiffre.

    Attributes
    ----------
    equities : numpy.ndarray, shape (n,)
        Les $EV, identiques à ceux d'`icm_equities` sur les mêmes arguments.
    erreur_type : numpy.ndarray, shape (n,)
        Erreur-type du $EV de chaque joueur. **Exactement zéro** sur le chemin
        exact : la récurrence de Harville ne tire rien au sort, son incertitude
        d'échantillonnage EST nulle (l'erreur de modèle, elle, n'est pas
        mesurée ici — voir la limite ci-dessous).
    erreur_places : numpy.ndarray, shape (n, k)
        Erreur-type de chaque probabilité de place. Zéro sur le chemin exact.
    exact : bool
        Vrai si aucun tirage n'a été fait.
    n_sims : int
        Tirages effectivement demandés (0 si exact).

    Limite à ne pas confondre
    -------------------------
    ``erreur_type`` mesure le BRUIT D'ÉCHANTILLONNAGE, pas la justesse du
    modèle. Un ICM exact reste une approximation de la réalité du tournoi
    (Harville suppose que la probabilité de finir devant est proportionnelle
    aux tapis) : l'erreur de modèle est ailleurs, et elle ne devient pas nulle
    parce que celle-ci l'est.
    """

    equities: np.ndarray
    erreur_type: np.ndarray
    erreur_places: np.ndarray
    exact: bool
    n_sims: int

    @property
    def demi_largeur_95(self) -> np.ndarray:
        """Demi-largeur de l'intervalle à 95 % (normal, 1,96 σ)."""
        return 1.959963984540054 * self.erreur_type


def icm_equities_mesurees(
    stacks: Sequence[float],
    payouts: Sequence[float],
    n_sims: int = 200_000,
    seed: int | None = 0,
    *,
    ordre_elimination: Sequence[int] | None = None,
) -> IcmMesure:
    r"""`icm_equities`, mais l'incertitude n'est plus jetée.

    Même calcul, mêmes chiffres — la seule différence est que la largeur
    d'intervalle produite par le Monte-Carlo remonte à l'appelant au lieu
    d'être perdue dans un ``probs, _ = ...``.

    Qui porte l'erreur : seuls les joueurs dont le classement a été TIRÉ. Un
    joueur à tapis nul reçoit sa place par la convention documentée dans
    `icm_equities` (partage égal, ou ordre déclaré) : aucun tirage, donc
    erreur-type nulle — et c'est vrai, pas une commodité.

    Examples
    --------
    Chemin exact : incertitude d'échantillonnage nulle, par construction.

    >>> m = icm_equities_mesurees([6000, 3000, 1000], [50, 30, 20])
    >>> m.exact, float(m.erreur_type.max())
    (True, 0.0)

    Et les $EV sont bien les mêmes que ceux de l'API historique :

    >>> import numpy as np
    >>> bool(np.allclose(m.equities, icm_equities([6000, 3000, 1000],
    ...                                           [50, 30, 20])))
    True
    """
    equities = icm_equities(stacks, payouts, n_sims=n_sims, seed=seed,
                            ordre_elimination=ordre_elimination)
    s, p = _validate(stacks, payouts)
    n, k = len(s), len(p)
    erreur_places = np.zeros((n, k), dtype=np.float64)
    erreur_type = np.zeros(n, dtype=np.float64)
    if not p:
        return IcmMesure(equities, erreur_type, erreur_places, True, 0)

    # Le tirage ne porte JAMAIS sur la table entière : les joueurs à tapis nul
    # sont servis par la branche de convention, sans hasard. C'est donc la
    # taille du sous-problème VIVANT qui décide du chemin, exactement comme
    # dans `icm_equities`.
    vivants = [i for i, v in enumerate(s) if v > 0.0]
    haut = p[: len(vivants)]
    if len(vivants) <= EXACT_LIMIT or not haut:
        return IcmMesure(equities, erreur_type, erreur_places, True, 0)

    probs, err_places = _finish_probs_mc(
        tuple(s[i] for i in vivants), len(haut), n_sims, seed)
    pay = np.asarray(haut, dtype=np.float64)
    erreur_places[np.ix_(vivants, range(len(haut)))] = err_places
    erreur_type[vivants] = erreur_type_exacte(probs, pay, n_sims)
    return IcmMesure(equities, erreur_type, erreur_places, False, n_sims)


def icm_dollar_ev(
    stacks: Sequence[float], payouts: Sequence[float], player: int, **kw
) -> float:
    """$EV d'un joueur donné."""
    eq = icm_equities(stacks, payouts, **kw)
    if not (0 <= player < len(eq)):
        raise IcmError("indice de joueur hors table.")
    return float(eq[player])


# ═══════════════════════════════════════════════════════════════════════════
# BUBBLE FACTOR & ÉQUITÉ REQUISE
# ═══════════════════════════════════════════════════════════════════════════


def _avec_ordre(kw: dict, etat: Sequence[float], sortant: int) -> dict:
    """``kw`` enrichi de « ``sortant`` est le dernier éliminé », s'il l'est.

    Si ``sortant`` a encore des jetons dans ``etat``, il n'est pas éliminé et
    rien n'est déclaré : `icm_equities` refuserait d'ordonner un vivant.
    """
    if etat[sortant] > 0.0:
        return kw
    return {**kw, "ordre_elimination": (sortant,)}


def bubble_factor(
    stacks: Sequence[float],
    payouts: Sequence[float],
    hero: int,
    villain: int,
    amount: float | None = None,
    **kw,
) -> float:
    r"""Bubble factor de hero contre villain (convention ICMIZER/HRC).

    .. math::
        BF = \frac{\$EV_{\text{hero}} - \$EV_{\text{hero}\,|\,\text{perd}}}
                  {\$EV_{\text{hero}\,|\,\text{gagne}} - \$EV_{\text{hero}}}

    C'est le rapport « ce que je perds en $ si je perds le pot » sur « ce que
    je gagne en $ si je le gagne », pour un même montant de jetons. En cash
    game, BF = 1 partout. En tournoi, BF > 1 dès qu'il y a des paliers — et il
    explose à la bulle contre les stacks qui te couvrent.

    Parameters
    ----------
    amount
        Jetons disputés. Par défaut : all-in pour le tapis effectif
        ``min(stack_hero, stack_villain)`` — la définition standard.

    Notes
    -----
    L'ordre d'élimination n'est PAS un paramètre : cette fonction le connaît
    par construction et le déclare elle-même à `icm_equities`. Dans l'état
    « le héros perd », le héros vient de sortir, donc APRÈS tous les joueurs
    déjà à zéro ; dans l'état « le héros gagne » et s'il couvre, c'est le
    vilain qui sort en dernier. Le lui passer serait une contradiction, d'où
    le refus explicite plutôt qu'un écrasement silencieux.
    """
    if "ordre_elimination" in kw:
        raise IcmError(
            "bubble_factor détermine lui-même l'ordre d'élimination : dans "
            "l'état perdant, le héros est le dernier sorti. Le déclarer de "
            "l'extérieur ne peut que le contredire.")
    s, p = _validate(stacks, payouts)
    if hero == villain:
        raise IcmError("hero et villain doivent différer.")
    for i in (hero, villain):
        if not (0 <= i < len(s)):
            raise IcmError("indice hors table.")
    eff = min(s[hero], s[villain])
    amt = eff if amount is None else float(amount)
    if not (0.0 < amt <= eff):
        raise IcmError("amount doit être dans (0, tapis effectif].")

    base = icm_dollar_ev(s, p, hero, **kw)

    win = list(s)
    win[hero] += amt
    win[villain] -= amt
    # Gagner, c'est éliminer : le vilain qui tombe à zéro sort MAINTENANT,
    # donc après les joueurs déjà éliminés. Sans effet sur le $EV du héros
    # (les vivants se partagent `p[:len(vivants)]` quoi qu'il arrive), mais le
    # vecteur rendu par `icm_equities` reste alors juste pour tout le monde.
    ev_win = icm_dollar_ev(win, p, hero, **_avec_ordre(kw, win, villain))

    lose = list(s)
    lose[hero] -= amt
    lose[villain] += amt
    # LE terme qui compte : le héros vient de mourir, il est le dernier sorti.
    ev_lose = icm_dollar_ev(lose, p, hero, **_avec_ordre(kw, lose, hero))

    gain = ev_win - base
    loss = base - ev_lose

    # Le rapport est un 0/0 dès que la structure ne met plus rien en jeu : si
    # tous les joueurs encore en lice touchent le même gain (satellite dont il
    # ne reste que des places équivalentes), déplacer des jetons ne change
    # aucune équité. `gain` et `loss` valent alors zéro EN THÉORIE, et ±1 ulp
    # en pratique.
    #
    # Comparer `gain` à zéro exactement laissait donc passer le résidu, et le
    # même cas dégénéré rendait deux réponses opposées au gré de l'arrondi :
    # −1,0 sur ([1, 2, 3], [1, 1, 1]) — un facteur de bulle NÉGATIF, qui n'a
    # aucun sens et faisait ensuite échouer `analyse_icm_spot` sur le message
    # trompeur « bubble factor doit être > 0 » — et +inf sur
    # ([10, 20, 30, 40], [7, 7, 7, 7]).
    #
    # Le garde-fou se compare donc à l'ÉCHELLE des $EV, pas à zéro. Le seuil
    # 1e-12 est encadré par la mesure (banc_invariants_icm.py --seuils) : le
    # résidu d'arrondi du cas dégénéré pèse ~1e-16 en relatif, tandis que le
    # gain légitime d'un heads-up à 1 jeton contre 1e9 pèse ~3,3e-10.
    #
    # LIMITE MESURÉE, à ne pas découvrir plus tard : le garde-fou renvoie
    # +inf dès que le rapport des tapis atteint ~1e12 en heads-up, où le gain
    # légitime (3,3e-13 en relatif) passe sous le seuil. Ce n'est pas une
    # perte : à ce rapport, la soustraction de deux $EV voisins a déjà perdu
    # tous ses chiffres significatifs — le calcul rendait 0,99996 au lieu
    # de 1 dès 1e11, et du bruit pur au-delà. Répondre +inf (« je ne peux pas
    # trancher, suppose la pression maximale ») est prudent et cohérent, là
    # où un nombre plausible et faux serait le pire des deux. Un tournoi réel
    # ne dépasse pas un rapport de ~1e6.
    echelle = max(abs(base), abs(ev_win), abs(ev_lose))
    if gain <= 1e-12 * echelle:
        return math.inf
    # L'équité ICM est croissante en jetons (invariant de monotonie) : perdre
    # des jetons ne peut pas rapporter. Une perte négative est du bruit.
    return float(max(loss, 0.0) / gain)


def risk_premium(bf: float) -> float:
    r"""Prime de risque : équité supplémentaire requise vs cash game.

    Pour un pot symétrique (all-in 50/50 du montant), l'équité de
    neutralité $EV passe de 50 % à :math:`BF/(1+BF)` ; la prime est l'écart.

    >>> risk_premium(1.0)
    0.0
    >>> round(risk_premium(1.5), 4)
    0.1
    """
    if bf < 0:
        raise IcmError("bubble factor négatif.")
    if math.isinf(bf):
        return 0.5
    return float(bf / (1.0 + bf) - 0.5)


def icm_required_equity(pot: float, bet: float, bf: float) -> float:
    r"""Équité requise pour un call en tournoi — le pot odds corrigé ICM.

    En cash : :math:`\alpha = b/(P+2b)` (F10). En tournoi, chaque jeton risqué
    pèse BF fois plus que chaque jeton gagnable :

    .. math::
        \alpha_{ICM} = \frac{BF \cdot b}{P + b + BF \cdot b}

    Se réduit au pot odds classique quand BF = 1.

    >>> round(icm_required_equity(100, 75, 1.0), 4)   # cash
    0.3
    >>> round(icm_required_equity(100, 75, 2.0), 4)   # bulle sévère
    0.4615
    """
    if pot < 0 or bet <= 0:
        raise IcmError("pot >= 0 et bet > 0 requis.")
    if bf <= 0:
        raise IcmError("bubble factor doit être > 0.")
    if math.isinf(bf):
        return 1.0
    return float(bf * bet / (pot + bet + bf * bet))


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSE DE SPOT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class IcmSpot:
    stacks: tuple[float, ...]
    payouts: tuple[float, ...]
    hero: int
    villain: int
    pot: float
    bet: float


@dataclass(frozen=True, slots=True)
class IcmAnalysis:
    dollar_ev: tuple[float, ...]
    hero_ev: float
    bubble: float
    premium: float
    alpha_cash: float
    alpha_icm: float
    verdict: str

    def explain(self) -> str:
        lines = [
            "$EV par joueur : " + "  ".join(f"{v:.2f}" for v in self.dollar_ev),
            f"bubble factor hero→villain : {self.bubble:.3f}"
            f"  (prime de risque {self.premium * 100:+.1f} pts)",
            f"équité requise : cash {self.alpha_cash * 100:.1f} %"
            f"  →  ICM {self.alpha_icm * 100:.1f} %",
            f"→ {self.verdict}",
        ]
        return "\n".join(lines)


def analyse_icm_spot(spot: IcmSpot, hero_equity: float | None = None) -> IcmAnalysis:
    """Analyse complète d'un spot de call en tournoi.

    C'est le point de jonction avec F10/F13 : ``alpha_icm`` remplace le pot
    odds brut dans l'analyse de bluff-catch dès que le spot est en tournoi.
    """
    ev = icm_equities(spot.stacks, spot.payouts)
    bf = bubble_factor(spot.stacks, spot.payouts, spot.hero, spot.villain)
    a_cash = spot.bet / (spot.pot + 2.0 * spot.bet)
    a_icm = icm_required_equity(spot.pot, spot.bet, bf)

    if hero_equity is None:
        verdict = (f"il faut {a_icm * 100:.1f} % d'équité pour payer "
                   f"(contre {a_cash * 100:.1f} % en cash).")
    elif hero_equity >= a_icm:
        verdict = (f"CALL : {hero_equity * 100:.1f} % ≥ {a_icm * 100:.1f} % requis "
                   f"malgré l'ICM.")
    elif hero_equity >= a_cash:
        verdict = (f"FOLD ICM : {hero_equity * 100:.1f} % suffirait en cash "
                   f"({a_cash * 100:.1f} %) mais pas en tournoi "
                   f"({a_icm * 100:.1f} %). C'est LE fold que les joueurs de "
                   f"cash ratent.")
    else:
        verdict = f"FOLD : {hero_equity * 100:.1f} % < {a_cash * 100:.1f} % même en cash."

    return IcmAnalysis(
        dollar_ev=tuple(float(x) for x in ev),
        hero_ev=float(ev[spot.hero]),
        bubble=bf,
        premium=risk_premium(bf),
        alpha_cash=float(a_cash),
        alpha_icm=float(a_icm),
        verdict=verdict,
    )


# ═══════════════════════════════════════════════════════════════════════════
# L5 — BOUNTIES PKO (progressive knockout)
# ═══════════════════════════════════════════════════════════════════════════
#
# Règle PKO standard : éliminer un joueur rapporte la MOITIÉ de sa prime en
# cash ; l'autre moitié s'ajoute à ta propre prime. Ta propre prime ne t'est
# versée que si tu GAGNES le tournoi (le vainqueur encaisse sa prime).
#
# Valeur exacte d'une capture, sous Harville :
#     BV = B_vilain · (½ + ½ · P(héros finit 1er | stacks après victoire))
# avec P(1er) = s/S — le premier étage de Harville, exact.
#
# L'équité requise du call se calcule par différences de $EV, sans
# approximation de conversion jetons↔$ :
#     r* = ($EV_fold − $EV_perd) / ($EV_gagne + BV·1{vilain éliminé} − $EV_perd)
#
# Convention d'états (P = pot TOTAL déjà engagé, y compris la mise adverse ;
# b = ce que héros doit payer) :
#     fold   : vilain += P
#     gagne  : héros += P  (sa mise revient avec le pot)
#     perd   : héros −= b ; vilain += P + b


@dataclass(frozen=True, slots=True)
class PkoSpot:
    stacks: tuple[float, ...]
    payouts: tuple[float, ...]
    bounties: tuple[float, ...]
    hero: int
    villain: int
    pot: float          # pot total, mise adverse incluse
    bet: float          # montant à payer

    villain_all_in: bool | None = None
    """Le vilain est-il à tapis ET couvert par le héros ? (la prime est-elle
    capturable ?)

    ``None`` — le défaut — laisse `analyse_pko_spot` l'INFÉRER du tapis
    résiduel, ce qui est fragile : la convention de `PkoSpot` veut les jetons
    du vilain à tapis dans ``pot`` et son tapis à zéro, mais un résidu
    (arrondi d'ante, tapis lu à l'écran, saisie) suffisait à faire
    disparaître la prime **en silence**. Mesuré sur une table finale de MTT
    PKO réel (3 337 entrants × 5 000 jetons, 6 685 000 jetons en jeu), du
    temps du repli ``1e-12 · Σ tapis`` : un résidu de 1 jeton — 0,0003 % du
    tapis du vilain — faisait passer l'équité exigée de 49,95 % à 53,59 %.
    Un chiffre faux et plausible, du côté du fold. Le repli juge désormais le
    résidu à l'échelle de la transaction (`SEUIL_RESIDU_TRANSACTION`), ce qui
    neutralise cette classe d'artefacts — mais l'inférence reste une
    inférence.

    Le déclarer coûte un booléen et supprime la classe entière de défauts.
    `spot_pko_face_a_tapis` le renseigne toujours.
    """

    unite_jeton: float | None = None
    """Valeur d'UN jeton, dans l'unité de ``stacks`` — repli d'inférence.

    Quand ``villain_all_in`` vaut ``None`` et que celle-ci est donnée, le
    vilain est réputé éliminé si son tapis résiduel est sous le demi-jeton.
    C'est le seul seuil qui ait un sens physique : entre zéro et un jeton, il
    n'existe rien à une table de poker.

    ARBITRAGE ASSUMÉ : ce critère n'est PAS invariant d'échelle, et il ne peut
    pas l'être. « Un jeton » est une grandeur dimensionnée ; exprimer les
    tapis en blindes ou en millions change ce qu'il vaut. Les deux exigences
    — invariance d'échelle et comparaison en jetons entiers — sont donc
    incompatibles tant que l'unité n'accompagne pas les tapis. C'est
    précisément ce que ce champ fournit : l'unité voyage AVEC les nombres,
    et le couple (tapis, unite_jeton) redevient invariant d'échelle — doubler
    les deux ne change aucun verdict. C'est l'invariance d'échelle qui prime,
    à condition de la porter sur la bonne quantité.

    Laissée à ``None`` sans ``villain_all_in``, l'inférence retombe sur
    l'échelle de la TRANSACTION : un résidu inférieur à
    ``SEUIL_RESIDU_TRANSACTION · max(pot, bet)`` est réputé nul au sens du
    jeu. Voir `SEUIL_RESIDU_TRANSACTION` pour les deux bornes mesurées qui
    encadrent ce seuil — et pourquoi l'ancien repli ``1e-12 · Σ tapis``
    comparait le résidu à une grandeur qui n'a rien à voir avec lui.
    """


@dataclass(frozen=True, slots=True)
class PkoAnalysis:
    ev_fold: float
    ev_win: float
    ev_lose: float
    bounty_value: float          # BV si le call couvre le vilain, sinon 0
    villain_eliminated: bool
    required_no_bounty: float    # équité requise ICM pure
    required_with_bounty: float  # équité requise avec la prime
    discount_pts: float          # points d'équité offerts par la prime
    verdict: str
    source_elimination: str = "échelle de la transaction (repli)"
    """D'où vient ``villain_eliminated`` : déclaration, unité de jeton, ou
    repli sur l'échelle de la transaction. Sans cette trace, impossible de
    savoir si la prime a été décidée ou devinée."""
    signaux: tuple[str, ...] = ()
    """Anomalies EXPLICITES là où le code tronquait en silence (``min``/``max``
    sur les équités requises). Vide dans le cas nominal."""

    def explain(self) -> str:
        lines = [
            f"$EV : fold {self.ev_fold:.2f} · gagne {self.ev_win:.2f}"
            f" · perd {self.ev_lose:.2f}",
            (f"prime capturable : {self.bounty_value:.2f} $"
             f"  [{self.source_elimination}]"
             if self.villain_eliminated else
             "le call ne couvre pas le vilain : pas de prime en jeu"
             f"  [{self.source_elimination}]"),
            f"équité requise : ICM {self.required_no_bounty * 100:.1f} %"
            f"  →  PKO {self.required_with_bounty * 100:.1f} %"
            f"  (−{self.discount_pts * 100:.1f} pts)",
            *(f"⚠ {x}" for x in self.signaux),
            f"→ {self.verdict}",
        ]
        return "\n".join(lines)


def bounty_capture_value(
    stacks_after_win: Sequence[float],
    hero: int,
    villain_bounty: float,
    **kw,
) -> float:
    """Valeur $ exacte de la capture : ½ cash + ½ sur sa propre prime.

    La moitié ajoutée à sa propre prime ne vaut que P(finir 1er), qui sous
    Harville est la part de jetons — exact, pas une approximation.

    >>> # Héros à 200/300 jetons après la victoire, prime 50 :
    >>> round(bounty_capture_value([200, 0, 100], 0, 50.0), 4)
    41.6667
    """
    if villain_bounty < 0:
        raise IcmError("prime négative.")
    s = [float(x) for x in stacks_after_win]
    total = sum(s)
    if total <= 0 or not (0 <= hero < len(s)):
        raise IcmError("stacks ou indice héros invalides.")
    p_first = s[hero] / total
    return float(villain_bounty * (0.5 + 0.5 * p_first))


TOLERANCE_EQUITE = 1e-12
"""Bande de bruit autour de 0 et de 1 pour les équités requises.

Sous l'ICM, ``0 ≤ r ≤ 1`` est un THÉORÈME et non un hasard : la monotonie
donne ``$EV_perte ≤ $EV_fold ≤ $EV_gain`` (le héros a strictement moins de
jetons quand il perd, strictement plus quand il gagne). Un dépassement de
l'ordre de l'ulp est donc de l'arrondi, et on colle à la borne ; au-delà,
c'est une information, et elle doit sortir.
"""


def _equite_requise(r: float, nom: str, signaux: list[str]) -> float:
    """Rend ``r`` SANS troncature, en signalant ce qu'un clamp aurait masqué.

    Les anciens ``min(max(r, 0.0), 1.0)`` faisaient disparaître deux
    informations fortes :

    * ``r < 0`` : payer est +EV même avec 0 % d'équité. Impossible dans le
      modèle écrit ici (voir `TOLERANCE_EQUITE`) — donc si le nombre sort,
      c'est que le modèle ou la saisie est en cause, et le taire est le pire
      des choix. Ce sera aussi la valeur légitime le jour où la perte de sa
      PROPRE prime sera modélisée : en PKO, buster coûte sa prime, ce qui
      abaisse encore l'équité de neutralité.
    * ``r > 1`` : aucune main ne peut payer. Sous ICM, gagner le pot ne peut
      pas rapporter moins que le coucher, donc c'est une erreur de saisie —
      typiquement des tapis qui ne cadrent pas avec ``pot``/``bet``.
    """
    if r < -TOLERANCE_EQUITE:
        signaux.append(
            f"{nom} : équité requise NÉGATIVE ({r:+.4%}). Payer serait +EV "
            f"même avec 0 % d'équité — sous ce modèle c'est impossible "
            f"(monotonie ICM), donc c'est le modèle ou la saisie qu'il faut "
            f"regarder, pas la main.")
        return float(r)
    if r > 1.0 + TOLERANCE_EQUITE:
        signaux.append(
            f"{nom} : équité requise > 100 % ({r:.4%}). Gagner le pot "
            f"rapporterait moins que le coucher, ce que l'ICM interdit : "
            f"vérifie la cohérence entre `stacks`, `pot` et `bet`.")
        return float(r)
    return float(min(max(r, 0.0), 1.0))


def _verdict_pko(r_pko: float, r_icm: float,
                 hero_equity: float | None) -> str:
    """Phrase de conclusion d'un call PKO.

    Fonction séparée pour une raison mesurable : les deux premières branches
    portent sur des équités requises HORS de [0, 1], et le chemin exact de
    l'ICM ne peut pas les produire (0 ≤ r ≤ 1 est un théorème, cf.
    `TOLERANCE_EQUITE` — vérifié sur 2 584 spots aléatoires, zéro sortie). Les
    tester à travers `analyse_pko_spot` demanderait de fabriquer un ICM faux ;
    les tester ici demande un appel. Le code n'est pas mort pour autant : le
    chemin Monte-Carlo (> 12 joueurs) estime les trois $EV, et une estimation
    n'est pas monotone.
    """
    if r_pko < 0.0:
        return (f"CALL AUTOMATIQUE : l'équité requise est NÉGATIVE "
                f"({r_pko * 100:.1f} %) — payer est +EV même avec 0 % "
                f"d'équité.")
    if r_pko > 1.0:
        return (f"FOLD FORCÉ : l'équité requise dépasse 100 % "
                f"({r_pko * 100:.1f} %) — aucune main ne peut payer. "
                f"Vérifie la saisie : sous l'ICM, gagner le pot ne peut "
                f"pas rapporter moins que le coucher.")
    if hero_equity is None:
        return (f"il faut {r_pko * 100:.1f} % d'équité pour payer "
                f"(ICM pur : {r_icm * 100:.1f} %).")
    if hero_equity >= r_pko:
        extra = " — la prime fait basculer le call" if hero_equity < r_icm else ""
        return f"CALL : {hero_equity * 100:.1f} % ≥ {r_pko * 100:.1f} %{extra}."
    return f"FOLD : {hero_equity * 100:.1f} % < {r_pko * 100:.1f} %."


SEUIL_RESIDU_TRANSACTION = 1e-5
"""Fraction de la transaction (``max(pot, bet)``) sous laquelle un tapis
résiduel est réputé NUL AU SENS DU JEU, quand ni ``villain_all_in`` ni
``unite_jeton`` ne sont fournis.

POURQUOI la transaction et pas ``Σ tapis`` : le résidu du vilain sort de
l'arithmétique de SA transaction — ``stack − engagement``, où l'engagement
vit dans ``pot``/``bet``. Les tapis des autres joueurs n'entrent jamais dans
cette soustraction ; en faire l'échelle du seuil, c'est rendre le verdict
dépendant d'une grandeur étrangère au calcul qui a produit le nombre jugé —
la dépendance à l'unité du seuil absolu, recréée en relatif (tour 4 du
conseil). Mesuré sur l'exemple du tour 4 : un jeton de saisie résiduel sur
un all-in de 288 000 (tournoi à 16,7 M) restait « vivant » pour le seuil
``1e-12 · Σ``, la prime tombait à zéro et l'équité exigée montait de
36,6 % à 42,2 % — un chiffre faux et plausible, du côté du fold.

Le seuil est encadré par deux bornes MESURÉES :

* **au-dessous, ce qui doit être vu comme zéro** — l'annulation flottante de
  ``stack − bet`` pèse quelques ulp de l'opérande, soit ~1e-12 de la
  transaction (7 ordres sous le seuil) ; un jeton de départ résiduel
  (arrondi d'ante, tapis lu à l'écran, saisie) sur une table où l'all-in en
  cours pèse ≥ 3e5 unités : ≤ 3,3e-6, soit 3× sous le seuil ;
* **au-dessus, ce qui doit rester un joueur vivant** — le plus petit jeton
  qui circule vaut au moins bb/100 (en ligne l'affichage ne descend pas
  sous ce pas dans les structures réelles ; en live la chip race retire les
  coupures inférieures), et un all-in ne dépasse pas ~300 bb : un vrai
  résidu pèse au moins 1/(100 · 300) = 3,3e-5 de la transaction, soit 3× au
  DESSUS du seuil.

1e-5 est la moyenne géométrique de ces deux bornes (√(3,3e-6 · 3,3e-5) =
1,04e-5). Le rapport résidu/transaction étant sans dimension, le critère est
invariant d'échelle — c'est ce que le seuil absolu n'était pas. LIMITE
ASSUMÉE : entre les deux bornes, aucune inférence n'est possible sans
l'unité ; c'est exactement ce que ``villain_all_in`` et ``unite_jeton``
fournissent, et ils PRIMENT toujours sur ce repli.
"""


def _vilain_elimine(spot: PkoSpot, s: tuple[float, ...],
                    v: int) -> tuple[bool, str]:
    """Le vilain est-il éliminable par ce call ? Et d'où vient la réponse ?

    Trois régimes, du plus sûr au moins sûr :

    1. ``spot.villain_all_in`` déclaré : on le croit, après avoir vérifié
       qu'il ne contredit pas les tapis fournis. C'est le seul régime qui ne
       dépende ni de l'unité ni de l'échelle.
    2. ``spot.unite_jeton`` fourni : moins d'un demi-jeton = éliminé. Le
       couple (tapis, unité) reste invariant d'échelle. La déclaration
       d'unité est une ASSERTION que les tapis sont exacts en jetons
       entiers : un résidu d'un jeton plein y est un joueur vivant, là où le
       repli 3 le lirait comme un artefact.
    3. ni l'un ni l'autre : le résidu est jugé à l'échelle de la TRANSACTION
       qui l'a produit — nul au sens du jeu s'il pèse moins de
       ``SEUIL_RESIDU_TRANSACTION · max(pot, bet)``. L'ancien repli
       ``1e-12 · Σ tapis`` comparait le résidu aux tapis des AUTRES joueurs,
       qui n'entrent jamais dans ``stack − bet`` : sur une table finale de
       MTT réel il valait 6,7e-6 jeton, et tout artefact de saisie — fût-il
       d'un millième de jeton — faisait tomber la prime en silence.
    """
    if spot.villain_all_in is not None:
        seuil = 1e-12 * sum(s)
        if spot.villain_all_in and s[v] > seuil:
            raise IcmError(
                f"villain_all_in=True mais le vilain garde {s[v]:g} jeton(s) "
                f"dans `stacks`. `PkoSpot` attend les tapis APRÈS engagement : "
                f"les jetons du vilain à tapis appartiennent à `pot`, pas à "
                f"`stacks`. `spot_pko_face_a_tapis` fait la conversion depuis "
                f"les tapis lus à l'écran.")
        return bool(spot.villain_all_in), "déclaré par l'appelant"
    if spot.unite_jeton is not None:
        if spot.unite_jeton <= 0.0:
            raise IcmError("unite_jeton doit être > 0.")
        return (s[v] < 0.5 * spot.unite_jeton,
                f"unité de jeton ({spot.unite_jeton:g})")
    transaction = max(spot.pot, spot.bet)
    return (s[v] <= SEUIL_RESIDU_TRANSACTION * transaction,
            "échelle de la transaction (repli)")


def analyse_pko_spot(spot: PkoSpot, hero_equity: float | None = None) -> PkoAnalysis:
    """Analyse exacte d'un call PKO par différences de $EV (+ prime).

    >>> # Winner-take-all 3-way, stacks égaux, prime 50 : le call 50/50
    >>> # devient un call à 30.8 % grâce à la prime.
    >>> s = PkoSpot(stacks=(100.0, 0.0, 100.0), payouts=(100.0,),
    ...             bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
    ...             pot=100.0, bet=100.0)
    >>> a = analyse_pko_spot(s)
    >>> round(a.required_no_bounty, 6), round(a.required_with_bounty, 6)
    (0.5, 0.307692)
    """
    s, p = _validate(spot.stacks, spot.payouts)
    n = len(s)
    if len(spot.bounties) != n:
        raise IcmError("il faut une prime par joueur.")
    if any(b < 0 for b in spot.bounties):
        raise IcmError("prime négative.")
    h, v = spot.hero, spot.villain
    if h == v or not (0 <= h < n) or not (0 <= v < n):
        raise IcmError("indices héros/vilain invalides.")
    if spot.bet <= 0 or spot.pot < 0:
        raise IcmError("pot >= 0 et bet > 0 requis.")
    if spot.bet > s[h]:
        raise IcmError("héros ne peut pas payer plus que son stack.")

    fold_s = list(s); fold_s[v] += spot.pot
    win_s = list(s);  win_s[h] += spot.pot
    lose_s = list(s); lose_s[h] -= spot.bet; lose_s[v] += spot.pot + spot.bet

    ev_fold = icm_dollar_ev(fold_s, p, h)
    # Gagner élimine le vilain, perdre élimine le héros — et dans les deux cas
    # le sortant sort APRÈS ceux qui étaient déjà à zéro. Sans cette
    # déclaration, `icm_equities` partagerait les places restantes à parts
    # égales et sous-estimerait $EV_perte (voir `ordre_elimination`).
    ev_win = icm_dollar_ev(win_s, p, h, **_avec_ordre({}, win_s, v))
    ev_lose = icm_dollar_ev(lose_s, p, h, **_avec_ordre({}, lose_s, h))

    eliminated, source = _vilain_elimine(spot, s, v)
    bv = (bounty_capture_value(win_s, h, spot.bounties[v])
          if eliminated else 0.0)

    denom_icm = ev_win - ev_lose
    denom_pko = ev_win + bv - ev_lose
    if denom_icm <= 0 or denom_pko <= 0:
        raise IcmError("spot dégénéré : gagner ne rapporte rien.")

    signaux: list[str] = []
    r_icm = _equite_requise(
        (ev_fold - ev_lose) / denom_icm, "ICM pur", signaux)
    r_pko = _equite_requise(
        (ev_fold - ev_lose) / denom_pko, "PKO", signaux)

    verdict = _verdict_pko(r_pko, r_icm, hero_equity)

    return PkoAnalysis(ev_fold, ev_win, ev_lose, bv, eliminated,
                       r_icm, r_pko, r_icm - r_pko, verdict,
                       source, tuple(signaux))


def spot_pko_face_a_tapis(
    tapis: Sequence[float],
    payouts: Sequence[float],
    primes: Sequence[float],
    hero: int,
    villain: int,
    *,
    blindes_mortes: float = 0.0,
    deja_engage_hero: float = 0.0,
) -> PkoSpot:
    """Construit le spot « il part à tapis, je paie » depuis les tapis RÉELS.

    `PkoSpot` attend des tapis **après** engagement : le vilain à tapis y
    figure à zéro, ses jetons étant comptés dans ``pot``. Cette convention
    est correcte mais se retourne facilement — je m'y suis pris les pieds en
    analysant une vraie table, en passant le tapis entier du vilain : il
    n'était alors pas vu comme éliminé, la prime valait zéro, et l'équité
    exigée sortait à 90 % au lieu de 53 %. Un chiffre faux et plausible, le
    pire genre.

    Cette fonction prend les tapis tels qu'on les lit à l'écran et fait la
    conversion.

    Parameters
    ----------
    tapis : sequence of float
        Tapis observés, dans la même unité que ``payouts`` n'a pas à
        partager (les jetons suffisent, l'ICM est invariant d'échelle).
    primes : sequence of float
        Une prime par joueur, en euros.
    blindes_mortes : float
        Ce qui traîne déjà au pot sans appartenir aux deux joueurs : petite
        blinde couchée, antes de tout le monde.
    deja_engage_hero : float
        Ce que le héros a déjà posé (sa grosse blinde, typiquement) et qu'il
        n'a donc pas à payer une seconde fois.

    Returns
    -------
    PkoSpot
        Prêt pour `analyse_pko_spot`.

    Examples
    --------
    >>> # Vilain à 19,25 bb part à tapis ; héros à 35,64 bb en grosse blinde.
    >>> s = spot_pko_face_a_tapis(
    ...     [35.64, 19.25], [100.0, 40.0], [8.13, 4.17], hero=0, villain=1,
    ...     blindes_mortes=1.4, deja_engage_hero=1.0)
    >>> s.stacks[1], round(s.pot, 2), round(s.bet, 2)
    (0.0, 20.65, 18.25)
    """
    n = len(tapis)
    if not (0 <= hero < n) or not (0 <= villain < n) or hero == villain:
        raise IcmError("indices héros/vilain invalides.")

    # Le vilain ne peut engager que ce qu'il a, et le héros ne peut perdre
    # que jusqu'à concurrence de son propre tapis.
    engage = min(float(tapis[villain]), float(tapis[hero]))
    apres = list(float(t) for t in tapis)
    apres[villain] = float(tapis[villain]) - engage   # 0 s'il est couvert
    apres[hero] = float(tapis[hero]) - deja_engage_hero

    pot = engage + blindes_mortes
    bet = engage - deja_engage_hero
    if bet <= 0:
        raise IcmError("rien à payer : le vilain n'engage pas plus que "
                       "ce que le héros a déjà posé.")
    # Cette fonction SAIT qui couvre qui : elle vient de le calculer. Le
    # déclarer supprime toute inférence en aval — c'est le correctif du défaut
    # que la docstring ci-dessus raconte, poussé jusqu'au bout. Le laisser à
    # `None` reviendrait à re-deviner depuis un tapis résiduel ce qu'on a sous
    # la main exactement.
    return PkoSpot(stacks=apres, payouts=list(payouts),
                   bounties=list(primes), hero=hero, villain=villain,
                   pot=pot, bet=bet,
                   villain_all_in=float(tapis[villain]) <= float(tapis[hero]))


# ═══════════════════════════════════════════════════════════════════════════
# L4 — FGS LÉGER : érosion des blindes futures
# ═══════════════════════════════════════════════════════════════════════════
#
# La critique standard de l'ICM statique : il ignore QUI paiera les blindes.
# Le FGS complet (HRC) simule les mains futures avec équilibres push-fold ;
# la version légère, au premier ordre, fait « passer » les blindes : à chaque
# main future, tout le monde se couche sauf la BB (modèle du walk), les jetons
# sont conservés, les stacks courts saignent. L'ICM recalculé sur ces stacks
# érodés donne des $EV et bubble factors FGS-ajustés.
#
# Hypothèse documentée : érosion fold-everything — pessimiste pour les stacks
# courts, neutre pour les gros. C'est la correction de premier ordre ; le FGS
# à équilibres viendra avec le solveur préflop (Phase 2).


@dataclass(frozen=True, slots=True)
class FgsResult:
    equities: np.ndarray        # (n_hands+1, n_players) — ligne 0 = statique
    stacks_path: np.ndarray     # (n_hands+1, n_players)
    deltas: np.ndarray          # equities[k] − equities[0]

    @property
    def static(self) -> np.ndarray:
        return self.equities[0]

    @property
    def fgs_mean(self) -> np.ndarray:
        """$EV moyen sur l'horizon — la valeur FGS-ajustée."""
        return self.equities[1:].mean(axis=0)


def fgs_equities(
    stacks: Sequence[float],
    payouts: Sequence[float],
    button: int,
    sb: float,
    bb: float,
    ante: float = 0.0,
    n_hands: int | None = None,
    **kw,
) -> FgsResult:
    """ICM sur stacks érodés par les blindes des ``n_hands`` prochaines mains.

    Modèle du walk : SB paie sb, chaque joueur l'ante, la BB ramasse tout —
    conservation des jetons garantie. Un joueur à 0 est éliminé de la
    rotation (son $EV reste défini : 0 jeton → plus bas rang Harville).
    Ce que le modèle mesure : l'ORDRE de passage des blindes — un stack court
    qui doit poster avant de encaisser son walk perd du $EV pendant tout
    l'intervalle, ce que l'ICM statique ne voit pas.

    >>> import numpy as np
    >>> # bouton en 1 : le short (indice 2) poste la SB dès la main suivante
    >>> r = fgs_equities([5000, 3000, 500, 1500], [50, 30, 20],
    ...                  button=1, sb=100, bb=200, n_hands=4)
    >>> bool(r.deltas[1:, 2].min() < -1.0)   # il saigne > 1 pt de $EV
    True
    >>> float(np.abs(r.equities.sum(axis=1) - 100.0).max()) < 1e-6
    True
    """
    s, p = _validate(stacks, payouts)
    n = len(s)
    if not (0 <= button < n):
        raise IcmError("indice de bouton invalide.")
    if sb < 0 or bb <= 0 or ante < 0:
        raise IcmError("blindes invalides (bb > 0, sb >= 0, ante >= 0).")
    if n_hands is None:
        n_hands = sum(1 for x in s if x > 0)
    if n_hands < 1:
        raise IcmError("n_hands >= 1 requis.")

    cur = list(s)
    path = [list(cur)]
    eqs = [list(icm_equities(cur, p, **kw))]
    btn = button
    for _ in range(n_hands):
        alive = [i for i in range(n) if cur[i] > 0]
        if len(alive) < 2:
            path.append(list(cur))
            eqs.append(list(icm_equities(cur, p, **kw)))
            continue
        # `suivants` = les vivants rangés par distance de SIÈGE au bouton, le
        # bouton EXCLU. La distance est prise modulo le nombre de sièges et non
        # modulo le nombre de vivants : les places vides sont sautées sans que
        # l'ancrage dérive, ce que `test_icm_fgs_blindes.py` mesure sur une
        # orbite complète après élimination.
        suivants = sorted(alive, key=lambda i: ((i - btn - 1) % n))
        if len(alive) == 2:
            # HEADS-UP : le bouton EST la small blind (règle universelle, TDA
            # 2022 §33). Appliquer la règle à trois joueurs et plus — « SB =
            # suivant du bouton » — inverse les deux derniers joueurs du
            # tournoi, c'est-à-dire exactement les deux places dont l'écart de
            # gain est le plus grand. Le bouton est ici le premier vivant à
            # partir du bouton INCLUS : quand le bouton meurt, son siège reste
            # l'ancre mais ne peut plus poster.
            sb_i = sorted(alive, key=lambda i: ((i - btn) % n))[0]
            bb_i = suivants[0] if suivants[0] != sb_i else suivants[1]
        else:
            sb_i, bb_i = suivants[0], suivants[1]
        pot = 0.0
        pay_sb = min(cur[sb_i], sb)
        cur[sb_i] -= pay_sb
        pot += pay_sb
        for i in alive:                   # antes (celle de la BB revient au pot
            if i != bb_i:                 # qu'elle ramasse : net nul, omise)
                a = min(cur[i], ante)
                cur[i] -= a
                pot += a
        cur[bb_i] += pot                  # walk : la BB ramasse tout
        # Le bouton avance d'UN SIÈGE VIVANT, toujours. À trois joueurs et plus
        # ce siège est celui de la SB (`suivants[0] == sb_i`), ce qui redonne
        # la règle usuelle « le bouton passe à l'ancienne small blind ». En
        # heads-up c'est celui de la BB, et c'est la seule écriture qui fait
        # alterner les deux joueurs : `btn = sb_i` y figeait le bouton sur le
        # même siège, donc le même joueur payait la SB à toutes les mains.
        btn = suivants[0]
        path.append(list(cur))
        eqs.append(list(icm_equities(cur, p, **kw)))

    equities = np.array(eqs, dtype=np.float64)
    stacks_path = np.array(path, dtype=np.float64)
    return FgsResult(equities, stacks_path, equities - equities[0])


def fgs_bubble_factor(
    stacks: Sequence[float],
    payouts: Sequence[float],
    hero: int,
    villain: int,
    button: int,
    sb: float,
    bb: float,
    ante: float = 0.0,
    n_hands: int | None = None,
    **kw,
) -> tuple[float, float]:
    """(BF statique, BF sur stacks érodés à l'horizon) — l'écart mesure la
    pression des blindes sur la décision présente."""
    static = bubble_factor(stacks, payouts, hero, villain, **kw)
    r = fgs_equities(stacks, payouts, button, sb, bb, ante, n_hands, **kw)
    eroded = r.stacks_path[-1]
    if eroded[hero] <= 0 or eroded[villain] <= 0:
        return static, math.inf
    fgs = bubble_factor(eroded.tolist(), payouts, hero, villain, **kw)
    return static, fgs
