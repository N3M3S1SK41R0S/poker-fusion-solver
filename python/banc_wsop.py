#!/usr/bin/env python
r"""Banc WSOP — chercher les erreurs PROUVABLES des pros des World Series 2023.

    python banc_wsop.py                        le banc complet
    python banc_wsop.py --detail               + le journal décision par décision
    python banc_wsop.py --demonstration        seulement : les détecteurs sont-ils vivants ?
    python banc_wsop.py --payouts "..."        active le volet ICM (gains NON fournis
                                               par le corpus : c'est toi qui les déclares)
    python banc_wsop.py --villain large        autre range supposée par le conseiller
    python banc_wsop.py --corpus CHEMIN        autre racine de fichiers .phh

Durée mesurée sur la machine de développement : ≈ 2 min pour le banc complet
(l'essentiel du coût est l'énumération exacte au flop, ≈ 5 s par décision :
1 081 mains adverses × 990 runouts).

Ce que ce banc cherche, et pourquoi c'est difficile
---------------------------------------------------
Le commanditaire veut savoir si le logiciel « fait mieux » que ce qui a été
joué. La seule formulation honnête de cet objectif est : trouver les décisions
où le CALCUL prouve l'erreur, et la chiffrer.

Le piège, qui fausse toute l'analyse du poker, est d'évaluer une décision par
ce qui s'est passé APRÈS. Le pro couche, on dit pousser, la main s'arrête :
l'alternative n'existe pas. Et connaître toutes les cartes — ce corpus les
donne — rend le piège plus tentant encore, parce qu'une décision mauvaise
AVEC le recul pouvait être juste compte tenu de l'information disponible AU
MOMENT de jouer.

Le banc classe donc chaque décision dans l'une de trois catégories, et c'est
la classification qui fait tout le travail :

**A — ERREUR PROUVABLE SANS LE RECUL.** La décision est mauvaise quelles que
soient les cartes adverses. C'est la seule catégorie où l'on peut dire « le
pro s'est trompé ». Quatre critères, de force décroissante, jamais fondus
ensemble dans le rapport :

  A1  *suivi réfuté sans aucune hypothèse* — même contre la main adverse la
      PLUS favorable au joueur parmi les 1 081 (ou 1 035, ou 990) mains
      encore possibles, payer perd des jetons. Aucune range n'est supposée :
      la borne porte sur toutes les mains possibles une par une.
      **Ce critère est structurellement inatteignable à une vraie table**, et
      le banc le démontre plutôt que de le taire (section 2 bis) : la cote du
      pot ne peut jamais exiger plus de 50 % d'équité, et il existe presque
      toujours une main adverse qui rend au moins 50 % au payeur. Il est
      implémenté, testé, vivant — et il ne se déclenchera pas.
  A2  *couché réfuté sans aucune hypothèse* — même contre la main adverse la
      MOINS favorable, suivre bat coucher. Exige en plus que le suivi FERME
      le coup, faute de quoi les rues suivantes pourraient renverser le
      calcul. En pratique, cela revient à coucher une main qui ne peut pas
      perdre : c'est rare, mais ce n'est pas impossible.
  A3  *suivi réfuté contre la range la plus large plausible* — l'équité
      contre une range uniforme (les 1 081 mains à poids égal) reste sous la
      cote. Une hypothèse, nommée : que la range du vilain ne soit pas, en
      moyenne, PLUS FAIBLE qu'une main tirée au hasard. Le sens du calcul est
      sûr : resserrer une range ne peut que baisser l'équité du payeur.
  A4  *action strictement dominée* — se coucher alors que checker est
      gratuit. Zéro hypothèse, zéro calcul.

  Volet ICM : le corpus **ne contient pas les gains**. Sans eux, aucune
  violation ICM ne se calcule, et le banc le dit plutôt que d'inventer une
  structure. Deux choses restent vraies sans les gains, et le rapport les
  chiffre : (i) le bubble factor mesuré est ≥ 1 sur toutes les configurations
  de tapis réelles de ce corpus croisées avec une grille de structures
  décroissantes, donc un suivi réfuté en jetons (A1, A3) l'est *a fortiori*
  en $EV ; (ii) pour un couché réfuté en jetons (A2), l'ICM peut au contraire
  l'excuser — le rapport donne alors le bubble factor SEUIL au-delà duquel le
  couché redevient correct, et laisse le lecteur juger s'il est plausible.

**B — ÉCART AVEC LE RECUL SEUL.** Notre conseiller aurait joué autrement, et
avec les cartes adverses connues cela aurait rapporté davantage. **Ce n'est
PAS une erreur du pro.** C'est le sophisme du résultat, et le rapport le
nomme comme tel à chaque ligne.

**C — DÉSACCORD SANS PREUVE.** Notre conseiller dit autre chose, ni l'un ni
l'autre ne se prouve. À compter, pas à interpréter.

Ce que le banc NE fait PAS — à lire avant de citer un chiffre
-------------------------------------------------------------
1. **L'échantillon est BIAISÉ.** 83 mains d'un Poker Players Championship à
   50 000 $, table finale de jour 5, tapis profonds, mains retenues pour la
   diffusion. Aucun taux de ce rapport n'est représentatif du jeu
   professionnel, et le banc n'en publie aucun comme tel.
2. **65 des 83 mains ne sont pas du hold'em** — c'est un championnat *mixte*
   (Omaha hi-lo, stud, razz, triple draw). Elles sont refusées par le lecteur,
   pas mal lues. Il reste 18 mains : 11 no-limit, 7 limit.
3. **Aucun critère de TAILLE de mise n'est implémenté**, et ce n'est pas un
   oubli. « Une mise qu'aucune range de valeur ne justifie » suppose de
   connaître qui paie et qui se couche — donc la stratégie adverse. Il n'en
   existe aucune forme sans hypothèse, et le banc préfère ne rien dire que
   dire quelque chose d'invérifiable. Ce qu'il faudrait pour l'obtenir : une
   range adverse mesurée sur ces joueurs, ce que 18 mains ne donnent pas.
4. **Le conseiller est souvent hors de son domaine déclaré** sur ce corpus
   (préflop face à une relance, postflop multiway, face à un tapis adverse).
   Ces spots sont comptés comme des REFUS, pas comme des désaccords : le
   nombre de décisions où le conseiller refuse de conclure est un livrable à
   part entière, aussi important que les autres.
5. **Les bornes préflop sont Monte-Carlo**, pas exactes (matrice 169×169,
   3 000 tirages par case). Une tolérance déclarée les protège
   (:data:`TOLERANCE_PREFLOP`), et le rapport mesure l'écart de la matrice à
   un calcul plus précis. Les bornes postflop, elles, sont EXACTES par
   énumération complète.

Rejouabilité
------------
Aucun aléa non semé : ordre de lecture stable (:func:`iter_phh_files`),
énumération exacte dès le flop, Monte-Carlo semé préflop. Deux exécutions
rendent les mêmes chiffres.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from banc_corpus_pluribus import (  # noqa: E402
    _moyenne_ic,
    intention_conseiller,
    intention_joueur,
)
from pfs.analysis.spot_advisor import Spot, advise, parse_cards  # noqa: E402
from pfs.core.equity import evaluate7  # noqa: E402
from pfs.core.icm import bubble_factor, icm_required_equity  # noqa: E402
from pfs.core.range_model import COMBO_TO_GROUP, Range, combo_index  # noqa: E402
from pfs.data.hand_history import ActionType, ParsedHand, Street  # noqa: E402
from pfs.data.phh import (  # noqa: E402
    Decision,
    PhhError,
    iter_decisions,
    iter_phh_files,
    parse_phh_file,
    phh_players,
)
from pfs.solver.pushfold import equity_matrix_169  # noqa: E402

CORPUS_DEFAUT = Path(
    r"C:\Users\pierr\Documents\POKERFUSIONSOLVER\corpus\phh-dataset\data\wsop\2023")

N_CARTES = 52

TOLERANCE_PREFLOP = 0.03
"""Marge de sécurité, en points d'équité, sur les bornes PRÉFLOP.

Préflop il n'existe pas d'énumération abordable : hero et vilain fixés, il
reste C(48,5) = 1 712 304 boards, et il faudrait le faire pour chacune des
1 225 mains adverses. Le banc lit donc les bornes dans la matrice 169×169 de
``solver.pushfold``, qui est du Monte-Carlo à 3 000 tirages par case (erreur-
type au pire √(0,25/3000) = 0,91 point) et qui raisonne par GROUPE, donc sans
les blockers des cartes du héros.

La tolérance est posée à 3 points, soit 3,3 fois l'erreur-type la plus
défavorable, et le rapport MESURE l'écart réel matrice ↔ calcul précis
(section « fiabilité des bornes préflop »). Elle joue toujours dans le sens
qui protège le pro : elle est ajoutée à la borne haute avant d'accuser un
suivi, retranchée à la borne basse avant d'accuser un couché.
"""

MC_PREFLOP = 200_000
"""Tirages du Monte-Carlo préflop (équité contre une range uniforme, et
contre la main adverse réelle). Erreur-type ≈ √(0,25/200000) = 0,11 point.
Semé, donc rejouable au chiffre près."""

BLOC_COMBOS = 64
"""Combos adverses traités d'un coup dans l'énumération exacte.

Compromis mémoire/vitesse mesuré au flop : 64 combos × 990 runouts ≈ 63 000
lignes de 7 cartes par paquet, soit ≈ 3,5 Mo par tableau. Monter le bloc ne
gagne plus rien et fait exploser la mémoire (le produit cartésien complet
demanderait 1,07 million de lignes, deux fois).
"""

GRILLE_GAINS_TEST = (
    (100.0,),
    (65.0, 35.0),
    (50.0, 30.0, 20.0),
    (40.0, 25.0, 18.0, 10.0, 7.0),
    (36.0, 22.0, 15.0, 11.0, 9.0),
)
"""Structures de gains décroissantes servant UNIQUEMENT à mesurer que le
bubble factor reste ≥ 1 sur les tapis réels de ce corpus.

Ce ne sont PAS les gains des WSOP 2023 — le corpus ne les donne pas. Elles
vont du winner-take-all (qui rend les jetons linéaires, BF = 1 exactement, et
sert de témoin bas) à une structure plate à cinq places, qui est le régime où
la pression ICM est la plus forte à cinq joueurs.
"""


# ═══════════════════════════════════════════════════════════════════════════
# ÉQUITÉ : BORNES EXACTES SUR TOUTES LES MAINS ADVERSES POSSIBLES
# ═══════════════════════════════════════════════════════════════════════════


def _forces(cartes7: np.ndarray) -> np.ndarray:
    """Force de mains de 7 cartes, en entiers comparables (plus grand = mieux)."""
    return evaluate7(cartes7).astype(np.int64)


def _runouts(restantes: list[int], besoin: int) -> np.ndarray:
    """Toutes les façons de compléter le board avec ``besoin`` cartes."""
    if besoin <= 0:
        return np.empty((1, 0), dtype=np.int64)
    return np.array(list(itertools.combinations(restantes, besoin)), dtype=np.int64)


def equites_par_combo(hero: list[int], board: list[int],
                      bloc: int = BLOC_COMBOS) -> tuple[np.ndarray, np.ndarray]:
    """Équité EXACTE du héros contre **chaque** main adverse possible.

    Parameters
    ----------
    hero : list of int
        Les deux cartes du héros (indices 0-51).
    board : list of int
        Les 3, 4 ou 5 cartes communes visibles. Préflop (0 carte) n'est PAS
        traité ici : l'énumération y coûterait 1,7 million de boards par main
        adverse, et :func:`bornes_equite` bascule sur la matrice 169×169.
    bloc : int
        Nombre de mains adverses traitées par paquet (mémoire).

    Returns
    -------
    combos : ndarray of shape (N, 2)
        Les mains adverses possibles, dans l'ordre lexicographique des indices
        de carte. N vaut 1 081 au flop, 1 035 au turn, 990 à la river.
    equites : ndarray of shape (N,)
        L'équité du héros contre chacune, partage compté pour moitié. Exacte :
        tous les runouts sont énumérés, pas échantillonnés.

    Notes
    -----
    C'est la brique de la catégorie A. Le maximum de ``equites`` est la borne
    la plus favorable au payeur qui existe **sans supposer quoi que ce soit**
    de la stratégie adverse ; son minimum est la borne la plus défavorable.
    Une décision réfutée par ces bornes-là l'est pour toutes les ranges à la
    fois, puisqu'une range n'est qu'une moyenne pondérée de ces mains.

    Examples
    --------
    Board river où le héros a la roue et le vilain ne peut faire mieux
    qu'égaler avec la même quinte : l'équité vaut 1 contre presque tout, et
    ½ contre les mains qui partagent.

    >>> from pfs.analysis.spot_advisor import parse_cards
    >>> c, eq = equites_par_combo(parse_cards("Ah Kd"),
    ...                           parse_cards("Qs Js Ts 3c 2d"))
    >>> float(eq.max()), int(c.shape[0])
    (1.0, 990)
    """
    if len(board) not in (3, 4, 5):
        raise ValueError(
            f"énumération exacte impossible sur un board de {len(board)} "
            "cartes : préflop, il faudrait 1,7 million de runouts par main "
            "adverse. Passer par bornes_equite().")
    mortes = set(hero) | set(board)
    if len(mortes) != len(hero) + len(board):
        raise ValueError("collision de cartes entre la main du héros et le board.")
    restantes = [c for c in range(N_CARTES) if c not in mortes]
    combos = np.array(list(itertools.combinations(restantes, 2)), dtype=np.int64)
    besoin = 5 - len(board)
    ro = _runouts(restantes, besoin)
    board_a = np.asarray(board, dtype=np.int64)
    hero_a = np.asarray(hero, dtype=np.int64)
    eq = np.empty(combos.shape[0], dtype=np.float64)

    for debut in range(0, combos.shape[0], bloc):
        paquet = combos[debut:debut + bloc]
        n = paquet.shape[0]
        if besoin == 0:
            plateau = np.tile(board_a, (n, 1))
            h = _forces(np.concatenate([np.tile(hero_a, (n, 1)), plateau], axis=1))
            v = _forces(np.concatenate([paquet, plateau], axis=1))
            eq[debut:debut + n] = (h > v) + 0.5 * (h == v)
            continue
        nr = ro.shape[0]
        i_ro = np.repeat(np.arange(nr), n)
        i_co = np.tile(np.arange(n), nr)
        R, C = ro[i_ro], paquet[i_co]
        # Un runout ne peut pas contenir une carte que le vilain tient déjà.
        garde = ~(R[:, :, None] == C[:, None, :]).any(axis=(1, 2))
        R, C, i_co = R[garde], C[garde], i_co[garde]
        m = R.shape[0]
        plateau = np.concatenate([np.tile(board_a, (m, 1)), R], axis=1)
        h = _forces(np.concatenate([np.tile(hero_a, (m, 1)), plateau], axis=1))
        v = _forces(np.concatenate([C, plateau], axis=1))
        valeur = (h > v) + 0.5 * (h == v)
        eq[debut:debut + n] = (np.bincount(i_co, weights=valeur, minlength=n)
                               / np.bincount(i_co, minlength=n))
    return combos, eq


def equite_vs_main(hero: list[int], vilain: list[int], board: list[int],
                   futures: tuple[int, ...] = ()) -> float:
    """Équité EXACTE du héros contre **une** main adverse connue.

    ``futures`` sont les cartes que le coup a réellement distribuées après ce
    point de décision : les compter, c'est le recul assumé de la catégorie B.
    Ce qui n'a jamais été distribué est énuméré uniformément.

    >>> from pfs.analysis.spot_advisor import parse_cards
    >>> equite_vs_main(parse_cards("Ah Ad"), parse_cards("Kh Kd"),
    ...                parse_cards("2c 7d 9s 3h 4c"))
    1.0
    """
    plateau = list(board) + [c for c in futures if c not in board]
    mortes = set(hero) | set(vilain) | set(plateau)
    if len(mortes) != len(hero) + len(vilain) + len(plateau):
        raise ValueError("collision de cartes entre héros, vilain et board.")
    besoin = 5 - len(plateau)
    if besoin < 0:
        raise ValueError(f"board de {len(plateau)} cartes : plus de 5.")
    restantes = [c for c in range(N_CARTES) if c not in mortes]
    ro = _runouts(restantes, besoin)
    n = ro.shape[0]
    base = np.concatenate([np.tile(np.asarray(plateau, dtype=np.int64), (n, 1)), ro],
                          axis=1)
    h = _forces(np.concatenate([np.tile(np.asarray(hero, dtype=np.int64), (n, 1)),
                                base], axis=1))
    v = _forces(np.concatenate([np.tile(np.asarray(vilain, dtype=np.int64), (n, 1)),
                                base], axis=1))
    return float(np.mean((h > v) + 0.5 * (h == v)))


_MATRICE_169: np.ndarray | None = None


def _matrice() -> np.ndarray:
    """Matrice 169×169 d'équités préflop, mémoïsée (cache disque du solveur)."""
    global _MATRICE_169
    if _MATRICE_169 is None:
        _MATRICE_169 = equity_matrix_169()
    return _MATRICE_169


@dataclass(frozen=True, slots=True)
class Bornes:
    """Bornes d'équité du héros au moment de décider.

    Attributes
    ----------
    basse, haute : float
        Équité contre la main adverse la moins / la plus favorable au héros,
        **tolérance déjà appliquée** dans le sens qui protège le joueur.
    uniforme : float
        Équité contre une range uniforme (toutes les mains possibles à poids
        égal), tolérance déjà appliquée vers le haut.
    exacte : bool
        Vrai si les bornes viennent d'une énumération complète (postflop),
        faux si de la matrice 169×169 Monte-Carlo (préflop).
    n_mains : int
        Nombre de mains adverses considérées.
    tolerance : float
        Marge appliquée (0 postflop).
    brutes : tuple of float
        (basse, haute, uniforme) AVANT tolérance — pour que le rapport puisse
        montrer ce que la marge a coûté.
    """

    basse: float
    haute: float
    uniforme: float
    exacte: bool
    n_mains: int
    tolerance: float
    brutes: tuple[float, float, float]


def bornes_equite(hero: list[int], board: list[int]) -> Bornes:
    """Bornes d'équité, exactes postflop et Monte-Carlo tolérancées préflop.

    Postflop : énumération complète (:func:`equites_par_combo`), tolérance
    nulle — il n'y a rien à tolérancer, le calcul est exact.

    Préflop : la borne basse est le minimum de la ligne du héros dans la
    matrice 169×169, la haute son maximum, l'uniforme un Monte-Carlo semé
    contre la range pleine. Les trois reçoivent :data:`TOLERANCE_PREFLOP`
    dans le sens qui rend une accusation PLUS difficile.
    """
    if board:
        _, eq = equites_par_combo(hero, board)
        b, h, u = float(eq.min()), float(eq.max()), float(eq.mean())
        return Bornes(b, h, u, True, int(eq.shape[0]), 0.0, (b, h, u))

    from pfs.core.equity import equity_vs_range  # import local : coût d'import

    ligne = _matrice()[int(COMBO_TO_GROUP[combo_index(hero[0], hero[1])])]
    b, h = float(ligne.min()), float(ligne.max())
    u = float(equity_vs_range(hero, Range.full(), (), n_sims=MC_PREFLOP,
                              seed=0).equity)
    t = TOLERANCE_PREFLOP
    return Bornes(max(0.0, b - t), min(1.0, h + t), min(1.0, u + t), False,
                  int(ligne.shape[0]), t, (b, h, u))


# ═══════════════════════════════════════════════════════════════════════════
# COTES, EV, ET CE QUI FERME LE COUP
# ═══════════════════════════════════════════════════════════════════════════


def alpha_requise(pot_gagnable: float, a_payer: float) -> float:
    r"""Équité minimale pour que suivre batte coucher, en jetons.

    .. math:: \alpha = \frac{c}{P + c}

    où :math:`P` est le pot **déjà gagnable** (mise adverse comprise, part
    non couvrable exclue) et :math:`c` ce qu'il reste à payer.

    Attention : ce n'est PAS la formule ``b/(P+2b)`` de
    :func:`~pfs.core.bluffcatch.required_equity`, qui attend un pot mesuré
    AVANT la mise adverse. Les deux coïncident quand on leur donne le pot
    qu'elles demandent ; les confondre sous-estime la cote et donc innocente
    à tort un suivi.

    **Cette formule ne peut jamais dépasser ½**, et c'est ce qui rend le
    critère A1 inatteignable. Pour que le héros doive ``c``, il faut qu'un
    adversaire ait engagé au moins ``engagé_héros + c`` ; le pot gagnable
    vaut donc au moins ``c``, d'où :math:`\alpha = c/(P+c) \le c/(c+c) = ½`.
    L'égalité demanderait un pot gagnable réduit à la seule mise adverse,
    c'est-à-dire un héros n'ayant rien engagé et aucun autre joueur : la
    borne n'est pas atteinte à une vraie table.

    >>> round(alpha_requise(175.0, 75.0), 4)
    0.3
    >>> alpha_requise(75.0, 75.0)     # le pot gagnable minimal possible
    0.5
    """
    if a_payer <= 0.0:
        raise ValueError("il n'y a rien à payer : la cote n'est pas définie.")
    if pot_gagnable < 0.0:
        raise ValueError("pot gagnable négatif.")
    return a_payer / (pot_gagnable + a_payer)


def ev_suivre(equite: float, pot_gagnable: float, a_payer: float) -> float:
    r"""EV(suivre) − EV(coucher), en jetons.

    .. math:: \Delta = e \cdot P - (1 - e) \cdot c

    Coucher vaut zéro par construction (les jetons déjà engagés ne sont plus
    au joueur), donc ce nombre EST le prix de la décision.

    >>> ev_suivre(0.5, 175.0, 75.0)
    50.0
    """
    return equite * pot_gagnable - (1.0 - equite) * a_payer


def ferme_le_coup(d: Decision) -> bool:
    """Vrai si suivre clôt définitivement l'enjeu du héros dans ce coup.

    C'est la condition sans laquelle **aucun couché ne peut être réfuté** :
    si le coup continue, l'adversaire peut remiser sur les rues suivantes et
    l'équité brute du héros n'est plus ce qu'il réalise. Trois cas suffisent
    et se lisent tous dans la décision :

    * le suivi met le héros à tapis (``to_call >= stack``) — il ne paiera
      plus jamais rien ;
    * plus aucun adversaire n'a de jetons derrière (``stack_effectif == 0``)
      — personne ne peut miser ;
    * river en tête-à-tête (``actifs == 2``) — suivre ferme le tour d'enchères
      et il n'y a pas de rue après.

    Tout le reste est déclaré ouvert, quitte à refuser de conclure : c'est le
    sens de la prudence de ce banc.

    >>> from pfs.data.hand_history import ActionType, Street
    >>> d = Decision("h", 0, Street.RIVER, "p", 1, "BB", (), (), 100.0, 100.0,
    ...              50.0, 500.0, 500.0, 0.0, 2, 1, ActionType.FOLD, 0.0)
    >>> ferme_le_coup(d)
    True
    """
    if d.to_call >= d.stack - 1e-9:
        return True
    if d.stack_effectif <= 1e-9:
        return True
    return d.street is Street.RIVER and d.actifs == 2


# ═══════════════════════════════════════════════════════════════════════════
# CATÉGORIE A — LA PREUVE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PreuveA:
    """Une erreur prouvée, avec de quoi la refaire à la main.

    Attributes
    ----------
    critere : str
        « A1 », « A2 », « A3 » ou « A4 ».
    action_correcte : str
        Ce qu'il fallait faire.
    cout_jetons, cout_bb : float
        Ce que l'erreur coûte, en jetons puis en grosses blindes. Toujours
        positif : c'est un coût.
    equite_utilisee : float
        L'équité qui porte la preuve (borne haute pour un suivi, basse pour
        un couché).
    alpha : float
        L'équité qu'il aurait fallu.
    raisonnement : str
        Le calcul en toutes lettres, chiffres compris.
    hypothese : str
        Vide pour les critères sans hypothèse (A1, A2, A4) ; la nommer pour A3.
    """

    critere: str
    action_correcte: str
    cout_jetons: float
    cout_bb: float
    equite_utilisee: float
    alpha: float
    raisonnement: str
    hypothese: str = ""


def prouver(d: Decision, bornes: Bornes | None, big_blind: float) -> PreuveA | None:
    """Cherche une erreur PROUVABLE dans une décision. ``None`` = rien à dire.

    L'ordre d'examen va du critère le plus fort au plus faible, et le premier
    qui conclut arrête : une décision n'est comptée qu'une fois, sous le
    critère le plus solide qui la réfute.

    Notes
    -----
    Aucun de ces critères ne regarde ce qui s'est passé ensuite. Le seul
    argument utilisé est : « avec l'information disponible à cet instant, une
    autre action rapporte davantage quelles que soient les cartes adverses ».
    """
    bb = big_blind if big_blind > 0 else 1.0

    # ── A4 : action strictement dominée ──────────────────────────────────
    if d.action is ActionType.FOLD and d.to_call <= 0.0:
        equite = bornes.basse if bornes is not None else 0.0
        return PreuveA(
            critere="A4",
            action_correcte="CHECKER",
            cout_jetons=equite * d.pot_gagnable,
            cout_bb=equite * d.pot_gagnable / bb,
            equite_utilisee=equite,
            alpha=0.0,
            raisonnement=(
                f"Rien à payer (à suivre : 0) et le joueur se couche. Checker "
                f"coûte zéro et conserve un pot de {d.pot_gagnable:,.0f} jetons. "
                f"Coucher est strictement dominé : il n'existe aucune main "
                f"adverse, aucune range et aucune structure de gains sous "
                f"laquelle abandonner gratuitement un pot rapporte davantage "
                f"que de le regarder gratuitement."),
        )

    if bornes is None or d.to_call <= 0.0:
        return None
    alpha = alpha_requise(d.pot_gagnable, d.to_call)

    # ── A1 : suivi réfuté sans aucune hypothèse ──────────────────────────
    if d.action in (ActionType.CALL, ActionType.ALLIN) and d.montant > 0.0:
        delta = ev_suivre(bornes.haute, d.pot_gagnable, d.to_call)
        if bornes.haute < alpha and delta < 0.0:
            return PreuveA(
                critere="A1",
                action_correcte="COUCHER",
                cout_jetons=-delta,
                cout_bb=-delta / bb,
                equite_utilisee=bornes.haute,
                alpha=alpha,
                raisonnement=(
                    f"Payer {d.to_call:,.0f} pour un pot gagnable de "
                    f"{d.pot_gagnable:,.0f} demande {alpha * 100:.1f} % d'équité. "
                    f"Contre la main adverse LA PLUS FAVORABLE des "
                    f"{bornes.n_mains} encore possibles, cette main n'en "
                    f"réalise que {bornes.haute * 100:.1f} %"
                    + ("" if bornes.exacte else
                       f" (borne Monte-Carlo, tolérance +{bornes.tolerance * 100:.0f} pt)")
                    + f". Toute range étant une moyenne pondérée de ces mains, "
                    f"aucune ne peut relever l'équité au-dessus de ce maximum : "
                    f"suivre perd {-delta:,.0f} jetons quoi que tienne "
                    f"l'adversaire."),
            )

    # ── A2 : couché réfuté sans aucune hypothèse ─────────────────────────
    if d.action is ActionType.FOLD and ferme_le_coup(d):
        delta = ev_suivre(bornes.basse, d.pot_gagnable, d.to_call)
        if bornes.basse > alpha and delta > 0.0:
            return PreuveA(
                critere="A2",
                action_correcte="SUIVRE",
                cout_jetons=delta,
                cout_bb=delta / bb,
                equite_utilisee=bornes.basse,
                alpha=alpha,
                raisonnement=(
                    f"Payer {d.to_call:,.0f} pour un pot gagnable de "
                    f"{d.pot_gagnable:,.0f} demande {alpha * 100:.1f} % d'équité. "
                    f"Contre la main adverse LA MOINS FAVORABLE des "
                    f"{bornes.n_mains} encore possibles, cette main en réalise "
                    f"encore {bornes.basse * 100:.1f} %"
                    + ("" if bornes.exacte else
                       f" (borne Monte-Carlo, tolérance −{bornes.tolerance * 100:.0f} pt)")
                    + f", et le suivi ferme le coup — aucune rue ne peut plus "
                    f"renverser le calcul. Coucher jette {delta:,.0f} jetons "
                    f"quoi que tienne l'adversaire."),
            )

    # ── A3 : suivi réfuté contre la range la plus large plausible ────────
    if d.action in (ActionType.CALL, ActionType.ALLIN) and d.montant > 0.0:
        delta = ev_suivre(bornes.uniforme, d.pot_gagnable, d.to_call)
        if bornes.uniforme < alpha and delta < 0.0:
            return PreuveA(
                critere="A3",
                action_correcte="COUCHER",
                cout_jetons=-delta,
                cout_bb=-delta / bb,
                equite_utilisee=bornes.uniforme,
                alpha=alpha,
                raisonnement=(
                    f"Payer {d.to_call:,.0f} pour un pot gagnable de "
                    f"{d.pot_gagnable:,.0f} demande {alpha * 100:.1f} % d'équité. "
                    f"Contre une range UNIFORME — les {bornes.n_mains} mains "
                    f"possibles à poids égal, la plus large qui existe — cette "
                    f"main n'en réalise que {bornes.uniforme * 100:.1f} %. "
                    f"Resserrer la range ne peut que baisser ce chiffre : "
                    f"suivre perd {-delta:,.0f} jetons."),
                hypothese=(
                    "que la range de l'adversaire ne soit pas, en moyenne, "
                    "PLUS FAIBLE qu'une main tirée au hasard — vrai de toute "
                    "range de valeur, non démontré pour une range très "
                    "polarisée."),
            )
    return None


def bubble_factor_seuil(pot_gagnable: float, a_payer: float, equite: float) -> float:
    r"""Bubble factor à partir duquel un couché redevient correct.

    :func:`~pfs.core.icm.icm_required_equity` attend un pot mesuré AVANT la
    mise ; on résout donc, avec :math:`P' = P - c` :

    .. math::
        \frac{BF\,c}{P' + c + BF\,c} = e
        \quad\Longleftrightarrow\quad
        BF^\star = \frac{e\,(P' + c)}{c\,(1 - e)} = \frac{e\,P}{c\,(1 - e)}

    où :math:`P` est le pot **gagnable** (mise adverse comprise), c'est-à-dire
    l'argument attendu ici.

    Sert à la catégorie A2 : un couché réfuté en jetons n'est une erreur en
    tournoi que si le bubble factor réel est **inférieur** à ce seuil. Rend
    ``inf`` quand l'équité vaut 1 : rien ne peut alors excuser le couché.

    >>> round(bubble_factor_seuil(250.0, 75.0, 0.5), 4)   # P' = 175, c = 75
    3.3333
    >>> from pfs.core.icm import icm_required_equity
    >>> round(icm_required_equity(175.0, 75.0, 3.3333333), 6)  # revient à e
    0.5
    """
    if a_payer <= 0.0:
        raise ValueError("rien à payer : le seuil n'a pas de sens.")
    if not (0.0 <= equite <= 1.0):
        raise ValueError("équité hors de [0, 1].")
    if equite >= 1.0:
        return math.inf
    return equite * pot_gagnable / (a_payer * (1.0 - equite))


# ═══════════════════════════════════════════════════════════════════════════
# CATÉGORIE B — LE RECUL, ASSUMÉ COMME TEL
# ═══════════════════════════════════════════════════════════════════════════


def valeur_avec_cartes_connues(d: Decision, main: ParsedHand) -> tuple[float, bool] | None:
    """EV(suivre) − EV(coucher) calculée AVEC les cartes adverses, en jetons.

    C'est le nombre du sophisme du résultat : il utilise une information que
    le joueur n'avait pas. Le banc le calcule quand même, parce que c'est lui
    qui sépare la catégorie B de la catégorie C — mais il ne le présente
    jamais comme une erreur.

    Returns
    -------
    (valeur, ferme) or None
        ``valeur`` en jetons ; ``ferme`` dit si le suivi fermait le coup —
        sinon le chiffre ignore les mises des rues suivantes et n'est qu'un
        ordre de grandeur. ``None`` quand le calcul demanderait une hypothèse :
        plus d'un adversaire encore en jeu, ou cartes adverses non déclarées
        par le fichier.
    """
    if d.to_call <= 0.0 or len(d.cards) != 2:
        return None
    couches = {a.player for a in main.actions if a.action is ActionType.FOLD}
    # Les joueurs couchés APRÈS cette décision sont encore en jeu ici ; on ne
    # peut donc pas se fier au seul jeu de couchés de fin de main. `actifs`
    # porte l'information juste à l'instant de la décision.
    if d.actifs != 2:
        return None
    adverses = [s for s in main.seats
                if s.player != d.player and s.player not in couches]
    if len(adverses) != 1 or len(adverses[0].cards) != 2:
        return None
    hero = parse_cards(" ".join(d.cards))
    vilain = parse_cards(" ".join(adverses[0].cards))
    board = parse_cards(" ".join(d.board))
    futures = tuple(parse_cards(" ".join(main.board))[len(board):])
    if set(hero) & set(vilain):
        return None
    eq = equite_vs_main(hero, vilain, board, futures)
    return ev_suivre(eq, d.pot_gagnable, d.to_call), ferme_le_coup(d)


# ═══════════════════════════════════════════════════════════════════════════
# LE CONSEILLER FACE AU SPOT
# ═══════════════════════════════════════════════════════════════════════════


def hors_domaine(d: Decision) -> tuple[str, ...]:
    """TOUTES les hypothèses du conseiller que ce spot viole. Vide = domaine OK.

    Le conseiller ANNONCE ses hypothèses ; les lui violer et compter sa
    réponse comme un avis serait fabriquer un désaccord. Trois violations se
    lisent dans la décision :

    * **préflop après une relance** — ``_advise_preflop_deep`` déclare
      « suppose que personne n'a ouvert devant » et rend pourtant un verdict
      d'OUVERTURE ;
    * **postflop multiway** — tout le postflop du conseiller suppose UN
      adversaire ;
    * **face à un tapis adverse** — le régime qui conviendrait (range de call
      d'équilibre face à un jam) n'existe pas : le conseiller n'a que la chart
      d'ouverture, qui répond à une autre question.

    La fonction rend la LISTE et non le premier motif : un même spot cumule
    souvent deux violations (préflop relancé ET face à un tapis), et n'en
    compter qu'une ferait disparaître un manque du logiciel du tableau des
    motifs — c'est arrivé, d'où cette signature.
    """
    motifs = []
    if d.street is Street.PREFLOP and d.relances_avant >= 1:
        motifs.append("préflop face à une relance (la chart suppose un pot "
                      "non ouvert)")
    if d.street is not Street.PREFLOP and d.actifs > 2:
        motifs.append("postflop multiway (le modèle suppose un adversaire)")
    if d.stack_effectif <= 1e-9 and d.to_call > 0.0:
        motifs.append("face à un tapis adverse (aucun régime « payer un jam »)")
    return tuple(motifs)


def construire_spot(d: Decision, main: ParsedHand, villain: str) -> Spot:
    """Le ``Spot`` que le conseiller aurait reçu d'une capture d'écran.

    ``pot`` reçoit ``pot_gagnable − to_call`` et non ``pot_gagnable`` :
    :class:`~pfs.analysis.spot_advisor.Spot` documente son champ ``pot`` comme
    « pot AVANT la mise adverse », et c'est bien ce que ses formules attendent
    (``required_equity(P, b) = b/(P+2b)`` ne redonne la vraie cote que si
    ``P`` exclut la mise). Lui passer le pot mise comprise sous-estimerait
    l'équité requise et innocenterait des suivis à tort.
    """
    return Spot(
        hero=" ".join(d.cards),
        board=" ".join(d.board),
        pot=max(0.0, d.pot_gagnable - d.to_call),
        bet=d.to_call,
        stack=d.stack_effectif,
        big_blind=main.big_blind,
        position=d.position,
        villain=villain,
        players=d.actifs,
        solver_budget_s=0.0,
    )


def accord(d: Decision, vu: str, fait: str) -> bool | None:
    """Accord sur l'axe binaire que le conseiller sait trancher.

    Préflop et face à une mise, l'axe est jouer / passer — le conseiller n'a
    pas de régime pour distinguer suivre de relancer. Sans mise, l'axe est
    miser / checker. ``None`` : il ne tranche pas.
    """
    if vu in ("MIXTE", "INDECIS", "REFUS") or vu.startswith("?"):
        return None
    if d.street is Street.PREFLOP or d.to_call > 0.0:
        return (vu != "PASSER") == (fait != "PASSER")
    return (vu == "AGRESSER") == (fait == "AGRESSER")


# ═══════════════════════════════════════════════════════════════════════════
# REJEU
# ═══════════════════════════════════════════════════════════════════════════


_RE_VARIANTE = re.compile(r"^variant\s*=\s*['\"]([^'\"]+)['\"]", re.M)


def variante(main: ParsedHand) -> str:
    """Variante déclarée par le fichier (« NT », « FT »)."""
    m = _RE_VARIANTE.search(main.raw or "")
    return m.group(1) if m else "?"


def tapis_avant_chaque_decision(main: ParsedHand) -> list[dict[str, float]]:
    """Tapis restants de TOUS les joueurs avant chaque décision volontaire.

    ``iter_decisions`` ne rend que le tapis du joueur qui parle ; l'ICM a
    besoin de la table entière. Ce rejeu parallèle applique la même
    comptabilité (les POST débitent, les FOLD ne débitent rien) et le banc
    **vérifie** ensuite que le tapis du joueur qui parle coïncide avec celui
    de ``Decision.stack`` — si les deux lectures divergeaient, le volet ICM
    serait faux sans prévenir.
    """
    reste = {s.player: s.stack for s in main.seats}
    out: list[dict[str, float]] = []
    for a in main.actions:
        if a.action is ActionType.POST:
            reste[a.player] = reste.get(a.player, 0.0) - a.amount
            continue
        out.append(dict(reste))
        if a.action is not ActionType.FOLD:
            reste[a.player] = reste.get(a.player, 0.0) - a.amount
    return out


@dataclass
class Ligne:
    """Une décision analysée, telle que le rapport la lit."""

    fichier: str
    variante: str
    ordre: int
    joueur: str
    rue: str
    position: str
    cartes: str
    board: str
    action: str
    pot_gagnable: float
    a_payer: float
    big_blind: float
    ferme: bool = False
    categorie: str = "?"
    preuve: PreuveA | None = None
    conseil: str = ""
    conseil_action: str = ""
    recul: float | None = None
    gain_conseil: float | None = None
    recul_ferme: bool = False
    motif_refus: str = ""
    motifs_refus: tuple[str, ...] = ()
    alpha: float | None = None
    bornes: Bornes | None = None


@dataclass
class Resultat:
    mains_lues: int = 0
    mains_refusees: int = 0
    refus_variantes: Counter = field(default_factory=Counter)
    variantes_lues: Counter = field(default_factory=Counter)
    decisions: int = 0
    non_analysables: int = 0
    lignes: list[Ligne] = field(default_factory=list)
    moteurs: Counter = field(default_factory=Counter)
    incoherences_tapis: int = 0
    tapis_reels: list[tuple[float, ...]] = field(default_factory=list)
    secondes: float = 0.0


def rejouer(corpus: Path, villain: str, bavard: bool) -> Resultat:
    """Rejoue le corpus et classe chaque décision en A, B, C, accord ou refus."""
    res = Resultat()
    depart = time.perf_counter()

    for fichier in iter_phh_files(corpus):
        try:
            main = parse_phh_file(fichier, is_tournament=True)
        except PhhError as exc:
            res.mains_refusees += 1
            msg = str(exc)
            m = re.search(r"«\s*([^»]+?)\s*»", msg)
            res.refus_variantes[m.group(1) if m and msg.startswith("variante")
                                else msg[:60]] += 1
            continue
        except (OSError, UnicodeDecodeError) as exc:
            res.mains_refusees += 1
            res.refus_variantes[f"{type(exc).__name__}"] += 1
            continue
        res.mains_lues += 1
        res.variantes_lues[variante(main)] += 1
        noms = phh_players(fichier.read_text(encoding="utf-8"))
        cles = [s.player for s in main.seats]
        res.tapis_reels.append(tuple(s.stack for s in main.seats))

        tapis = tapis_avant_chaque_decision(main)
        for d, table in zip(iter_decisions(main), tapis, strict=True):
            res.decisions += 1
            if abs(table.get(d.player, 0.0) - d.stack) > 1e-6:
                res.incoherences_tapis += 1
            ligne = Ligne(
                fichier=fichier.stem,
                variante=variante(main),
                ordre=d.ordre,
                joueur=noms[cles.index(d.player)] if d.player in cles else "?",
                rue=d.street.value,
                position=d.position,
                cartes="/".join(d.cards),
                board=" ".join(d.board),
                action=d.action.value,
                pot_gagnable=d.pot_gagnable,
                a_payer=d.to_call,
                big_blind=main.big_blind,
                ferme=ferme_le_coup(d),
            )
            if len(d.cards) != 2:
                # Le fichier n'a pas déclaré les cartes (« d dh p3 ???? ») :
                # ni preuve ni conseil ne sont possibles. Le compter comme un
                # accord ou un désaccord serait inventer.
                res.non_analysables += 1
                ligne.categorie = "NON ANALYSABLE"
                ligne.motif_refus = "cartes du joueur non déclarées par le fichier"
                res.lignes.append(ligne)
                continue

            hero = parse_cards(" ".join(d.cards))
            board = parse_cards(" ".join(d.board)) if d.board else []
            bornes = bornes_equite(hero, board) if (d.to_call > 0.0 or
                                                    d.action is ActionType.FOLD) else None
            ligne.bornes = bornes
            if d.to_call > 0.0:
                ligne.alpha = alpha_requise(d.pot_gagnable, d.to_call)

            preuve = prouver(d, bornes, main.big_blind)
            if preuve is not None:
                ligne.categorie = "A"
                ligne.preuve = preuve
                res.lignes.append(ligne)
                continue

            motifs = hors_domaine(d)
            if motifs:
                ligne.categorie = "REFUS"
                ligne.motifs_refus = motifs
                ligne.motif_refus = " + ".join(motifs)
                res.lignes.append(ligne)
                continue
            try:
                conseil = advise(construire_spot(d, main, villain))
            except (ValueError, ZeroDivisionError) as exc:
                ligne.categorie = "REFUS"
                ligne.motif_refus = f"advise() refuse le spot : {exc}"[:100]
                res.lignes.append(ligne)
                continue
            res.moteurs[conseil.regime.split(" ")[0] if conseil.regime else "?"] += 1
            ligne.conseil = conseil.action
            vu = intention_conseiller(conseil)
            fait = intention_joueur(d)
            ligne.conseil_action = f"{vu} / joué {fait}"
            verdict = accord(d, vu, fait)
            if verdict is None:
                ligne.categorie = "REFUS"
                ligne.motif_refus = (
                    "le conseiller ne tranche pas (réponse en fréquence ou "
                    "indifférence déclarée)" if vu != "REFUS" else
                    "le conseiller n'a aucun modèle pour ce spot")
                res.lignes.append(ligne)
                continue
            if verdict:
                ligne.categorie = "ACCORD"
                res.lignes.append(ligne)
                continue

            recul = valeur_avec_cartes_connues(d, main)
            if recul is None:
                ligne.categorie = "C"
                res.lignes.append(ligne)
                continue
            valeur, ferme = recul
            ligne.recul, ligne.recul_ferme = valeur, ferme
            # Le conseil vaut-il mieux AVEC les cartes connues ? Le signe
            # dépend du sens du désaccord : si le conseiller voulait payer là
            # où le joueur a couché, le gain du conseil est +valeur ; s'il
            # voulait coucher là où le joueur a payé, c'est −valeur.
            ligne.gain_conseil = valeur if vu != "PASSER" else -valeur
            ligne.categorie = "B" if ligne.gain_conseil > 0.0 else "C"
            res.lignes.append(ligne)

        if bavard:
            print(f"  … {res.mains_lues} mains lues, {res.decisions} décisions, "
                  f"{time.perf_counter() - depart:.0f} s",
                  file=sys.stderr, flush=True)

    res.secondes = time.perf_counter() - depart
    return res


# ═══════════════════════════════════════════════════════════════════════════
# LES DÉTECTEURS SONT-ILS VIVANTS ?
# ═══════════════════════════════════════════════════════════════════════════


def _decision(action: ActionType, cartes: str, board: str, pot: float,
              a_payer: float, stack: float, eff: float, rue: Street,
              actifs: int = 2) -> Decision:
    """Fabrique une décision synthétique pour la section de démonstration."""
    return Decision(
        hand_id="démonstration", ordre=0, street=rue, player="p", seat=1,
        position="BB", cards=tuple(cartes.split()), board=tuple(board.split()),
        pot=pot, pot_gagnable=pot, to_call=a_payer, stack=stack,
        stack_effectif=eff, engage=0.0, actifs=actifs, relances_avant=1,
        action=action, montant=a_payer if action is not ActionType.FOLD else 0.0)


CAS_DEMONSTRATION: tuple[tuple[str, Decision, str], ...] = (
    ("A1 se déclenche — mais seulement sur un spot IMPOSSIBLE à une vraie "
     "table : payer 900 pour un pot gagnable de 100, soit 90 % d'équité "
     "requise. Quinte royale au tableau, donc 50 % d'équité garantie contre "
     "n'importe quelle main. À une vraie table la cote ne dépasse jamais "
     "50 % (voir alpha_requise) : ce détecteur est vivant et inatteignable.",
     _decision(ActionType.CALL, "3d 2c", "As Ks Qs Js Ts", 100.0, 900.0,
               5000.0, 5000.0, Street.RIVER),
     "A1"),
    ("A2 se déclenche — coucher la QUINTE MAXIMALE (6♦5♣ sur 2♣ 7♦ 9♠ 3♥ 4♣ : "
     "7-6-5-4-3, aucune couleur possible, aucune quinte supérieure) pour "
     "1 jeton dans un pot de 10 000, suivi à tapis.",
     _decision(ActionType.FOLD, "6d 5c", "2c 7d 9s 3h 4c", 10001.0, 1.0,
               1.0, 1.0, Street.RIVER),
     "A2"),
    ("A3 se déclenche — payer 900 dans 1 000 au turn avec 7♣2♦ sur "
     "A♥ K♦ Q♣ J♠ : 47,4 % d'équité requise, l'équité contre une range "
     "uniforme reste très en dessous.",
     _decision(ActionType.CALL, "7c 2d", "Ah Kd Qc Js", 1000.0, 900.0,
               5000.0, 5000.0, Street.TURN),
     "A3"),
    ("A4 se déclenche — se coucher alors que checker est gratuit.",
     _decision(ActionType.FOLD, "Ah Kd", "2c 7d 9s", 500.0, 0.0, 1000.0,
               1000.0, Street.FLOP),
     "A4"),
    ("rien à dire — payer 100 dans 1 000 avec A♥A♦ servi : le détecteur de "
     "suivi ne doit pas crier.",
     _decision(ActionType.CALL, "Ah Ad", "2c 7d 9s 3h 4c", 1000.0, 100.0,
               5000.0, 5000.0, Street.RIVER),
     ""),
    ("rien à dire — coucher 900 dans 1 000 avec A♥K♦ sur 2♣ 7♦ 9♠ 3♥ 4♣ : le "
     "héros perd contre une quinte 6-5, donc sa borne basse est nulle et le "
     "détecteur de couché ne doit pas crier.",
     _decision(ActionType.FOLD, "Ah Kd", "2c 7d 9s 3h 4c", 1000.0, 900.0,
               900.0, 900.0, Street.RIVER),
     ""),
)
"""Spots FABRIQUÉS dont la réponse est connue à l'avance.

Quatre doivent déclencher un critère précis, deux ne doivent rien déclencher.
Le tuple est exposé au module pour que les tests l'épinglent sans dupliquer
les spots — s'ils divergeaient, le rapport et les tests ne parleraient plus
de la même chose.
"""


def demonstration() -> list[tuple[str, str, PreuveA | None]]:
    """Passe les détecteurs sur :data:`CAS_DEMONSTRATION`.

    Un détecteur qui ne se déclenche jamais sur le corpus réel ne prouve rien
    tant qu'on n'a pas montré qu'il PEUT se déclencher. Cette section fait
    tourner exactement le même :func:`prouver` que le rejeu. Elle est
    rejouable et sert de garde-fou contre du code mort.
    """
    out = []
    for titre, d, attendu in CAS_DEMONSTRATION:
        b = bornes_equite(parse_cards(" ".join(d.cards)),
                          parse_cards(" ".join(d.board)))
        out.append((titre, attendu, prouver(d, b, 100.0)))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MESURES DE FIABILITÉ
# ═══════════════════════════════════════════════════════════════════════════


def fiabilite_bornes_preflop(n_paires: int = 20) -> tuple[float, float, int]:
    """Écart mesuré entre la matrice 169×169 et un calcul plus précis.

    La matrice est du Monte-Carlo à 3 000 tirages par case et raisonne par
    groupe (donc sans blockers). Cette fonction rejoue ``n_paires`` cases
    tirées de façon déterministe avec :data:`MC_PREFLOP` tirages et une main
    héros CONCRÈTE, puis rend (écart moyen absolu, écart max, n).

    C'est ce qui justifie :data:`TOLERANCE_PREFLOP` : si l'écart max mesuré
    dépassait la tolérance, la tolérance serait trop faible et le banc
    pourrait accuser un pro à tort.
    """
    from pfs.core.equity import equity_vs_range
    from pfs.core.range_model import group_name, parse_range
    from pfs.solver.pushfold import _rep_combo

    mat = _matrice()
    rng = np.random.default_rng(20230622)
    ecarts = []
    for _ in range(n_paires):
        g, j = (int(x) for x in rng.integers(0, 169, size=2))
        hero = list(_rep_combo(g))
        cible = parse_range(group_name(j))
        if cible.weights[combo_index(hero[0], hero[1])] > 0 and g == j:
            continue
        try:
            mesure = equity_vs_range(hero, cible, (), n_sims=MC_PREFLOP,
                                     seed=int(g) * 169 + int(j)).equity
        except Exception:  # noqa: BLE001 — range vidée par les blockers
            continue
        ecarts.append(abs(float(mat[g, j]) - float(mesure)))
    if not ecarts:
        return (0.0, 0.0, 0)
    return (float(np.mean(ecarts)), float(np.max(ecarts)), len(ecarts))


def borne_haute_minimale(n_tirages: int = 200, graine: int = 20230622
                         ) -> tuple[float, int]:
    """Le plus petit maximum d'équité observé sur des rivers TIRÉES AU SORT.

    Soutient la deuxième moitié de l'argument « A1 est inatteignable » : la
    première moitié (α ≤ ½) est une démonstration, celle-ci est une mesure.
    Tire ``n_tirages`` situations de river complètes (2 cartes au héros,
    5 au tableau) et rend le minimum, sur ces tirages, du maximum d'équité du
    héros contre les 990 mains adverses possibles.

    Semé, donc rejouable. Rend (minimum, n).
    """
    rng = np.random.default_rng(graine)
    pire = 1.0
    for _ in range(n_tirages):
        cartes = [int(c) for c in rng.choice(N_CARTES, size=7, replace=False)]
        _, eq = equites_par_combo(cartes[:2], cartes[2:])
        pire = min(pire, float(eq.max()))
    return pire, n_tirages


def bubble_factors_mesures(tapis: list[tuple[float, ...]]) -> tuple[float, float, int]:
    """(BF minimum, BF maximum, nombre de mesures) sur les tapis RÉELS du corpus.

    Croise chaque vecteur de tapis de départ observé avec
    :data:`GRILLE_GAINS_TEST` et toutes les paires (héros, vilain). Le
    minimum est le chiffre qui compte : tant qu'il vaut 1 ou plus, un suivi
    réfuté en jetons l'est *a fortiori* en $EV, et la catégorie A n'a pas
    besoin des vrais gains pour tenir.
    """
    valeurs = []
    vus: set[tuple[float, ...]] = set()
    for st in tapis:
        if st in vus:
            continue
        vus.add(st)
        for gains in GRILLE_GAINS_TEST:
            for h in range(len(st)):
                for v in range(len(st)):
                    if h == v:
                        continue
                    eff = min(st[h], st[v])
                    try:
                        valeurs.append(bubble_factor(list(st), list(gains), h, v,
                                                     amount=eff * (1.0 - 1e-9)))
                    except Exception:  # noqa: BLE001 — configuration hors domaine ICM
                        continue
    finis = [v for v in valeurs if math.isfinite(v)]
    if not finis:
        return (0.0, 0.0, 0)
    return (min(finis), max(finis), len(finis))


# ═══════════════════════════════════════════════════════════════════════════
# VOLET ICM (optionnel : le corpus ne donne pas les gains)
# ═══════════════════════════════════════════════════════════════════════════


def volet_icm(res: Resultat, gains: tuple[float, ...]) -> list[str]:
    """Ce que des gains DÉCLARÉS PAR L'UTILISATEUR changeraient à la catégorie A.

    Rend des lignes de rapport. Pour chaque suivi encore innocenté en jetons,
    on regarde si l'équité requise ICM le condamne ; pour chaque couché
    réfuté en jetons, on donne le bubble factor seuil.
    """
    lignes = [
        f"  Gains DÉCLARÉS PAR TOI : {', '.join(f'{g:g}' for g in gains)}.",
        "  Ils ne viennent pas du corpus — le format PHH ne les porte pas.",
        "  Tout ce qui suit dépend donc d'une donnée extérieure au dépôt.",
        "",
    ]
    trouve = 0
    for lg in res.lignes:
        if lg.bornes is None or lg.alpha is None or lg.a_payer <= 0:
            continue
        if lg.action not in ("call", "allin"):
            continue
        pot_avant = max(0.0, lg.pot_gagnable - lg.a_payer)
        # Bubble factor le plus fort de la grille : c'est lui qui condamne le
        # plus de suivis, donc c'est lui qu'il faut montrer pour être complet.
        bf_max = 1.0
        for st in res.tapis_reels:
            for h in range(len(st)):
                for v in range(len(st)):
                    if h == v:
                        continue
                    try:
                        bf = bubble_factor(list(st), list(gains), h, v,
                                           amount=min(st[h], st[v]) * (1.0 - 1e-9))
                    except Exception:  # noqa: BLE001
                        continue
                    if math.isfinite(bf):
                        bf_max = max(bf_max, bf)
        alpha_icm = icm_required_equity(pot_avant, lg.a_payer, bf_max)
        if lg.bornes.uniforme < alpha_icm <= 1.0 and lg.bornes.uniforme >= lg.alpha:
            trouve += 1
            lignes.append(
                f"  {lg.fichier} #{lg.ordre} {lg.joueur} — suivi correct en "
                f"jetons ({lg.bornes.uniforme * 100:.1f} % contre "
                f"{lg.alpha * 100:.1f} % requis) mais l'équité requise passe à "
                f"{alpha_icm * 100:.1f} % au bubble factor le plus fort de tes "
                f"gains ({bf_max:.2f}). Condamné sous ICM, pas en jetons.")
    if not trouve:
        lignes.append("  Aucun suivi ne bascule : la structure déclarée ne "
                      "condamne rien que les jetons n'aient déjà innocenté.")
    return lignes


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═══════════════════════════════════════════════════════════════════════════

_TRAIT = "─" * 78


def _titre(txt: str) -> None:
    print()
    print(_TRAIT)
    print(f"  {txt}")
    print(_TRAIT)


def rapport(res: Resultat, args: argparse.Namespace) -> None:
    cat = Counter(lg.categorie for lg in res.lignes)
    analysables = res.decisions - res.non_analysables

    _titre("BANC WSOP 2023 — les erreurs PROUVABLES des pros")
    print(f"  corpus                : {args.corpus}")
    print(f"  mains du corpus       : {res.mains_lues + res.mains_refusees}")
    print(f"  mains lues (hold'em)  : {res.mains_lues}   "
          f"({', '.join(f'{k}×{v}' for k, v in sorted(res.variantes_lues.items()))})")
    print(f"  mains refusées        : {res.mains_refusees}   "
          "(variantes hors hold'em — refusées, pas mal lues)")
    if res.refus_variantes:
        print("    " + ", ".join(f"{k}×{v}" for k, v in
                                 sorted(res.refus_variantes.items())))
    print(f"  range adverse supposée: « {args.villain} » (n'entre PAS dans la "
          "catégorie A)")
    print(f"  durée                 : {res.secondes:.1f} s")
    if res.incoherences_tapis:
        print(f"  ⚠ {res.incoherences_tapis} incohérences entre les deux lectures "
              "de tapis — le volet ICM est suspect.")

    _titre("1. LE COMPTE — combien de décisions, et dans quelle catégorie")
    print(f"  décisions rejouées                          : {res.decisions:>5}")
    print(f"  dont non analysables (cartes non déclarées) : {res.non_analysables:>5}")
    print(f"  décisions analysées                         : {analysables:>5}")
    print()
    print(f"  A — erreur PROUVABLE sans le recul          : {cat['A']:>5}")
    print(f"  B — écart avec le recul seul (sophisme)     : {cat['B']:>5}")
    print(f"  C — désaccord sans preuve                   : {cat['C']:>5}")
    print(f"  accord conseiller / joueur                  : {cat['ACCORD']:>5}")
    print(f"  REFUS — le conseiller ne conclut pas        : {cat['REFUS']:>5}")
    print()
    print("  A, B et C ne se recouvrent pas : la catégorie A est décidée AVANT")
    print("  d'interroger le conseiller (elle ne dépend pas de lui), puis les")
    print("  refus, puis les accords, puis B contre C selon le recul.")

    _titre("2. CATÉGORIE A — les erreurs prouvables")
    fautes = [lg for lg in res.lignes if lg.categorie == "A"]
    if not fautes:
        print("  AUCUNE. Sur les 83 mains du corpus, dont 18 lisibles, et sur")
        print(f"  les {analysables} décisions analysées, le calcul ne réfute AUCUNE")
        print("  décision de ces cinq joueurs.")
        print()
        print("  Ce n'est pas un échec du banc : c'est le résultat attendu d'un")
        print("  buy-in à 50 000 $. Un banc qui trouverait des dizaines")
        print("  d'erreurs prouvables chez Brian Rast, Matthew Ashton, James")
        print("  Obst, Talal Shakerchi et Kristopher Tong serait suspect avant")
        print("  d'être impressionnant — il faudrait alors soupçonner le banc,")
        print("  pas les joueurs.")
    else:
        print(f"  {len(fautes)} décision(s) réfutée(s). Liste complète :")
        for lg in fautes:
            p = lg.preuve
            assert p is not None
            print()
            print(f"    ── {lg.fichier} · décision #{lg.ordre} · critère {p.critere}")
            print(f"       main      : {lg.joueur} ({lg.position}), {lg.cartes}"
                  + (f" sur {lg.board}" if lg.board else " (préflop)"))
            print(f"       rue       : {lg.rue}  ·  variante {lg.variante}")
            print(f"       joué      : {lg.action.upper()}"
                  + (f" {lg.a_payer:,.0f}" if lg.a_payer else ""))
            print(f"       correct   : {p.action_correcte}")
            print(f"       coût      : {p.cout_jetons:,.0f} jetons  "
                  f"= {p.cout_bb:.2f} bb  (bb = {lg.big_blind:,.0f})")
            print(f"       preuve    : {p.raisonnement}")
            if p.hypothese:
                print(f"       hypothèse : {p.hypothese}")
            else:
                print("       hypothèse : AUCUNE — la preuve tient contre "
                      "chaque main adverse possible, une par une.")
            if p.critere == "A2":
                seuil = bubble_factor_seuil(lg.pot_gagnable, lg.a_payer,
                                            p.equite_utilisee)
                print(f"       réserve   : en tournoi, ce couché redevient "
                      f"correct si le bubble factor dépasse {seuil:.2f}. "
                      "Le corpus ne donne pas les gains ; à toi de juger.")

    _titre("2 bis. POURQUOI A1 NE SE DÉCLENCHERA JAMAIS — et pourquoi le dire")
    print("  Fait STRUCTUREL, vrai indépendamment de ce corpus et de ces")
    print("  joueurs. Deux bornes se croisent :")
    print()
    print("    · la cote du pot ne peut jamais exiger plus de 50 % d'équité.")
    print("      Pour devoir c, il faut qu'un adversaire ait engagé au moins")
    print("      (engagé_héros + c) ; le pot gagnable vaut donc au moins c, et")
    print("      α = c/(P+c) ≤ c/(c+c) = ½.")
    hautes = [lg.bornes.haute for lg in res.lignes
              if lg.bornes is not None and lg.bornes.exacte]
    if hautes:
        print("    · la borne HAUTE d'équité ne descend pas sous 50 %. Sur les")
        print(f"      {len(hautes)} décisions postflop à bornes exactes de ce "
              "corpus :")
        print(f"      minimum {min(hautes) * 100:.1f} %, maximum "
              f"{max(hautes) * 100:.1f} %.")
    pire, n_tir = borne_haute_minimale()
    print(f"      Et sur {n_tir} rivers tirées au sort (semées, rejouables) : la")
    print(f"      borne haute la plus basse observée vaut {pire * 100:.1f} %. La")
    print("      raison : il existe presque toujours une main adverse qui")
    print("      n'améliore pas le tableau, et contre elle le héros partage au")
    print("      pire — donc 50 %.")
    print()
    print("  Conséquence : A1 est implémenté, testé, VIVANT (section 7 le")
    print("  montre sur un spot fabriqué à 90 % de cote), et il ne peut pas")
    print("  se déclencher à une vraie table. Le taire aurait laissé croire")
    print("  qu'une catégorie A vide signifie « aucune erreur » alors qu'elle")
    print("  signifie ici, pour ce critère, « aucune erreur détectable ».")
    print("  C'est A3 qui porte réellement la réfutation d'un suivi.")

    _titre("3. À QUELLE DISTANCE DE LA PREUVE ? — les décisions les plus proches")
    print("  Une catégorie A vide ne dit pas à quel point on en était loin.")
    print("  Marge = équité de la borne − équité requise, en points.")
    print("    · pour un SUIVI, il faudrait une marge NÉGATIVE pour le réfuter :")
    print("      les valeurs ci-dessous, toutes positives, disent de combien de")
    print("      points d'équité chaque suivi est au-dessus de la cote.")
    print("    · pour un COUCHÉ, il faudrait une marge POSITIVE : les valeurs")
    print("      négatives disent de combien le couché est justifié.")
    print()
    proches_suivi: list[tuple[float, Ligne]] = []
    proches_couche: list[tuple[float, Ligne]] = []
    for lg in res.lignes:
        if lg.bornes is None or lg.alpha is None or lg.a_payer <= 0:
            continue
        if lg.action in ("call", "allin"):
            proches_suivi.append((lg.bornes.uniforme - lg.alpha, lg))
        elif lg.action == "fold":
            proches_couche.append((lg.bornes.basse - lg.alpha, lg))
    print("  a) SUIVIS — marge contre la range uniforme (critère A3) :")
    for marge, lg in sorted(proches_suivi)[:5]:
        print(f"     {marge * 100:+6.1f} pt  {lg.fichier} #{lg.ordre:<2} "
              f"{lg.joueur:<16} {lg.cartes:<6} {lg.rue:<8} "
              f"paie {lg.a_payer:>9,.0f} dans {lg.pot_gagnable:>10,.0f}")
    if not proches_suivi:
        print("     aucun suivi face à une mise dans le corpus lu.")
    print()
    print("  b) COUCHÉS FERMANT LE COUP — marge contre la main adverse la plus")
    print("     défavorable (critère A2). Les couchés qui ne ferment pas le")
    print("     coup ne figurent pas : leur réfutation demanderait de savoir")
    print("     ce que l'adversaire aurait misé aux rues suivantes.")
    fermants = [(m, lg) for m, lg in proches_couche if lg.ferme]
    for marge, lg in sorted(fermants, key=lambda t: -t[0])[:5]:
        print(f"     {marge * 100:+6.1f} pt  {lg.fichier} #{lg.ordre:<2} "
              f"{lg.joueur:<16} {lg.cartes:<6} {lg.rue:<8} "
              f"paierait {lg.a_payer:>9,.0f} dans {lg.pot_gagnable:>10,.0f}")
    if not fermants:
        print("     aucun couché fermant le coup dans le corpus lu.")

    _titre("4. CATÉGORIE B — le sophisme du résultat, nommé comme tel")
    bs = [lg for lg in res.lignes if lg.categorie == "B"]
    print("  Ces décisions ne sont PAS des erreurs. Le conseiller aurait joué")
    print("  autrement et, EN REGARDANT LES CARTES ADVERSES, cela aurait")
    print("  rapporté davantage. Le joueur ne les voyait pas. Évaluer une")
    print("  décision par ce qui s'est passé ensuite est exactement le")
    print("  raisonnement que ce projet refuse — la colonne existe pour")
    print("  montrer combien il y en aurait, pas pour accuser.")
    print()
    if bs:
        for lg in bs:
            reserve = ("" if lg.recul_ferme else
                       "   (⚠ chiffre partiel : ignore les mises des rues "
                       "suivantes)")
            print(f"    {lg.fichier} #{lg.ordre:<2} {lg.joueur:<16} {lg.cartes:<6} "
                  f"{lg.rue:<8} joué {lg.action:<5} · conseil {lg.conseil_action}")
            print(f"      valeur du SUIVI avec les cartes adverses : "
                  f"{lg.recul:+,.0f} jetons")
            print(f"      ce que le CONSEIL aurait rapporté de plus : "
                  f"{lg.gain_conseil:+,.0f} jetons "
                  f"({(lg.gain_conseil or 0.0) / (lg.big_blind or 1.0):+.2f} bb)"
                  f"{reserve}")
        gains = [lg.gain_conseil for lg in bs if lg.gain_conseil is not None]
        m, sd, err = _moyenne_ic(gains)
        print()
        print(f"    total {len(bs)} lignes, gain moyen du conseil {m:+,.0f} jetons "
              f"(écart-type {sd:,.0f}, erreur-type {err:,.0f}).")
        print("    Cette moyenne n'autorise AUCUNE conclusion : deux points, et")
        print("    surtout un chiffre construit sur une information que le")
        print("    joueur n'avait pas.")
    else:
        print("    Aucune. Le recul ne donne raison à notre conseiller sur")
        print("    aucun des désaccords calculables de ce corpus.")

    _titre("5. CATÉGORIE C — désaccords sans preuve, à compter")
    cs = [lg for lg in res.lignes if lg.categorie == "C"]
    print(f"  {len(cs)} décision(s). Ni le calcul ni le recul ne tranchent.")
    print("  Répartition par sens du désaccord :")
    sens = Counter(lg.conseil_action for lg in cs)
    for k, v in sens.most_common():
        print(f"    {v:>4} × conseil {k}")
    print()
    print("  Répartition par rue :")
    for k, v in Counter(lg.rue for lg in cs).most_common():
        print(f"    {v:>4} × {k}")
    if args.detail and cs:
        print()
        for lg in cs:
            recul = ("recul non calculable" if lg.recul is None
                     else f"recul {lg.recul:+,.0f} jetons")
            print(f"    {lg.fichier} #{lg.ordre:<2} {lg.joueur:<16} {lg.cartes:<6} "
                  f"{lg.rue:<8} joué {lg.action:<5} · {lg.conseil_action} · {recul}")

    _titre("6. QUAND LE CONSEILLER REFUSE DE CONCLURE — un livrable à part entière")
    refus = [lg for lg in res.lignes if lg.categorie == "REFUS"]
    print(f"  {len(refus)} décision(s) sur {analysables} analysées "
          f"({len(refus) / max(analysables, 1) * 100:.1f} %).")
    print("  Un refus n'est pas un échec : c'est le conseiller qui reconnaît")
    print("  un spot hors de son domaine DÉCLARÉ. Le compter comme un accord")
    print("  gonflerait le logiciel, le compter comme un désaccord le")
    print("  noircirait. Motifs mesurés (un spot peut en cumuler deux, la")
    print("  somme des lignes dépasse donc le total) :")
    print()
    par_motif: Counter = Counter()
    for lg in refus:
        for m in (lg.motifs_refus or (lg.motif_refus,)):
            par_motif[m] += 1
    for motif, k in par_motif.most_common():
        print(f"    {k:>4} × {motif}")
    print()
    print("  Ce que ce tableau dit du logiciel, indépendamment des chiffres :")
    print("  il n'existe AUCUN régime pour « défendre face à une relance")
    print("  préflop », ni pour « payer un tapis adverse ». Sur une table")
    print("  finale à cinq joueurs en tapis courts, ce sont les deux spots les")
    print("  plus fréquents. C'est le manque le plus net que ce corpus révèle,")
    print("  et il est mesuré ici, pas supposé.")

    _titre("7. LES DÉTECTEURS SONT-ILS VIVANTS ?")
    attendus = sum(1 for _, _, a in CAS_DEMONSTRATION if a)
    print("  Une catégorie A vide peut vouloir dire « aucune erreur » ou")
    print(f"  « détecteur mort ». Ces {len(CAS_DEMONSTRATION)} spots FABRIQUÉS "
          "passent par le même")
    print(f"  prouver() que le corpus. {attendus} doivent se déclencher, "
          f"{len(CAS_DEMONSTRATION) - attendus} non.")
    print()
    ok = True
    for titre, attendu, p in demonstration():
        obtenu = p.critere if p is not None else ""
        bon = obtenu == attendu
        ok = ok and bon
        print(f"    [{'OK ' if bon else 'RATÉ'}] {titre}")
        print(f"           attendu « {attendu or 'rien'} », "
              f"obtenu « {obtenu or 'rien'} »"
              + (f" — coût {p.cout_jetons:,.0f} jetons" if p else ""))
    print()
    print("  → " + ("les quatre critères sont vivants et ne crient pas à tort."
                    if ok else "AU MOINS UN DÉTECTEUR EST FAUX : rapport à jeter."))

    _titre("8. FIABILITÉ DES BORNES")
    print("  Postflop, les bornes sont EXACTES : chaque main adverse possible")
    print("  est énumérée, chaque runout aussi. Rien à tolérancer.")
    print()
    moy, mx, n = fiabilite_bornes_preflop()
    print("  Préflop, elles viennent de la matrice 169×169 du solveur")
    print(f"  (Monte-Carlo, 3 000 tirages par case). Écart mesuré sur {n} cases")
    print(f"  contre un Monte-Carlo à {MC_PREFLOP:,} tirages avec une main héros")
    print(f"  concrète : moyenne {moy * 100:.2f} pt, maximum {mx * 100:.2f} pt.")
    print(f"  Tolérance appliquée : {TOLERANCE_PREFLOP * 100:.0f} pt, soit "
          f"{'au-dessus du' if TOLERANCE_PREFLOP > mx else 'SOUS le'} maximum "
          "mesuré.")
    if TOLERANCE_PREFLOP <= mx:
        print("  ⚠ La tolérance ne couvre pas l'écart mesuré : une accusation")
        print("    préflop serait fragile. Aucune n'est portée dans ce rapport ;")
        print("    si la catégorie A en contenait une, il faudrait la refaire.")
    print()
    bmin, bmax, nbf = bubble_factors_mesures(res.tapis_reels)
    print(f"  Bubble factor mesuré sur {nbf} configurations (tapis RÉELS du")
    print(f"  corpus × {len(GRILLE_GAINS_TEST)} structures de gains décroissantes) :")
    print(f"    minimum {bmin:.4f} · maximum {bmax:.4f}")
    print("  Le minimum est le chiffre utile : tant qu'il vaut 1 ou plus, un")
    print("  suivi réfuté en JETONS l'est aussi en $EV, et la catégorie A")
    print("  n'a pas besoin des gains — que le corpus ne donne pas — pour")
    print("  tenir en tournoi. La réciproque est fausse : un COUCHÉ réfuté en")
    print("  jetons peut être excusé par l'ICM, d'où le bubble factor seuil")
    print("  affiché sous chaque A2.")

    if args.payouts:
        _titre("9. VOLET ICM — sur des gains que TU déclares")
        for ligne in volet_icm(res, tuple(float(x) for x in
                                          args.payouts.replace(";", ",").split(",")
                                          if x.strip())):
            print(ligne)
    else:
        _titre("9. VOLET ICM — non évalué, et voici pourquoi")
        print("  Le format PHH ne porte pas la structure de gains, et le")
        print("  corpus ne la donne nulle part. Sans elle, aucune violation")
        print("  ICM ne se calcule. Le banc pourrait aller chercher les gains")
        print("  publiés des WSOP 2023 : ce serait une donnée EXTÉRIEURE au")
        print("  dépôt, non rejouable depuis le corpus, et une affirmation")
        print("  chiffrée reposant dessus ne respecterait pas la règle du")
        print("  projet. --payouts \"50, 30, 20\" active le volet en assumant")
        print("  que les gains viennent de toi.")

    _titre("10. LIMITES — à lire avant de citer un chiffre de ce rapport")
    for ligne in (
        "ÉCHANTILLON BIAISÉ. 83 mains d'une table finale de championnat mixte",
        "à 50 000 $ de buy-in, jour 5, retenues pour la diffusion. Aucun taux",
        "de ce rapport n'est représentatif du jeu professionnel, et aucun",
        "n'est publié comme tel. 18 mains de hold'em, cinq joueurs.",
        "",
        "AUCUN CRITÈRE DE TAILLE DE MISE. « Une mise qu'aucune range de valeur",
        "ne justifie » demande de savoir qui paie et qui se couche, donc la",
        "stratégie adverse. Il n'en existe pas de forme sans hypothèse. Le",
        "banc ne l'implémente pas plutôt que de l'implémenter mal.",
        "",
        "LES COUCHÉS SONT PRESQUE INRÉFUTABLES, et c'est structurel : prouver",
        "qu'un couché est mauvais demande une borne INFÉRIEURE sur l'équité,",
        "donc de supposer que l'adversaire n'est pas plus fort qu'une range",
        "donnée. Seule la borne sur la main adverse la plus défavorable s'en",
        "passe, et elle n'accuse qu'avec des cotes énormes. Le déséquilibre",
        "entre suivis et couchés dans la section 3 vient de là, pas des joueurs.",
        "",
        "LA CATÉGORIE B N'EST PAS UNE ERREUR. Elle utilise les cartes adverses,",
        "que le joueur ne voyait pas. Elle est là pour être comptée et écartée.",
        "",
        "LE CONSEILLER EST HORS DOMAINE SUR LA MAJORITÉ DE CE CORPUS (section",
        "6). Les catégories B et C ne portent donc que sur la minorité de spots",
        "qu'il couvre, et ne mesurent pas sa qualité globale.",
    ):
        print(f"  {ligne}")
    print(_TRAIT)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAUT)
    p.add_argument("--villain", default="moyenne",
                   help="range adverse supposée PAR LE CONSEILLER (n'entre pas "
                        "dans la catégorie A)")
    p.add_argument("--payouts", default="",
                   help="structure de gains décroissante, DÉCLARÉE PAR TOI : le "
                        "corpus ne la contient pas")
    p.add_argument("--detail", action="store_true",
                   help="journal décision par décision de la catégorie C")
    p.add_argument("--demonstration", action="store_true",
                   help="seulement : les détecteurs se déclenchent-ils ?")
    p.add_argument("--silencieux", action="store_true",
                   help="pas de progression sur stderr")
    args = p.parse_args(argv)

    if args.demonstration:
        _titre("LES DÉTECTEURS SONT-ILS VIVANTS ?")
        ok = True
        for titre, attendu, preuve in demonstration():
            obtenu = preuve.critere if preuve is not None else ""
            bon = obtenu == attendu
            ok = ok and bon
            print(f"  [{'OK ' if bon else 'RATÉ'}] {titre}")
            print(f"         attendu « {attendu or 'rien'} », obtenu "
                  f"« {obtenu or 'rien'} »")
            if preuve is not None:
                print(f"         {preuve.raisonnement}")
        print(_TRAIT)
        return 0 if ok else 1

    if not args.corpus.exists():
        print(f"corpus introuvable : {args.corpus}", file=sys.stderr)
        return 2

    res = rejouer(args.corpus, args.villain, not args.silencieux)
    rapport(res, args)
    return 1 if res.incoherences_tapis else 0


if __name__ == "__main__":
    raise SystemExit(main())
