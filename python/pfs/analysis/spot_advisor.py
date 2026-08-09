"""Conseiller de spot — « qu'est-ce qu'il fallait faire ? » sur une main jouée.

Conçu pour l'étude d'une main **terminée** (capture d'écran, note, souvenir) :
on décrit le spot avec ce qu'une image donne — tes cartes, le board, le pot,
la mise subie, les tapis, la position — et le conseiller rend le verdict
adossé aux solveurs du projet.

Quatre régimes, choisis selon le spot :

* **préflop tapis court en tournoi** (``stacks`` *et* ``payouts`` fournis) :
  équilibre de Nash jam/fold résolu en **$EV ICM** (Malmuth-Harville), avec
  l'écart au chipEV affiché — c'est cet écart qui coûte cher en Twister ;
* **préflop tapis court heads-up chipEV** (≤ 25 bb, héros premier de parole) :
  équilibre de Nash jam/fold exact (``solver.pushfold``) ;
* **préflop tapis profond** : comparaison à la range d'ouverture de
  référence de la position (``GTO_PRESETS``, charts approximatives assumées) ;
* **postflop** : river **résolue au CFR** range contre range quand personne
  n'a misé (taille, fréquence et EV de ta main) ; sinon équité exacte de ta
  main contre une range adverse plausible, confrontée aux cotes du pot —
  corrigées par le bubble factor dès que le spot est un spot de tournoi.

Honnêteté (NEMESIS) : « certain » est réservé aux verdicts qui ne supposent
RIEN du jeu adverse — ni range de mains, ni action prêtée à un joueur. Il ne
reste que le Nash push/fold à deux joueurs présents : la range de call du
vilain y est un résultat du solve, et il n'y a personne d'autre à supposer
couché. Sont donc « indicatifs » : le régime ICM à trois joueurs et plus (les
autres sont supposés déjà couchés — une hypothèse sur leur stratégie), les
charts d'ouverture, et le solve river, dont l'équilibre est exact POUR CES
RANGES : ce sont les ranges qui sont l'hypothèse.

« Certain » qualifie le modèle, pas la précision numérique. Le fictitious
play s'arrête sur un certificat ε-Nash, et sur les mains que l'équilibre
laisse quasi indifférentes un solve plus long peut inverser le verdict : le
conseiller le dit alors dans ses raisons, avec les bandes mesurées
(``_INDIFFERENCE_BB``, ``_INDIFFERENCE_CAGNOTTE``). Il déclare toujours
l'hypothèse et le seuil qui ferait basculer la décision — c'est cette
sensibilité qui a une valeur d'étude.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from pfs.core.bluffcatch import minimum_defence_frequency, required_equity
from pfs.core.equity import equity_vs_range
from pfs.core.icm import bubble_factor, icm_required_equity, risk_premium
from pfs.core.range_model import (
    COMBO_TO_GROUP,
    GTO_PRESETS,
    RANKS,
    SUITS,
    Range,
    combo_index,
    group_name,
    parse_range,
)
from pfs.solver.postflop import IP, OOP, PostflopError, PostflopSolver
from pfs.solver.pushfold import equity_matrix_169, solve_hu_pushfold

__all__ = ["Spot", "Advice", "advise", "parse_cards"]

MAX_PUSHFOLD_BB = 25.0

RIVER_SOLVE_ITERATIONS = 400
"""Itérations DCFR visées par le solve river.

Mesuré sur le board Kd 7h 2c 9s 3d, pot 100, tapis 200, tailles
0,33 / 0,75 / 1,25 × pot, 2 mises par street. Le solve est déterministe : ces
chiffres se rejouent au chiffre près, et un test les épingle. Après 400
itérations l'exploitabilité vaut, pour deux ranges « large » (390 combos
écrits, 328 une fois les cartes du board retirées), **0,052 % du pot sur
l'arbre en position** (11 nœuds) et **0,139 % hors de position** (20 nœuds) ;
pour deux ranges « moyenne » (230 → 196 combos) 0,055 % et 0,130 % ; pour
deux ranges « serrée » (70 → 62 combos) 0,083 % et 0,159 %. La décroissance
mesurée est proche de 1/t — sur l'arbre en position : 0,470 % à 50
itérations, 0,217 % à 100, 0,107 % à 200, 0,052 % à 400.
"""

RIVER_SOLVE_BUDGET_S = 2.0
"""Budget de temps du solve river, en secondes.

Le cas le plus lourd est celui **hors de position** : le conseiller y
construit 20 nœuds contre 11 en position, parce qu'un check du héros laisse
le vilain miser derrière. Ce compte ne dépend pas de la machine et un test
l'épingle. Le temps, lui, en dépend : sur la machine de mesure les 400
itérations demandent de l'ordre de 3 s sur l'arbre hors de position (deux
ranges « large », board Kd 7h 2c 9s 3d, pot 100, tapis 200) contre 1,6 s en
position. **Le budget de 2,0 s ne va donc pas au bout hors de position** : il
coupe vers 350 itérations, et l'exploitabilité passe de 0,139 % à 0,159 % du
pot. Le budget est vérifié entre deux paquets de 50 itérations ; dépassé, le
conseiller s'arrête là où il en est et **déclare** dans ses raisons les
itérations faites et l'exploitabilité atteinte. À 0 ou moins il ne solve pas
du tout et retombe sur l'équité — c'est la porte de sortie testable.
"""

_RIVER_CHUNK = 50

_MIXTE_MIN = 0.01
"""Fréquence en deçà de laquelle une action n'est pas listée dans « MIXTE — ».

Filtre de lecture, et non seuil de bruit : une action jouée moins d'une fois
sur cent ne s'exécute pas à une table, et la moyenne DCFR ne descend jamais
exactement à zéro (mesuré sur le solve golden — board Kd 7h 2c 9s 3d, deux
ranges « large », arbre en position, 400 itérations : aucune des 1312 entrées
de la racine n'est nulle, la plus petite vaut 1,6e-4). Ce que le filtre cache
est borné et mesuré : au plus 2,2 points de fréquence pour un combo donné,
0,2 point en médiane. ``Advice.mix`` expose la répartition complète, sans
filtre — c'est là qu'on lit ce que l'affichage résume.
"""

_INDIFFERENCE_BB = 0.05
"""Bande d'indifférence du push/fold chipEV, en bb.

Le fictitious play s'arrête sur un certificat ε-Nash ou sur la stabilité de
ses moyennes : sa stratégie n'est pas toujours pure, et l'écart d'EV par
groupe garde une incertitude. Mesurée en rejouant chaque solve avec un
plafond de 1000 itérations (48 tapis de 1,1 à 25,0 bb, pas de 0,5 bb) : 43
groupes changent de verdict jam/fold entre les deux solves, et le plus
tranché d'entre eux n'affichait que 0,029 bb d'écart (54s à 22,1 bb). Le
seuil est posé à 0,05 bb, 1,7 fois au-dessus de cette bande.
"""

_INDIFFERENCE_CAGNOTTE = 0.001
"""Bande d'indifférence du push/fold ICM, en fraction de la cagnotte.

Même protocole sur 36 configurations ICM (héros de 5 à 20 bb, tables de 3 et
4 joueurs, gains 65/35 et 50/30/20, référence à 1000 itérations) : 52 groupes
changent de verdict, le plus tranché à 0,034 % de la cagnotte ([12, 12, 12]
bb, gains 65/35). Le seuil est posé à 0,1 %, 3,0 fois au-dessus.
"""

_ACTION_PURE = 0.85
"""Convention d'affichage : au-dessus, une action résume la stratégie.

Ce n'est pas une mesure mais un choix de lecture — au-dessous, annoncer la
seule action de tête masquerait un mélange que le solveur juge nécessaire, et
c'est justement le mélange qui est la réponse.
"""

# Ranges adverses plausibles par défaut (postflop), en l'absence de lecture.
# Un vilain qui MISE sur une rue avancée est plus fort qu'un vilain qui checke.
_DEFAULT_VILLAIN = {
    "large": "22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, "
             "A7o+, K9o+, QTo+, JTo",
    "moyenne": "22+, A8s+, A5s-A2s, K9s+, QTs+, JTs, T9s, 98s, "
               "ATo+, KJo+, QJo",
    "serree": "TT+, AJs+, KQs, AKo, AQo",
}

# Positions qui parlent en dernier postflop. La small blind est un cas à part :
# en heads-up elle est en position, à trois et plus elle ne l'est pas.
_POSITIONS_IP = frozenset({"IP", "BTN", "BU", "BOUTON", "CO"})

_EQUITY_MATRIX = None


def _equity_matrix():
    global _EQUITY_MATRIX
    if _EQUITY_MATRIX is None:
        _EQUITY_MATRIX = equity_matrix_169()
    return _EQUITY_MATRIX


_CARD_RE = re.compile(r"([2-9TJQKAtjqka]|10)\s*([shdcSHDC♠♥♦♣])")
_SUIT_ALIAS = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def parse_cards(text: str) -> list[int]:
    """Lit des cartes écrites librement → indices 0-51.

    Accepte « AhKd », « As Kd », « A♠ K♦ », « 10h Jc », majuscules ou non —
    tout ce qu'on recopie depuis une capture d'écran.

    >>> [c // 4 for c in parse_cards("Ah Kd")]
    [0, 1]
    """
    out: list[int] = []
    for rank, suit in _CARD_RE.findall(text or ""):
        r = "T" if rank == "10" else rank.upper()
        s = _SUIT_ALIAS.get(suit, suit.lower())
        out.append(RANKS.index(r) * 4 + SUITS.index(s))
    if len(set(out)) != len(out):
        raise ValueError(f"carte en double dans « {text} ».")
    return out


def _nombres(spec: str | Sequence[float]) -> tuple[float, ...]:
    """Lit « 10, 40, 1 » (ou une séquence déjà numérique) → tuple de flottants.

    La forme texte existe pour que la saisie en rafale de ``analyser_main``
    (« clé valeur ») puisse renseigner des tapis et une structure de gains
    sans conversion côté appelant.

    >>> _nombres("50, 30, 20")
    (50.0, 30.0, 20.0)
    """
    if isinstance(spec, str):
        return tuple(float(x) for x in spec.replace(";", ",").split(",")
                     if x.strip())
    return tuple(float(x) for x in spec)


@dataclass(slots=True)
class Spot:
    """Un spot de poker tel qu'une capture d'écran le donne.

    Parameters
    ----------
    hero : str
        Tes deux cartes (« Ah Kd », « A♠K♦ »…).
    board : str
        Les cartes communes, vide en préflop.
    pot : float
        Pot AVANT la mise adverse, en jetons ou en bb (même unité partout).
    bet : float
        Mise à payer (0 si personne n'a misé).
    stack : float
        Ton tapis effectif (min des deux tapis en jeu).
    big_blind : float
        Valeur de la big blind, pour convertir en bb.
    position : str
        « BTN », « SB », « BB », « CO », « MP », « UTG », ou « ip »/« oop ».
    villain : str
        Range adverse supposée : « large », « moyenne », « serree », ou une
        range explicite (« TT+, AQs+ »).
    players : int
        Joueurs encore en jeu (2 = heads-up).
    stacks : str or sequence of float
        Tapis de TOUS les joueurs de la table (« 10, 40, 1 »), même unité que
        ``pot``. Avec ``payouts``, bascule le préflop en régime ICM et corrige
        les cotes du pot postflop par le bubble factor.
    payouts : str or sequence of float
        Structure de gains décroissante (« 65, 35 » ou « 50, 30, 20 »). Une
        structure winner-take-all (« 100 ») rend les jetons linéaires : l'ICM
        y coïncide alors exactement avec le chipEV.
    hero_seat, villain_seat : int
        Indices du héros et du défenseur dans ``stacks``.
    ante : float
        Ante payée par chaque joueur, même unité que ``pot``.
    hero_range : str
        Ta range supposée pour le solve river. Vide → la même que ``villain``
        (un solve sur ta seule main serait transparent : le vilain la
        connaîtrait, et la question de la fréquence de bluff disparaîtrait).
    bet_sizes : str or sequence of float
        Tailles de mise testées par le solveur river, en fraction du pot.
    max_bets : int
        Mises et relances autorisées par street dans l'arbre du solve.
    solver_budget_s : float
        Budget de temps du solve river en secondes ; ≤ 0 désactive le solve
        et retombe sur l'équité contre la range supposée.
    """

    hero: str
    board: str = ""
    pot: float = 0.0
    bet: float = 0.0
    stack: float = 0.0
    big_blind: float = 1.0
    position: str = "BTN"
    villain: str = "moyenne"
    players: int = 2
    stacks: str | Sequence[float] = ""
    payouts: str | Sequence[float] = ""
    hero_seat: int = 0
    villain_seat: int = 1
    ante: float = 0.0
    hero_range: str = ""
    bet_sizes: str | Sequence[float] = "0.33, 0.75, 1.25"
    max_bets: int = 2
    solver_budget_s: float = RIVER_SOLVE_BUDGET_S


@dataclass(slots=True)
class Advice:
    """Verdict du conseiller sur un spot.

    Attributes
    ----------
    action : str
        Le verdict, prêt à lire : « JAM (tapis) », « FOLD », « CALL (juste) »,
        « MISER 75 (75 % du pot) », « MIXTE — … ».
    confidence : str
        « certain » ou « indicatif », selon le critère de l'en-tête du module.
    regime : str
        Le moteur qui a tranché, avec ses paramètres.
    equity, required, mdf : float or None
        Équité de ta main, équité requise par les cotes (corrigée par le
        bubble factor en tournoi) et fréquence de défense minimale, dans
        ``[0, 1]``.
    ev_bb : float or None
        Espérance de la décision conseillée, en big blinds, toujours comptée
        depuis l'abandon du pot : EV(jam) − EV(fold) en push/fold, EV du call
        face à une mise, EV de ta main dans le solve river (où un
        check-abattage perdu vaut 0).
    ev_icm : float or None
        ``EV(jam) − EV(fold)`` en unités de gain — régime ICM seulement.
    bubble : float or None
        Bubble factor héros → vilain.
    size : float or None
        Taille de la mise conseillée, dans l'unité du pot.
    frequency : float or None
        Fréquence de l'action de **tête**, celle qui ouvre ``action``. Pour
        une main mixte dont l'action de tête est un check, la fréquence de
        mise ne se lit donc PAS ici mais dans ``mix``.
    mix : tuple of (str, float)
        Toutes les actions du nœud avec leur fréquence pour ta main, de la
        plus fréquente à la moins fréquente, sans filtre — alors que
        ``action`` masque celles qui tombent sous ``_MIXTE_MIN``.
    villain_defence : float or None
        Part de la range adverse qui ne se couche pas face à la mise
        conseillée, à l'équilibre.
    hand : str
        Groupe de la grille 169 auquel ta main appartient (« AA », « K9o »…).
    reasons : list of str
        Ce qui fonde le verdict, chiffres compris.
    assumptions : list of str
        Ce qu'il a fallu supposer — à relire si tu avais une lecture.
    """

    action: str
    confidence: str
    regime: str
    equity: float | None = None
    required: float | None = None
    mdf: float | None = None
    ev_bb: float | None = None
    ev_icm: float | None = None
    bubble: float | None = None
    size: float | None = None
    frequency: float | None = None
    mix: tuple[tuple[str, float], ...] = ()
    villain_defence: float | None = None
    hand: str = ""
    reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            "──────────────────────────────────────────────────────",
            f"  {self.hand}  →  {self.action}",
            f"  ({self.regime} · {self.confidence})",
            "──────────────────────────────────────────────────────",
        ]
        if self.equity is not None:
            lines.append(f"  Ton équité          : {self.equity * 100:5.1f} %")
        if self.required is not None:
            lines.append(f"  Équité requise      : {self.required * 100:5.1f} %"
                         "   (cotes du pot)")
        if self.mdf is not None:
            lines.append(f"  MDF                 : {self.mdf * 100:5.1f} %")
        if self.villain_defence is not None:
            lines.append(f"  Défense du vilain   : "
                         f"{self.villain_defence * 100:5.1f} %   (à l'équilibre)")
        if self.size is not None:
            lines.append(f"  Taille conseillée   : {self.size:5.1f}")
        if self.frequency is not None:
            lines.append(f"  Fréq. action de tête: {self.frequency * 100:5.1f} %")
        if len(self.mix) > 1:
            lines.append("  Mélange complet     : " + " · ".join(
                f"{lab} {f * 100:.1f} %" for lab, f in self.mix))
        if self.ev_bb is not None:
            lines.append(f"  EV                  : {self.ev_bb:+5.2f} bb")
        if self.ev_icm is not None:
            lines.append(f"  EV ICM              : {self.ev_icm:+7.4f} "
                         "  (unités de gain)")
        if self.bubble is not None:
            lines.append(f"  Bubble factor       : {self.bubble:5.2f}")
        if self.reasons:
            lines.append("")
            lines += [f"  · {r}" for r in self.reasons]
        if self.assumptions:
            lines.append("")
            lines.append("  Hypothèses (à ajuster si tu avais une lecture) :")
            lines += [f"    – {a}" for a in self.assumptions]
        lines.append("──────────────────────────────────────────────────────")
        return "\n".join(lines)


def _hand_label(cards: list[int]) -> str:
    g = int(COMBO_TO_GROUP[combo_index(cards[0], cards[1])])
    return group_name(g)


@lru_cache(maxsize=64)
def _villain_range(spec: str):
    """Range adverse, mémoïsée : en session d'entraînement, on enchaîne les
    mains avec la même range supposée — inutile de la re-parser à chaque fois.
    """
    return parse_range(_DEFAULT_VILLAIN.get(spec, spec))


@lru_cache(maxsize=256)
def _nash_solution(eff_bb_deci: int, ante_deci: int = 0):
    """Équilibre jam/fold chipEV mémoïsé par dixième de bb.

    En session d'entraînement on repasse sans cesse sur les mêmes tapis
    (10, 12, 15 bb…) : le solve n'est fait qu'une fois par valeur.
    """
    return solve_hu_pushfold(eff_bb_deci / 10.0, sb=0.5, bb=1.0,
                             ante=ante_deci / 10.0, equity=_equity_matrix())


@lru_cache(maxsize=128)
def _nash_icm(stacks_deci: tuple[int, ...], payouts: tuple[float, ...],
              hero: int, villain: int, ante_deci: int):
    """Équilibre jam/fold résolu en $EV ICM, mémoïsé par configuration."""
    stacks_bb = [d / 10.0 for d in stacks_deci]
    return solve_hu_pushfold(stacks_bb[hero], payouts=list(payouts),
                             stacks=stacks_bb, sb=0.5, bb=1.0,
                             ante=ante_deci / 10.0, hero=hero, villain=villain,
                             equity=_equity_matrix())


def _bubble(stacks: Sequence[float], payouts: Sequence[float],
            hero: int, villain: int) -> float:
    """Bubble factor héros → vilain pour un all-in au tapis effectif.

    Le montant disputé est amputé d'un milliardième parce que la branche
    perdante met le héros à zéro et que la récurrence de Harville divise par
    la somme des tapis encore en course. Cette somme ne s'annule que si la
    récurrence doit descendre jusqu'au joueur busté, c'est-à-dire quand la
    structure de gains classe **autant de places qu'il y a de joueurs** —
    ``len(payouts) >= len(stacks)`` après la troncature de ``icm._validate``.
    Mesuré : 2 joueurs / 2 gains, 3/3, 4/4 et 5/5 lèvent
    ``ZeroDivisionError`` ; 2/1, 3/2 et 4/3 passent sans amputation. Le
    golden 3 joueurs / 2 gains du conseiller n'en a donc pas besoin, mais
    ``stacks="10, 40"`` + ``payouts="65, 35"`` (2/2) et
    ``payouts="50, 30, 20"`` sur trois tapis (3/3) en dépendent.

    Coût de l'amputation, mesuré : 8,5e-9 en relatif là où la valeur exacte
    existe (3/2, tapis 10/40/1 bb, gains 65/35 : 5,021764802 contre
    5,021764845) et moins de 7,1e-9 par rapport à une amputation mille fois
    plus fine là où elle n'existe pas. C'est le même plancher relatif que
    ``solver.pushfold`` applique à ses états terminaux.
    """
    eff = min(stacks[hero], stacks[villain])
    return bubble_factor(stacks, payouts, hero, villain,
                         amount=eff * (1.0 - 1e-9))


def _tournoi(spot: Spot) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Tapis et gains du spot, ou deux tuples vides si ce n'est pas un tournoi."""
    stacks = _nombres(spot.stacks)
    payouts = _nombres(spot.payouts)
    if not stacks or not payouts:
        return (), ()
    n = len(stacks)
    h, v = spot.hero_seat, spot.villain_seat
    if h == v or not (0 <= h < n) or not (0 <= v < n):
        raise ValueError(f"hero_seat={h} et villain_seat={v} doivent différer "
                         f"et rester dans [0, {n}).")
    if any(s <= 0.0 for s in stacks):
        raise ValueError("tapis nul ou négatif : un joueur déjà éliminé n'a "
                         "pas de place à classer — retire-le de « stacks ».")
    if sum(payouts) <= 0.0:
        raise ValueError("cagnotte nulle : sans gain à départager, l'ICM n'a "
                         "rien à dire — laisse « payouts » vide pour le chipEV.")
    return stacks, payouts


# ═══════════════════════════════════════════════════════════════════════════
# PRÉFLOP
# ═══════════════════════════════════════════════════════════════════════════


def _advise_preflop_short(spot: Spot, hero: list[int], eff_bb: float) -> Advice:
    """Tapis court heads-up : l'équilibre de Nash jam/fold tranche."""
    g = int(COMBO_TO_GROUP[combo_index(hero[0], hero[1])])
    sol = _nash_solution(round(eff_bb * 10))
    ev = float(sol.ev_jam_par_groupe[g])
    action = "JAM (tapis)" if ev >= 0 else "FOLD"
    reasons = [
        f"À {eff_bb:.1f} bb effectifs, le spot se réduit à pousser ou passer.",
        f"EV(jam) − EV(fold) = {ev:+.2f} bb à l'équilibre.",
    ]
    if abs(ev) < _INDIFFERENCE_BB:
        reasons.append(
            f"Main de frontière : |EV| = {abs(ev):.3f} bb, dans la bande où "
            "l'équilibre ne tranche pas (un solve plus long peut retourner le "
            "verdict). Les deux options se valent, l'erreur y coûte presque "
            "rien.")
    if not sol.converged:
        reasons.append("Équilibre non certifié dans le plafond d'itérations : "
                       "verdict à prendre comme indicatif.")
    return Advice(
        action=action,
        confidence="certain" if sol.converged else "indicatif",
        regime=f"Nash push/fold {eff_bb:.1f} bb",
        ev_bb=ev,
        hand=_hand_label(hero),
        reasons=reasons,
        assumptions=[
            "Heads-up, chipEV : les jetons sont supposés linéaires. Renseigne "
            "« stacks » et « payouts » pour le régime ICM — il resserre le "
            "call du vilain (aucun contre-exemple sur 36 configurations "
            "balayées) mais déplace le jam dans les deux sens : il l'élargit "
            "dans 19 de ces 36 configurations.",
        ],
    )


def _advise_preflop_icm(spot: Spot, hero: list[int],
                        stacks: tuple[float, ...],
                        payouts: tuple[float, ...]) -> Advice:
    """Tournoi tapis court : Nash jam/fold résolu en $EV ICM, écart au chipEV.

    Le même équilibre est calculé deux fois — en jetons puis en dollars
    Malmuth-Harville — parce que c'est l'écart entre les deux qui a une
    valeur d'étude : il chiffre ce que la structure de gains ajoute (ou
    retire) à une décision correcte en cash game.
    """
    h, v = spot.hero_seat, spot.villain_seat
    bb = spot.big_blind if spot.big_blind > 0 else 1.0
    deci = tuple(int(round(s / bb * 10)) for s in stacks)
    if any(d <= 0 for d in deci):
        raise ValueError("un tapis vaut moins d'un dixième de big blind : "
                         "hors du domaine du solveur push/fold.")
    stacks_bb = tuple(d / 10.0 for d in deci)
    ante_deci = int(round(spot.ante / bb * 10))
    eff_bb = min(stacks_bb[h], stacks_bb[v])
    if not (1.0 + ante_deci / 10.0 < eff_bb <= MAX_PUSHFOLD_BB):
        return _advise_preflop_deep(spot, hero)

    payouts_key = tuple(float(p) for p in payouts)
    sol = _nash_icm(deci, payouts_key, h, v, ante_deci)
    chip = _nash_solution(int(round(eff_bb * 10)), ante_deci)
    g = int(COMBO_TO_GROUP[combo_index(hero[0], hero[1])])
    ev_icm = float(sol.ev_jam_par_groupe[g])
    ev_chip = float(chip.ev_jam_par_groupe[g])
    bf = _bubble(stacks_bb, payouts_key, h, v)
    prime = risk_premium(bf)
    cagnotte = sum(payouts_key)

    action = "JAM (tapis)" if ev_icm >= 0 else "FOLD"
    reasons = [
        f"{len(stacks_bb)} joueurs "
        f"({', '.join(f'{s:.1f}' for s in stacks_bb)} bb), gains "
        f"{', '.join(f'{p:g}' for p in payouts_key)} : à {eff_bb:.1f} bb "
        "effectifs le spot se réduit à pousser ou passer.",
        f"EV(jam) − EV(fold) = {ev_icm:+.4f} en unités de gain "
        f"({ev_icm / cagnotte * 100:+.3f} % de la cagnotte) ; le même spot "
        f"vaut {ev_chip:+.2f} bb en chipEV.",
        f"Bubble factor héros → vilain {bf:.2f} : un jeton risqué pèse "
        f"{bf:.2f} fois un jeton gagné (prime de risque "
        f"{prime * 100:+.1f} points d'équité).",
        f"Ranges d'équilibre : jam {chip.jam_pct * 100:.1f} % (chipEV) → "
        f"{sol.jam_pct * 100:.1f} % (ICM) ; call du vilain "
        f"{chip.call_pct * 100:.1f} % → {sol.call_pct * 100:.1f} %.",
    ]
    if ev_chip >= 0.0 > ev_icm:
        reasons.append("Jam correct en jetons, perdant en dollars : c'est "
                       "exactement l'erreur que l'ICM sanctionne.")
    elif ev_icm >= 0.0 > ev_chip:
        reasons.append("Fold en jetons, jam en dollars : la structure de "
                       "gains paie ici la prise de risque.")
    if abs(bf - 1.0) < 1e-6:
        reasons.append("Bubble factor 1 : cette structure rend les jetons "
                       "linéaires, l'ICM ne change donc rien au chipEV.")
    if abs(ev_icm) < _INDIFFERENCE_CAGNOTTE * cagnotte:
        reasons.append(
            f"Main de frontière : |EV| = {abs(ev_icm) / cagnotte * 100:.3f} % "
            "de la cagnotte, dans la bande où l'équilibre ne tranche pas — un "
            "solve plus long peut retourner le verdict.")
    tete_a_tete = len(stacks_bb) == 2
    if not tete_a_tete:
        reasons.append(
            f"Verdict indicatif : à {len(stacks_bb)} joueurs le solve suppose "
            "les autres déjà couchés, ce qui est une hypothèse sur leur "
            "stratégie et non un résultat.")
    if not sol.converged:
        reasons.append("Équilibre non certifié dans le plafond d'itérations : "
                       "verdict à prendre comme indicatif.")

    return Advice(
        action=action,
        confidence="certain" if (sol.converged and tete_a_tete) else "indicatif",
        regime=f"Nash push/fold ICM {eff_bb:.1f} bb ({len(stacks_bb)} joueurs)",
        ev_bb=ev_chip,
        ev_icm=ev_icm,
        bubble=bf,
        hand=_hand_label(hero),
        reasons=reasons,
        assumptions=[
            "Le solveur place le héros en small blind face au vilain en big "
            "blind." if tete_a_tete else
            "Le solveur place le héros en small blind face au vilain en big "
            "blind ; les autres joueurs sont supposés déjà couchés — leurs "
            "tapis pèsent dans le $EV mais ils ne contestent pas le pot "
            "(repli N-way de solver.pushfold).",
            "$EV Malmuth-Harville : P(finir à la place k) découle des seuls "
            "tapis — ni niveau de jeu, ni position, ni blindes futures.",
        ],
    )


def _advise_preflop_deep(spot: Spot, hero: list[int]) -> Advice:
    """Tapis profond : comparaison à la range d'ouverture de la position."""
    pos = spot.position.upper()
    preset = GTO_PRESETS.get(pos)
    label = _hand_label(hero)
    if preset is None:
        return Advice(
            action="—",
            confidence="indicatif",
            regime="pas de chart pour cette position",
            hand=label,
            reasons=[f"Position « {spot.position} » inconnue ; "
                     f"positions couvertes : {', '.join(GTO_PRESETS)}."],
        )
    rng = parse_range(preset)
    weight = float(rng.weights[combo_index(hero[0], hero[1])])
    if weight >= 0.99:
        action, why = "OUVRIR (relance)", "Dans la range d'ouverture standard."
    elif weight <= 0.01:
        action, why = "FOLD", "Hors de la range d'ouverture standard."
    else:
        action = f"MIXTE — ouvrir {weight * 100:.0f} % du temps"
        why = "Main de frontière : le solveur la joue en fréquence mixte."
    return Advice(
        action=action,
        confidence="indicatif",
        regime=f"chart d'ouverture {pos}",
        hand=label,
        reasons=[why],
        assumptions=[
            "Chart 6-max 100 bb sans ante, approximative (à recalibrer sur "
            "tes propres solves) — et suppose que personne n'a ouvert devant.",
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# POSTFLOP
# ═══════════════════════════════════════════════════════════════════════════


def _hero_en_position(spot: Spot) -> bool:
    """Vrai si le héros parle en dernier postflop.

    La small blind est le cas litigieux : elle est en position en heads-up,
    hors de position dès qu'un bouton s'intercale — d'où le test sur le
    nombre de joueurs.
    """
    pos = spot.position.strip().upper()
    if pos in _POSITIONS_IP:
        return True
    return pos == "SB" and spot.players == 2


def _libelle(label: str, montant: float, pot: float) -> str:
    """Nom d'affichage français d'une action de l'arbre du solveur postflop."""
    if label == "check":
        return "CHECK"
    if label == "fold":
        return "FOLD"
    if label == "call":
        return f"PAYER {montant:g}"
    part = montant / pot * 100.0 if pot > 0 else 0.0
    if label == "all-in":
        return f"TAPIS ({montant:g}, {part:.0f} % du pot)"
    return f"MISER {montant:g} ({part:.0f} % du pot)"


def _range_hero(spot: Spot, hero: list[int]) -> tuple[Range, str, bool]:
    """Range héros du solve river : (range, spécification lue, main ajoutée ?).

    Sans ``hero_range`` déclarée, on suppose la même range que le vilain. Un
    solve sur ta seule main serait transparent — le vilain la connaîtrait et
    ne paierait jamais un bluff — donc la fréquence de bluff, qui est la
    question posée, n'aurait plus de sens.
    """
    spec = spot.hero_range or spot.villain
    poids = _villain_range(spec).weights.copy()
    k = combo_index(hero[0], hero[1])
    ajoutee = poids[k] < 1.0
    poids[k] = 1.0
    return Range(poids), spec, ajoutee


def _solve_river(spot: Spot, hero: list[int], board: list[int]) -> Advice | None:
    """Résout la river au CFR et rend la taille, la fréquence et l'EV.

    Rend ``None`` quand le spot ne s'y prête pas — budget nul, pot ou tapis
    inconnu, arbre impossible à construire — auquel cas l'appelant retombe
    sur le régime d'équité.

    L'accès aux nœuds internes du solveur (``_nodes``, ``_traverse_fixed``,
    ``_compat_mass``) est nécessaire ici : l'API publique agrège la stratégie
    et l'EV sur toute la range, alors que le conseil porte sur UN combo — le
    tien.
    """
    budget = float(spot.solver_budget_s)
    tailles = _nombres(spot.bet_sizes)
    if budget <= 0.0 or spot.pot <= 0.0 or spot.stack <= 0.0 or not tailles:
        return None

    rng_hero, spec_hero, ajoutee = _range_hero(spot, hero)
    rng_vil = _villain_range(spot.villain)
    en_position = _hero_en_position(spot)
    try:
        if en_position:
            # le vilain a checké : on lui retire toute mise à la racine, ce qui
            # fait démarrer l'arbre sur TA décision et rend l'EV conditionnelle
            solver = PostflopSolver(
                board, rng_vil, rng_hero, pot=spot.pot, stack=spot.stack,
                oop_bet_fracs=(), ip_bet_fracs=tailles, raise_fracs=tailles,
                max_bets=spot.max_bets)
            place, chemin = IP, ("check",)
        else:
            solver = PostflopSolver(
                board, rng_hero, rng_vil, pot=spot.pot, stack=spot.stack,
                oop_bet_fracs=tailles, ip_bet_fracs=tailles,
                raise_fracs=tailles, max_bets=spot.max_bets)
            place, chemin = OOP, ()
    except PostflopError:
        return None

    cartes = solver.players[place].cards
    trouve = np.nonzero(
        ((cartes[:, 0] == hero[0]) & (cartes[:, 1] == hero[1]))
        | ((cartes[:, 0] == hero[1]) & (cartes[:, 1] == hero[0])))[0]
    if trouve.size == 0:
        return None
    k = int(trouve[0])

    depart = time.perf_counter()
    faites = 0
    while faites < RIVER_SOLVE_ITERATIONS:
        solver.solve(_RIVER_CHUNK)
        faites += _RIVER_CHUNK
        if time.perf_counter() - depart >= budget:
            break
    ecoule = time.perf_counter() - depart

    idx = solver.node_at(chemin)
    node = solver._nodes[idx]
    freqs = solver.average_strategy(idx)[:, k]
    ordre = [int(a) for a in np.argsort(-freqs)]
    tete = ordre[0]
    freq = float(freqs[tete])
    melange = tuple((_libelle(node.labels[a], node.amounts[a], spot.pot),
                     float(freqs[a])) for a in ordre)
    if freq >= _ACTION_PURE:
        action = _libelle(node.labels[tete], node.amounts[tete], spot.pot)
    else:
        action = "MIXTE — " + " / ".join(
            f"{lab} {f * 100:.0f} %" for lab, f in melange if f >= _MIXTE_MIN)

    q = solver.players[1 - place].weights
    cfv = solver._traverse_fixed(solver._root, place, q.copy(), -1, False)
    compat = solver._compat_mass(place, q, -1)
    ev = float(cfv[k] / compat[k]) if compat[k] > 0 else 0.0
    expl = solver.exploitability()

    taille = mdf = defense = None
    if node.labels[tete] != "check":
        taille = float(node.amounts[tete])
        mdf = minimum_defence_frequency(spot.pot, taille)
        enfant = node.children[tete]
        if enfant >= 0:
            reponse = solver._nodes[enfant]
            sigma_v = solver.average_strategy(enfant)
            poids_v = solver.players[reponse.player].weights
            garde = [a for a, lab in enumerate(reponse.labels) if lab != "fold"]
            total = float(poids_v.sum())
            if total > 0:
                defense = float(sum(float((sigma_v[a] * poids_v).sum())
                                    for a in garde) / total)

    reasons = [
        "Personne n'a misé : la question est la taille et la fréquence, pas "
        "une cote à payer.",
        f"River résolue au CFR range contre range : {faites} itérations en "
        f"{ecoule:.2f} s, exploitabilité {expl * 100:.2f} % du pot "
        "(0 = équilibre exact).",
        f"EV de ta main dans ce spot : {ev:+.4f} sur un pot de {spot.pot:g} "
        f"(un check-abattage perdu vaudrait 0).",
    ]
    if mdf is not None:
        reasons.append(
            f"Face à cette mise, le vilain doit défendre {mdf * 100:.1f} % "
            "(MDF) sous peine d'être bluffable à volonté"
            + (f" ; à l'équilibre il défend {defense * 100:.1f} %."
               if defense is not None else "."))
    if freq < _ACTION_PURE:
        reasons.append("Le solveur mélange : aucune option ne domine, la "
                       "fréquence EST la réponse.")
    if faites < RIVER_SOLVE_ITERATIONS:
        reasons.append(f"Budget de {budget:.1f} s atteint avant les "
                       f"{RIVER_SOLVE_ITERATIONS} itérations visées : "
                       "l'exploitabilité ci-dessus dit ce que ça coûte.")

    assumptions = [
        f"Ranges supposées : héros « {spec_hero} », vilain "
        f"« {spot.villain} ». L'équilibre est exact POUR CES RANGES — "
        "c'est la range qui est l'hypothèse, pas le solve.",
        f"Héros {'en position' if en_position else 'hors de position'} ; "
        f"tailles testées {', '.join(f'{t:g}' for t in tailles)} × pot, "
        f"{spot.max_bets} mise(s) par street, tapis {spot.stack:g}, "
        f"arbre de {len(solver._nodes)} nœuds.",
    ]
    if ajoutee:
        assumptions.append("Ta main n'était pas dans la range héros supposée : "
                           "elle y a été ajoutée avec un poids 1.")
    if _nombres(spot.payouts):
        assumptions.append(
            "Solve en jetons : la structure de gains n'entre pas dans l'arbre "
            "postflop, alors que sous ICM le risque coûte plus cher — lis les "
            "tailles ci-dessus comme un plafond.")

    return Advice(
        action=action,
        confidence="indicatif",
        regime="solveur CFR à la river",
        mdf=mdf,
        ev_bb=ev / (spot.big_blind if spot.big_blind > 0 else 1.0),
        size=taille,
        frequency=freq,
        mix=melange,
        villain_defence=defense,
        hand=_hand_label(hero),
        reasons=reasons,
        assumptions=assumptions,
    )


def _advise_postflop(spot: Spot, hero: list[int], board: list[int]) -> Advice:
    """Postflop : solve river si personne n'a misé, sinon équité vs cotes."""
    if spot.bet <= 0.0 and len(board) == 5:
        conseil = _solve_river(spot, hero, board)
        if conseil is not None:
            return conseil

    villain = _villain_range(spot.villain)
    eq = equity_vs_range(hero, villain, board).equity
    label = _hand_label(hero)
    street = {3: "flop", 4: "turn", 5: "river"}[len(board)]
    assumptions = [
        f"Range adverse supposée « {spot.villain} » — c'est une hypothèse, "
        "pas une lecture.",
    ]

    if spot.bet <= 0:
        # personne n'a misé : miser pour la valeur au-dessus de la parité
        action = "MISER (valeur)" if eq >= 0.55 else (
            "CHECK" if eq >= 0.40 else "CHECK (ou bluff choisi)")
        reasons = [
            f"Au {street}, ta main réalise {eq * 100:.1f} % d'équité contre "
            "cette range.",
            "Personne n'a misé : la question est value ou contrôle du pot.",
        ]
        if eq >= 0.55:
            reasons.append("Au-dessus de 55 %, une mise de valeur est payée "
                           "par assez de mains plus faibles.")
        return Advice(action=action, confidence="indicatif",
                      regime=f"équité exacte au {street}", equity=eq,
                      hand=label, reasons=reasons, assumptions=assumptions)

    alpha_cash = required_equity(spot.pot, spot.bet)
    mdf = minimum_defence_frequency(spot.pot, spot.bet)
    stacks, payouts = _tournoi(spot)
    bf = None
    alpha = alpha_cash
    if stacks:
        bf = _bubble(stacks, payouts, spot.hero_seat, spot.villain_seat)
        alpha = icm_required_equity(spot.pot, spot.bet, bf)
    margin = eq - alpha
    ev_call = eq * (spot.pot + spot.bet) - (1 - eq) * spot.bet
    if spot.big_blind > 0:
        ev_call /= spot.big_blind

    if margin > 0.06:
        action = "CALL (confortable)"
    elif margin > 0.0:
        action = "CALL (juste)"
    elif margin > -0.04:
        action = "MARGINAL — proche de l'indifférence"
    else:
        action = "FOLD"

    reasons = [
        f"Au {street}, ta main vaut {eq * 100:.1f} % contre cette range ; "
        f"il en faut {alpha * 100:.1f} % pour payer {spot.bet:.0f} dans "
        f"{spot.pot:.0f}.",
        f"Marge : {margin * 100:+.1f} points d'équité.",
    ]
    if bf is not None:
        reasons.append(
            f"Bubble factor {bf:.2f} : l'équité requise passe de "
            f"{alpha_cash * 100:.1f} % (cotes brutes) à {alpha * 100:.1f} % "
            "en tournoi.")
        if eq >= alpha_cash > 0 and eq < alpha:
            reasons.append("Call correct en cash game, fold en tournoi : "
                           "c'est LE fold que les joueurs de cash ratent.")
        assumptions.append(
            "EV affichée en jetons : sous ICM la vraie monnaie est le $EV, "
            "la marge d'équité ci-dessus est le critère qui tranche.")
    # seuil de bascule : la décision tient-elle si la range change ?
    reasons.append(
        f"Bascule à {alpha * 100:.1f} % d'équité — si tu le crois plus "
        f"{'serré' if margin > 0 else 'large'} que supposé, la décision "
        "s'inverse."
    )
    if eq >= 0.75:
        reasons.append("Assez fort pour relancer plutôt que simplement payer.")
    return Advice(
        action=action, confidence="indicatif",
        regime=f"équité exacte au {street} vs cotes du pot"
               + (" (ICM)" if bf is not None else ""),
        equity=eq, required=alpha, mdf=mdf, ev_bb=ev_call, bubble=bf,
        hand=label, reasons=reasons, assumptions=assumptions,
    )


def advise(spot: Spot) -> Advice:
    """Rend le verdict sur un spot déjà joué : que fallait-il faire ?

    Examples
    --------
    >>> a = advise(Spot(hero="Ah Ad", stack=10, big_blind=1, position="BTN"))
    >>> a.action.startswith("JAM")
    True

    Le même spot dans un tournoi 3-max où le héros est couvert et où un
    joueur est à un souffle de sauter — l'ICM resserre :

    >>> a = advise(Spot(hero="Kh 9d", big_blind=1, stacks="10, 40, 1",
    ...                 payouts="65, 35", players=3))
    >>> a.action
    'FOLD'
    """
    hero = parse_cards(spot.hero)
    if len(hero) != 2:
        raise ValueError(f"il faut exactement 2 cartes pour le héros, "
                         f"reçu {len(hero)} depuis « {spot.hero} ».")
    board = parse_cards(spot.board) if spot.board else []
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board de 0, 3, 4 ou 5 cartes, reçu {len(board)}.")
    if set(hero) & set(board):
        raise ValueError("une carte du board est aussi dans ta main.")

    if board:
        return _advise_postflop(spot, hero, board)

    stacks, payouts = _tournoi(spot)
    if stacks:
        return _advise_preflop_icm(spot, hero, stacks, payouts)

    eff_bb = spot.stack / spot.big_blind if spot.big_blind > 0 else 0.0
    if spot.players == 2 and 1.0 <= eff_bb <= MAX_PUSHFOLD_BB:
        return _advise_preflop_short(spot, hero, eff_bb)
    return _advise_preflop_deep(spot, hero)
