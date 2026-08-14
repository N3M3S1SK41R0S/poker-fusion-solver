#!/usr/bin/env python
r"""Banc CORPUS — nos conseils auraient-ils été bons sur 10 000 mains de Pluribus ?

    python banc_corpus_pluribus.py                  le rejeu complet
    python banc_corpus_pluribus.py --mains 500      un échantillon rapide
    python banc_corpus_pluribus.py --verifier       seulement l'audit du lecteur
    python banc_corpus_pluribus.py --population     seulement les repères VPIP/PFR
    python banc_corpus_pluribus.py --villain large  autre range adverse supposée
    python banc_corpus_pluribus.py --joueur MrBlue  rejouer un humain plutôt que le bot
    python banc_corpus_pluribus.py --corpus CHEMIN  autre racine de fichiers .phh

Durées mesurées sur la machine de développement : 6 s pour ``--verifier``
(lecture seule des 10 000 fichiers), 11 s pour ``--population``, **23 min**
pour le rejeu complet. Le coût est presque entièrement dans l'équité exacte
au flop — 990 runouts × la range adverse à chaque spot, soit ≈ 350 ms — et
il n'est pas réductible sans renoncer à l'exactitude, donc au caractère
rejouable au chiffre près.

Résultat du rejeu complet au 14 août 2026 (icm.py corrigé du seuil PKO,
range adverse « moyenne », intervalle de Wilson à 95 %) ::

    préflop · tapis court (< 15 bb)       17 spots · accord 84,6 %
    préflop · profond (≥ 15 bb)       10 282 spots · accord 86,3 % [12,9;14,4]
    postflop · flop                    2 317 spots · accord 63,0 % [35,1;39,1]
    postflop · turn                    1 513 spots · accord 63,8 % [33,8;38,7]
    postflop · river                   1 040 spots · accord 64,4 % [32,7;38,5]
    TOUS RÉGIMES                      15 169 spots · accord 78,0 % [21,3;22,7]

Les intervalles portent le DÉSACCORD. Le désaccord postflop (~36 %) n'est
pas une mesure du postflop seul : la famille la plus nombreuse d'un corpus
6-max est la DÉFENSE face à une ouverture, à laquelle le conseiller répond
avec la chart d'ouverture — un modèle appliqué hors de son domaine, compté
tel quel (voir la section 9 du rapport). Cash 6-max à 100 bb sans ICM :
aucun de ces chiffres ne valide le conseil en tournoi ni le tapis court.

Pourquoi ce fichier existe
--------------------------
Le logiciel n'avait jamais été confronté à autre chose qu'aux mains de son
utilisateur. Pluribus (Brown & Sandholm, *Science*, 2019) est le premier bot
surhumain au poker à six joueurs, et ses 10 000 mains sont publiques. Ce n'est
pas la vérité — c'est une **référence forte**. La lecture qui en découle :

* un désaccord PONCTUEL ne dit rien : deux stratégies correctes divergent
  sur les mains de frontière, et Pluribus lui-même mélange ;
* un désaccord SYSTÉMATIQUE sur une famille de spots signale un défaut chez
  nous — ou, au minimum, une famille où notre modèle n'a rien à dire et
  répond quand même. C'est le livrable principal de ce banc.

Ce que le banc mesure exactement
--------------------------------
Il rejoue chaque main, s'arrête à chaque décision du joueur suivi, construit
le ``Spot`` correspondant avec l'information disponible À CET INSTANT (jamais
les cartes à venir), demande au conseiller ce qu'il aurait fait, et compare
sur un axe binaire propre à chaque régime :

============================  =========================================
préflop                       jouer (ouvrir/suivre) ou passer
postflop face à une mise      continuer (suivre/relancer) ou passer
postflop sans mise            miser ou checker
============================  =========================================

L'axe binaire est celui que le modèle du conseiller sait trancher : sa
comparaison équité / cotes du pot répond « payer ou coucher », pas « payer ou
relancer ». Le détail complet verdict × action jouée est imprimé sous chaque
famille, pour que l'agrégat ne cache pas ce qu'il résume.

Les taux sont rendus **par régime**, jamais en un seul chiffre : préflop tapis
court (< 15 bb effectifs), préflop profond, puis postflop **rue par rue**
(flop, turn, river). Un taux global moyennerait des moteurs différents — chart
d'ouverture, Nash push/fold, équité à 990 runouts, équité exacte — et ne
dirait rien. Le total « tous régimes confondus » n'est imprimé que pour la
comptabilité, avec la réserve écrite à côté.

Chaque régime et chaque famille marquée sont ensuite CARACTÉRISÉS sur quatre
axes comptés séparément — position, profondeur, texture de board
(appariement, couleur, connexité) et type de main. Séparément, et non
croisés : le croisement ferait des cases de dix spots, où le hasard fabrique
n'importe quel motif.

Ce que le banc NE mesure PAS — à lire avant de citer un chiffre
---------------------------------------------------------------
1. **Pluribus joue du CASH GAME six-max, sans ICM.** L'utilisateur joue des
   tournois. Un taux d'accord mesuré ici ne valide RIEN du conseil en
   tournoi : ni le régime ICM, ni le bubble factor, ni le push/fold.
2. **Les tapis valent 100 bb à chaque main** (10 000 jetons, blindes 50/100,
   remis à niveau entre les mains). Le corpus ne contient donc aucun spot de
   tapis court : le solveur push/fold, cœur du conseil en Twister, sort de ce
   banc sans une seule mesure. Ce n'est pas une déduction — le rapport
   COMPTE les moteurs réellement sollicités (section 2, « quel moteur a
   répondu ») et affiche le total du régime push/fold.
3. **Le postflop du conseiller suppose UN adversaire.** Les pots multiway y
   sont hors domaine ; ils sont comptés à part, jamais fondus dans le total.
4. **La range adverse est une hypothèse** (``--villain``, « moyenne » par
   défaut). Le taux d'accord postflop en dépend ; le banc le remesure sur
   demande avec « large » et « serree ».
5. **Le coût en bb « selon notre modèle » est auto-référentiel** : il chiffre
   la conviction du conseiller, pas la vérité. Seul le coût RÉALISÉ des
   divergences à la river heads-up est indépendant de nous — le corpus donne
   les cartes de tout le monde, l'abattage est donc calculable exactement.

Rejouabilité
------------
Aucun aléa : ordre de lecture des fichiers stable (:func:`iter_phh_files`),
équités exactes par énumération dès le flop, solveur river désactivé par
défaut. Deux exécutions rendent les mêmes chiffres.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pfs.analysis.reperes import CORPUS_PLURIBUS  # noqa: E402
from pfs.analysis.spot_advisor import Advice, Spot, advise, parse_cards  # noqa: E402
from pfs.core.equity import evaluate5, evaluate7, hand_category  # noqa: E402
from pfs.core.range_model import GTO_PRESETS, parse_range  # noqa: E402
from pfs.data.hand_history import (  # noqa: E402
    ActionType,
    ParsedHand,
    Street,
    player_key,
)
from pfs.data.phh import (  # noqa: E402
    Decision,
    PhhError,
    iter_decisions,
    iter_phh_files,
    parse_phh_file,
    phh_players,
    positions_par_siege,
)

#: Même résolution que partout ailleurs — ``PFS_CORPUS``, sinon
#: ``<parent du dépôt>/corpus`` (voir :func:`pfs.analysis.reperes.
#: racine_corpus`) : une seule définition du chemin du corpus dans le projet.
#: ``--corpus`` reste le moyen d'en désigner un autre.
CORPUS_DEFAUT = CORPUS_PLURIBUS

N_MIN_FAMILLE = 100
"""Effectif en deçà duquel une famille n'est jamais déclarée systématique.

Sous 100 spots, la borne de Wilson est si large qu'une famille purement
aléatoire peut afficher 60 % de désaccord. Le seuil ne dit pas qu'en dessous
il n'y a rien : il dit que le banc ne sait pas.
"""

SEUIL_SYSTEMATIQUE = 0.50
"""Un désaccord est « systématique » si la BORNE BASSE de son intervalle de
Wilson à 95 % dépasse ce seuil.

Le critère porte sur la borne basse, pas sur le taux observé : une famille de
120 spots à 55 % de désaccord a une borne basse de 46 % et n'est donc PAS
déclarée systématique. Il faut que même la lecture pessimiste conclue « on
diverge plus souvent qu'on ne s'accorde ».
"""

SEUIL_MARQUE = 0.25
"""Second palier, plus bas : « désaccord marqué ». Un quart de divergence sur
une famille nombreuse mérite d'être lu, sans mériter le mot systématique."""

N_MIN_AXE = 50
"""Effectif minimal de CHAQUE côté pour qu'un axe de caractérisation parle.

Les axes (texture de board, type de main, position, profondeur) découpent une
famille déjà découpée : leurs cases sont petites, et c'est exactement là que
la pêche aux corrélations attrape des motifs de hasard. Le banc n'annonce un
écart d'axe que si les deux côtés comparés atteignent ce seuil ET que leurs
intervalles de Wilson à 95 % sont DISJOINTS — un critère plus dur que « les
taux diffèrent ».
"""

SEUIL_TAPIS_COURT_BB = 15.0
"""Frontière entre « préflop tapis court » et « préflop profond », en bb.

C'est l'axe demandé par la question posée au banc, et il ne coïncide PAS avec
la bascule interne du conseiller : ``advise()`` ne passe en Nash push/fold
qu'à ``players == 2`` et sous ``MAX_PUSHFOLD_BB`` (25 bb). Un spot à 12 bb
effectifs à cinq joueurs est donc « tapis court » pour ce rapport et traité
par la chart d'ouverture par le conseiller. Le rapport imprime les deux
lectures — le régime par profondeur (section 2) et le moteur réellement
sollicité (« quel moteur a répondu ») — pour que l'écart soit visible plutôt
que déduit.
"""

REGIME_COURT = f"préflop · tapis court (< {SEUIL_TAPIS_COURT_BB:.0f} bb)"
REGIME_PROFOND = f"préflop · profond (≥ {SEUIL_TAPIS_COURT_BB:.0f} bb)"

ORDRE_REGIMES: tuple[str, ...] = (
    REGIME_COURT,
    REGIME_PROFOND,
    "postflop · flop",
    "postflop · turn",
    "postflop · river",
)
"""Ordre d'affichage des régimes.

Explicite plutôt qu'alphabétique : l'ordre alphabétique intercalerait la
river entre le flop et le turn, et un lecteur qui parcourt la colonne des
taux lirait une progression par rue qui n'existe pas.
"""


# ═══════════════════════════════════════════════════════════════════════════
# OUTILS
# ═══════════════════════════════════════════════════════════════════════════


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Intervalle de Wilson à 95 % pour une proportion.

    Choisi plutôt que l'intervalle normal parce qu'il reste correct aux
    extrêmes (0 succès, ou 100 %), là où le normal donne des bornes hors
    ``[0, 1]`` — et les familles intéressantes de ce banc sont justement
    celles où le désaccord frôle 0 ou 100 %.

    >>> low, high = wilson(50, 100)
    >>> round(low, 4), round(high, 4)
    (0.4038, 0.5962)
    >>> wilson(0, 0)
    (0.0, 1.0)
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def _force7(cartes: list[int]) -> int:
    """Force d'une main de 7 cartes (plus grand = plus fort)."""
    return int(evaluate7(np.asarray([cartes], dtype=np.int64))[0])


_CHIFFRES = re.compile(r"[\d]+(?:[.,]\d+)?")


def _moteur(regime: str) -> str:
    """Nom du moteur qui a tranché, débarrassé de ses paramètres numériques.

    ``Advice.regime`` porte les valeurs du spot (« Nash push/fold 12.5 bb »,
    « chart d'ouverture CO ») : les agréger tels quels donnerait des centaines
    de lignes. Ce qui intéresse le rapport est QUEL moteur a répondu, pas
    avec quel tapis.

    >>> _moteur("Nash push/fold 12.5 bb")
    'Nash push/fold N bb'
    >>> _moteur("équité exacte au flop vs cotes du pot")
    'équité exacte au flop vs cotes du pot'
    """
    return _CHIFFRES.sub("N", regime)


def _moyenne_ic(xs: list[float]) -> tuple[float, float, float]:
    """(moyenne, écart-type, erreur-type) d'un échantillon.

    L'écart-type est celui de l'échantillon (dénominateur *n−1*) : il décrit
    la dispersion des spots, qui est énorme quand la mesure est un résultat
    d'abattage. L'erreur-type, elle, décrit la précision de la moyenne, et
    c'est elle seule qui autorise à conclure.

    >>> m, sd, err = _moyenne_ic([1.0, 2.0, 3.0, 4.0])
    >>> round(m, 6), round(sd, 6), round(err, 6)
    (2.5, 1.290994, 0.645497)
    >>> _moyenne_ic([])
    (0.0, 0.0, 0.0)
    """
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = sum(xs) / n
    if n == 1:
        return (m, 0.0, 0.0)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return (m, sd, sd / math.sqrt(n))


@dataclass
class Famille:
    """Comptes d'une famille de spots.

    ``indecis`` regroupe ce que le conseiller refuse de trancher pour un spot
    isolé — « MIXTE » (fréquence) et « MARGINAL » (indifférence) — et
    ``refus`` ce sur quoi il n'a aucun modèle du tout. Les deux sont exclus
    du dénominateur du taux d'accord : compter un « je ne sais pas » comme un
    désaccord gonflerait artificiellement le défaut, le compter comme un
    accord le cacherait.
    """

    n: int = 0
    accord: int = 0
    desaccord: int = 0
    indecis: int = 0
    refus: int = 0
    erreurs: int = 0
    croise: Counter = field(default_factory=Counter)
    couts_modele: list[float] = field(default_factory=list)
    #: axe → valeur → [accords, désaccords]. Ne compte QUE les spots tranchés,
    #: pour la même raison que le taux : un « je ne sais pas » rangé d'un côté
    #: fabriquerait l'écart qu'on cherche à mesurer.
    axes: dict[str, dict[str, list[int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(
            lambda: [0, 0])))

    def compter_axes(self, axes: dict[str, str], accord: bool) -> None:
        """Enregistre un spot TRANCHÉ dans chacun de ses axes."""
        for nom, valeur in axes.items():
            self.axes[nom][valeur][0 if accord else 1] += 1

    def fusionner_axes(self, autre: "Famille") -> None:
        """Ajoute les comptes d'axes de ``autre`` — pour la ligne « tous
        régimes confondus », qui doit porter les mêmes axes que ses parties."""
        for nom, valeurs in autre.axes.items():
            for valeur, (a, d) in valeurs.items():
                cible = self.axes[nom][valeur]
                cible[0] += a
                cible[1] += d

    @property
    def tranches(self) -> int:
        return self.accord + self.desaccord

    @property
    def taux_desaccord(self) -> float:
        return self.desaccord / self.tranches if self.tranches else 0.0

    def sens(self) -> tuple[int, int]:
        """(désaccords où NOUS jouons plus, désaccords où nous jouons moins).

        « Plus » veut dire : le conseiller met de l'argent (ouvrir, payer,
        miser) là où le joueur suivi s'arrête. « Moins » l'inverse.

        C'est cette asymétrie, et non le taux brut, qui distingue un défaut
        d'un simple bruit de frontière. Deux stratégies correctes divergent
        sur les mains limites — mais elles divergent alors dans les DEUX
        sens, à peu près également. Un déséquilibre marqué dit qu'un des deux
        modèles penche.
        """
        plus = moins = 0
        for (vu, fait), k in self.croise.items():
            if vu in ("MIXTE", "INDECIS", "REFUS") or vu.startswith("?"):
                continue
            actif_vu = vu in ("AGRESSER", "CONTINUER")
            actif_fait = fait in ("AGRESSER", "CONTINUER")
            if actif_vu and not actif_fait:
                plus += k
            elif actif_fait and not actif_vu:
                moins += k
        return plus, moins


@dataclass
class StatsJoueur:
    """VPIP / PFR d'un joueur, au total et par position."""

    mains: int = 0
    vpip: int = 0
    pfr: int = 0
    par_position: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0, 0]))

    def ajouter(self, position: str, vpip: bool, pfr: bool) -> None:
        self.mains += 1
        self.vpip += int(vpip)
        self.pfr += int(pfr)
        c = self.par_position[position]
        c[0] += 1
        c[1] += int(vpip)
        c[2] += int(pfr)


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFICATION D'UN SPOT
# ═══════════════════════════════════════════════════════════════════════════


def intention_conseiller(a: Advice) -> str:
    """Verdict du conseiller ramené à l'axe binaire de son régime.

    ``AGRESSER`` (ouvrir, miser, jam), ``CONTINUER`` (payer), ``CHECK``,
    ``PASSER`` (coucher), ``MIXTE`` / ``INDECIS`` (il ne tranche pas),
    ``REFUS`` (aucun modèle).

    >>> from pfs.analysis.spot_advisor import Advice
    >>> intention_conseiller(Advice("OUVRIR (relance)", "indicatif", "chart"))
    'AGRESSER'
    >>> intention_conseiller(Advice("MARGINAL — proche de l'indifférence",
    ...                             "indicatif", "équité"))
    'INDECIS'
    """
    act = a.action
    if act == "—":
        return "REFUS"
    if act.startswith("MIXTE"):
        return "MIXTE"
    if act.startswith("MARGINAL"):
        return "INDECIS"
    if act.startswith(("OUVRIR", "MISER", "JAM", "TAPIS")):
        return "AGRESSER"
    if act.startswith(("CALL", "PAYER")):
        return "CONTINUER"
    if act.startswith("CHECK"):
        return "CHECK"
    if act.startswith("FOLD"):
        return "PASSER"
    return f"?{act}"


def intention_joueur(d: Decision) -> str:
    """Ce que le joueur suivi a réellement fait, sur le même vocabulaire."""
    if d.action is ActionType.FOLD:
        return "PASSER"
    if d.action is ActionType.CHECK:
        return "CHECK"
    if d.action is ActionType.CALL:
        return "CONTINUER"
    return "AGRESSER"


def profondeur_bb(d: Decision, main: ParsedHand) -> float:
    """Tapis effectif de la décision, en big blinds, à CET instant.

    ``stack_effectif`` et non ``stack`` : ce qui décide du régime est ce qui
    peut encore être misé face à quelqu'un, pas ce que le joueur a devant lui.
    Un tapis de 200 bb en face d'un adversaire à 8 bb est un spot de 8 bb.

    Rend ``math.inf`` si la big blind est nulle — le fichier serait alors
    illisible en bb, et classer un tel spot « tapis court » sur un quotient
    indéfini fabriquerait le régime le plus intéressant du rapport.
    """
    bb = main.big_blind
    return d.stack_effectif / bb if bb > 0 else math.inf


def regime_du_spot(d: Decision, prof_bb: float) -> str:
    """Régime d'un spot : préflop court / préflop profond / postflop par rue.

    Découpage imposé par la question posée : un taux global mélangerait le
    push/fold, l'ouverture à 100 bb et trois rues postflop, qui ne sollicitent
    pas les mêmes moteurs et n'ont aucune raison d'avoir le même taux
    d'accord. La rue est portée par le régime lui-même, et non reléguée dans
    la famille, parce que le flop, le turn et la river sont trois modèles
    différents chez nous (équité à 990 runouts, à 44, exacte).
    """
    if d.street is Street.PREFLOP:
        return REGIME_COURT if prof_bb < SEUIL_TAPIS_COURT_BB else REGIME_PROFOND
    return f"postflop · {d.street.value}"


def famille_du_spot(d: Decision, limpeurs: int,
                    prof_bb: float) -> tuple[str, str]:
    """(régime, famille détaillée) d'une décision.

    Le régime est l'agrégat que demande le rapport ; la famille est la maille
    où un défaut se voit — c'est elle qui porte le test « systématique ».

    ``prof_bb`` est obligatoire, sans valeur par défaut : une profondeur
    implicite classerait silencieusement tout un corpus de tapis courts en
    « profond », et le régime le plus sensible du rapport serait vide sans
    que rien ne tombe.
    """
    regime = regime_du_spot(d, prof_bb)
    if d.street is Street.PREFLOP:
        if d.relances_avant >= 1:
            return (regime, "préflop · face à une relance")
        if limpeurs > 0:
            return (regime, "préflop · ouverture après limp")
        return (regime, f"préflop · ouverture propre · {d.position}")
    face = "face à une mise" if d.to_call > 0 else "sans mise"
    table = "heads-up" if d.actifs == 2 else f"{d.actifs} joueurs"
    return (regime, f"postflop · {face} · {table} · {d.street.value}")


# ── caractérisation : texture du board, type de main, position, profondeur ──


def texture_board(board: Sequence[str]) -> dict[str, str]:
    """Trois axes indépendants décrivant les cartes communes visibles.

    Trois axes SÉPARÉS et non un libellé unique : croiser appariement ×
    couleur × connexité donnerait huit cases par famille, soit une dizaine de
    spots chacune sur les familles les plus nombreuses de ce corpus — un
    effectif où le hasard fabrique n'importe quel motif. Chaque axe garde
    ainsi tout l'effectif de la famille.

    L'as compte comme haute ET comme basse pour la connexité : sans ça, A-2-9
    passerait pour un board déconnecté alors que la roue s'y tire.

    « Connecté » veut dire : deux rangs distincts séparés d'au plus deux crans.
    Le seuil est choisi pour que l'axe DÉCOUPE la population — une valeur qui
    écrase l'autre ne caractérise rien. Sur les 22 100 flops possibles,
    énumérés : **14 960 connectés (67,69 %) contre 7 140 secs (32,31 %)** ;
    à rangs strictement adjacents on aurait 8 944 / 22 100 (40,47 %), qui
    découpe aussi, mais laisserait les tirages ventraux du côté « sec ».

    Pouvoir de coupe des deux autres axes, mesuré de la même façon sur les
    mêmes 22 100 flops : appariement 3 796 / 18 304 (17,18 % appairés),
    couleur 1 144 / 20 956 (5,18 % à trois assorties). **L'axe couleur est
    donc déséquilibré** : sur un régime de 2 000 spots il n'y aura qu'une
    centaine de flops monotones, tout juste de quoi atteindre
    :data:`N_MIN_AXE`. C'est une limite de cet axe, pas un défaut du corpus —
    le rapport l'affiche avec son intervalle, qui sera large.

    Ces trois comptes sont épinglés par
    ``test_le_seuil_decoupe_vraiment_la_population``.

    >>> t = texture_board(("Ah", "Kh", "Qh"))
    >>> t["appariement"], t["couleur"], t["connexité"]
    ('non appairé', '3+ assorties', 'connecté')
    >>> texture_board(("2c", "7d", "Kh"))["connexité"]
    'sec'
    >>> texture_board(("Ac", "2d", "9h"))["connexité"]
    'connecté'
    >>> texture_board(("8c", "8d", "Kh"))["appariement"]
    'appairé'
    """
    cartes = parse_cards(" ".join(board))
    rangs = [c >> 2 for c in cartes]
    couleurs = [c & 3 for c in cartes]
    # 12 = as, 0 = deux : l'ordre naturel pour mesurer un écart.
    valeurs = sorted({12 - r for r in rangs})
    if 12 in valeurs:
        valeurs = [-1, *valeurs]
    return {
        "appariement": ("appairé" if len(set(rangs)) < len(rangs)
                        else "non appairé"),
        "couleur": ("3+ assorties"
                    if max(Counter(couleurs).values()) >= 3
                    else "2 assorties au plus"),
        "connexité": ("connecté"
                      if any(b - a <= 2 for a, b in zip(valeurs, valeurs[1:]))
                      else "sec"),
    }


def _code_meilleure_main(cartes: list[int]) -> int:
    """Force de la meilleure main de 5 cartes tirée de ``cartes`` (5 à 7).

    ``evaluate7`` ne prend que sept cartes ; au flop il n'y en a que cinq et
    au turn six. Énumérer les sous-mains de cinq avec ``evaluate5`` — l'oracle
    scalaire du projet — coûte au plus 21 évaluations et évite d'inventer des
    cartes pour compléter.
    """
    if len(cartes) == 7:
        return int(evaluate7(np.asarray([cartes], dtype=np.int64))[0])
    sous = np.asarray(list(itertools.combinations(cartes, 5)), dtype=np.int64)
    return int(evaluate5(sous).max())


def type_de_main(cards: Sequence[str], board: Sequence[str]) -> str:
    """Type de la main du héros, sur l'axe qui a un sens à cette rue.

    Préflop, ce qui décide est la forme des deux cartes ; postflop, ce qui
    décide est la main FAITE contre ce board. Trois paliers de chaque côté :
    au-delà, les cases deviennent trop petites pour que le banc conclue.

    Les tirages ne sont PAS distingués ici : une main à tirage couleur nu est
    comptée « hauteur ». C'est une limite assumée de cet axe — il caractérise
    la main faite, pas son potentiel, et une famille où le désaccord viendrait
    des tirages apparaîtrait donc sous « hauteur » sans être expliquée.

    >>> type_de_main(("As", "Ad"), ())
    'paire servie'
    >>> type_de_main(("As", "Kd"), ())
    'dépareillée'
    >>> type_de_main(("As", "Ks"), ())
    'assortie'
    >>> type_de_main(("As", "Kd"), ("Ah", "7c", "2d"))
    'une paire'
    >>> type_de_main(("As", "Kd"), ("Qh", "7c", "2d"))
    'hauteur'
    >>> type_de_main(("As", "Kd"), ("Ah", "Kc", "2d"))
    'deux paires ou mieux'
    """
    if not board:
        cartes = parse_cards(" ".join(cards))
        if len(cartes) != 2:
            return "?"
        if cartes[0] >> 2 == cartes[1] >> 2:
            return "paire servie"
        return "assortie" if cartes[0] & 3 == cartes[1] & 3 else "dépareillée"
    cartes = parse_cards(" ".join(cards)) + parse_cards(" ".join(board))
    if not (5 <= len(cartes) <= 7):
        return "?"
    categorie = hand_category(_code_meilleure_main(cartes))
    if categorie == "hauteur":
        return "hauteur"
    return "une paire" if categorie == "paire" else "deux paires ou mieux"


def axes_du_spot(d: Decision, prof_bb: float) -> dict[str, str]:
    """Les axes de caractérisation d'un spot, prêts à être comptés.

    Ce sont eux que la question posée exige de chiffrer : position,
    profondeur, texture de board, type de main. Ils sont comptés séparément
    pour chaque famille ET pour chaque régime — c'est au niveau du régime que
    l'effectif est assez gros pour que la comparaison entre deux valeurs d'un
    axe ait une chance de trancher.
    """
    axes = {
        "position": d.position or "?",
        "profondeur": (f"< {SEUIL_TAPIS_COURT_BB:.0f} bb"
                       if prof_bb < SEUIL_TAPIS_COURT_BB
                       else f"≥ {SEUIL_TAPIS_COURT_BB:.0f} bb"),
        "type de main": type_de_main(d.cards, d.board),
        "joueurs actifs": ("heads-up" if d.actifs == 2
                           else f"{d.actifs} joueurs"),
    }
    if d.board:
        for nom, valeur in texture_board(d.board).items():
            axes[f"board · {nom}"] = valeur
    return axes


def construire_spot(d: Decision, main: ParsedHand, villain: str,
                    budget_river: float) -> Spot:
    """Le ``Spot`` que le conseiller aurait reçu d'une capture d'écran.

    ``pot`` reçoit ``pot_gagnable`` et non ``pot`` : les jetons qu'un
    adversaire a surmisés au-delà de ce que le héros peut couvrir lui
    reviendront, et les compter fausserait la cote.
    """
    return Spot(
        hero=" ".join(d.cards),
        board=" ".join(d.board),
        pot=d.pot_gagnable,
        bet=d.to_call,
        stack=d.stack_effectif,
        big_blind=main.big_blind,
        position=d.position,
        villain=villain,
        players=d.actifs,
        solver_budget_s=budget_river,
    )


def _accord(regime_face: str, conseil: str, joue: str) -> bool | None:
    """Accord sur l'axe binaire du régime. ``None`` = le conseiller ne tranche pas.

    Préflop l'axe est jouer/passer : le conseiller n'a que « ouvrir » et
    « coucher » en magasin, il ne sait pas dire « suivre ». Compter un limp
    de Pluribus comme un désaccord avec « OUVRIR » mélangerait deux questions
    (faut-il jouer cette main ? faut-il la jouer en relançant ?) dont la
    seconde est hors du modèle. Le tableau croisé imprimé sous chaque famille
    garde la distinction.
    """
    if conseil in ("MIXTE", "INDECIS", "REFUS") or conseil.startswith("?"):
        return None
    if regime_face == "preflop":
        return (conseil != "PASSER") == (joue != "PASSER")
    if regime_face == "face_mise":
        return (conseil != "PASSER") == (joue != "PASSER")
    # sans mise : miser ou checker
    return (conseil == "AGRESSER") == (joue == "AGRESSER")


# ═══════════════════════════════════════════════════════════════════════════
# REJEU
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Resultat:
    mains: int = 0
    mains_refusees: int = 0
    decisions: int = 0
    spots: int = 0
    joueurs_mains: int = 0
    couches_verifies: int = 0
    familles: dict[str, Famille] = field(default_factory=dict)
    regimes: dict[str, Famille] = field(default_factory=dict)
    stats: dict[str, StatsJoueur] = field(default_factory=dict)
    noms: dict[str, str] = field(default_factory=dict)
    # divergences river heads-up : (coût réalisé en bb, verdict, action jouée)
    river_exact: list[tuple[float, str, str]] = field(default_factory=list)
    river_exact_accords: list[float] = field(default_factory=list)
    ouvertures: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0]))
    moteurs: Counter = field(default_factory=Counter)
    secondes: float = 0.0
    refus_exemples: Counter = field(default_factory=Counter)
    #: (catégorie du non-verdict, moteur consulté) → nombre de spots. C'est la
    #: réponse chiffrée à « pourquoi le conseiller ne conclut-il pas ? ».
    motifs_sans_verdict: Counter = field(default_factory=Counter)
    #: profondeurs effectives préflop réellement rencontrées, en bb : le
    #: rapport ne DÉDUIT pas que le corpus est à 100 bb, il le mesure.
    profondeurs_preflop: list[float] = field(default_factory=list)


def _famille(table: dict[str, Famille], cle: str) -> Famille:
    if cle not in table:
        table[cle] = Famille()
    return table[cle]


def cout_realise_river(d: Decision, main: ParsedHand) -> float | None:
    """Valeur EXACTE du suivi, en bb, sur une river heads-up.

    Le corpus donne les cartes de tous les joueurs : à la river, board
    complet et un seul adversaire encore en jeu, l'abattage est calculable
    sans supposer quoi que ce soit. Par rapport au fold (qui vaut 0), suivre
    rapporte ``+pot_gagnable`` si la main gagne, ``−to_call`` si elle perd,
    et la moitié de l'écart en cas de partage.

    Rend ``None`` dès qu'une hypothèse serait nécessaire : pas la river, plus
    d'un adversaire, aucune mise à payer, ou cartes adverses absentes.

    Ce chiffre est **réalisé**, pas espéré : une seule distribution par spot.
    Sa moyenne sur N divergences a donc une variance élevée, et le banc
    l'affiche avec son écart-type.
    """
    if d.street is not Street.RIVER or d.actifs != 2 or d.to_call <= 0:
        return None
    if len(d.board) != 5 or len(d.cards) != 2:
        return None
    couches = {a.player for a in main.actions if a.action is ActionType.FOLD}
    adverses = [s for s in main.seats
                if s.player != d.player and s.player not in couches]
    if len(adverses) != 1 or len(adverses[0].cards) != 2:
        return None
    board = parse_cards(" ".join(d.board))
    moi = _force7(parse_cards(" ".join(d.cards)) + board)
    lui = _force7(parse_cards(" ".join(adverses[0].cards)) + board)
    bb = main.big_blind or 1.0
    if moi > lui:
        return d.pot_gagnable / bb
    if moi < lui:
        return -d.to_call / bb
    return (d.pot_gagnable - d.to_call) / 2.0 / bb


def rejouer(corpus: Path, joueur: str | None, mains_max: int, villain: str,
            budget_river: float, bavard: bool) -> Resultat:
    """Rejoue le corpus et confronte le conseiller aux décisions du joueur.

    ``joueur = None`` ne rejoue aucune décision : seuls les repères de
    population sont calculés, et le banc devient instantané.

    La main est lue **sans héros déclaré**. Le désigner par son nom serait
    tentant, mais ferait REFUSER toutes les mains où il n'est pas assis :
    Pluribus joue les 10 000 mains du corpus, mais MrBlue n'en joue que
    9 121 et MrBrown 1 378. Le rapport aurait alors compté 879 mains
    « refusées » qui sont simplement des mains sans lui.
    """
    res = Resultat()
    cle_joueur = player_key(joueur) if joueur else None
    depart = time.perf_counter()

    for i, fichier in enumerate(iter_phh_files(corpus)):
        if mains_max and i >= mains_max:
            break
        try:
            main = parse_phh_file(fichier)
        except PhhError as exc:
            res.mains_refusees += 1
            res.refus_exemples[str(exc)[:120]] += 1
            continue
        except (OSError, UnicodeDecodeError) as exc:
            res.mains_refusees += 1
            res.refus_exemples[f"{type(exc).__name__}: {exc}"[:120]] += 1
            continue
        res.mains += 1

        # ── repères de population : tous les joueurs de la table ──────────
        noms = phh_players(fichier.read_text(encoding="utf-8"))
        positions = positions_par_siege(main)
        for nom, siege in zip(noms, sorted(s.seat for s in main.seats)):
            cle = player_key(nom)
            res.noms[cle] = nom
            res.stats.setdefault(cle, StatsJoueur()).ajouter(
                positions.get(siege, "?"),
                main.voluntarily_put_in_pot(cle),
                main.preflop_raise(cle))
            res.joueurs_mains += 1
        couches = {a.player for a in main.actions if a.action is ActionType.FOLD}
        res.couches_verifies += len(couches)

        # ── rejeu des décisions ──────────────────────────────────────────
        if cle_joueur is None:
            continue
        limpeurs = 0
        rue = Street.PREFLOP
        for d in iter_decisions(main):
            res.decisions += 1
            if d.street is not rue:
                rue = d.street
                limpeurs = 0
            limp_avant = limpeurs
            if d.action is ActionType.CALL and d.relances_avant == 0:
                limpeurs += 1

            if d.player != cle_joueur:
                continue
            res.spots += 1

            prof = profondeur_bb(d, main)
            if d.street is Street.PREFLOP and math.isfinite(prof):
                res.profondeurs_preflop.append(prof)
            regime, nom_famille = famille_du_spot(d, limp_avant, prof)
            fam = _famille(res.familles, nom_famille)
            reg = _famille(res.regimes, regime)
            fam.n += 1
            reg.n += 1

            try:
                conseil = advise(construire_spot(d, main, villain, budget_river))
            except (ValueError, ZeroDivisionError) as exc:
                fam.erreurs += 1
                reg.erreurs += 1
                res.refus_exemples[f"advise: {exc}"[:120]] += 1
                continue

            res.moteurs[_moteur(conseil.regime)] += 1
            vu = intention_conseiller(conseil)
            fait = intention_joueur(d)
            fam.croise[(vu, fait)] += 1
            reg.croise[(vu, fait)] += 1

            # fréquence d'ouverture, chart contre observation
            if (d.street is Street.PREFLOP and d.relances_avant == 0
                    and limp_avant == 0 and d.position in GTO_PRESETS):
                c = res.ouvertures[d.position]
                c[0] += 1
                c[1] += int(fait == "AGRESSER")

            axe = ("preflop" if d.street is Street.PREFLOP else
                   "face_mise" if d.to_call > 0 else "sans_mise")
            verdict = _accord(axe, vu, fait)
            if verdict is None:
                cible = "refus" if vu == "REFUS" else "indecis"
                setattr(fam, cible, getattr(fam, cible) + 1)
                setattr(reg, cible, getattr(reg, cible) + 1)
                res.motifs_sans_verdict[(vu, _moteur(conseil.regime))] += 1
            else:
                caracteres = axes_du_spot(d, prof)
                fam.compter_axes(caracteres, verdict)
                reg.compter_axes(caracteres, verdict)
                if verdict:
                    fam.accord += 1
                    reg.accord += 1
                else:
                    fam.desaccord += 1
                    reg.desaccord += 1
                    if conseil.ev_bb is not None and d.to_call > 0:
                        fam.couts_modele.append(abs(conseil.ev_bb))

            exact = cout_realise_river(d, main)
            if exact is not None:
                if verdict is False:
                    # signe : ce que le CONSEIL aurait rapporté de plus
                    gain = exact if vu != "PASSER" else -exact
                    res.river_exact.append((gain, vu, fait))
                elif verdict is True:
                    res.river_exact_accords.append(
                        exact if vu != "PASSER" else -exact)

        if bavard and res.mains % 1000 == 0:
            ecoule = time.perf_counter() - depart
            print(f"  … {res.mains} mains, {res.spots} spots, {ecoule:.0f} s",
                  file=sys.stderr, flush=True)

    res.secondes = time.perf_counter() - depart
    return res


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═══════════════════════════════════════════════════════════════════════════

_TRAIT = "─" * 78


def _titre(txt: str) -> None:
    print()
    print(_TRAIT)
    print(f"  {txt}")
    print(_TRAIT)


def _ligne_famille(nom: str, f: Famille) -> None:
    if f.tranches == 0:
        print(f"  {nom:<46} {f.n:>5} spots — jamais tranché")
        return
    taux = f.taux_desaccord
    bas, haut = wilson(f.desaccord, f.tranches)
    marque = ("  ⇐ SYSTÉMATIQUE" if f.n >= N_MIN_FAMILLE and bas >= SEUIL_SYSTEMATIQUE
              else "  ⇐ marqué" if f.n >= N_MIN_FAMILLE and bas >= SEUIL_MARQUE
              else "")
    print(f"  {nom:<46} {f.n:>5} spots · accord "
          f"{(1 - taux) * 100:5.1f} % · désaccord {taux * 100:5.1f} % "
          f"[{bas * 100:4.1f} ; {haut * 100:4.1f}]{marque}")
    if f.indecis or f.refus or f.erreurs:
        print(f"  {'':46}       dont {f.indecis} indécis, {f.refus} sans "
              f"modèle, {f.erreurs} illisibles (hors taux)")
    plus, moins = f.sens()
    if plus + moins >= 30:
        bp, hp = wilson(plus, plus + moins)
        penche = ("  ⇐ NOUS JOUONS PLUS" if bp >= 0.75 else
                  "  ⇐ NOUS JOUONS MOINS" if hp <= 0.25 else "")
        print(f"  {'':46}       sens : {plus} fois nous mettons de l'argent "
              f"et pas lui, {moins} fois l'inverse "
              f"({plus / (plus + moins) * 100:.0f} % "
              f"[{bp * 100:.0f} ; {hp * 100:.0f}]){penche}")


def _croise(f: Famille, indent: str = "        ") -> None:
    total = sum(f.croise.values())
    if not total:
        return
    for (vu, fait), k in sorted(f.croise.items(), key=lambda kv: -kv[1]):
        print(f"{indent}conseil {vu:<9} · joué {fait:<9} : {k:>5} "
              f"({k / total * 100:4.1f} %)")


#: (motif dans ``Advice.regime``, composant du conseiller, ce que la
#: comparaison vaut pour lui). **L'ordre compte** : le premier motif reconnu
#: gagne, et les motifs se recouvrent — « équité exacte au flop vs cotes du
#: pot (ICM) » contient les trois. Sans priorité, un même spot serait compté
#: dans trois composants et le total dépasserait le nombre de spots.
COMPOSANTS: tuple[tuple[str, str, str], ...] = (
    ("push/fold ICM", "Nash push/fold ICM + bubble factor",
     "hors d'atteinte : cash game, aucune structure de gains"),
    ("(ICM)", "correction ICM des cotes du pot postflop",
     "hors d'atteinte : cash game, aucune structure de gains"),
    ("Nash push/fold", "Nash push/fold chipEV",
     "hors d'atteinte : ce corpus n'a aucun tapis court"),
    ("chart d'ouverture", "ranges d'ouverture (GTO_PRESETS)",
     "CONFRONTÉ, à un bot d'équilibre 6-max 100 bb — le régime nominal des "
     "charts"),
    ("pas de chart", "aveu d'absence de chart",
     "compté en section 4 : c'est un refus honnête, pas une erreur"),
    ("solveur CFR", "solveur CFR à la river",
     "sollicité seulement avec --budget-river > 0, désactivé par défaut"),
    ("vs cotes du pot", "équité exacte postflop vs cotes du pot",
     "CONFRONTÉ, mais sous une range adverse SUPPOSÉE : la range est testée "
     "autant que le calcul"),
    ("équité exacte", "équité exacte postflop sans mise à payer",
     "CONFRONTÉ, même réserve, et sur un seuil de 55 % qui est une "
     "convention d'affichage, pas un solve"),
)


def _classer_moteur(moteur: str) -> str:
    """Composant du conseiller derrière un nom de régime. Premier motif gagne.

    Rend ``« non classé »`` plutôt que de ranger un régime inconnu dans un
    composant plausible : un moteur neuf doit être VISIBLE dans le rapport,
    pas absorbé.

    >>> _classer_moteur("équité exacte au flop vs cotes du pot (ICM)")
    'correction ICM des cotes du pot postflop'
    >>> _classer_moteur("équité exacte au flop vs cotes du pot")
    'équité exacte postflop vs cotes du pot'
    >>> _classer_moteur("équité exacte au turn")
    'équité exacte postflop sans mise à payer'
    >>> _classer_moteur("moteur inventé demain")
    '« non classé »'
    """
    for motif, nom, _ in COMPOSANTS:
        if motif in moteur:
            return nom
    return "« non classé »"


def _composants_sollicites(moteurs: Counter) -> list[tuple[str, int, str]]:
    """(composant, spots mesurés, portée) — compté, jamais supposé.

    Un composant peut afficher 0 : c'est le résultat qui compte le plus dans
    ce rapport, parce qu'il dit précisément ce que la comparaison NE valide
    pas. Les lignes ICM sont celles qui intéressent l'utilisateur, qui joue
    des tournois.
    """
    par_composant: Counter = Counter()
    for moteur, k in moteurs.items():
        par_composant[_classer_moteur(moteur)] += k
    sortie = [(nom, par_composant.pop(nom, 0), portee)
              for _, nom, portee in COMPOSANTS]
    for reste, k in sorted(par_composant.items()):
        sortie.append((reste, k, "MOTEUR INCONNU DE CE BANC — à classer avant "
                                 "de citer le rapport"))
    return sortie


def _disjoints(k1: int, n1: int, k2: int, n2: int) -> bool:
    """Les deux intervalles de Wilson à 95 % sont-ils DISJOINTS ?

    Critère volontairement plus dur que « les taux diffèrent » : sur un corpus
    de 10 000 mains découpé en régimes puis en axes, comparer des taux bruts
    ferait apparaître des écarts partout. Deux intervalles disjoints est une
    condition suffisante pour rejeter l'égalité au seuil de 5 % (elle est même
    conservatrice : elle rate des écarts réels, ce qui est le bon sens de
    l'erreur ici).

    >>> _disjoints(90, 100, 10, 100)
    True
    >>> _disjoints(52, 100, 48, 100)
    False
    """
    b1, h1 = wilson(k1, n1)
    b2, h2 = wilson(k2, n2)
    return h1 < b2 or h2 < b1


def _axes(f: Famille, indent: str = "        ") -> None:
    """Caractérisation d'une famille : où le désaccord se concentre-t-il ?

    Pour chaque axe, chaque valeur est comparée au RESTE de l'axe (toutes les
    autres valeurs poolées), et non à sa voisine : comparer deux à deux
    multiplierait les tests, donc les faux positifs. Une valeur n'est marquée
    que si elle et son complément atteignent :data:`N_MIN_AXE` et que leurs
    intervalles de Wilson sont disjoints.
    """
    if not f.axes:
        return
    for nom in sorted(f.axes):
        valeurs = f.axes[nom]
        total_n = sum(a + d for a, d in valeurs.values())
        total_k = sum(d for _, d in valeurs.values())
        if total_n == 0 or len(valeurs) < 2:
            continue
        print(f"{indent}axe « {nom} » :")
        for valeur, (a, d) in sorted(valeurs.items(),
                                     key=lambda kv: -(kv[1][0] + kv[1][1])):
            n = a + d
            reste_n, reste_k = total_n - n, total_k - d
            bas, haut = wilson(d, n)
            marque = ""
            if (n >= N_MIN_AXE and reste_n >= N_MIN_AXE
                    and _disjoints(d, n, reste_k, reste_n)):
                marque = ("  ⇐ ÉCART (vs reste de l'axe : "
                          f"{reste_k / reste_n * 100:.1f} %)")
            print(f"{indent}  {valeur:<24} {n:>5} tranchés · désaccord "
                  f"{d / n * 100:5.1f} % [{bas * 100:4.1f} ; {haut * 100:4.1f}]"
                  f"{marque}")


def rapport(res: Resultat, args: argparse.Namespace) -> None:
    joueur = args.joueur
    seul_pop = bool(getattr(args, "population", False))
    _titre(f"BANC CORPUS PLURIBUS — le conseiller face à « {joueur} »"
           if not seul_pop else
           "BANC CORPUS PLURIBUS — repères de population seuls")
    if seul_pop:
        print("  --population : le conseiller n'a PAS été appelé. Les sections")
        print("  2 à 6 sont donc vides, ce n'est pas un résultat.")
    print(f"  corpus            : {args.corpus}")
    print(f"  mains lues        : {res.mains}"
          + (f"  (refusées : {res.mains_refusees})" if res.mains_refusees else ""))
    print(f"  décisions rejouées: {res.decisions}")
    print(f"  spots de {joueur:<9}: {res.spots}")
    print(f"  range adverse     : « {args.villain} » (hypothèse, cf. --villain)")
    print(f"  solveur river     : "
          + (f"{args.budget_river:.1f} s" if args.budget_river > 0
             else "désactivé (le régime d'équité prend le relais)"))
    print(f"  durée             : {res.secondes:.1f} s")

    _titre("1. AUDIT DU LECTEUR PHH — la comptabilité tient-elle ?")
    print("  Chaque main est refusée si un joueur encaisse un montant négatif,")
    print("  ou si un joueur COUCHÉ encaisse quoi que ce soit. C'est le")
    print("  contrôle qui tomberait si « cbr » était lu comme un incrément.")
    print()
    print(f"  mains acceptées               : {res.mains}")
    print(f"  mains refusées                : {res.mains_refusees}")
    print(f"  joueurs-mains vérifiés        : {res.joueurs_mains}")
    print(f"  dont couchés (encaisse == 0)  : {res.couches_verifies}")
    if res.refus_exemples:
        print("  motifs de refus :")
        for motif, k in res.refus_exemples.most_common(5):
            print(f"    {k:>5} × {motif}")
    print()
    _portee_de_l_invariant()

    _titre("2. TAUX D'ACCORD PAR RÉGIME")
    print("  Pluribus n'est pas la vérité. Ces taux mesurent une DIVERGENCE,")
    print("  pas une erreur. Les indécis (MIXTE, MARGINAL) et les spots sans")
    print("  modèle sont hors du dénominateur — les compter d'un côté ou de")
    print("  l'autre fabriquerait le résultat.")
    print()
    total = Famille()
    # Les régimes attendus sont imprimés MÊME VIDES : un régime absent du
    # rapport se lirait « pas mesuré », un régime à zéro spot se lit « ce
    # corpus n'en contient pas », et c'est une information.
    for nom in ORDRE_REGIMES:
        f = res.regimes.get(nom)
        if f is None:
            print(f"  {nom:<46}     0 spot — absent de ce corpus")
            continue
        _ligne_famille(nom, f)
        total.n += f.n
        total.accord += f.accord
        total.desaccord += f.desaccord
        total.indecis += f.indecis
        total.refus += f.refus
        total.erreurs += f.erreurs
        total.croise.update(f.croise)
        total.fusionner_axes(f)
    inattendus = [n for n in res.regimes if n not in ORDRE_REGIMES]
    for nom in sorted(inattendus):
        _ligne_famille(f"{nom} (hors nomenclature)", res.regimes[nom])
    print()
    _ligne_famille("TOUS RÉGIMES CONFONDUS", total)
    print("  Ce total est donné pour la comptabilité, PAS pour être cité : il")
    print("  moyenne des régimes qui n'ont ni le même moteur ni le même")
    print("  effectif. Les lignes au-dessus sont le résultat.")
    print()
    print("  QUEL MOTEUR A RÉPONDU (compté, pas supposé) :")
    for moteur, k in res.moteurs.most_common():
        print(f"    {k:>6}  {moteur}")
    court = sum(k for m, k in res.moteurs.items() if "push/fold" in m)
    n_court = res.regimes[REGIME_COURT].n if REGIME_COURT in res.regimes else 0
    print()
    print(f"  Régime « préflop tapis court » : {n_court} spot(s) par la")
    print(f"  profondeur, {court} spot(s) par le moteur (Nash push/fold).")
    if res.profondeurs_preflop:
        p = sorted(res.profondeurs_preflop)
        print(f"  Profondeur effective préflop MESURÉE sur {len(p)} spots : "
              f"min {p[0]:.1f} bb ·")
        print(f"  médiane {p[len(p) // 2]:.1f} bb · max {p[-1]:.1f} bb. Le "
              "corpus est donc")
        print("  bien à tapis constants — c'est mesuré ici, pas supposé.")
    print("  Le conseiller ne passe en Nash push/fold qu'à deux joueurs sous")
    print("  25 bb effectifs : les deux comptes ci-dessus peuvent différer, et")
    print("  l'écart dirait qu'un spot court est traité par la chart. Le")
    print("  solveur qui sert en Twister n'est pas sollicité ici : ce banc ne")
    print("  dit rien de lui, ni en bien ni en mal.")

    _titre("3. FAMILLES DE SPOTS — le livrable principal")
    print("  Fait STRUCTUREL, vrai indépendamment des chiffres ci-dessous :")
    print("  le conseiller n'a AUCUN régime pour « préflop face à une relance ».")
    print("  advise() y retombe sur _advise_preflop_deep, dont l'hypothèse")
    print("  déclarée est « suppose que personne n'a ouvert devant ». Il rend")
    print("  donc un verdict d'OUVERTURE là où la question est de défendre")
    print("  face à un open — sans jamais dire qu'il est hors de son domaine.")
    print("  C'est la famille la plus nombreuse d'un corpus 6-max.")
    print()
    print(f"  « SYSTÉMATIQUE » : ≥ {N_MIN_FAMILLE} spots ET borne basse de")
    print(f"  l'intervalle de Wilson à 95 % ≥ {SEUIL_SYSTEMATIQUE * 100:.0f} % de désaccord.")
    print(f"  « marqué » : même effectif, borne basse ≥ {SEUIL_MARQUE * 100:.0f} %.")
    print()
    print("  La ligne « sens » est le chiffre qui compte le plus. Un TAUX de")
    print("  désaccord élevé peut n'être que du bruit de frontière — deux")
    print("  stratégies correctes divergent sur les mains limites, mais dans")
    print("  les DEUX sens à parts égales. Un SENS déséquilibré, lui, ne peut")
    print("  pas s'expliquer par le hasard des mains limites : il dit que")
    print("  notre modèle penche. « NOUS JOUONS PLUS » est affiché quand au")
    print("  moins 75 % des désaccords (borne basse à 95 %) vont dans ce sens.")
    print()
    ordre = sorted(res.familles.items(),
                   key=lambda kv: (-kv[1].taux_desaccord, -kv[1].n))
    for nom, f in ordre:
        _ligne_famille(nom, f)
    print()
    print("  RÉCAPITULATIF — familles où le désaccord a un SENS :")
    penchees = []
    for nom, f in ordre:
        plus, moins = f.sens()
        if plus + moins < 30:
            continue
        bp, hp = wilson(plus, plus + moins)
        if bp >= 0.75:
            penchees.append((nom, "nous jouons PLUS", plus, moins, bp, hp))
        elif hp <= 0.25:
            penchees.append((nom, "nous jouons MOINS", plus, moins, bp, hp))
    if penchees:
        for nom, sens, plus, moins, bp, hp in penchees:
            part = plus / (plus + moins)
            print(f"    {nom:<46} {sens} "
                  f"({plus} contre {moins}, {part * 100:.0f} % "
                  f"[{bp * 100:.0f} ; {hp * 100:.0f}])")
    else:
        print("    aucune : les désaccords se répartissent dans les deux sens.")
    print()
    print("  Détail des familles où le désaccord est systématique ou marqué :")
    vus = False
    for nom, f in ordre:
        bas, _ = wilson(f.desaccord, f.tranches)
        if f.n >= N_MIN_FAMILLE and bas >= SEUIL_MARQUE:
            vus = True
            print(f"\n    {nom} ({f.n} spots)")
            _croise(f)
            print("      caractérisation :")
            _axes(f, indent="        ")
    if not vus:
        print("    aucune.")

    _titre("3 bis. CARACTÉRISATION PAR RÉGIME "
           "(position, profondeur, board, main)")
    print("  Le découpage en familles épuise vite l'effectif ; les axes sont")
    print("  donc aussi comptés au niveau du RÉGIME, où il reste assez de")
    print("  spots pour que la comparaison ait une chance de trancher.")
    print()
    print("  Une valeur n'est marquée « ÉCART » que si elle ET le reste de")
    print(f"  son axe comptent ≥ {N_MIN_AXE} spots tranchés, et que leurs")
    print("  intervalles de Wilson à 95 % sont DISJOINTS. Sur 10 000 mains,")
    print("  comparer des taux bruts ferait apparaître des motifs partout :")
    print("  ce critère en rate volontairement pour n'en inventer aucun.")
    for nom in ORDRE_REGIMES:
        f = res.regimes.get(nom)
        if f is None or f.tranches == 0:
            continue
        print(f"\n    {nom} ({f.tranches} spots tranchés sur {f.n})")
        _axes(f, indent="      ")

    _titre("4. QUAND LE CONSEILLER REFUSE DE CONCLURE")
    print(f"  sans aucun modèle (« — »)        : {total.refus:>6}"
          f"  ({total.refus / max(total.n, 1) * 100:.1f} % des spots)")
    print(f"  indécis (MIXTE / MARGINAL)       : {total.indecis:>6}"
          f"  ({total.indecis / max(total.n, 1) * 100:.1f} %)")
    print(f"  spots illisibles par le conseiller: {total.erreurs:>6}")
    print()
    print("  POURQUOI, compté et non raconté — (catégorie, moteur consulté) :")
    if res.motifs_sans_verdict:
        for (categorie, moteur), k in res.motifs_sans_verdict.most_common():
            print(f"    {k:>6}  {categorie:<8} · {moteur}")
    else:
        print("    aucun : le conseiller a tranché tous les spots rejoués.")
    print()
    print("  PAR RÉGIME :")
    for nom in ORDRE_REGIMES:
        f = res.regimes.get(nom)
        if f is None or f.n == 0:
            continue
        sans = f.refus + f.indecis
        print(f"    {nom:<40} {sans:>5} / {f.n:>5} spots "
              f"({sans / f.n * 100:5.1f} %) sans verdict")
    print()
    print("  Un « — » signale une position sans chart (la big blind). Un")
    print("  « MIXTE » est une réponse en fréquence : elle a du sens sur mille")
    print("  mains, pas sur celle-ci. Un « MARGINAL » dit que l'équité est à")
    print("  moins de 4 points de la cote — l'erreur y coûte presque rien.")
    print("  Ces trois-là sont des refus HONNÊTES. Ce qui manque au conseiller")
    print("  est ailleurs : il ne refuse PAS les spots hors de son domaine")
    print("  (défense face à un open, pot multiway), il y répond quand même.")
    print("  Le compte ci-dessus est donc un plancher, pas le total de ce que")
    print("  le conseiller ne sait pas faire.")

    _titre("5. COÛT EN BB DES DÉSACCORDS")
    couts = [c for f in res.familles.values() for c in f.couts_modele]
    if couts:
        print(f"  a) SELON NOTRE PROPRE MODÈLE (auto-référentiel) — {len(couts)} "
              "désaccords")
        print("     face à une mise, où l'EV du suivi est calculable :")
        print(f"     moyenne {sum(couts) / len(couts):+.2f} bb · médiane "
              f"{sorted(couts)[len(couts) // 2]:+.2f} bb · max "
              f"{max(couts):+.2f} bb")
        print("     Ce chiffre suppose notre range adverse : il mesure ce que")
        print("     NOUS croyons que la divergence coûte, pas ce qu'elle coûte.")
    else:
        print("  a) aucun désaccord chiffrable par le modèle.")
    print()
    if res.river_exact:
        gains = [g for g, _, _ in res.river_exact]
        m, sd, err = _moyenne_ic(gains)
        print(f"  b) RÉSULTAT RÉALISÉ, indépendant de nous — {len(gains)} "
              "divergences")
        print("     à la river en tête-à-tête, board complet et cartes adverses")
        print("     connues (le corpus les donne) : l'abattage est exact.")
        print(f"     Suivre notre conseil aurait rapporté {m:+.2f} bb par spot")
        print(f"     (écart-type {sd:.2f} bb, erreur-type {err:.2f} bb).")
        low, high = m - 1.96 * err, m + 1.96 * err
        print(f"     Intervalle à 95 % : [{low:+.2f} ; {high:+.2f}] bb.")
        verdict = ("notre conseil aurait GAGNÉ" if low > 0 else
                   "notre conseil aurait PERDU" if high < 0 else
                   "l'échantillon ne tranche pas")
        print(f"     → {verdict}.")
        pour = Counter((vu, fait) for _, vu, fait in res.river_exact)
        print("       détail par cellule, avec son PROPRE intervalle : le")
        print("       total ci-dessus mélange des divergences de sens opposé,")
        print("       et une cellule peut trancher là où le total ne tranche pas.")
        for (vu, fait), _ in pour.most_common():
            sous = [g for g, v, f2 in res.river_exact if (v, f2) == (vu, fait)]
            mc, sc, ec = _moyenne_ic(sous)
            marque = ("  ⇐ tranche" if len(sous) >= 30 and abs(mc) > 1.96 * ec
                      else "")
            print(f"       conseil {vu:<9} · joué {fait:<9} : {len(sous):>4} "
                  f"spots, {mc:+6.2f} bb  [{mc - 1.96 * ec:+6.2f} ; "
                  f"{mc + 1.96 * ec:+6.2f}]{marque}")
        print("     Réserve : la contrepartie chiffrée est un suivi sec. Quand")
        print("     la ligne « joué AGRESSER » ci-dessus est nombreuse, ce que")
        print("     Pluribus a réellement gagné inclut de l'équité de fold que")
        print("     ce calcul ignore — le chiffre lui est donc défavorable.")
    else:
        print("  b) aucune divergence river en tête-à-tête dans cet échantillon.")
    if res.river_exact_accords:
        g = res.river_exact_accords
        print(f"     Repère — sur les {len(g)} river où les deux s'accordent, la")
        print(f"     même mesure vaut {sum(g) / len(g):+.2f} bb. Ce n'est pas un")
        print("     bruit de fond mais la valeur des décisions non contestées :")
        print("     c'est le niveau auquel comparer le chiffre ci-dessus.")

    _titre("6. LA CHART D'OUVERTURE CONTRE L'OBSERVATION")
    print("  Fréquence d'ouverture prescrite par GTO_PRESETS (moyenne des poids")
    print("  sur les 1326 combos) contre celle observée, sur les spots où")
    print(f"  {joueur} ouvre l'action (personne n'a relancé ni limpé).")
    print()
    print(f"  {'position':<10}{'chart':>10}{'observé':>10}{'écart':>10}"
          f"{'spots':>8}")
    for pos in ("UTG", "MP", "CO", "BTN", "SB"):
        if pos not in res.ouvertures:
            continue
        n, k = res.ouvertures[pos]
        if not n:
            continue
        chart = float(np.mean(parse_range(GTO_PRESETS[pos]).weights))
        obs = k / n
        print(f"  {pos:<10}{chart * 100:9.1f} %{obs * 100:9.1f} %"
              f"{(obs - chart) * 100:+9.1f} pt{n:8d}")
    print()
    print("  Un écart positif dit que le joueur ouvre PLUS large que notre")
    print("  chart. Sur un bot d'équilibre à 100 bb, c'est un repère utile")
    print("  pour recalibrer les charts — pas une preuve qu'elles sont fausses :")
    print("  les charts visent le 6-max sans ante, ce corpus aussi, mais rien")
    print("  ne garantit que Pluribus optimise le même critère.")

    _titre("7. REPÈRES DE POPULATION MESURÉS (VPIP / PFR / écart)")
    print("  Ce que le logiciel compare aujourd'hui à des fourchettes de manuel,")
    print("  mesuré ici. ATTENTION à qui est mesuré : 12 professionnels invités")
    print("  et un bot surhumain, en cash game 6-max 100 bb. Ce n'est PAS la")
    print("  population d'un tournoi en ligne — les fourchettes de manuel visent")
    print("  un pool récréatif, ces chiffres-ci un pool d'élite.")
    print()
    print(f"  {'joueur':<12}{'mains':>7}{'VPIP':>9}{'PFR':>9}{'écart':>9}")
    lignes = sorted(res.stats.items(), key=lambda kv: -kv[1].mains)
    ecarts, vpips, pfrs = [], [], []
    for cle, s in lignes:
        if not s.mains:
            continue
        v, p = s.vpip / s.mains, s.pfr / s.mains
        print(f"  {res.noms.get(cle, cle[:10]):<12}{s.mains:>7}"
              f"{v * 100:8.1f} %{p * 100:8.1f} %{(v - p) * 100:8.1f} pt")
        vpips.append(v)
        pfrs.append(p)
        ecarts.append(v - p)
    if ecarts:
        def quartiles(xs: list[float]) -> str:
            q = sorted(xs)
            n = len(q)
            return (f"min {q[0] * 100:.1f} · Q1 {q[n // 4] * 100:.1f} · "
                    f"médiane {q[n // 2] * 100:.1f} · Q3 {q[3 * n // 4] * 100:.1f} · "
                    f"max {q[-1] * 100:.1f}")
        print()
        print(f"  distribution sur {len(ecarts)} joueurs (en %) :")
        print(f"    VPIP  : {quartiles(vpips)}")
        print(f"    PFR   : {quartiles(pfrs)}")
        print(f"    écart : {quartiles(ecarts)}")
        tot_m = sum(s.mains for _, s in lignes)
        tot_v = sum(s.vpip for _, s in lignes)
        tot_p = sum(s.pfr for _, s in lignes)
        print(f"    poolé : VPIP {tot_v / tot_m * 100:.1f} % · "
              f"PFR {tot_p / tot_m * 100:.1f} % · "
              f"écart {(tot_v - tot_p) / tot_m * 100:.1f} pt "
              f"({tot_m} joueurs-mains)")

    _titre("8. VPIP / PFR PAR POSITION (poolé sur tous les joueurs)")
    par_pos: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for _, s in res.stats.items():
        for pos, c in s.par_position.items():
            d = par_pos[pos]
            d[0] += c[0]
            d[1] += c[1]
            d[2] += c[2]
    print(f"  {'position':<10}{'mains':>8}{'VPIP':>9}{'PFR':>9}{'écart':>9}")
    for pos in ("UTG", "MP", "CO", "BTN", "SB", "BB"):
        if pos not in par_pos:
            continue
        n, v, p = par_pos[pos]
        if not n:
            continue
        print(f"  {pos:<10}{n:>8}{v / n * 100:8.1f} %{p / n * 100:8.1f} %"
              f"{(v - p) / n * 100:8.1f} pt")

    _titre("9. CE QUE CETTE COMPARAISON VALIDE — ET CE QU'ELLE NE TOUCHE PAS")
    print("  Composant par composant du conseiller, avec le nombre de spots")
    print("  où il a RÉELLEMENT répondu (compté dans Advice.regime, pas")
    print("  supposé). Un zéro n'est pas un manque du banc : c'est la mesure")
    print("  de ce que ce corpus ne peut pas dire.")
    print()
    for nom, k, portee in _composants_sollicites(res.moteurs):
        etat = f"{k:>6} spots" if k else "     — jamais sollicité"
        print(f"    {nom:<42}{etat}")
        print(f"      {portee}")
    print()
    print("  Il manque un composant à cette liste parce qu'il n'existe pas :")
    print("  la DÉFENSE face à une ouverture. Le conseiller y répond avec la")
    print("  chart d'ouverture de la position, qui suppose un pot non ouvert.")
    print("  Les spots correspondants sont donc comptés dans la ligne")
    print("  « ranges d'ouverture » ci-dessus alors qu'ils posent une autre")
    print("  question — c'est la famille la plus nombreuse d'un corpus 6-max,")
    print("  et le désaccord qu'on y mesure ne dit pas que la chart est")
    print("  fausse : il dit qu'on l'applique là où elle n'a rien à faire.")

    _titre("9 bis. LIMITES — à lire avant de citer un chiffre de ce rapport")
    for ligne in (
        "Cash game 6-max sans ICM. AUCUN chiffre ci-dessus ne valide le",
        "conseil en tournoi : ni l'ICM, ni le bubble factor, ni le push/fold.",
        "",
        "Tapis constants à 100 bb : zéro spot de tapis court, donc zéro",
        "mesure sur le solveur qui sert en Twister.",
        "",
        "Pluribus est une référence, pas un oracle. Un désaccord ponctuel ne",
        "prouve rien ; seules les familles marquées ci-dessus le sont assez",
        "pour valoir un examen.",
        "",
        f"Le postflop suppose UN adversaire et la range « {args.villain} ».",
        "Les familles multiway sont hors du domaine du modèle et comptées",
        "à part. Relancer avec --villain large / serree pour la sensibilité.",
        "",
        "Le coût « selon notre modèle » n'est pas une mesure indépendante.",
        "Seul le coût réalisé à la river en tête-à-tête l'est.",
    ):
        print(f"  {ligne}")
    print(_TRAIT)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════


def audit_lecteur(corpus: Path, mains_max: int) -> int:
    """Passe rapide : le lecteur avale-t-il tout le corpus sans se contredire ?

    Deux sortes de refus, qui n'ont pas le même sens et ne sortent donc pas
    par le même code :

    * **hors domaine** — une variante que le lecteur ne prétend pas lire
      (Omaha, stud, lowball). C'est le comportement voulu, pas un défaut : le
      banc rend 0. Mesuré sur la table finale des WSOP 2023, qui est un
      championnat *mixte* : 18 mains de hold'em lues, 65 refusées, réparties
      sur sept variantes (FO/8, F7S, FR, PO, N2L1D, F7S/8, F2L3D).
    * **comptabilité** — le déroulé lu ne recolle pas aux tapis de fin. Là,
      c'est le lecteur ou le fichier qui est faux : le banc rend 1.
    """
    ok = folds = joueurs = 0
    hors_domaine = comptable = autre = 0
    motifs: Counter = Counter()
    t0 = time.perf_counter()
    for i, f in enumerate(iter_phh_files(corpus)):
        if mains_max and i >= mains_max:
            break
        try:
            h = parse_phh_file(f)
        except PhhError as exc:
            msg = str(exc)
            motifs[msg[:140]] += 1
            if msg.startswith("variante"):
                hors_domaine += 1
            elif "comptabilité" in msg:
                comptable += 1
            else:
                autre += 1
            continue
        ok += 1
        joueurs += len(h.seats)
        folds += len({a.player for a in h.actions
                      if a.action is ActionType.FOLD})
    dt = time.perf_counter() - t0
    _titre("AUDIT DU LECTEUR PHH")
    print(f"  corpus                        : {corpus}")
    print(f"  mains acceptées               : {ok}")
    print(f"  refus « variante hors domaine »: {hors_domaine}   (attendu)")
    print(f"  refus « comptabilité »         : {comptable}   (défaut)")
    print(f"  autres refus                   : {autre}")
    print(f"  joueurs-mains vérifiés        : {joueurs}")
    print(f"  dont couchés (encaisse == 0)  : {folds}")
    print(f"  durée                         : {dt:.1f} s")
    if motifs:
        print("  motifs :")
        for m, k in motifs.most_common(10):
            print(f"    {k:>5} × {m}")
    print()
    _portee_de_l_invariant()
    print(_TRAIT)
    return 1 if (comptable or autre) else 0


def _portee_de_l_invariant() -> None:
    """Ce que l'invariant comptable attrape — et ce qu'il n'attrape pas.

    Mesuré en cassant volontairement le lecteur et en relançant
    ``--verifier`` sur les 10 000 mains. Un contrôle dont on ne connaît pas
    la portée ne rassure sur rien.
    """
    print("  PORTÉE MESURÉE DE CE CONTRÔLE (mutations jouées sur le lecteur,")
    print("  --verifier relancé sur les 10 000 mains de Pluribus) :")
    for texte in (
        "lire « cbr » comme un incrément     →  994 mains refusées  ATTRAPÉ",
        "supprimer le rendu de la surmise    →    0 main refusée    MANQUÉ",
        "compter l'ante dans la mise de rue  →    0 main refusée    MANQUÉ",
        "ne pas plafonner « cc » au tapis    →    0 main refusée    MANQUÉ",
    ):
        print(f"    {texte}")
    print("  Les trois manqués le sont pour des raisons connues : un joueur")
    print("  couché n'a jamais de surmise à récupérer (son encaissement reste")
    print("  nul dans les deux lectures), Pluribus joue sans ante, et personne")
    print("  n'y est jamais assez court pour qu'un « cc » soit plafonné.")
    print("  Ces trois-là sont épinglés par tests/test_phh.py, dont chaque")
    print("  test a été vérifié TOMBANT sous la mutation correspondante.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAUT)
    p.add_argument("--joueur", default="Pluribus",
                   help="joueur dont on rejoue les décisions")
    p.add_argument("--mains", type=int, default=0,
                   help="ne lire que les N premières mains (ordre stable)")
    p.add_argument("--villain", default="moyenne",
                   help="range adverse supposée : large, moyenne, serree, "
                        "ou une range explicite")
    p.add_argument("--budget-river", type=float, default=0.0,
                   help="budget du solveur CFR à la river, en secondes "
                        "(0 = désactivé ; ~2 s par spot sinon)")
    p.add_argument("--verifier", action="store_true",
                   help="seulement l'audit comptable du lecteur")
    p.add_argument("--population", action="store_true",
                   help="seulement les repères VPIP/PFR (rapide)")
    p.add_argument("--silencieux", action="store_true",
                   help="pas de progression sur stderr")
    args = p.parse_args(argv)

    if not args.corpus.exists():
        print(f"corpus introuvable : {args.corpus}", file=sys.stderr)
        return 2

    if args.verifier:
        return audit_lecteur(args.corpus, args.mains)

    # Le rejeu du conseiller est ce qui coûte (≈ 350 ms par spot au flop, où
    # l'équité énumère les 990 runouts) : --population le saute entièrement.
    res = rejouer(args.corpus, None if args.population else args.joueur,
                  args.mains, args.villain, args.budget_river,
                  not args.silencieux)
    rapport(res, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
