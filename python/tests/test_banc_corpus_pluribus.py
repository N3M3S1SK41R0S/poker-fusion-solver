"""Tests du banc de rejeu du corpus Pluribus.

Un banc qui produit des chiffres pour un rapport est du code de production :
ses règles de classement et ses formules doivent être épinglées comme le
reste. Sont testés ici les trois endroits où une erreur silencieuse
produirait un rapport faux mais crédible :

1. la borne de Wilson, qui décide seule quelle famille est déclarée
   « systématique » ;
2. la traduction verdict ↔ action, où une chaîne mal reconnue ferait compter
   un accord pour un désaccord ;
3. le coût RÉALISÉ à la river, seul chiffre du rapport qui ne dépende pas de
   nos propres hypothèses — et donc le seul qu'on puisse citer sans réserve.
"""

from __future__ import annotations

import math
import sys
import unittest
from collections import Counter
from pathlib import Path

from pfs.analysis.spot_advisor import Advice
from pfs.data.hand_history import ActionType, Street
from pfs.data.phh import iter_decisions, parse_phh

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import banc_corpus_pluribus as banc  # noqa: E402


class TestWilson(unittest.TestCase):
    """La borne qui décide de « systématique » doit être exacte."""

    def test_valeur_de_reference(self) -> None:
        """50 succès sur 100, refait à la main avec z = 1,959963985.

        z² = 3,8414588 ; dénominateur 1 + z²/n = 1,038414588 ; centre
        (0,5 + z²/200) / 1,038414588 = 0,5 ; demi-largeur
        z·√(0,25/100 + z²/40000) / 1,038414588
        = 1,959963985 × 0,05095131 / 1,038414588 = 0,0961685.
        D'où [0,4038315 ; 0,5961685], symétrique parce que p = ½.
        """
        bas, haut = banc.wilson(50, 100)
        self.assertAlmostEqual(bas, 0.4038315, places=6)
        self.assertAlmostEqual(haut, 0.5961685, places=6)
        self.assertAlmostEqual((bas + haut) / 2, 0.5, places=12)

    def test_reste_dans_zero_un_aux_extremes(self) -> None:
        """C'est la raison du choix de Wilson : l'intervalle normal sort de
        [0, 1] à 0 et à 100 % de succès, et le banc travaille justement là."""
        self.assertEqual(banc.wilson(0, 40)[0], 0.0)
        self.assertLess(banc.wilson(0, 40)[1], 0.10)
        self.assertEqual(banc.wilson(40, 40)[1], 1.0)
        self.assertGreater(banc.wilson(40, 40)[0], 0.90)

    def test_sans_observation_l_intervalle_est_total(self) -> None:
        self.assertEqual(banc.wilson(0, 0), (0.0, 1.0))

    def test_la_borne_basse_croit_avec_l_effectif(self) -> None:
        """À taux constant, plus de spots = plus de certitude.

        C'est cette propriété qui empêche une famille de 12 spots à 60 % de
        désaccord d'être déclarée systématique.
        """
        petit = banc.wilson(12, 20)[0]
        grand = banc.wilson(600, 1000)[0]
        self.assertLess(petit, 0.5)
        self.assertGreater(grand, 0.5)


class TestMoyenneIC(unittest.TestCase):
    """L'erreur-type décide si le coût réalisé « tranche » ou non."""

    def test_valeurs_a_la_main(self) -> None:
        """[1, 2, 3, 4] : moyenne 2,5 ; variance échantillon 5/3 ;
        écart-type √(5/3) = 1,290994 ; erreur-type /√4 = 0,645497."""
        m, sd, err = banc._moyenne_ic([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(m, 2.5)
        self.assertAlmostEqual(sd, 1.2909944, places=6)
        self.assertAlmostEqual(err, 0.6454972, places=6)

    def test_denominateur_n_moins_un(self) -> None:
        """Avec n−1 l'écart-type de [0, 2] vaut √2 ; avec n il vaudrait 1.

        Le choix n'est pas cosmétique : c'est lui qui fixe la largeur de
        l'intervalle affiché sous le coût réalisé.
        """
        _, sd, _ = banc._moyenne_ic([0.0, 2.0])
        self.assertAlmostEqual(sd, math.sqrt(2.0), places=12)

    def test_cas_degeneres(self) -> None:
        self.assertEqual(banc._moyenne_ic([]), (0.0, 0.0, 0.0))
        self.assertEqual(banc._moyenne_ic([7.0]), (7.0, 0.0, 0.0))


class TestMoteur(unittest.TestCase):
    """Le rapport COMPTE les moteurs sollicités plutôt que de les supposer."""

    def test_les_parametres_numeriques_sont_effaces(self) -> None:
        """Sans ça, « Nash push/fold 12.5 bb » et « … 13.0 bb » feraient deux
        lignes, et le rapport aurait des centaines de lignes illisibles."""
        self.assertEqual(banc._moteur("Nash push/fold 12.5 bb"),
                         "Nash push/fold N bb")
        self.assertEqual(banc._moteur("Nash push/fold 13 bb"),
                         "Nash push/fold N bb")
        self.assertEqual(banc._moteur("Nash push/fold ICM 8.0 bb (3 joueurs)"),
                         "Nash push/fold ICM N bb (N joueurs)")

    def test_les_noms_de_moteur_restent_lisibles(self) -> None:
        """Les régimes sans chiffre traversent intacts — c'est eux qu'on lit."""
        for regime in ("chart d'ouverture CO",
                       "équité exacte au flop vs cotes du pot",
                       "solveur CFR à la river",
                       "pas de chart pour cette position"):
            self.assertEqual(banc._moteur(regime), regime)

    def test_le_comptage_du_tapis_court_repose_sur_ce_nom(self) -> None:
        """Le rapport conclut « 0 spot de tapis court » en cherchant
        « push/fold » dans ces noms : la sous-chaîne doit survivre."""
        self.assertIn("push/fold", banc._moteur("Nash push/fold 12.5 bb"))
        self.assertIn("push/fold",
                      banc._moteur("Nash push/fold ICM 8.0 bb (3 joueurs)"))


class TestTraductionDesVerdicts(unittest.TestCase):
    """Une chaîne mal reconnue transformerait un accord en désaccord."""

    def _a(self, action: str) -> Advice:
        return Advice(action=action, confidence="indicatif", regime="test")

    def test_tous_les_verdicts_du_conseiller_sont_reconnus(self) -> None:
        """Aucun verdict ne doit retomber dans la branche « ? ».

        La liste vient des chaînes réellement construites par
        ``spot_advisor`` : si l'un de ses libellés change, ce test tombe
        plutôt que de laisser le banc compter des désaccords fantômes.
        """
        attendu = {
            "JAM (tapis)": "AGRESSER",
            "FOLD": "PASSER",
            "OUVRIR (relance)": "AGRESSER",
            "MIXTE — ouvrir 70 % du temps": "MIXTE",
            "—": "REFUS",
            "CALL (confortable)": "CONTINUER",
            "CALL (juste)": "CONTINUER",
            "MARGINAL — proche de l'indifférence": "INDECIS",
            "MISER (valeur)": "AGRESSER",
            "CHECK": "CHECK",
            "CHECK (ou bluff choisi)": "CHECK",
            "MISER 75 (75 % du pot)": "AGRESSER",
            "TAPIS (200, 200 % du pot)": "AGRESSER",
            "PAYER 50": "CONTINUER",
            "MIXTE — CHECK 60 % / MISER 33 (33 % du pot) 40 %": "MIXTE",
        }
        for action, intention in attendu.items():
            with self.subTest(action=action):
                self.assertEqual(banc.intention_conseiller(self._a(action)),
                                 intention)

    def test_un_libelle_inconnu_est_signale_pas_avale(self) -> None:
        """Il doit être VISIBLE, pas rangé dans une catégorie plausible."""
        self.assertTrue(
            banc.intention_conseiller(self._a("RELANCER À MORT")).startswith("?"))


class TestAxesDAccord(unittest.TestCase):
    """Chaque régime a l'axe binaire que son modèle sait trancher."""

    def test_preflop_ouvrir_ou_passer(self) -> None:
        self.assertTrue(banc._accord("preflop", "AGRESSER", "AGRESSER"))
        self.assertTrue(banc._accord("preflop", "PASSER", "PASSER"))
        self.assertFalse(banc._accord("preflop", "AGRESSER", "PASSER"))
        self.assertFalse(banc._accord("preflop", "PASSER", "AGRESSER"))

    def test_preflop_un_limp_compte_comme_jouer(self) -> None:
        """Le conseiller ne sait pas dire « suivre » : sur l'axe jouer/passer,
        « OUVRIR » et un limp s'accordent. Le tableau croisé du rapport garde
        la distinction, qui est une autre question."""
        self.assertTrue(banc._accord("preflop", "AGRESSER", "CONTINUER"))
        self.assertFalse(banc._accord("preflop", "PASSER", "CONTINUER"))

    def test_face_a_une_mise_relancer_c_est_continuer(self) -> None:
        """Le modèle équité/cotes répond « payer ou coucher ». Une relance de
        Pluribus n'est pas un désaccord avec « CALL » : les deux refusent de
        coucher."""
        self.assertTrue(banc._accord("face_mise", "CONTINUER", "AGRESSER"))
        self.assertTrue(banc._accord("face_mise", "CONTINUER", "CONTINUER"))
        self.assertFalse(banc._accord("face_mise", "CONTINUER", "PASSER"))
        self.assertTrue(banc._accord("face_mise", "PASSER", "PASSER"))

    def test_sans_mise_miser_ou_checker(self) -> None:
        self.assertTrue(banc._accord("sans_mise", "AGRESSER", "AGRESSER"))
        self.assertTrue(banc._accord("sans_mise", "CHECK", "CHECK"))
        self.assertFalse(banc._accord("sans_mise", "CHECK", "AGRESSER"))
        self.assertFalse(banc._accord("sans_mise", "AGRESSER", "CHECK"))

    def test_ce_que_le_conseiller_ne_tranche_pas_sort_du_taux(self) -> None:
        """MIXTE, MARGINAL et « — » rendent None : ni accord, ni désaccord.

        Les compter d'un côté fabriquerait le résultat — c'est exactement le
        genre de choix qui rend un rapport faux sans qu'aucun test ne tombe.
        """
        for verdict in ("MIXTE", "INDECIS", "REFUS", "?RELANCER"):
            for axe in ("preflop", "face_mise", "sans_mise"):
                self.assertIsNone(banc._accord(axe, verdict, "AGRESSER"))


class TestSensDuDesaccord(unittest.TestCase):
    """Le SENS du désaccord est le chiffre le plus fort du rapport.

    Un taux élevé peut n'être que du bruit de frontière ; un sens déséquilibré
    ne peut pas. Une erreur ici inverserait la conclusion du banc.
    """

    def _f(self, **cellules) -> "banc.Famille":
        f = banc.Famille()
        for cle, k in cellules.items():
            vu, fait = cle.split("_")
            f.croise[(vu, fait)] = k
        return f

    def test_face_a_une_mise_payer_quand_il_couche_c_est_jouer_plus(self) -> None:
        f = self._f(CONTINUER_PASSER=96, PASSER_CONTINUER=4,
                    CONTINUER_CONTINUER=107, PASSER_PASSER=48)
        self.assertEqual(f.sens(), (96, 4))

    def test_une_relance_adverse_n_est_pas_nous_qui_jouons_moins(self) -> None:
        """Conseil « payer », joué « relancer » : les deux mettent de
        l'argent. Ce n'est ni un désaccord sur l'axe du modèle, ni un sens."""
        self.assertEqual(self._f(CONTINUER_AGRESSER=34).sens(), (0, 0))

    def test_sans_mise_miser_quand_il_checke_c_est_jouer_plus(self) -> None:
        f = self._f(AGRESSER_CHECK=201, CHECK_AGRESSER=191,
                    CHECK_CHECK=449, AGRESSER_AGRESSER=150)
        self.assertEqual(f.sens(), (201, 191))

    def test_preflop_un_limp_compte_comme_de_l_argent_mis(self) -> None:
        """Conseil FOLD, limp joué : c'est bien NOUS qui jouons moins."""
        self.assertEqual(self._f(PASSER_CONTINUER=12).sens(), (0, 12))
        self.assertEqual(self._f(AGRESSER_CONTINUER=12).sens(), (0, 0))

    def test_ce_que_le_conseiller_ne_tranche_pas_n_a_pas_de_sens(self) -> None:
        f = self._f(MIXTE_AGRESSER=50, REFUS_PASSER=30, INDECIS_CONTINUER=20)
        self.assertEqual(f.sens(), (0, 0))

    def test_plus_et_moins_recouvrent_exactement_les_desaccords(self) -> None:
        """Invariant : sur l'axe binaire, chaque désaccord a un sens et un seul.

        Si un cas s'échappait, le rapport afficherait un sens qui ne recouvre
        pas son propre taux — et personne ne le verrait.
        """
        for axe, cellules in (
            ("preflop", {"AGRESSER_PASSER": 3, "PASSER_AGRESSER": 5,
                         "PASSER_CONTINUER": 7, "AGRESSER_AGRESSER": 11,
                         "AGRESSER_CONTINUER": 2, "PASSER_PASSER": 13}),
            ("face_mise", {"CONTINUER_PASSER": 3, "PASSER_CONTINUER": 5,
                           "PASSER_AGRESSER": 7, "CONTINUER_AGRESSER": 11,
                           "CONTINUER_CONTINUER": 2, "PASSER_PASSER": 13}),
            ("sans_mise", {"AGRESSER_CHECK": 3, "CHECK_AGRESSER": 5,
                           "CHECK_CHECK": 11, "AGRESSER_AGRESSER": 13}),
        ):
            with self.subTest(axe=axe):
                f = self._f(**cellules)
                attendus = sum(
                    k for (vu, fait), k in f.croise.items()
                    if banc._accord(axe, vu, fait) is False)
                self.assertEqual(sum(f.sens()), attendus)


def _decision(**kw):
    """Une décision de rejeu, préflop CO par défaut, à modifier par mot-clé."""
    from pfs.data.phh import Decision
    base = dict(hand_id="x", ordre=0, street=Street.PREFLOP, player="p",
                seat=1, position="CO", cards=("As", "Kd"), board=(),
                pot=150.0, pot_gagnable=150.0, to_call=100.0, stack=9900.0,
                stack_effectif=9900.0, engage=0.0, actifs=6,
                relances_avant=0, action=ActionType.FOLD, montant=0.0)
    base.update(kw)
    return Decision(**base)


class TestFamilleDuSpot(unittest.TestCase):
    """Le classement d'un spot décide dans quelle famille il pèse."""

    def _d(self, **kw) -> "banc.Decision":
        return _decision(**kw)

    def test_ouverture_propre_contre_apres_limp(self) -> None:
        """La chart d'ouverture suppose un pot non ouvert : un limp devant
        change le spot, et le mélanger avec les ouvertures propres diluerait
        le seul régime où la chart est dans son domaine."""
        self.assertEqual(banc.famille_du_spot(self._d(), 0, 99.0)[1],
                         "préflop · ouverture propre · CO")
        self.assertEqual(banc.famille_du_spot(self._d(), 1, 99.0)[1],
                         "préflop · ouverture après limp")

    def test_face_a_une_relance_est_sa_propre_famille(self) -> None:
        self.assertEqual(
            banc.famille_du_spot(self._d(relances_avant=1), 0, 99.0)[1],
            "préflop · face à une relance")

    def test_postflop_separe_heads_up_et_multiway(self) -> None:
        """Le modèle postflop suppose UN adversaire : fondre les pots
        multiway dans le même chiffre masquerait un hors-domaine."""
        hu = self._d(street=Street.FLOP, board=("2c", "7d", "9h"), actifs=2)
        multi = self._d(street=Street.FLOP, board=("2c", "7d", "9h"), actifs=3)
        self.assertEqual(banc.famille_du_spot(hu, 0, 99.0)[1],
                         "postflop · face à une mise · heads-up · flop")
        self.assertEqual(banc.famille_du_spot(multi, 0, 99.0)[1],
                         "postflop · face à une mise · 3 joueurs · flop")

    def test_sans_mise_est_separe_de_face_a_une_mise(self) -> None:
        d = self._d(street=Street.TURN, board=("2c", "7d", "9h", "Js"),
                    to_call=0.0, actifs=2)
        self.assertEqual(banc.famille_du_spot(d, 0, 99.0)[1],
                         "postflop · sans mise · heads-up · turn")

    def test_la_famille_ne_depend_pas_de_la_profondeur(self) -> None:
        """La profondeur pilote le RÉGIME, pas la famille : sinon chaque
        famille préflop se dédoublerait et perdrait la moitié de son
        effectif, donc la moitié de sa puissance statistique."""
        for prof in (3.0, 14.9, 15.0, 100.0):
            with self.subTest(prof=prof):
                self.assertEqual(banc.famille_du_spot(self._d(), 0, prof)[1],
                                 "préflop · ouverture propre · CO")


class TestRegimeDuSpot(unittest.TestCase):
    """Les régimes du rapport sont ceux que la question exige.

    Un taux global mélange des moteurs différents : c'est ce découpage qui
    fait que le rapport dit quelque chose. Une erreur ici viderait un régime
    au profit d'un autre sans qu'aucun total ne bouge.
    """

    def test_la_frontiere_du_tapis_court_est_a_quinze_bb(self) -> None:
        """Strictement en dessous : court. À 15,0 exactement : profond.

        La borne est celle écrite dans :data:`banc.SEUIL_TAPIS_COURT_BB` ;
        le test la relit pour rester juste si elle change, mais épingle le
        SENS de la comparaison, qui est ce qu'une erreur inverserait.
        """
        seuil = banc.SEUIL_TAPIS_COURT_BB
        self.assertEqual(banc.regime_du_spot(_decision(), seuil - 0.1),
                         banc.REGIME_COURT)
        self.assertEqual(banc.regime_du_spot(_decision(), seuil),
                         banc.REGIME_PROFOND)
        self.assertEqual(banc.regime_du_spot(_decision(), seuil + 0.1),
                         banc.REGIME_PROFOND)

    def test_le_postflop_est_separe_rue_par_rue(self) -> None:
        """Flop, turn et river n'ont pas le même modèle chez nous (990
        runouts, 44 runouts, exact) : les fondre en « postflop » cacherait
        lequel des trois diverge."""
        for rue, board, attendu in (
            (Street.FLOP, ("2c", "7d", "9h"), "postflop · flop"),
            (Street.TURN, ("2c", "7d", "9h", "Js"), "postflop · turn"),
            (Street.RIVER, ("2c", "7d", "9h", "Js", "4d"), "postflop · river"),
        ):
            with self.subTest(rue=rue):
                d = _decision(street=rue, board=board)
                self.assertEqual(banc.regime_du_spot(d, 100.0), attendu)

    def test_la_profondeur_ne_touche_pas_le_postflop(self) -> None:
        """« Tapis court » est un régime PRÉFLOP : au flop, la question n'est
        plus pousser ou passer."""
        d = _decision(street=Street.FLOP, board=("2c", "7d", "9h"))
        self.assertEqual(banc.regime_du_spot(d, 2.0), "postflop · flop")

    def test_tous_les_regimes_produits_sont_dans_la_nomenclature(self) -> None:
        """ORDRE_REGIMES pilote l'affichage : un régime hors liste serait
        rejeté en bas du rapport sous « hors nomenclature ». Aucun régime
        normalement produit ne doit s'y retrouver."""
        boards = {Street.PREFLOP: (),
                  Street.FLOP: ("2c", "7d", "9h"),
                  Street.TURN: ("2c", "7d", "9h", "Js"),
                  Street.RIVER: ("2c", "7d", "9h", "Js", "4d")}
        produits = {banc.regime_du_spot(_decision(street=r, board=b), prof)
                    for r, b in boards.items() for prof in (5.0, 100.0)}
        self.assertEqual(produits, set(banc.ORDRE_REGIMES))


class TestProfondeurBb(unittest.TestCase):
    """La profondeur décide du régime : la mesurer faux le viderait."""

    class _Main:
        def __init__(self, bb: float) -> None:
            self.big_blind = bb

    def test_c_est_le_tapis_effectif_qui_compte(self) -> None:
        """Un tapis de 200 bb face à un adversaire à 8 bb est un spot de
        8 bb : c'est ce qui peut encore être misé qui fait le régime."""
        d = _decision(stack=20000.0, stack_effectif=800.0)
        self.assertAlmostEqual(banc.profondeur_bb(d, self._Main(100.0)), 8.0)

    def test_big_blind_nulle_ne_fabrique_pas_un_tapis_court(self) -> None:
        """Sans big blind lisible, le quotient n'a pas de sens. Rendre 0
        classerait le spot dans le régime le plus intéressant du rapport ;
        l'infini le laisse en « profond », c'est-à-dire hors du régime que
        ce banc ne sait de toute façon pas mesurer."""
        d = _decision(stack_effectif=800.0)
        self.assertEqual(banc.profondeur_bb(d, self._Main(0.0)), math.inf)
        self.assertEqual(banc.regime_du_spot(d, math.inf), banc.REGIME_PROFOND)


class TestTextureBoard(unittest.TestCase):
    """La texture est un des axes qui caractérisent un défaut."""

    def test_appariement(self) -> None:
        self.assertEqual(banc.texture_board(("8c", "8d", "Kh"))["appariement"],
                         "appairé")
        self.assertEqual(banc.texture_board(("2c", "7d", "Kh"))["appariement"],
                         "non appairé")

    def test_appariement_vu_au_turn_et_a_la_river(self) -> None:
        """Le board s'apparie souvent APRÈS le flop : ne regarder que trois
        cartes classerait un turn appairé comme non appairé."""
        self.assertEqual(
            banc.texture_board(("2c", "7d", "Kh", "7s"))["appariement"],
            "appairé")

    def test_couleur(self) -> None:
        """Le seuil est à trois cartes d'une même couleur : c'est à partir de
        là qu'une couleur est faite ou à un tirage."""
        self.assertEqual(banc.texture_board(("Ah", "Kh", "Qh"))["couleur"],
                         "3+ assorties")
        self.assertEqual(banc.texture_board(("Ah", "Kh", "Qd"))["couleur"],
                         "2 assorties au plus")
        self.assertEqual(
            banc.texture_board(("Ah", "Kh", "Qd", "2h"))["couleur"],
            "3+ assorties")

    def test_connexite(self) -> None:
        self.assertEqual(banc.texture_board(("9c", "8d", "2h"))["connexité"],
                         "connecté")
        self.assertEqual(banc.texture_board(("9c", "7d", "2h"))["connexité"],
                         "connecté")
        self.assertEqual(banc.texture_board(("Kc", "7d", "2h"))["connexité"],
                         "sec")

    def test_l_as_compte_aussi_comme_carte_basse(self) -> None:
        """A-2 est adjacent (la roue), A-9 ne l'est pas.

        Le cas discriminant est A-2-9 : sans la règle « l'as compte aussi
        comme basse », l'écart A→2 vaut douze crans et le board passerait
        pour sec. A-2-3 ne prouverait rien — 2 et 3 sont déjà adjacents entre
        eux et le board serait dit connecté de toute façon.
        """
        self.assertEqual(banc.texture_board(("Ac", "2d", "9h"))["connexité"],
                         "connecté")
        self.assertEqual(banc.texture_board(("Ac", "3d", "9h"))["connexité"],
                         "connecté")
        # l'as ne rend pas TOUT connecté : A-5 fait quatre crans.
        self.assertEqual(banc.texture_board(("Ac", "5d", "9h"))["connexité"],
                         "sec")

    def test_le_seuil_decoupe_vraiment_la_population(self) -> None:
        """Un axe dont une valeur écrase l'autre ne caractérise rien.

        Énumération EXHAUSTIVE des 22 100 flops possibles (C(52,3)), tous
        axes comptés :

        =============  ====================  =======
        axe            répartition           part
        =============  ====================  =======
        connexité      14 960 / 7 140        67,69 %
        appariement    3 796 / 18 304        17,18 %
        couleur        1 144 / 20 956         5,18 %
        =============  ====================  =======

        Ce test épingle le pouvoir de coupe de chaque axe, pas une propriété
        du corpus : il tombe si quelqu'un resserre la connexité aux rangs
        strictement adjacents (8 944 au lieu de 14 960) ou déplace le seuil
        de couleur. Il documente aussi, en dur, que l'axe couleur est
        déséquilibré — c'est une limite à lire dans le rapport, pas un bug.
        """
        from itertools import combinations
        noms = [r + s for r in "AKQJT98765432" for s in "shdc"]
        flops = list(combinations(noms, 3))
        self.assertEqual(len(flops), 22100)
        textures = [banc.texture_board(f) for f in flops]
        for axe, attendu in (
            ("connexité", {"connecté": 14960, "sec": 7140}),
            ("appariement", {"appairé": 3796, "non appairé": 18304}),
            ("couleur", {"3+ assorties": 1144, "2 assorties au plus": 20956}),
        ):
            with self.subTest(axe=axe):
                self.assertEqual(dict(Counter(t[axe] for t in textures)),
                                 attendu)
        # La connexité et l'appariement coupent assez pour caractériser ; la
        # couleur non — le test le dit plutôt que de le taire.
        connexite = Counter(t["connexité"] for t in textures)
        self.assertGreater(min(connexite.values()) / 22100, 0.20)

    def test_les_trois_axes_sont_toujours_rendus(self) -> None:
        """Un axe manquant ferait disparaître silencieusement une colonne du
        rapport pour toute une famille."""
        for board in (("2c", "7d", "9h"), ("2c", "7d", "9h", "Js"),
                      ("2c", "7d", "9h", "Js", "4d")):
            with self.subTest(board=board):
                self.assertEqual(set(banc.texture_board(board)),
                                 {"appariement", "couleur", "connexité"})


class TestTypeDeMain(unittest.TestCase):
    """L'axe « type de main », préflop par la forme, postflop par la force."""

    def test_preflop_trois_formes(self) -> None:
        self.assertEqual(banc.type_de_main(("As", "Ad"), ()), "paire servie")
        self.assertEqual(banc.type_de_main(("As", "Ks"), ()), "assortie")
        self.assertEqual(banc.type_de_main(("As", "Kd"), ()), "dépareillée")

    def test_postflop_c_est_la_main_faite(self) -> None:
        """Au flop il n'y a que cinq cartes : évaluer sept en inventerait
        deux, et un board de flop rendrait des mains fausses."""
        self.assertEqual(banc.type_de_main(("As", "Kd"), ("Qh", "7c", "2d")),
                         "hauteur")
        self.assertEqual(banc.type_de_main(("As", "Kd"), ("Ah", "7c", "2d")),
                         "une paire")
        self.assertEqual(banc.type_de_main(("As", "Kd"), ("Ah", "Kc", "2d")),
                         "deux paires ou mieux")

    def test_la_meilleure_main_est_prise_au_turn(self) -> None:
        """Au turn il y a SIX cartes : il faut énumérer les vingt et une… non,
        les six sous-mains de cinq, pas garder les cinq premières.

        Cas discriminant : héros 7s 2d, board Ah Ac Kd Kh. Les cinq premières
        cartes de la concaténation (7s 2d Ah Ac Kd) ne font qu'une paire ;
        la meilleure combinaison (Ah Ac Kd Kh + 7s) en fait deux.
        """
        self.assertEqual(
            banc.type_de_main(("7s", "2d"), ("Ah", "Ac", "Kd", "Kh")),
            "deux paires ou mieux")
        self.assertEqual(
            banc.type_de_main(("As", "Kd"), ("Ah", "7c", "2d", "3s")),
            "une paire")

    def test_la_river_passe_par_l_evaluateur_sept_cartes(self) -> None:
        """Sept cartes : c'est ``evaluate7`` qui tranche, et il doit trouver
        la suite 2-3-4-5-6 alors que le board porte aussi K et Q."""
        self.assertEqual(
            banc.type_de_main(("As", "Kd"), ("Ah", "7c", "2d", "3s", "Kh")),
            "deux paires ou mieux")
        self.assertEqual(
            banc.type_de_main(("2s", "3d"), ("4h", "5c", "6d", "Ks", "Qh")),
            "deux paires ou mieux")

    def test_la_paire_du_board_seule_compte_comme_paire(self) -> None:
        """Limite assumée et écrite : l'axe décrit la main FAITE, sans savoir
        si le héros y participe. Le test l'épingle pour que personne ne lise
        « une paire » comme « le héros a touché »."""
        self.assertEqual(banc.type_de_main(("7s", "2d"), ("Ah", "Ac", "9d")),
                         "une paire")

    def test_un_board_illisible_est_signale_pas_avale(self) -> None:
        self.assertEqual(banc.type_de_main(("As",), ()), "?")


class TestAxesDuSpot(unittest.TestCase):
    """Les axes comptés doivent couvrir ce que la question demande."""

    def test_preflop_pas_d_axe_de_board(self) -> None:
        axes = banc.axes_du_spot(_decision(), 100.0)
        self.assertEqual(axes["position"], "CO")
        self.assertEqual(axes["type de main"], "dépareillée")
        self.assertTrue(all(not k.startswith("board") for k in axes))

    def test_postflop_les_quatre_axes_demandes(self) -> None:
        d = _decision(street=Street.FLOP, board=("Ah", "Kh", "Qh"), actifs=2,
                      cards=("7s", "2d"))
        axes = banc.axes_du_spot(d, 100.0)
        self.assertEqual(axes["position"], "CO")
        self.assertEqual(axes["profondeur"], "≥ 15 bb")
        self.assertEqual(axes["board · couleur"], "3+ assorties")
        self.assertEqual(axes["board · connexité"], "connecté")
        self.assertEqual(axes["type de main"], "hauteur")
        self.assertEqual(axes["joueurs actifs"], "heads-up")

    def test_l_axe_profondeur_suit_le_meme_seuil_que_le_regime(self) -> None:
        """Deux seuils divergents feraient dire au rapport qu'un spot est
        court sur un axe et profond sur l'autre."""
        court = banc.axes_du_spot(_decision(),
                                  banc.SEUIL_TAPIS_COURT_BB - 0.1)
        profond = banc.axes_du_spot(_decision(), banc.SEUIL_TAPIS_COURT_BB)
        self.assertNotEqual(court["profondeur"], profond["profondeur"])
        self.assertEqual(court["profondeur"], "< 15 bb")


class TestComptageDesAxes(unittest.TestCase):
    """La comptabilité des axes doit recoller au taux qu'elle explique."""

    def test_chaque_axe_totalise_les_spots_tranches(self) -> None:
        """Invariant : sur un axe, la somme des cases vaut le nombre de spots
        tranchés de la famille. S'il s'en échappait, le rapport afficherait
        une caractérisation qui ne couvre pas son propre taux."""
        f = banc.Famille()
        for i in range(7):
            d = _decision(street=Street.FLOP, board=("Ah", "7c", "2d"),
                          actifs=2)
            f.compter_axes(banc.axes_du_spot(d, 100.0), accord=i % 2 == 0)
            f.accord += int(i % 2 == 0)
            f.desaccord += int(i % 2 != 0)
        for nom, valeurs in f.axes.items():
            with self.subTest(axe=nom):
                self.assertEqual(sum(a + d for a, d in valeurs.values()),
                                 f.tranches)
                self.assertEqual(sum(d for _, d in valeurs.values()),
                                 f.desaccord)

    def test_les_non_tranches_n_entrent_pas_dans_les_axes(self) -> None:
        """Un MIXTE compté d'un côté fabriquerait l'écart qu'on cherche."""
        f = banc.Famille()
        f.indecis = 5
        f.refus = 3
        self.assertEqual(sum(len(v) for v in f.axes.values()), 0)

    def test_la_fusion_conserve_les_totaux(self) -> None:
        a, b = banc.Famille(), banc.Famille()
        axes = {"position": "CO"}
        for _ in range(3):
            a.compter_axes(axes, accord=True)
        for _ in range(4):
            b.compter_axes(axes, accord=False)
        a.fusionner_axes(b)
        self.assertEqual(a.axes["position"]["CO"], [3, 4])


class TestComposantsSollicites(unittest.TestCase):
    """« Quelle partie du conseiller cette comparaison valide-t-elle ? »

    C'est la question à laquelle la section 9 répond, et elle y répond en
    COMPTANT. Une erreur de classement ferait dire au rapport que l'ICM a été
    éprouvé sur un corpus de cash game — exactement le contresens que ce banc
    existe pour empêcher.
    """

    def test_le_motif_le_plus_specifique_gagne(self) -> None:
        """« équité exacte au flop vs cotes du pot (ICM) » contient les trois
        motifs. Sans priorité, un seul spot serait compté trois fois."""
        self.assertEqual(
            banc._classer_moteur("équité exacte au flop vs cotes du pot (ICM)"),
            "correction ICM des cotes du pot postflop")
        self.assertEqual(
            banc._classer_moteur("équité exacte au flop vs cotes du pot"),
            "équité exacte postflop vs cotes du pot")
        self.assertEqual(banc._classer_moteur("équité exacte au turn"),
                         "équité exacte postflop sans mise à payer")

    def test_les_deux_push_fold_ne_sont_pas_confondus(self) -> None:
        """Le chipEV et l'ICM sont deux composants distincts : le second est
        celui qui sert en Twister, et le rapport doit pouvoir dire qu'il n'a
        pas été touché même si le premier l'avait été."""
        self.assertEqual(
            banc._classer_moteur("Nash push/fold ICM N bb (N joueurs)"),
            "Nash push/fold ICM + bubble factor")
        self.assertEqual(banc._classer_moteur("Nash push/fold N bb"),
                         "Nash push/fold chipEV")

    def test_un_moteur_inconnu_reste_visible(self) -> None:
        """Rangé dans un composant plausible, un moteur neuf gonflerait
        silencieusement une couverture qu'il ne fournit pas."""
        self.assertEqual(banc._classer_moteur("solveur turn range vs range"),
                         "« non classé »")

    def test_tous_les_regimes_du_conseiller_sont_classes(self) -> None:
        """La liste vient des chaînes que ``spot_advisor`` construit vraiment.

        Si l'un de ses libellés change, ce test tombe plutôt que de laisser
        la section 9 déclarer « jamais sollicité » un composant qui l'a été.
        """
        for regime in ("chart d'ouverture CO",
                       "pas de chart pour cette position",
                       "Nash push/fold N bb",
                       "Nash push/fold ICM N bb (N joueurs)",
                       "équité exacte au flop",
                       "équité exacte au turn",
                       "équité exacte au river",
                       "équité exacte au flop vs cotes du pot",
                       "équité exacte au river vs cotes du pot (ICM)",
                       "solveur CFR à la river"):
            with self.subTest(regime=regime):
                self.assertNotEqual(banc._classer_moteur(regime),
                                    "« non classé »")

    def test_le_total_par_composant_recolle_au_total_des_moteurs(self) -> None:
        """Invariant : aucun spot perdu, aucun spot compté deux fois."""
        moteurs = Counter({
            "chart d'ouverture CO": 900,
            "pas de chart pour cette position": 300,
            "équité exacte au flop": 120,
            "équité exacte au flop vs cotes du pot": 80,
            "équité exacte au river vs cotes du pot (ICM)": 7,
            "moteur inventé demain": 3,
        })
        lignes = banc._composants_sollicites(moteurs)
        self.assertEqual(sum(k for _, k, _ in lignes), sum(moteurs.values()))
        self.assertEqual(len({nom for nom, _, _ in lignes}), len(lignes))

    def test_un_composant_jamais_sollicite_est_affiche_a_zero(self) -> None:
        """Le zéro EST le résultat : c'est lui qui dit que le push/fold de
        tournoi n'est pas validé par ce banc."""
        lignes = dict((nom, k) for nom, k, _ in
                      banc._composants_sollicites(Counter(
                          {"chart d'ouverture CO": 10})))
        self.assertEqual(lignes["Nash push/fold ICM + bubble factor"], 0)
        self.assertEqual(lignes["Nash push/fold chipEV"], 0)
        self.assertEqual(lignes["ranges d'ouverture (GTO_PRESETS)"], 10)


class TestDisjonctionDesIntervalles(unittest.TestCase):
    """Le critère qui décide si un axe « parle » — donc s'il est cité."""

    def test_un_ecart_franc_est_reconnu(self) -> None:
        self.assertTrue(banc._disjoints(90, 100, 10, 100))

    def test_un_ecart_de_bruit_ne_l_est_pas(self) -> None:
        """52 % contre 48 % sur 100 spots chacun : les intervalles se
        chevauchent largement. Le déclarer significatif serait exactement la
        pêche aux corrélations que ce banc refuse."""
        self.assertFalse(banc._disjoints(52, 100, 48, 100))

    def test_le_critere_est_conservateur(self) -> None:
        """Deux intervalles disjoints impliquent un écart significatif ; la
        réciproque est fausse, et c'est voulu.

        55/100 contre 40/100 : le test de comparaison de deux proportions
        rejette l'égalité (z = 0,15 / 0,0706 = 2,12, p ≈ 0,034), alors que
        les intervalles de Wilson [45,2 ; 64,4] et [30,9 ; 49,8] se
        chevauchent. Le banc ne le marquera donc PAS. Il rate des écarts
        réels — c'est le bon sens de l'erreur quand le risque est d'inventer
        des familles de défauts sur 10 000 mains.
        """
        self.assertFalse(banc._disjoints(55, 100, 40, 100))
        # …et il ne rate pas ce qui est franc : 60 contre 40 est marqué.
        self.assertTrue(banc._disjoints(60, 100, 40, 100))

    def test_sans_effectif_rien_n_est_disjoint(self) -> None:
        self.assertFalse(banc._disjoints(0, 0, 0, 0))


class TestCoutRealiseRiver(unittest.TestCase):
    """Le seul chiffre du rapport indépendant de nos hypothèses.

    Main construite à la main (blindes 50/100, deux joueurs, tapis 10 000) :
    p1 monte à 300, p2 suit ; flop, turn et river passent en check ; à la
    river p2 mise 400 et p1 doit décider.

    À cet instant p1 a engagé 300, p2 700 : pot 1 000, à payer 400. Suivre
    vaut donc **+1 000 jetons (10 bb)** si p1 gagne, **−400 (−4 bb)** s'il
    perd, et **+300 (+3 bb)** en cas de partage — (pot − mise) / 2.
    """

    GABARIT = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [50, 100]
starting_stacks = [10000, 10000]
actions = [
  "d dh p1 {h1}", "d dh p2 {h2}",
  "p1 cbr 300", "p2 cc",
  "d db {flop}", "p2 cc", "p1 cc",
  "d db {turn}", "p2 cc", "p1 cc",
  "d db {river}", "p2 cbr 400", "p1 cc",
]
finishing_stacks = [{f1}, {f2}]
"""

    def _river(self, h1: str, h2: str, flop: str, turn: str, river: str,
               f1: int, f2: int):
        main = parse_phh(self.GABARIT.format(h1=h1, h2=h2, flop=flop,
                                             turn=turn, river=river,
                                             f1=f1, f2=f2))
        d = [x for x in iter_decisions(main)
             if x.street is Street.RIVER and x.to_call > 0][0]
        return d, main

    def test_le_spot_est_bien_celui_decrit(self) -> None:
        d, _ = self._river("AsKs", "2c3d", "QsJsTs", "2h", "5c", 10700, 9300)
        self.assertEqual((d.pot, d.pot_gagnable, d.to_call), (1000.0, 1000.0, 400.0))

    def test_gagnant_le_suivi_vaut_le_pot(self) -> None:
        """p1 a la quinte flush royale : +1 000 jetons = +10 bb."""
        d, main = self._river("AsKs", "2c3d", "QsJsTs", "2h", "5c", 10700, 9300)
        self.assertAlmostEqual(banc.cout_realise_river(d, main), 10.0)

    def test_perdant_le_suivi_ne_coute_que_la_mise(self) -> None:
        """p1 n'a rien : −400 jetons = −4 bb (le pot était déjà perdu)."""
        d, main = self._river("2c3d", "AsKs", "QsJsTs", "2h", "5c", 9300, 10700)
        self.assertAlmostEqual(banc.cout_realise_river(d, main), -4.0)

    def test_partage_la_moitie_de_l_ecart(self) -> None:
        """Board AKQJT : les deux jouent le tableau. (1 000 − 400) / 2 = +3 bb.

        Le cas qu'une formule bâclée rate : rendre 0, ou la moitié du pot.
        """
        d, main = self._river("2c3d", "4s6s", "AhKhQd", "Jc", "Ts", 10000, 10000)
        self.assertAlmostEqual(banc.cout_realise_river(d, main), 3.0)

    def test_refus_quand_une_hypothese_serait_necessaire(self) -> None:
        """Hors river, hors tête-à-tête ou sans mise à payer : None.

        Le chiffre ne vaut que parce qu'il ne suppose RIEN ; l'étendre à un
        turn (où il resterait une carte à venir) le rendrait faux.
        """
        main = parse_phh(self.GABARIT.format(h1="AsKs", h2="2c3d",
                                             flop="QsJsTs", turn="2h",
                                             river="5c", f1=10700, f2=9300))
        for d in iter_decisions(main):
            if d.street is not Street.RIVER or d.to_call <= 0:
                self.assertIsNone(banc.cout_realise_river(d, main),
                                  msg=f"décision #{d.ordre} {d.street}")


if __name__ == "__main__":
    unittest.main()
