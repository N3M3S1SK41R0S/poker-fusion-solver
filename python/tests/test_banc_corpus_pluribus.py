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


class TestFamilleDuSpot(unittest.TestCase):
    """Le classement d'un spot décide dans quelle famille il pèse."""

    def _d(self, **kw) -> "banc.Decision":
        from pfs.data.phh import Decision
        base = dict(hand_id="x", ordre=0, street=Street.PREFLOP, player="p",
                    seat=1, position="CO", cards=("As", "Kd"), board=(),
                    pot=150.0, pot_gagnable=150.0, to_call=100.0, stack=9900.0,
                    stack_effectif=9900.0, engage=0.0, actifs=6,
                    relances_avant=0, action=ActionType.FOLD, montant=0.0)
        base.update(kw)
        return Decision(**base)

    def test_ouverture_propre_contre_apres_limp(self) -> None:
        """La chart d'ouverture suppose un pot non ouvert : un limp devant
        change le spot, et le mélanger avec les ouvertures propres diluerait
        le seul régime où la chart est dans son domaine."""
        self.assertEqual(banc.famille_du_spot(self._d(), 0)[1],
                         "préflop · ouverture propre · CO")
        self.assertEqual(banc.famille_du_spot(self._d(), 1)[1],
                         "préflop · ouverture après limp")

    def test_face_a_une_relance_est_sa_propre_famille(self) -> None:
        self.assertEqual(banc.famille_du_spot(self._d(relances_avant=1), 0)[1],
                         "préflop · face à une relance")

    def test_postflop_separe_heads_up_et_multiway(self) -> None:
        """Le modèle postflop suppose UN adversaire : fondre les pots
        multiway dans le même chiffre masquerait un hors-domaine."""
        hu = self._d(street=Street.FLOP, board=("2c", "7d", "9h"), actifs=2)
        multi = self._d(street=Street.FLOP, board=("2c", "7d", "9h"), actifs=3)
        self.assertEqual(banc.famille_du_spot(hu, 0)[1],
                         "postflop · face à une mise · heads-up · flop")
        self.assertEqual(banc.famille_du_spot(multi, 0)[1],
                         "postflop · face à une mise · 3 joueurs · flop")

    def test_sans_mise_est_separe_de_face_a_une_mise(self) -> None:
        d = self._d(street=Street.TURN, board=("2c", "7d", "9h", "Js"),
                    to_call=0.0, actifs=2)
        self.assertEqual(banc.famille_du_spot(d, 0)[1],
                         "postflop · sans mise · heads-up · turn")


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
