"""Tests du modèle de DÉFENSE préflop face à une ouverture.

Le banc Pluribus a mesuré le défaut (section 9 de son rapport) : le
conseiller répondait à « faut-il défendre ? » avec la chart d'OUVERTURE de la
position, un modèle appliqué hors de son domaine. Le modèle testé ici répond
à la bonne question, avec des seuils DÉRIVÉS — cote du pot exacte, MDF face à
la taille, part du héros — jamais posés au doigt mouillé.

Goldens ancrés sur des calculs refaits à la main (blindes 0,5/1) :

* BB face à CO 2,5 bb : payer 1,5 pour un pot final de 5,5 → α = 3/11 ;
  l'ouvreur risque 2,5 pour 1,5 d'argent mort → MDF = 1,5/4 = 37,5 %.
* mini-ouverture (2 bb) : α = 1/4,5 ; MDF = 1,5/3,5 = 42,86 %.
* ouverture énorme (10 bb) : α = 9/20,5 ; MDF = 1,5/11,5 = 13,04 %.
* blinde contre blinde (SB ouvre à 3 bb) : α = 2/6 = 1/3 ; la blinde déjà
  investie de l'ouvreur est déduite de son risque → MDF = 1,5/4 = 37,5 %,
  la valeur exacte.
* CO face à UTG (4 défenseurs restants) : α = 2,5/6,5 ; MDF partagée
  symétriquement → part du héros 1 − 0,625^(1/4) = 11,09 % des mains.
"""

from __future__ import annotations

import doctest
import sys
import unittest
from pathlib import Path

import pfs.analysis.spot_advisor as sa
import pfs.core.range_model as rm
from pfs.analysis.spot_advisor import Spot, advise

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _bb_vs_co(main: str, **kw) -> "sa.Advice":
    """BB face à une ouverture CO à 2,5 bb, blindes 0,5/1 : pot affiché 4."""
    base = dict(hero=main, position="BB", opener="CO", pot=4.0, bet=1.5,
                stack=97.5, big_blind=1.0, players=2)
    base.update(kw)
    return advise(Spot(**base))


def load_tests(loader, tests, ignore):
    """Les goldens écrits dans les docstrings font partie de la suite."""
    tests.addTests(doctest.DocTestSuite(sa))
    tests.addTests(doctest.DocTestSuite(rm))
    return tests


class TestSeuilsDerives(unittest.TestCase):
    """Chaque seuil sort d'une formule de la docstring, pas d'un réglage."""

    def test_cote_du_pot_exacte(self) -> None:
        a = _bb_vs_co("Kh 9d")
        self.assertAlmostEqual(a.required, 1.5 / 5.5, places=9)

    def test_mdf_bb_vs_co(self) -> None:
        # R = 1,5 + 1 (blinde du héros) − 0 ; P0 = 4 − 2,5 = 1,5.
        a = _bb_vs_co("Kh 9d")
        self.assertAlmostEqual(a.mdf, 1.5 / 4.0, places=9)

    def test_mini_ouverture(self) -> None:
        a = advise(Spot(hero="Kh 9d", position="BB", opener="CO", pot=3.5,
                        bet=1.0, stack=98.0, big_blind=1.0))
        self.assertAlmostEqual(a.required, 1.0 / 4.5, places=9)
        self.assertAlmostEqual(a.mdf, 1.5 / 3.5, places=9)
        self.assertTrue(a.action.startswith("CALL"))  # défense LARGE à 2 bb

    def test_ouverture_enorme(self) -> None:
        a = advise(Spot(hero="Kh 9d", position="BB", opener="CO", pot=11.5,
                        bet=9.0, stack=90.0, big_blind=1.0))
        self.assertAlmostEqual(a.required, 9.0 / 20.5, places=9)
        self.assertAlmostEqual(a.mdf, 1.5 / 11.5, places=9)
        self.assertEqual(a.action, "FOLD")            # défense SERRÉE à 10 bb

    def test_blinde_contre_blinde(self) -> None:
        # SB ouvre à 3 : sa blinde de 0,5 est déduite de son risque.
        a = advise(Spot(hero="Kh 9d", position="BB", opener="SB", pot=4.0,
                        bet=2.0, stack=97.0, big_blind=1.0, players=2))
        self.assertAlmostEqual(a.required, 1.0 / 3.0, places=9)
        self.assertAlmostEqual(a.mdf, 1.5 / 4.0, places=9)
        self.assertTrue(any("Blinde contre blinde" in r for r in a.reasons))

    def test_mdf_partagee_entre_defenseurs(self) -> None:
        # CO face à UTG, BTN + SB + BB encore derrière : 4 défenseurs.
        a = advise(Spot(hero="Ah Kd", position="CO", opener="UTG", pot=4.0,
                        bet=2.5, stack=97.5, big_blind=1.0, players=5))
        self.assertAlmostEqual(a.required, 2.5 / 6.5, places=9)
        self.assertAlmostEqual(a.mdf, 1.5 / 4.0, places=9)
        part = 1.0 - (1.0 - 1.5 / 4.0) ** 0.25        # 11,09 %
        self.assertTrue(any(f"{part * 100:.1f} %" in r for r in a.reasons))

    def test_sans_ouvreur_declare_la_blinde_esperee_est_deduite(self) -> None:
        # Sans « opener », P(SB)·0,5 bb est déduit du risque de l'ouvreur :
        # la MDF s'élargit par rapport au cas « ouvreur hors des blindes ».
        avec = advise(Spot(hero="Kh 9d", position="BB", opener="CO", pot=3.5,
                           bet=1.0, stack=98.0, big_blind=1.0))
        sans = advise(Spot(hero="Kh 9d", position="BB", pot=3.5, bet=1.0,
                           stack=98.0, big_blind=1.0))
        self.assertGreater(sans.mdf, avec.mdf)

    def test_icm_releve_la_cote(self) -> None:
        cash = _bb_vs_co("Kh 9d")
        icm = _bb_vs_co("Kh 9d", stacks="30, 40, 5", payouts="65, 35",
                        players=3, hero_seat=0, villain_seat=1)
        self.assertGreater(icm.bubble, 1.0)
        self.assertGreater(icm.required, cash.required)
        self.assertEqual(icm.action, "FOLD")          # K9o call en cash…
        self.assertTrue(cash.action.startswith("CALL"))


class TestVerdicts(unittest.TestCase):
    """Le verdict suit le classement par équité contre la range d'ouverture."""

    def test_premium_relance_pour_la_valeur(self) -> None:
        a = _bb_vs_co("Ah Ad")
        self.assertEqual(a.action, "CALL ou 3-BET (valeur)")
        self.assertEqual(a.confidence, "indicatif")

    def test_poubelle_couche(self) -> None:
        self.assertEqual(_bb_vs_co("7h 2d").action, "FOLD")

    def test_main_moyenne_defend(self) -> None:
        a = _bb_vs_co("Kh 9d")
        self.assertEqual(a.action, "CALL (défense)")

    def test_bluff_candidate_sous_le_seuil_avec_blocker(self) -> None:
        # K3o : juste sous le seuil de call, roi en blocker → 3-bet bluff en
        # fréquence. Le verdict est MIXTE : le banc le classe « indécis »,
        # jamais accord ni désaccord — c'est une réponse en fréquence.
        a = _bb_vs_co("Kh 3d")
        self.assertTrue(a.action.startswith("MIXTE — 3-bet bluff"), a.action)

    def test_frontiere_est_marginale_pas_tranchee(self) -> None:
        # K6o mesuré à ~1 point du seuil : dans le bruit Monte-Carlo de la
        # matrice d'équité (±0,9 pt par paire) — le modèle refuse de trancher.
        a = _bb_vs_co("Kh 6d")
        self.assertTrue(a.action.startswith("MARGINAL"), a.action)

    def test_l_equite_et_le_seuil_sont_declares(self) -> None:
        a = _bb_vs_co("Kh 9d")
        self.assertIsNotNone(a.equity)
        self.assertTrue(0.0 < a.equity < 1.0)
        self.assertTrue(any("range d'ouverture" in r for r in a.reasons))
        self.assertTrue(any("Cote du pot exacte" in r for r in a.reasons))
        self.assertTrue(any("MDF" in r for r in a.reasons))
        self.assertTrue(any("OUVERTURE" in x for x in a.assumptions))

    def test_l_ouvreur_serre_baisse_l_equite(self) -> None:
        # La même main vaut moins contre une ouverture UTG que contre une
        # ouverture BTN — c'est la range qui porte l'information.
        vs_utg = _bb_vs_co("Ah Jd", opener="UTG")
        vs_btn = _bb_vs_co("Ah Jd", opener="BTN")
        self.assertLess(vs_utg.equity, vs_btn.equity)


class TestRoutage(unittest.TestCase):
    """La défense ne s'empare QUE des spots « face à une ouverture »."""

    def test_pot_non_ouvert_reste_a_la_chart(self) -> None:
        a = advise(Spot(hero="Ah Kd", stack=100, big_blind=1,
                        position="BTN", players=6))
        self.assertEqual(a.regime, "chart d'ouverture BTN")
        self.assertTrue(a.action.startswith("OUVRIR"))

    def test_un_limp_devant_reste_a_la_chart(self) -> None:
        # BTN, 1 bb à payer = un limp, pas une ouverture.
        a = advise(Spot(hero="Ah Kd", position="BTN", pot=2.5, bet=1.0,
                        stack=100, big_blind=1, players=4))
        self.assertEqual(a.regime, "chart d'ouverture BTN")

    def test_la_bb_face_a_une_mini_ouverture_est_une_defense(self) -> None:
        # Pour la BB, 1 bb à payer N'est PAS un limp (un limp = 0 à payer).
        a = advise(Spot(hero="Ah Kd", position="BB", pot=3.5, bet=1.0,
                        stack=98, big_blind=1, players=2))
        self.assertIn("défense face à une ouverture", a.regime)

    def test_utg_face_a_une_mise_retombe_sur_la_chart(self) -> None:
        # Rien ne peut avoir OUVERT avant UTG : la mise subie est une
        # surrelance, hors du domaine du modèle — comportement inchangé.
        a = advise(Spot(hero="Ah Kd", position="UTG", pot=4.0, bet=2.5,
                        stack=97.5, big_blind=1, players=6))
        self.assertEqual(a.regime, "chart d'ouverture UTG")

    def test_la_bb_sans_chart_est_maintenant_conseillee(self) -> None:
        # AVANT : position BB inconnue des charts → « — » (aucun modèle).
        # C'était le refus le plus nombreux du banc Pluribus.
        a = _bb_vs_co("Kh 9d")
        self.assertNotEqual(a.action, "—")

    def test_le_tapis_court_garde_la_priorite_nash(self) -> None:
        a = advise(Spot(hero="Ah Ad", position="BB", pot=4.0, bet=1.5,
                        stack=10, big_blind=1, players=2))
        self.assertIn("Nash push/fold", a.regime)

    def test_le_postflop_est_inchange(self) -> None:
        a = advise(Spot(hero="Ah Qd", board="Qs 7d 2c", pot=50, bet=25,
                        stack=300, big_blind=10))
        self.assertIn("équité exacte au flop", a.regime)

    def test_le_regime_porte_le_motif_chart_pour_le_banc(self) -> None:
        # Le banc classe les moteurs par motif : « chart d'ouverture » range
        # ce régime sous « ranges d'ouverture (GTO_PRESETS) » — ce qui est
        # exact : c'est bien la brique utilisée, côté vilain cette fois.
        a = _bb_vs_co("Kh 9d")
        self.assertIn("chart d'ouverture", a.regime)


class TestAxeBinaireDuBanc(unittest.TestCase):
    """Chaque libellé du modèle tombe dans la bonne case du banc Pluribus.

    Un libellé non reconnu serait compté « ? » et sortirait du taux d'accord
    sans que personne ne le voie — c'est le défaut que
    ``test_banc_corpus_pluribus`` épingle pour les libellés historiques.
    """

    def test_les_libelles_sont_reconnus_par_le_banc(self) -> None:
        import banc_corpus_pluribus as banc
        from pfs.analysis.spot_advisor import Advice

        attendus = {
            "CALL (défense)": "CONTINUER",
            "CALL ou 3-BET (valeur)": "CONTINUER",
            "MARGINAL — au seuil de défense": "INDECIS",
            "MIXTE — 3-bet bluff 96 % du temps, sinon FOLD": "MIXTE",
            "FOLD": "PASSER",
        }
        for action, intention in attendus.items():
            a = Advice(action=action, confidence="indicatif", regime="test")
            self.assertEqual(banc.intention_conseiller(a), intention, action)


if __name__ == "__main__":
    unittest.main()
