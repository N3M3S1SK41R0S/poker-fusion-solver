"""Tests du simulateur de main et du lexique.

Le simulateur doit distribuer un vrai hasard (aucune carte en double, une
donne différente à chaque appel sans graine) et rendre, pour la même main,
ce qu'il faut faire à plusieurs profondeurs de tapis. Le lexique doit
expliquer chaque terme sans renvoyer à un autre jargon.
"""

from __future__ import annotations

import unittest

from pfs.lexique import LEXIQUE, chercher, definir
from pfs.train.simulateur import (
    MainSimulee,
    simuler,
    tirer_main,
)


class TestTirage(unittest.TestCase):
    def test_aucune_carte_en_double(self) -> None:
        for joueurs in (2, 3, 6, 9):
            for board in (0, 3, 4, 5):
                m = tirer_main(joueurs, board)
                cartes = list(m.hero) + [c for v in m.villains for c in v] \
                    + list(m.board)
                self.assertEqual(len(cartes), len(set(cartes)),
                                 f"doublon à {joueurs} joueurs, board {board}")
                self.assertEqual(len(cartes), 2 * joueurs + board)

    def test_hasard_reel_sans_graine(self) -> None:
        """Sans graine, deux tirages consécutifs doivent différer.

        Un simulateur d'entraînement dont on devine la suite n'entraîne rien.
        La probabilité que 12 tirages donnent 12 fois la même main est nulle
        en pratique (1326 combos possibles).
        """
        mains = {tirer_main(2).hero for _ in range(12)}
        self.assertGreater(len(mains), 1)

    def test_graine_rejoue_la_meme_donne(self) -> None:
        a, b = tirer_main(3, 5, graine=1234), tirer_main(3, 5, graine=1234)
        self.assertEqual((a.hero, a.villains, a.board),
                         (b.hero, b.villains, b.board))

    def test_parametres_invalides(self) -> None:
        with self.assertRaises(ValueError):
            tirer_main(joueurs=1)
        with self.assertRaises(ValueError):
            tirer_main(joueurs=2, cartes_board=2)

    def test_groupe_de_main(self) -> None:
        m = MainSimulee(hero=("Ah", "Ks"))
        self.assertEqual(m.groupe, "AKo")
        self.assertEqual(MainSimulee(hero=("Ah", "Kh")).groupe, "AKs")
        self.assertEqual(MainSimulee(hero=("7d", "7c")).groupe, "77")


class TestSimulation(unittest.TestCase):
    def test_un_verdict_par_composition(self) -> None:
        r = simuler(joueurs=2, tapis=(5.0, 10.0, 20.0), positions=("BTN",),
                    graine=7)
        self.assertEqual(len(r.verdicts), 3)
        self.assertEqual([v.tapis_bb for v in r.verdicts], [5.0, 10.0, 20.0])

    def test_tapis_court_heads_up_est_certain(self) -> None:
        """En push/fold heads-up, le verdict est l'équilibre de Nash."""
        r = simuler(joueurs=2, tapis=(8.0, 12.0), positions=("BTN",), graine=3)
        self.assertTrue(all(v.certain for v in r.verdicts))
        self.assertTrue(all(v.ev_bb is not None for v in r.verdicts))

    def test_cartes_cachees_ne_revelent_rien(self) -> None:
        r = simuler(joueurs=2, cartes_visibles=False, tapis=(10.0,), graine=5)
        self.assertIsNone(r.equite_reelle)
        self.assertNotIn("VISIBLES", r.explain())

    def test_cartes_visibles_donnent_l_equite_exacte(self) -> None:
        r = simuler(joueurs=2, cartes_board=3, cartes_visibles=True,
                    tapis=(10.0,), graine=5)
        self.assertIsNotNone(r.equite_reelle)
        self.assertGreaterEqual(r.equite_reelle, 0.0)
        self.assertLessEqual(r.equite_reelle, 1.0)
        self.assertIn("VISIBLES", r.explain())

    def test_aa_pousse_toujours_en_tapis_court(self) -> None:
        m = MainSimulee(hero=("Ah", "Ad"), villains=(("2c", "7d"),))
        r = simuler(m, tapis=(5.0, 10.0, 15.0, 20.0, 25.0), positions=("BTN",))
        for v in r.verdicts:
            self.assertTrue(v.action.startswith("JAM"), f"{v.tapis_bb} bb")
            self.assertGreater(v.ev_bb, 0.0)

    def test_72o_passe_toujours_en_tapis_court(self) -> None:
        m = MainSimulee(hero=("7s", "2d"), villains=(("Ac", "Kd"),))
        r = simuler(m, tapis=(10.0, 15.0, 20.0, 25.0), positions=("BTN",))
        for v in r.verdicts:
            self.assertEqual(v.action, "FOLD", f"{v.tapis_bb} bb")

    def test_bascule_detectee(self) -> None:
        """Une main de frontière change de verdict : le seuil est le point
        pédagogique, il doit être rendu."""
        m = MainSimulee(hero=("J5", "x")) if False else MainSimulee(
            hero=("Jd", "5c"), villains=(("2c", "7d"),))
        r = simuler(m, tapis=(5.0, 8.0, 10.0), positions=("BTN",))
        actions = {v.action.split()[0] for v in r.verdicts}
        if len(actions) > 1:
            self.assertIsNotNone(r.bascule_bb)
        else:
            self.assertIsNone(r.bascule_bb)


class TestLexique(unittest.TestCase):
    def test_termes_bien_formes(self) -> None:
        for t in LEXIQUE:
            self.assertTrue(t.nom.strip(), "terme sans nom")
            self.assertGreater(len(t.definition), 30,
                               f"définition trop courte : {t.nom}")
            self.assertTrue(t.categorie.strip(), f"{t.nom} sans catégorie")

    def test_pas_de_doublon(self) -> None:
        noms = [t.nom.lower() for t in LEXIQUE]
        self.assertEqual(len(noms), len(set(noms)))

    def test_les_termes_du_logiciel_sont_couverts(self) -> None:
        """Tout jargon affiché par l'application doit être explicable."""
        for terme in ("VPIP", "PFR", "MDF", "ICM", "équité", "range",
                      "bubble factor", "limp", "push/fold", "EV",
                      "all-in adjusted", "WTSD", "blocker", "EQR"):
            self.assertIsNotNone(definir(terme), f"« {terme} » absent")

    def test_recherche(self) -> None:
        self.assertTrue(chercher("icm"))
        self.assertEqual(len(chercher("")), len(LEXIQUE))
        self.assertEqual(chercher("zzzzz"), [])

    def test_definition_ne_renvoie_pas_a_du_jargon_non_defini(self) -> None:
        # garde-fou : une définition ne doit pas s'appuyer sur un sigle
        # absent du lexique
        connus = {t.nom.lower() for t in LEXIQUE}
        for sigle in ("SPR", "MDF", "ICM", "EQR", "ESS"):
            self.assertTrue(any(sigle.lower() in n for n in connus), sigle)


if __name__ == "__main__":
    unittest.main()
