"""Tests des drills ciblés — les fuites mesurées converties en exercices.

Goldens ancrés sur l'équilibre jam/fold heads-up, mesurés une fois avec le
solveur du projet (chipEV). Sans ante, le conseiller et la revue font le même
appel et rendent donc la même valeur :

    AA  @ 10 bb : EV(jam) − EV(fold) = +3.4911 bb   → jam pur
    72o @ 10 bb :                      −0.4882 bb   → fold pur
    Q7o @ 11 bb :                      −0.0366 bb   → fold pur
    Q7o @ 12 bb :                      −0.0776 bb   → fold pur
    QJo @ 12 bb :                      +0.5885 bb   → jam pur

Avec ante, le solve de la revue s'écarte de celui du conseiller — c'est la
raison d'être des tests ``TestAnte`` :

    72o @ 10 bb, ante 0.20 bb : −0.5612 bb   (14,96 % au-delà du sans-ante)
    QJo @ 12 bb, ante 0.10 bb : +0.7554 bb   (28,36 %)
    Q7o @ 12 bb, ante 0.15 bb : +0.0116 bb   (le SIGNE bascule)

Ces valeurs servent aussi de repères de coût : 20 jams Q7o à 11 bb coûtent
0,732 bb, donc plus qu'un unique jam 72o à 10 bb (0,488 bb). C'est exactement
l'arbitrage gravité/fréquence que le tri doit produire.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pfs.analysis.pushfold_review import MAX_EFF_BB, PushFoldReview, ShoveSpot
from pfs.analysis.session_review import review_hands
from pfs.analysis.spot_advisor import MAX_PUSHFOLD_BB, Spot, advise
from pfs.data.hand_history import ParsedHand, parse_ipoker, player_key
from pfs.train.drill import Grade
from pfs.train.leak_drills import (
    SessionDrillsCibles,
    SpotDrill,
    _cartes_representatives,
    _contexte_du_pot,
    _memes_verdicts,
    generer_drills,
    generer_drills_dossier,
    generer_drills_mains,
)

HERO = player_key("H")

# EV Nash mesurées (bb, chipEV) — voir docstring du module.
EV_AA_10 = 3.4911
EV_72O_10 = -0.4882
EV_Q7O_11 = -0.0366
EV_Q7O_12 = -0.0776
EV_QJO_12 = 0.5885
EV_72O_10_ANTE20 = -0.5612
EV_QJO_12_ANTE10 = 0.7554
EV_Q7O_12_ANTE15 = 0.0116

ECART_ANTE = "verdict retourné par l'ante (non robuste)"
ECART_INCERTAIN = "réponse non certaine (hors périmètre du solveur)"
ECART_LIMP_NUL = "limp sans coût mesurable"
ECART_MULTIWAY = "multiway (hors solveur heads-up)"
ECART_PROFOND = "tapis > 25 bb (hors push/fold)"
ECART_RELANCE = "relance non-jam (hors push/fold)"


def _main_xml(
    cartes: str,
    action: str,
    *,
    stack: int = 100,
    bb: int = 10,
    ante: int = 0,
    gamecode: str = "1",
) -> str:
    """Une main heads-up où H est bouton/SB (10 bb effectifs par défaut).

    Parameters
    ----------
    cartes : str
        Cartes du héros au format iPoker (« SA HA »).
    action : str
        « jam » (all-in), « fold », « limp » (suivi de la SB) ou « raise ».
    ante : int
        Ante par joueur, en jetons (0 = pas d'ante).
    """
    demi = bb // 2
    if action == "jam":
        act = f'<action no="5" player="H" sum="{stack}" type="23"/>'
        h_bet, v_bet, h_win, v_win = stack, bb + ante, 0, stack + bb + ante
    elif action == "limp":
        act = f'<action no="5" player="H" sum="{demi}" type="3"/>'
        h_bet = v_bet = bb + ante
        h_win, v_win = 0, 2 * (bb + ante)
    elif action == "raise":
        act = f'<action no="5" player="H" sum="{3 * bb}" type="5"/>'
        h_bet, v_bet = 3 * bb + ante, bb + ante
        h_win, v_win = 0, 4 * bb + 2 * ante
    else:
        act = '<action no="5" player="H" sum="0" type="0"/>'
        h_bet, v_bet = demi + ante, bb + ante
        h_win, v_win = 0, bb + demi + 2 * ante
    antes = ""
    if ante:
        antes = (f'<action no="1" player="H" sum="{ante}" type="15"/>'
                 f'<action no="2" player="V" sum="{ante}" type="15"/>')
    return f"""<?xml version="1.0"?>
<session sessioncode="1">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>1</tournamentcode></general>
 <game gamecode="{gamecode}">
  <general><smallblind>{demi}</smallblind><bigblind>{bb}</bigblind>
   <players>
    <player bet="{h_bet}" chips="{stack}" dealer="1" name="H" seat="1" win="{h_win}"/>
    <player bet="{v_bet}" chips="{stack}" dealer="0" name="V" seat="2" win="{v_win}"/>
   </players></general>
  <round no="0">
   {antes}
   <action no="3" player="H" sum="{demi}" type="1"/>
   <action no="4" player="V" sum="{bb}" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">{cartes}</cards>
   <cards player="V" type="Pocket">X X</cards>
   {act}</round>
 </game>
</session>"""


def _table_six_xml(cartes: str = "SA HA", bb: int = 40, ante: int = 3,
                   stack: int = 400) -> str:
    """Une table à six où H est au bouton, tout le monde a payé l'ante.

    Le héros parle en premier dans le modèle de la revue : personne ne peut
    s'être couché devant lui, le spot est donc multiway.
    """
    demi = bb // 2
    sieges = "".join(
        f'<player bet="{ante}" chips="{stack}" dealer="{1 if i == 1 else 0}" '
        f'name="P{i}" seat="{i}" win="0"/>'
        for i in range(2, 7)
    )
    antes = "".join(
        f'<action no="{i}" player="{"H" if i == 1 else f"P{i}"}" '
        f'sum="{ante}" type="15"/>'
        for i in range(1, 7)
    )
    return f"""<?xml version="1.0"?>
<session sessioncode="1">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>1</tournamentcode></general>
 <game gamecode="6max">
  <general><smallblind>{demi}</smallblind><bigblind>{bb}</bigblind>
   <players>
    <player bet="{ante}" chips="{stack}" dealer="1" name="H" seat="1" win="0"/>
    {sieges}
   </players></general>
  <round no="0">
   {antes}
   <action no="7" player="P2" sum="{demi}" type="1"/>
   <action no="8" player="P3" sum="{bb}" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">{cartes}</cards>
   <action no="9" player="H" sum="0" type="0"/></round>
 </game>
</session>"""


def _mains(*specs: tuple) -> list[ParsedHand]:
    """Parse une suite de mains ``(cartes, action, stack, bb[, ante])``."""
    out: list[ParsedHand] = []
    for i, spec in enumerate(specs):
        cartes, action, stack, bb, *reste = spec
        out.extend(
            parse_ipoker(_main_xml(cartes, action, stack=stack, bb=bb,
                                   ante=reste[0] if reste else 0,
                                   gamecode=str(i + 1)))
        )
    return out


class TestCartesRepresentatives(unittest.TestCase):
    """Le représentant canonique doit désigner le bon groupe, sans ambiguïté."""

    def test_paire_suited_offsuit(self) -> None:
        self.assertEqual(_cartes_representatives("AA"), "As Ah")
        self.assertEqual(_cartes_representatives("A5s"), "As 5s")
        self.assertEqual(_cartes_representatives("A5o"), "As 5h")

    def test_le_groupe_est_bien_retrouve(self) -> None:
        for nom in ("AA", "72o", "A5s", "T9s", "K2o"):
            conseil = advise(Spot(hero=_cartes_representatives(nom), stack=10,
                                  big_blind=1, position="BTN", players=2))
            self.assertEqual(conseil.hand, nom)

    def test_groupe_invalide_rejete(self) -> None:
        for mauvais in ("", "A", "AXo", "A5x", "A5so"):
            with self.assertRaises(ValueError):
                _cartes_representatives(mauvais)


class TestMemesVerdicts(unittest.TestCase):
    """Le garde-fou ante : deux solves qui divergent ⇒ pas de drill."""

    def test_meme_signe_accepte(self) -> None:
        self.assertTrue(_memes_verdicts(+0.4, +0.1))
        self.assertTrue(_memes_verdicts(-0.4, -0.1))

    def test_signes_opposes_refuses(self) -> None:
        self.assertFalse(_memes_verdicts(+0.02, -0.02))
        self.assertFalse(_memes_verdicts(-0.02, +0.02))

    def test_seul_le_signe_est_controle_pas_la_magnitude(self) -> None:
        # Les deux EV mesurées de 72o à 10 bb avec 0,20 bb d'ante : même
        # décision, 14,96 % d'écart. Le garde-fou l'accepte — c'est pourquoi
        # le drill n'affiche jamais celle du conseiller.
        self.assertTrue(_memes_verdicts(EV_72O_10, EV_72O_10_ANTE20))
        ecart = abs(EV_72O_10_ANTE20 - EV_72O_10) / abs(EV_72O_10)
        self.assertAlmostEqual(ecart, 0.1496, places=3)


class TestGenerationDepuisMains(unittest.TestCase):
    """Bout en bout, contre le vrai solveur."""

    def test_jam_trop_lache_donne_un_drill_fold(self) -> None:
        rap = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)
        self.assertEqual(len(rap), 1)
        d = rap[0]
        self.assertEqual(d.main, "72o")
        self.assertEqual(d.bonne_reponse, "FOLD")
        self.assertEqual(d.action_jouee, "jam")
        self.assertEqual(d.fuite, "jam trop lâche")
        self.assertEqual(d.options, ("JAM", "FOLD"))
        self.assertAlmostEqual(d.ev_jam_moins_fold, EV_72O_10, places=3)
        self.assertAlmostEqual(d.cout_total_bb, -EV_72O_10, places=3)
        self.assertEqual(d.occurrences, 1)
        self.assertTrue(d.cartes_reelles)
        self.assertEqual(d.cartes, "7s 2d")

    def test_fold_trop_serre_donne_un_drill_jam(self) -> None:
        rap = generer_drills_mains(_mains(("SA HA", "fold", 100, 10)), hero=HERO)
        d = rap[0]
        self.assertEqual(d.main, "AA")
        self.assertEqual(d.bonne_reponse, "JAM")
        self.assertEqual(d.fuite, "fold trop serré")
        self.assertAlmostEqual(d.ev_jam_moins_fold, EV_AA_10, places=3)
        self.assertAlmostEqual(d.cout_total_bb, EV_AA_10, places=3)

    def test_decisions_correctes_ne_produisent_aucun_drill(self) -> None:
        rap = generer_drills_mains(
            _mains(("SA HA", "jam", 100, 10), ("S7 D2", "fold", 100, 10)),
            hero=HERO,
        )
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.n_spots_corrects, 2)
        self.assertEqual(rap.n_erreurs, 0)
        self.assertEqual(rap.cout_cible_bb, 0.0)

    def test_limp_en_tapis_court_est_cible(self) -> None:
        # QJo à 12 bb : le jam est rentable (+0,59 bb), donc limper le coûte
        rap = generer_drills_mains(_mains(("SQ HJ", "limp", 120, 10)), hero=HERO)
        self.assertEqual(len(rap), 1)
        d = rap[0]
        self.assertEqual(d.main, "QJo")
        self.assertEqual(d.action_jouee, "limp")
        self.assertEqual(d.fuite, "limp")
        self.assertEqual(d.bonne_reponse, "JAM")
        self.assertAlmostEqual(d.cout_total_bb, EV_QJO_12, places=3)

    def test_pot_et_mise_du_spot(self) -> None:
        rap = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)
        d = rap[0]
        self.assertAlmostEqual(d.pot_bb, 1.5)     # SB + BB, aucune ante ici
        self.assertAlmostEqual(d.mise_bb, 0.5)    # complément de la SB
        self.assertAlmostEqual(d.ante_bb, 0.0)
        self.assertAlmostEqual(d.tapis_bb, 10.0)


class TestPriorisation(unittest.TestCase):
    """Fréquence × gravité : le tri se fait sur le coût CUMULÉ."""

    @classmethod
    def setUpClass(cls) -> None:
        # 20 jams Q7o à 11 bb (0,0366 bb l'unité) contre 1 jam 72o à 10 bb
        # (0,4882 bb) : le total du premier (0,732 bb) doit passer devant.
        specs = [("SQ H7", "jam", 110, 10)] * 20 + [("S7 D2", "jam", 100, 10)]
        cls.rap = generer_drills_mains(_mains(*specs), hero=HERO)

    def test_les_repetitions_sont_regroupees(self) -> None:
        self.assertEqual(len(self.rap), 2)
        repete = next(d for d in self.rap if d.main == "Q7o")
        self.assertEqual(repete.occurrences, 20)
        self.assertEqual(len(repete.mains_ids), 20)
        self.assertAlmostEqual(repete.cout_unitaire_bb, -EV_Q7O_11, places=3)
        self.assertAlmostEqual(repete.cout_total_bb, 20 * -EV_Q7O_11, places=2)

    def test_une_erreur_frequente_passe_devant_une_erreur_chere(self) -> None:
        self.assertEqual(self.rap[0].main, "Q7o")
        self.assertEqual(self.rap[1].main, "72o")
        # …alors qu'à l'unité, l'erreur isolée coûte 13 fois plus cher
        self.assertLess(self.rap[0].cout_unitaire_bb,
                        self.rap[1].cout_unitaire_bb)
        self.assertGreater(self.rap[0].priorite, self.rap[1].priorite)

    def test_max_drills_garde_les_plus_couteux(self) -> None:
        rap = generer_drills_mains(
            _mains(*([("SQ H7", "jam", 110, 10)] * 20
                     + [("S7 D2", "jam", 100, 10)])),
            hero=HERO, max_drills=1,
        )
        self.assertEqual(len(rap), 1)
        self.assertEqual(rap[0].main, "Q7o")

    def test_max_drills_ne_tronque_pas_la_comptabilite(self) -> None:
        # La coupe porte sur l'affichage : 21 erreurs restent 21 erreurs.
        specs = [("SQ H7", "jam", 110, 10)] * 20 + [("S7 D2", "jam", 100, 10)]
        rap = generer_drills_mains(_mains(*specs), hero=HERO, max_drills=1)
        self.assertEqual(rap.n_erreurs, 21)
        self.assertEqual(rap.n_spots_juges, 21)

    def test_max_drills_invalide(self) -> None:
        with self.assertRaises(ValueError):
            generer_drills(PushFoldReview(), max_drills=0)


class TestCoutDesLimps(unittest.TestCase):
    """Le coût d'un limp n'est pas une mesure, et rien ne doit le laisser croire."""

    @classmethod
    def setUpClass(cls) -> None:
        # Un limp (QJo @ 12 bb, 0,5885 bb « affichés ») et une vraie mesure
        # (72o @ 10 bb, 0,4882 bb) dans le même corpus.
        cls.rap = generer_drills_mains(
            _mains(("SQ HJ", "limp", 120, 10), ("S7 D2", "jam", 100, 10)),
            hero=HERO,
        )
        cls.limp = next(d for d in cls.rap if d.fuite == "limp")
        cls.jam = next(d for d in cls.rap if d.fuite == "jam trop lâche")

    def test_le_limp_est_declare_non_mesure(self) -> None:
        self.assertFalse(self.limp.cout_mesure)
        self.assertTrue(self.jam.cout_mesure)

    def test_bornes_du_cout(self) -> None:
        # Mesure : les deux bornes coïncident. Limp : [0 ; coût + mise], la
        # borne haute correspondant au limp que le héros abandonne ensuite
        # sans remettre un jeton — il perd 1 bb au lieu des 0,5 bb du fold.
        self.assertEqual(self.jam.bornes_du_cout,
                         (self.jam.cout_total_bb, self.jam.cout_total_bb))
        bas, haut = self.limp.bornes_du_cout
        self.assertEqual(bas, 0.0)
        self.assertAlmostEqual(haut, EV_QJO_12 + 0.5, places=3)

    def test_bornes_du_cout_croissent_avec_les_occurrences(self) -> None:
        rap = generer_drills_mains(
            _mains(*([("SQ HJ", "limp", 120, 10)] * 3)), hero=HERO)
        d = rap[0]
        self.assertEqual(d.occurrences, 3)
        self.assertAlmostEqual(d.bornes_du_cout[1], 3 * EV_QJO_12 + 3 * 0.5,
                               places=3)

    def test_explication_dun_limp_ne_facture_pas_le_cout_comme_un_fait(self) -> None:
        e = self.limp.explication()
        self.assertIn("NON MESURÉES", e)
        self.assertIn("[0.00 ; 1.09] bb", e)      # 0,5885 + 0,5
        self.assertNotIn("bb perdues", e)

    def test_explication_dune_vraie_mesure_reste_affirmative(self) -> None:
        e = self.jam.explication()
        self.assertIn("bb perdues", e)
        self.assertNotIn("NON MESURÉES", e)

    def test_le_rapport_isole_la_part_non_mesuree(self) -> None:
        r = self.rap
        self.assertAlmostEqual(r.cout_non_mesure_bb, EV_QJO_12, places=3)
        self.assertAlmostEqual(r.cout_mesure_bb, -EV_72O_10, places=3)
        self.assertAlmostEqual(r.cout_mesure_bb + r.cout_non_mesure_bb,
                               r.cout_cible_bb, places=9)
        # 54,7 % du chiffre affiché ne sort pas du solveur
        part = r.cout_non_mesure_bb / r.cout_cible_bb
        self.assertAlmostEqual(part, 0.5466, places=3)

    def test_le_rapport_encadre_le_cout_reel(self) -> None:
        bas, haut = self.rap.bornes_du_cout
        self.assertAlmostEqual(bas, -EV_72O_10, places=3)
        self.assertAlmostEqual(haut, -EV_72O_10 + EV_QJO_12 + 0.5, places=3)
        self.assertLessEqual(bas, self.rap.cout_cible_bb)
        self.assertLessEqual(self.rap.cout_cible_bb, haut)

    def test_le_rapport_dit_que_le_total_est_une_convention(self) -> None:
        txt = self.rap.explain()
        self.assertIn("NON MESURÉ", txt)
        self.assertIn("Coût réel encadré par", txt)
        self.assertIn("convention", txt)
        self.assertIn("(non mesuré)", txt)

    def test_sans_limp_le_rapport_nannonce_aucune_convention(self) -> None:
        rap = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)
        self.assertEqual(rap.cout_non_mesure_bb, 0.0)
        self.assertNotIn("NON MESURÉ", rap.explain())
        self.assertEqual(rap.bornes_du_cout,
                         (rap.cout_cible_bb, rap.cout_cible_bb))


class TestAnte(unittest.TestCase):
    """Heads-up AVEC ante : un seul solve derrière l'EV, le coût et le pot."""

    @classmethod
    def setUpClass(cls) -> None:
        # 72o jammé à 10 bb avec 0,20 bb d'ante (bb = 10 jetons, ante = 2).
        cls.rap = generer_drills_mains(
            _mains(("S7 D2", "jam", 100, 10, 2)), hero=HERO)
        cls.d = cls.rap[0]

    def test_lante_est_bien_lue(self) -> None:
        self.assertAlmostEqual(self.d.ante_bb, 0.20)
        self.assertAlmostEqual(self.d.tapis_bb, 10.0)

    def test_ev_affichee_et_cout_viennent_du_meme_solve(self) -> None:
        d = self.d
        self.assertAlmostEqual(d.ev_jam_moins_fold, EV_72O_10_ANTE20, places=3)
        self.assertAlmostEqual(d.cout_total_bb, -EV_72O_10_ANTE20, places=3)
        # l'identité exacte est le point : plus aucun mélange de deux solves
        self.assertAlmostEqual(d.cout_unitaire_bb, -d.ev_jam_moins_fold,
                               places=9)

    def test_lev_sans_ante_est_conservee_comme_contre_epreuve(self) -> None:
        d = self.d
        self.assertAlmostEqual(d.ev_sans_ante_bb, EV_72O_10, places=3)
        ecart = abs(d.ev_jam_moins_fold - d.ev_sans_ante_bb)
        self.assertAlmostEqual(ecart / abs(d.ev_sans_ante_bb), 0.1496, places=3)

    def test_le_pot_compte_deux_antes(self) -> None:
        self.assertAlmostEqual(self.d.pot_bb, 1.5 + 2 * 0.20)
        self.assertAlmostEqual(self.d.mise_bb, 0.5)

    def test_le_regime_et_le_corrige_nomment_lante(self) -> None:
        self.assertIn("ante 0.20 bb", self.d.regime)
        e = self.d.explication()
        self.assertIn("ante 0.20 bb", e)
        self.assertIn("Contre-épreuve sans ante", e)
        self.assertIn("-0.49 bb", e)              # la valeur sans ante
        self.assertIn("-0.56 bb", e)              # celle qui a chiffré le coût

    def test_sans_ante_aucune_contre_epreuve_nest_affichee(self) -> None:
        rap = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)
        self.assertNotIn("Contre-épreuve", rap[0].explication())

    def test_un_limp_avec_ante_reste_coherent(self) -> None:
        rap = generer_drills_mains(_mains(("SQ HJ", "limp", 120, 10, 1)),
                                   hero=HERO)
        d = rap[0]
        self.assertAlmostEqual(d.ante_bb, 0.10)
        self.assertAlmostEqual(d.ev_jam_moins_fold, EV_QJO_12_ANTE10, places=3)
        self.assertAlmostEqual(d.cout_total_bb, EV_QJO_12_ANTE10, places=3)
        self.assertAlmostEqual(d.pot_bb, 1.7)
        self.assertFalse(d.cout_mesure)

    def test_lante_qui_retourne_le_verdict_ecarte_le_spot(self) -> None:
        # Q7o couché à 12 bb : le fold est correct sans ante (−0,0776 bb) et
        # devient une erreur avec 0,15 bb d'ante (+0,0116 bb). Les deux solves
        # ne désignent pas la même action → aucun drill, un écart compté.
        self.assertLess(EV_Q7O_12, 0.0)
        self.assertGreater(EV_Q7O_12_ANTE15, 0.0)
        rap = generer_drills_mains(_mains(("SQ H7", "fold", 240, 20, 3)),
                                   hero=HERO)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.n_erreurs, 0)
        self.assertEqual(rap.ecartes[ECART_ANTE], 1)


class TestContexteDuPot(unittest.TestCase):
    """Le pot affiché doit être celui du modèle qui a chiffré la réponse."""

    def test_sans_main_le_pot_est_celui_dune_structure_sans_ante(self) -> None:
        self.assertEqual(_contexte_du_pot(None), (1.5, 0.5, 0.0))

    def test_le_pot_suit_le_modele_a_deux_joueurs_pas_la_table(self) -> None:
        # Table à six, ante 4 jetons pour une bb de 40 → 0,10 bb par joueur.
        # La table a 2,10 bb au pot ; le solve chipEV de la revue n'en compte
        # que deux antes, soit 1,70 bb — c'est ce dernier qui est affiché.
        main = parse_ipoker(_table_six_xml(ante=4))[0]
        self.assertEqual(len(main.seats), 6)
        pot, mise, ante = _contexte_du_pot(main)
        self.assertAlmostEqual(ante, 0.10)
        self.assertAlmostEqual(pot, 1.70)
        self.assertAlmostEqual(mise, 0.5)
        self.assertAlmostEqual(1.5 + len(main.seats) * ante, 2.10)

    def test_lante_est_arrondie_comme_la_cle_du_solve(self) -> None:
        # 3 jetons pour une bb de 40 valent 0,075 bb, mais la revue mémoïse son
        # solve au centième de bb (round(7.5) = 8) : le pot affiché suit cette
        # granularité, sans quoi il ne serait pas celui du modèle résolu.
        main = parse_ipoker(_table_six_xml(ante=3))[0]
        pot, _, ante = _contexte_du_pot(main)
        self.assertAlmostEqual(main.ante / main.big_blind, 0.075)
        self.assertAlmostEqual(ante, 0.08)
        self.assertAlmostEqual(pot, 1.66)

    def test_le_pipeline_ne_juge_aucun_spot_a_plus_de_deux_joueurs(self) -> None:
        # Le héros au bouton parle en premier : personne ne peut s'être couché
        # devant lui, donc une table à six est toujours multiway et écartée.
        # L'écart mesuré ci-dessus est donc inatteignable par le pipeline.
        rap = generer_drills_mains(parse_ipoker(_table_six_xml(ante=4)), hero=HERO)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.n_spots_juges, 0)
        self.assertEqual(rap.ecartes[ECART_MULTIWAY], 1)


class TestHonnetete(unittest.TestCase):
    """Ce qu'on refuse de transformer en drill, et qu'on compte à part."""

    def test_tapis_profond_ecarte_sans_drill(self) -> None:
        rap = generer_drills_mains(_mains(("SA HA", "fold", 1000, 10)), hero=HERO)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.ecartes[ECART_PROFOND], 1)
        self.assertEqual(rap.n_ecartes, 1)

    def test_reponse_non_certaine_ecartee(self) -> None:
        # Spot fabriqué à 30 bb : hors du régime où le conseiller est certain
        revue = PushFoldReview(spots=[
            ShoveSpot(hand_id="x", hand="A5o", eff_bb=30.0, decision="jam",
                      nash_jam_freq=0.0, ev_jam_minus_fold=-0.5,
                      verdict="jam trop lâche", cost_bb=0.5)
        ])
        rap = generer_drills(revue)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.ecartes[ECART_INCERTAIN], 1)
        # écartée une seule fois : elle ne compte pas AUSSI comme une erreur
        self.assertEqual(rap.n_erreurs, 0)

    def test_le_garde_fou_incertain_est_hors_datteinte_du_pipeline(self) -> None:
        # Le domaine de la revue est inclus dans celui où advise() est certain :
        # le compteur ne peut donc pas bouger via generer_drills_mains. Le
        # balayage part de 1,1 bb parce que le solveur refuse un tapis effectif
        # ≤ bb + ante (à 1,0 bb pile, c'est la revue elle-même qui lève).
        self.assertGreaterEqual(MAX_PUSHFOLD_BB, MAX_EFF_BB)
        eff = 1.1
        while eff <= MAX_EFF_BB + 1e-9:
            conseil = advise(Spot(hero="As Kd", stack=eff, big_blind=1.0,
                                  position="BTN", players=2))
            self.assertEqual(conseil.confidence, "certain", f"à {eff} bb")
            eff = round(eff + 0.5, 1)
        self.assertEqual(
            advise(Spot(hero="As Kd", stack=MAX_EFF_BB, big_blind=1.0,
                        position="BTN", players=2)).confidence,
            "certain",
        )

    def test_le_rapport_marque_les_garde_fous(self) -> None:
        txt = generer_drills(PushFoldReview()).explain()
        self.assertIn("(garde-fou)", txt)
        self.assertIn("n'est pas un filtrage observé", txt)

    def test_verdict_retourne_par_lante_ecarte(self) -> None:
        # Le conseiller (sans ante) dit JAM pour AA ; une revue qui prétend
        # l'inverse signale une structure que le drill n'afficherait pas.
        revue = PushFoldReview(spots=[
            ShoveSpot(hand_id="x", hand="AA", eff_bb=10.0, decision="fold",
                      nash_jam_freq=0.0, ev_jam_minus_fold=-1.0,
                      verdict="fold trop serré", cost_bb=1.0)
        ])
        rap = generer_drills(revue)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.ecartes[ECART_ANTE], 1)
        self.assertEqual(rap.n_erreurs, 0)

    def test_limp_sans_cout_mesurable_ecarte(self) -> None:
        # 72o limpé à 10 bb : le jam était perdant, le coût du limp n'est pas
        # défini par ce modèle → aucun drill fabriqué
        rap = generer_drills_mains(_mains(("S7 D2", "limp", 100, 10)), hero=HERO)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.ecartes[ECART_LIMP_NUL], 1)

    def test_cartes_representatives_signalees(self) -> None:
        # Sans historique, les cartes affichées ne sont plus celles jouées
        revue = PushFoldReview(spots=[
            ShoveSpot(hand_id="x", hand="72o", eff_bb=10.0, decision="jam",
                      nash_jam_freq=0.0, ev_jam_minus_fold=EV_72O_10,
                      verdict="jam trop lâche", cost_bb=-EV_72O_10)
        ])
        d = generer_drills(revue)[0]
        self.assertFalse(d.cartes_reelles)
        self.assertEqual(d.cartes, "7s 2h")
        self.assertEqual(d.bonne_reponse, "FOLD")
        # main absente ⇒ ante inconnue ⇒ pot de la structure sans ante
        self.assertAlmostEqual(d.pot_bb, 1.5)
        self.assertAlmostEqual(d.ante_bb, 0.0)


class TestComptabilite(unittest.TestCase):
    """Chaque décision est comptée une fois et une seule."""

    @classmethod
    def setUpClass(cls) -> None:
        # 1 correcte, 1 erreur, 1 relance non-jam, 1 tapis profond
        cls.rap = generer_drills_mains(
            _mains(
                ("SA HA", "jam", 100, 10),
                ("S7 D2", "jam", 100, 10),
                ("SK HQ", "raise", 100, 10),
                ("SA HA", "fold", 1000, 10),
            ),
            hero=HERO,
        )

    def test_identite_comptable(self) -> None:
        r = self.rap
        self.assertEqual(r.n_spots_juges, r.n_spots_corrects + r.n_erreurs)
        self.assertEqual(r.n_decisions_vues, r.n_spots_juges + r.n_ecartes)
        self.assertEqual(r.n_decisions_vues, 4)

    def test_une_relance_nest_pas_comptee_deux_fois(self) -> None:
        r = self.rap
        self.assertEqual(r.n_spots_corrects, 1)
        self.assertEqual(r.n_erreurs, 1)
        self.assertEqual(r.n_spots_juges, 2)      # la relance n'y est pas
        self.assertEqual(r.ecartes[ECART_RELANCE], 1)
        self.assertEqual(r.ecartes[ECART_PROFOND], 1)
        self.assertEqual(r.n_ecartes, 2)

    def test_les_occurrences_couvrent_exactement_les_erreurs(self) -> None:
        self.assertEqual(sum(d.occurrences for d in self.rap), self.rap.n_erreurs)

    def test_le_rapport_affiche_le_bilan(self) -> None:
        txt = self.rap.explain()
        self.assertIn("Décisions préflop vues      : 4", txt)
        self.assertIn("(1 correctes, 1 erreurs)", txt)


class TestRapport(unittest.TestCase):
    def test_rapport_nomme_les_fuites(self) -> None:
        rap = generer_drills_mains(
            _mains(("S7 D2", "jam", 100, 10), ("SA HA", "fold", 100, 10)),
            hero=HERO,
        )
        txt = rap.explain()
        self.assertIn("DRILLS CIBLÉS", txt)
        self.assertIn("Jam trop lâche", txt)
        self.assertIn("Fold trop serré", txt)
        self.assertIn("72o", txt)
        self.assertIn("AA", txt)
        self.assertIn("Spots écartés", txt)

    def test_rapport_vide_le_dit(self) -> None:
        txt = generer_drills(PushFoldReview()).explain()
        self.assertIn("Aucune fuite chiffrable", txt)

    def test_agregation_par_famille_de_fuite(self) -> None:
        rap = generer_drills_mains(
            _mains(("S7 D2", "jam", 100, 10), ("SA HA", "fold", 100, 10)),
            hero=HERO,
        )
        fuites = dict((f, c) for f, _, c in rap.fuites)
        self.assertAlmostEqual(fuites["fold trop serré"], EV_AA_10, places=3)
        self.assertAlmostEqual(fuites["jam trop lâche"], -EV_72O_10, places=3)
        # les plus chères d'abord : AA (3,49 bb) avant 72o (0,49 bb)
        self.assertEqual(rap.fuites[0][0], "fold trop serré")

    def test_question_et_explication_en_francais(self) -> None:
        d = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)[0]
        q = d.question()
        self.assertIn("72o", q)
        self.assertIn("préflop", q)
        self.assertIn("JAM ou FOLD", q)          # les deux options, à égalité
        self.assertNotIn("Bonne réponse", q)     # l'énoncé ne souffle rien
        self.assertNotIn("EV", q)
        e = d.explication()
        self.assertIn("Bonne réponse : FOLD", e)
        self.assertIn("Fuite ciblée", e)

    def test_profil_affiche_dans_le_rapport(self) -> None:
        mains = _mains(("S7 D2", "jam", 100, 10), ("SA HA", "fold", 100, 10))
        profil = review_hands(mains).profile
        rap = generer_drills_mains(mains, hero=HERO, profil=profil)
        self.assertIs(rap.profil, profil)
        txt = rap.explain()
        self.assertIn("Profil observé", txt)
        self.assertIn("VPIP", txt)
        self.assertIn("PFR", txt)
        self.assertIn(f"{profil.vpip - profil.pfr:.1f} points", txt)

    def test_sans_profil_le_bloc_est_absent(self) -> None:
        rap = generer_drills_mains(_mains(("S7 D2", "jam", 100, 10)), hero=HERO)
        self.assertIsNone(rap.profil)
        self.assertNotIn("Profil observé", rap.explain())

    def test_indexation_et_tranche(self) -> None:
        rap = generer_drills_mains(
            _mains(("S7 D2", "jam", 100, 10), ("SA HA", "fold", 100, 10)),
            hero=HERO,
        )
        self.assertIsInstance(rap[0], SpotDrill)
        self.assertIsInstance(rap[0:1], list)
        self.assertEqual(rap[0:2], list(rap))


class TestDossier(unittest.TestCase):
    """La porte d'entrée du chantier : lire un dossier d'historiques."""

    def test_lecture_recursive_et_fichiers_perdus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            (racine / "sous").mkdir()
            (racine / "a.xml").write_text(
                _main_xml("S7 D2", "jam", gamecode="a"), encoding="utf-8")
            (racine / "sous" / "b.xml").write_text(
                _main_xml("SA HA", "fold", gamecode="b"), encoding="utf-8")
            (racine / "casse.xml").write_text("<session", encoding="utf-8")
            (racine / "notes.txt").write_text("ignoré", encoding="utf-8")
            # un dossier nommé « .xml » : rglob le remonte, la lecture échoue
            (racine / "illisible.xml").mkdir()
            rap = generer_drills_dossier(racine, avec_profil=False)
        self.assertEqual(rap.n_mains, 2)          # le sous-dossier est lu
        self.assertEqual(rap.n_fichiers_illisibles, 1)
        self.assertEqual(rap.n_fichiers_sans_main, 1)   # le XML cassé
        self.assertEqual(len(rap), 2)
        self.assertEqual({d.main for d in rap}, {"72o", "AA"})
        txt = rap.explain()
        self.assertIn("Fichiers illisibles ignorés : 1", txt)
        self.assertIn("Fichiers sans main lisible  : 1", txt)

    def test_profil_calcule_a_la_demande(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            (racine / "a.xml").write_text(
                _main_xml("S7 D2", "jam"), encoding="utf-8")
            avec = generer_drills_dossier(racine, avec_profil=True)
            sans = generer_drills_dossier(racine, avec_profil=False)
        self.assertIsNotNone(avec.profil)
        self.assertIn("Profil observé", avec.explain())
        self.assertIsNone(sans.profil)
        self.assertNotIn("Profil observé", sans.explain())

    def test_max_drills_transmis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            racine = Path(tmp)
            (racine / "a.xml").write_text(
                _main_xml("S7 D2", "jam", gamecode="a"), encoding="utf-8")
            (racine / "b.xml").write_text(
                _main_xml("SA HA", "fold", gamecode="b"), encoding="utf-8")
            rap = generer_drills_dossier(racine, avec_profil=False, max_drills=1)
        self.assertEqual(len(rap), 1)
        self.assertEqual(rap[0].main, "AA")       # la fuite la plus chère
        self.assertEqual(rap.n_erreurs, 2)

    def test_dossier_vide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rap = generer_drills_dossier(tmp, avec_profil=False)
        self.assertEqual(len(rap), 0)
        self.assertEqual(rap.n_mains, 0)
        self.assertEqual(rap.n_fichiers_illisibles, 0)
        self.assertIn("Aucune fuite chiffrable", rap.explain())


class TestSessionSM2(unittest.TestCase):
    """Le branchement sur la répétition espacée existante."""

    def setUp(self) -> None:
        specs = [("SQ H7", "jam", 110, 10)] * 20 + [("S7 D2", "jam", 100, 10)]
        self.drills = list(generer_drills_mains(_mains(*specs), hero=HERO))
        self.session = SessionDrillsCibles(self.drills)

    def test_le_plus_couteux_sort_en_premier(self) -> None:
        self.assertEqual(self.session.prochain().main, "Q7o")

    def test_une_erreur_revient_immediatement(self) -> None:
        d = self.session.prochain()
        note = self.session.repondre(d, "JAM")     # la bonne réponse est FOLD
        self.assertFalse(note.correcte)
        self.assertIs(note.grade, Grade.BLACKOUT)
        self.assertAlmostEqual(note.cout_bb, d.cout_unitaire_bb)
        self.assertEqual(self.session.prochain().cle, d.cle)

    def test_une_reussite_espace_le_spot(self) -> None:
        d = self.session.prochain()
        note = self.session.repondre(d, "fold")    # casse et espaces tolérés
        self.assertTrue(note.correcte)
        self.assertIs(note.grade, Grade.PERFECT)
        self.assertEqual(self.session.srs.card(d.cle).interval_days, 1.0)
        self.assertNotEqual(self.session.prochain().cle, d.cle)

    def test_reponse_lente_est_degradee(self) -> None:
        d = self.session.prochain()
        note = self.session.repondre(d, "FOLD", secondes=12.0)
        self.assertTrue(note.correcte)
        self.assertIs(note.grade, Grade.GOOD)

    def test_epuisement_puis_reprise_apres_le_delai(self) -> None:
        for d in self.drills:
            self.session.repondre(d, d.bonne_reponse)
        self.assertIsNone(self.session.prochain())
        self.session.srs.advance(2.0)
        self.assertIsNotNone(self.session.prochain())

    def test_option_inconnue_rejetee(self) -> None:
        with self.assertRaises(ValueError):
            self.session.repondre(self.drills[0], "CALL")

    def test_score(self) -> None:
        for d in self.drills:
            self.session.repondre(d, d.bonne_reponse)
        s = self.session.score()
        self.assertEqual(s["n"], float(len(self.drills)))
        self.assertEqual(s["exactitude"], 1.0)
        self.assertEqual(s["bb_manquees"], 0.0)
        self.assertAlmostEqual(s["bb_ciblees"],
                               sum(d.priorite for d in self.drills))

    def test_score_avant_toute_reponse(self) -> None:
        s = self.session.score()
        self.assertEqual(s["n"], 0.0)
        self.assertGreater(s["bb_ciblees"], 0.0)

    def test_score_a_toujours_les_memes_cles(self) -> None:
        attendues = {"n", "exactitude", "bb_manquees", "bb_ciblees", "maitrise"}
        vierge = self.session.score()
        self.assertEqual(set(vierge), attendues)
        self.assertEqual(vierge["exactitude"], 0.0)
        self.assertEqual(vierge["bb_manquees"], 0.0)
        self.assertEqual(vierge["maitrise"], 0.0)
        self.session.repondre(self.drills[0], self.drills[0].bonne_reponse)
        self.assertEqual(set(self.session.score()), attendues)

    def test_session_vide_refusee(self) -> None:
        with self.assertRaises(ValueError):
            SessionDrillsCibles([])

    def test_cles_dupliquees_refusees(self) -> None:
        with self.assertRaises(ValueError):
            SessionDrillsCibles([self.drills[0], self.drills[0]])


if __name__ == "__main__":
    unittest.main()
