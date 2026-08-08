"""Tests de la revue de session — équité des tapis + profil statistique.

Golden : AA contre KK all-in préflop. L'équité préflop de AA vs KK est
≈ 81,5 % (valeur classique), calculée ici au moment du tapis (0 carte de
board), indépendamment du déroulé — c'est toute la finesse de la métrique
« all-in adjusted » : on mesure l'équité entrée, pas le résultat.
"""

from __future__ import annotations

import unittest

from pfs.analysis import review_hands
from pfs.data.hand_history import Street, parse_ipoker, player_key

# AA (héros) vs KK (vilain), tapis préflop, board neutre → AA gagne 2000.
AA_VS_KK = """<?xml version="1.0"?>
<session sessioncode="1">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>1</tournamentcode></general>
 <game gamecode="1">
  <general><smallblind>10</smallblind><bigblind>20</bigblind>
   <players>
    <player bet="1000" chips="1000" dealer="1" name="H" seat="1" win="2000"/>
    <player bet="1000" chips="1000" dealer="0" name="V" seat="2" win="0"/>
   </players></general>
  <round no="0">
   <action no="1" player="H" sum="10" type="1"/>
   <action no="2" player="V" sum="20" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">SA HA</cards>
   <cards player="V" type="Pocket">SK HK</cards>
   <action no="3" player="H" sum="1000" type="23"/>
   <action no="4" player="V" sum="980" type="7"/></round>
  <round no="2"><cards type="Flop">C2 D7 S9</cards></round>
  <round no="3"><cards type="Turn">H3</cards></round>
  <round no="4"><cards type="River">D5</cards></round>
 </game>
</session>"""

# Héros couche préflop (BB) après une relance : pas VPIP, pas de tapis.
HERO_FOLDS = """<?xml version="1.0"?>
<session sessioncode="2">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>1</tournamentcode></general>
 <game gamecode="2">
  <general><smallblind>10</smallblind><bigblind>20</bigblind>
   <players>
    <player bet="20" chips="1000" dealer="0" name="H" seat="1" win="0"/>
    <player bet="60" chips="1000" dealer="1" name="V" seat="2" win="30"/>
   </players></general>
  <round no="0">
   <action no="1" player="V" sum="10" type="1"/>
   <action no="2" player="H" sum="20" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">C7 D2</cards>
   <cards player="V" type="Pocket">X X</cards>
   <action no="3" player="V" sum="60" type="23"/>
   <action no="4" player="H" sum="0" type="0"/></round>
 </game>
</session>"""


class TestAllInEquity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hands = parse_ipoker(AA_VS_KK)
        cls.report = review_hands(cls.hands, hero=player_key("H"))

    def test_one_measured_spot(self) -> None:
        self.assertEqual(self.report.n_allin_total, 1)
        self.assertEqual(self.report.n_allin_measured, 1)
        self.assertEqual(self.report.n_allin_skipped, 0)

    def test_spot_is_preflop(self) -> None:
        spot = self.report.allin_spots[0]
        self.assertIs(spot.street, Street.PREFLOP)
        self.assertEqual(spot.board, ())      # équité entrée = préflop, 0 carte
        self.assertEqual(spot.hero_cards, ("As", "Ah"))
        self.assertEqual(spot.villain_cards, ("Ks", "Kh"))

    def test_equity_aa_vs_kk(self) -> None:
        # AA vs KK préflop ≈ 0.815 ; Monte-Carlo 20 000, marge large
        eq = self.report.allin_spots[0].equity
        self.assertGreater(eq, 0.79)
        self.assertLess(eq, 0.84)

    def test_pot_and_expectation(self) -> None:
        spot = self.report.allin_spots[0]
        self.assertEqual(spot.pot, 2000.0)
        self.assertEqual(spot.realized, 2000.0)
        # espéré ≈ 0.815 × 2000 ≈ 1630 ; réalisé 2000 → chance > 0
        self.assertAlmostEqual(spot.expected, spot.equity * 2000.0, places=6)
        self.assertGreater(spot.luck_bb, 0.0)   # a gagné plus que l'équité


class TestProfileStats(unittest.TestCase):
    def test_fold_hand_not_vpip(self) -> None:
        rep = review_hands(parse_ipoker(HERO_FOLDS), hero=player_key("H"))
        self.assertEqual(rep.profile.n_hands, 1)
        self.assertEqual(rep.profile.vpip_yes, 0)      # a couché la BB
        self.assertEqual(rep.profile.vpip_opp, 1)
        self.assertEqual(rep.n_allin_total, 0)         # pas de tapis
        # net : a perdu sa BB (20) → -1 bb
        self.assertAlmostEqual(rep.profile.net_bb, -1.0)

    def test_aggregate_two_hands(self) -> None:
        hands = parse_ipoker(AA_VS_KK) + parse_ipoker(HERO_FOLDS)
        rep = review_hands(hands, hero=player_key("H"))
        self.assertEqual(rep.profile.n_hands, 2)
        self.assertEqual(rep.n_allin_measured, 1)
        # AA : VPIP+PFR oui ; fold : VPIP non → 1/2 VPIP, 1/2 PFR
        self.assertEqual(rep.profile.vpip_yes, 1)
        self.assertEqual(rep.profile.pfr_yes, 1)


if __name__ == "__main__":
    unittest.main()
