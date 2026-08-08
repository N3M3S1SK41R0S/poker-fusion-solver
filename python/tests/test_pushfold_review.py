"""Tests de la revue shove/fold — décisions préflop face au Nash jam/fold.

Goldens ancrés sur des faits d'équilibre non discutables à 10 bb HU :
AA est un jam pur, 72o un fold pur. Le coût d'un écart est l'EV sacrifiée,
lue directement dans ``ev_jam_par_groupe`` du solveur (bb, chipEV).
"""

from __future__ import annotations

import unittest

from pfs.analysis.pushfold_review import _judge, review_pushfold_hands
from pfs.data.hand_history import parse_ipoker, player_key


def _hand_xml(hero_cards: str, hero_action: str, *, stack: int = 100,
              bb: int = 10, gamecode: str = "1") -> str:
    """Une main HU où H est bouton/SB (10 bb effectifs par défaut).

    hero_action : "jam" (all-in type 23) ou "fold" (type 0).
    """
    if hero_action == "jam":
        act = f'<action no="3" player="H" sum="{stack}" type="23"/>'
        h_bet, v_bet, h_win, v_win = stack, bb, 0, stack + bb
    else:
        act = '<action no="3" player="H" sum="0" type="0"/>'
        h_bet, v_bet, h_win, v_win = bb // 2, bb, 0, bb + bb // 2
    return f"""<?xml version="1.0"?>
<session sessioncode="1">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>1</tournamentcode></general>
 <game gamecode="{gamecode}">
  <general><smallblind>{bb // 2}</smallblind><bigblind>{bb}</bigblind>
   <players>
    <player bet="{h_bet}" chips="{stack}" dealer="1" name="H" seat="1" win="{h_win}"/>
    <player bet="{v_bet}" chips="{stack}" dealer="0" name="V" seat="2" win="{v_win}"/>
   </players></general>
  <round no="0">
   <action no="1" player="H" sum="{bb // 2}" type="1"/>
   <action no="2" player="V" sum="{bb}" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">{hero_cards}</cards>
   <cards player="V" type="Pocket">X X</cards>
   {act}</round>
 </game>
</session>"""


class TestJudge(unittest.TestCase):
    """Le jugement pur, indépendant du solveur."""

    def test_jam_correct_when_ev_positive(self) -> None:
        self.assertEqual(_judge("jam", 1.0, +0.8), ("ok", 0.0))

    def test_jam_loose_costs_the_ev_lost(self) -> None:
        verdict, cost = _judge("jam", 0.0, -0.35)
        self.assertEqual(verdict, "jam trop lâche")
        self.assertAlmostEqual(cost, 0.35)

    def test_fold_correct_when_ev_negative(self) -> None:
        self.assertEqual(_judge("fold", 0.0, -0.5), ("ok", 0.0))

    def test_fold_too_tight_costs_the_ev_left(self) -> None:
        verdict, cost = _judge("fold", 1.0, +0.42)
        self.assertEqual(verdict, "fold trop serré")
        self.assertAlmostEqual(cost, 0.42)

    def test_limp_is_flagged(self) -> None:
        verdict, cost = _judge("limp", 1.0, +0.6)
        self.assertEqual(verdict, "limp")
        self.assertAlmostEqual(cost, 0.6)


class TestShoveReview(unittest.TestCase):
    """Bout en bout, contre le vrai solveur (matrice d'équité en cache)."""

    HERO = player_key("H")

    def test_aa_jam_is_correct(self) -> None:
        hands = parse_ipoker(_hand_xml("SA HA", "jam"))
        rev = review_pushfold_hands(hands, hero=self.HERO)
        self.assertEqual(len(rev.spots), 1)
        spot = rev.spots[0]
        self.assertEqual(spot.hand, "AA")
        self.assertEqual(spot.decision, "jam")
        self.assertGreater(spot.ev_jam_minus_fold, 0.0)  # AA : jam évident
        self.assertEqual(spot.verdict, "ok")
        self.assertEqual(spot.cost_bb, 0.0)

    def test_72o_jam_is_a_leak(self) -> None:
        # 7♠2♦ à 10 bb : jam perdant à l'équilibre → écart chiffré
        hands = parse_ipoker(_hand_xml("S7 D2", "jam"))
        rev = review_pushfold_hands(hands, hero=self.HERO)
        spot = rev.spots[0]
        self.assertEqual(spot.hand, "72o")
        self.assertLess(spot.ev_jam_minus_fold, 0.0)
        self.assertEqual(spot.verdict, "jam trop lâche")
        self.assertGreater(spot.cost_bb, 0.0)
        self.assertAlmostEqual(rev.total_cost_bb, spot.cost_bb)
        self.assertEqual(rev.n_mistakes, 1)

    def test_aa_fold_is_a_leak(self) -> None:
        hands = parse_ipoker(_hand_xml("SA HA", "fold"))
        rev = review_pushfold_hands(hands, hero=self.HERO)
        spot = rev.spots[0]
        self.assertEqual(spot.decision, "fold")
        self.assertEqual(spot.verdict, "fold trop serré")
        self.assertGreater(spot.cost_bb, 0.0)

    def test_72o_fold_is_correct(self) -> None:
        hands = parse_ipoker(_hand_xml("S7 D2", "fold"))
        rev = review_pushfold_hands(hands, hero=self.HERO)
        self.assertEqual(rev.spots[0].verdict, "ok")
        self.assertEqual(rev.n_mistakes, 0)

    def test_deep_stack_is_skipped(self) -> None:
        # 100 bb effectifs : hors périmètre push/fold, compté à part
        hands = parse_ipoker(_hand_xml("SA HA", "jam", stack=1000, bb=10))
        rev = review_pushfold_hands(hands, hero=self.HERO)
        self.assertEqual(rev.spots, [])
        self.assertEqual(rev.n_skipped_deep, 1)

    def test_report_renders(self) -> None:
        hands = parse_ipoker(_hand_xml("S7 D2", "jam"))
        txt = review_pushfold_hands(hands, hero=self.HERO).explain()
        self.assertIn("SHOVE/FOLD", txt)
        self.assertIn("72o", txt)


if __name__ == "__main__":
    unittest.main()
