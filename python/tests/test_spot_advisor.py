"""Tests du conseiller de spot — « qu'est-ce qu'il fallait faire ? ».

Goldens ancrés sur des vérités non discutables : AA pousse à 10 bb, 72o
passe ; les cotes du pot sont exactes (payer 75 dans 100 exige 30 %) ; une
main qui bat 100 % de la range adverse a 100 % d'équité.
"""

from __future__ import annotations

import unittest

from pfs.analysis.spot_advisor import Spot, advise, parse_cards


class TestParseCards(unittest.TestCase):
    def test_compact_and_spaced(self) -> None:
        self.assertEqual(parse_cards("AhKd"), parse_cards("Ah Kd"))
        self.assertEqual(len(parse_cards("Qs 7d 2c 9h 3s")), 5)

    def test_ten_written_as_10(self) -> None:
        # « 10h » et « Th » désignent la même carte
        self.assertEqual(parse_cards("10h"), parse_cards("Th"))

    def test_unicode_suits_and_case(self) -> None:
        self.assertEqual(parse_cards("A♠ K♦"), parse_cards("as kd"))

    def test_duplicate_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_cards("Ah Ah")


class TestPreflopShortStack(unittest.TestCase):
    """Régime Nash push/fold — le solveur tranche, verdict certain."""

    def test_aces_jam(self) -> None:
        a = advise(Spot(hero="Ah Ad", stack=10, big_blind=1, position="BTN"))
        self.assertTrue(a.action.startswith("JAM"))
        self.assertEqual(a.confidence, "certain")
        self.assertEqual(a.hand, "AA")
        self.assertGreater(a.ev_bb, 0.0)

    def test_seven_deuce_folds(self) -> None:
        a = advise(Spot(hero="7s 2d", stack=10, big_blind=1, position="BTN"))
        self.assertEqual(a.action, "FOLD")
        self.assertEqual(a.hand, "72o")
        self.assertLess(a.ev_bb, 0.0)

    def test_deep_stack_uses_chart_not_nash(self) -> None:
        a = advise(Spot(hero="Ah Ad", stack=100, big_blind=1, position="BTN"))
        self.assertEqual(a.confidence, "indicatif")
        self.assertIn("chart", a.regime)


class TestPostflop(unittest.TestCase):
    def test_nuts_gets_full_equity(self) -> None:
        # quinte royale sur board As Ks Qs : bat toute range → 100 %
        a = advise(Spot(hero="Js Ts", board="As Ks Qs 2d 3c",
                        pot=100, bet=50, stack=300, big_blind=10))
        self.assertAlmostEqual(a.equity, 1.0, places=6)
        self.assertTrue(a.action.startswith("CALL"))

    def test_pot_odds_are_exact(self) -> None:
        # payer 75 dans un pot de 100 exige 75/(100+75+75) = 30 %
        a = advise(Spot(hero="Ah Qd", board="Qs 7d 2c 9h 3s",
                        pot=100, bet=75, stack=300, big_blind=10))
        self.assertAlmostEqual(a.required, 0.30, places=6)
        self.assertAlmostEqual(a.mdf, 100 / 175, places=6)

    def test_air_folds_facing_a_bet(self) -> None:
        a = advise(Spot(hero="5c 4d", board="Qs 7d 2c 9h Ks",
                        pot=100, bet=60, stack=300, big_blind=10))
        self.assertEqual(a.action, "FOLD")
        self.assertLess(a.equity, a.required)

    def test_no_bet_faced_is_a_value_question(self) -> None:
        a = advise(Spot(hero="Ah Kh", board="Kd 7h 2c",
                        pot=50, bet=0, stack=300, big_blind=10))
        self.assertIsNone(a.required)          # pas de cote : personne n'a misé
        self.assertIn("MISER", a.action)       # top paire meilleur kicker

    def test_assumption_is_always_declared(self) -> None:
        a = advise(Spot(hero="Ah Qd", board="Qs 7d 2c", pot=50, bet=25,
                        stack=300, big_blind=10))
        self.assertTrue(a.assumptions)         # la range supposée est dite
        self.assertEqual(a.confidence, "indicatif")


class TestValidation(unittest.TestCase):
    def test_card_in_hand_and_board(self) -> None:
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah Kd", board="Ah 7d 2c", pot=10, bet=5))

    def test_wrong_hero_card_count(self) -> None:
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah", stack=10, big_blind=1))

    def test_impossible_board_length(self) -> None:
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah Kd", board="Qs 7d", pot=10, bet=5))


if __name__ == "__main__":
    unittest.main()
