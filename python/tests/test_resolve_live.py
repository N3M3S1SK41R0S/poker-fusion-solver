"""Tests P4 — LE différenciateur : re-solve depuis la range inférée (F3→L1).

Le test de valeur : contre un vilain dont la range réelle est PLAFONNÉE
(aucune main forte), la stratégie re-solvée depuis la range inférée doit
battre la stratégie GTO-préset figée — face à la même range, vilain
répondant au mieux dans les deux cas. Le gain mesuré EST la valeur de
l'inférence en direct, celle que personne d'autre ne calcule.
"""

from __future__ import annotations

import unittest

from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.engine import FusionEngine, ResolveReport


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


BOARD = [c(x) for x in ("Ks", "7d", "2c", "9h", "3s")]
HERO = "TT+, AKs, AKo, KQs"                 # value-lourd
CAPPED = "22-99, A2s-A9s, KTs-KJs, QTs+"    # vilain plafonné : zéro overpair
REF = "22+, ATs+, KTs+, QTs+, JTs, ATo+, KQo"  # préset « équilibré »


class TestResolveLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = FusionEngine()
        # simule une inférence aboutie : le filtre part de la range plafonnée
        cls.engine.start_hand("villain#1", prior_range=parse_range(CAPPED))
        cls.report = cls.engine.resolve_spot(
            "villain#1",
            board=BOARD, pot=100, hero_range=parse_range(HERO),
            hero_position="ip", stack=300, bet_fracs=(0.75,),
            max_bets=2, iterations=250,
            reference_villain=parse_range(REF),
        )

    def test_uses_inferred_range(self) -> None:
        self.assertIn("inférée", self.report.villain_source)
        self.assertGreater(self.report.lam, 0.5)

    def test_inference_gain_is_positive(self) -> None:
        """LE test : le re-solve bat le GTO figé contre le vilain plafonné."""
        self.assertGreater(self.report.exploit_gain, 2.0)   # > 2 % du pot
        self.assertGreater(self.report.ev_hero_exploit,
                           self.report.ev_hero_gto_locked)

    def test_solve_quality(self) -> None:
        self.assertLess(self.report.exploitability, 0.03)
        self.assertTrue(self.report.root_actions)

    def test_report_explains(self) -> None:
        txt = self.report.explain()
        self.assertIn("gain d'inférence", txt)
        self.assertIn("λ=", txt)


class TestResolveFallback(unittest.TestCase):
    def test_no_observations_falls_back_to_preset(self) -> None:
        eng = FusionEngine()
        rep = eng.resolve_spot(
            "inconnu#9", board=BOARD, pot=100,
            hero_range=parse_range(HERO), hero_position="ip", stack=300,
            iterations=150, reference_villain=parse_range(REF),
        )
        self.assertIn("préset", rep.villain_source)
        self.assertAlmostEqual(rep.lam, 0.0)
        # exploit vs sa propre référence : gain ≈ 0 (au bruit CFR près)
        self.assertLess(abs(rep.exploit_gain), 3.0)

    def test_asymmetric_stacks_effective(self) -> None:
        eng = FusionEngine()
        rep = eng.resolve_spot(
            "x#1", board=BOARD, pot=100, hero_range=parse_range(HERO),
            hero_position="oop", hero_stack=500, villain_stack=120,
            iterations=100, reference_villain=parse_range(REF),
        )
        self.assertIsInstance(rep, ResolveReport)

    def test_validation(self) -> None:
        eng = FusionEngine()
        with self.assertRaises(ValueError):
            eng.resolve_spot("x", board=BOARD, pot=100,
                             hero_range=parse_range(HERO),
                             hero_position="middle", stack=100)
        with self.assertRaises(ValueError):
            eng.resolve_spot("x", board=BOARD, pot=100,
                             hero_range=parse_range(HERO))  # pas de stack
        with self.assertRaises(ValueError):
            from pfs.core.rake import WINAMAX_MICRO
            eng.resolve_spot("x", board=BOARD, pot=100,
                             hero_range=parse_range(HERO), stack=100,
                             game_format="mtt", rake=WINAMAX_MICRO)


if __name__ == "__main__":
    unittest.main()
