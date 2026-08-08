"""Tests du moteur d'équité — oracle brute-force + équités publiées.

Deux niveaux de vérification :
1. ``evaluate7`` == max des 21 sous-mains de 5 cartes évaluées par l'oracle
   scalaire indépendant (2 000 mains aléatoires + cas adversariaux construits).
2. Équités de référence connues de la littérature (Sklansky ; tout calculateur
   d'équité standard) : AA vs KK ≈ 81.9 %, AKs vs QQ ≈ 46 %, etc.
"""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from pfs.core.equity import (
    EquityResult,
    _eval5_scalar,
    equity_vs_range,
    evaluate7,
    hand_category,
)
from pfs.core.range_model import RANKS, SUITS, RangeError, parse_range


def c(txt: str) -> int:
    """'As' → indice de carte."""
    return RANKS.index(txt[0]) * 4 + SUITS.index(txt[1])


def cs(*txts: str) -> list[int]:
    return [c(t) for t in txts]


def oracle7(hand: list[int]) -> int:
    """Référence : max des 21 sous-mains de 5 cartes."""
    return max(_eval5_scalar(sub) for sub in itertools.combinations(hand, 5))


class TestEvaluator(unittest.TestCase):
    def test_categories_on_constructed_hands(self) -> None:
        cases = [
            (cs("As", "Ks", "Qs", "Js", "Ts", "2d", "3c"), "quinte flush"),
            (cs("Ah", "5h", "4h", "3h", "2h", "Kd", "Kc"), "quinte flush"),  # roue
            (cs("Ah", "Ad", "Ac", "As", "Kd", "2c", "3h"), "carré"),
            (cs("Ah", "Ad", "Ac", "Kd", "Ks", "2c", "3h"), "full"),
            (cs("Ah", "Ad", "Ac", "Kd", "Kс".replace("с", "c"), "Qd", "Qs"), "full"),
            (cs("Ah", "Kh", "9h", "5h", "2h", "Qd", "Js"), "couleur"),
            (cs("9c", "8d", "7h", "6s", "5c", "Ad", "Ks"), "suite"),
            (cs("Ah", "2d", "3c", "4s", "5h", "Kd", "Qc"), "suite"),        # roue
            (cs("Ah", "Ad", "Ac", "Kd", "Qs", "2c", "3h"), "brelan"),
            (cs("Ah", "Ad", "Kc", "Kd", "Qs", "2c", "3h"), "deux paires"),
            (cs("Ah", "Ad", "Kc", "Qd", "Js", "2c", "3h"), "paire"),
            (cs("Ah", "Kd", "Qc", "Jd", "9s", "2c", "3h"), "hauteur"),
        ]
        for hand, want in cases:
            code = int(evaluate7(np.array([hand]))[0])
            self.assertEqual(hand_category(code), want, f"main {hand}")

    def test_adversarial_flush_bit_clobbering(self) -> None:
        """Deux cartes de même rang, une seule dans la couleur : le bit de la
        couleur ne doit pas être écrasé (bug classique d'écriture indexée)."""
        # Couleur à pique avec Ks ; Kd hors couleur ne doit rien effacer.
        hand = cs("As", "Ks", "Kd", "9s", "5s", "2s", "7c")
        code = int(evaluate7(np.array([hand]))[0])
        self.assertEqual(hand_category(code), "couleur")
        self.assertEqual(code, oracle7(hand))

    def test_two_trips_is_full_house(self) -> None:
        hand = cs("Ah", "Ad", "Ac", "Kd", "Ks", "Kc", "2h")
        code = int(evaluate7(np.array([hand]))[0])
        self.assertEqual(hand_category(code), "full")
        self.assertEqual(code, oracle7(hand))

    def test_three_pairs_kicker(self) -> None:
        hand = cs("Ah", "Ad", "Kc", "Kd", "2s", "2c", "Qh")
        code = int(evaluate7(np.array([hand]))[0])
        self.assertEqual(hand_category(code), "deux paires")
        self.assertEqual(code, oracle7(hand))

    def test_steel_wheel_vs_higher_flush(self) -> None:
        """Quinte flush A-5 bat une couleur à l'as — catégorie prime."""
        wheel = cs("Ah", "2h", "3h", "4h", "5h", "Kd", "Qc")
        flush = cs("Ah", "Kh", "Qh", "Jh", "9h", "2d", "3c")
        codes = evaluate7(np.array([wheel, flush]))
        self.assertGreater(int(codes[0]), int(codes[1]))

    def test_random_sample_matches_oracle(self) -> None:
        """2 000 mains aléatoires : accord exact avec l'oracle 21-sous-mains."""
        g = np.random.default_rng(42)
        hands = np.array([g.choice(52, size=7, replace=False)
                          for _ in range(2000)])
        got = evaluate7(hands)
        for i in range(2000):
            self.assertEqual(int(got[i]), oracle7([int(x) for x in hands[i]]),
                             f"main {[int(x) for x in hands[i]]}")

    def test_validation(self) -> None:
        with self.assertRaises(RangeError):
            evaluate7(np.zeros((3, 6), dtype=int))
        with self.assertRaises(RangeError):
            evaluate7(np.array([[0, 1, 2, 3, 4, 5, 99]]))


class TestEquityExact(unittest.TestCase):
    def test_river_locked_win(self) -> None:
        r = equity_vs_range(cs("Ah", "Ad"), parse_range("KK"),
                            cs("2s", "7d", "9c", "Jh", "3s"))
        self.assertTrue(r.exact)
        self.assertAlmostEqual(r.equity, 1.0, places=9)
        self.assertEqual(r.n_scenarios, 6)   # 6 combos de KK

    def test_board_plays_full_chop(self) -> None:
        """Broadway sur le board : tout le monde partage."""
        r = equity_vs_range(cs("2h", "3d"), parse_range("77"),
                            cs("As", "Kd", "Qc", "Jh", "Th"))
        self.assertAlmostEqual(r.equity, 0.5, places=9)
        self.assertAlmostEqual(r.tie, 1.0, places=9)

    def test_turn_flush_draw_exact_outs(self) -> None:
        """Tirage couleur au turn contre brelan : équité exacte 7/44.

        Héros Ah Kh, board Qh 7h 2s 9d, vilain 99 (3 combos restants après le
        blocage du 9d). Rivières gagnantes = cœurs qui ne remplissent pas le
        vilain : 9h donne carré (et 2h donne full 999+22) → il reste
        Jh Th 8h 6h 5h 4h 3h, soit 7 rivières sur 44 — pour CHAQUE combo,
        qu'il contienne le 9h (qui bloque une rivière perdante) ou non.
        """
        r = equity_vs_range(cs("Ah", "Kh"), parse_range("99"),
                            cs("Qh", "7h", "2s", "9d"))
        self.assertTrue(r.exact)
        self.assertEqual(r.n_scenarios, 3 * 44)  # 3 combos × 44 rivières
        self.assertAlmostEqual(r.equity, 7 / 44, places=9)

    def test_flop_set_vs_overpair(self) -> None:
        """Set contre overpair au flop : la valeur canonique ~90 %."""
        r = equity_vs_range(cs("7h", "7d"), parse_range("AA"),
                            cs("7s", "2d", "9c"))
        self.assertTrue(r.exact)
        self.assertGreater(r.equity, 0.87)
        self.assertLess(r.equity, 0.95)

    def test_dead_card_collision_raises(self) -> None:
        with self.assertRaises(RangeError):
            equity_vs_range(cs("Ah", "Ah"), parse_range("KK"), [])
        with self.assertRaises(RangeError):
            equity_vs_range(cs("Ah", "Kd"), parse_range("KK"),
                            cs("Ah", "2c", "3d"))

    def test_range_fully_blocked_raises(self) -> None:
        # Villain = AA exactement, mais héros tient 2 as et le board les 2 autres
        with self.assertRaises(RangeError):
            equity_vs_range(cs("Ah", "Ad"), parse_range("AA"),
                            cs("As", "Ac", "5d"))


class TestEquityPreflop(unittest.TestCase):
    """Équités préflop de référence (valeurs publiées, tolérance MC)."""

    def _eq(self, hero: list[int], rng: str) -> EquityResult:
        return equity_vs_range(hero, parse_range(rng), n_sims=200_000, seed=1)

    def test_aa_vs_kk(self) -> None:
        r = self._eq(cs("Ah", "Ad"), "KK")
        self.assertAlmostEqual(r.equity, 0.8255, delta=0.01)
        self.assertFalse(r.exact)
        self.assertLess(r.std_error, 0.002)

    def test_aks_vs_qq(self) -> None:
        r = self._eq(cs("Ah", "Kh"), "QQ")
        self.assertAlmostEqual(r.equity, 0.46, delta=0.012)

    def test_ako_vs_22_coinflip(self) -> None:
        r = self._eq(cs("Ah", "Kd"), "22")
        self.assertAlmostEqual(r.equity, 0.47, delta=0.015)

    def test_aa_vs_full_random_range(self) -> None:
        from pfs.core.range_model import Range
        r = equity_vs_range(cs("Ah", "Ad"), Range.full(),
                            n_sims=200_000, seed=1)
        self.assertAlmostEqual(r.equity, 0.85, delta=0.012)

    def test_dominated_hand(self) -> None:
        """AQo dominée par la range AK : ~25 % (24.3 vs AKs, 25.9 vs AKo)."""
        r = self._eq(cs("Ah", "Qd"), "AKs,AKo")
        self.assertAlmostEqual(r.equity, 0.254, delta=0.015)


if __name__ == "__main__":
    unittest.main()


class TestEquityMultiway(unittest.TestCase):
    """L3 — héros contre plusieurs ranges (partage espéré du pot)."""

    def test_river_locked_multiway_win(self) -> None:
        from pfs.core.equity import equity_multiway
        r = equity_multiway(cs("Ah", "Ad"),
                            [parse_range("KK"), parse_range("QQ")],
                            cs("2s", "7d", "9c", "Jh", "3s"))
        self.assertTrue(r.exact)
        self.assertAlmostEqual(r.equity, 1.0, places=9)

    def test_three_way_chop_exact_shares(self) -> None:
        """Board broadway : chacun 1/3 exactement (partage à trois)."""
        from pfs.core.equity import equity_multiway
        r = equity_multiway(cs("2h", "3d"),
                            [parse_range("66"), parse_range("55")],
                            cs("As", "Kd", "Qc", "Jh", "Th"))
        self.assertTrue(r.exact)
        self.assertAlmostEqual(r.equity, 1 / 3, places=9)
        self.assertAlmostEqual(r.tie, 1.0, places=9)

    def test_exact_matches_mc(self) -> None:
        """Le MC converge vers l'énumération exacte (même spot river)."""
        from pfs.core.equity import _multiway_mc, _villain_combos, equity_multiway
        hero, board = cs("Ah", "Kh"), cs("Kd", "7s", "2c", "9h", "3d")
        vill = [parse_range("QQ,JJ"), parse_range("77,66")]
        ex = equity_multiway(hero, vill, board)
        dead = set(hero) | set(board)
        packs = [_villain_combos(v, dead) for v in vill]
        mc = _multiway_mc(hero, board, packs, dead, n_sims=60_000, seed=3)
        self.assertTrue(ex.exact)
        self.assertFalse(mc.exact)
        self.assertAlmostEqual(ex.equity, mc.equity, delta=4 * max(mc.std_error, 1e-3))

    def test_published_three_way_preflop(self) -> None:
        """AA vs KK vs QQ ≈ 66-67 % (valeur publiée ~66.5)."""
        from pfs.core.equity import equity_multiway
        r = equity_multiway(cs("Ah", "Ad"),
                            [parse_range("KK"), parse_range("QQ")],
                            n_sims=120_000, seed=2)
        self.assertAlmostEqual(r.equity, 0.665, delta=0.015)

    def test_single_villain_delegates(self) -> None:
        from pfs.core.equity import equity_multiway
        r1 = equity_multiway(cs("Ah", "Ad"), [parse_range("KK")],
                             cs("2s", "7d", "9c", "Jh", "3s"))
        r2 = equity_vs_range(cs("Ah", "Ad"), parse_range("KK"),
                             cs("2s", "7d", "9c", "Jh", "3s"))
        self.assertAlmostEqual(r1.equity, r2.equity, places=12)

    def test_validation(self) -> None:
        from pfs.core.equity import equity_multiway
        with self.assertRaises(RangeError):
            equity_multiway(cs("Ah", "Ad"), [])
