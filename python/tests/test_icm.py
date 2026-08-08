"""Tests F14 — ICM Malmuth-Harville, bubble factor, équité requise.

Valeurs de référence :
- Malmuth-Harville : P(i finit 1er) = s_i/S ; places suivantes par récursion.
- 3 joueurs à stacks égaux, payouts 50/30/20 → chacun 33.3333 %.
- α_ICM = BF·b/(P+b+BF·b) (se réduit au pot odds si BF=1).
- Prime de risque = BF/(1+BF) − 1/2.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pfs.core.icm import (
    IcmError,
    IcmSpot,
    analyse_icm_spot,
    bubble_factor,
    icm_dollar_ev,
    icm_equities,
    icm_required_equity,
    risk_premium,
)


class TestIcmEquities(unittest.TestCase):
    def test_equal_stacks_equal_ev(self) -> None:
        eq = icm_equities([1000, 1000, 1000], [50, 30, 20])
        for e in eq:
            self.assertAlmostEqual(float(e), 100 / 3, places=6)

    def test_sum_equals_prize_pool(self) -> None:
        eq = icm_equities([5000, 3000, 1500, 500], [50, 30, 20])
        self.assertAlmostEqual(float(np.sum(eq)), 100.0, places=9)

    def test_nonlinearity_big_stack_discounted(self) -> None:
        """Cœur de l'ICM : le chip-leader vaut MOINS que sa part de jetons."""
        eq = icm_equities([6000, 3000, 1000], [50, 30, 20])
        self.assertLess(float(eq[0]), 60.0)      # 60 % des jetons, < 60 % du pool
        self.assertGreater(float(eq[2]), 10.0)   # 10 % des jetons, > 10 % du pool

    def test_two_players_winner_take_all_is_chip_ev(self) -> None:
        """Winner-take-all HU : l'ICM se réduit au chipEV exact."""
        eq = icm_equities([7000, 3000], [100])
        self.assertAlmostEqual(float(eq[0]), 70.0, places=9)
        self.assertAlmostEqual(float(eq[1]), 30.0, places=9)

    def test_harville_exact_three_players(self) -> None:
        """Vérification manuelle complète sur 3 joueurs (calcul à la main).

        Stacks 50/30/20 (S=100), payouts 50/30/20.
        P(A 1er)=.5 ; P(B 1er)=.3 ; P(C 1er)=.2
        P(A 2e) = .3·(50/70) + .2·(50/80) = 0.339286 → $EV_A = 25+10.178571+3.214286
        P(B 2e) = .5·(30/50) + .2·(30/80) = 0.375     → $EV_B = 15+11.25+6.5
        P(C 2e) = .5·(20/50) + .3·(20/70) = 0.285714  → $EV_C = 10+8.571429+10.285714
        Somme = 38.392857 + 32.75 + 28.857143 = 100 ✓
        """
        eq = icm_equities([50, 30, 20], [50, 30, 20])
        self.assertAlmostEqual(float(eq[0]), 38.392857, places=5)
        self.assertAlmostEqual(float(eq[1]), 32.75, places=5)
        self.assertAlmostEqual(float(eq[2]), 28.857143, places=5)

    def test_zero_stack_zero_ev(self) -> None:
        eq = icm_equities([5000, 5000, 0], [60, 40])
        self.assertAlmostEqual(float(eq[2]), 40.0 * 0 + 0.0, places=9)

    def test_monte_carlo_beyond_exact_limit(self) -> None:
        stacks = list(range(1000, 14000, 1000))  # 13 joueurs → MC
        eq = icm_equities(stacks, [50, 30, 20], n_sims=100_000, seed=7)
        self.assertAlmostEqual(float(np.sum(eq)), 100.0, places=6)
        # Monotonie : plus gros stack → plus gros $EV
        self.assertTrue(all(eq[i] <= eq[i + 1] + 1e-9 for i in range(12)))

    def test_dollar_ev_single_player(self) -> None:
        v = icm_dollar_ev([5000, 3000, 1500, 500], [100, 60, 40], player=0)
        self.assertGreater(v, 50.0)
        with self.assertRaises(IcmError):
            icm_dollar_ev([100, 100], [50], player=5)

    def test_validation(self) -> None:
        with self.assertRaises(IcmError):
            icm_equities([], [50])
        with self.assertRaises(IcmError):
            icm_equities([-100, 200], [50])
        with self.assertRaises(IcmError):
            icm_equities([100, 200], [30, -10])


class TestBubbleFactor(unittest.TestCase):
    STACKS = [5000, 3000, 1500, 500]
    PAYOUTS = [50, 30, 20]  # 4 joueurs, 3 payés → bulle

    def test_bubble_factor_above_one_on_bubble(self) -> None:
        bf = bubble_factor(self.STACKS, self.PAYOUTS, hero=1, villain=0)
        self.assertGreater(bf, 1.0)

    def test_covering_stack_hurts_more(self) -> None:
        """BF contre un stack qui te couvre >> BF contre un shortstack."""
        bf_cover = bubble_factor(self.STACKS, self.PAYOUTS, hero=1, villain=0)
        bf_short = bubble_factor(self.STACKS, self.PAYOUTS, hero=1, villain=3)
        self.assertGreater(bf_cover, bf_short)
        self.assertGreater(bf_cover, 2.0)   # bulle sévère
        self.assertLess(bf_short, 1.5)      # risque limité au tapis du short

    def test_cash_game_equivalent_bf_one(self) -> None:
        """Payout proportionnel aux jetons (≈ cash game) → BF = 1."""
        # 2 joueurs winner-take-all : $EV linéaire en jetons → BF exactement 1
        bf = bubble_factor([6000, 4000], [100], hero=1, villain=0)
        self.assertAlmostEqual(bf, 1.0, places=9)

    def test_validation(self) -> None:
        with self.assertRaises(IcmError):
            bubble_factor(self.STACKS, self.PAYOUTS, hero=1, villain=1)
        with self.assertRaises(IcmError):
            bubble_factor(self.STACKS, self.PAYOUTS, hero=0, villain=1,
                          amount=99999)


class TestRequiredEquity(unittest.TestCase):
    def test_reduces_to_pot_odds_at_bf_one(self) -> None:
        # Cash : call 75 dans pot 100 → 75/250 = 0.3
        self.assertAlmostEqual(icm_required_equity(100, 75, 1.0), 0.3, places=9)

    def test_bubble_inflates_required_equity(self) -> None:
        self.assertAlmostEqual(icm_required_equity(100, 75, 2.0),
                               150 / 325, places=9)  # 0.461538

    def test_monotone_in_bf(self) -> None:
        prev = 0.0
        for bf in (1.0, 1.5, 2.0, 3.0, 5.0):
            cur = icm_required_equity(100, 100, bf)
            self.assertGreater(cur, prev)
            prev = cur

    def test_risk_premium_goldens(self) -> None:
        self.assertAlmostEqual(risk_premium(1.0), 0.0, places=9)
        self.assertAlmostEqual(risk_premium(1.5), 0.1, places=9)
        self.assertAlmostEqual(risk_premium(math.inf), 0.5, places=9)

    def test_validation(self) -> None:
        with self.assertRaises(IcmError):
            icm_required_equity(100, 0, 1.0)
        with self.assertRaises(IcmError):
            icm_required_equity(100, 50, 0.0)
        self.assertAlmostEqual(icm_required_equity(100, 50, math.inf), 1.0)


class TestSpotAnalysis(unittest.TestCase):
    def test_bubble_fold_verdict(self) -> None:
        """45 % d'équité : call trivial en cash-like, fold clair à la bulle."""
        spot = IcmSpot(stacks=(5000, 3000, 1500, 500), payouts=(50, 30, 20),
                       hero=1, villain=0, pot=400, bet=2800)
        a = analyse_icm_spot(spot, hero_equity=0.45)
        self.assertGreater(a.alpha_icm, a.alpha_cash)
        self.assertIn("FOLD", a.explain())

    def test_call_when_equity_clears_icm_bar(self) -> None:
        spot = IcmSpot(stacks=(5000, 3000, 1500, 500), payouts=(50, 30, 20),
                       hero=1, villain=3, pot=200, bet=400)
        a = analyse_icm_spot(spot, hero_equity=0.80)
        self.assertIn("CALL", a.explain())


class TestPko(unittest.TestCase):
    """L5 — bounties PKO : ½ cash + ½ sur sa propre prime × P(1er)."""

    def _spot(self, **kw) -> "PkoSpot":
        from pfs.core.icm import PkoSpot
        base = dict(stacks=(100.0, 0.0, 100.0), payouts=(100.0,),
                    bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
                    pot=100.0, bet=100.0)
        base.update(kw)
        return PkoSpot(**base)

    def test_winner_take_all_goldens_by_hand(self) -> None:
        """Calcul à la main : fold 33.33, gagne 66.67, perd 0 ;
        BV = 50·(½+½·200/300) = 41.667 ; r* = 33.33/108.33 = 0.307692."""
        from pfs.core.icm import analyse_pko_spot
        a = analyse_pko_spot(self._spot())
        self.assertAlmostEqual(a.ev_fold, 100 / 3, places=6)
        self.assertAlmostEqual(a.ev_win, 200 / 3, places=6)
        self.assertAlmostEqual(a.ev_lose, 0.0, places=9)
        self.assertAlmostEqual(a.bounty_value, 50 * (0.5 + 0.5 * 2 / 3), places=6)
        self.assertAlmostEqual(a.required_no_bounty, 0.5, places=9)
        self.assertAlmostEqual(a.required_with_bounty, (100 / 3) / (200 / 3 + 125 / 3),
                               places=9)
        self.assertAlmostEqual(a.discount_pts, 0.5 - 4 / 13, places=9)

    def test_bounty_flips_marginal_call(self) -> None:
        from pfs.core.icm import analyse_pko_spot
        a = analyse_pko_spot(self._spot(), hero_equity=0.40)
        self.assertIn("CALL", a.verdict)
        self.assertIn("prime", a.verdict)
        b = analyse_pko_spot(self._spot(bounties=(0.0, 0.0, 0.0)),
                             hero_equity=0.40)
        self.assertIn("FOLD", b.verdict)

    def test_no_bounty_when_villain_not_covered(self) -> None:
        """Vilain pas à tapis : la prime n'est pas en jeu."""
        from pfs.core.icm import analyse_pko_spot
        a = analyse_pko_spot(self._spot(stacks=(100.0, 40.0, 100.0)))
        self.assertFalse(a.villain_eliminated)
        self.assertAlmostEqual(a.bounty_value, 0.0)
        self.assertAlmostEqual(a.required_no_bounty, a.required_with_bounty,
                               places=12)

    def test_bigger_bounty_bigger_discount(self) -> None:
        from pfs.core.icm import analyse_pko_spot
        small = analyse_pko_spot(self._spot(bounties=(50.0, 20.0, 50.0)))
        big = analyse_pko_spot(self._spot(bounties=(50.0, 200.0, 50.0)))
        self.assertGreater(big.discount_pts, small.discount_pts)

    def test_validation(self) -> None:
        from pfs.core.icm import analyse_pko_spot
        with self.assertRaises(IcmError):
            analyse_pko_spot(self._spot(bounties=(50.0, 50.0)))
        with self.assertRaises(IcmError):
            analyse_pko_spot(self._spot(bet=500.0))
        with self.assertRaises(IcmError):
            analyse_pko_spot(self._spot(bounties=(-1.0, 50.0, 50.0)))


class TestFgs(unittest.TestCase):
    """L4 — FGS léger : érosion des blindes, conservation des jetons."""

    STACKS = [5000.0, 3000.0, 500.0, 1500.0]
    PAYOUTS = [50.0, 30.0, 20.0]

    def test_chip_conservation_every_hand(self) -> None:
        from pfs.core.icm import fgs_equities
        r = fgs_equities(self.STACKS, self.PAYOUTS, button=1, sb=100, bb=200,
                         ante=25, n_hands=8)
        for k in range(r.stacks_path.shape[0]):
            self.assertAlmostEqual(float(r.stacks_path[k].sum()), 10_000.0,
                                   places=6)
            self.assertAlmostEqual(float(r.equities[k].sum()), 100.0, places=6)

    def test_short_posting_first_bleeds(self) -> None:
        from pfs.core.icm import fgs_equities
        r = fgs_equities(self.STACKS, self.PAYOUTS, button=1, sb=100, bb=200,
                         n_hands=4)
        self.assertLess(float(r.deltas[1:, 2].min()), -1.0)

    def test_static_row_matches_icm(self) -> None:
        from pfs.core.icm import fgs_equities, icm_equities
        r = fgs_equities(self.STACKS, self.PAYOUTS, button=0, sb=100, bb=200)
        np.testing.assert_allclose(r.static,
                                   icm_equities(self.STACKS, self.PAYOUTS),
                                   atol=1e-9)

    def test_elimination_by_blinds(self) -> None:
        """Un stack plus petit que la SB disparaît de la rotation, $EV → plancher."""
        from pfs.core.icm import fgs_equities
        r = fgs_equities([5000.0, 80.0, 3000.0, 1920.0], self.PAYOUTS,
                         button=0, sb=100, bb=200, n_hands=6)
        self.assertAlmostEqual(float(r.stacks_path[-1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(r.stacks_path[-1].sum()), 10_000.0, places=6)

    def test_fgs_bubble_factor_moves(self) -> None:
        from pfs.core.icm import fgs_bubble_factor
        st, fg = fgs_bubble_factor(self.STACKS, self.PAYOUTS, hero=2, villain=0,
                                   button=1, sb=100, bb=200, n_hands=2)
        self.assertGreater(st, 1.0)
        self.assertNotAlmostEqual(st, fg, places=3)

    def test_validation(self) -> None:
        from pfs.core.icm import fgs_equities
        with self.assertRaises(IcmError):
            fgs_equities(self.STACKS, self.PAYOUTS, button=9, sb=100, bb=200)
        with self.assertRaises(IcmError):
            fgs_equities(self.STACKS, self.PAYOUTS, button=0, sb=100, bb=0)


if __name__ == "__main__":
    unittest.main()
