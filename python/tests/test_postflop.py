"""Tests L1 — solveur postflop range-vs-range.

Trois étages de preuve :
1. **Micro** : le CFV d'abattage O(n log n) avec card removal == double boucle
   brute-force O(n²), sur ranges denses avec blockers (le cœur du solveur).
2. **Nash analytique** : le spot de polarisation nuts/air contre bluff-catchers
   a une solution exacte connue (bluff = b/(P+2b) de la range de mise, call =
   P/(P+b) des catchers, EV = 3P/4 vs P/4 pour pot-size bet) — le solveur doit
   la retrouver à ε près.
3. **Invariants** : somme des EV = pot (somme constante, précision machine),
   exploitabilité décroissante, turn cohérent avec chance node.
"""

from __future__ import annotations

import unittest

import numpy as np

from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.solver.postflop import (
    IP,
    OOP,
    PostflopError,
    PostflopSolver,
)


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def cs(*ts: str) -> list[int]:
    return [c(t) for t in ts]


RIVER_DRY = cs("2s", "2d", "7h", "8h", "Kc")


class TestShowdownCfvBruteForce(unittest.TestCase):
    """Le calcul trié+préfixes == la double boucle, blockers inclus."""

    def _brute(self, solver: PostflopSolver, trav: int, q, pot, inv):
        p = solver.players[trav]
        o = solver.players[1 - trav]
        s_p, s_o = solver._strengths[-1] if trav == OOP else solver._strengths[-1][::-1]
        out = np.zeros(p.n)
        for i in range(p.n):
            a, b = p.cards[i]
            for j in range(o.n):
                if a in (o.cards[j][0], o.cards[j][1]) or \
                   b in (o.cards[j][0], o.cards[j][1]):
                    continue
                if s_p[i] > s_o[j]:
                    out[i] += q[j] * (pot - inv)
                elif s_p[i] == s_o[j]:
                    out[i] += q[j] * (0.5 * pot - inv)
                else:
                    out[i] += q[j] * (-inv)
        return out

    def test_dense_ranges_with_blockers(self) -> None:
        oop = parse_range("22+, A2s+, KTs+, ATo+")
        ip = parse_range("55+, A9s+, KQs, AJo+")
        s = PostflopSolver(RIVER_DRY, oop, ip, pot=60, stack=200)
        rng = np.random.default_rng(0)
        for trav in (OOP, IP):
            q = rng.random(s.players[1 - trav].n)
            ctx = s._sd_ctx[(trav, -1)]
            fast = ctx.cfv(q, pot=60.0, invested_p=25.0)
            slow = self._brute(s, trav, q, 60.0, 25.0)
            np.testing.assert_allclose(fast, slow, atol=1e-9)

    def test_identical_ranges_pair_combos(self) -> None:
        """Ranges identiques : le combo exactement identique (impossible) est
        bien retiré — le cas qui casse les implémentations naïves."""
        r = parse_range("QQ, JJ, TT")
        s = PostflopSolver(RIVER_DRY, r, r, pot=40, stack=100)
        q = np.ones(s.players[IP].n)
        ctx = s._sd_ctx[(OOP, -1)]
        fast = ctx.cfv(q, pot=40.0, invested_p=0.0)
        slow = self._brute(s, OOP, q, 40.0, 0.0)
        np.testing.assert_allclose(fast, slow, atol=1e-9)


class TestPolarizationNash(unittest.TestCase):
    """Solution analytique exacte : AA (nuts) + 33 (air) contre QQ, PSB.

    Nash : IP mise tout AA + la moitié de 33 (ratio bluff 1/3 de la range de
    mise = b/(P+2b)) ; QQ paie 1/2 (1−α, α = b/(P+b)) ; EV_IP = 3P/4.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = PostflopSolver(RIVER_DRY, parse_range("QQ"),
                                    parse_range("AA,33"), pot=100, stack=100,
                                    bet_fracs=(1.0,), max_bets=1)
        cls.solver.solve(800)
        cls.result = cls.solver.result()

    def test_evs_match_analytic(self) -> None:
        self.assertAlmostEqual(self.result.ev_ip, 75.0, delta=0.6)
        self.assertAlmostEqual(self.result.ev_oop, 25.0, delta=0.6)

    def test_constant_sum_machine_precision(self) -> None:
        self.assertAlmostEqual(self.result.ev_oop + self.result.ev_ip,
                               100.0, places=8)

    def test_low_exploitability(self) -> None:
        self.assertLess(self.result.exploitability, 0.005)

    def test_ip_bluffs_air_at_half(self) -> None:
        # nœud IP = enfant du check OOP
        root = self.solver._nodes[self.solver._root]
        ip_idx = root.children[root.labels.index("check")]
        node = self.solver._nodes[ip_idx]
        sigma = self.solver.average_strategy(ip_idx)
        bet_a = next(a for a, l in enumerate(node.labels) if l != "check")
        cards = self.solver.players[IP].cards
        # 33 = paires de rang 3 → valeur 1 (v = 12 − rang) ; rang de '3' = 11
        is_air = ((cards >> 2) == RANKS.index("3")).all(axis=1)
        is_nuts = ((cards >> 2) == RANKS.index("A")).all(axis=1)
        bluff_freq = float(sigma[bet_a][is_air].mean())
        value_freq = float(sigma[bet_a][is_nuts].mean())
        self.assertAlmostEqual(bluff_freq, 0.5, delta=0.05)
        self.assertGreater(value_freq, 0.97)

    def test_oop_calls_at_mdf(self) -> None:
        root = self.solver._nodes[self.solver._root]
        ip_idx = root.children[root.labels.index("check")]
        ip_node = self.solver._nodes[ip_idx]
        bet_a = next(a for a, l in enumerate(ip_node.labels) if l != "check")
        oop_idx = ip_node.children[bet_a]
        node = self.solver._nodes[oop_idx]
        sigma = self.solver.average_strategy(oop_idx)
        call_a = node.labels.index("call")
        call_freq = float(sigma[call_a].mean())
        self.assertAlmostEqual(call_freq, 0.5, delta=0.05)


class TestInvariants(unittest.TestCase):
    def test_constant_sum_generic_spot(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("22+, ATs+, KQs"),
                           parse_range("77+, AJs+, AQo+"),
                           pot=80, stack=300, bet_fracs=(0.5, 1.25),
                           max_bets=2)
        s.solve(120)
        ev0, ev1 = s.values()
        self.assertAlmostEqual(ev0 + ev1, 80.0, places=8)

    def test_exploitability_decreases(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ+, AKs"),
                           parse_range("JJ+, AQs+"), pot=50, stack=150)
        s.solve(40)
        early = s.exploitability()
        s.solve(360)
        late = s.exploitability()
        self.assertLess(late, early)
        self.assertLess(late, 0.02)

    def test_exploitability_nonnegative(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("AA"),
                           parse_range("KK"), pot=100, stack=100)
        s.solve(200)
        self.assertGreaterEqual(s.exploitability(), -1e-9)

    def test_nut_range_gets_the_pot(self) -> None:
        """AA contre KK, board 2s2d7h8h3c (pas de K !) : AA gagne toujours.

        Les ranges étant d'information commune, KK ne paie jamais un bet de
        la range 100 % nuts : l'EV de AA converge vers exactement le pot.
        (Premier jet de ce test : board avec Kc — KK y fait un FULL et le
        solveur rendait 100 à KK. Le solveur avait raison, pas le test.)
        """
        board = cs("2s", "2d", "7h", "8h", "3c")
        s = PostflopSolver(board, parse_range("AA"),
                           parse_range("KK"), pot=100, stack=200)
        s.solve(300)
        ev0, ev1 = s.values()
        self.assertGreater(ev0, 99.0)     # AA prend ~tout le pot
        self.assertLess(ev1, 1.0)


class TestTurnSolving(unittest.TestCase):
    """Turn = chance node sur les rivières — petit mais réel."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = PostflopSolver(cs("2s", "2d", "7h", "8h"),
                                    parse_range("QQ, 99"),
                                    parse_range("KK, 55"),
                                    pot=60, stack=180,
                                    bet_fracs=(0.75,), max_bets=1)
        cls.solver.solve(150)

    def test_constant_sum(self) -> None:
        ev0, ev1 = self.solver.values()
        self.assertAlmostEqual(ev0 + ev1, 60.0, places=7)

    def test_exploitability_reasonable(self) -> None:
        self.assertLess(self.solver.exploitability(), 0.03)

    def test_chance_masks_river_collisions(self) -> None:
        """La masse μ ne dépend pas de la rivière ; les EV restent finies."""
        ev0, ev1 = self.solver.values()
        self.assertTrue(np.isfinite(ev0) and np.isfinite(ev1))


class TestValidation(unittest.TestCase):
    def test_bad_inputs(self) -> None:
        r = parse_range("AA")
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("2s", "2d", "7h"), r, r, pot=10, stack=10)
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=0, stack=10)
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=10, stack=10, bet_fracs=())
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("2s", "2s", "7h", "8h", "Kc"), r, r,
                           pot=10, stack=10)

    def test_blocked_range_raises(self) -> None:
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("As", "Ad", "Ah", "Ac", "Kc"),
                           parse_range("AA"), parse_range("KK"),
                           pot=10, stack=10)


if __name__ == "__main__":
    unittest.main()
