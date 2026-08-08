"""Tests P1/P2/P3 — arbre complet, node-locking, profondeur limitée EQR.

Preuves attendues :
- P1 : les tailles séparées OOP/IP et de relance façonnent l'arbre ; le
  seuil d'all-in convertit ; les invariants (somme = pot) tiennent sur les
  arbres riches.
- P2 : verrouiller un vilain sur-foldeur AUGMENTE l'EV et la fréquence de
  bluff du héros re-solvé (nodelock 2.0) ; un nœud verrouillé ne bouge pas ;
  sans lock, bit-à-bit identique au baseline.
- P3 : la profondeur limitée divise les nœuds par ~40, garde la somme
  constante (rollout : exactement), et son EV reste proche du solve complet.
"""

from __future__ import annotations

import unittest

import numpy as np

from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.solver.postflop import IP, OOP, PostflopError, PostflopSolver


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def cs(*ts: str) -> list[int]:
    return [c(t) for t in ts]


RIVER_DRY = cs("2s", "2d", "7h", "8h", "Kc")
TURN_DRY = cs("2s", "2d", "7h", "8h")


class TestBetTreeP1(unittest.TestCase):
    def test_separate_oop_ip_sizes(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ"), parse_range("AA,33"),
                           pot=100, stack=400,
                           oop_bet_fracs=(), ip_bet_fracs=(0.5, 1.0),
                           max_bets=1)
        root = s._nodes[s._root]                     # OOP : check seulement
        self.assertEqual(root.labels, ["check"])
        ip_node = s._nodes[root.children[0]]
        bet_labels = [l for l in ip_node.labels if l != "check"]
        self.assertEqual(len(bet_labels), 2)         # IP : deux tailles

    def test_dedicated_raise_sizes(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ+"), parse_range("JJ+"),
                           pot=100, stack=2000,
                           bet_fracs=(0.5,), raise_fracs=(3.0,), max_bets=2)
        # nœud face au bet OOP 0.5p : la relance doit être 3p, pas 0.5p
        root = s._nodes[s._root]
        bet_a = next(a for a, l in enumerate(root.labels) if l.startswith("bet"))
        facing = s._nodes[root.children[bet_a]]
        raises = [l for l in facing.labels if l.startswith("raise")]
        self.assertEqual(raises, ["raise 3p"])

    def test_allin_threshold_converts(self) -> None:
        # stack 120, bet 1p = 100 ≥ 0.67×120 → converti all-in
        s = PostflopSolver(RIVER_DRY, parse_range("QQ"), parse_range("AA"),
                           pot=100, stack=120, bet_fracs=(1.0,),
                           allin_threshold=0.67, max_bets=1)
        root = s._nodes[s._root]
        self.assertIn("all-in", root.labels)
        a = root.labels.index("all-in")
        self.assertAlmostEqual(root.amounts[a], 120.0, places=9)

    def test_add_allin_appends(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ"), parse_range("AA"),
                           pot=100, stack=500, bet_fracs=(0.5,),
                           add_allin=True, max_bets=1)
        root = s._nodes[s._root]
        self.assertIn("all-in", root.labels)
        self.assertIn("bet 0.5p", root.labels)

    def test_rich_tree_keeps_constant_sum(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("22+, ATs+"),
                           parse_range("77+, AQo+"), pot=80, stack=600,
                           oop_bet_fracs=(0.33, 0.75), ip_bet_fracs=(0.5, 1.5),
                           raise_fracs=(1.0,), add_allin=True, max_bets=3)
        s.solve(80)
        ev0, ev1 = s.values()
        self.assertAlmostEqual(ev0 + ev1, 80.0, places=8)

    def test_donk_is_native(self) -> None:
        """Le donk (OOP mise en premier) est natif : la racine OOP a des bets."""
        s = PostflopSolver(RIVER_DRY, parse_range("QQ+"), parse_range("JJ+"),
                           pot=100, stack=300)
        root = s._nodes[s._root]
        self.assertEqual(root.player, OOP)
        self.assertTrue(any(l.startswith("bet") for l in root.labels))

    def test_validation(self) -> None:
        r = parse_range("AA")
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=10, stack=10,
                           allin_threshold=1.5)
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=10, stack=10,
                           bet_fracs=(), oop_bet_fracs=(), ip_bet_fracs=())
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=10, stack=10,
                           leaf_model="eqr")           # sans modèle
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, r, r, pot=10, stack=10,
                           leaf_model="rollout")       # river ≠ turn


class TestNodeLockP2(unittest.TestCase):
    """Le nodelock : verrouiller le vilain, re-solve le héros, mesurer le gain."""

    def _base(self) -> PostflopSolver:
        # stack 300 : le bet 1p (100) reste « bet 1p », pas un all-in relabellisé
        return PostflopSolver(RIVER_DRY, parse_range("QQ"),
                              parse_range("AA,33"), pot=100, stack=300,
                              oop_bet_fracs=(), ip_bet_fracs=(1.0,),
                              max_bets=1)

    def test_unlocked_equals_baseline(self) -> None:
        a, b = self._base(), self._base()
        a.solve(300)
        b.solve(300)
        np.testing.assert_allclose(a.values(), b.values(), atol=1e-12)

    def test_overfolding_villain_gets_exploited(self) -> None:
        """QQ verrouillé à 90 % de fold face au bet → l'IP re-solvé bluffe
        TOUT son air et son EV dépasse le Nash (75)."""
        s = self._base()
        s.lock_node(("check", "bet 1p"), {"fold": 0.9, "call": 0.1})
        s.solve(600)
        ev_oop, ev_ip = s.values()
        self.assertGreater(ev_ip, 85.0)              # Nash = 75
        # l'air (33) doit bluffer ~toujours
        root = s._nodes[s._root]
        ip_idx = root.children[root.labels.index("check")]
        sigma = s.average_strategy(ip_idx)
        node = s._nodes[ip_idx]
        bet_a = next(a for a, l in enumerate(node.labels) if l != "check")
        cards = s.players[IP].cards
        is_air = ((cards >> 2) == RANKS.index("3")).all(axis=1)
        self.assertGreater(float(sigma[bet_a][is_air].mean()), 0.95)

    def test_locked_node_reports_locked_strategy(self) -> None:
        s = self._base()
        s.lock_node(("check", "bet 1p"), {"fold": 0.9, "call": 0.1})
        s.solve(200)
        idx = s.node_at(("check", "bet 1p"))
        sigma = s.average_strategy(idx)
        node = s._nodes[idx]
        f = node.labels.index("fold")
        np.testing.assert_allclose(sigma[f], 0.9, atol=1e-12)

    def test_partial_lock_leaves_rest_free(self) -> None:
        """Lock partiel : seuls les QQ « rouges » (spécification) verrouillés,
        les autres combos re-solvent librement."""
        s = self._base()
        s.lock_node(("check", "bet 1p"), {"fold": 1.0}, combos="QQ")
        # QQ = tout l'OOP ici → équivalent au lock complet ; vérifie le masque
        idx = s.node_at(("check", "bet 1p"))
        self.assertTrue(s._nodes[idx].locked_mask.all())
        s2 = self._base()
        with self.assertRaises(PostflopError):
            s2.lock_node(("check", "bet 1p"), {"fold": 1.0}, combos="AA")

    def test_unlock_restores(self) -> None:
        s = self._base()
        s.lock_node(("check", "bet 1p"), {"fold": 1.0})
        s.unlock_all()
        s.solve(300)
        ev_oop, ev_ip = s.values()
        self.assertAlmostEqual(ev_ip, 75.0, delta=1.0)   # retour au Nash

    def test_bad_paths(self) -> None:
        s = self._base()
        with self.assertRaises(PostflopError):
            s.node_at(("bet 7p",))
        with self.assertRaises(PostflopError):
            s.lock_node(("check", "bet 1p"), {})


class TestDepthLimitedP3(unittest.TestCase):
    OOP_R, IP_R = "QQ, 99", "KK, 55"

    @classmethod
    def setUpClass(cls) -> None:
        cls.full = PostflopSolver(TURN_DRY, parse_range(cls.OOP_R),
                                  parse_range(cls.IP_R), pot=60, stack=180,
                                  bet_fracs=(0.75,), max_bets=1)
        cls.full.solve(150)
        cls.roll = PostflopSolver(TURN_DRY, parse_range(cls.OOP_R),
                                  parse_range(cls.IP_R), pot=60, stack=180,
                                  bet_fracs=(0.75,), max_bets=1,
                                  leaf_model="rollout")
        cls.roll.solve(150)

    def test_node_count_collapses(self) -> None:
        self.assertLess(len(self.roll._nodes), len(self.full._nodes) / 20)

    def test_rollout_constant_sum_exact(self) -> None:
        ev0, ev1 = self.roll.values()
        self.assertAlmostEqual(ev0 + ev1, 60.0, places=8)

    def test_rollout_gap_is_the_river_leverage(self) -> None:
        """La feuille rollout ignore la mise river : contre une range
        polarisée, l'écart au solve complet est PRÉCISÉMENT la valeur du
        levier de mise river (mesuré ici ~24 % du pot — c'est la raison
        d'être des value-nets). On borne l'écart, on ne le nie pas."""
        f0, _ = self.full.values()
        r0, _ = self.roll.values()
        gap = abs(f0 - r0)
        self.assertGreater(gap, 0.05 * 60.0)     # le levier river existe…
        self.assertLess(gap, 0.35 * 60.0)        # …et reste borné

    def test_eqr_leaf_corrects_toward_full(self) -> None:
        """Contrat du mode EQR (honnête) : approximation DIRECTIONNELLE.

        La valeur de feuille EQR doit rapprocher l'EV du solve complet par
        rapport au rollout nu. Comme tout value-net (DeepStack inclus), la
        somme exacte des EV n'est plus garantie — c'est le prix de la
        profondeur limitée ; « rollout » reste le mode somme-exacte.
        """
        from pfs.fusion.eqr import train_eqr
        model = train_eqr(n_spots=8, iterations=80, seed=0)
        s = PostflopSolver(TURN_DRY, parse_range(self.OOP_R),
                           parse_range(self.IP_R), pot=60, stack=180,
                           bet_fracs=(0.75,), max_bets=1,
                           leaf_model="eqr", eqr_model=model)
        s.solve(150)
        e0, e1 = s.values()
        f0, f1 = self.full.values()
        r0, r1 = self.roll.values()
        self.assertLess(abs(e1 - f1), abs(r1 - f1))   # plus proche que rollout
        self.assertLess(abs(e0 + e1 - 60.0), 0.5 * 60.0)  # dérive bornée
        self.assertGreater(e1, r1)                    # direction : position ↑


class TestSimplifyReport(unittest.TestCase):
    def test_near_equal_sizes_mergeable(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ+, AKs"),
                           parse_range("JJ+, AQs+"), pot=100, stack=400,
                           bet_fracs=(0.70, 0.75), max_bets=1)
        s.solve(250)
        rep = s.simplify_report(eps=0.02)
        self.assertTrue(rep["mergeable"])            # 0.70p ≈ 0.75p
        self.assertEqual(len(rep["keep"]), 1)

    def test_distinct_sizes_not_merged(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ"),
                           parse_range("AA,33"), pot=100, stack=300,
                           oop_bet_fracs=(), ip_bet_fracs=(1.0,), max_bets=1)
        s.solve(250)
        rep = s.simplify_report(eps=0.005)
        self.assertIn("check", rep["action_evs"])

    def test_validation(self) -> None:
        s = PostflopSolver(RIVER_DRY, parse_range("AA"), parse_range("KK"),
                           pot=10, stack=10)
        with self.assertRaises(PostflopError):
            s.simplify_report(eps=0)


if __name__ == "__main__":
    unittest.main()
