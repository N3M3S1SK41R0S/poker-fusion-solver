"""Tests L6 — population mining : comptages, priors empirical-Bayes, SQLite.

Vérité de comptage établie à la main sur la main Winamax de référence :
Dave = héros (exclu). Alice fold BB (VPIP non, PFR non), Bob open-raise
(VPIP oui, PFR oui), Carol fold (non/non). Une relance existe → occasions de
3bet pour les trois. Bob c-bet au flop ; personne d'autre que Dave (exclu)
n'y fait face → aucune occasion fold_to_cbet pour le pool.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pfs.data.hand_history import parse_text
from pfs.data.population import (
    PopulationError,
    PopulationMiner,
    stake_band,
)

WNMX = """Winamax Poker - CashGame - HandId: #12345-678-1234567890 - Holdem no limit (0.50€)/(1€) - 2026/08/06 20:14:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Bob raises 2€ to 3€
Carol folds
Dave calls 2.50€
Alice folds
*** FLOP *** [Ks 7d 2c]
Dave checks
Bob bets 4€
Dave folds
Bob collected 7€ from pot
*** SUMMARY ***
Total pot 7€"""


class TestStakeBand(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(stake_band(0.02, False), "NL2")
        self.assertEqual(stake_band(1.0, False), "NL100")
        self.assertEqual(stake_band(999, True), "MTT")
        with self.assertRaises(PopulationError):
            stake_band(0, False)


class TestIngestHandCounts(unittest.TestCase):
    def test_counts_verified_by_hand(self) -> None:
        m = PopulationMiner()
        hand = parse_text(WNMX)
        m.ingest(hand)                      # héros (Dave) exclu
        band = "NL100"
        # VPIP : Alice non, Bob oui, Carol non → 1/3
        self.assertEqual(m.occasions(band, "vpip"), 3)
        self.assertAlmostEqual(m.rate(band, "vpip"), 1 / 3)
        # PFR : Bob seul → 1/3
        self.assertAlmostEqual(m.rate(band, "pfr"), 1 / 3)
        # 3bet : une relance existe → occasion pour les 3, personne ne 3bet
        self.assertEqual(m.occasions(band, "three_bet"), 3)
        self.assertAlmostEqual(m.rate(band, "three_bet"), 0.0)
        # fold_to_cbet : seul le héros a fait face au c-bet → 0 occasion pool
        self.assertEqual(m.occasions(band, "fold_to_cbet"), 0)

    def test_hero_included_when_asked(self) -> None:
        m = PopulationMiner()
        m.ingest(parse_text(WNMX), exclude_hero=False)
        self.assertEqual(m.occasions("NL100", "vpip"), 4)
        self.assertAlmostEqual(m.rate("NL100", "vpip"), 2 / 4)  # Bob + Dave

    def test_ingest_text_multi(self) -> None:
        m = PopulationMiner()
        hands, obs = m.ingest_text(WNMX + "\n\n" + WNMX)
        self.assertEqual(hands, 2)
        self.assertEqual(m.occasions("NL100", "vpip"), 6)
        self.assertGreater(obs, 0)


class TestPrior(unittest.TestCase):
    def _mined(self, k: int, n: int) -> PopulationMiner:
        m = PopulationMiner()
        for i in range(n):
            m._add("NL100", "vpip", i < k)
        return m

    def test_no_data_gives_jeffreys(self) -> None:
        p = PopulationMiner().prior_for("NL100", "vpip")
        self.assertAlmostEqual(p.alpha, 0.5)
        self.assertAlmostEqual(p.beta, 0.5)
        self.assertEqual(p.n_pop, 0)

    def test_strength_capped_by_pool_information(self) -> None:
        """40 occasions ne méritent pas un prior de 30 : n0_eff = 4."""
        p = self._mined(10, 40).prior_for("NL100", "vpip", n0=30.0)
        self.assertAlmostEqual(p.strength, 4.0, places=9)
        # 3000 occasions → plafond atteint : n0_eff = 30
        p2 = self._mined(750, 3000).prior_for("NL100", "vpip", n0=30.0)
        self.assertAlmostEqual(p2.strength, 30.0, places=9)

    def test_prior_mean_tracks_pool_rate(self) -> None:
        p = self._mined(750, 3000).prior_for("NL100", "vpip", n0=30.0)
        self.assertAlmostEqual(p.mean, 0.25, delta=0.01)

    def test_prior_feeds_tracker(self) -> None:
        """Le prior s'injecte dans F1 et s'efface sous les observations."""
        from pfs.fusion.dynamic_beta import DynamicBetaTracker
        pop = self._mined(750, 3000).prior_for("NL100", "vpip", n0=30.0)
        t = DynamicBetaTracker("vpip", discount=1.0, prior_mean=pop.mean,
                               prior_strength=pop.strength)
        self.assertAlmostEqual(t.belief.mean, 0.25, delta=0.02)
        t.update_batch(successes=45, trials=60)     # joueur réel : 75 %
        # conjugaison exacte : (7.5+45)/(30+60) = 0.5833 — l'observation domine
        self.assertAlmostEqual(t.belief.mean, 52.5 / 90.0, delta=0.01)

    def test_validation(self) -> None:
        with self.assertRaises(PopulationError):
            PopulationMiner().prior_for("NL100", "vpip", n0=-1)
        with self.assertRaises(PopulationError):
            PopulationMiner().rate("NL100", "vpip")


class TestPersistence(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        m = PopulationMiner()
        m.ingest_text(WNMX)
        with TemporaryDirectory() as d:
            path = Path(d) / "pop.sqlite"
            m.save(path)
            m2 = PopulationMiner.load(path)
        self.assertEqual(m2.occasions("NL100", "vpip"), 3)
        self.assertEqual(m2.n_hands, 1)
        self.assertAlmostEqual(m2.rate("NL100", "vpip"), 1 / 3)

    def test_table_report(self) -> None:
        m = PopulationMiner()
        m.ingest_text(WNMX)
        rows = m.table()
        self.assertTrue(rows)
        r = next(x for x in rows if x["stat"] == "vpip")
        self.assertEqual(r["n"], 3)
        self.assertLess(r["ci_low"], r["rate"])
        self.assertLess(r["rate"], r["ci_high"])


if __name__ == "__main__":
    unittest.main()
