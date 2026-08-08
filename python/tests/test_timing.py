"""Tests L2 — tells temporels : log-normal, Welford, CUSUM, Wilson.

Le test central est adversarial : sur du BRUIT pur, le détecteur de tells ne
doit (presque) rien déclarer ; sur un tell PLANTÉ (snap-bet = air), il doit
le trouver. C'est le même contrat que la TDA (F7) : pas d'exploits fantômes.
"""

from __future__ import annotations

import math
import random
import unittest

from pfs.fusion.timing import (
    ActionClass,
    StrengthBin,
    TimingError,
    TimingObservation,
    TimingProfile,
    TimingTell,
    strength_bin_from_equity,
    wilson_interval,
)


class TestWilson(unittest.TestCase):
    def test_golden_8_of_10(self) -> None:
        """À la main : p=0.8, z²=3.8416 → centre 0.71674, demi-largeur 0.22658."""
        lo, hi = wilson_interval(8, 10)
        self.assertAlmostEqual(lo, 0.49016, places=4)
        self.assertAlmostEqual(hi, 0.94332, places=4)

    def test_extremes(self) -> None:
        lo, hi = wilson_interval(0, 20)
        self.assertAlmostEqual(lo, 0.0, places=9)
        self.assertLess(hi, 0.20)
        lo, hi = wilson_interval(20, 20)
        self.assertGreater(lo, 0.80)
        self.assertAlmostEqual(hi, 1.0, places=9)

    def test_validation(self) -> None:
        with self.assertRaises(TimingError):
            wilson_interval(5, 0)
        with self.assertRaises(TimingError):
            wilson_interval(-1, 10)


class TestSurprise(unittest.TestCase):
    def test_baseline_then_zscore(self) -> None:
        p = TimingProfile()
        rng = random.Random(7)
        # ligne de base log-normale : médiane 4 s, σ_log 0.35
        for _ in range(60):
            t = math.exp(math.log(4.0) + rng.gauss(0, 0.35))
            p.observe(TimingObservation(ActionClass.BET_BIG, t))
        self.assertLess(abs(p.surprise(ActionClass.BET_BIG, 4.0)), 0.8)
        self.assertLess(p.surprise(ActionClass.BET_BIG, 0.5), -2.0)   # snap
        self.assertGreater(p.surprise(ActionClass.BET_BIG, 40.0), 2.0)  # tank

    def test_needs_min_n(self) -> None:
        p = TimingProfile(min_n=8)
        for _ in range(3):
            p.observe(TimingObservation(ActionClass.CALL, 3.0))
        self.assertNotEqual(p.surprise(ActionClass.CALL, 3.0),
                            p.surprise(ActionClass.CALL, 3.0))  # nan != nan

    def test_global_fallback(self) -> None:
        """Classe jamais vue : la ligne de base globale prend le relais."""
        p = TimingProfile()
        for _ in range(20):
            p.observe(TimingObservation(ActionClass.CHECK, 2.0))
        z = p.surprise(ActionClass.RAISE, 2.0)
        self.assertEqual(z, z)          # pas nan
        self.assertLess(abs(z), 1.0)

    def test_surprise_computed_before_update(self) -> None:
        """La 1re observation d'une classe mûre ne se juge pas elle-même."""
        p = TimingProfile()
        for _ in range(30):
            p.observe(TimingObservation(ActionClass.CALL, 3.0))
        z = p.observe(TimingObservation(ActionClass.CALL, 30.0))
        self.assertGreater(z, 2.0)      # jugée contre AVANT, pas après

    def test_validation(self) -> None:
        p = TimingProfile()
        with self.assertRaises(TimingError):
            p.observe(TimingObservation(ActionClass.CALL, 0.0))
        with self.assertRaises(TimingError):
            TimingProfile(min_n=1)


class TestCusum(unittest.TestCase):
    def test_regime_change_detected(self) -> None:
        """Passage médiane 4 s → 12 s : le CUSUM sonne en peu de mains."""
        p = TimingProfile()
        rng = random.Random(11)
        for _ in range(50):
            t = math.exp(math.log(4.0) + rng.gauss(0, 0.3))
            p.observe(TimingObservation(ActionClass.CALL, t))
        self.assertEqual(p.drift_alarms, 0)
        for _ in range(12):
            t = math.exp(math.log(12.0) + rng.gauss(0, 0.3))
            p.observe(TimingObservation(ActionClass.CALL, t))
        self.assertGreaterEqual(p.drift_alarms, 1)

    def test_stationary_alarm_rate_matches_arl(self) -> None:
        """Sous H₀, (k=0.5, h=5) → ARL₀ ≈ 470 obs (Page) : sur 300
        observations stationnaires, on attend 0 ou 1 alarme — pas une rafale."""
        p = TimingProfile()
        rng = random.Random(13)
        for _ in range(300):
            t = math.exp(math.log(5.0) + rng.gauss(0, 0.4))
            p.observe(TimingObservation(ActionClass.CALL, t))
        self.assertLessEqual(p.drift_alarms, 1)


class TestTells(unittest.TestCase):
    def _profile_with_baseline(self, seed: int = 3) -> TimingProfile:
        p = TimingProfile()
        rng = random.Random(seed)
        for _ in range(80):
            t = math.exp(math.log(5.0) + rng.gauss(0, 0.4))
            p.observe(TimingObservation(ActionClass.BET_BIG, t))
        return p

    def test_planted_tell_found(self) -> None:
        """Tell planté : les snap-bets de ce joueur sont de l'air à 85 %."""
        p = self._profile_with_baseline()
        rng = random.Random(5)
        for _ in range(40):     # snaps : air à 85 %
            t = math.exp(math.log(5.0) - 1.2 + rng.gauss(0, 0.15))
            s = StrengthBin.AIR if rng.random() < 0.85 else StrengthBin.STRONG
            p.record_showdown(ActionClass.BET_BIG, t, s)
        for _ in range(40):     # temps normaux : force équilibrée
            t = math.exp(math.log(5.0) + rng.gauss(0, 0.15))
            s = (StrengthBin.STRONG if rng.random() < 0.5 else StrengthBin.MEDIUM)
            p.record_showdown(ActionClass.BET_BIG, t, s)
        tells = p.tells()
        self.assertTrue(tells)
        # le tell planté est déclaré (ses miroirs — « normal n'est jamais
        # air » — sont aussi valides et peuvent le devancer au classement)
        planted = [t for t in tells
                   if t.tercile == "rapide" and t.strength is StrengthBin.AIR]
        self.assertTrue(planted)
        self.assertGreater(planted[0].rate, planted[0].baseline)
        self.assertGreater(planted[0].ci_low, planted[0].baseline)

    def test_pure_noise_stays_quiet(self) -> None:
        """Sur du bruit : force indépendante du temps → (presque) aucun tell.

        Chaque cellule×force a ~5 % de faux positifs par construction de
        l'IC ; on tolère ≤ 2 déclarations sur 18 cellules-forces possibles.
        """
        p = self._profile_with_baseline(seed=17)
        rng = random.Random(19)
        strengths = list(StrengthBin)
        for _ in range(150):
            t = math.exp(math.log(5.0) + rng.gauss(0, 0.4))
            p.record_showdown(ActionClass.BET_BIG, t, rng.choice(strengths))
        self.assertLessEqual(len(p.tells()), 2)

    def test_small_sample_reports_nothing(self) -> None:
        p = self._profile_with_baseline()
        p.record_showdown(ActionClass.BET_BIG, 1.0, StrengthBin.AIR)
        p.record_showdown(ActionClass.BET_BIG, 1.1, StrengthBin.AIR)
        self.assertEqual(p.tells(), [])

    def test_summary_serializable(self) -> None:
        import json
        p = self._profile_with_baseline()
        s = p.summary()
        json.dumps(s)               # ne lève pas
        self.assertGreater(s["n_obs"], 0)
        self.assertIn("bet_big", s["baselines"])

    def test_strength_bin_mapping(self) -> None:
        self.assertIs(strength_bin_from_equity(0.10), StrengthBin.AIR)
        self.assertIs(strength_bin_from_equity(0.50), StrengthBin.MEDIUM)
        self.assertIs(strength_bin_from_equity(0.90), StrengthBin.STRONG)
        with self.assertRaises(TimingError):
            strength_bin_from_equity(1.5)


if __name__ == "__main__":
    unittest.main()
