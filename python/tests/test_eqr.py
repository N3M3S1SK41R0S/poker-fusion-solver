"""Tests L7 — EQR apprise depuis les solves du solveur postflop (L1).

Le fait de cohérence central : l'avantage de position doit ÉMERGER des
solutions Nash — le coefficient is_ip est positif sans qu'on l'ait codé.
"""

from __future__ import annotations

import unittest

from pfs.fusion.eqr import (
    FEATURES,
    EqrError,
    EqrModel,
    EqrSample,
    fit,
    generate_samples,
)


class TestEqrPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = generate_samples(n_spots=14, iterations=100, seed=0)
        cls.model = fit(cls.samples)

    def test_samples_shape(self) -> None:
        self.assertGreaterEqual(len(self.samples), 8)
        for s in self.samples:
            self.assertTrue(0.0 < s.equity < 1.0)
            self.assertTrue(0.1 < s.eqr < 3.0)

    def test_fit_quality(self) -> None:
        self.assertGreater(self.model.r2, 0.15)
        self.assertLessEqual(self.model.r2, 1.0)
        self.assertEqual(len(self.model.coef), len(FEATURES))

    def test_position_premium_emerges(self) -> None:
        """Le solveur n'a aucun trait « position » codé en dur : le
        coefficient IP > 0 sort des équilibres eux-mêmes."""
        beta_ip = self.model.coef[FEATURES.index("is_ip")]
        self.assertGreater(beta_ip, 0.0)

    def test_nut_share_positive(self) -> None:
        self.assertGreater(self.model.coef[FEATURES.index("nut_share")], 0.0)

    def test_prediction_bounds_and_direction(self) -> None:
        ip = self.model.predict(0.55, True, 2.0, 0.2, 0.1)
        oop = self.model.predict(0.55, False, 2.0, 0.2, 0.1)
        self.assertGreater(ip, oop)         # même main, mieux en position
        for v in (ip, oop):
            self.assertTrue(0.2 <= v <= 2.5)

    def test_deterministic_with_seed(self) -> None:
        again = fit(generate_samples(n_spots=14, iterations=100, seed=0))
        self.assertEqual(self.model.coef, again.coef)

    def test_validation(self) -> None:
        with self.assertRaises(EqrError):
            self.model.predict(1.5, True, 2.0, 0.2, 0.1)
        with self.assertRaises(EqrError):
            self.model.predict(0.5, True, -1.0, 0.2, 0.1)
        with self.assertRaises(EqrError):
            fit([EqrSample(0.5, 1.0, 0.0, 0.2, 0.1, 1.0)] * 3)
        with self.assertRaises(EqrError):
            generate_samples(n_spots=1)


if __name__ == "__main__":
    unittest.main()
