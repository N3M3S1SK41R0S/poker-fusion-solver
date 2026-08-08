"""Tests — abstraction par buckets de force de main (Johanson et al., 2013).

Valeurs de référence (déterministes : énumération exacte des runouts, card
removal exact) :

- Flop sec Ks 7d 2c : EHS(AsAh) ≈ 0,887 > EHS(KhQh) ≈ 0,874 > EHS(8h7h) ≈
  0,705 > EHS(QhJh) ≈ 0,468 > EHS(6h3d) ≈ 0,186 — paire d'as > top paire >
  paire moyenne > overcards > hauteur.
- Board 9h 8h 7s (draw vs made) : QhJh (tirage couleur + gutshot + deux
  overcards, EHS ≈ 0,646) et Ad9d (top paire top kicker, EHS ≈ 0,653) ont la
  même force espérée à 0,007 près, mais σ(QhJh) ≈ 0,347 ≈ 2 × σ(Ad9d) ≈ 0,164 :
  la distribution sépare ce que l'espérance confond (Johanson et al., 2013).
  NB : sur ce board, 6h5h est une SUITE FAITE (9-8-7-6-5 + tirage quinte
  flush), pas un tirage — son σ ≈ 0,115 est le plus bas du lot, ce qui
  illustre le phénomène (main faite ⇒ force stable) au lieu de l'infirmer.
- Oracle croisé : au turn, EHS d'un combo = équité exacte de ce combo contre
  ``Range.full()`` rendue par ``pfs.core.equity`` — deux chemins de calcul
  indépendants qui énumèrent le même espace (river × combo adverse).
- Carré d'as sur As Ah Ad Kc : nuts absolu sur les 46 rivières → EHS = 1 et
  σ = 0, exacts (le card removal exact rend le percentile des nuts égal à 1).
"""

from __future__ import annotations

import itertools
import time
import unittest

import numpy as np

from pfs.core.equity import equity_vs_range
from pfs.core.range_model import RANKS, SUITS, Range, combo_cards, parse_range
from pfs.solver.abstraction import (
    AbstractionError,
    BucketSummary,
    abstract_range,
    bucket_assignments,
    distribution_aware_features,
    expected_hand_strength,
)


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def board(txt: str) -> list[int]:
    return [c(t) for t in txt.split()]


def combos(*txt: str) -> np.ndarray:
    return np.array([[c(t[:2]), c(t[2:])] for t in txt], dtype=np.int64)


FLOP_SEC = board("Ks 7d 2c")
FLOP_DRAW = board("9h 8h 7s")
TURN = board("Ks 7d 2c 9h")
TURN_QUADS = board("As Ah Ad Kc")


class TestEHSMonotonie(unittest.TestCase):
    def test_ordre_des_forces_flop_sec(self) -> None:
        """Sur Ks 7d 2c : AA > KQ (top paire) > 87 (paire moyenne) > QJ > 63o."""
        ehs = expected_hand_strength(
            combos("AsAh", "KhQh", "8h7h", "QhJh", "6h3d"), FLOP_SEC
        )
        aa, kq, mid, qj, junk = (float(x) for x in ehs)
        self.assertGreater(aa, kq)
        self.assertGreater(kq, mid)
        self.assertGreater(mid, qj)
        self.assertGreater(qj, junk)

    def test_valeurs_goldens(self) -> None:
        """Valeurs exactes figées (énumération complète, card removal exact)."""
        ehs = expected_hand_strength(combos("AsAh", "QhJh"), FLOP_SEC)
        self.assertAlmostEqual(float(ehs[0]), 0.8873, delta=1e-3)
        self.assertAlmostEqual(float(ehs[1]), 0.4684, delta=1e-3)

    def test_bornes_et_forme(self) -> None:
        ehs = expected_hand_strength(combos("AsAh", "6h3d"), FLOP_SEC)
        self.assertEqual(ehs.shape, (2,))
        self.assertEqual(ehs.dtype, np.float64)
        self.assertTrue(np.all(ehs >= 0.0) and np.all(ehs <= 1.0))

    def test_nuts_absolu_ehs_1_sigma_0_exacts(self) -> None:
        """Carré d'as au turn As Ah Ad Kc : imbattable et jamais partagé sur
        les 46 rivières (aucune couleur ni quinte flush possible) → HS = 1 sur
        chaque runout, donc EHS = 1.0 et σ = 0.0 EXACTS — l'ancre qui prouve
        le card removal (sans lui, les 45 combos fantômes contenant Ac
        plafonneraient le percentile à ≈ 0,979)."""
        feats = distribution_aware_features(combos("Ac2h"), TURN_QUADS)
        self.assertEqual(float(feats[0, 0]), 1.0)
        self.assertEqual(float(feats[0, 1]), 0.0)


class TestOracleEquite(unittest.TestCase):
    def test_ehs_turn_egale_equite_contre_range_complete(self) -> None:
        """Oracle indépendant : au turn, EHS = moyenne sur les rivières de
        P(gagner) + ½·P(partager) contre une main aléatoire — exactement ce
        que ``equity_vs_range(hero, Range.full(), turn)`` énumère (46 rivières
        × 990 combos adverses). Les deux implémentations n'ont aucun code
        commun au-delà de ``evaluate7``."""
        for hand in ("AsAh", "QhJh", "6h5h"):
            hero = [c(hand[:2]), c(hand[2:])]
            ehs = float(expected_hand_strength(np.array([hero]), TURN)[0])
            ref = equity_vs_range(hero, Range.full(), TURN)
            self.assertTrue(ref.exact)
            self.assertAlmostEqual(ehs, ref.equity, places=12)


class TestInvarianceIsomorphisme(unittest.TestCase):
    def test_permutation_de_couleurs_valeurs_identiques(self) -> None:
        """EHS(σ(combo), σ(board)) == EHS(combo, board) au bit près : les deux
        requêtes transitent par la même table canonique (Waugh, 2013)."""
        perm = {0: 1, 1: 2, 2: 3, 3: 0}  # s→h, h→d, d→c, c→s

        def sigma(card: int) -> int:
            return 4 * (card // 4) + perm[card % 4]

        hands = combos("AsAh", "KhQh", "8h7h", "QhJh", "6h3d")
        base = distribution_aware_features(hands, FLOP_SEC)
        image = distribution_aware_features(
            np.vectorize(sigma)(hands), [sigma(x) for x in FLOP_SEC]
        )
        self.assertTrue(np.array_equal(base, image))


class TestDrawVsMade(unittest.TestCase):
    """Le cœur de Johanson et al. (2013) : même EHS, distributions opposées."""

    def test_tirage_plus_variable_quune_main_faite_de_meme_ehs(self) -> None:
        """Sur 9h 8h 7s : QhJh (gros tirage, Q-high) et Ad9d (top paire faite)
        ont un EHS quasi identique (≈ 0,65 tous deux), mais l'écart-type de la
        force du tirage est plus du double de celui de la main faite —
        l'abstraction distribution-aware DOIT pouvoir les séparer."""
        feats = distribution_aware_features(combos("QhJh", "Ad9d"), FLOP_DRAW)
        ehs_draw, std_draw = (float(x) for x in feats[0])
        ehs_made, std_made = (float(x) for x in feats[1])
        self.assertLess(abs(ehs_draw - ehs_made), 0.05)   # « même EHS approximatif »
        self.assertGreater(std_draw, std_made + 0.10)     # mesuré : 0,347 vs 0,164

    def test_main_faite_forte_a_sigma_faible(self) -> None:
        """6h5h sur 9h 8h 7s est une suite FAITE (9-8-7-6-5) : force déjà
        réalisée, donc σ inférieur à celui du tirage QhJh — et même inférieur
        à celui du set 9c9d (qui, lui, redraw au full sur board pairé)."""
        feats = distribution_aware_features(combos("6h5h", "QhJh", "9c9d"), FLOP_DRAW)
        self.assertGreater(float(feats[0, 0]), 0.85)             # EHS de la suite faite
        self.assertLess(float(feats[0, 1]), float(feats[1, 1]))  # σ(65s) < σ(QJs)
        self.assertLess(float(feats[0, 1]), float(feats[2, 1]))  # σ(65s) < σ(99)

    def test_coherence_features_et_ehs(self) -> None:
        """La colonne 0 des features EST expected_hand_strength (même table)."""
        hands = combos("QhJh", "Ad9d", "6h5h")
        feats = distribution_aware_features(hands, FLOP_DRAW)
        ehs = expected_hand_strength(hands, FLOP_DRAW)
        self.assertEqual(feats.shape, (3, 2))
        self.assertTrue(np.array_equal(feats[:, 0], ehs))
        self.assertTrue(np.all(feats[:, 1] >= 0.0))


class TestSousEchantillonnage(unittest.TestCase):
    def test_reproductible_a_graine_fixee(self) -> None:
        hands = combos("AsAh", "QhJh")
        a = expected_hand_strength(hands, FLOP_SEC, n_rollouts=300, seed=3)
        b = expected_hand_strength(hands, FLOP_SEC, n_rollouts=300, seed=3)
        self.assertTrue(np.array_equal(a, b))

    def test_graines_differentes_echantillons_differents(self) -> None:
        hands = combos("AsAh", "QhJh")
        a = expected_hand_strength(hands, FLOP_SEC, n_rollouts=300, seed=3)
        b = expected_hand_strength(hands, FLOP_SEC, n_rollouts=300, seed=4)
        self.assertFalse(np.array_equal(a, b))

    def test_proche_de_l_exact(self) -> None:
        """300 runouts sur 1 176 : erreur-type ≈ σ/√300 ≤ 0,02 — tolérance 0,05."""
        hands = combos("AsAh", "QhJh")
        approx = expected_hand_strength(hands, FLOP_SEC, n_rollouts=300, seed=3)
        exact = expected_hand_strength(hands, FLOP_SEC)
        self.assertLess(float(np.max(np.abs(approx - exact))), 0.05)

    def test_n_rollouts_couvrant_tout_egale_l_exact(self) -> None:
        """n_rollouts ≥ nombre total de runouts → énumération exacte (la graine
        devient sans effet et l'entrée de cache est partagée)."""
        hands = combos("AsAh", "QhJh")
        exact = expected_hand_strength(hands, FLOP_SEC)
        full = expected_hand_strength(hands, FLOP_SEC, n_rollouts=1176, seed=99)
        over = expected_hand_strength(hands, FLOP_SEC, n_rollouts=10**6, seed=7)
        self.assertTrue(np.array_equal(exact, full))
        self.assertTrue(np.array_equal(exact, over))

    def test_echantillon_trop_court_pour_couvrir_un_combo(self) -> None:
        """Au turn avec n_rollouts=1, une seule rivière est tirée : les combos
        contenant cette carte n'ont aucun runout valide — demander TOUS les
        combos vivants doit donc lever une erreur explicite."""
        remaining = [x for x in range(52) if x not in TURN]
        every = np.array(list(itertools.combinations(remaining, 2)), dtype=np.int64)
        with self.assertRaises(AbstractionError):
            expected_hand_strength(every, TURN, n_rollouts=1)


class TestBucketsPercentile(unittest.TestCase):
    def test_masses_egales_a_un_combo_pres(self) -> None:
        gen = np.random.default_rng(7)
        strengths = gen.permutation(np.linspace(0.0, 1.0, 100))
        labels = bucket_assignments(strengths, n_buckets=8)
        sizes = np.bincount(labels, minlength=8)
        self.assertEqual(int(sizes.sum()), 100)
        self.assertTrue(np.all(sizes > 0))
        self.assertLessEqual(int(sizes.max() - sizes.min()), 1)

    def test_masses_egales_taille_non_divisible(self) -> None:
        gen = np.random.default_rng(11)
        strengths = gen.permutation(np.linspace(0.0, 1.0, 97))
        sizes = np.bincount(bucket_assignments(strengths, n_buckets=8), minlength=8)
        self.assertLessEqual(int(sizes.max() - sizes.min()), 1)

    def test_ordre_des_buckets_suit_la_force(self) -> None:
        gen = np.random.default_rng(13)
        strengths = gen.random(200)
        labels = bucket_assignments(strengths, n_buckets=6)
        by_strength = labels[np.argsort(strengths, kind="stable")]
        self.assertTrue(np.all(np.diff(by_strength) >= 0))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertTrue(np.all((labels >= 0) & (labels < 6)))

    def test_masse_ponderee_equilibree(self) -> None:
        """Avec des poids de range, chaque bucket reçoit W/K à un poids près."""
        gen = np.random.default_rng(17)
        strengths = gen.random(150)
        weights = 0.1 + 0.9 * gen.random(150)
        labels = bucket_assignments(strengths, n_buckets=5, weights=weights)
        masses = np.bincount(labels, weights=weights, minlength=5)
        target = weights.sum() / 5.0
        self.assertLess(float(np.max(np.abs(masses - target))),
                        float(weights.max()) + 1e-12)

    def test_un_seul_bucket(self) -> None:
        labels = bucket_assignments(np.array([0.1, 0.9, 0.5]), n_buckets=1)
        self.assertTrue(np.array_equal(labels, np.zeros(3, dtype=np.int64)))

    def test_forces_toutes_egales_masses_preservees(self) -> None:
        """Égalités parfaites : l'équilibre des masses prime (documenté)."""
        sizes = np.bincount(
            bucket_assignments(np.full(10, 0.5), n_buckets=2), minlength=2
        )
        self.assertTrue(np.array_equal(sizes, np.array([5, 5])))


class TestBucketsKmeans(unittest.TestCase):
    def _trois_paquets(self) -> np.ndarray:
        jitter = np.linspace(-0.02, 0.02, 10)
        return np.concatenate([0.1 + jitter, 0.5 + jitter, 0.9 + jitter])

    def test_deterministe_deux_executions_identiques(self) -> None:
        gen = np.random.default_rng(23)
        strengths = gen.random(300)
        a = bucket_assignments(strengths, n_buckets=8, method="kmeans")
        b = bucket_assignments(strengths, n_buckets=8, method="kmeans")
        self.assertTrue(np.array_equal(a, b))

    def test_retrouve_trois_paquets_separes(self) -> None:
        """Trois modes bien séparés → k-means colle aux modes (ce que les
        quantiles de masse ne garantissent pas)."""
        strengths = self._trois_paquets()
        labels = bucket_assignments(strengths, n_buckets=3, method="kmeans")
        expected = np.repeat(np.array([0, 1, 2]), 10)
        self.assertTrue(np.array_equal(labels, expected))

    def test_ordre_des_buckets_suit_la_force(self) -> None:
        gen = np.random.default_rng(29)
        strengths = gen.random(200)
        labels = bucket_assignments(strengths, n_buckets=5, method="kmeans")
        by_strength = labels[np.argsort(strengths, kind="stable")]
        self.assertTrue(np.all(np.diff(by_strength) >= 0))

    def test_un_seul_bucket(self) -> None:
        labels = bucket_assignments(
            np.array([0.1, 0.9, 0.5]), n_buckets=1, method="kmeans"
        )
        self.assertTrue(np.array_equal(labels, np.zeros(3, dtype=np.int64)))

    def test_poids_acceptes(self) -> None:
        strengths = self._trois_paquets()
        weights = np.full(30, 0.5)
        labels = bucket_assignments(
            strengths, n_buckets=3, method="kmeans", weights=weights
        )
        self.assertTrue(np.array_equal(labels, np.repeat(np.array([0, 1, 2]), 10)))


class TestAbstractRange(unittest.TestCase):
    def test_conservation_de_la_masse(self) -> None:
        rng_ = parse_range("TT+, AQs+, AJo+")
        buckets = abstract_range(rng_, TURN, n_buckets=4)
        live = rng_.remove_blockers(TURN)
        self.assertAlmostEqual(
            sum(b.weight for b in buckets.values()), live.n_combos, places=9
        )

    def test_ehs_moyen_croissant_avec_le_bucket(self) -> None:
        buckets = abstract_range(parse_range("TT+, AQs+, AJo+"), TURN, n_buckets=4)
        keys = sorted(buckets)
        means = [buckets[k].mean_ehs for k in keys]
        self.assertTrue(all(a <= b for a, b in zip(means, means[1:])))
        self.assertLess(means[0], means[-1])
        for k in keys:
            self.assertIsInstance(buckets[k], BucketSummary)
            self.assertTrue(0.0 <= buckets[k].mean_ehs <= 1.0)

    def test_representants_format_et_plafond(self) -> None:
        buckets = abstract_range(
            parse_range("TT+, AQs+, AJo+"), TURN, n_buckets=3, max_representatives=2
        )
        for summary in buckets.values():
            self.assertGreaterEqual(len(summary.combos), 1)
            self.assertLessEqual(len(summary.combos), 2)
            for combo in summary.combos:
                self.assertRegex(combo, r"^[AKQJT98765432][shdc][AKQJT98765432][shdc]$")

    def test_poids_fractionnaires_conserves(self) -> None:
        rng_ = parse_range("QQ:0.5, AKs, A5s:0.25")
        buckets = abstract_range(rng_, TURN, n_buckets=2)
        live = rng_.remove_blockers(TURN)
        self.assertAlmostEqual(
            sum(b.weight for b in buckets.values()), live.n_combos, places=9
        )

    def test_methode_kmeans(self) -> None:
        buckets = abstract_range(
            parse_range("TT+, AQs+, AJo+"), TURN, n_buckets=3, method="kmeans"
        )
        live = parse_range("TT+, AQs+, AJo+").remove_blockers(TURN)
        self.assertAlmostEqual(
            sum(b.weight for b in buckets.values()), live.n_combos, places=9
        )
        self.assertTrue(set(buckets) <= set(range(3)))

    def test_bucket_unique(self) -> None:
        buckets = abstract_range(parse_range("KK, QQ"), TURN_QUADS, n_buckets=1)
        self.assertEqual(sorted(buckets), [0])

    def test_range_vide_apres_blockers(self) -> None:
        """AA sur As Ah Ad Kc : les 6 combos d'as heurtent le board → erreur."""
        with self.assertRaises(AbstractionError):
            abstract_range(parse_range("AA"), TURN_QUADS, n_buckets=2)


class TestValidation(unittest.TestCase):
    def test_board_invalide(self) -> None:
        good = combos("AsAh")
        for bad in ([], board("Ks 7d"), board("Ks 7d 2c 9h 3s"),
                    [c("Ks"), c("Ks"), c("2c")], [0, 4, 52], [0, 4, -1]):
            with self.assertRaises(AbstractionError):
                expected_hand_strength(good, bad)

    def test_combos_invalides(self) -> None:
        for bad in (np.array([0, 5]),                       # 1-D
                    np.empty((0, 2), dtype=np.int64),       # vide
                    np.array([[0, 5, 9]]),                  # (n, 3)
                    np.array([[0.0, 5.0]]),                 # non entier
                    np.array([[0, 52]]),                    # hors domaine
                    np.array([[7, 7]])):                    # cartes identiques
            with self.assertRaises(AbstractionError):
                expected_hand_strength(bad, FLOP_SEC)

    def test_combo_heurtant_le_board(self) -> None:
        with self.assertRaises(AbstractionError):
            expected_hand_strength(combos("KsAh"), FLOP_SEC)

    def test_n_rollouts_invalide(self) -> None:
        for bad in (0, -2, "300", 1.5, True):
            with self.assertRaises(AbstractionError):
                expected_hand_strength(combos("AsAh"), FLOP_SEC, n_rollouts=bad)

    def test_seed_invalide(self) -> None:
        for bad in ("x", 1.5, True):
            with self.assertRaises(AbstractionError):
                expected_hand_strength(
                    combos("AsAh"), FLOP_SEC, n_rollouts=300, seed=bad
                )

    def test_bucket_strengths_invalides(self) -> None:
        for bad in (np.array([[0.1, 0.2]]), np.array([]),
                    np.array([0.1, np.nan]), np.array([0.1, np.inf])):
            with self.assertRaises(AbstractionError):
                bucket_assignments(bad, n_buckets=2)

    def test_bucket_n_buckets_invalide(self) -> None:
        s = np.array([0.1, 0.5, 0.9])
        for bad in (0, -1, 2.5, True):
            with self.assertRaises(AbstractionError):
                bucket_assignments(s, n_buckets=bad)

    def test_bucket_methode_inconnue(self) -> None:
        with self.assertRaises(AbstractionError):
            bucket_assignments(np.array([0.1, 0.9]), n_buckets=2, method="foo")

    def test_bucket_poids_invalides(self) -> None:
        s = np.array([0.1, 0.5, 0.9])
        for bad in (np.array([1.0, 1.0]),          # taille
                    np.array([1.0, -1.0, 1.0]),    # négatif
                    np.array([0.0, 0.0, 0.0]),     # somme nulle
                    np.array([1.0, np.nan, 1.0])):  # non fini
            with self.assertRaises(AbstractionError):
                bucket_assignments(s, n_buckets=2, weights=bad)

    def test_abstract_range_arguments_invalides(self) -> None:
        with self.assertRaises(AbstractionError):
            abstract_range(np.ones(1326), TURN, n_buckets=2)  # pas une Range
        with self.assertRaises(AbstractionError):
            abstract_range(parse_range("AA"), TURN, n_buckets=2, max_representatives=0)


class TestPerformance(unittest.TestCase):
    def test_ehs_exact_de_cent_combos_sous_30_secondes(self) -> None:
        """Budget de la Phase 2 : ≈ 100 combos sur un flop FRAIS (aucune entrée
        de cache) en énumération exacte — mesuré ≈ 8 s, borne 30 s. La table
        couvrant les 1 326 combos d'un coup, le coût est indépendant de n."""
        rng_ = parse_range("22+, ATs+, KQs, AQo+")
        fresh = board("Qc 9h 4s")
        live = rng_.remove_blockers(fresh)
        idx = np.nonzero(live.weights > 0.0)[0]
        cards = np.array([combo_cards(int(i)) for i in idx], dtype=np.int64)
        self.assertGreaterEqual(cards.shape[0], 100)

        start = time.monotonic()
        ehs = expected_hand_strength(cards, fresh)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 30.0)
        self.assertEqual(ehs.shape, (cards.shape[0],))
        self.assertTrue(np.all((ehs >= 0.0) & (ehs <= 1.0)))


if __name__ == "__main__":
    unittest.main()
