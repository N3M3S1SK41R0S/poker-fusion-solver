"""Tests de l'effet de bunching — vérités calculables à la main + validation croisée.

Quatre niveaux de vérification :
1. Cas dégénéré exact : un folder dont la range de fold est UN combo précis
   (As Ad) → tout combo cible touchant As ou Ad a un multiplicateur
   exactement nul, les autres exactement 1 — pour le Monte-Carlo ET
   l'approximation pairwise.
2. Direction du biais : trois folders serrés (folds de UTG/MP/CO, presets
   GTO) → les petites cartes sont appauvries, les grosses enrichies ;
   ratio multiplicateur(AKs)/multiplicateur(22) mesuré ≈ 1.10.
3. Convergence croisée : MC (50 000 sims, graine fixe) contre pairwise —
   corrélation mesurée ≈ 0.995 (> 0.95 exigé), écart max mesuré ≈ 0.011
   (< 0.03 exigé) : les corrélations entre folders, négligées par
   pairwise, restent un effet de second ordre sur des folds larges.
4. Cas limites et validation des entrées (0 folder = identité, méthode
   inconnue, cartes mortes invalides, modèles incohérents…).
"""

from __future__ import annotations

import unittest

import numpy as np

from pfs.core.bunching import (
    BunchingError,
    apply_bunching,
    bunching_weights_mc,
    bunching_weights_pairwise,
    fold_range_from_play,
)
from pfs.core.range_model import (
    GTO_PRESETS,
    N_COMBOS,
    RANKS,
    SUITS,
    Range,
    combo_cards,
    combo_index,
    parse_range,
)


def c(txt: str) -> int:
    """'As' → indice de carte (rang*4 + couleur)."""
    return RANKS.index(txt[0]) * 4 + SUITS.index(txt[1])


def single_combo_range(a: str, b: str) -> Range:
    """Range dégénérée : poids 1 sur UN combo exact, 0 partout ailleurs."""
    w = np.zeros(N_COMBOS)
    w[combo_index(c(a), c(b))] = 1.0
    return Range(w)


def combos_touching(cards: set[int]) -> np.ndarray:
    """Masque booléen (1326,) des combos utilisant au moins une des cartes."""
    return np.array(
        [bool(set(combo_cards(i)) & cards) for i in range(N_COMBOS)], dtype=bool
    )


def group_mask(spec: str) -> np.ndarray:
    """Masque booléen des combos d'une range texte (ex. 'AKs')."""
    return parse_range(spec).weights > 0.0


# Folds des trois positions serrées — la configuration canonique du bunching :
# BB défend contre un fold généralisé, son paquet résiduel est déformé.
TIGHT_FOLDERS: list[Range] = [
    fold_range_from_play(parse_range(GTO_PRESETS[p])) for p in ("UTG", "MP", "CO")
]


class TestFoldRangeFromPlay(unittest.TestCase):
    def test_complement_is_one_minus_weights(self) -> None:
        play = parse_range("AA, KK:0.4")
        fold = fold_range_from_play(play)
        np.testing.assert_allclose(fold.weights, 1.0 - play.weights, atol=1e-15)

    def test_mixed_frequency_folds_at_complement(self) -> None:
        """KK joué à 40 % → foldé à 60 % ; AA jamais foldé ; 72o toujours."""
        fold = fold_range_from_play(parse_range("AA, KK:0.4"))
        self.assertAlmostEqual(
            fold.weights[combo_index(c("As"), c("Ah"))], 0.0, places=12
        )
        self.assertAlmostEqual(
            fold.weights[combo_index(c("Ks"), c("Kh"))], 0.6, places=12
        )
        self.assertAlmostEqual(
            fold.weights[combo_index(c("7s"), c("2d"))], 1.0, places=12
        )

    def test_full_play_range_folds_nothing(self) -> None:
        self.assertEqual(fold_range_from_play(Range.full()).n_combos, 0.0)

    def test_rejects_non_range(self) -> None:
        with self.assertRaises(BunchingError):
            fold_range_from_play("AA")  # type: ignore[arg-type]


class TestDegenerateFolder(unittest.TestCase):
    """Folder dont le fold = 100 % le combo AsAd : vérité exacte à la main.

    P(As libre ET Ad libre) = 0 pour tout combo touchant As ou Ad (le folder
    tient TOUJOURS ces deux cartes), 1 pour tous les autres — dans les deux
    estimateurs, sans tolérance statistique (le tirage MC est déterministe :
    un seul combo possible).
    """

    def setUp(self) -> None:
        self.folder = single_combo_range("As", "Ad")
        self.target = Range.full()
        self.touch = combos_touching({c("As"), c("Ad")})

    def _check_exact(self, mult: np.ndarray) -> None:
        np.testing.assert_allclose(mult[self.touch], 0.0, atol=1e-12)
        np.testing.assert_allclose(mult[~self.touch], 1.0, atol=1e-12)
        # Sondes nominatives : le combo exact, un combo à une seule carte
        # bloquée, un combo indemne.
        self.assertAlmostEqual(mult[combo_index(c("As"), c("Ad"))], 0.0, places=12)
        self.assertAlmostEqual(mult[combo_index(c("As"), c("Kd"))], 0.0, places=12)
        self.assertAlmostEqual(mult[combo_index(c("Ah"), c("Ac"))], 1.0, places=12)

    def test_pairwise_exact_zeros_and_ones(self) -> None:
        self._check_exact(bunching_weights_pairwise(self.target, [self.folder], []))

    def test_mc_exact_zeros_and_ones(self) -> None:
        self._check_exact(
            bunching_weights_mc(self.target, [self.folder], [], n_sims=2_000, seed=7)
        )


class TestBiasDirection(unittest.TestCase):
    """Trois folds serrés (UTG/MP/CO) : petites cartes appauvries, AK enrichi.

    Les ranges de JEU serrées gardent les grosses cartes, donc leurs FOLDS
    regorgent de petites : la masse de fold par deuce dépasse la masse par
    as, et le facteur (1 − m(a) − m(b) + m(ab)) pénalise 22 plus que AKs.
    Ratio mesuré (multiplicateurs moyens par groupe) : pairwise ≈ 1.100,
    MC 20 000 sims ≈ 1.104.
    """

    def test_pairwise_ratio_aks_over_22(self) -> None:
        mult = bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS, [])
        ratio = mult[group_mask("AKs")].mean() / mult[group_mask("22")].mean()
        self.assertGreater(ratio, 1.05)

    def test_mc_ratio_aks_over_22(self) -> None:
        mult = bunching_weights_mc(
            Range.full(), TIGHT_FOLDERS, [], n_sims=20_000, seed=0
        )
        ratio = mult[group_mask("AKs")].mean() / mult[group_mask("22")].mean()
        self.assertGreater(ratio, 1.05)

    def test_small_pairs_depleted_versus_premiums(self) -> None:
        """Le biais est monotone au-delà du couple AKs/22 : tout le bas de
        grille (22-66) est en moyenne sous les premiums (mesuré ≈ 1.095)."""
        mult = bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS, [])
        low = mult[group_mask("22,33,44,55,66")].mean()
        big = mult[group_mask("AA,KK,AKs,AKo")].mean()
        self.assertGreater(big / low, 1.05)


class TestConvergence(unittest.TestCase):
    def test_mc_agrees_with_pairwise(self) -> None:
        """MC (50 000 sims, graine 0) vs pairwise sur 3 folders GTO.

        Valeurs mesurées (graine 0) : corrélation ≈ 0.995, écart max
        ≈ 0.011, écart moyen ≈ 0.003 — l'écart résiduel mélange le bruit MC
        (erreur type ~0.002 par combo) et les corrélations entre folders
        que pairwise néglige. Seuils : corrélation > 0.95 (exigence), écart
        max < 0.03 (3× la valeur mesurée).
        """
        pw = bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS, [])
        mc = bunching_weights_mc(
            Range.full(), TIGHT_FOLDERS, [], n_sims=50_000, seed=0
        )
        corr = float(np.corrcoef(mc, pw)[0, 1])
        self.assertGreater(corr, 0.95)
        self.assertLess(float(np.max(np.abs(mc - pw))), 0.03)

    def test_mc_is_deterministic_given_seed(self) -> None:
        a = bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], n_sims=3_000, seed=5)
        b = bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], n_sims=3_000, seed=5)
        np.testing.assert_array_equal(a, b)

    def test_multipliers_bounded_and_max_normalised(self) -> None:
        """Convention documentée : multiplicateurs dans [0,1], max = 1."""
        for mult in (
            bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS, []),
            bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], n_sims=2_000, seed=1),
        ):
            self.assertGreaterEqual(float(mult.min()), 0.0)
            self.assertEqual(float(mult.max()), 1.0)


class TestNoFolders(unittest.TestCase):
    def test_multipliers_all_exactly_one(self) -> None:
        ones = np.ones(N_COMBOS)
        np.testing.assert_array_equal(
            bunching_weights_pairwise(Range.full(), [], []), ones
        )
        np.testing.assert_array_equal(
            bunching_weights_mc(Range.full(), [], [], n_sims=10, seed=0), ones
        )

    def test_apply_bunching_is_identity(self) -> None:
        target = parse_range(GTO_PRESETS["BTN"])
        for method in ("pairwise", "mc"):
            out = apply_bunching(target, [], [], method=method)
            np.testing.assert_allclose(out.weights, target.weights, atol=1e-15)


class TestApplyBunching(unittest.TestCase):
    def test_weights_never_increase(self) -> None:
        """Multiplicateurs ≤ 1 (normalisation au max) : la masse est rognée."""
        target = parse_range(GTO_PRESETS["SB"])
        out = apply_bunching(target, TIGHT_FOLDERS, [], method="pairwise")
        self.assertTrue(np.all(out.weights <= target.weights + 1e-12))
        self.assertLess(out.n_combos, target.n_combos)

    def test_dead_card_combos_are_removed(self) -> None:
        """Les combos cibles utilisant une carte morte sortent à poids nul."""
        target = Range.full()
        dead = [c("As"), c("Kd"), c("7c")]
        out = apply_bunching(target, TIGHT_FOLDERS[:2], dead, method="pairwise")
        touch = combos_touching(set(dead))
        np.testing.assert_allclose(out.weights[touch], 0.0, atol=1e-15)
        self.assertGreater(float(out.weights[~touch].min()), 0.0)

    def test_mc_method_dispatch_and_determinism(self) -> None:
        target = parse_range(GTO_PRESETS["BTN"])
        out1 = apply_bunching(
            target, TIGHT_FOLDERS[:1], [], method="mc", n_sims=4_000, seed=3
        )
        out2 = apply_bunching(
            target, TIGHT_FOLDERS[:1], [], method="mc", n_sims=4_000, seed=3
        )
        self.assertIsInstance(out1, Range)
        np.testing.assert_array_equal(out1.weights, out2.weights)

    def test_pairwise_matches_manual_product(self) -> None:
        """apply = poids × multiplicateurs, puis retrait des cartes mortes."""
        target = parse_range(GTO_PRESETS["MP"])
        dead = [c("Qh"), c("2c")]
        mult = bunching_weights_pairwise(target, TIGHT_FOLDERS[:2], dead)
        expected = Range(
            np.clip(target.weights * mult, 0.0, 1.0)
        ).remove_blockers(dead)
        out = apply_bunching(target, TIGHT_FOLDERS[:2], dead, method="pairwise")
        np.testing.assert_allclose(out.weights, expected.weights, atol=1e-15)

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(BunchingError):
            apply_bunching(Range.full(), TIGHT_FOLDERS, [], method="exact")


class TestValidation(unittest.TestCase):
    def test_bad_n_sims_and_seed(self) -> None:
        with self.assertRaises(BunchingError):
            bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], n_sims=0)
        with self.assertRaises(BunchingError):
            bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], n_sims=2.5)  # type: ignore[arg-type]
        with self.assertRaises(BunchingError):
            bunching_weights_mc(Range.full(), TIGHT_FOLDERS, [], seed=-1)

    def test_bad_dead_cards(self) -> None:
        for dead in ([52], [-1], [c("As"), c("As")], [1.5]):
            with self.assertRaises(BunchingError):
                bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS, dead)

    def test_empty_fold_range_raises(self) -> None:
        with self.assertRaises(BunchingError):
            bunching_weights_pairwise(Range.full(), [Range.empty()], [])

    def test_fold_range_blocked_by_dead_cards_raises(self) -> None:
        """Folder = AsAd uniquement, mais As est mort : modèle incohérent."""
        folder = single_combo_range("As", "Ad")
        with self.assertRaises(BunchingError):
            bunching_weights_pairwise(Range.full(), [folder], [c("As")])

    def test_folder_ranges_must_be_ranges(self) -> None:
        with self.assertRaises(BunchingError):
            bunching_weights_pairwise(Range.full(), [0.5], [])  # type: ignore[list-item]
        with self.assertRaises(BunchingError):
            # Une Range seule au lieu d'une séquence de Ranges.
            bunching_weights_pairwise(Range.full(), TIGHT_FOLDERS[0], [])  # type: ignore[arg-type]

    def test_target_must_be_range(self) -> None:
        with self.assertRaises(BunchingError):
            bunching_weights_mc("AA", TIGHT_FOLDERS, [])  # type: ignore[arg-type]

    def test_mutually_blocking_folders_raise_in_mc(self) -> None:
        """Deux folders exigeant tous deux AsAd : aucune simulation cohérente.

        Le MC (modèle joint) détecte l'impossibilité ; pairwise, qui traite
        chaque folder isolément, ne le peut pas — divergence documentée.
        """
        folder = single_combo_range("As", "Ad")
        with self.assertRaises(BunchingError):
            bunching_weights_mc(
                Range.full(), [folder, folder], [], n_sims=500, seed=1
            )


if __name__ == "__main__":
    unittest.main()
