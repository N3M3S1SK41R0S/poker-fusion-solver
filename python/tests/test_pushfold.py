"""Tests du solveur push/fold Nash — jam/fold HU, chipEV et $EV ICM.

Valeurs de référence de la littérature jam/fold (Miltersen & Sørensen 2007 ;
charts HoldemResources) :

- 10 bb chipEV : la SB jamme ≈ 58 % des mains, la BB paie ≈ 37-40 % ;
- 5 bb : le jam dépasse 70 % ; la range s'élargit quand le tapis fond ;
- AA est toujours jam et toujours call ;
- une bulle ICM resserre la range de call du défenseur (effet canonique).

Tolérances larges (±3-4 pts de %) : la matrice d'équité est Monte-Carlo
(±0,9 pt d'erreur-type par paire à 3 000 simulations). Tout est semé, donc
chaque valeur testée est reproductible à l'identique.

NOTE : la première exécution construit la matrice 169×169 (≈ 4 min sur
2 cœurs, ≈ 7 min en séquentiel) puis la met en cache disque dans
``pfs/solver/`` — les exécutions suivantes durent quelques secondes.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import pfs.solver.pushfold as pf
from pfs.core.range_model import GROUP_COMBO_COUNT, N_COMBOS, N_GROUPS, group_name
from pfs.solver.pushfold import (
    PushFoldError,
    PushFoldSolution,
    chart,
    equity_matrix_169,
    solve_hu_pushfold,
)

# Indices de groupes utilisés partout (grille 13×13, ligne × 13 + colonne).
AA, AKS, AKO, KK, QQ, O32 = 0, 1, 13, 14, 28, 167

_EQ: np.ndarray | None = None


def setUpModule() -> None:
    """Construit (1re fois) ou charge la matrice d'équité partagée."""
    global _EQ
    _EQ = equity_matrix_169()


def _solve(stack: float, **kw) -> PushFoldSolution:
    """Raccourci : résout avec la matrice partagée du module."""
    return solve_hu_pushfold(stack, equity=_EQ, **kw)


class TestEquityMatrix(unittest.TestCase):
    """Qualité de la matrice 169×169 : bornes, symétrie, goldens publiés."""

    def test_forme_et_bornes(self) -> None:
        self.assertEqual(_EQ.shape, (N_GROUPS, N_GROUPS))
        self.assertTrue(np.all(np.isfinite(_EQ)))
        self.assertGreaterEqual(float(_EQ.min()), 0.0)
        self.assertLessEqual(float(_EQ.max()), 1.0)

    def test_antisymetrie_structurelle(self) -> None:
        """Hors diagonale, E[i,j] + E[j,i] = 1 exactement (par construction,
        et c'est une égalité vraie par symétrie des couleurs)."""
        s = _EQ + _EQ.T
        off = ~np.eye(N_GROUPS, dtype=bool)
        self.assertLess(float(np.abs(s[off] - 1.0).max()), 1e-12)

    def test_symetrie_recalcul_independant(self) -> None:
        """Le triangle inférieur (complété par 1 − E[j,i]) est confronté à un
        recalcul Monte-Carlo DIRECT : les graines ``i·169+j`` avec i > j
        n'ont jamais servi au build, l'estimation est donc indépendante.
        Attendu : |e_direct(i,j) + E[j,i] − 1| ≤ 0.03 (deux bruits MC à
        ±0,9 pt chacun)."""
        rng = np.random.default_rng(42)
        devs = []
        for _ in range(40):
            i = int(rng.integers(1, N_GROUPS))
            j = int(rng.integers(0, i))          # i > j : triangle inférieur
            e_direct = pf._pair_equity(i, j, 3000)
            devs.append(abs(e_direct + float(_EQ[j, i]) - 1.0))
        devs_arr = np.array(devs)
        self.assertLess(float(devs_arr.max()), 0.03)
        self.assertLess(float(devs_arr.mean()), 0.02)

    def test_diagonale_miroir(self) -> None:
        """Groupe contre lui-même : équité exacte 1/2 (symétrie), MC ±0,03."""
        diag = np.diag(_EQ)
        self.assertLess(float(np.abs(diag - 0.5).max()), 0.03)

    def test_aa_contre_range_complete(self) -> None:
        """AA vs main aléatoire ≈ 85 % (valeur publiée), pondérée par le
        card-removal exact des groupes."""
        counts = pf._unblocked_counts()
        aa_full = float((counts[AA] * _EQ[AA]).sum() / counts[AA].sum())
        self.assertAlmostEqual(aa_full, 0.85, delta=0.02)

    def test_goldens_publies(self) -> None:
        """Matchups classiques (mêmes références que test_equity)."""
        self.assertAlmostEqual(float(_EQ[AA, KK]), 0.8255, delta=0.03)
        self.assertAlmostEqual(float(_EQ[AKS, QQ]), 0.46, delta=0.03)
        self.assertAlmostEqual(float(_EQ[AKO, QQ]), 0.433, delta=0.03)
        # sanity d'indexation : les noms correspondent bien
        self.assertEqual(group_name(AKS), "AKs")
        self.assertEqual(group_name(AKO), "AKo")
        self.assertEqual(group_name(O32), "32o")

    def test_cache_disque(self) -> None:
        """Relire le cache redonne exactement la même matrice."""
        again = equity_matrix_169()
        self.assertTrue(np.array_equal(_EQ, again))
        hits = list(Path(pf.__file__).parent.glob("_pushfold_equity169_v*.npy"))
        self.assertTrue(hits, "le cache disque .npy doit exister")

    def test_n_sims_trop_faible(self) -> None:
        with self.assertRaises(PushFoldError):
            equity_matrix_169(n_sims=50)


class TestPushFoldChipEV(unittest.TestCase):
    """Équilibres chipEV HU contre les références de la littérature."""

    def test_10bb_reference(self) -> None:
        """10 bb : jam ≈ 58 % (référence HRC), call ≈ 37-40 %."""
        sol = _solve(10.0)
        self.assertGreaterEqual(sol.jam_pct, 0.45)
        self.assertLessEqual(sol.jam_pct, 0.68)
        self.assertGreaterEqual(sol.call_pct, 0.30)
        self.assertLessEqual(sol.call_pct, 0.50)

    def test_5bb_jam_large(self) -> None:
        """5 bb : la SB jamme au moins 70 % des mains."""
        self.assertGreaterEqual(_solve(5.0).jam_pct, 0.70)

    def test_monotonie_en_stack(self) -> None:
        """Le jam s'élargit strictement quand le tapis raccourcit."""
        j5, j10, j20 = (_solve(s).jam_pct for s in (5.0, 10.0, 20.0))
        self.assertGreater(j5, j10)
        self.assertGreater(j10, j20)

    def test_aa_toujours_jam_et_call(self) -> None:
        """AA est jam ET call à fréquence 1, à tous les tapis ≤ 20 bb."""
        for s in (5.0, 10.0, 20.0):
            sol = _solve(s)
            self.assertAlmostEqual(float(sol.jam_range[AA]), 1.0, places=9)
            self.assertAlmostEqual(float(sol.call_range[AA]), 1.0, places=9)

    def test_32o_fold_a_10bb(self) -> None:
        sol = _solve(10.0)
        self.assertAlmostEqual(float(sol.jam_range[O32]), 0.0, places=9)

    def test_convergence(self) -> None:
        for s in (5.0, 10.0, 20.0):
            sol = _solve(s)
            self.assertTrue(sol.converged)
            self.assertLess(sol.n_iter, 200)

    def test_determinisme(self) -> None:
        """Deux résolutions identiques : graines fixes, zéro aléa résiduel."""
        a, b = _solve(10.0), _solve(10.0)
        self.assertTrue(np.array_equal(a.jam_range, b.jam_range))
        self.assertTrue(np.array_equal(a.call_range, b.call_range))
        self.assertTrue(np.array_equal(a.ev_jam_par_groupe, b.ev_jam_par_groupe))
        self.assertEqual(a.n_iter, b.n_iter)

    def test_ev_jam_coherent_avec_la_range(self) -> None:
        """Équilibre pur certifié ⇒ jam ssi EV(jam) − EV(fold) > 0."""
        sol = _solve(10.0)
        self.assertGreater(float(sol.ev_jam_par_groupe[AA]), 0.0)
        self.assertTrue(np.array_equal(
            sol.jam_range, (sol.ev_jam_par_groupe > 0.0).astype(np.float64)))

    def test_ante_elargit_le_jam(self) -> None:
        """Plus de mort au milieu ⇒ jam plus large (10 bb, ante 12,5 %)."""
        self.assertGreater(_solve(10.0, ante=0.125).jam_pct,
                           _solve(10.0).jam_pct + 0.01)

    def test_pct_ponderes_par_combos(self) -> None:
        """jam_pct doit être la moyenne des fréquences pondérée en combos."""
        sol = _solve(10.0)
        attendu = float((sol.jam_range * GROUP_COMBO_COUNT).sum() / N_COMBOS)
        self.assertAlmostEqual(sol.jam_pct, attendu, places=12)


class TestPushFoldIcm(unittest.TestCase):
    """$EV ICM : effet bulle, cohérence winner-take-all, dégénérescences."""

    PAYOUTS = (50.0, 30.0, 20.0)
    STACKS = (30.0, 10.0, 10.0, 10.0)   # BB chip-leader (0), héros SB mid (1)

    def test_bulle_resserre_le_call(self) -> None:
        """Bulle 4 joueurs / 3 payés, héros mid-stack couvert par le
        chip-leader : la range de call ICM est nettement plus serrée qu'en
        chipEV (l'effet bulle canonique) — et le jam se resserre aussi."""
        chip = _solve(10.0)
        icm = _solve(10.0, payouts=self.PAYOUTS, stacks=self.STACKS,
                     hero=1, villain=0)
        self.assertLess(icm.call_pct, chip.call_pct - 0.05)
        self.assertLess(icm.jam_pct, chip.jam_pct)
        self.assertTrue(icm.converged)
        self.assertLess(icm.n_iter, 200)

    def test_aa_reste_jam_et_call_sous_icm(self) -> None:
        icm = _solve(10.0, payouts=self.PAYOUTS, stacks=self.STACKS,
                     hero=1, villain=0)
        self.assertAlmostEqual(float(icm.jam_range[AA]), 1.0, places=9)
        self.assertAlmostEqual(float(icm.call_range[AA]), 1.0, places=9)

    def test_winner_take_all_hu_egal_chipev(self) -> None:
        """HU winner-take-all : le $EV est affine en jetons, l'équilibre ICM
        doit coïncider EXACTEMENT avec l'équilibre chipEV."""
        chip = _solve(10.0)
        wta = _solve(10.0, payouts=[1.0], stacks=[10.0, 10.0])
        self.assertTrue(np.array_equal(chip.jam_range, wta.jam_range))
        self.assertTrue(np.array_equal(chip.call_range, wta.call_range))

    def test_payouts_plats_jeu_trivial(self) -> None:
        """Payouts identiques ⇒ aucun enjeu : personne ne jamme ni ne paie
        (EV strictement égaux → fold par défaut), convergence immédiate."""
        sol = _solve(10.0, payouts=[50.0, 50.0], stacks=[10.0, 10.0])
        self.assertTrue(sol.converged)
        self.assertAlmostEqual(sol.jam_pct, 0.0, places=9)

    def test_determinisme_icm(self) -> None:
        a = _solve(10.0, payouts=self.PAYOUTS, stacks=self.STACKS,
                   hero=1, villain=0)
        b = _solve(10.0, payouts=self.PAYOUTS, stacks=self.STACKS,
                   hero=1, villain=0)
        self.assertTrue(np.array_equal(a.jam_range, b.jam_range))
        self.assertTrue(np.array_equal(a.call_range, b.call_range))
        self.assertEqual(a.n_iter, b.n_iter)


class TestChart(unittest.TestCase):
    def test_structure_13x13(self) -> None:
        sol = _solve(10.0)
        lignes = chart(sol).splitlines()
        self.assertEqual(len(lignes), 15)            # en-tête + 13 + résumé
        self.assertTrue(lignes[0].lstrip().startswith("A"))
        for r, ligne in enumerate(lignes[1:14]):
            cellules = ligne.split()
            self.assertEqual(len(cellules), 14)      # étiquette + 13 cases
            self.assertEqual(cellules[0], "AKQJT98765432"[r])
            self.assertTrue(set(cellules[1:]) <= {"+", "."})

    def test_aa_marque_plus_et_dechets_point(self) -> None:
        sol = _solve(10.0)
        lignes = chart(sol).splitlines()
        self.assertEqual(lignes[1].split()[1], "+")          # case AA
        self.assertEqual(lignes[13].split()[12], ".")        # case 32o
        self.assertIn("58", lignes[14])                      # résumé ~58.x %

    def test_seuil_invalide(self) -> None:
        sol = _solve(10.0)
        with self.assertRaises(PushFoldError):
            chart(sol, threshold=0.0)
        with self.assertRaises(PushFoldError):
            chart(sol, threshold=1.5)

    def test_range_mal_dimensionnee(self) -> None:
        mauvais = PushFoldSolution(np.ones(5), np.ones(5), 1.0, 1.0,
                                   np.zeros(5), 1, True)
        with self.assertRaises(PushFoldError):
            chart(mauvais)


class TestValidation(unittest.TestCase):
    """Chaque paramétrage incohérent doit lever PushFoldError, pas planter."""

    def test_stack_invalide(self) -> None:
        for bad in (0.0, -3.0, float("nan")):
            with self.assertRaises(PushFoldError):
                _solve(bad)

    def test_blindes_invalides(self) -> None:
        with self.assertRaises(PushFoldError):
            _solve(10.0, bb=0.0)
        with self.assertRaises(PushFoldError):
            _solve(10.0, sb=-0.5)
        with self.assertRaises(PushFoldError):
            _solve(10.0, ante=-0.1)

    def test_iterations_et_tolerance(self) -> None:
        with self.assertRaises(PushFoldError):
            _solve(10.0, max_iter=0)
        with self.assertRaises(PushFoldError):
            _solve(10.0, tol=0.0)

    def test_stacks_et_payouts_incoherents(self) -> None:
        with self.assertRaises(PushFoldError):
            _solve(10.0, stacks=[10.0, 10.0])            # stacks sans payouts
        with self.assertRaises(PushFoldError):
            _solve(10.0, payouts=[50.0, 30.0])           # payouts sans stacks
        with self.assertRaises(PushFoldError):
            _solve(10.0, payouts=[50.0, 30.0], stacks=[12.0, 10.0])  # héros ≠ 12
        with self.assertRaises(PushFoldError):
            _solve(10.0, payouts=[50.0, 30.0], stacks=[10.0, 10.0],
                   hero=0, villain=0)
        with self.assertRaises(PushFoldError):
            _solve(10.0, payouts=[30.0, 50.0], stacks=[10.0, 10.0])  # croissants

    def test_tapis_effectif_degenere(self) -> None:
        """Tapis ≤ bb + ante : la BB est engagée d'office, jeu dégénéré."""
        with self.assertRaises(PushFoldError):
            _solve(0.8)

    def test_matrice_equite_invalide(self) -> None:
        with self.assertRaises(PushFoldError):
            solve_hu_pushfold(10.0, equity=np.zeros((5, 5)))
        with self.assertRaises(PushFoldError):
            solve_hu_pushfold(10.0, equity=np.full((N_GROUPS, N_GROUPS), 2.0))


if __name__ == "__main__":
    unittest.main()
