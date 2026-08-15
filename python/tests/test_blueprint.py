"""Tests — infrastructure du blueprint flop (``pfs.solver.blueprint``).

Ce que ces tests prouvent :

- le DÉNOMBREMENT posé dans la docstring du module : 1 755 classes, somme
  des poids = C(52,3) = 22 100 côté flops bruts, et C(50,3) = 19 600 flops
  desservis pour toute main fixée (les mêmes 1 755 classes suffisent) ;
- le MAGASIN : sauve/recharge bit à bit, écriture atomique, canonisation
  des clés (sauver sous une image de l'orbite marque LA classe) ;
- le MANIFESTE : reprise après interruption (readonly du disque : une
  entrée sans fichier redevient à faire), invalidation propre par réglage ;
- la REQUÊTE : aller-retour flop réel ↔ canonique ↔ stratégie re-projetée,
  vérifié sur des flops qui diffèrent d'une permutation de couleurs, y
  compris contre un calcul direct d'EHS sur le flop réel ;
- le PIPELINE : classe → buckets EHS → entrée du solveur, sur une classe.
"""

from __future__ import annotations

import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pfs.core.range_model import (
    N_COMBOS,
    RANKS,
    SUITS,
    Range,
    combo_cards,
    combo_index,
    parse_range,
)
from pfs.solver.abstraction import expected_hand_strength
from pfs.solver.blueprint import (
    BlueprintError,
    BlueprintSettings,
    BlueprintStore,
    N_FLOP_CLASSES,
    TOTAL_FLOPS,
    bucketed_inputs,
    class_key,
    combo_permutation,
    enumerate_classes,
    query_flop,
)
from pfs.solver.isomorphism import canonical_board, suit_mapping


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def board(txt: str) -> list[int]:
    return [c(t) for t in txt.split()]


def apply_suits(cards: list[int], perm: dict[int, int]) -> list[int]:
    return [4 * (x // 4) + perm[x % 4] for x in cards]


SETTINGS = BlueprintSettings(
    iterations=10, n_buckets=4, method="percentile",
    n_rollouts=128, seed=0, solver_version="test-v0",
)

FLOP = board("Ks 7d 2c")
PERM = {0: 2, 1: 0, 2: 3, 3: 1}          # une permutation de couleurs quelconque
FLOP_IMAGE = apply_suits(FLOP, PERM)      # même classe, autres couleurs


class TestDenombrement(unittest.TestCase):
    def test_1755_classes_poids_22100(self) -> None:
        """Côté flops bruts : Σ poids = C(52,3) = 22 100, jamais codé en dur."""
        classes = enumerate_classes()
        self.assertEqual(len(classes), N_FLOP_CLASSES)
        self.assertEqual(sum(k.weight for k in classes), TOTAL_FLOPS)
        self.assertEqual(TOTAL_FLOPS, math.comb(52, 3))
        self.assertTrue(all(k.weight >= 1 for k in classes))

    def test_representants_canoniques_et_cles_uniques(self) -> None:
        classes = enumerate_classes()
        keys = {k.key for k in classes}
        self.assertEqual(len(keys), N_FLOP_CLASSES)
        for k in classes[::97]:               # échantillon régulier
            self.assertEqual(canonical_board(k.board), k.board)
            self.assertEqual(class_key(k.board), k.key)

    def test_ordre_deterministe(self) -> None:
        boards = [k.board for k in enumerate_classes()]
        self.assertEqual(boards, sorted(boards))

    def test_cote_mains_fixees_19600(self) -> None:
        """Pour une main fixée, C(50,3) = 19 600 flops compatibles, TOUS
        desservis par les 1 755 classes : Σ_classes |orbite ∩ compatibles|
        = 19 600 — le dénombrement « côté mains fixées » de la docstring."""
        hand = {c("As"), c("Kd")}
        valid = {k.board for k in enumerate_classes()}
        per_class: dict[tuple[int, int, int], int] = {}
        n = 0
        for flop in itertools.combinations(range(52), 3):
            if hand & set(flop):
                continue
            n += 1
            canon = canonical_board(flop)
            self.assertIn(canon, valid)
            per_class[canon] = per_class.get(canon, 0) + 1
        self.assertEqual(n, math.comb(50, 3))
        self.assertEqual(sum(per_class.values()), 19600)

    def test_cle_flop_seulement(self) -> None:
        with self.assertRaises(BlueprintError):
            class_key(board("As Ks Qs 2c"))


class TestReglages(unittest.TestCase):
    def test_empreinte_stable_et_discriminante(self) -> None:
        a = BlueprintSettings(iterations=100, n_buckets=8)
        b = BlueprintSettings(iterations=100, n_buckets=8)
        c_ = BlueprintSettings(iterations=101, n_buckets=8)
        self.assertEqual(a.digest(), b.digest())
        self.assertNotEqual(a.digest(), c_.digest())

    def test_aller_retour_dict(self) -> None:
        s = BlueprintSettings(iterations=7, n_buckets=3, method="kmeans",
                              n_rollouts=96, seed=5, solver_version="x-v2")
        self.assertEqual(BlueprintSettings.from_dict(s.to_dict()), s)

    def test_validations(self) -> None:
        for bad in (dict(iterations=0, n_buckets=8),
                    dict(iterations=True, n_buckets=8),
                    dict(iterations=10, n_buckets=0),
                    dict(iterations=10, n_buckets=8, method="foo"),
                    dict(iterations=10, n_buckets=8, n_rollouts=0),
                    dict(iterations=10, n_buckets=8, seed="x"),
                    dict(iterations=10, n_buckets=8, solver_version="")):
            with self.assertRaises(BlueprintError):
                BlueprintSettings(**bad)


class _AvecMagasin(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pfs-blueprint-")
        self.root = Path(self._tmp.name)
        self.store = BlueprintStore(self.root, SETTINGS)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestMagasin(_AvecMagasin):
    def _solution(self) -> dict[str, np.ndarray]:
        gen = np.random.default_rng(3)
        strat = gen.random((2, N_COMBOS), dtype=np.float64).astype(np.float32)
        return {
            "strat/racine": strat,
            "labels": np.array(["check", "bet 75%"]),
            "ev": np.array([1.25]),
        }

    def test_sauve_recharge_identique(self) -> None:
        arrays = self._solution()
        self.store.save_solution(FLOP, arrays, combo_keys=("strat/racine",),
                                 meta={"exploitability": 0.012},
                                 elapsed_s=1.5)
        sol = self.store.load_solution(FLOP)
        self.assertEqual(sol.board, canonical_board(FLOP))
        self.assertEqual(sol.combo_keys, ("strat/racine",))
        self.assertEqual(sol.meta, {"exploitability": 0.012})
        for name, arr in arrays.items():
            self.assertEqual(sol.arrays[name].dtype, arr.dtype, name)
            self.assertTrue(np.array_equal(sol.arrays[name], arr), name)

    def test_sauver_sous_une_image_marque_la_classe(self) -> None:
        self.store.save_solution(FLOP_IMAGE, self._solution())
        self.assertTrue(self.store.is_done(FLOP))
        self.assertTrue(self.store.is_done(canonical_board(FLOP)))
        sol = self.store.load_solution(FLOP)          # via n'importe quelle image
        self.assertEqual(sol.board, canonical_board(FLOP))

    def test_progression_ponderee(self) -> None:
        self.store.save_solution(board("As Ks Qs"), self._solution())   # orbite 4
        p = self.store.progress()
        self.assertEqual(p["n_done"], 1)
        self.assertEqual(p["weight_done"], 4)
        self.assertEqual(p["weight_total"], TOTAL_FLOPS)
        self.assertEqual(len(self.store.pending()), N_FLOP_CLASSES - 1)

    def test_charger_classe_absente(self) -> None:
        with self.assertRaises(BlueprintError):
            self.store.load_solution(FLOP)

    def test_validations_de_sauvegarde(self) -> None:
        with self.assertRaises(BlueprintError):
            self.store.save_solution(FLOP, {})
        with self.assertRaises(BlueprintError):
            self.store.save_solution(FLOP, {"_prive": np.zeros(3)})
        with self.assertRaises(BlueprintError):
            self.store.save_solution(
                FLOP, {"s": np.zeros((2, 10))}, combo_keys=("s",))
        with self.assertRaises(BlueprintError):
            self.store.save_solution(
                FLOP, {"s": np.zeros((2, N_COMBOS))}, combo_keys=("absent",))
        with self.assertRaises(BlueprintError):
            self.store.save_solution(FLOP, {"s": np.zeros(3)},
                                     meta={"f": object()})


class TestManifesteReprise(_AvecMagasin):
    def _sauver(self, store: BlueprintStore, flop_txt: str) -> None:
        store.save_solution(board(flop_txt),
                            {"strat": np.zeros((1, N_COMBOS))},
                            combo_keys=("strat",))

    def test_reprise_apres_interruption(self) -> None:
        """Deux classes faites, « crash » (l'objet meurt), relance : le
        manifeste sait ce qui est fait, pending() liste le reste."""
        self._sauver(self.store, "Ks 7d 2c")
        self._sauver(self.store, "As Ks Qs")
        del self.store
        repris = BlueprintStore(self.root, SETTINGS)      # même réglage
        self.assertEqual(len(repris.done_keys()), 2)
        self.assertTrue(repris.is_done(board("Ks 7d 2c")))
        self.assertEqual(len(repris.pending()), N_FLOP_CLASSES - 2)

    def test_fichier_disparu_classe_redevient_a_faire(self) -> None:
        """Interruption simulée au pire moment : manifeste écrit, fichier
        détruit ensuite. La classe redevient À FAIRE (le disque prime)."""
        self._sauver(self.store, "Ks 7d 2c")
        key = class_key(board("Ks 7d 2c"))
        (self.store.directory / "classes" / f"{key}.npz").unlink()
        repris = BlueprintStore(self.root, SETTINGS)
        self.assertFalse(repris.is_done(board("Ks 7d 2c")))
        self.assertEqual(len(repris.pending()), N_FLOP_CLASSES)

    def test_reglage_different_invalide_proprement(self) -> None:
        """Un autre réglage n'hérite de RIEN — et ne détruit rien."""
        self._sauver(self.store, "Ks 7d 2c")
        autres = BlueprintSettings(
            iterations=SETTINGS.iterations + 1, n_buckets=SETTINGS.n_buckets,
            method=SETTINGS.method, n_rollouts=SETTINGS.n_rollouts,
            seed=SETTINGS.seed, solver_version=SETTINGS.solver_version,
        )
        neuf = BlueprintStore(self.root, autres)
        self.assertEqual(len(neuf.done_keys()), 0)
        self.assertNotEqual(neuf.directory, self.store.directory)
        # l'ancien calcul est intact
        self.assertTrue(BlueprintStore(self.root, SETTINGS).is_done(
            board("Ks 7d 2c")))

    def test_manifeste_contradictoire_refuse(self) -> None:
        """Réglages édités à la main dans le manifeste → ouverture refusée."""
        path = self.store.directory / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["settings"]["iterations"] = 999
        path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(BlueprintError):
            BlueprintStore(self.root, SETTINGS)


class TestRequeteReprojetee(_AvecMagasin):
    def test_flop_canonique_permutation_identite(self) -> None:
        canon = canonical_board(FLOP)
        strat = np.arange(N_COMBOS, dtype=np.float64)[None, :]
        self.store.save_solution(canon, {"strat": strat}, combo_keys=("strat",))
        q = query_flop(self.store, canon)
        self.assertEqual(q.mapping, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertTrue(np.array_equal(q.arrays["strat"], strat))

    def test_aller_retour_par_permutation_de_couleurs(self) -> None:
        """La solution encode l'index canonique de chaque combo ; interrogée
        depuis une image du flop, la valeur du combo réel i doit être
        combo_index(σ(i)) où σ = suit_mapping(flop réel)."""
        strat = np.arange(N_COMBOS, dtype=np.float64)[None, :]
        self.store.save_solution(FLOP, {"strat": strat}, combo_keys=("strat",))
        q = query_flop(self.store, FLOP_IMAGE)
        self.assertEqual(q.canonical, canonical_board(FLOP))
        self.assertEqual(tuple(sorted(FLOP_IMAGE)), q.board)
        m = suit_mapping(FLOP_IMAGE)
        for i in (0, 17, 421, 900, N_COMBOS - 1):
            a, b = combo_cards(i)
            j = combo_index(4 * (a // 4) + m[a % 4], 4 * (b // 4) + m[b % 4])
            self.assertEqual(float(q.arrays["strat"][0, i]), float(j))

    def test_reprojection_egale_calcul_direct_ehs(self) -> None:
        """Le test sémantique de bout en bout : une « stratégie » = l'EHS des
        1 326 combos, stockée pour la classe. Interrogée depuis une image du
        flop, elle doit coïncider BIT À BIT avec l'EHS calculée directement
        sur le flop réel (l'invariance de Waugh, traversée par le magasin)."""
        canon = canonical_board(FLOP)

        def ehs_complet(b: list[int] | tuple[int, ...]) -> np.ndarray:
            dead = set(b)
            live = [i for i in range(N_COMBOS)
                    if not set(combo_cards(i)) & dead]
            cards = np.array([combo_cards(i) for i in live], dtype=np.int64)
            vals = expected_hand_strength(cards, b, n_rollouts=128, seed=0)
            full = np.full(N_COMBOS, np.nan)
            full[live] = vals
            return full

        self.store.save_solution(canon, {"ehs": ehs_complet(canon)},
                                 combo_keys=("ehs",))
        q = query_flop(self.store, FLOP_IMAGE)
        direct = ehs_complet(FLOP_IMAGE)
        self.assertTrue(np.array_equal(q.arrays["ehs"], direct, equal_nan=True))

    def test_tableaux_non_combo_inchanges(self) -> None:
        arrays = {"strat": np.zeros((1, N_COMBOS)),
                  "ev": np.array([3.14, 2.71])}
        self.store.save_solution(FLOP, arrays, combo_keys=("strat",))
        q = query_flop(self.store, FLOP_IMAGE)
        self.assertTrue(np.array_equal(q.arrays["ev"], arrays["ev"]))

    def test_poids_de_classe_expose(self) -> None:
        self.store.save_solution(board("As Ks Qs"),
                                 {"s": np.zeros(N_COMBOS)}, combo_keys=("s",))
        q = query_flop(self.store, board("Ah Kh Qh"))
        self.assertEqual(q.weight, 4)                     # orbite monotone AKQ

    def test_permutation_combos_bijection(self) -> None:
        perm = combo_permutation(PERM)
        self.assertEqual(perm.shape, (N_COMBOS,))
        self.assertEqual(len(set(perm.tolist())), N_COMBOS)
        ident = combo_permutation({0: 0, 1: 1, 2: 2, 3: 3})
        self.assertTrue(np.array_equal(ident, np.arange(N_COMBOS)))
        with self.assertRaises(BlueprintError):
            combo_permutation({0: 0, 1: 1, 2: 2, 3: 2})


class TestPipelineAbstraction(unittest.TestCase):
    """Classe → buckets EHS (abstraction.py) → entrée du solveur, sur UNE classe."""

    OOP = "88+, ATs+, KQs, AQo+"
    IP = "55+, A2s+, KTs+, QTs+, JTs, T9s, 98s, ATo+, KQo"

    def _entrees(self, flop: list[int] | tuple[int, ...]):
        return bucketed_inputs(flop, parse_range(self.OOP),
                               parse_range(self.IP), SETTINGS)

    def test_entree_complete_et_coherente(self) -> None:
        e = self._entrees(FLOP)
        self.assertEqual(e.canonical, canonical_board(FLOP))
        self.assertEqual(e.n_buckets, SETTINGS.n_buckets)
        for p in (e.oop, e.ip):
            n = p.cards.shape[0]
            self.assertGreater(n, 0)
            self.assertEqual(p.cards.shape, (n, 2))
            for arr in (p.combo_indices, p.weights, p.ehs, p.buckets):
                self.assertEqual(arr.shape, (n,))
            self.assertTrue(np.all((p.ehs >= 0.0) & (p.ehs <= 1.0)))
            self.assertTrue(np.all((p.buckets >= 0)
                                   & (p.buckets < SETTINGS.n_buckets)))
            self.assertTrue(np.all(p.weights > 0.0))
            # les cartes correspondent bien aux index de combo
            for k in range(0, n, max(1, n // 7)):
                self.assertEqual(tuple(p.cards[k]),
                                 combo_cards(int(p.combo_indices[k])))

    def test_buckets_croissants_avec_la_force(self) -> None:
        e = self._entrees(FLOP)
        for p in (e.oop, e.ip):
            ordre = np.argsort(p.ehs, kind="stable")
            self.assertTrue(np.all(np.diff(p.buckets[ordre]) >= 0))
            self.assertEqual(len(set(p.buckets.tolist())), SETTINGS.n_buckets)

    def test_invariance_par_permutation_de_couleurs(self) -> None:
        """Les ranges par groupe sont invariantes de couleur : sur l'image du
        flop, chaque combo réel i porte l'EHS et le bucket du combo σ(i) sur
        le flop CANONIQUE — le pipeline entier respecte l'isomorphisme."""
        base = self._entrees(canonical_board(FLOP))
        image = self._entrees(FLOP_IMAGE)
        m = suit_mapping(FLOP_IMAGE)
        perm = combo_permutation(m)
        ehs_base = dict(zip(base.oop.combo_indices.tolist(),
                            base.oop.ehs.tolist()))
        bkt_base = dict(zip(base.oop.combo_indices.tolist(),
                            base.oop.buckets.tolist()))
        for i, ehs_i, bkt_i in zip(image.oop.combo_indices.tolist(),
                                   image.oop.ehs.tolist(),
                                   image.oop.buckets.tolist()):
            j = int(perm[i])
            self.assertAlmostEqual(ehs_i, ehs_base[j], places=15)
            self.assertEqual(bkt_i, bkt_base[j])

    def test_erreurs(self) -> None:
        with self.assertRaises(BlueprintError):
            bucketed_inputs(board("Ks 7d 2c 9h"), parse_range("AA"),
                            parse_range("KK"), SETTINGS)
        with self.assertRaises(BlueprintError):
            bucketed_inputs(FLOP, np.ones(N_COMBOS),        # pas une Range
                            parse_range("KK"), SETTINGS)
        with self.assertRaises(BlueprintError):
            bucketed_inputs(board("As Ah Ad"), parse_range("AA"),
                            parse_range("KK"), SETTINGS)    # range OOP vidée
        with self.assertRaises(BlueprintError):
            bucketed_inputs(FLOP, parse_range("AA"),
                            parse_range("KK"), settings="x")


if __name__ == "__main__":
    unittest.main()
