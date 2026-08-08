"""Tests L8 — isomorphisme de couleurs des boards (Waugh, 2013).

Valeurs de référence :
- 22 100 = C(52,3) flops bruts → 1 755 classes canoniques
  (Burnside : (22 100 + 6·2 938 + 8·299)/24 = 42 120/24 = 1 755).
- Turn (4 cartes) : 16 432 classes, même lemme.
- Multiplicités d'orbites comptées à la main : monotone AKQ → 4,
  bicolore AKQ → 12 (par motif de couleur), arc-en-ciel AKQ → 24,
  AAK → 12 + 12 (roi assorti à un as, ou non).
"""

from __future__ import annotations

import random
import unittest

from pfs.core.range_model import RANKS, SUITS
from pfs.solver.isomorphism import (
    IsoError,
    board_str,
    canonical_board,
    enumerate_canonical_flops,
    n_canonical,
    suit_mapping,
)


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def board(txt: str) -> list[int]:
    return [c(t) for t in txt.split()]


def apply_suits(cards: list[int], perm: list[int]) -> list[int]:
    return [4 * (x // 4) + perm[x % 4] for x in cards]


class TestCanonicalBoard(unittest.TestCase):
    def test_textures_distinctes_memes_rangs(self) -> None:
        """Monocolore / bicolore / arc-en-ciel : 3 classes distinctes pour AKQ."""
        mono = canonical_board(board("As Ks Qs"))
        deux = canonical_board(board("As Ks Qd"))
        arc = canonical_board(board("As Kd Qc"))
        self.assertNotEqual(mono, deux)
        self.assertNotEqual(mono, arc)
        self.assertNotEqual(deux, arc)
        # Représentants canoniques vérifiés à la main.
        self.assertEqual(board_str(mono), "As Ks Qs")
        self.assertEqual(board_str(deux), "As Ks Qh")
        self.assertEqual(board_str(arc), "As Kh Qd")

    def test_bicolore_le_motif_de_couleur_discrimine(self) -> None:
        """QUELS rangs partagent la couleur compte : A-K, A-Q et K-Q assortis
        sont trois classes distinctes (hiérarchie des tirages couleur)."""
        ak = canonical_board(board("As Ks Qd"))
        aq = canonical_board(board("As Kd Qs"))
        kq = canonical_board(board("Ad Ks Qs"))
        self.assertNotEqual(ak, aq)
        self.assertNotEqual(ak, kq)
        self.assertNotEqual(aq, kq)

    def test_invariance_permutation_couleurs_flop(self) -> None:
        """200 flops aléatoires : permuter les couleurs (et mélanger l'ordre)
        ne change pas la forme canonique."""
        rng = random.Random(42)
        for _ in range(200):
            flop = rng.sample(range(52), 3)
            perm = rng.sample(range(4), 4)
            image = apply_suits(flop, perm)
            rng.shuffle(image)
            self.assertEqual(canonical_board(image), canonical_board(flop))

    def test_ordre_des_cartes_indifferent(self) -> None:
        self.assertEqual(canonical_board(board("Qd Ks As")),
                         canonical_board(board("As Ks Qd")))

    def test_idempotence(self) -> None:
        """canonical_board(canonical_board(b)) == canonical_board(b)."""
        rng = random.Random(7)
        for k in (3, 4, 5):
            for _ in range(50):
                cb = canonical_board(rng.sample(range(52), k))
                self.assertEqual(canonical_board(cb), cb)

    def test_resultat_trie_croissant_dans_le_domaine(self) -> None:
        rng = random.Random(3)
        for k in (3, 4, 5):
            for _ in range(30):
                cb = canonical_board(rng.sample(range(52), k))
                self.assertEqual(len(cb), k)
                self.assertEqual(list(cb), sorted(set(cb)))
                self.assertTrue(all(0 <= x < 52 for x in cb))


class TestTurnRiver(unittest.TestCase):
    def test_canonical_sur_4_et_5_cartes(self) -> None:
        turn = canonical_board(board("As Ks Qd 2c"))
        river = canonical_board(board("As Ks Qd 2c 7h"))
        self.assertEqual(len(turn), 4)
        self.assertEqual(len(river), 5)

    def test_invariance_turn(self) -> None:
        rng = random.Random(1010)
        for _ in range(100):
            b4 = rng.sample(range(52), 4)
            perm = rng.sample(range(4), 4)
            image = apply_suits(b4, perm)
            rng.shuffle(image)
            self.assertEqual(canonical_board(image), canonical_board(b4))

    def test_invariance_river(self) -> None:
        rng = random.Random(2020)
        for _ in range(100):
            b5 = rng.sample(range(52), 5)
            perm = rng.sample(range(4), 4)
            image = apply_suits(b5, perm)
            rng.shuffle(image)
            self.assertEqual(canonical_board(image), canonical_board(b5))


class TestSuitMapping(unittest.TestCase):
    def test_mapping_envoie_sur_la_forme_canonique(self) -> None:
        rng = random.Random(11)
        for k in (3, 4, 5):
            for _ in range(50):
                b = rng.sample(range(52), k)
                m = suit_mapping(b)
                image = tuple(sorted(4 * (x // 4) + m[x % 4] for x in b))
                self.assertEqual(image, canonical_board(b))

    def test_mapping_est_une_bijection_des_couleurs(self) -> None:
        m = suit_mapping(board("As Ks Qd"))
        self.assertEqual(sorted(m), [0, 1, 2, 3])
        self.assertEqual(sorted(m.values()), [0, 1, 2, 3])

    def test_board_deja_canonique_mapping_identite(self) -> None:
        cb = canonical_board(board("Ah Kd Qc"))
        self.assertEqual(suit_mapping(cb), {0: 0, 1: 1, 2: 2, 3: 3})


class TestEnumerationFlops(unittest.TestCase):
    def test_1755_classes_multiplicites_22100(self) -> None:
        """L'oracle : 1 755 classes (valeur publiée), orbites couvrant C(52,3)."""
        classes = enumerate_canonical_flops()
        self.assertEqual(len(classes), 1755)
        self.assertEqual(sum(classes.values()), 22100)
        self.assertTrue(all(m >= 1 for m in classes.values()))

    def test_chaque_cle_est_son_propre_canonique(self) -> None:
        for b in enumerate_canonical_flops():
            self.assertEqual(canonical_board(b), b)

    def test_coherence_enumeration_et_canonical_board(self) -> None:
        """Le chemin vectorisé (clés int64) et le chemin pur Python coïncident."""
        classes = enumerate_canonical_flops()
        rng = random.Random(5)
        for _ in range(100):
            self.assertIn(canonical_board(rng.sample(range(52), 3)), classes)

    def test_multiplicites_goldens(self) -> None:
        """Orbites comptées à la main : 4, 12, 24, et les deux classes AAK."""
        classes = enumerate_canonical_flops()
        self.assertEqual(classes[canonical_board(board("As Ks Qs"))], 4)
        self.assertEqual(classes[canonical_board(board("As Ks Qd"))], 12)
        self.assertEqual(classes[canonical_board(board("Ah Kd Qc"))], 24)
        aak_assorti = canonical_board(board("As Ah Ks"))
        aak_sec = canonical_board(board("As Ah Kd"))
        self.assertNotEqual(aak_assorti, aak_sec)
        self.assertEqual(classes[aak_assorti], 12)
        self.assertEqual(classes[aak_sec], 12)

    def test_dictionnaire_neuf_a_chaque_appel(self) -> None:
        """Muter le dict retourné ne corrompt pas le cache interne."""
        d = enumerate_canonical_flops()
        d.clear()
        self.assertEqual(len(enumerate_canonical_flops()), 1755)


class TestNCanonical(unittest.TestCase):
    def test_flop_1755(self) -> None:
        self.assertEqual(n_canonical("flop"), 1755)

    def test_normalisation_de_la_chaine(self) -> None:
        self.assertEqual(n_canonical("  FLOP "), 1755)

    def test_turn_16432_burnside(self) -> None:
        """(C(52,4) + 6·19 253 + 3·325 + 8·884 + 6·13)/24 = 16 432."""
        self.assertEqual(n_canonical("turn"), 16432)

    def test_rue_inconnue(self) -> None:
        for mauvaise in ("preflop", "", "6th street"):
            with self.assertRaises(IsoError):
                n_canonical(mauvaise)


class TestValidation(unittest.TestCase):
    def test_carte_dupliquee(self) -> None:
        with self.assertRaises(IsoError):
            canonical_board([0, 0, 4])
        with self.assertRaises(IsoError):
            suit_mapping(board("As Ks As"))

    def test_carte_hors_domaine(self) -> None:
        with self.assertRaises(IsoError):
            canonical_board([0, 4, 52])
        with self.assertRaises(IsoError):
            canonical_board([-1, 4, 8])

    def test_taille_de_board_invalide(self) -> None:
        for mauvais in ([], [0], [0, 4], [0, 4, 8, 12, 16, 20]):
            with self.assertRaises(IsoError):
                canonical_board(mauvais)
        with self.assertRaises(IsoError):
            suit_mapping([0, 4])


if __name__ == "__main__":
    unittest.main()
