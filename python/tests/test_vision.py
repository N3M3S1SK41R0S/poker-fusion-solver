"""Tests du recogniseur de cartes — pHash sur le deck PMU étiqueté.

Propriétés verrouillées :
  1. déterminisme et auto-reconnaissance (chaque gabarit se reconnaît) ;
  2. DISCRIMINATION — les 52 cartes sont mutuellement séparées d'une large
     marge (le point qui a motivé le passage en 256 bits : pique vs trèfle
     de même rang ne tenait qu'à 2 bits en 64 bits) ;
  3. ROBUSTESSE À L'ÉCHELLE — une carte agrandie (comme à l'écran) reste
     correctement reconnue : c'est tout l'intérêt du pHash ;
  4. robustesse au bruit ;
  5. reconnaissance multi-cartes par régions d'intérêt ;
  6. rejet d'une image qui n'est pas une carte (confiance insuffisante).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pfs.vision import build_templates, identify_card, phash, recognize_cards
from pfs.vision.card_recognizer import _DECK_DIR, load_templates
from pfs.vision.phash import HASH_BITS, hamming


def _sprite(card: str) -> Image.Image:
    return Image.open(_DECK_DIR / f"{card}.png").convert("RGBA")


class TestPhash(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = np.zeros((20, 15), dtype=np.uint8)
        a[5:15, 5:10] = 255
        self.assertEqual(phash(a), phash(a))

    def test_self_distance_zero(self) -> None:
        h = phash(_sprite("Ah"))
        self.assertEqual(hamming(h, h), 0)

    def test_hash_width(self) -> None:
        # 256 bits attendus (bloc 16×16)
        self.assertEqual(HASH_BITS, 256)


class TestDeckIntegrity(unittest.TestCase):
    def test_fifty_two_distinct_cards(self) -> None:
        tpl = load_templates()
        self.assertEqual(len(tpl), 52)
        self.assertEqual(len(set(tpl.values())), 52)   # aucune signature dupliquée

    def test_all_thirteen_ranks_four_suits(self) -> None:
        tpl = load_templates()
        from collections import Counter
        suits = Counter(c[1] for c in tpl)
        self.assertEqual(set(suits), set("shdc"))
        self.assertTrue(all(n == 13 for n in suits.values()))


class TestRecognition(unittest.TestCase):
    def setUp(self) -> None:
        self.tpl = load_templates()
        self.cards = list(self.tpl)

    def test_every_template_identifies_itself(self) -> None:
        for card in self.cards:
            m = identify_card(_DECK_DIR / f"{card}.png", self.tpl)
            self.assertEqual(m.card, card)
            self.assertEqual(m.distance, 0)
            self.assertGreater(m.confidence, 0.95)

    def test_cards_are_well_separated(self) -> None:
        # séparation minimale entre cartes distinctes ≥ 20 bits (mesurée 30)
        worst = HASH_BITS
        for i, a in enumerate(self.cards):
            for b in self.cards[i + 1:]:
                worst = min(worst, hamming(self.tpl[a], self.tpl[b]))
        self.assertGreaterEqual(worst, 20)

    def test_scale_invariance(self) -> None:
        # à l'écran les cartes sont plus grandes que le gabarit 15×20
        for scale in (3, 5, 8):
            for card in self.cards:
                im = _sprite(card)
                big = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
                self.assertEqual(identify_card(big, self.tpl).card, card,
                                 f"{card} raté à l'échelle ×{scale}")

    def test_noise_robustness(self) -> None:
        rng = np.random.default_rng(20260808)
        for card in self.cards:
            im = _sprite(card).resize((75, 100), Image.LANCZOS)
            im = im.filter(ImageFilter.GaussianBlur(0.8))
            arr = np.asarray(im.convert("RGB")).astype(np.int16)
            arr = np.clip(arr + rng.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
            self.assertEqual(identify_card(arr, self.tpl).card, card)

    def test_recognize_multiple_by_roi(self) -> None:
        # compose une "table" : 5 cartes agrandies posées à des ROI connues
        board = ["As", "Kh", "Qd", "Jc", "Ts"]
        cw, ch, gap = 60, 80, 10
        sheet = Image.new("RGBA", (len(board) * (cw + gap) + gap, 140),
                          (20, 80, 40, 255))
        rois = []
        for i, card in enumerate(board):
            x, y = gap + i * (cw + gap), 30
            sheet.paste(_sprite(card).resize((cw, ch), Image.LANCZOS), (x, y))
            rois.append((x, y, cw, ch))
        matches = recognize_cards(sheet, rois, self.tpl)
        self.assertEqual([m.card for m in matches], board)
        self.assertTrue(all(m.accepted for m in matches))


class TestFramingTolerance(unittest.TestCase):
    """Un humain ne cadre pas au pixel près : le bord doit être retrouvé.

    Sans recadrage automatique, une sélection élargie de 6 px (du décor
    autour de la carte) faisait chuter la reconnaissance à 0/8 — mesuré.
    """

    CW, CH, X, Y = 60, 80, 100, 50
    CARDS = ["Ah", "Ks", "Qd", "7c", "2s", "Th"]

    def _table(self, card: str) -> Image.Image:
        t = Image.new("RGB", (400, 200), (20, 83, 45))   # feutre vert
        im = _sprite(card).resize((self.CW, self.CH), Image.LANCZOS)
        t.paste(im, (self.X, self.Y), im)
        return t

    def test_exact_framing(self) -> None:
        for card in self.CARDS:
            crop = self._table(card).crop(
                (self.X, self.Y, self.X + self.CW, self.Y + self.CH))
            self.assertEqual(identify_card(crop).card, card)

    def test_generous_framing_is_recovered(self) -> None:
        # cadrage large : du décor tout autour — doit rester correct
        for pad in (6, 12, 20):
            for card in self.CARDS:
                crop = self._table(card).crop(
                    (self.X - pad, self.Y - pad,
                     self.X + self.CW + pad, self.Y + self.CH + pad))
                self.assertEqual(identify_card(crop).card, card,
                                 f"{card} raté avec une marge de {pad} px")

    def test_adjacent_cards_do_not_confuse(self) -> None:
        # board serré : un cadrage large mord sur les voisines ; c'est la
        # carte VISÉE (composante la plus grande) qui doit sortir
        board = ["As", "Kh", "Qd", "Jc", "Ts"]
        cw, ch, gap = 60, 80, 8
        t = Image.new("RGB", (700, 220), (20, 83, 45))
        pos = []
        for i, card in enumerate(board):
            x, y = 60 + i * (cw + gap), 70
            im = _sprite(card).resize((cw, ch), Image.LANCZOS)
            t.paste(im, (x, y), im)
            pos.append((x, y))
        for card, (x, y) in zip(board, pos):
            crop = t.crop((x - 20, y - 20, x + cw + 20, y + ch + 20))
            self.assertEqual(identify_card(crop).card, card,
                             f"{card} confondue avec une carte voisine")

    def test_autocrop_never_degrades_exact_crop(self) -> None:
        # sur un gabarit déjà parfait, le recadrage ne doit rien casser
        for card in self.CARDS:
            self.assertEqual(identify_card(_sprite(card)).card, card)


class TestRejection(unittest.TestCase):
    def test_blank_image_is_not_a_card(self) -> None:
        blank = np.full((80, 60, 3), 255, dtype=np.uint8)
        self.assertIsNone(identify_card(blank).card)

    def test_pure_noise_is_not_a_card(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
        # soit rejeté (card None), soit très basse confiance
        m = identify_card(noise)
        self.assertTrue(m.card is None or m.confidence < 0.5)


class TestBuilder(unittest.TestCase):
    def test_rebuild_matches_shipped(self) -> None:
        # reconstruire depuis les PNG doit redonner les signatures livrées
        rebuilt = build_templates(_DECK_DIR)
        shipped = load_templates()
        self.assertEqual(rebuilt, shipped)


if __name__ == "__main__":
    unittest.main()
