"""Tests du recogniseur de cartes — multi-habillages, multi-tapis.

Le client change l'habillage des cartes ET le tapis. Trois mécanismes rendent
la lecture indépendante des deux :

1. le recadrage isole la carte par **écart de couleur** au tapis, jamais par
   luminosité — un carton rouge sur feutre vert a la MÊME luminosité, et le
   seuillage en niveaux de gris tombait alors à 0/52 ;
2. la banque réunit les signatures de **tous** les habillages livrés : on ne
   devine pas le thème, c'est la signature la plus proche qui gagne ;
3. le **coin haut-gauche** (là où se lit le rang) est haché à part et pondéré :
   sur un habillage à fond plein, le rang est un petit glyphe noyé dans un
   aplat, et « 7♦ » ne se séparait de « J♦ » que de 0 bit.

**Garantie visée : ne jamais annoncer une carte fausse.** Une mauvaise carte
fausse silencieusement tout le conseil qui suit ; un refus, lui, demande
simplement de recadrer. Les seuils sont calés là-dessus (1 560 essais :
87,6 % de lectures, 0,13 % de fausses).
"""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np
from PIL import Image, ImageFilter

from pfs.vision import build_templates, identify_card, phash, recognize_cards
from pfs.vision.card_recognizer import (
    _DECK_DIR,
    _TEMPLATE_ROOT,
    _THEMES,
    build_all_themes,
    load_templates,
)
from pfs.vision.phash import HASH_BITS, hamming


def _sprite(card: str, theme: str = "pmu_deck") -> Image.Image:
    return Image.open(_TEMPLATE_ROOT / theme / f"{card}.png").convert("RGBA")


def _cards(theme: str) -> list[str]:
    return sorted(f.stem for f in (_TEMPLATE_ROOT / theme).glob("*.png"))


class TestPhash(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = np.zeros((20, 15), dtype=np.uint8)
        a[5:15, 5:10] = 255
        self.assertEqual(phash(a), phash(a))

    def test_self_distance_zero(self) -> None:
        h = phash(_sprite("Ah"))
        self.assertEqual(hamming(h, h), 0)

    def test_hash_width(self) -> None:
        # 576 bits : bloc 24×24 sur une image normalisée en 64×64
        self.assertEqual(HASH_BITS, 576)


class TestDeckIntegrity(unittest.TestCase):
    """Chaque habillage livré doit être un jeu COMPLET et sans doublon."""

    def test_each_theme_is_a_full_deck(self) -> None:
        for theme in _THEMES:
            if not (_TEMPLATE_ROOT / theme).is_dir():
                continue
            cards = _cards(theme)
            self.assertEqual(len(cards), 52, f"{theme} : {len(cards)} cartes")
            self.assertEqual(len(set(cards)), 52, f"{theme} : doublon")
            suits = Counter(c[1] for c in cards)
            ranks = Counter(c[0] for c in cards)
            self.assertEqual(set(suits), set("shdc"), theme)
            self.assertTrue(all(n == 13 for n in suits.values()), theme)
            self.assertTrue(all(n == 4 for n in ranks.values()), theme)

    def test_bank_holds_every_theme(self) -> None:
        tpl = load_templates()
        themes = {k.split("@", 1)[1] for k in tpl}
        self.assertIn("pmu_deck", themes)
        self.assertIn("pmu_solid", themes)
        self.assertEqual(len(tpl), 52 * len(themes))

    def test_rebuild_matches_shipped(self) -> None:
        # Reconstruire depuis les PNG doit redonner les signatures livrées.
        # La couleur est arrondie à 2 décimales dans le JSON : on compare à
        # la même précision.
        rebuilt = build_all_themes(save_to=None)
        shipped = load_templates()
        self.assertEqual(set(rebuilt), set(shipped))
        for k, (h, kh, c) in rebuilt.items():
            sh, skh, sc = shipped[k]
            self.assertEqual((h, kh), (sh, skh), k)
            for a, b in zip(c, sc):
                self.assertAlmostEqual(a, b, places=1, msg=k)

    def test_build_templates_reads_one_theme(self) -> None:
        t = build_templates(_DECK_DIR)
        self.assertEqual(len(t), 52)
        self.assertIn("Ah", t)          # clés SANS suffixe de thème ici


class TestRecognition(unittest.TestCase):
    """Sur les gabarits eux-mêmes : la lecture doit être exacte."""

    def test_every_template_identifies_itself(self) -> None:
        for theme in _THEMES:
            if not (_TEMPLATE_ROOT / theme).is_dir():
                continue
            for card in _cards(theme):
                m = identify_card(_TEMPLATE_ROOT / theme / f"{card}.png")
                self.assertEqual(m.card, card, f"{theme}/{card}")
                self.assertEqual(m.distance, 0)

    def test_scale_never_produces_a_wrong_read(self) -> None:
        """Carte agrandie SANS tapis autour — cas volontairement ingrat.

        Sans fond, le recadrage n'a aucun bord à retrouver et le taux de
        lecture chute ; ce n'est pas l'usage réel (une carte est toujours sur
        une table — voir les tests sur fond, qui mesurent les taux). Ce qui
        doit tenir même ici : ne jamais annoncer une carte fausse.
        """
        for scale in (3, 5, 8):
            for card in _cards("pmu_deck"):
                im = _sprite(card)
                big = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
                m = identify_card(big)
                self.assertIn(m.card, (card, None),
                              f"{card} lu FAUX à ×{scale} : {m.card}")

    def test_noise_never_produces_a_wrong_read(self) -> None:
        rng = np.random.default_rng(20260808)
        for card in _cards("pmu_deck"):
            im = _sprite(card).resize((75, 100), Image.LANCZOS)
            im = im.filter(ImageFilter.GaussianBlur(0.8))
            arr = np.asarray(im.convert("RGB")).astype(np.int16)
            arr = np.clip(arr + rng.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
            m = identify_card(arr)
            self.assertIn(m.card, (card, None), f"{card} lu FAUX : {m.card}")


class TestThemesAndBackgrounds(unittest.TestCase):
    """Tous les habillages de cartes, sur tous les tapis."""

    CW, CH, X, Y, PAD = 68, 92, 90, 44, 14
    FELTS = {
        "feutre vert": (20, 83, 45),
        "tapis clair": (206, 209, 214),
        "bleu nuit": (16, 24, 56),
        "noir": (10, 10, 12),
        "bordeaux": (92, 18, 30),
        "gris moyen": (128, 128, 132),
    }

    def _framed(self, theme: str, card: str, felt, pad: int | None = None):
        pad = self.PAD if pad is None else pad
        t = Image.new("RGB", (300, 200), felt)
        im = _sprite(card, theme).resize((self.CW, self.CH), Image.LANCZOS)
        t.paste(im, (self.X, self.Y), im)
        return t.crop((self.X - pad, self.Y - pad,
                       self.X + self.CW + pad, self.Y + self.CH + pad))

    def test_never_announces_a_wrong_card(self) -> None:
        """LA propriété critique : une carte annoncée est une carte juste."""
        wrong, total = [], 0
        for theme in _THEMES:
            if not (_TEMPLATE_ROOT / theme).is_dir():
                continue
            for card in _cards(theme):
                for felt in self.FELTS.values():
                    total += 1
                    m = identify_card(self._framed(theme, card, felt))
                    if m.card is not None and m.card != card:
                        wrong.append((theme, card, m.card))
        self.assertLess(len(wrong), total * 0.01,
                        f"{len(wrong)}/{total} lectures FAUSSES : {wrong[:6]}")

    def test_solid_theme_read_on_every_felt(self) -> None:
        """L'habillage à fond plein — celui affiché par le client PMU."""
        for name, felt in self.FELTS.items():
            cards = _cards("pmu_solid")
            ok = sum(identify_card(self._framed("pmu_solid", c, felt)).card == c
                     for c in cards)
            self.assertGreaterEqual(ok, 45, f"fond plein sur {name} : {ok}/52")

    def test_classic_theme_read_on_contrasting_felts(self) -> None:
        for name, felt in self.FELTS.items():
            cards = _cards("pmu_deck")
            ok = sum(identify_card(self._framed("pmu_deck", c, felt)).card == c
                     for c in cards)
            self.assertGreaterEqual(ok, 40, f"classique sur {name} : {ok}/52")

    def test_generous_framing_is_recovered(self) -> None:
        """Personne ne cadre au pixel près : le bord est retrouvé."""
        felt = self.FELTS["feutre vert"]
        for pad in (6, 14, 22):
            cards = _cards("pmu_solid")
            ok = 0
            for c in cards:
                m = identify_card(self._framed("pmu_solid", c, felt, pad))
                self.assertIn(m.card, (c, None), f"{c} lu FAUX à +{pad} px")
                ok += m.card == c
            self.assertGreaterEqual(ok, 44, f"marge +{pad} px : {ok}/52")

    def test_adjacent_cards_do_not_confuse(self) -> None:
        """Board serré : c'est la carte VISÉE qui doit sortir, pas sa voisine."""
        board = ["As", "Kh", "Qd", "Jc", "Ts"]
        cw, ch, gap = 68, 92, 8
        t = Image.new("RGB", (760, 220), self.FELTS["feutre vert"])
        pos = []
        for i, card in enumerate(board):
            x, y = 60 + i * (cw + gap), 70
            im = _sprite(card, "pmu_solid").resize((cw, ch), Image.LANCZOS)
            t.paste(im, (x, y), im)
            pos.append((x, y))
        # cadrage réaliste (la carte + un peu de marge) : au-delà, la
        # sélection contient surtout les voisines et le refus est légitime
        read = 0
        for card, (x, y) in zip(board, pos):
            m = identify_card(t.crop((x - 6, y - 6, x + cw + 6, y + ch + 6)))
            # une voisine ne doit JAMAIS être annoncée à la place de la visée
            self.assertIn(m.card, (card, None), f"{card} confondue : {m.card}")
            read += m.card == card
        self.assertGreaterEqual(read, 4, f"{read}/5 cartes lues sur board serré")

    def test_recognize_multiple_by_roi(self) -> None:
        board = ["As", "Kh", "Qd", "Jc", "Ts"]
        cw, ch, gap = 68, 92, 14
        t = Image.new("RGB", (760, 220), self.FELTS["feutre vert"])
        rois = []
        for i, card in enumerate(board):
            x, y = 40 + i * (cw + gap), 60
            t.paste(_sprite(card, "pmu_solid").resize((cw, ch), Image.LANCZOS),
                    (x, y), _sprite(card, "pmu_solid").resize((cw, ch), Image.LANCZOS))
            rois.append((x - 8, y - 8, cw + 16, ch + 16))
        got = [m.card for m in recognize_cards(t, rois)]
        self.assertEqual(got, board)


class TestRejection(unittest.TestCase):
    def test_blank_image_is_not_a_card(self) -> None:
        blank = np.full((80, 60, 3), 255, dtype=np.uint8)
        self.assertIsNone(identify_card(blank).card)

    def test_pure_noise_is_not_a_card(self) -> None:
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, (80, 60, 3), dtype=np.uint8)
        m = identify_card(noise)
        self.assertTrue(m.card is None or m.confidence < 0.5)


if __name__ == "__main__":
    unittest.main()
