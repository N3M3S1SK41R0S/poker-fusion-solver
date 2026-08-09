"""Tests de la localisation automatique des cartes sur une table.

Ce que ces tests verrouillent, et pourquoi ces chiffres-là.

Le banc EST ce fichier : `_bench()` engendre 144 tables (2 habillages × 6 tapis
× 3 tailles de board × 2 échelles de carte × 2 niveaux de bruit) à graine fixe,
avec leur DÉCOR — sièges, avatars, piles de jetons, boutons d'action, montants,
cartes face cachée. Sans décor la mesure serait complaisante : une table qui ne
contient que des cartes ne peut pas produire de fausse carte. Il n'y a pas de
script d'audit hors dépôt : tous les taux annoncés dans la docstring de
`table_detector` se rejouent en changeant une constante du module et en
relançant `_bench()`.

Quatre propriétés sont vérifiées, dans l'ordre de leur importance :

1. **Aucune boîte fantôme.** Une carte inventée fausse silencieusement tout le
   conseil qui suit. La mesure vaut 0/986 : le test exige donc ZÉRO, sans
   marge — une régression à MAX_RATIO = 0,85 (2,5 % de fantômes) doit casser.
2. **Les cartes sont trouvées.** 664/672 = 98,8 % des cartes du héros et du
   board sont localisées à IoU ≥ 0,5. Les seuils gardent ici 2 à 3 points de
   marge, parce qu'un réglage voisin ne doit pas casser la suite.
3. **Les rôles sont justes.** 659/672 = 98,1 %, et jamais un dos de carte
   adverse promu en carte du héros ou du board (0/720).
4. **Les justifications écrites tiennent.** Chaque borne du détecteur est
   remise à sa valeur d'avant sur un sous-banc et doit redonner le défaut
   qu'elle corrige (`TestAblations`) ; la limite JPEG et le recouvrement des
   densités d'arêtes sont mesurés ici, pas seulement racontés.

Les distracteurs sont ceux qui trompent réellement un détecteur de rectangles,
avec leurs valeurs MESURÉES ici : les trois piles de jetons donnent des boîtes
de rapport 0,92 / 0,80 / 0,71, striées à 0,833 isolées et à 0,769–0,775 telles
que le détecteur les cadre sur une table ; les avatars de siège donnent des
boîtes recalées de rapport 0,846 à 0,872.
"""

from __future__ import annotations

import io
import itertools
import random
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from pfs.vision import table_detector
from pfs.vision.phash import _rgb_array
from pfs.vision.synth_table import _SEATS, FELTS, TableSpec, _chip_stack, render_table
from pfs.vision.table_detector import (
    INNER_MAX,
    MAX_RATIO,
    MIN_RATIO,
    QUIET_SIDES,
    CardBox,
    detect_card_boxes,
    read_table,
)

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK = [r + s for r in RANKS for s in SUITS]

FELT_RGB = (20, 83, 45)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / (aw * ah + bw * bh - inter)


def _best(box, others) -> float:
    return max((_iou(box, o) for o in others), default=0.0)


def _bench(themes=("pmu_deck", "pmu_solid"), felts=tuple(FELTS),
           boards=(0, 3, 5), sizes=((52, 70), (68, 92)), noises=(0.0, 6.0)):
    """Cas du banc : mêmes tirages à chaque exécution (graine fixe)."""
    rng = random.Random(20260808)
    for i, (theme, felt, nb, (cw, ch), noise) in enumerate(
            itertools.product(themes, felts, boards, sizes, noises)):
        cards = rng.sample(DECK, 2 + nb)
        yield TableSpec(hero=(cards[0], cards[1]), board=tuple(cards[2:]),
                        theme=theme, felt=felt, card_w=cw, card_h=ch,
                        noise=noise, blur=0.6 if noise else 0.0, seed=i,
                        villains=i % 6)


# 144 tables à ~0,33 s : on ne les lit qu'une fois pour tout le module.
_READS: list[tuple[TableSpec, object, object]] = []


def _reads() -> list[tuple[TableSpec, object, object]]:
    if not _READS:
        for spec in _bench():
            st = render_table(spec)
            _READS.append((spec, st, read_table(st.image)))
    return _READS


def _counts(reads, keep=lambda spec: True) -> tuple[int, int]:
    """Cartes du héros et du board retrouvées à IoU ≥ 0,5, et total attendu."""
    found = total = 0
    for spec, st, tr in reads:
        if not keep(spec):
            continue
        boxes = [b.box for b in tr.all]
        total += len(st.all_boxes)
        found += sum(1 for g in st.all_boxes if _best(g, boxes) >= 0.5)
    return found, total


def _phantoms(reads) -> tuple[list[CardBox], int]:
    """Boîtes qui ne recouvrent AUCUNE carte réelle, et total produit."""
    ghosts: list[CardBox] = []
    produced = 0
    for _, st, tr in reads:
        for b in tr.all:
            produced += 1
            if _best(b.box, st.every_card) < 0.3:
                ghosts.append(b)
    return ghosts, produced


def _reread(reads, transform=lambda image: image):
    """Relit un sous-banc avec les constantes ACTUELLES du module."""
    return [(spec, st, read_table(transform(st.image))) for spec, st, _ in reads]


def _seat_avatars(size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Carrés d'avatar du décor — même géométrie que `_draw_decor`."""
    w, h = size
    side = int(h * 0.058) + 1
    return [(int(w * fx), int(h * fy), side, side) for fx, fy in _SEATS]


def _chip_stack_image(n: int) -> Image.Image:
    """Une pile de `n` jetons seule sur du feutre, à la largeur du décor."""
    img = Image.new("RGB", (120, 160), FELT_RGB)
    _chip_stack(ImageDraw.Draw(img), 40, 130, 23, n)
    return img


def _ink_box(img: Image.Image) -> tuple[int, int, int, int]:
    """Boîte englobante de ce qui n'est pas le feutre."""
    a = np.asarray(img).astype(int)
    ys, xs = np.where(np.abs(a - np.array(FELT_RGB)).sum(axis=2) > 12)
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _inner_density(img: Image.Image, box: tuple[int, int, int, int]) -> float:
    """Densité d'arêtes du cœur d'une boîte, exactement comme le détecteur."""
    edges = np.logical_or(*table_detector._edge_maps(_rgb_array(img)))
    x, y, w, h = box
    m = max(2, min(w, h) // 8)
    core = table_detector._rect(edges, x + m, y + m, w - 2 * m, h - 2 * m)
    return float(core.mean())


# Tapis dont la clarté approche celle d'un carton blanc : c'est là que la
# couleur ne sépare plus rien et que seul le gradient au bord tient.
_LIGHT_FELTS = ("tapis clair", "gris moyen")


class TestBoxGeometry(unittest.TestCase):
    """Ce qu'une boîte de carte doit être, indépendamment du taux de réussite."""

    def setUp(self) -> None:
        self.table = render_table(TableSpec(
            hero=("Ah", "Kd"), board=("2c", "7s", "Td"), theme="pmu_solid",
            felt="feutre vert", villains=3, seed=7))

    def test_boxes_have_a_card_ratio(self) -> None:
        for b in detect_card_boxes(self.table.image):
            self.assertTrue(MIN_RATIO <= b.w / b.h <= MAX_RATIO,
                            f"rapport hors gabarit : {b.box}")

    def test_boxes_stay_inside_the_image(self) -> None:
        W, H = self.table.image.size
        for b in detect_card_boxes(self.table.image):
            self.assertGreaterEqual(b.x, 0)
            self.assertGreaterEqual(b.y, 0)
            self.assertLessEqual(b.x + b.w, W)
            self.assertLessEqual(b.y + b.h, H)

    def test_no_two_boxes_are_nested(self) -> None:
        """Deux cartes ne s'emboîtent jamais : une boîte dans une autre est un
        artefact d'appariement, pas une carte."""
        boxes = [b.box for b in detect_card_boxes(self.table.image)]
        for a, b in itertools.combinations(boxes, 2):
            x0, y0 = max(a[0], b[0]), max(a[1], b[1])
            x1 = min(a[0] + a[2], b[0] + b[2])
            y1 = min(a[1] + a[3], b[1] + b[3])
            inter = max(0, x1 - x0) * max(0, y1 - y0)
            self.assertLess(inter / min(a[2] * a[3], b[2] * b[3]), 0.6,
                            f"{a} et {b} s'emboîtent")

    def test_detection_is_deterministic(self) -> None:
        first = [b.box for b in detect_card_boxes(self.table.image)]
        second = [b.box for b in detect_card_boxes(self.table.image)]
        self.assertEqual(first, second)


class TestEmptyAndDegenerate(unittest.TestCase):
    def test_uniform_image_has_no_card(self) -> None:
        flat = np.full((300, 400, 3), 60, dtype=np.uint8)
        self.assertEqual(detect_card_boxes(flat), [])
        self.assertEqual(read_table(flat).all, [])

    def test_pure_noise_invents_no_card(self) -> None:
        rng = np.random.default_rng(20260808)
        noise = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        self.assertEqual(detect_card_boxes(noise), [])

    def test_table_without_cards_stays_empty(self) -> None:
        """Décor complet, aucune carte : rien ne doit être annoncé."""
        st = render_table(TableSpec(hero=(), board=(), felt="gris moyen",
                                    villains=0, seed=3))
        self.assertEqual(read_table(st.image).all, [])

    def test_tiny_image_does_not_crash(self) -> None:
        self.assertEqual(detect_card_boxes(np.zeros((9, 7, 3), np.uint8)), [])


class TestLocalisationRate(unittest.TestCase):
    """Les taux, sur le banc décoré. Seuils sous la mesure, avec marge."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.reads = _reads()

    def test_hero_and_board_are_localised(self) -> None:
        """Mesure : 664/672 = 98,8 %. Seuil 0,98, soit 5 cartes de marge."""
        found, total = _counts(self.reads)
        self.assertGreaterEqual(found / total, 0.98,
                                f"{found}/{total} cartes localisées")

    def test_no_phantom_box(self) -> None:
        """Une boîte qui ne recouvre AUCUNE carte réelle est une carte inventée.

        Aucune marge ici, contrairement aux autres seuils : la mesure vaut
        exactement 0/986, et c'est la propriété la plus importante du module.
        """
        ghosts, produced = _phantoms(self.reads)
        self.assertEqual(ghosts, [], f"{len(ghosts)}/{produced} boîtes fantômes")

    def test_white_cards_on_a_light_felt(self) -> None:
        """Le cas qui a motivé la réécriture : carton blanc sur tapis clair,
        où l'écart de couleur au feutre ne vaut que 37 sur 255.

        Mesure : 107/112 = 95,5 % (gris moyen 94,6 %, tapis clair 96,4 %).
        """
        found, total = _counts(
            self.reads, lambda s: s.theme == "pmu_deck" and s.felt in _LIGHT_FELTS)
        self.assertGreaterEqual(found / total, 0.92,
                                f"deck classique sur tapis clair : {found}/{total}")

    def test_noise_does_not_collapse_the_rate(self) -> None:
        """Mesure : 330/336 = 98,2 %."""
        found, total = _counts(self.reads, lambda s: s.noise > 0)
        self.assertGreaterEqual(found / total, 0.96, f"sous bruit : {found}/{total}")

    def test_small_cards_are_found(self) -> None:
        """Une table réduite dessine des cartes de 52 px de large.
        Mesure : 328/336 = 97,6 %."""
        found, total = _counts(self.reads, lambda s: s.card_w == 52)
        self.assertGreaterEqual(found / total, 0.95,
                                f"petites cartes : {found}/{total}")


class TestJpegLimit(unittest.TestCase):
    """La limite qui fait échouer le critère de la phase A.

    Le banc est en PNG ; une capture d'écran réelle ne l'est presque jamais.
    Sur la tranche la plus fragile (habillage plein, cartes de 52 px, sans
    bruit) la localisation passe de 84/84 en PNG à 53/84 = 63,1 % à q = 75.
    Les fantômes, eux, restent à zéro : la compression fait PERDRE des cartes,
    elle n'en invente pas.

    Le balayage sur les 144 tables (q = 95 → 96,6 %, q = 90 → 95,5 %,
    q = 85 → 93,3 %, q = 75 → 88,1 %, q = 60 → 75,0 %) est écrit dans la
    docstring de `table_detector` ; il se rejoue avec ce même `_bench()`, mais
    on n'en garde ici qu'un point, pour ne pas payer cinq passes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.reads = [r for r in _reads() if r[0].theme == "pmu_solid"
                     and r[0].card_w == 52 and r[0].noise == 0.0]

    @staticmethod
    def _recompress(quality: int):
        def transform(image):
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return transform

    def test_png_slice_is_perfect(self) -> None:
        found, total = _counts(self.reads)
        self.assertEqual(found, total, f"référence PNG : {found}/{total}")

    def test_jpeg_75_breaks_the_phase_a_criterion(self) -> None:
        reads = _reread(self.reads, self._recompress(75))
        found, total = _counts(reads)
        ghosts, _ = _phantoms(reads)
        self.assertLess(found / total, 0.80,
                        f"JPEG q=75 : {found}/{total} — la limite se serait "
                        "résorbée, il faut la re-chiffrer")
        self.assertEqual(ghosts, [], "la compression ne doit pas inventer de carte")


class TestAblations(unittest.TestCase):
    """Chaque borne remise à sa valeur d'avant doit rendre le défaut qu'elle
    corrige — sinon la justification écrite dans `table_detector` est un décor.

    Les ablations tournent sur les 12 tables bruitées à tapis sombre : toutes
    les boîtes fantômes du banc y naissent, et rejouer le banc entier coûterait
    une passe de 45 s par constante pour la même conclusion.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.reads = [r for r in _reads() if r[0].noise > 0 and r[0].card_w == 68
                     and r[0].felt in ("bleu nuit", "noir")]

    def test_the_sub_bench_is_clean_by_default(self) -> None:
        found, total = _counts(self.reads)
        ghosts, _ = _phantoms(self.reads)
        self.assertEqual((found, len(ghosts)), (total, 0),
                         f"{found}/{total} cartes, {len(ghosts)} fantômes")

    def test_quiet_sides_does_not_sort_the_candidates(self) -> None:
        """Le test des abords calmes ne TRIE rien au niveau candidate.

        Sur les 9 298 candidates de ce sous-banc, exiger trois abords calmes en
        garde 22,9 % parmi les bonnes et 23,6 % parmi les fantômes : la
        sélectivité est nulle, et même à l'envers. C'est le dédoublonnage qui
        transforme ce non-tri en zéro fantôme, pas le tri lui-même.
        """
        seen: list[tuple[tuple[int, int, int, int], int]] = []
        original = table_detector._looks_like_a_card

        def probe(rgb, edges, box):
            _, outside = table_detector._side_windows(box)
            quiet = 0
            for outs in outside:
                band = table_detector._rect(edges, *outs)
                if band.size and float(band.mean()) < table_detector.QUIET_MAX:
                    quiet += 1
            seen.append((box, quiet))
            return original(rgb, edges, box)

        good = good_quiet = bad = bad_quiet = 0
        with patch.object(table_detector, "_looks_like_a_card", probe):
            for _, st, _ in self.reads:
                seen.clear()
                detect_card_boxes(st.image)
                for box, quiet in seen:
                    best = _best(box, st.every_card)
                    if best >= 0.5:
                        good += 1
                        good_quiet += quiet >= QUIET_SIDES
                    elif best < 0.3:
                        bad += 1
                        bad_quiet += quiet >= QUIET_SIDES
        kept_good, kept_bad = good_quiet / good, bad_quiet / bad
        self.assertLess(abs(kept_good - kept_bad), 0.10,
                        f"garde {100 * kept_good:.1f} % des bonnes et "
                        f"{100 * kept_bad:.1f} % des fantômes : le test trierait "
                        "vraiment les candidates, la docstring est à réécrire")

    def test_quiet_sides_is_what_removes_the_phantoms(self) -> None:
        """Sans le test des abords calmes : 35 fantômes sur ce sous-banc.

        Le mécanisme n'est pas un tri des candidates — la localisation ne bouge
        pas, 56/56 dans les deux cas — c'est ce qui survit au dédoublonnage.
        """
        with patch.object(table_detector, "QUIET_SIDES", 0):
            reads = _reread(self.reads)
        ghosts, _ = _phantoms(reads)
        found, total = _counts(reads)
        self.assertGreaterEqual(len(ghosts), 10, "l'ablation ne défait plus rien")
        self.assertEqual(found, total, f"{found}/{total} — la localisation a bougé")

    def test_four_quiet_sides_collapses_the_localisation(self) -> None:
        """Exiger les QUATRE abords calmes : 39/56 = 69,6 %. La carte voisine
        suffit alors à disqualifier une vraie carte."""
        with patch.object(table_detector, "QUIET_SIDES", 4):
            found, total = _counts(_reread(self.reads))
        self.assertLess(found / total, 0.80, f"{found}/{total} cartes localisées")

    def test_the_ratio_ceiling_is_fixed_by_seat_avatars(self) -> None:
        """Desserrée à 0,88, la borne fait entrer 14 fantômes sur ce sous-banc,
        tous posés sur un avatar de siège, sans gagner une seule carte."""
        with patch.object(table_detector, "MAX_RATIO", 0.88):
            reads = _reread(self.reads)
        ghosts, _ = _phantoms(reads)
        found, total = _counts(reads)
        self.assertGreaterEqual(len(ghosts), 5, "l'ablation ne défait plus rien")
        self.assertEqual(found, total, "desserrer la borne ne gagne aucune carte")
        avatars = _seat_avatars(self.reads[0][1].image.size)
        for b in ghosts:
            self.assertTrue(0.84 <= b.w / b.h <= 0.88,
                            f"rapport {b.w / b.h:.3f}, hors de la bande des avatars")
            self.assertGreaterEqual(max(_iou(b.box, a) for a in avatars), 0.3,
                                    f"{b.box} n'est pas posé sur un avatar de siège")


class TestInnerDensityIsATradeOff(unittest.TestCase):
    """INNER_MAX ne sépare pas deux populations, il tranche dans une seule.

    Densité d'arêtes intérieure des 672 vraies cartes : médiane 0,463, p90
    0,661, p99 0,777, maximum 0,819. Les piles de jetons, telles que le
    détecteur les CADRE au milieu d'une table, lisent 0,769 et 0,775 — donc
    11 vraies cartes sur 672 (1,6 %) sont plus striées que le plus calme des
    tas de jetons. Aucun plafond ne garde toutes les cartes en rejetant tous
    les jetons ; celui-ci se place sous toute la zone de recouvrement et coûte
    les 4,8 % de cartes qui le dépassent.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.dens = np.array([_inner_density(st.image, g)
                             for _, st, _ in _reads() for g in st.all_boxes])
        # Le tapis noir sans bruit est la seule tranche où desserrer le plafond
        # laisse passer les jetons : 24 fantômes sur ces 12 tables.
        cls.dark = [r for r in _reads() if r[0].felt == "noir" and r[0].noise == 0.0]

    def test_real_cards_are_mostly_quiet(self) -> None:
        """Mesure sur les 672 cartes : médiane 0,463, p90 0,661, p99 0,777."""
        self.assertEqual(self.dens.size, 672)
        self.assertLess(float(np.median(self.dens)), 0.50)
        self.assertLess(float(np.percentile(self.dens, 90)), 0.70)

    def test_the_ceiling_costs_about_five_percent_of_the_cards(self) -> None:
        """4,8 % des vraies cartes dépassent 0,72 : c'est le prix payé pour
        écarter les jetons. Si ce chiffre s'envole, le réglage est à revoir."""
        above = float((self.dens > INNER_MAX).mean())
        self.assertLess(above, 0.06, f"{100 * above:.1f} % des cartes au-dessus")
        self.assertGreater(above, 0.0, "plus aucune carte au-dessus : re-chiffrer")

    def test_chip_boxes_fall_inside_the_card_distribution(self) -> None:
        """Le recouvrement, mesuré : les boîtes de jetons qui passent quand on
        desserre le plafond à 0,78 sont MOINS striées que 1,6 % des cartes."""
        with patch.object(table_detector, "INNER_MAX", 0.78):
            loose = _reread(self.dark)
        chips = [_inner_density(st.image, b.box) for _, st, tr in loose
                 for b in tr.all if _best(b.box, st.every_card) < 0.3]
        self.assertGreaterEqual(len(chips), 10, "l'ablation ne défait plus rien")
        self.assertGreater(float(self.dens.max()), min(chips),
                           "les jetons seraient enfin séparables : re-chiffrer")
        overlap = float((self.dens > min(chips)).mean())
        self.assertGreater(overlap, 0.005,
                           f"{100 * overlap:.1f} % de cartes au-dessus des jetons")


class TestDecorIsReal(unittest.TestCase):
    """Le décor doit être ce que les docstrings prétendent qu'il est."""

    def test_chip_stacks_span_the_ratio_window(self) -> None:
        """Les trois piles donnent 24×26, 24×30 et 24×34, soit 0,92 / 0,80 /
        0,71 : elles encadrent la fenêtre du détecteur au lieu de viser un
        seul rapport."""
        for n, wanted in ((6, (24, 26)), (7, (24, 30)), (8, (24, 34))):
            *_, w, h = _ink_box(_chip_stack_image(n))
            self.assertEqual((w, h), wanted, f"pile de {n} jetons")

    def test_chip_stacks_are_striped_enough_to_be_rejected(self) -> None:
        """Densité d'arêtes mesurée : 0,833 pour les trois piles, au-dessus du
        plafond 0,72 — c'est ce qui les écarte, pas leur rapport."""
        for n in (6, 7, 8):
            img = _chip_stack_image(n)
            self.assertGreater(_inner_density(img, _ink_box(img)), INNER_MAX,
                               f"pile de {n} jetons")

    def test_amounts_are_drawn_not_just_declared(self) -> None:
        """`texts` serait une vérité-terrain mensongère si les montants
        n'étaient pas peints : deux tables identiques au pot près doivent
        donner deux images différentes."""
        a = render_table(TableSpec(hero=("Ah", "Kd"), pot=1.5, seed=11))
        b = render_table(TableSpec(hero=("Ah", "Kd"), pot=88.25, seed=11))
        self.assertNotEqual(a.texts["pot"], b.texts["pot"])
        self.assertFalse(np.array_equal(np.asarray(a.image), np.asarray(b.image)),
                         "le pot n'est pas dessiné sur l'image")


class TestLabelTooCloseToACard(unittest.TestCase):
    """Pourquoi le banc écrit le tapis du héros 16 px sous ses cartes, et pas 6.

    À 6 px les glyphes tombent DANS la bande d'abord basse que le détecteur
    exige calme. Sur les 18 tables les plus fragiles du banc (deck classique,
    cartes de 52 px, bruitées), le second libellé fait perdre 11 cartes ; sur
    le banc entier, déplacer le libellé de 16 px à 6 px fait passer la
    localisation de 664/672 à 652/672, soit 98,8 % → 97,0 %.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.reads = [r for r in _reads() if r[0].theme == "pmu_deck"
                     and r[0].card_w == 52 and r[0].noise > 0]

    @staticmethod
    def _glue(spec: TableSpec, texts: dict[str, str]):
        def transform(image):
            copy = image.copy()
            ImageDraw.Draw(copy).text(
                (spec.size[0] // 2 - 24, int(spec.size[1] * 0.72) + spec.card_h + 6),
                texts["hero_stack"], fill=(238, 238, 210))
            return copy
        return transform

    def test_a_glued_label_costs_cards(self) -> None:
        before, total = _counts(self.reads)
        glued = [(spec, st, read_table(self._glue(spec, st.texts)(st.image)))
                 for spec, st, _ in self.reads]
        after, _ = _counts(glued)
        self.assertGreaterEqual(before - after, 5,
                                f"{before}/{total} → {after}/{total} : l'effet a "
                                "disparu, la marge de 16 px est à re-justifier")


class TestRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reads = _reads()

    def test_roles_are_right(self) -> None:
        """Mesure : 659/672 = 98,1 %."""
        right = total = 0
        for _, st, tr in self.reads:
            labelled = ([(b.box, "hero") for b in tr.hero]
                        + [(b.box, "board") for b in tr.board]
                        + [(b.box, "?") for b in tr.others])
            truth = ([(g, "hero") for g in st.hero_boxes]
                     + [(g, "board") for g in st.board_boxes])
            for g, role in truth:
                total += 1
                best = max((_iou(g, d) for d, _ in labelled), default=0.0)
                if best < 0.5:
                    continue
                got = max(labelled, key=lambda d: _iou(g, d[0]))[1]
                right += got == role
        self.assertGreaterEqual(right / total, 0.97,
                                f"{right}/{total} rôles corrects")

    def test_face_down_cards_are_never_promoted(self) -> None:
        """Un dos de carte adverse dans la main du héros fausserait tout.
        Mesure : 0/720."""
        promoted = total = 0
        for _, st, tr in self.reads:
            claimed = [b.box for b in tr.hero + tr.board]
            for g in st.villain_boxes:
                total += 1
                promoted += _best(g, claimed) >= 0.5
        self.assertEqual(promoted, 0, f"{promoted}/{total} dos promus")

    def test_hero_is_never_more_than_four_cards(self) -> None:
        for _, _, tr in self.reads:
            self.assertLessEqual(len(tr.hero), 4)

    def test_board_size_is_plausible(self) -> None:
        for _, _, tr in self.reads:
            self.assertIn(len(tr.board), (0, 3, 4, 5))

    def test_board_sits_above_hero(self) -> None:
        for _, _, tr in self.reads:
            if not tr.hero or not tr.board:
                continue
            self.assertLess(max(b.y + b.h for b in tr.board),
                            max(b.y + b.h for b in tr.hero))

    def test_preflop_has_no_board(self) -> None:
        for _, st, tr in self.reads:
            if st.board:
                continue
            self.assertEqual(tr.board, [], "board annoncé sans board dessiné")

    def test_roles_partition_the_boxes(self) -> None:
        for _, _, tr in self.reads:
            boxes = [b.box for b in tr.all]
            self.assertEqual(len(boxes), len(set(boxes)))


class TestCardBox(unittest.TestCase):
    def test_centre_and_box(self) -> None:
        b = CardBox(10, 20, 30, 40, "hero")
        self.assertEqual(b.box, (10, 20, 30, 40))
        self.assertAlmostEqual(b.cx, 25.0)
        self.assertAlmostEqual(b.cy, 40.0)


if __name__ == "__main__":
    unittest.main()
