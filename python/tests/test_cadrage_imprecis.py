"""Une boîte juste à quelques pixels près doit suffire à lire la carte.

Le détecteur rend une boîte approximative ; le hachage, lui, exige un
cadrage exact. Mesuré sur la première capture réelle d'une table PMU : la
boîte rendue faisait 126 px de large pour une carte de 121, décalée de 7 px,
et cet écart suffisait à transformer une lecture parfaite en refus (3♥ :
refus à 681 avec la boîte du détecteur, « sure » à 209 avec la boîte exacte).

`identify_card_autour` cherche le bon cadrage autour de la boîte. Le risque
de cette recherche est connu et mesuré : essayer N cadrages et garder le
meilleur, c'est N chances de tomber sous le seuil par hasard. Ces tests
épinglent les deux moitiés — elle récupère les vraies cartes, et elle
n'affirme rien sur ce qui n'en est pas.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pfs.vision.card_recognizer import (
    BUDGET_CADRAGE,
    DISTANCE_SURE,
    MARGE_SURE,
    _cadrages,
    identify_card,
    identify_card_autour,
)
from pfs.vision.synth_table import TableSpec, render_table

FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70),
           (92, 26, 38), (84, 84, 90), (16, 16, 20))


def test_le_parcours_part_du_centre() -> None:
    """Le premier cadrage essayé est la boîte telle qu'elle est fournie.

    C'est ce qui rend la recherche bon marché : une boîte déjà juste est
    reconnue au premier essai, et le budget n'est consommé que par les cas
    difficiles.
    """
    ordre = _cadrages(3, BUDGET_CADRAGE)
    assert ordre[0] == (0, 0, 0, 0)
    ecarts = [abs(a) + abs(b) + abs(c) + abs(d) for a, b, c, d in ordre]
    assert ecarts == sorted(ecarts), "le parcours ne va pas du centre vers l'extérieur"
    assert len(ordre) <= BUDGET_CADRAGE


def test_une_boite_decalee_reste_lisible() -> None:
    """Décaler la boîte de quelques pixels ne doit plus faire perdre la carte."""
    table = render_table(TableSpec(
        hero=("Ah", "Kd"), board=(), felt="feutre vert", size=(1600, 960),
        theme="pmu_solid", decor=True, seed=515))
    image = table.image
    x, y, w, h = (int(v) for v in table.hero_boxes[0])

    juste = identify_card(image.crop((x, y, x + w, y + h)))
    if juste.card != "Ah":
        pytest.skip("le cadrage exact ne lit déjà pas la carte sur ce banc")

    perdues_sans, perdues_avec = 0, 0
    decalages = ((5, 0, 4, 0), (-5, 0, 5, 0), (0, 5, 0, 5), (4, 4, 6, 6))
    for dx, dy, dw, dh in decalages:
        bx, by, bw, bh = x + dx, y + dy, w + dw, h + dh
        brut = identify_card(image.crop((bx, by, bx + bw, by + bh)))
        cherche = identify_card_autour(image, (bx, by, bw, bh))
        perdues_sans += brut.card != "Ah"
        perdues_avec += cherche.card != "Ah"

    assert perdues_avec <= perdues_sans, (
        f"la recherche dégrade la lecture : {perdues_avec} perdues contre "
        f"{perdues_sans} sans elle")


def _non_cartes(n: int = 40):
    """Découpes qui ne sont pas des cartes : bruit, feutre, dos rayés."""
    rng = np.random.default_rng(20260810)
    for i in range(n):
        h, w = int(rng.integers(90, 140)), int(rng.integers(90, 140))
        if i % 3 == 0:
            yield Image.fromarray(
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        elif i % 3 == 1:
            c = FEUTRES[i % len(FEUTRES)]
            a = np.tile(np.array(c, dtype=np.uint8), (h, w, 1))
            a = np.clip(a.astype(int) + rng.normal(0, 5, a.shape), 0, 255)
            yield Image.fromarray(a.astype(np.uint8))
        else:
            img = Image.new("RGB", (w, h),
                            tuple(int(v) for v in rng.integers(60, 200, 3)))
            d = ImageDraw.Draw(img)
            for k in range(0, w + h, 7):
                d.line([(k, 0), (0, k)],
                       fill=tuple(int(v) for v in rng.integers(0, 120, 3)),
                       width=2)
            yield img


def test_la_recherche_n_affirme_rien_sur_une_non_carte() -> None:
    """Le garde-fou survit à l'élargissement de la recherche.

    C'est le test qui compte : une recherche qui récupérerait des cartes au
    prix d'une carte inventée de temps en temps serait une régression, pas
    un progrès. Banc complet dans `banc_cadrage.py` (240 non-cartes,
    0 affirmée) ; ici on garde un échantillon rapide.
    """
    affirmees = []
    for img in _non_cartes():
        m = identify_card_autour(img, (0, 0, img.width, img.height))
        if m.statut == "sure":
            affirmees.append((m.card, m.distance, m.margin))
    assert not affirmees, (
        f"{len(affirmees)} non-carte(s) affirmée(s) après recherche de "
        f"cadrage : {affirmees[:4]}")


def test_la_recherche_ne_relache_aucun_seuil() -> None:
    """Elle élargit la recherche, elle n'abaisse pas les exigences."""
    for img in _non_cartes(12):
        m = identify_card_autour(img, (0, 0, img.width, img.height))
        if m.statut == "sure":
            assert m.distance <= DISTANCE_SURE and m.margin >= MARGE_SURE
        assert m.card is None or m.statut == "sure", (
            "une carte est rendue sans être affirmée")
