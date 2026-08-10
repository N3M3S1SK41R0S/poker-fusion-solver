"""Pourquoi la localisation se fait à PLEINE ÉCHELLE.

Réduire l'image avant de chercher les cartes semblait gratuit : le détecteur
raisonne en fractions de l'image, son coût croît avec le nombre de pixels.
Une première mesure, sur une seule famille de tables, l'a paru confirmer, et
le raccourci est devenu le chemin par défaut avec la mention « 1280 est un
optimum ».

Le banc complet (`banc_localisation.py --large`, 54 tables par
configuration) a démenti : à 1280, la localisation tombe à 56,0 % sur le
deck classique, 70,2 % avec des cartes de 52×70, 79,8 % sur des images
bruitées. Pleine échelle : 100 % partout.

Ces tests épinglent les deux moitiés de la décision — la pleine échelle
tient, et la réduction est réellement lossy. Le second est le plus utile :
sans lui, quelqu'un (moi) réactivera le raccourci en le croyant sans coût.
"""

from __future__ import annotations

import pytest

from pfs.vision.live import (
    LARGEUR_SANS_REDUCTION,
    _agrandir,
    _reduire,
)
from pfs.vision.synth_table import TableSpec, render_table
from pfs.vision.table_detector import CardBox, read_table

TAILLE = (2560, 1529)
FEUTRES = ("feutre vert", "bleu nuit", "gris moyen")


def _boite(v) -> CardBox:
    if isinstance(v, CardBox):
        return v
    x, y, w, h = v
    return CardBox(x=x, y=y, w=w, h=h)


def _recouvrement(a: CardBox, b: CardBox) -> float:
    """Part de la plus petite des deux boîtes couverte par l'intersection."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / min(a.w * a.h, b.w * b.h) if inter else 0.0


def _localisees(table, largeur: int) -> tuple[int, int, int]:
    """(cartes retrouvées, rôles justes, cartes attendues) pour une largeur."""
    image = table.image
    if largeur:
        cherchee, k = _reduire(image, largeur)
        remise = lambda b: _agrandir(b, k, image.width, image.height)  # noqa: E731
    else:
        cherchee, k, remise = image, 1.0, (lambda b: b)
    lu = read_table(cherchee)
    boites = [(r, remise(b)) for r in ("hero", "board", "others")
              for b in getattr(lu, r)]

    verite = [("hero", _boite(v)) for v in table.hero_boxes]
    verite += [("board", _boite(v)) for v in table.board_boxes]

    prises: set[int] = set()
    trouvees = roles = 0
    for role_vrai, cible in verite:
        meilleur, score = None, 0.0
        for idx, (_, bb) in enumerate(boites):
            if idx in prises:
                continue
            s = _recouvrement(cible, bb)
            if s > score:
                meilleur, score = idx, s
        if meilleur is not None and score >= 0.5:
            prises.add(meilleur)
            trouvees += 1
            if boites[meilleur][0] == role_vrai:
                roles += 1
    return trouvees, roles, len(verite)


def _tables(theme: str = "pmu_solid", carte=(68, 92), bruit=0.0):
    for i, feutre in enumerate(FEUTRES):
        yield render_table(TableSpec(
            hero=("Ah", "Kd"), board=("Ts", "4h", "9d", "2c", "Kh"),
            felt=feutre, size=TAILLE, theme=theme,
            card_w=carte[0], card_h=carte[1], noise=bruit,
            decor=True, seed=4242 + i))


def test_la_pleine_echelle_localise_tout() -> None:
    """Aucune carte perdue, aucun rôle inversé, sur trois feutres."""
    for table in _tables():
        trouvees, roles, attendues = _localisees(table,
                                                 LARGEUR_SANS_REDUCTION)
        assert trouvees == attendues, (
            f"{attendues - trouvees} carte(s) perdue(s) à pleine échelle")
        assert roles == attendues, (
            f"{attendues - roles} rôle(s) faux à pleine échelle")


def test_la_pleine_echelle_tient_aussi_sur_le_deck_classique() -> None:
    """Le cas où la réduction s'effondre le plus (56 % à 1280)."""
    for table in _tables(theme="pmu_deck"):
        trouvees, roles, attendues = _localisees(table,
                                                 LARGEUR_SANS_REDUCTION)
        assert trouvees == attendues and roles == attendues


def test_la_reduction_perd_vraiment_des_cartes() -> None:
    """La réduction n'est PAS gratuite — c'est la raison de l'avoir rejetée.

    Si ce test venait à passer « trop bien » (plus aucune perte à 1280), ce
    serait le signe qu'un correctif du détecteur a changé la donne, et que
    la décision mérite d'être reprise — pas qu'il faut supprimer le test.
    """
    perdues = 0
    total = 0
    for table in _tables(theme="pmu_deck"):
        trouvees, _, attendues = _localisees(table, 1280)
        perdues += attendues - trouvees
        total += attendues
    assert perdues > 0, (
        "réduire à 1280 ne perd plus aucune carte du deck classique : "
        "re-mesure avec banc_localisation.py --large avant de conclure, "
        "puis reprends la décision de LARGEUR_SANS_REDUCTION.")


def test_agrandir_reste_dans_l_image() -> None:
    """Une boîte au bord ne déborde jamais après remise à l'échelle."""
    coin = CardBox(x=0, y=0, w=45, h=60, role="hero")
    b = _agrandir(coin, 2.0, 2560, 1529)
    assert b.x >= 0 and b.y >= 0
    assert b.x + b.w <= 2560 and b.y + b.h <= 1529

    bord = CardBox(x=1275, y=760, w=5, h=5, role="board")
    b = _agrandir(bord, 2.0, 2560, 1529)
    assert b.x + b.w <= 2560 and b.y + b.h <= 1529, (
        "une boîte collée au bord droit déborde après agrandissement")


def test_reduction_conserve_les_proportions() -> None:
    """L'image réduite garde le rapport largeur/hauteur, et le facteur est juste."""
    from PIL import Image

    image = Image.new("RGB", TAILLE, (10, 90, 40))
    reduite, k = _reduire(image, 1280)
    assert reduite.width == 1280
    assert abs(reduite.width / reduite.height
               - image.width / image.height) < 0.01
    assert abs(k - image.width / reduite.width) < 1e-9

    # Une image déjà petite n'est pas touchée : la réduire l'aurait dégradée
    # pour rien, et un facteur ≠ 1 déplacerait les boîtes.
    petite = Image.new("RGB", (800, 600), (0, 0, 0))
    inchangee, k1 = _reduire(petite, 1280)
    assert inchangee.size == petite.size and k1 == 1.0


@pytest.mark.parametrize("largeur", [0, 1280])
def test_les_deux_chemins_rendent_des_boites_dans_l_image(largeur) -> None:
    """Quelle que soit la largeur, aucune boîte ne sort du cadre."""
    table = next(iter(_tables()))
    image = table.image
    if largeur:
        cherchee, k = _reduire(image, largeur)
        boites = [_agrandir(b, k, image.width, image.height)
                  for r in ("hero", "board", "others")
                  for b in getattr(read_table(cherchee), r)]
    else:
        boites = [b for r in ("hero", "board", "others")
                  for b in getattr(read_table(image), r)]
    for b in boites:
        assert 0 <= b.x and 0 <= b.y
        assert b.x + b.w <= image.width and b.y + b.h <= image.height
