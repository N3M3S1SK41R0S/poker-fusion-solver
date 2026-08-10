"""La réduction avant localisation ne doit rien coûter en cartes trouvées.

`pfs.vision.live` réduit la capture à 1280 px de large avant de localiser
les cartes, puis découpe dans l'image d'origine. C'est ce qui rend la boucle
de calibration utilisable : à pleine échelle, localiser sur une capture
2560×1529 prend 7,6 s.

Le risque du raccourci est double — perdre des cartes devenues trop petites,
ou mal replacer les boîtes après remise à l'échelle. Ces deux tests le
mesurent sur des tables synthétiques de la taille d'un vrai écran.
"""

from __future__ import annotations

import pytest

from pfs.vision.live import LARGEUR_DETECTION, _agrandir, _reduire
from pfs.vision.synth_table import TableSpec, render_table
from pfs.vision.table_detector import CardBox, read_table

TAILLE = (2560, 1529)


def _recouvrement(a: CardBox, b: CardBox) -> float:
    """Part de la plus petite des deux boîtes couverte par l'intersection."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / min(a.w * a.h, b.w * b.h) if inter else 0.0


@pytest.fixture(scope="module")
def table():
    return render_table(TableSpec(
        hero=("Ah", "Kd"), board=("Ts", "4h", "9d", "2c", "Kh"),
        felt="feutre vert", size=TAILLE, decor=True, seed=4242))


def test_reduction_conserve_les_proportions() -> None:
    """L'image réduite garde le rapport largeur/hauteur, et le facteur est juste."""
    from PIL import Image

    image = Image.new("RGB", TAILLE, (10, 90, 40))
    reduite, k = _reduire(image)
    assert reduite.width == LARGEUR_DETECTION
    assert abs(reduite.width / reduite.height
               - image.width / image.height) < 0.01
    assert abs(k - image.width / reduite.width) < 1e-9
    # Une image déjà petite n'est pas touchée : réduire l'aurait dégradée
    # pour rien, et un facteur ≠ 1 déplacerait les boîtes.
    petite = Image.new("RGB", (800, 600), (0, 0, 0))
    inchangee, k1 = _reduire(petite)
    assert inchangee.size == petite.size and k1 == 1.0


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


def _boite(v) -> CardBox:
    """Vérité-terrain du générateur, ramenée à une `CardBox`."""
    if isinstance(v, CardBox):
        return v
    x, y, w, h = v
    return CardBox(x=x, y=y, w=w, h=h)


def test_la_reduction_ne_perd_aucune_carte(table) -> None:
    """Toutes les VRAIES cartes restent localisées après réduction.

    La comparaison porte sur la vérité-terrain du générateur, pas sur ce que
    trouve le détecteur à pleine échelle : celui-ci produit aussi des boîtes
    fantômes (32 sur le banc de 48 tables), que la réduction élimine
    justement en lissant les petits décors. Exiger de retrouver ses fantômes
    reviendrait à protéger un défaut.
    """
    image = table.image
    reduite, k = _reduire(image)
    trouvees = read_table(reduite)
    boites = [_agrandir(b, k, image.width, image.height)
              for b in list(trouvees.hero) + list(trouvees.board)
              + list(trouvees.others)]

    attendues = [_boite(v) for v in
                 list(table.hero_boxes) + list(table.board_boxes)]
    assert attendues, "le générateur doit fournir une vérité-terrain"

    manquantes = [
        v for v in attendues
        if max((_recouvrement(v, c) for c in boites), default=0.0) < 0.5
    ]
    assert not manquantes, (
        f"{len(manquantes)} carte(s) perdue(s) par la réduction à "
        f"{LARGEUR_DETECTION} px : {[(b.x, b.y) for b in manquantes]}")


def test_la_reduction_ne_cree_pas_de_fantome(table) -> None:
    """Aucune boîte détectée qui ne corresponde à une vraie carte.

    Une boîte fantôme envoie une découpe de feutre au recogniseur, qui la
    refuse et l'archive : le banc d'échecs se remplit de bruit et la mesure
    perd son sens.
    """
    image = table.image
    reduite, k = _reduire(image)
    trouvees = read_table(reduite)
    boites = [_agrandir(b, k, image.width, image.height)
              for b in list(trouvees.hero) + list(trouvees.board)
              + list(trouvees.others)]
    attendues = [_boite(v) for v in
                 list(table.hero_boxes) + list(table.board_boxes)]

    fantomes = [
        c for c in boites
        if max((_recouvrement(v, c) for v in attendues), default=0.0) < 0.5
    ]
    assert not fantomes, (
        f"{len(fantomes)} boîte(s) sans carte : "
        f"{[(b.x, b.y, b.w, b.h) for b in fantomes]}")


def test_les_roles_survivent_a_la_reduction(table) -> None:
    """Héros et board restent correctement séparés après réduction."""
    reduite, _ = _reduire(table.image)
    lu = read_table(reduite)
    assert len(lu.hero) == len(table.hero_boxes), (
        f"{len(lu.hero)} carte(s) héros lues pour "
        f"{len(table.hero_boxes)} attendues")
    assert len(lu.board) == len(table.board_boxes), (
        f"{len(lu.board)} carte(s) de board lues pour "
        f"{len(table.board_boxes)} attendues")
