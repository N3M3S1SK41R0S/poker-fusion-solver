"""Une découpe qui n'est pas une carte ne doit JAMAIS sortir « sure ».

Défaut constaté lors de la première lecture en direct : une découpe de décor
(42×56 px, sur le feutre) a été lue « 4h », statut « sure », à un écart de
703 — parce que sa marge valait 44. La marge dit qu'un gabarit devance ses
concurrents ; elle ne dit pas qu'il ressemble à l'image. Sur une découpe qui
ne ressemble à rien, le classement des gabarits est arbitraire, et l'écart
entre le premier et le deuxième l'est tout autant.

C'est le pire mode d'échec possible pour cet outil : un refus se voit et
s'archive, une carte inventée avec aplomb se propage. `DISTANCE_SURE` exige
désormais que la lecture affirmée soit aussi PROCHE du gabarit.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pfs.vision.card_recognizer import (
    DISTANCE_SURE,
    MAX_ACCEPT_DISTANCE,
    identify_card,
)

#: Teintes de feutre rencontrées sur les tables (vert, clair, bleu, bordeaux,
#: gris, noir) : le fond sur lequel les fausses découpes sont prélevées.
FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70),
           (92, 26, 38), (84, 84, 90), (16, 16, 20))


def _bruit(rng) -> Image.Image:
    h, w = int(rng.integers(40, 120)), int(rng.integers(30, 90))
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def _feutre(rng, i: int) -> Image.Image:
    c = FEUTRES[i % len(FEUTRES)]
    h, w = int(rng.integers(40, 120)), int(rng.integers(30, 90))
    a = np.tile(np.array(c, dtype=np.uint8), (h, w, 1))
    a = np.clip(a.astype(int) + rng.normal(0, 4, a.shape), 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def _dos(rng) -> Image.Image:
    """Dos de carte : aplat coloré strié — illisible par nature."""
    h, w = int(rng.integers(60, 120)), int(rng.integers(45, 90))
    img = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(60, 200, 3)))
    d = ImageDraw.Draw(img)
    for k in range(0, w + h, 7):
        d.line([(k, 0), (0, k)],
               fill=tuple(int(x) for x in rng.integers(0, 120, 3)), width=2)
    return img


def _jeton(rng) -> Image.Image:
    h, w = int(rng.integers(40, 100)), int(rng.integers(35, 85))
    img = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(0, 80, 3)))
    d = ImageDraw.Draw(img)
    d.ellipse([w * 0.1, h * 0.1, w * 0.9, h * 0.9],
              fill=tuple(int(x) for x in rng.integers(120, 255, 3)))
    d.text((w * 0.35, h * 0.4), str(int(rng.integers(0, 99))), fill=(0, 0, 0))
    return img


def _non_cartes(n: int = 240):
    rng = np.random.default_rng(20260810)
    fabriques = (lambda i: _bruit(rng), lambda i: _feutre(rng, i),
                 lambda i: _dos(rng), lambda i: _jeton(rng))
    for i in range(n):
        yield fabriques[i % len(fabriques)](i)


def test_aucune_non_carte_n_est_affirmee() -> None:
    """240 découpes sans carte : aucune ne doit ressortir « sure »."""
    affirmees = [identify_card(img) for img in _non_cartes()]
    fautives = [m for m in affirmees if m.statut == "sure"]
    assert not fautives, (
        f"{len(fautives)} non-carte(s) affirmée(s) : "
        f"{[(m.card, m.distance, m.margin) for m in fautives[:5]]}")
    assert all(m.card is None for m in affirmees), (
        "une carte a été rendue alors qu'aucune découpe n'en contient")


def test_le_plancher_de_bruit_reste_au_dessus_du_seuil() -> None:
    """L'écart minimal d'une non-carte doit rester au-dessus de DISTANCE_SURE.

    C'est l'hypothèse qui rend le seuil valide. Si elle tombe — parce que le
    hachage a changé, ou parce qu'on a ajouté un habillage — le seuil doit
    être re-mesuré, pas le test supprimé.
    """
    plancher = min(identify_card(img).distance for img in _non_cartes())
    assert plancher > DISTANCE_SURE, (
        f"une non-carte atteint l'écart {plancher}, sous le seuil "
        f"DISTANCE_SURE={DISTANCE_SURE} : le seuil n'est plus protecteur.")


def test_le_seuil_tombe_dans_le_vide_entre_les_deux_populations() -> None:
    """Le seuil sépare deux nuages disjoints, il ne coupe pas dans l'un d'eux.

    C'est ce qui distingue un seuil mesuré d'un seuil réglé à la main : les
    vraies cartes cadrées plafonnent à 599, les non-cartes démarrent à 658.
    Tant que le vide existe, le seuil est robuste des deux côtés ; s'il se
    referme, aucune valeur ne conviendra et il faudra un autre signal que la
    distance.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import test_vision as tv
    except ImportError:  # pragma: no cover - dépend du banc d'habillages
        pytest.skip("banc d'habillages indisponible")

    cadre = tv.TestThemesAndBackgrounds()
    hauts = []
    for theme in ("pmu_deck", "pmu_solid"):
        for _, feutre in list(cadre.FELTS.items())[:3]:
            for c in tv._cards(theme):
                m = identify_card(cadre._framed(theme, c, feutre))
                if m.best_guess == c:
                    hauts.append(m.distance)
    assert hauts, "aucune carte cadrée n'a été identifiée"

    plafond_vraies = max(hauts)
    plancher_fausses = min(identify_card(i).distance for i in _non_cartes())
    assert plafond_vraies < DISTANCE_SURE < plancher_fausses, (
        f"le seuil {DISTANCE_SURE} ne tombe plus dans le vide : vraies "
        f"cartes jusqu'à {plafond_vraies}, non-cartes à partir de "
        f"{plancher_fausses}.")


def test_les_vrais_gabarits_restent_affirmes() -> None:
    """Le durcissement ne doit rien coûter aux cartes réellement lisibles."""
    from pathlib import Path

    from pfs.vision.card_recognizer import _DECK_DIR

    dossier = Path(_DECK_DIR)
    if not dossier.is_dir():
        pytest.skip("gabarits absents")
    pngs = sorted(dossier.glob("*.png"))
    assert pngs, "aucun gabarit à vérifier"
    for png in pngs:
        m = identify_card(png)
        assert m.statut == "sure", (
            f"{png.stem} n'est plus affirmée (écart {m.distance}, "
            f"marge {m.margin})")
        assert m.card == png.stem


def test_les_seuils_restent_ordonnes() -> None:
    """Le seuil d'affirmation ne dépasse jamais celui de plausibilité.

    Affirmer une carte que l'on juge par ailleurs implausible n'aurait aucun
    sens. L'égalité, elle, est légitime et c'est l'état actuel : les deux
    seuils ont convergé vers le même vide mesuré entre les vraies cartes
    (≤ 599) et les non-cartes (≥ 658). Au-dessus, rien n'est plausible ;
    en dessous, c'est la marge qui départage « sure » de « propose ».

    Ce test a été écrit avec un `<` strict alors que le plafond valait encore
    900 ; il est passé au `<=` quand la mesure sur capture réelle a fait
    descendre le plafond. La contrainte réelle a toujours été celle-ci.
    """
    assert DISTANCE_SURE <= MAX_ACCEPT_DISTANCE
