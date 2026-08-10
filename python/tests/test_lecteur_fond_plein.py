"""Lire une carte à fond plein : la couleur donne la famille, le chiffre le rang.

L'idée vient de Pierre, et elle est meilleure que le hachage perceptuel sur
cet habillage. Le hachage compare la carte ENTIÈRE : il paie donc le bandeau
d'équité qui masque le bas, un cadrage à sept pixels près, et l'échelle. Un
jeu « 4 couleurs à fond plein » ne demande rien de tout cela — la teinte est
l'enseigne, et il ne reste qu'un chiffre blanc à reconnaître.

Ces tests épinglent les trois propriétés qui font la valeur de la méthode :
elle lit juste, elle est indifférente au cadrage, et elle refuse ce qui n'est
pas une carte.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pfs.vision.card_recognizer import _TEMPLATE_ROOT
from pfs.vision.lecteur_fond_plein import (
    ECART_COULEUR_MAX,
    FAMILLES,
    RANGS,
    couleur_dominante,
    famille_de_couleur,
    lire_carte_fond_plein,
)

_SOLID = _TEMPLATE_ROOT / "pmu_solid"
FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70),
           (92, 26, 38), (84, 84, 90), (16, 16, 20))


@pytest.fixture(scope="module")
def gabarits():
    if not _SOLID.is_dir():
        pytest.skip("habillage pmu_solid absent")
    return _SOLID


def test_les_52_cartes_se_lisent(gabarits) -> None:
    """Aucune carte manquée, aucune confondue."""
    faux, refus = [], []
    for r in RANGS:
        for s in FAMILLES:
            p = gabarits / f"{r}{s}.png"
            if not p.exists():
                continue
            m = lire_carte_fond_plein(p)
            if m.carte is None:
                refus.append((f"{r}{s}", m.motif))
            elif m.carte != f"{r}{s}":
                faux.append((f"{r}{s}", m.carte))
    assert not faux, f"cartes confondues : {faux}"
    assert not refus, f"cartes refusées : {refus}"


def test_les_quatre_familles_sont_separees(gabarits) -> None:
    """La teinte de fond identifie l'enseigne sans ambiguïté.

    C'est l'hypothèse qui fonde toute la méthode : si deux familles se
    rapprochaient, le seuil de couleur cesserait de protéger.
    """
    teintes = {}
    for s in FAMILLES:
        vues = []
        for r in RANGS:
            p = gabarits / f"{r}{s}.png"
            if p.exists():
                vues.append(couleur_dominante(p)[0])
        assert vues
        teintes[s] = np.mean(vues, axis=0)
        dispersion = float(np.max(np.std(vues, axis=0)))
        assert dispersion < 5.0, (
            f"la teinte de {s} varie d'un rang à l'autre ({dispersion:.1f})")

    familles = list(teintes)
    for i, a in enumerate(familles):
        for b in familles[i + 1:]:
            d = float(np.linalg.norm(teintes[a] - teintes[b]))
            assert d > 2 * ECART_COULEUR_MAX, (
                f"{a} et {b} ne sont séparées que de {d:.1f}, pour un seuil "
                f"de {ECART_COULEUR_MAX} : une confusion devient possible")


def test_une_teinte_quelconque_n_est_aucune_famille() -> None:
    """Le filtre de couleur est ce qui protège des non-cartes."""
    rng = np.random.default_rng(7)
    acceptees = 0
    for _ in range(400):
        c = rng.integers(0, 256, 3).astype(float)
        if famille_de_couleur(c)[0] is not None:
            acceptees += 1
    assert acceptees / 400 < 0.05, (
        f"{acceptees}/400 teintes au hasard passent pour une famille : le "
        "seuil de couleur est trop large")


def _non_cartes(n: int = 60):
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


def test_rien_qui_ne_soit_une_carte_n_est_lu() -> None:
    """Bruit, feutre, dos de cartes : tout doit être refusé."""
    lues = [m for m in (lire_carte_fond_plein(i) for i in _non_cartes())
            if m.carte is not None]
    assert not lues, (
        f"{len(lues)} non-carte(s) lue(s) : "
        f"{[(m.carte, round(m.ecart_couleur, 1)) for m in lues[:5]]}")


def test_la_lecture_survit_a_un_cadrage_approximatif(gabarits) -> None:
    """Élargir, resserrer ou décaler la découpe ne doit rien changer.

    C'est l'avantage décisif sur le hachage, qui perdait la carte pour sept
    pixels : le chiffre est cherché dans le coin, pas supposé à une place.
    """
    from PIL import Image as PILImage

    p = gabarits / "5c.png"
    if not p.exists():
        pytest.skip("gabarit 5c absent")
    src = PILImage.open(p).convert("RGB")
    w, h = src.size

    # Découpe posée sur du feutre, puis recadrée de plusieurs façons.
    marge = 14
    fond = PILImage.new("RGB", (w + 2 * marge, h + 2 * marge), FEUTRES[0])
    fond.paste(src, (marge, marge))

    cadrages = {
        "exact": (marge, marge, w, h),
        "large": (marge - 8, marge - 8, w + 16, h + 16),
        "serré": (marge + 4, marge + 3, w - 8, h - 6),
        "décalé": (marge + 5, marge + 4, w, h),
        "bas coupé": (marge, marge, w, int(h * 0.66)),
    }
    ratés = []
    for nom, (x, y, ww, hh) in cadrages.items():
        m = lire_carte_fond_plein(fond.crop((x, y, x + ww, y + hh)))
        if m.carte != "5c":
            ratés.append((nom, m.carte, m.motif))
    assert not ratés, f"cadrages perdus : {ratés}"
