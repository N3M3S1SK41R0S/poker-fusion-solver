"""L'ENSEIGNE d'un gabarit doit correspondre à son étiquette.

Pourquoi ce fichier existe
--------------------------
`pmu_deck/7h.png` et `7d.png` étaient INTERVERTIS depuis leur extraction. Le
vrai 7 de cœur était donc lu « 7d », statut « sure », à un écart de 0 et une
marge de 487 : une lecture fausse avec une confiance maximale, le pire mode
d'échec possible pour cet outil.

Le défaut a survécu à plus de mille tests parce que ceux qui vérifient les
gabarits sont TAUTOLOGIQUES : ils confrontent chaque gabarit à lui-même
(« test_every_template_identifies_itself »). Un gabarit mal étiqueté gagne
toujours contre lui-même, à distance 0. Aucune vérification de ce type ne
peut détecter un échange d'étiquettes — seule une mesure portant sur le
CONTENU de l'image le peut.

Le discriminant
---------------
Un cœur a deux lobes en haut : l'encre se répartit sur les côtés et laisse
le centre creux. Un carreau a une pointe unique : l'encre se concentre au
centre. On mesure donc, dans le haut du symbole, l'encre des côtés moins
celle du centre. Mesuré sur les 26 cartes rouges livrées ::

    cœurs    n=13   min −0,208   médiane −0,094   max +0,311
    carreaux n=13   min −0,533   médiane −0,508   max −0,320

Les deux populations sont séparées par un vide de 0,11, sans recouvrement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pfs.vision.card_recognizer import _TEMPLATE_ROOT

RANGS = "23456789TJQKA"

#: Frontière entre les deux populations, posée dans le vide mesuré
#: (cœurs ≥ −0,208, carreaux ≤ −0,320).
SEUIL_LOBES = -0.26


def _glyphe(chemin: Path) -> np.ndarray | None:
    """Le SYMBOLE seul, normalisé en 24×24.

    Il occupe la moitié basse de la vignette, sous le rang. On l'isole par
    sa boîte englobante d'encre pour que la mesure ne dépende ni de la
    taille du gabarit ni de la place que prend le rang au-dessus.
    """
    a = np.asarray(Image.open(chemin).convert("RGB"), dtype=float)
    bas = a[int(a.shape[0] * 0.52):, :, :]
    encre = bas.min(axis=2) < 200
    ys, xs = np.where(encre)
    if not len(ys):
        return None
    g = encre[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    fixe = Image.fromarray((g * 255).astype(np.uint8)).resize(
        (24, 24), Image.LANCZOS)
    return np.asarray(fixe, dtype=float) / 255.0


def _lobes(g: np.ndarray) -> float:
    """Encre des côtés moins encre du centre, dans le haut du symbole.

    Positif ou proche de zéro : deux lobes, donc un cœur. Nettement négatif :
    une pointe unique, donc un carreau.
    """
    haut = g[:6, :]
    centre = haut[:, 9:15].mean()
    cotes = (haut[:, 3:9].mean() + haut[:, 15:21].mean()) / 2
    return float(cotes - centre)


def _rouges(theme: str):
    d = _TEMPLATE_ROOT / theme
    for r in RANGS:
        for s in "hd":
            p = d / f"{r}{s}.png"
            if p.exists():
                yield f"{r}{s}", s, p


def test_pmu_deck_existe() -> None:
    assert (_TEMPLATE_ROOT / "pmu_deck").is_dir(), "gabarits pmu_deck absents"


def test_chaque_carte_rouge_porte_la_bonne_enseigne() -> None:
    """Un cœur étiqueté doit ressembler à un cœur, un carreau à un carreau.

    C'est ce test — et lui seul — qui aurait attrapé l'échange 7h/7d.
    """
    fautives = []
    for nom, suit, chemin in _rouges("pmu_deck"):
        g = _glyphe(chemin)
        if g is None:
            continue
        v = _lobes(g)
        attendu_coeur = suit == "h"
        vu_coeur = v >= SEUIL_LOBES
        if attendu_coeur != vu_coeur:
            fautives.append((nom, round(v, 3),
                             "carreau" if attendu_coeur else "cœur"))
    assert not fautives, (
        "gabarit(s) dont l'enseigne contredit l'étiquette — une lecture "
        f"fausse et affirmée en découlerait : {fautives}")


def test_les_deux_populations_restent_separees() -> None:
    """Le seuil doit rester dans un vide, sinon il ne prouve plus rien.

    Si les deux nuages se recouvrent un jour (autre habillage, autre taille
    de vignette), ce test le dit avant que le précédent ne devienne un
    tirage à pile ou face.
    """
    coeurs, carreaux = [], []
    for _, suit, chemin in _rouges("pmu_deck"):
        g = _glyphe(chemin)
        if g is None:
            continue
        (coeurs if suit == "h" else carreaux).append(_lobes(g))
    assert coeurs and carreaux
    assert min(coeurs) > max(carreaux), (
        f"les enseignes ne se séparent plus : cœurs ≥ {min(coeurs):.3f}, "
        f"carreaux ≤ {max(carreaux):.3f}")
    assert max(carreaux) < SEUIL_LOBES < min(coeurs), (
        f"le seuil {SEUIL_LOBES} n'est plus dans le vide "
        f"[{max(carreaux):.3f}, {min(coeurs):.3f}]")


def test_le_controle_detecte_bien_un_echange() -> None:
    """Le test précédent échouerait-il si l'on ré-intervertissait 7h et 7d ?

    Sans cette vérification, rien ne garantit que le contrôle ait le moindre
    pouvoir de détection — c'est précisément le reproche fait aux tests
    tautologiques qu'il remplace.
    """
    d = _TEMPLATE_ROOT / "pmu_deck"
    gh, gd = _glyphe(d / "7h.png"), _glyphe(d / "7d.png")
    if gh is None or gd is None:
        pytest.skip("gabarits 7h/7d absents")
    # Étiquettes échangées : le cœur serait jugé carreau, et réciproquement.
    assert _lobes(gd) < SEUIL_LOBES <= _lobes(gh), (
        "l'échange 7h/7d ne serait pas détecté par ce contrôle")
