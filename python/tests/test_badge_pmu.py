"""Le détecteur de badge « PMU PLAY » — tests rapides, sans les 57 captures.

Ce que ces tests garantissent
-----------------------------
1. **Les sprites livrés se retrouvent eux-mêmes** — recollés sur un fond
   synthétique aux positions calibrées, chaque marqueur rend un score ~1,0
   et ``vu=True``. C'est le test de non-corruption des gabarits : si un
   sprite du dépôt est altéré, c'est ici que ça casse.
2. **Le jitter mesuré (±4 px) est réellement toléré**, et pas un pixel de
   plus n'est promis.
3. **Fond nu et tables synthétiques sans badge** ⇒ ``vu=False`` — les mêmes
   négatifs que l'étude (dos bleus, aucun filigrane).
4. **Taille non calibrée ⇒ refus explicite** (fail-closed) : la robustesse
   multi-échelles n'est pas mesurée, le détecteur ne doit pas deviner.
5. **La lecture ne décide rien** : ``vu=False`` sans refus n'est qu'un
   « rien vu sur CETTE frame » — c'est le gate qui interprétera.

Ce que ces tests ne garantissent pas
------------------------------------
Les seuils (0,25 / 0,80) viennent des 57 captures réelles (~108 Mo, non
versionnées) ; leur validité se rejoue avec `banc_badge_pmu.py`. Le dernier
test rejoue 3 captures réelles quand l'archive locale existe — il est SAUTÉ
sinon, et il ne faut pas le confondre avec une garantie permanente.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from pfs.vision.badge_pmu import (
    BOITE_FILIGRANE_CLAIR,
    BOITE_FILIGRANE_SOMBRE,
    SEUIL_DOS,
    SEUIL_FILIGRANE,
    TAILLE_CALIBREE,
    _DOSSIER_SPRITES,
    detecter_badge,
)

CAPTURES = (Path(os.environ.get("LOCALAPPDATA") or str(Path.home()))
            / "PokerFusionSolver" / "captures" / "sessions")
SESSION_CLAIRE = CAPTURES / "100_-_6-max_Turbo_Re-buy_1197329110"
SESSION_SOMBRE = CAPTURES / "300_7-max_KO_1197327559"


def _fond(couleur: tuple[int, int, int] = (20, 83, 45)) -> Image.Image:
    """Un feutre uni à la taille calibrée — aucun badge dessus."""
    return Image.new("RGB", TAILLE_CALIBREE, couleur)


def _sprite(nom: str) -> Image.Image:
    return Image.open(_DOSSIER_SPRITES / nom).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LES SPRITES SE RETROUVENT EUX-MÊMES AUX POSITIONS CALIBRÉES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("nom", "boite", "theme"), [
    ("sprite_pmu_play_clair.png", BOITE_FILIGRANE_CLAIR, "clair"),
    ("sprite_pmu_play_sombre.png", BOITE_FILIGRANE_SOMBRE, "sombre"),
])
def test_filigrane_recolle_est_vu(nom: str, boite: tuple[int, int, int, int],
                                  theme: str) -> None:
    """Sprite recollé à sa position calibrée : score ~1,0, thème identifié."""
    image = _fond()
    image.paste(_sprite(nom), boite[:2])
    lu = detecter_badge(image)
    assert lu.refus is None
    assert lu.filigrane_score > 0.99, "le sprite ne se retrouve pas lui-même"
    assert lu.vu
    assert lu.theme == theme
    assert lu.position == boite[:2]


def test_filigrane_tolere_le_jitter_calibre() -> None:
    """±4 px de dérive (le jitter mesuré) ne coûtent pas la détection."""
    image = _fond()
    x, y = BOITE_FILIGRANE_CLAIR[:2]
    image.paste(_sprite("sprite_pmu_play_clair.png"), (x + 4, y - 4))
    lu = detecter_badge(image)
    assert lu.vu and lu.theme == "clair"
    assert lu.filigrane_score > 0.99


def test_dos_est_vu_partout() -> None:
    """Le dos orange se cherche PLEIN CADRE : recollé loin de sa boîte
    d'origine (les sièges varient), il est vu, et sa position est rendue."""
    image = _fond()
    image.paste(_sprite("sprite_dos_pmu_play.png"), (300, 900))
    lu = detecter_badge(image)
    assert lu.refus is None
    assert lu.dos_score > 0.99
    assert lu.vu
    assert lu.theme is None, "le dos seul ne révèle pas le thème"
    assert lu.position == (300, 900)
    assert lu.filigrane_score < SEUIL_FILIGRANE


# ═══════════════════════════════════════════════════════════════════════════
# 2. RIEN À VOIR ⇒ RIEN N'EST VU
# ═══════════════════════════════════════════════════════════════════════════

def test_fond_nu_rien_vu() -> None:
    """Feutre uni : aucun marqueur, aucun refus — juste « rien vu »."""
    lu = detecter_badge(_fond())
    assert lu.refus is None
    assert not lu.vu
    assert lu.position is None and lu.theme is None
    assert lu.filigrane_score < SEUIL_FILIGRANE
    assert lu.dos_score < SEUIL_DOS


@pytest.mark.parametrize("felt", ["feutre vert", "bordeaux"])
def test_table_synthetique_sans_badge(felt: str) -> None:
    """Les négatifs de l'étude : tables du dépôt, dos bleus, sans filigrane.

    « bordeaux » est le pire cas mesuré (dos 0,569, et 44 991 avant le
    plancher de variance σ ≥ 5) ; « feutre vert » le tapis nominal.
    """
    from pfs.vision.synth_table import TableSpec, render_table

    table = render_table(TableSpec(
        hero=("Ah", "Kd"), board=("Ts", "4h", "9d"), felt=felt,
        size=TAILLE_CALIBREE, theme="pmu_solid",
        card_w=102, card_h=138, decor=True, seed=1))
    lu = detecter_badge(table.image)
    assert lu.refus is None
    assert not lu.vu, (f"faux positif sur {felt} : filigrane "
                       f"{lu.filigrane_score:.3f}, dos {lu.dos_score:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TAILLE NON CALIBRÉE ⇒ REFUS EXPLICITE (FAIL-CLOSED)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("taille", [(1280, 720), (1920, 1080), (960, 680)])
def test_mauvaise_taille_refusee(taille: tuple[int, int]) -> None:
    """Autre taille = autre échelle, jamais mesurée : refus, pas de devinette."""
    lu = detecter_badge(Image.new("RGB", taille, (20, 83, 45)))
    assert lu.refus is not None
    assert "non calibré" in lu.refus
    assert f"{taille[0]}×{taille[1]}" in lu.refus, "le refus doit dire ce qu'il a reçu"
    assert not lu.vu
    assert lu.filigrane_score == 0.0 and lu.dos_score == 0.0


def test_le_refus_prime_sur_le_contenu() -> None:
    """Même un vrai sprite recollé sur une image hors calibration est refusé :
    le score qu'on lui donnerait n'aurait pas été mesuré."""
    image = Image.new("RGB", (960, 680), (20, 83, 45))
    image.paste(_sprite("sprite_dos_pmu_play.png"), (100, 100))
    lu = detecter_badge(image)
    assert lu.refus is not None and not lu.vu


# ═══════════════════════════════════════════════════════════════════════════
# 4. TROIS CAPTURES RÉELLES (sauté si l'archive locale est absente)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (SESSION_CLAIRE / "0000.png").exists()
    or not (SESSION_SOMBRE / "0010.png").exists(),
    reason="captures réelles absentes (~108 Mo, non versionnées) — "
           "rejouer banc_badge_pmu.py quand elles existent")
def test_trois_captures_reelles() -> None:
    """0000 clair (les deux marqueurs), 0010 sombre, 0026 fin de main (rien).

    Les scores attendus sont ceux de l'étude : filigrane 1,0 sur les frames
    sources des sprites, dos 1,0, et 0026 = abattage où board et bandeau
    SIT OUT masquent les deux marqueurs (filigrane ≤ 0,086, dos 0,630).
    """
    lu = detecter_badge(Image.open(SESSION_CLAIRE / "0000.png"))
    assert lu.vu and lu.theme == "clair"
    assert lu.filigrane_score > 0.99 and lu.dos_score > 0.99

    lu = detecter_badge(Image.open(SESSION_SOMBRE / "0010.png"))
    assert lu.vu and lu.theme == "sombre"
    assert lu.filigrane_score > 0.99

    lu = detecter_badge(Image.open(SESSION_CLAIRE / "0026.png"))
    assert not lu.vu and lu.refus is None, \
        "0026 est une fin de main SANS marqueur : la voir serait un faux positif"
