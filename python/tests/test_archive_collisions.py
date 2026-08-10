"""Deux échecs archivés dans la même seconde ne doivent pas s'écraser.

Défaut constaté en calibration réelle : une lecture de table soumet jusqu'à
huit découpes d'un coup. Les noms de fichiers étant horodatés à la seconde,
toutes celles qui partageaient le même statut se sont écrasées — l'archive
ne gardait qu'une découpe par seconde et par statut, sans qu'aucune erreur
ne le signale. Un banc d'essai qui perd sa matière en silence est pire
qu'aucun banc.
"""

from __future__ import annotations

import base64
import io

import pytest

from pfs.vision.archive import enregistrer_capture, enregistrer_echec


@pytest.fixture(autouse=True)
def archive_isolee(tmp_path, monkeypatch):
    """Écrit dans un dossier jetable, pas dans l'archive de l'utilisateur."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    return tmp_path


def _png(couleur: tuple[int, int, int]) -> str:
    from PIL import Image

    tampon = io.BytesIO()
    Image.new("RGB", (8, 12), couleur).save(tampon, format="PNG")
    return base64.b64encode(tampon.getvalue()).decode("ascii")


def test_huit_echecs_simultanes_sont_tous_conserves() -> None:
    """Une lecture de table complète archive ses huit découpes."""
    couleurs = [(i * 30, 10, 200 - i * 20) for i in range(8)]
    chemins = [enregistrer_echec(_png(c), {"statut": "refus", "distance": 700})
               for c in couleurs]

    assert len({p.name for p in chemins}) == 8, (
        f"collision de noms : {[p.name for p in chemins]}")
    assert all(p.exists() for p in chemins)
    # Les contenus doivent différer : un même nom réécrit produirait des
    # fichiers identiques, ce que la seule unicité des noms ne prouve pas.
    assert len({p.read_bytes() for p in chemins}) == 8


def test_le_diagnostic_suit_sa_decoupe() -> None:
    """Chaque image garde SON diagnostic, pas celui de la suivante."""
    import json

    a = enregistrer_echec(_png((255, 0, 0)),
                          {"statut": "refus", "distance": 111})
    b = enregistrer_echec(_png((0, 255, 0)),
                          {"statut": "refus", "distance": 222})
    assert a.name != b.name
    assert json.loads(a.with_suffix(".json").read_text(
        encoding="utf-8"))["distance"] == 111
    assert json.loads(b.with_suffix(".json").read_text(
        encoding="utf-8"))["distance"] == 222


def test_les_captures_entieres_ne_se_recouvrent_pas() -> None:
    """Deux captures collées coup sur coup restent deux fichiers."""
    a = enregistrer_capture(_png((1, 2, 3)))
    b = enregistrer_capture(_png((4, 5, 6)))
    assert a.name != b.name
    assert a.read_bytes() != b.read_bytes()
