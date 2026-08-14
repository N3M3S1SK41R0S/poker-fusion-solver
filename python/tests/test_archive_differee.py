"""L'archivage des échecs de ``lire_capture`` est différé, jamais perdu.

Le chantier vitesse du 14 août 2026 a sorti les écritures disque de l'archive
du chemin de la réponse (thread nommé ``pfs-archive-echecs``, non daemon).
Ce fichier verrouille les trois propriétés qui rendent ce déplacement sûr :

1. la réponse de la route est STRICTEMENT celle d'avant — aucune clé
   d'archive, les mêmes cartes, les mêmes statuts ;
2. chaque découpe non sûre d'un rôle héros/board finit BIEN sur disque,
   avec le même diagnostic que l'archivage synchrone écrivait (statut,
   distance, marge, rôle, origine « collage », boîte) ;
3. le dossier de destination est celui de la REQUÊTE : résolu pendant
   l'appel, pas au moment de l'écriture — un ``LOCALAPPDATA`` restauré
   entre-temps (ce que font tous les tests) n'égare aucun fichier.

La carte illisible est fabriquée sans rien mocker : l'intérieur d'une carte
du héros est recouvert d'un aplat étranger aux teintes du jeu. La détection
garde la boîte (bords et fond blanc intacts, intérieur calme) ; la
reconnaissance, elle, refuse — le chemin réel de bout en bout.
"""

from __future__ import annotations

import base64
import io
import json
import threading

import pytest

from pfs.app.server import API
from pfs.vision.synth_table import TableSpec, render_table


def _capture_avec_carte_illisible() -> tuple[str, int]:
    """Une table dont la 1re carte du héros est détectable mais illisible.

    Returns
    -------
    tuple
        (image_b64, nombre de cartes du héros attendues).
    """
    spec = TableSpec(hero=("Ah", "Kd"), board=("2c", "7d", "Jh"),
                     size=(1200, 760), decor=False)
    synth = render_table(spec)
    img = synth.image
    x, y, w, h = synth.hero_boxes[0]
    # Aplat posé à 6 px du bord : les arêtes de la carte restent nettes, la
    # marge blanche reste blanche, mais le glyphe disparaît sous une teinte
    # qui n'est d'aucune famille du deck — la lecture doit refuser.
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([x + 6, y + 6, x + w - 7, y + h - 7], fill=(96, 128, 96))
    tampon = io.BytesIO()
    img.save(tampon, format="PNG")
    return base64.b64encode(tampon.getvalue()).decode("ascii"), 2


def _rejoindre_archives() -> None:
    """Attend la fin de tous les threads d'archive en vol."""
    for t in threading.enumerate():
        if t.name == "pfs-archive-echecs":
            t.join(timeout=30)


def test_les_echecs_finissent_sur_disque_dans_le_dossier_de_la_requete(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    b64, _ = _capture_avec_carte_illisible()

    rep = API.lire_capture({"image_b64": b64})

    # 1. La réponse est celle du contrat historique : rien de l'archive n'y
    #    figure, et la carte sabotée n'est PAS affirmée.
    assert not {"archive", "echecs", "chemin"} & set(rep)
    rates = [c for c in rep["hero"] + rep["board"] if c["statut"] != "sure"]
    assert rates, "le sabotage n'a produit aucun échec — le test ne teste rien"

    # 2. Chaque échec est sur disque, diagnostic identique à ce que
    #    l'archivage synchrone écrivait.
    _rejoindre_archives()
    d = tmp_path / "PokerFusionSolver" / "captures" / "echecs"
    pngs = sorted(d.glob("*.png"))
    assert len(pngs) == len(rates), (
        f"{len(rates)} échec(s) dans la réponse, {len(pngs)} sur disque")
    diags = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
             for p in pngs]
    for diag in diags:
        assert diag["origine"] == "collage"
        assert diag["role"] in ("hero", "board")
        assert {"statut", "distance", "margin", "best_guess",
                "boite"} <= set(diag)
    # le diagnostic reprend exactement les statuts/boîtes de la réponse
    assert sorted((d["statut"], tuple(d["boite"])) for d in diags) == sorted(
        (c["statut"], tuple(c["boite"])) for c in rates)


def test_une_capture_sans_echec_ne_cree_aucun_thread_d_archive(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    spec = TableSpec(hero=("Ah", "Kd"), board=(), size=(1200, 760),
                     decor=False)
    tampon = io.BytesIO()
    render_table(spec).image.save(tampon, format="PNG")
    b64 = base64.b64encode(tampon.getvalue()).decode("ascii")

    rep = API.lire_capture({"image_b64": b64})
    assert rep["main"] == ["Ah", "Kd"]
    _rejoindre_archives()
    d = tmp_path / "PokerFusionSolver" / "captures" / "echecs"
    assert sorted(d.glob("*.png")) == [], (
        "une lecture entièrement sûre ne doit rien archiver")


def test_le_dossier_est_fige_avant_le_thread(tmp_path, monkeypatch) -> None:
    """Le point 3 : l'écriture atterrit dans le dossier de la requête même si
    l'environnement change juste après la réponse — c'est la situation de
    chaque test qui restaure ``LOCALAPPDATA`` à sa sortie."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    b64, _ = _capture_avec_carte_illisible()
    rep = API.lire_capture({"image_b64": b64})
    autre = tmp_path / "ailleurs"
    monkeypatch.setenv("LOCALAPPDATA", str(autre))   # trop tard, et c'est voulu
    _rejoindre_archives()
    d = tmp_path / "PokerFusionSolver" / "captures" / "echecs"
    rates = [c for c in rep["hero"] + rep["board"] if c["statut"] != "sure"]
    assert len(sorted(d.glob("*.png"))) == len(rates)
    assert not (autre / "PokerFusionSolver").exists()
