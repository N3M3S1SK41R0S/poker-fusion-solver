"""Drills depuis les fuites + emplacements + persistance — par les ROUTES RÉELLES.

Même doctrine que ``test_parcours_complet.py`` : un test qui appelle
``generer_drills_dossier()`` en Python ne prouve rien sur le routage HTTP ni
sur ce que l'onglet S'entraîner obtient. Ici on part de FICHIERS iPoker écrits
sur disque (fabriqués en ``tmp_path``, jamais les historiques réels de
l'utilisateur), on monte le VRAI serveur sur un port libre, et on lui parle en
HTTP avec urllib.

Ce que le parcours traverse
---------------------------
``POST /api/drill/fuites`` (dossier → drills → session SM-2) →
``POST /api/drill/fuites/answer`` (raté, le spot revient ; réussi, on avance)
→ fin de session avec score. Plus ``POST /api/emplacements`` (détection des
dossiers PMU, environnement contrôlé) et les vérifications STRUCTURELLES de la
page servie : le bouton « M'entraîner sur mes fuites », et la persistance
localStorage qui ne couvre QUE le chemin d'historiques et la fenêtre de
calibration — jamais les montants du spot.

Les EV goldens (AA/72o à 10 bb) sont celles de ``test_leak_drills.py``,
mesurées une fois sur le solveur du projet.

Limite assumée (NEMESIS)
------------------------
La persistance vit dans le JavaScript de la page et aucun moteur JS n'est
disponible ici : les tests « persistance » vérifient, sur les octets
réellement servis par ``GET /``, que localStorage n'est touché que par les
deux helpers et que chaque écriture porte une clé autorisée — pas que le
navigateur exécute le tout. C'est une vérification structurelle, nommée comme
telle ; le comportement réel se vérifie au navigateur.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pfs.app.server import create_server
from pfs.data import emplacements

# Le fabricant de mains iPoker et les EV goldens des drills : mêmes valeurs,
# même XML — dupliquer l'un ou l'autre ici finirait par les faire diverger.
from test_leak_drills import EV_72O_10, EV_AA_10, _main_xml

#: Jeton explicite : le serveur de test ne touche jamais au jeton persistant.
JETON = "jeton-du-parcours-fuites"


# ═══════════════════════════════════════════════════════════════════════════
# LE SERVEUR RÉEL, ET COMMENT ON LUI PARLE
# ═══════════════════════════════════════════════════════════════════════════


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base() -> str:
    """Le vrai serveur, sur un port libre, servi dans un fil de discussion."""
    srv, _ = create_server(_port_libre(), token=JETON)
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        fil.join(timeout=5)
        srv.server_close()


def _poster(base: str, route: str, charge: dict,
            jeton: str | None = JETON) -> tuple[int, dict]:
    """POST HTTP réel sur ``/api/<route>``. Renvoie (code, corps décodé)."""
    entetes = {"Content-Type": "application/json"}
    if jeton is not None:
        entetes["X-PFS-Token"] = jeton
    req = urllib.request.Request(
        f"{base}/api/{route}",
        data=json.dumps(charge).encode("utf-8"),
        headers=entetes, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as rep:
            return rep.status, json.loads(rep.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _obtenir(base: str, chemin: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{base}{chemin}", timeout=30) as rep:
            return rep.status, rep.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


@pytest.fixture()
def dossier_fuites(tmp_path: Path) -> Path:
    """Deux erreurs réelles fabriquées : AA couché (3,49 bb), 72o jammé (0,49 bb).

    Mains iPoker écrites sur disque comme le client PMU le ferait — jamais
    les historiques réels de la machine dans un test versionné.
    """
    (tmp_path / "a.xml").write_text(
        _main_xml("SA HA", "fold", gamecode="a"), encoding="utf-8")
    (tmp_path / "b.xml").write_text(
        _main_xml("S7 D2", "jam", gamecode="b"), encoding="utf-8")
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════
# LE MODULE PARTAGÉ D'EMPLACEMENTS (unitaire, sans serveur)
# ═══════════════════════════════════════════════════════════════════════════


def test_dossiers_connus_derivent_de_l_environnement(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    connus = emplacements.dossiers_connus()
    assert len(connus) == 2
    assert all(c.startswith(str(tmp_path)) for c in connus)
    assert all(c.endswith("data") for c in connus)


def test_la_detection_ne_rend_que_l_existant(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert emplacements.dossiers_detectes() == ()
    cible = tmp_path / "PMU PLAY 100% Poker" / "data"
    cible.mkdir(parents=True)
    assert emplacements.dossiers_detectes() == (str(cible),)


def test_repli_sans_variable_d_environnement(monkeypatch) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert emplacements.appdata_local() == Path.home() / "AppData" / "Local"


def test_recuperer_mains_importe_la_detection_partagee() -> None:
    """Le script CLI et la route utilisent la MÊME fonction — pas une copie."""
    racine = Path(__file__).resolve().parents[1]
    if str(racine) not in sys.path:
        sys.path.insert(0, str(racine))
    import recuperer_mains

    assert recuperer_mains.dossiers_connus is emplacements.dossiers_connus


# ═══════════════════════════════════════════════════════════════════════════
# LA ROUTE /api/emplacements
# ═══════════════════════════════════════════════════════════════════════════


def test_la_route_emplacements_rend_detectes_et_cherches(
        base: str, tmp_path, monkeypatch) -> None:
    """``dossiers`` = ce qui existe ; ``connus`` = où l'on a cherché.

    L'environnement est contrôlé : le test ne dépend pas de la présence d'un
    client PMU sur la machine qui l'exécute.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    cible = tmp_path / "PMU Poker" / "data"
    cible.mkdir(parents=True)
    code, r = _poster(base, "emplacements", {})
    assert code == 200, r
    assert r["dossiers"] == [str(cible)]
    assert len(r["connus"]) == 2
    assert str(cible) in r["connus"]


def test_une_detection_vide_reste_diagnosticable(base: str, tmp_path,
                                                 monkeypatch) -> None:
    """Aucun client installé ⇒ ``dossiers`` vide, mais ``connus`` dit où
    l'on a regardé — un champ muet ne se corrige pas."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    code, r = _poster(base, "emplacements", {})
    assert code == 200, r
    assert r["dossiers"] == []
    assert len(r["connus"]) == 2


# ═══════════════════════════════════════════════════════════════════════════
# LA ROUTE /api/drill/fuites : refus d'abord
# ═══════════════════════════════════════════════════════════════════════════


def test_sans_jeton_la_route_refuse(base: str, dossier_fuites: Path) -> None:
    code, rep = _poster(base, "drill/fuites",
                        {"dossier": str(dossier_fuites)}, jeton=None)
    assert code == 403
    assert "next" not in rep and "error" in rep


def test_un_dossier_manquant_est_refuse_avec_le_champ_nomme(base: str) -> None:
    code, rep = _poster(base, "drill/fuites", {})
    assert code == 400
    assert "dossier" in rep.get("error", "")


def test_un_dossier_introuvable_est_refuse(base: str) -> None:
    code, rep = _poster(base, "drill/fuites",
                        {"dossier": "C:/nexiste/vraiment/pas"})
    assert code == 400
    assert "introuvable" in rep.get("error", "")


def test_un_corpus_sans_fuite_ne_cree_pas_de_session(base: str,
                                                     tmp_path: Path) -> None:
    """Dossier vide ⇒ ``session: false`` n'est PAS une erreur : le rapport
    explique, et la route « next » dit ensuite qu'aucune session n'existe."""
    code, r = _poster(base, "drill/fuites", {"dossier": str(tmp_path)})
    assert code == 200, r
    assert r["session"] is False and r["drills"] == 0
    assert "Aucune fuite chiffrable" in r["rapport"]
    code, q = _poster(base, "drill/fuites/next", {})
    assert code == 200
    assert "aucune session" in q.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
# LE PARCOURS COMPLET : dossier → drills → réponses → score
# ═══════════════════════════════════════════════════════════════════════════


def test_le_parcours_complet_des_drills_de_fuites(base: str,
                                                  dossier_fuites: Path) -> None:
    """Disque → génération → session SM-2 → corrigés → score, tout en HTTP."""
    code, r = _poster(base, "drill/fuites", {"dossier": str(dossier_fuites)})
    assert code == 200, r
    assert r["session"] is True
    assert r["drills"] == 2 and r["n_mains"] == 2 and r["n_erreurs"] == 2
    assert r["bb_ciblees"] == pytest.approx(EV_AA_10 - EV_72O_10, abs=0.01)
    assert r["bb_non_mesurees"] == 0.0          # aucun limp dans ce corpus
    assert {f["fuite"] for f in r["fuites"]} == {"fold trop serré",
                                                 "jam trop lâche"}
    assert all(f["mesure"] for f in r["fuites"])
    assert "DRILLS CIBLÉS" in r["rapport"]

    # La fuite la plus chère d'abord : AA (3,49 bb) avant 72o (0,49 bb) —
    # et l'énoncé ne souffle JAMAIS la réponse.
    q = r["next"]
    assert q["main"] == "AA"
    assert q["options"] == ["JAM", "FOLD"]
    assert "JAM ou FOLD" in q["question"]
    assert "Bonne réponse" not in q["question"]
    assert "bonne_reponse" not in q
    assert not any("ev" == k.lower() or k.lower().startswith("ev_")
                   for k in q), f"l'énoncé expose une EV : {sorted(q)}"
    assert q["n_drills"] == 2 and q["n_reponses"] == 0

    # Réponse FAUSSE : corrigé complet, coût facturé, et le spot REVIENT
    # immédiatement (SM-2 remet l'intervalle à zéro).
    code, a = _poster(base, "drill/fuites/answer",
                      {"reponse": "FOLD", "seconds": 2.0})
    assert code == 200, a
    assert a["correcte"] is False
    assert a["bonne_reponse"] == "JAM"
    assert "Bonne réponse : JAM" in a["explication"]
    assert a["cout_bb"] == pytest.approx(EV_AA_10, abs=0.01)
    assert a["next"]["main"] == "AA", "un spot raté doit revenir tout de suite"

    # Bonne réponse (casse et espaces tolérés par le module) : on avance.
    code, a = _poster(base, "drill/fuites/answer",
                      {"reponse": "jam", "seconds": 1.0})
    assert code == 200, a
    assert a["correcte"] is True and a["cout_bb"] == 0.0
    assert a["next"]["main"] == "72o"

    # Dernière fuite : la session se boucle avec un score complet.
    code, a = _poster(base, "drill/fuites/answer",
                      {"reponse": "FOLD", "seconds": 1.0})
    assert code == 200, a
    assert a["correcte"] is True
    fin = a["next"]
    assert fin.get("fini") is True
    assert fin["score"]["n"] == 3.0
    assert fin["score"]["exactitude"] == pytest.approx(2 / 3)
    assert fin["score"]["bb_manquees"] == pytest.approx(EV_AA_10, abs=0.01)

    # Rejouer « next » ne corrompt pas l'état final.
    code, encore = _poster(base, "drill/fuites/next", {})
    assert code == 200 and encore.get("fini") is True


def test_une_reponse_hors_options_est_refusee_sans_casser_la_session(
        base: str, dossier_fuites: Path) -> None:
    code, r = _poster(base, "drill/fuites", {"dossier": str(dossier_fuites)})
    assert code == 200 and r["session"] is True
    code, a = _poster(base, "drill/fuites/answer", {"reponse": "CALL"})
    assert code == 400
    assert "JAM" in a.get("error", "") and "FOLD" in a.get("error", "")
    # La session n'est pas corrompue : la même question reste posée.
    code, q = _poster(base, "drill/fuites/next", {})
    assert code == 200 and q["main"] == "AA"


# ═══════════════════════════════════════════════════════════════════════════
# LA PAGE SERVIE : bouton, bascule, et persistance (structurel)
# ═══════════════════════════════════════════════════════════════════════════


def test_la_page_porte_le_bouton_et_la_session_de_fuites(base: str) -> None:
    """Le chaînon revue → entraînement existe dans les octets servis."""
    code, html = _obtenir(base, "/")
    assert code == 200
    assert "M'entraîner sur mes fuites" in html
    assert "drillFuitesStart" in html
    assert 'api("drill/fuites"' in html
    assert 'id="fuites-q"' in html
    assert 'goTab("s-train")' in html


def test_la_persistance_ne_couvre_que_le_chemin_et_la_fenetre(base: str) -> None:
    """Structurel : localStorage n'est touché que par les deux helpers, et
    chaque écriture porte une clé autorisée. Les montants du spot (pot, mise,
    tapis, blinde) ne sont JAMAIS persistés — des chiffres d'une vieille
    session rejoués en silence seraient un mensonge."""
    _, html = _obtenir(base, "/")
    # Tous les accès localStorage vivent dans lsGet/lsSet.
    assert html.count("localStorage.") == 2, (
        "un accès localStorage vit hors des helpers lsGet/lsSet")
    # Chaque APPEL de lsSet (hors définition) porte une clé autorisée.
    appels = [m.group(1).strip()
              for m in re.finditer(r"(?<!function )lsSet\(([^,]+),", html)]
    assert appels, "aucune écriture persistée : la persistance a disparu"
    assert set(appels) <= {"LS.rv", "LS.cal"}, appels
    # Les deux clés sont bien celles du chemin et de la fenêtre.
    assert '"pfs.rv-path"' in html and '"pfs.cal-fen"' in html
    # Et aucun identifiant de montant n'apparaît près d'une écriture.
    for m in re.finditer(r"(?<!function )lsSet\(", html):
        contexte = html[max(0, m.start() - 120):m.start() + 120]
        for champ in ("sp-pot", "sp-bet", "sp-stack", "sp-bb",
                      "sp-stacks", "sp-payouts"):
            assert champ not in contexte, (
                f"« {champ} » apparaît près d'une écriture persistée")


def test_le_prerempli_vient_de_la_detection_sans_la_memoriser(base: str) -> None:
    """Le préremplissage interroge /api/emplacements mais n'écrit PAS la
    détection dans localStorage : une détection n'est pas un choix — elle
    n'est mémorisée qu'à la première utilisation (frappe ou analyse)."""
    _, html = _obtenir(base, "/")
    assert 'api("emplacements"' in html
    debut = html.index("function initPersistance"
                       ) if "function initPersistance" in html else \
        html.index("initPersistance")
    fin = html.index("buildGrid();", debut)
    bloc = html[debut:fin]
    apres = bloc[bloc.index('api("emplacements"'):]
    assert "lsSet" not in apres, (
        "la détection est écrite dans localStorage sans geste de l'utilisateur")


def test_deux_chargements_successifs_rendent_la_meme_page(base: str) -> None:
    """Premier chargement et rechargement : même page, même détection —
    aucun état serveur ne fuit dans les octets servis."""
    code1, html1 = _obtenir(base, "/")
    code2, html2 = _obtenir(base, "/")
    assert code1 == code2 == 200
    assert html1 == html2
    code, r1 = _poster(base, "emplacements", {})
    _, r2 = _poster(base, "emplacements", {})
    assert code == 200 and r1 == r2
