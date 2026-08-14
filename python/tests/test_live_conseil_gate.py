"""La nouvelle porte du mode Live : le conseil n'existe que prouvé.

Miroir INVERSÉ de ``test_live_sans_conseil.py`` : là-bas on prouve que les
routes de calibration ne conseillent jamais ; ici on prouve que la route
``live/table`` ne conseille QUE lorsque le gate a dit LIVE — armement manuel
confirmé + badge « PMU PLAY » vu (ou son souvenir de moins de 60 s sur la
même fenêtre au même titre) — et jamais autrement : non armé, badge absent,
gate en panne, désarmé, titre changé, lecture insuffisante.

Le chemin est exécuté POUR DE VRAI : seule la capture d'écran est remplacée
par une table synthétique (détection, badge, gate, lecture des montants et
conseil s'exécutent réellement), comme dans le verrou existant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import pfs.app.mode_live as mode_live
import pfs.app.server as server
from pfs.app.server import API
from pfs.vision import live as vision_live
from pfs.vision.digit_ocr import _chemin_police
from pfs.vision.synth_table import FELTS, TableSpec, render_table

HWND = 777_001
TITRE = "100 - 6-max Turbo Re-buy 424242"

_POLICE = _chemin_police("segoeui.ttf") or _chemin_police("arial.ttf")
pytestmark = pytest.mark.skipif(
    _POLICE is None, reason="aucune police d'interface — le rendu des "
    "montants de test serait illisible par construction")

_GABARITS = Path(__file__).resolve().parents[1] / "pfs" / "vision" / \
    "templates" / "pmu_play"


def _frame(avec_badge: bool) -> bytes:
    """Une table 1920×1361 lisible de bout en bout, badge en option.

    Cartes aux dimensions des captures réelles (138×169) pour que les cadres
    de montants — relatifs à la hauteur de carte — englobent les textes du
    générateur. Le libellé « CALL … » du générateur est recouvert de feutre
    puis remplacé par un bouton « PAYER … » HORS du cadre du tapis : sur le
    client réel les deux rangées sont disjointes, le générateur les
    compresse (même raison que dans ``test_zones_montants``). La preuve
    play-money est le DOS de carte orange, collé dans une zone neutre — le
    filigrane central est sous le board, comme sur les vraies captures.
    """
    spec = TableSpec(hero=("Ah", "Kd"), board=("2c", "7d", "Jh"),
                     pot=6.75, hero_stack=42.5, to_call=2.25,
                     card_w=138, card_h=169, size=(1920, 1361),
                     taille_texte=22, decor=False)
    synth = render_table(spec)
    img = synth.image
    d = ImageDraw.Draw(img)
    feutre = FELTS["feutre vert"]
    # Recouvre le « CALL … » du générateur (rangée partagée avec le tapis).
    d.rectangle([1150, int(1361 * 0.86), 1920, int(1361 * 0.94)], fill=feutre)
    police = ImageFont.truetype(str(_POLICE), 22)
    d.text((1400, 1250), "PAYER 2,25 BB", fill=(240, 240, 228), font=police)
    if avec_badge:
        dos = Image.open(_GABARITS / "sprite_dos_pmu_play.png").convert("RGB")
        img.paste(dos, (940, 110))
    import io
    tampon = io.BytesIO()
    img.save(tampon, format="PNG")
    return tampon.getvalue()


@pytest.fixture()
def porte(monkeypatch, tmp_path):
    """Un mode Live neuf, capture injectée, archive détournée."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(mode_live, "_hwnd_par_titre", lambda titre: HWND)
    server._MODE_LIVE_INSTANCE = None
    frames = {"png": _frame(avec_badge=True)}
    monkeypatch.setattr(
        vision_live, "capturer_fenetre",
        lambda titre=None, timeout_s=vision_live._TIMEOUT_S: frames["png"])
    yield frames
    server._MODE_LIVE_INSTANCE = None


def _armer() -> dict:
    return API.live_armer({"fenetre": TITRE, "confirme": True})


# ═══════════════════════════════════════════════════════════════════════════
# 1. FAIL-CLOSED : tous les chemins qui doivent NE PAS conseiller
# ═══════════════════════════════════════════════════════════════════════════


def test_non_arme_aucun_conseil_meme_avec_badge(porte) -> None:
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "review"
    assert r["conseil"] is None
    assert "non prouvée" in r["conseil_refus"]
    assert "non armée" in r["gate"]["raison"]


def test_l_armement_exige_la_confirmation_explicite(porte) -> None:
    with pytest.raises(ValueError, match="confirme"):
        API.live_armer({"fenetre": TITRE})
    with pytest.raises(ValueError, match="confirme"):
        API.live_armer({"fenetre": TITRE, "confirme": "oui"})


def test_arme_sans_badge_reste_en_review(porte) -> None:
    porte["png"] = _frame(avec_badge=False)
    _armer()
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "review"
    assert r["gate"]["badge_vu"] is False
    assert r["conseil"] is None


def test_le_titre_play_money_ne_corrobore_pas_le_badge(porte) -> None:
    # Le profil PMU PLAY IGNORE le titre par construction (voir
    # ``ComplianceGate.profil_pmu_play``) : sur ce client les tables affichent
    # des euros fictifs et des marqueurs de tournoi, le titre ne discrimine
    # RIEN. Un titre « argent fictif » ne doit donc jamais tenir lieu de
    # corroboration à la place du badge — sinon il suffirait de renommer sa
    # fenêtre pour déverrouiller le conseil sans preuve pixel. Armé sous un tel
    # titre mais SANS badge à l'écran : le gate reste en review, aucun conseil.
    porte["png"] = _frame(avec_badge=False)
    titre_menteur = "PMU PLAY - argent fictif - table 7"
    API.live_armer({"fenetre": titre_menteur, "confirme": True})
    r = API.live_table({"fenetre": titre_menteur})
    assert r["gate"]["mode"] == "review"
    assert r["gate"]["badge_vu"] is False
    assert r["conseil"] is None


def test_une_panne_du_gate_ne_conseille_pas(porte, monkeypatch) -> None:
    _armer()
    appels: list[str] = []
    monkeypatch.setattr(API, "advise",
                        staticmethod(lambda p: appels.append("advise")))
    from pfs.compliance.gate import ComplianceGate
    monkeypatch.setattr(     # la classe : ComplianceGate a des __slots__
        ComplianceGate, "evaluate",
        lambda self, obs: (_ for _ in ()).throw(RuntimeError("panne")))
    with pytest.raises(RuntimeError):
        API.live_table({"fenetre": TITRE})
    assert appels == []


def test_le_desarmement_revoque_immediatement(porte) -> None:
    _armer()
    assert API.live_table({"fenetre": TITRE})["gate"]["mode"] == "live"
    API.live_desarmer({"fenetre": TITRE})
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "review"
    assert r["conseil"] is None


def test_le_verrouillage_par_audit_bloque_l_armement(porte) -> None:
    server._mode_live().gate.lock("main assistée jugée réelle par l'audit")
    from pfs.compliance.gate import GateLockedError
    with pytest.raises(GateLockedError):
        _armer()


def test_les_calculateurs_dorment_quand_le_gate_refuse(porte, monkeypatch) -> None:
    # Le tripwire est posé sur le conseiller AVANT l'appel : si la route le
    # convoque malgré un gate en REVIEW, le test explose ici.
    def piege(p):
        raise AssertionError("advise convoqué alors que le gate a dit REVIEW")
    monkeypatch.setattr(API, "advise", staticmethod(piege))
    r = API.live_table({"fenetre": TITRE})       # non armé
    assert r["gate"]["mode"] == "review" and r["conseil"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. LE CHEMIN AUTORISÉ — et sa mémoire fail-closed
# ═══════════════════════════════════════════════════════════════════════════


def test_arme_avec_badge_le_conseil_existe_et_est_complet(porte) -> None:
    _armer()
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "live"
    assert r["gate"]["badge_vu"] is True
    lecture = r["lecture"]
    assert lecture["main"] == ["Ah", "Kd"]
    assert lecture["montants"]["pot"]["valeur"] == 6.75
    assert lecture["montants"]["mise"]["valeur"] == 2.25
    assert lecture["montants"]["tapis"]["valeur"] == 42.5
    assert lecture["montants"]["blinde"]["valeur"] == 1.0
    assert r["conseil"] is not None and r["conseil"]["action"]


def test_le_conseil_est_celui_du_conseiller_rien_d_autre(porte) -> None:
    # Anti-fabrication (leçon du contournement G) : la route doit rendre
    # EXACTEMENT ce que le conseiller recalcule sur la même lecture.
    _armer()
    r = API.live_table({"fenetre": TITRE})
    m = r["lecture"]["montants"]
    attendu = API.advise({
        "hero": " ".join(r["lecture"]["main"]),
        "board": " ".join(r["lecture"]["tableau"]),
        "pot": m["pot"]["valeur"], "bet": m["mise"]["valeur"],
        "stack": m["tapis"]["valeur"], "big_blind": m["blinde"]["valeur"]})
    assert r["conseil"]["action"] == attendu["action"]
    assert r["conseil"]["equity"] == attendu["equity"]


def test_le_souvenir_du_badge_couvre_l_occultation(porte) -> None:
    _armer()
    assert API.live_table({"fenetre": TITRE})["gate"]["mode"] == "live"
    porte["png"] = _frame(avec_badge=False)      # board masque le badge
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "live"
    assert r["gate"]["badge_age_s"] is not None and r["gate"]["badge_age_s"] >= 0


def test_le_souvenir_expire_apres_60_secondes(porte, monkeypatch) -> None:
    _armer()
    API.live_table({"fenetre": TITRE})
    porte["png"] = _frame(avec_badge=False)
    horloge = {"t": 0.0}
    vrai = mode_live.time.monotonic()
    monkeypatch.setattr(mode_live.time, "monotonic",
                        lambda: vrai + horloge["t"])
    horloge["t"] = mode_live.BADGE_MEMOIRE_S + 5.0
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "review"
    assert r["conseil"] is None


def test_un_changement_de_titre_purge_le_souvenir(porte) -> None:
    _armer()
    API.live_table({"fenetre": TITRE})           # badge vu et mémorisé
    autre = "NL50 cash réelle maquillée"
    API.live_armer({"fenetre": autre, "confirme": True})   # même hwnd simulé
    porte["png"] = _frame(avec_badge=False)
    r = API.live_table({"fenetre": autre})
    assert r["gate"]["mode"] == "review"
    assert r["conseil"] is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONTRAT CLOS
# ═══════════════════════════════════════════════════════════════════════════


def test_le_contrat_de_la_route_est_clos(porte) -> None:
    _armer()
    r = API.live_table({"fenetre": TITRE})
    assert set(r) == {"gate", "lecture", "conseil"}
    assert set(r["gate"]) == {"mode", "raison", "verrouille", "badge_vu",
                              "badge_filigrane", "badge_dos", "badge_age_s",
                              "badge_refus", "signaux"}
    API.live_desarmer({})
    r2 = API.live_table({"fenetre": TITRE})
    assert set(r2) == {"gate", "lecture", "conseil", "conseil_refus"}
    assert r2["conseil"] is None


def test_la_lecture_insuffisante_refuse_le_conseil_en_le_disant(porte) -> None:
    # Un board seul, sans main du héros : gate LIVE mais rien à juger.
    spec = TableSpec(hero=(), board=("2c", "7d", "Jh"), pot=6.75,
                     card_w=138, card_h=169, size=(1920, 1361),
                     taille_texte=22, decor=False)
    synth = render_table(spec)
    dos = Image.open(_GABARITS / "sprite_dos_pmu_play.png").convert("RGB")
    synth.image.paste(dos, (940, 110))
    import io
    tampon = io.BytesIO()
    synth.image.save(tampon, format="PNG")
    porte["png"] = tampon.getvalue()
    _armer()
    r = API.live_table({"fenetre": TITRE})
    assert r["gate"]["mode"] == "live"
    assert r["conseil"] is None
    # Sans cartes du héros en bas, la détection promeut ce qu'elle peut :
    # peu importe le compte exact, il n'est pas 2 et le refus le DIT.
    assert "carte(s) sûre(s)" in r["conseil_refus"]
