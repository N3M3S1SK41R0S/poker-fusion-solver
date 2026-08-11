"""Le parcours utilisateur complet, de bout en bout, par les ROUTES RÉELLES.

Pourquoi ce fichier existe
--------------------------
Les cinq défauts graves trouvés récemment étaient tous **aux jointures** :
chaque composant était juste isolément, et c'est l'assemblage qui mentait.

  1. ``/api/advise`` construisait le ``Spot`` sans ``stacks`` ni ``payouts`` :
     le régime ICM était inatteignable, et le verdict INVERSE sur une bulle.
     Non détecté parce que tous les tests du régime ICM appelaient ``advise()``
     en direct, jamais la route HTTP.
  2. Le verdict se déclenchait sur des montants restés par défaut dans le
     formulaire après un collage.
  3. Deux gabarits de cartes intervertis : lecture FAUSSE et affirmée,
     survivant à des tests tautologiques (chaque gabarit comparé à lui-même).

Un test qui appelle ``API.advise(...)`` en Python ne prouve rien sur le
routage, ni sur le format d'échange, ni sur ce que l'utilisateur obtient.
Celui-ci part donc d'un **fichier sur disque**, monte le **vrai serveur** sur
un port libre et lui parle en **HTTP avec urllib**. Aucune fonction interne
n'est appelée en direct pour produire un résultat testé.

Ce que le parcours traverse
---------------------------
``GET /`` (la page et son jeton) → ``POST /api/recognize`` sur les découpes
PMU versionnées → ``POST /api/lire_capture`` sur une capture de table entière
→ ``POST /api/advise`` sur quatorze spots écrits à la main, couvrant les
quatre régimes : préflop chipEV (Nash push/fold), préflop **ICM** (avec tapis
ET gains — c'est le défaut n°1), postflop équité contre cotes du pot, et
charts d'ouverture en tapis profond. Plus les refus : une découpe illisible,
une capture sans carte lisible, une main absente.

Couverture réelle mesurée par ce seul parcours
----------------------------------------------
Banc rejouable : ``python banc_couverture_parcours.py [--modules]``
(coverage.py s'il est importable, sinon un traceur ``sys.settrace`` +
``threading.settrace`` dont le dénominateur vient des ``co_lines()`` du code
compilé — le serveur répond dans un fil par requête, sans
``threading.settrace`` tout le code des routes échapperait au compteur).

Mesure du 11 août 2026, Windows 11, Python 3.13.14, sur les 61 fichiers de
``pfs/`` :

  * coverage.py 7.15.4 : **2 897 lignes exécutables traversées sur 9 146,
    soit 31,7 %** ; **20 modules jamais atteints** ;
  * traceur ``sys.settrace`` : 3 592 sur 11 920, soit **30,1 %** (même
    numérateur logique, dénominateur plus large : coverage.py exclut ce que
    ses règles écartent, le traceur compte toute ligne atteignable).

Les nombres absolus bougent au fil du code — mesurés à deux heures d'écart le
même jour, ils avaient déjà glissé de deux lignes ; c'est le **rapport** qui
se relit, et il faut rejouer le banc plutôt que citer ce paragraphe.

Les modules les mieux traversés sont ceux de la chaîne :
``vision/synth_table`` 95,0 %, ``vision/table_detector`` 91,7 %,
``vision/lecteur_fond_plein`` 90,8 %, ``vision/phash`` 81,0 %,
``vision/card_recognizer`` 74,6 %, ``analysis/spot_advisor`` 68,7 %,
``solver/pushfold`` 67,6 %. Le serveur lui-même n'est traversé qu'à 42,0 %
(224 lignes sur 533) : ce parcours n'emprunte que 3 de ses 31 routes POST
(``recognize``, ``lire_capture``, ``advise``) plus la page et ses erreurs.

Le chiffre est bas, et c'est l'information la plus utile de ce fichier : les
1 095 tests qui existaient avant lui couvrent large en appelant les
composants un par un, mais
**moins d'un tiers du code est atteint par le chemin que l'utilisateur emprunte
réellement**. Les 68 % restants ne sont pas morts — ce sont les routes que ce
parcours ne visite pas (``solve``, ``postflop``, ``resolve``, ``review``,
``drill``, ``fusion``, ``bankroll``, ``hmm``, ``simuler``, ``lexique``…) et les
branches d'erreur. Mais tant qu'un parcours de bout en bout ne les traverse
pas, aucun test ne dit ce que l'utilisateur en obtiendrait par l'interface :
c'est exactement la zone où vivaient les cinq défauts. Le détail par module
est imprimé par ``--modules``.

Preuve que ce test échoue quand on réintroduit un défaut
--------------------------------------------------------
Vérifié le 11 août 2026 en cassant, en mesurant, puis en rétablissant le code
(retour au vert confirmé après chaque rétablissement) :

  * **Défaut n°1** — en retirant ``stacks=`` et ``payouts=`` de la
    construction du ``Spot`` dans ``pfs/app/server.py`` (``API.advise``) :
    **9 échecs**. Les trois verdicts ICM de la table (``ICM K9o``, ``ICM AA``,
    ``ICM bulle A7o``) reçoivent « JAM (tapis) » là où l'ICM rend « FOLD » ;
    ``postflop AJ tournoi`` reçoit « CALL (juste) » au lieu de « FOLD » ;
    ``test_le_regime_icm_est_atteignable_par_la_route``,
    ``test_l_icm_inverse_le_verdict_sur_une_bulle``,
    ``test_le_bubble_factor_releve_l_equite_requise_postflop`` (bubble factor
    ``None``), ``test_la_capture_entiere_alimente_le_verdict`` et
    ``test_un_tapis_nul_en_icm_est_refuse_avec_une_consigne`` (la route
    répond 200 avec un verdict de chart d'ouverture au lieu de refuser).
  * **Défaut n°2** — en remplaçant, dans ``pfs/app/ui.html``, le garde-fou
    ``if(manquants.length){…}else{adviseRun();}`` par un ``adviseRun();``
    inconditionnel : **1 échec**,
    ``test_apres_un_collage_les_montants_non_saisis_ne_declenchent_rien``
    (« le montant "sp-pot" n'est pas gardé »).
  * **Défaut n°3** — en intervertissant les fichiers ``5c.png`` et ``3h.png``
    du thème ``pmu_solid`` (les gabarits) : **2 échecs**,
    ``test_les_decoupes_reelles_se_lisent_par_la_route[pmu_play_hero_5c.png-5c]``
    (« lu ['3h'] au lieu de ['5c'] ») et son jumeau. Le défaut d'origine était
    sur ``pmu_deck``, mais le mécanisme et le test sont les mêmes : une
    vérité-terrain **externe** au gabarit.

Limite assumée (NEMESIS)
------------------------
Le garde-fou du défaut n°2 vit dans le JavaScript de la page. Aucun moteur JS
n'est disponible ici : le test vérifie donc, **sur les octets réellement
servis par ``GET /``**, que le garde existe, qu'il porte sur les quatre
montants, et que ``adviseRun()`` n'est atteignable que par sa branche
``else`` — pas que le navigateur l'exécute. C'est une vérification
structurelle, pas comportementale, et elle est nommée comme telle.
"""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from PIL import Image

from pfs.app.server import create_server
from pfs.vision.synth_table import TableSpec, render_table

DONNEES = Path(__file__).parent / "donnees"

#: Jeton explicite : le serveur de test ne doit jamais toucher au jeton
#: persistant de l'utilisateur (``%LOCALAPPDATA%/PokerFusionSolver/jeton``).
JETON = "jeton-du-parcours-complet"

#: Découpes réelles versionnées et leur vérité-terrain, lue à l'œil sur la
#: capture PMU du 10 août 2026. La vérité est EXTERNE au gabarit : c'est ce
#: qui rend le test non tautologique (défaut n°3).
DECOUPES_REELLES = {
    "pmu_play_hero_5c.png": "5c",
    "pmu_play_hero_3h.png": "3h",
}


# ═══════════════════════════════════════════════════════════════════════════
# LE SERVEUR RÉEL, ET COMMENT ON LUI PARLE
# ═══════════════════════════════════════════════════════════════════════════


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def _archive_jetable(tmp_path_factory) -> str:
    """Détourne l'archive des échecs vers un dossier temporaire.

    ``lire_capture`` et ``recognize`` archivent tout ce qu'ils ne lisent pas
    avec certitude, sous ``%LOCALAPPDATA%``. Un test qui écrit dans l'archive
    réelle de l'utilisateur y noierait ses vrais échecs — c'est-à-dire la
    matière même du banc d'essai.
    """
    ancien = os.environ.get("LOCALAPPDATA")
    racine = tmp_path_factory.mktemp("archive-parcours")
    os.environ["LOCALAPPDATA"] = str(racine)
    try:
        yield str(racine)
    finally:
        if ancien is None:
            os.environ.pop("LOCALAPPDATA", None)
        else:
            os.environ["LOCALAPPDATA"] = ancien


@pytest.fixture(scope="module")
def base(_archive_jetable) -> str:
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
    """POST HTTP réel sur ``/api/<route>``. Renvoie (code, corps décodé).

    Les erreurs HTTP ne lèvent pas : leur corps JSON est la réponse que
    l'interface reçoit, et c'est lui qu'on veut examiner.
    """
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
    """GET HTTP réel. Renvoie (code, corps texte)."""
    try:
        with urllib.request.urlopen(f"{base}{chemin}", timeout=30) as rep:
            return rep.status, rep.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def _b64(source: Path | Image.Image) -> str:
    """Encode une image comme le navigateur l'envoie (préfixe ``data:``)."""
    if isinstance(source, Path):
        octets = source.read_bytes()
    else:
        tampon = io.BytesIO()
        source.save(tampon, format="PNG")
        octets = tampon.getvalue()
    return "data:image/png;base64," + base64.b64encode(octets).decode("ascii")


def _table(hero: tuple[str, ...], board: tuple[str, ...], seed: int) -> str:
    """Capture de table entière, encodée : ce qu'on colle dans la page."""
    return _b64(render_table(TableSpec(
        hero=hero, board=board, size=(1200, 760), theme="pmu_solid",
        decor=True, villains=2, seed=seed)).image)


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 0 — LE TRANSPORT LUI-MÊME
# ═══════════════════════════════════════════════════════════════════════════


def test_la_page_est_servie_avec_son_jeton(base: str) -> None:
    """``GET /`` rend l'interface, jeton substitué.

    Sans cette substitution la page se charge et **tous** ses appels
    reviennent en 403 : le logiciel paraît « ne plus rien reconnaître » alors
    qu'aucun calcul n'a été demandé.
    """
    code, html = _obtenir(base, "/")
    assert code == 200
    assert JETON in html, "le jeton n'a pas été injecté dans la page"
    assert "__TOKEN__" not in html, "le gabarit de jeton est resté tel quel"


def test_sans_jeton_l_api_refuse_et_ne_conseille_rien(base: str) -> None:
    """Un 403 doit être un refus, pas un verdict dégradé."""
    code, rep = _poster(base, "advise", {"hero": "Ah Ad", "stack": 10},
                        jeton=None)
    assert code == 403
    assert "action" not in rep and "error" in rep


def test_une_route_inconnue_est_nommee(base: str) -> None:
    """Le 404 doit dire QUELLE route manque — c'est ce qui a fait perdre des
    heures quand un vieux serveur répondait à la place du neuf."""
    code, rep = _poster(base, "route/qui/nexiste/pas", {})
    assert code == 404
    assert "route/qui/nexiste/pas" in rep.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — DU DISQUE À LA CARTE : les découpes réelles versionnées
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("fichier,verite", sorted(DECOUPES_REELLES.items()))
def test_les_decoupes_reelles_se_lisent_par_la_route(base: str, fichier: str,
                                                     verite: str) -> None:
    """Une vraie découpe PMU, lue par ``POST /api/recognize``.

    La vérité-terrain est écrite ici, dans le test, et non dérivée du gabarit :
    intervertir deux gabarits fait échouer ce test, alors qu'un test qui
    compare chaque gabarit à lui-même resterait vert (défaut n°3).
    """
    code, rep = _poster(base, "recognize", {"image_b64": _b64(DONNEES / fichier)})
    assert code == 200, rep
    assert rep["cards"] == [verite], (
        f"lu {rep['cards']} au lieu de ['{verite}'] sur {fichier}")
    assert rep["detail"][0]["statut"] == "sure"


def test_une_decoupe_illisible_est_refusee_pas_devinee(base: str) -> None:
    """Le refus explicite (d) : « je ne sais pas » plutôt qu'une carte inventée.

    ``pmu_hero_tronque.png`` est une découpe réelle où le HUD mange le bas des
    cartes. Un logiciel qui devine ici est plus dangereux qu'un logiciel
    absent : le joueur croirait tenir une lecture.
    """
    code, rep = _poster(base, "recognize",
                        {"image_b64": _b64(DONNEES / "pmu_hero_tronque.png")})
    assert code == 200, rep
    assert rep["cards"] == [None], f"une carte a été affirmée : {rep['cards']}"
    detail = rep["detail"][0]
    assert detail["statut"] != "sure"
    # le diagnostic reste exposé — un refus muet ne se corrige pas
    assert detail["distance"] is not None and "margin" in detail
    # et surtout : aucun verdict ne s'est glissé dans la réponse
    assert not {"action", "regime", "confidence"} & set(rep)


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — DE LA CAPTURE ENTIÈRE AUX CARTES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def capture_preflop(base: str) -> dict:
    """Une table préflop entière, collée puis lue par la route."""
    code, rep = _poster(base, "lire_capture",
                        {"image_b64": _table(("Ah", "Ad"), (), seed=11)})
    assert code == 200, rep
    return rep


@pytest.fixture(scope="module")
def capture_postflop(base: str) -> dict:
    """Une table au turn, collée puis lue par la route."""
    code, rep = _poster(base, "lire_capture", {
        "image_b64": _table(("Ad", "Jh"), ("Qs", "9d", "4c", "2h"), seed=13)})
    assert code == 200, rep
    return rep


def test_la_capture_preflop_rend_la_main(capture_preflop: dict) -> None:
    assert capture_preflop["main"] == ["Ah", "Ad"]
    assert capture_preflop["tableau"] == []
    assert capture_preflop["sures"] == capture_preflop["total"] == 2


def test_la_capture_postflop_rend_la_main_et_le_board(
        capture_postflop: dict) -> None:
    assert capture_postflop["main"] == ["Ad", "Jh"]
    assert capture_postflop["tableau"] == ["Qs", "9d", "4c", "2h"]
    assert capture_postflop["sures"] == capture_postflop["total"] == 6


def test_une_capture_sans_carte_lisible_ne_rend_aucune_carte(base: str) -> None:
    """Refus (d), version capture entière : rien de lisible → rien d'affirmé."""
    vide = _b64(render_table(TableSpec(
        hero=(), board=(), size=(1200, 760), theme="pmu_solid",
        decor=True, villains=0, seed=17)).image)
    code, rep = _poster(base, "lire_capture", {"image_b64": vide})
    assert code == 200, rep
    assert rep["main"] == [] and rep["tableau"] == []
    assert rep["sures"] == 0
    # Le détecteur a cadré des boîtes (jetons, avatars) sans rien y lire :
    # c'est le comportement voulu — il montre ce qu'il a vu, il n'invente pas.
    for role in ("hero", "board"):
        for carte in rep[role]:
            assert carte["carte"] is None and carte["statut"] != "sure"


def test_une_capture_ne_fournit_aucun_montant(capture_postflop: dict) -> None:
    """Défaut n°2, moitié serveur : la capture ne donne QUE des cartes.

    Si la réponse contenait un pot ou un tapis, l'interface pourrait croire
    les tenir de l'image. Ils n'y sont pas : ils viendront du formulaire, donc
    de l'utilisateur — et c'est ce qui rend le garde-fou côté page nécessaire.
    """
    interdits = {"pot", "bet", "mise", "stack", "tapis", "big_blind", "bb",
                 "payouts", "stacks", "ante"}
    assert not interdits & set(capture_postflop), (
        f"la lecture prétend fournir des montants : "
        f"{sorted(interdits & set(capture_postflop))}")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — DES CARTES AU VERDICT : quatorze spots écrits à la main
# ═══════════════════════════════════════════════════════════════════════════
#
# Chaque ligne : (nom, charge utile, action attendue, fragment de régime,
# certitude attendue). L'action attendue est écrite à la main d'après le
# raisonnement noté en commentaire, PAS recopiée d'une sortie du moteur.

VERDICTS = [
    # ── préflop chipEV : équilibre de Nash jam/fold, heads-up ───────────────
    # AA à 10 bb : la meilleure main du jeu, tapis court, personne devant.
    ("chipEV AA 10bb",
     {"hero": "Ah Ad", "stack": 10, "big_blind": 1, "players": 2,
      "position": "SB"},
     "JAM", "Nash push/fold", "certain"),
    # A5s à 12 bb : as bloquant + tirage couleur, dans toute range de jam.
    ("chipEV A5s 12bb",
     {"hero": "As 5s", "stack": 12, "big_blind": 1, "players": 2,
      "position": "SB"},
     "JAM", "Nash push/fold", "certain"),
    # 95o à 20 bb : trop haut pour jammer une main sans équité de repli.
    ("chipEV 95o 20bb",
     {"hero": "9h 5d", "stack": 20, "big_blind": 1, "players": 2,
      "position": "SB"},
     "FOLD", "Nash push/fold", "certain"),
    # 5c 3h à 8 bb : la main réellement lue sur la capture PMU du 10 août.
    # Deux petites cartes dépareillées, aucune équité contre une range de call.
    ("chipEV 5c3h 8bb",
     {"hero": "5c 3h", "stack": 8, "big_blind": 1, "players": 2,
      "position": "SB"},
     "FOLD", "Nash push/fold", "certain"),

    # ── préflop ICM : le régime que la route jetait (défaut n°1) ────────────
    # K9o, 10 bb, un joueur à 1 bb : en jetons c'est un jam net ; en dollars,
    # sauter avant lui coûte la place qu'il est en train de perdre.
    ("ICM K9o",
     {"hero": "Kh 9d", "big_blind": 1, "stacks": "10, 40, 1",
      "payouts": "65, 35", "players": 3},
     "FOLD", "ICM", "indicatif"),
    # Le même spot SANS tapis ni gains : le chipEV veut pousser. C'est le
    # couple qui prouve que la bascule opère.
    ("chipEV K9o 10bb",
     {"hero": "Kh 9d", "stack": 10, "big_blind": 1, "players": 2,
      "position": "SB"},
     "JAM", "Nash push/fold", "certain"),
    # AA sous la même pression de bulle : aucune structure de gains ne rend
    # les as trop chers à jouer.
    ("ICM AA",
     {"hero": "Ah Ad", "big_blind": 1, "stacks": "10, 40, 1",
      "payouts": "65, 35", "players": 3},
     "JAM", "ICM", "indicatif"),
    # A7o sur une bulle 50/30/20 : le spot de référence du défaut n°1 — la
    # route rendait « OUVRIR (relance) », l'inverse du bon verdict.
    ("ICM bulle A7o",
     {"hero": "Ah 7d", "big_blind": 1, "stack": 10, "players": 3,
      "position": "SB", "pot": 1.5, "bet": 0,
      "stacks": "10, 40, 1", "payouts": "50, 30, 20"},
     "FOLD", "ICM", "indicatif"),

    # ── postflop : équité exacte contre les cotes du pot ────────────────────
    # Brelan de 7 au flop face à 75 dans 100 : payer demande 30 % d'équité,
    # la main en réalise l'écrasante majorité.
    ("postflop brelan",
     {"hero": "7s 7h", "board": "Qs 7d 2c", "pot": 100, "bet": 75,
      "stack": 300, "big_blind": 10, "position": "BTN"},
     "CALL", "vs cotes du pot", "indicatif"),
    # 32o sur Qs 9d 8c Kh face à 90 dans 100 : rien, pas même un tirage.
    ("postflop air",
     {"hero": "3s 2h", "board": "Qs 9d 8c Kh", "pot": 100, "bet": 90,
      "stack": 300, "big_blind": 10, "position": "BTN"},
     "FOLD", "vs cotes du pot", "indicatif"),
    # AJ au turn face à 50 dans 100 : 25 % requis en cash game, la main les a.
    ("postflop AJ cash",
     {"hero": "Ad Jh", "board": "Qs 9d 4c 2h", "pot": 100, "bet": 50,
      "stack": 300, "big_blind": 10, "position": "BTN", "players": 3},
     "CALL", "vs cotes du pot", "indicatif"),
    # LE MÊME spot en tournoi : le bubble factor fait passer l'équité requise
    # au-dessus de celle de la main. C'est le fold que les joueurs de cash
    # ratent — et il n'existe que si tapis et gains traversent la route.
    ("postflop AJ tournoi",
     {"hero": "Ad Jh", "board": "Qs 9d 4c 2h", "pot": 100, "bet": 50,
      "stack": 300, "big_blind": 10, "position": "BTN", "players": 3,
      "stacks": "300, 900, 60", "payouts": "50, 30, 20"},
     "FOLD", "(ICM)", "indicatif"),

    # ── préflop tapis profond : charts d'ouverture ──────────────────────────
    ("profond AKo BTN",
     {"hero": "Ah Kd", "stack": 100, "big_blind": 1, "players": 6,
      "position": "BTN"},
     "OUVRIR", "chart d'ouverture", "indicatif"),
    ("profond 72o UTG",
     {"hero": "7h 2d", "stack": 100, "big_blind": 1, "players": 6,
      "position": "UTG"},
     "FOLD", "chart d'ouverture", "indicatif"),
]


@pytest.mark.parametrize(
    "nom,charge,action,regime,certitude",
    VERDICTS, ids=[v[0] for v in VERDICTS])
def test_le_verdict_attendu_sort_de_la_route(base: str, nom: str, charge: dict,
                                             action: str, regime: str,
                                             certitude: str) -> None:
    """Quatorze verdicts écrits à la main, rendus par ``POST /api/advise``."""
    code, rep = _poster(base, "advise", charge)
    assert code == 200, rep
    assert rep["action"].upper().startswith(action.upper()), (
        f"{nom} : « {rep['action']} » au lieu de « {action}… » "
        f"(régime « {rep['regime']} »)")
    assert regime in rep["regime"], (
        f"{nom} : régime « {rep['regime']} » ne contient pas « {regime} »")
    assert rep["confidence"] == certitude, (
        f"{nom} : certitude « {rep['confidence']} » au lieu de « {certitude} »")


def test_le_regime_icm_est_atteignable_par_la_route(base: str) -> None:
    """Défaut n°1, nommé : tapis et gains doivent TRAVERSER la route.

    ``advise()`` bascule en ICM sur la seule présence de ``stacks`` et
    ``payouts``. La route les construisait puis les jetait ; le régime, pourtant
    implémenté, documenté et testé en direct, était donc inatteignable.
    """
    charge = {"hero": "Kh 9d", "big_blind": 1, "stacks": "10, 40, 1",
              "payouts": "65, 35", "players": 3}
    code, rep = _poster(base, "advise", charge)
    assert code == 200, rep
    assert "ICM" in rep["regime"], (
        f"régime « {rep['regime']} » : les tapis n'ont pas traversé la route")
    assert rep["bubble"] is not None and rep["bubble"] > 1.0
    assert rep["ev_icm"] is not None


def test_l_icm_inverse_le_verdict_sur_une_bulle(base: str) -> None:
    """Le cœur du défaut n°1 : ce n'était pas une option manquante, c'était un
    verdict FAUX. Si ces deux verdicts redevenaient identiques, il faut
    regarder le code — pas ajuster le test.

    ``players`` reste à 2 dans les deux jambes : sans tapis ni gains, le
    conseiller ne sait rien de la table et ne peut que traiter le spot comme
    un tapis court heads-up. C'est précisément l'information que la route
    jetait — et ce que le joueur croyait avoir donnée.
    """
    commun = {"hero": "Kh 9d", "big_blind": 1, "stack": 10, "players": 2,
              "position": "SB"}
    _, jetons = _poster(base, "advise", dict(commun))
    _, dollars = _poster(base, "advise",
                         dict(commun, stacks="10, 40, 1", payouts="65, 35"))
    assert jetons["action"].upper().startswith("JAM")
    assert dollars["action"].upper().startswith("FOLD")
    assert jetons["action"] != dollars["action"], (
        f"même verdict avec et sans ICM ({jetons['action']}) : "
        "la bascule n'opère plus")


def test_le_bubble_factor_releve_l_equite_requise_postflop(base: str) -> None:
    """Postflop, la correction ICM doit être VISIBLE dans les chiffres rendus.

    Un verdict qui bascule sans que l'équité requise bouge serait un hasard,
    pas une correction.
    """
    commun = {"hero": "Ad Jh", "board": "Qs 9d 4c 2h", "pot": 100, "bet": 50,
              "stack": 300, "big_blind": 10, "position": "BTN", "players": 3}
    _, cash = _poster(base, "advise", dict(commun))
    _, tournoi = _poster(base, "advise", dict(
        commun, stacks="300, 900, 60", payouts="50, 30, 20"))
    assert cash["bubble"] is None and tournoi["bubble"] > 1.0
    assert cash["equity"] == pytest.approx(tournoi["equity"]), (
        "l'équité de la main ne dépend pas de la structure de gains")
    assert tournoi["required"] > cash["required"], (
        f"équité requise inchangée ({cash['required']} → "
        f"{tournoi['required']}) : le bubble factor n'est pas appliqué")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — LA CHAÎNE ENTIÈRE, SANS RECOPIER LES CARTES
# ═══════════════════════════════════════════════════════════════════════════


def test_les_decoupes_reelles_alimentent_le_verdict(base: str) -> None:
    """Disque → ``recognize`` → ``advise``, sans qu'aucune carte soit recopiée.

    C'est le parcours que l'utilisateur fait vraiment quand il découpe deux
    cartes d'une capture PMU. La main envoyée au conseiller est celle que la
    route a LUE ; si la lecture se trompait, le verdict serait faux, et ce
    test le verrait.
    """
    lues: list[str] = []
    for fichier in sorted(DECOUPES_REELLES):
        code, rep = _poster(base, "recognize",
                            {"image_b64": _b64(DONNEES / fichier)})
        assert code == 200 and rep["detail"][0]["statut"] == "sure", rep
        lues.append(rep["cards"][0])
    assert sorted(lues) == sorted(DECOUPES_REELLES.values())

    code, verdict = _poster(base, "advise", {
        "hero": " ".join(lues), "stack": 8, "big_blind": 1, "players": 2,
        "position": "SB"})
    assert code == 200, verdict
    assert verdict["hand"] in ("53o", "35o"), verdict["hand"]
    assert verdict["action"].upper().startswith("FOLD")
    assert verdict["confidence"] == "certain"


def test_la_capture_entiere_alimente_le_verdict(base: str,
                                                capture_postflop: dict) -> None:
    """Capture entière → ``lire_capture`` → ``advise``, mêmes règles.

    Le couple cash/tournoi est rejoué ici sur les cartes **lues**, pas
    écrites : c'est la jointure complète, celle où vivaient les défauts.
    """
    main = " ".join(capture_postflop["main"])
    board = " ".join(capture_postflop["tableau"])
    commun = {"hero": main, "board": board, "pot": 100, "bet": 50,
              "stack": 300, "big_blind": 10, "position": "BTN", "players": 3}

    code, cash = _poster(base, "advise", dict(commun))
    assert code == 200, cash
    assert cash["action"].upper().startswith("CALL")

    code, tournoi = _poster(base, "advise", dict(
        commun, stacks="300, 900, 60", payouts="50, 30, 20"))
    assert code == 200, tournoi
    assert tournoi["action"].upper().startswith("FOLD")
    assert "(ICM)" in tournoi["regime"]


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — LES REFUS : dire « je ne sais pas » plutôt que conseiller
# ═══════════════════════════════════════════════════════════════════════════


#: (nom, charge utile, fragment attendu dans le message d'erreur). Le fragment
#: est exigé pour que le refus soit CORRIGEABLE : « erreur » tout court laisse
#: l'utilisateur devant un écran muet.
REFUS = [
    ("main absente", {"stack": 10, "big_blind": 1}, "2 cartes"),
    ("main vide", {"hero": "", "stack": 10, "big_blind": 1}, "2 cartes"),
    ("une seule carte", {"hero": "Ah", "stack": 10, "big_blind": 1}, "2 cartes"),
    ("carte illisible", {"hero": "Xz 9d", "stack": 10, "big_blind": 1},
     "2 cartes"),
    ("carte en double", {"hero": "Ah Ah", "stack": 10, "big_blind": 1},
     "double"),
    ("board dans la main",
     {"hero": "Ad Jh", "board": "Ad 9d 4c", "pot": 100, "bet": 50,
      "stack": 300, "big_blind": 10}, "board"),
    ("board de 2 cartes",
     {"hero": "Ad Jh", "board": "Qs 9d", "pot": 100, "bet": 50,
      "stack": 300, "big_blind": 10}, "board"),
]


@pytest.mark.parametrize("nom,charge,fragment", REFUS,
                         ids=[r[0] for r in REFUS])
def test_un_spot_indescriptible_est_refuse_pas_conseille(
        base: str, nom: str, charge: dict, fragment: str) -> None:
    """Un logiciel qui ne sait pas dire « je ne sais pas » est plus dangereux
    qu'un logiciel absent. Chaque refus doit être un CODE D'ERREUR et un
    message diagnostiquable, jamais un verdict par défaut."""
    code, rep = _poster(base, "advise", charge)
    assert code == 400, f"{nom} : la route a répondu {code} — {rep}"
    assert "action" not in rep, f"{nom} : un verdict a été rendu quand même"
    assert fragment in rep.get("error", ""), (
        f"{nom} : message peu diagnostiquable — {rep.get('error')!r}")


def test_un_tapis_nul_en_icm_est_refuse_avec_une_consigne(base: str) -> None:
    """Défaut n°5 : l'ICM plantait sur un tapis nul.

    Le refus doit dire quoi faire — « retire-le de stacks » — sinon
    l'utilisateur ne peut pas corriger sa saisie.
    """
    code, rep = _poster(base, "advise", {
        "hero": "Ah Ad", "big_blind": 1, "stacks": "10, 40, 0",
        "payouts": "50, 30, 20", "players": 3})
    assert code == 400, rep
    assert "action" not in rep
    assert "stacks" in rep.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — DÉFAUT N°2 : un collage ne doit pas déclencher de verdict
# ═══════════════════════════════════════════════════════════════════════════


def _bloc_apres_collage(html: str) -> str:
    """Isole, dans la page SERVIE, le bloc qui décide après une lecture.

    On lit les octets que le navigateur reçoit, pas le fichier source : c'est
    ce qui est réellement exécuté chez l'utilisateur.
    """
    debut = html.index("if(shotHero.length===2){")
    fin = html.index("function loadShot", debut)
    return html[debut:fin]


def test_apres_un_collage_les_montants_non_saisis_ne_declenchent_rien(
        base: str) -> None:
    """Défaut n°2, vérifié structurellement sur la page servie par ``GET /``.

    La capture ne donne que des cartes. Le pot, la mise, le tapis et la blinde
    restent ceux du formulaire — des valeurs par défaut que personne n'a
    saisies. Déclencher le verdict dessus, c'est conseiller sur des chiffres
    inventés sans le dire.

    Limite : aucun moteur JS ici. On vérifie que le garde EXISTE, qu'il porte
    sur les quatre montants, et que ``adviseRun()`` n'est atteignable que par
    sa branche ``else``.
    """
    code, html = _obtenir(base, "/")
    assert code == 200
    bloc = _bloc_apres_collage(html)

    for champ in ("sp-pot", "sp-bet", "sp-stack", "sp-bb"):
        assert champ in bloc, f"le montant « {champ} » n'est pas gardé"
    assert "dataset.saisi" in bloc, "le garde ne consulte pas la marque de saisie"

    assert "adviseRun()" in bloc, "le verdict n'est plus jamais déclenché"
    assert bloc.index("}else{") < bloc.index("adviseRun()"), (
        "le verdict est déclenché sans que les montants aient été saisis : "
        "« adviseRun() » n'est pas dans la branche « else » du garde")


def test_la_marque_de_saisie_ne_vient_que_d_une_frappe(base: str) -> None:
    """Le garde ne vaut que si RIEN d'autre ne marque un montant comme saisi.

    Un préremplissage, une lecture d'image ou un ``value=`` posé par le code
    qui poserait aussi ``dataset.saisi`` rouvrirait exactement le défaut n°2 :
    le garde resterait présent et laisserait passer.
    """
    _, html = _obtenir(base, "/")
    poses = [i for i in range(len(html))
             if html.startswith("dataset.saisi=", i)]
    assert len(poses) == 1, (
        f"{len(poses)} endroits marquent un montant comme saisi ; un seul est "
        "légitime (l'écouteur de frappe)")
    contexte = html[max(0, poses[0] - 200):poses[0]]
    assert 'addEventListener("input"' in contexte, (
        "la marque de saisie n'est pas posée par une frappe de l'utilisateur")


def test_les_montants_gardes_sont_ceux_qui_changent_le_verdict(base: str) -> None:
    """Les quatre champs gardés doivent être ceux dont dépend la décision.

    Vérifié par le moteur, pas par lecture : on fait varier chacun d'eux et on
    exige que le verdict ou ses chiffres bougent. Un garde qui protégerait des
    champs sans effet serait décoratif.
    """
    base_charge = {"hero": "Ad Jh", "board": "Qs 9d 4c 2h", "pot": 100,
                   "bet": 50, "stack": 300, "big_blind": 10, "position": "BTN"}
    _, temoin = _poster(base, "advise", dict(base_charge))
    for champ, autre in (("pot", 20.0), ("bet", 400.0), ("big_blind", 1.0)):
        _, varie = _poster(base, "advise", dict(base_charge, **{champ: autre}))
        assert (varie["action"], varie["required"], varie["ev_bb"]) != (
            temoin["action"], temoin["required"], temoin["ev_bb"]), (
            f"faire varier « {champ} » ne change rien au verdict : "
            "le garde-fou protégerait un champ sans effet")
