#!/usr/bin/env python
"""Banc de la garantie éthique : la frontière live / conseil est-elle tenue ?

    python banc_contournements_live.py
    python banc_contournements_live.py --seul D_appel_http --montrer
    python banc_contournements_live.py --test tests/un_autre_fichier.py

Pourquoi ce banc existe
-----------------------
`tests/test_live_sans_conseil.py` garde la contrainte la plus importante du
projet : le logiciel ne conseille jamais pendant une main d'argent réel. Un
test de cette portée doit être MESURÉ, pas cru sur parole — et la seule
mesure qui vaille est adverse : écrire les contournements, et vérifier qu'ils
échouent.

Depuis le 14 août 2026, le mode Live avec conseil (argent fictif) ajoute une
SECONDE frontière, inverse de la première : la route `live/table` DOIT
conseiller, mais UNIQUEMENT après un verdict de gate `mode == "live"`
(armement manuel confirmé + badge « PMU PLAY » vu). `tests/test_live_conseil_
gate.py` la garde. Le banc l'éprouve avec les contournements H–L.

Ce banc écrit treize fuites réelles dans le dépôt, une par une, lance le
fichier de test qui la garde, puis remet les sources d'origine. Chaque fuite
est FONCTIONNELLE : `--montrer` affiche le verdict qu'elle fait effectivement
sortir du mode live (« CALL (confortable) », « OUVRIR (relance) »,
« JAM (tapis) »…), sur une capture synthétique injectée. A–G sont mesurés
contre l'ANCIENNE frontière (`test_live_sans_conseil.py`), H–L contre la
NOUVELLE (`test_live_conseil_gate.py`) ; chaque contournement porte son
fichier de test dans `CONTOURNEMENTS`.

Mesure du 11 août 2026, sur les deux versions du test
-----------------------------------------------------
=====================  ==========================  =====================
contournement          test « analyse statique »   test renforcé
=====================  ==========================  =====================
TÉMOIN import statique  ÉCHOUE (attrapé)            ÉCHOUE (attrapé)
A  import dynamique     PASSE  ← trou               ÉCHOUE (attrapé)
B  chaîne libre         PASSE  ← trou               ÉCHOUE (attrapé)
C  autre route          PASSE  ← trou               ÉCHOUE (attrapé)
D  appel HTTP           PASSE  ← trou               ÉCHOUE (attrapé)
E  code dans le résumé  PASSE  ← trou               ÉCHOUE (attrapé)
F  code dans le statut  PASSE  ← trou               ÉCHOUE (attrapé)
=====================  ==========================  =====================

Contre-mesure du 11 août 2026 (revue adverse)
---------------------------------------------
Un huitième contournement — G, le contournement F déplacé dans
`card_recognizer` — passait au vert sur le test renforcé lui-même. Le
recalcul censé vérifier chaque valeur empruntait la MÊME fonction que le
chemin live : la fuite était recalculée à l'identique. Il est désormais
attrapé par la vérité-terrain exigée sur toute la famille de tables.

=====================  ==========================  =====================
contournement          test renforcé (1re version) test + vérité-terrain
=====================  ==========================  =====================
G  code dans le
   recogniseur          PASSE  ← trou               ÉCHOUE (attrapé)
=====================  ==========================  =====================

Mesure du 14 août 2026 — la porte live/table (H–L)
--------------------------------------------------
Cinq attaques nées de l'étude du mode Live, chacune sur un maillon distinct
de la preuve LIVE. Mesurées contre `tests/test_live_conseil_gate.py`.

=============================  ================  ==================================
contournement                  verdict           attrapé par
=============================  ================  ==================================
H  route sans gate             ÉCHOUE (attrapé)  test_le_contrat_de_la_route_est_clos
                                                 (+ non_arme, désarmement, calculateurs)
I  gate fail-open              ÉCHOUE (attrapé)  test_une_panne_du_gate_ne_conseille_pas
J  badge menteur               ÉCHOUE (attrapé)  test_arme_sans_badge_reste_en_review
                                                 (+ souvenir_expire, changement_de_titre)
K  titre corrobore             PASSE  ← trou     — puis test_le_titre_play_money_ne_
                                                 corrobore_pas_le_badge (ajouté) ⇒ attrapé
L  corroboration abandonnée    ÉCHOUE (attrapé)  test_arme_sans_badge_reste_en_review
=============================  ================  ==================================

K était un TROU : les tests existants n'exerçaient qu'un titre neutre, jamais
un titre « argent fictif » face au profil PMU PLAY — or ce profil IGNORE le
titre par construction (sur ce client il ne discrimine rien, voir
`ComplianceGate.profil_pmu_play`). Un nouveau test l'exige : armé sous un
titre menteur, badge absent, le gate doit rester REVIEW. Une fois posé, K est
attrapé — sans affaiblir aucun test existant.

Sécurité du banc
----------------
Les sources modifiées (`pfs/vision/live.py`, `pfs/app/server.py`,
`pfs/vision/card_recognizer.py`, et pour H–L `pfs/compliance/gate.py`,
`pfs/app/mode_live.py`, `pfs/vision/badge_pmu.py`) sont sauvegardées avant
toute écriture et restaurées dans un `finally`, puis comparées par empreinte
SHA-256. Le banc REFUSE de démarrer si une sauvegarde d'une exécution
précédente traîne encore — signe qu'une exécution a été interrompue et que le
dépôt est peut-être encore patché. Dans ce cas : comparer chaque source à sa
sauvegarde `.banc_contournements_sauvegarde`, restaurer la source patchée
depuis sa sauvegarde, puis effacer les sauvegardes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PY = Path(__file__).resolve().parent
RACINE = PY.parent
LIVE = PY / "pfs" / "vision" / "live.py"
SERVER = PY / "pfs" / "app" / "server.py"
RECO = PY / "pfs" / "vision" / "card_recognizer.py"
GATE = PY / "pfs" / "compliance" / "gate.py"
MODE_LIVE = PY / "pfs" / "app" / "mode_live.py"
BADGE = PY / "pfs" / "vision" / "badge_pmu.py"

#: Ancienne frontière : les routes de calibration ne conseillent JAMAIS.
TEST_ANCIEN = "tests/test_live_sans_conseil.py"
#: Nouvelle frontière : `live/table` ne conseille QUE prouvé LIVE.
TEST_CONSEIL = "tests/test_live_conseil_gate.py"
#: Défaut historique conservé pour la compatibilité de `--test`.
TEST = TEST_ANCIEN

#: Les sources que le banc a le droit de patcher — donc de sauvegarder et de
#: restaurer. `card_recognizer.py` s'y trouve depuis le contournement G : la
#: frontière à tenir est celle de la SORTIE du mode live, pas celle d'un
#: fichier, et le recogniseur est en amont de cette sortie. `gate.py`,
#: `mode_live.py` et `badge_pmu.py` s'y ajoutent le 14 août 2026 : la porte
#: live/conseil (H–L) court sur ces trois maillons.
PATCHABLES = (LIVE, SERVER, RECO, GATE, MODE_LIVE, BADGE)

#: Les sauvegardes vivent à côté des sources, avec un suffixe reconnaissable :
#: si le banc est tué, le fichier reste visible et le prochain lancement
#: refuse de partir plutôt que d'écraser une source déjà patchée.
SUFFIXE = ".banc_contournements_sauvegarde"


# ═══════════════════════════════════════════════════════════════════════════
# LES HUIT FUITES DE L'ANCIENNE FRONTIÈRE (TÉMOIN, A–G)
#
# « live/lire ne conseille JAMAIS. » Chacune est un patch textuel sur la
# source d'ORIGINE. Elles sont écrites comme les écrirait quelqu'un qui veut
# vraiment contourner la garantie, pas comme des cas d'école : c'est la seule
# façon de savoir ce que le test tient. Mesurées contre `test_live_sans_
# conseil.py`.
# ═══════════════════════════════════════════════════════════════════════════

def temoin_import_statique(src: str) -> str:
    """L'import naïf en tête de module — le seul cas que l'ancien test voyait.

    Sert de témoin de sensibilité : si CELUI-CI passait, le fichier de test
    serait vide de sens.
    """
    return src.replace(
        "from pfs.vision.archive import enregistrer_echec\n",
        "from pfs.analysis.spot_advisor import advise\n"
        "from pfs.vision.archive import enregistrer_echec\n",
    )


def a_import_dynamique(src: str) -> str:
    """`importlib` dans `lire_ecran`, verdict dans un champ au nom neutre.

    L'analyse AST ne voit que les noeuds `Import` / `ImportFrom` : un
    `importlib.import_module(...)` est un appel de fonction, donc invisible.
    Et le champ porteur s'appelle « commentaire » : aucun mot d'une liste de
    termes interdits ne s'y trouve.
    """
    src = src.replace(
        '    cartes: list[CarteLue] = field(default_factory=list)\n'
        '    image_b64: str = ""\n',
        '    cartes: list[CarteLue] = field(default_factory=list)\n'
        '    image_b64: str = ""\n'
        '    commentaire: str = ""\n',
    )
    return src.replace(
        '    return LectureLive(fenetre=titre or "auto",\n'
        '                       largeur=image.width, hauteur=image.height,\n'
        '                       cartes=cartes, image_b64=png_b64)\n',
        '    import importlib\n'
        '    _a = importlib.import_module("pfs.analysis")\n'
        '    _main = " ".join(c.carte for c in cartes\n'
        '                     if c.role == "hero" and c.carte)\n'
        '    _txt = ""\n'
        '    if _main.count(" ") == 1:\n'
        '        _tab = " ".join(c.carte for c in cartes\n'
        '                        if c.role == "board" and c.carte)\n'
        '        _av = _a.advise(_a.Spot(hero=_main, board=_tab, pot=10.0,\n'
        '                                bet=5.0, stack=100.0, big_blind=1.0))\n'
        '        _txt = f"{_av.action} - {_av.confidence}"\n'
        '    return LectureLive(fenetre=titre or "auto",\n'
        '                       largeur=image.width, hauteur=image.height,\n'
        '                       cartes=cartes, image_b64=png_b64,\n'
        '                       commentaire=_txt)\n',
    )


def b_chaine_libre(src: str) -> str:
    """Le verdict concaténé à `resume()` — aucun champ, aucun import statique.

    `resume()` est déjà affiché dans l'interface et dans le terminal : y
    ajouter huit caractères suffit.
    """
    return src.replace(
        "        bouts = [f\"{r} {' '.join(v)}\" for r, v in sorted(par_role.items())]\n"
        '        return f"{\' | \'.join(bouts)}  ({self.sures}/{len(self.cartes)} sûres)"\n',
        "        bouts = [f\"{r} {' '.join(v)}\" for r, v in sorted(par_role.items())]\n"
        '        base = f"{\' | \'.join(bouts)}  ({self.sures}/{len(self.cartes)} sûres)"\n'
        "        mains = [c.carte for c in self.cartes\n"
        "                 if c.role == 'hero' and c.carte]\n"
        "        if len(mains) == 2:\n"
        "            import importlib\n"
        '            _a = importlib.import_module("pfs.analysis")\n'
        "            av = _a.advise(_a.Spot(hero=' '.join(mains), pot=10.0,\n"
        "                                   bet=5.0, stack=100.0,\n"
        "                                   big_blind=1.0))\n"
        '            base += f"   [{av.action}]"\n'
        "        return base\n",
    )


def c_autre_route(src: str) -> str:
    """La route `live/lire` appelle une AUTRE route du serveur.

    `pfs/vision/live.py` n'est pas touché : un test qui ne regarde que ce
    fichier ne peut rien voir.
    """
    return src.replace(
        '        return {\n'
        '            "fenetre": lecture.fenetre,\n',
        '        _note = ""\n'
        '        _m = [c.carte for c in lecture.cartes\n'
        '              if c.role == "hero" and c.carte]\n'
        '        if len(_m) == 2:\n'
        '            _note = API.advise({"hero": " ".join(_m), "pot": 10,\n'
        '                                "bet": 5, "stack": 100,\n'
        '                                "big_blind": 1})["action"]\n'
        '        return {\n'
        '            "note": _note,\n'
        '            "fenetre": lecture.fenetre,\n',
    )


def d_appel_http(src: str) -> str:
    """Le verdict arrive par le RÉSEAU — aucun import à intercepter.

    Le banc démarre un point d'entrée HTTP d'appoint qui rend un verdict ;
    sur la machine de l'utilisateur ce serait `/api/advise` du serveur PFS
    lui-même. Noter le `try/except Exception` : il avale l'exception d'une
    sentinelle, d'où l'importance que celle-ci ENREGISTRE la violation avant
    de la lever.
    """
    return src.replace(
        '        return {\n'
        '            "fenetre": lecture.fenetre,\n',
        '        _note = ""\n'
        '        _url = os.environ.get("PFS_CONSEIL_URL", "")\n'
        '        if _url:\n'
        '            import json as _j\n'
        '            import urllib.request as _u\n'
        '            try:\n'
        '                _rep = _u.urlopen(_url, timeout=5).read()\n'
        '                _note = _j.loads(_rep)["action"]\n'
        '            except Exception as _e:\n'
        '                _note = f"conseil indisponible: {_e}"\n'
        '        return {\n'
        '            "note": _note,\n'
        '            "fenetre": lecture.fenetre,\n',
    )


def e_code_dans_resume(src: str) -> str:
    """Un conseil CODÉ : ni import, ni réseau, ni mot de vocabulaire.

    Une table de mains écrite en dur et un seul caractère ajouté au résumé.
    Aucun lexique ne peut l'attraper — seule la forme close de `resume()`.
    """
    return src.replace(
        "        bouts = [f\"{r} {' '.join(v)}\" for r, v in sorted(par_role.items())]\n"
        '        return f"{\' | \'.join(bouts)}  ({self.sures}/{len(self.cartes)} sûres)"\n',
        "        bouts = [f\"{r} {' '.join(v)}\" for r, v in sorted(par_role.items())]\n"
        '        base = f"{\' | \'.join(bouts)}  ({self.sures}/{len(self.cartes)} sûres)"\n'
        "        m = [c.carte for c in self.cartes\n"
        "             if c.role == 'hero' and c.carte]\n"
        "        if len(m) == 2 and m[0][0] in 'AKQJT' and m[1][0] in 'AKQJT':\n"
        "            base += ' *'\n"
        "        return base\n",
    )


def f_code_dans_statut(src: str) -> str:
    """Un conseil codé dans un champ à domaine FINI, et CONDITIONNEL.

    `statut` — trois valeurs possibles, aucun caractère en trop — détourné
    pour signifier « joue / ne joue pas », et seulement pour les mains hors
    AKQ. C'est ce contournement qui a montré qu'exercer UNE seule table de
    référence (héros Ah Kd) ne suffisait pas : la condition ne s'y déclenche
    jamais.
    """
    return src.replace(
        "        lues.append(CarteLue(\n"
        "            role=role, boite=(b.x, b.y, b.w, b.h),\n"
        "            carte=m.card, candidat=m.best_guess, statut=m.statut,\n"
        "            ecart=m.distance, marge=m.margin))\n",
        "        _s = m.statut\n"
        "        if role == 'hero' and m.card and m.card[0] not in 'AKQ':\n"
        "            _s = 'propose'\n"
        "        lues.append(CarteLue(\n"
        "            role=role, boite=(b.x, b.y, b.w, b.h),\n"
        "            carte=m.card, candidat=m.best_guess, statut=_s,\n"
        "            ecart=m.distance, marge=m.margin))\n",
    )


def g_statut_dans_le_recogniseur(src: str) -> str:
    """Le contournement F, déplacé d'un module vers le bas.

    Même sémantique (« joue / ne joue pas » codé dans `statut`), mais posé
    dans `card_recognizer.identify_card_autour` au lieu de `live.py`. C'est
    le contournement qui a montré la limite du RECALCUL : les tests
    recalculaient chaque valeur avec `identify_card_autour`, c'est-à-dire
    avec la fonction même qu'emprunte le chemin live — la fuite était donc
    recalculée à l'identique, et l'égalité tenait.

    Il est conditionné aux rangs 7/6/5 : ABSENTS de la table de référence
    (Ah Kd | Ts 4h 9d 2c Kh), PRÉSENTS dans la famille (7c 5d | 6h 2s Ad Qd).
    Ce n'est donc pas une fuite « hors échantillon » : la famille l'exerce,
    et il passait quand même. Ce qui l'attrape est la VÉRITÉ-TERRAIN de la
    table synthétique, seule référence qui ne vienne pas du recogniseur.
    """
    src = src.replace(
        "    direct = _lecture_fond_plein(image, (x, y, w, h))\n"
        "    if direct is not None:\n"
        "        return direct\n",
        "    direct = _lecture_fond_plein(image, (x, y, w, h))\n"
        "    if direct is not None:\n"
        "        return _teinter(direct)\n",
    )
    src = src.replace(
        "        m = identify_card(image.crop((xx, yy, xx + ww, yy + hh)), templates)\n"
        '        if m.statut == "sure":\n'
        "            return m\n",
        "        m = identify_card(image.crop((xx, yy, xx + ww, yy + hh)), templates)\n"
        '        if m.statut == "sure":\n'
        "            return _teinter(m)\n",
    )
    return src.replace(
        "def identify_card_autour(\n",
        "def _teinter(m):\n"
        "    # « propose » = ne joue pas. Domaine fini, aucun caractère en trop,\n"
        "    # aucun mot du lexique, aucun import, aucun réseau.\n"
        "    if m.card and m.card[0] in '765':\n"
        "        from dataclasses import replace as _r\n"
        "        return _r(m, statut='propose')\n"
        "    return m\n"
        "\n"
        "\n"
        "def identify_card_autour(\n", 1)


# ═══════════════════════════════════════════════════════════════════════════
# LES CINQ FUITES CONTRE LA PORTE LIVE / CONSEIL (14 août 2026)
#
# La frontière neuve n'est plus « live/lire ne conseille jamais » mais
# « live/table ne conseille QUE prouvé LIVE ». Chaque patch attaque un maillon
# DISTINCT de cette preuve : la garde de la route (H), la robustesse du gate
# face à la panne (I), le détecteur de badge (J), la sélection des signaux du
# profil PMU PLAY (K), la règle de corroboration du gate (L). Ils sont mesurés
# contre `tests/test_live_conseil_gate.py`.
# ═══════════════════════════════════════════════════════════════════════════

def h_route_sans_gate(src: str) -> str:
    """La route `live_table` rend un conseil sans vérifier `mode == "live"`.

    La garde qui renvoie tôt en REVIEW est neutralisée (``if False:``) : le
    calcul du conseil devient inconditionnel dès que la lecture suffit — non
    armé, désarmé, badge absent, tout donne un conseil. Le corps de la garde
    reste syntaxiquement valide mais mort, donc la fuite est un vrai conseil,
    pas une erreur d'import.

    Mesure du 14 août 2026 : ÉCHOUE (attrapé) —
    `test_non_arme_aucun_conseil_meme_avec_badge` en tête, épaulé par
    `test_le_desarmement_revoque_immediatement`,
    `test_les_calculateurs_dorment_quand_le_gate_refuse` et
    `test_le_contrat_de_la_route_est_clos`.
    """
    return src.replace(
        '        if resultat["gate"]["mode"] != "live":\n',
        '        if False:  # FUITE H : garde du gate neutralisée '
        '(mode != "live" ignoré)\n',
    )


def i_gate_fail_open(src: str) -> str:
    """Un `try/except` autour de `gate.evaluate` avale la panne et continue LIVE.

    C'est l'exact inverse du fail-closed : au lieu de laisser la panne
    remonter (aucun conseil), on fabrique une décision LIVE et on descend au
    conseil. Le gate en panne devient un gate qui dit toujours oui.

    Mesure du 14 août 2026 : ÉCHOUE (attrapé) —
    `test_une_panne_du_gate_ne_conseille_pas` : il pose une `evaluate` qui
    lève, attend que l'exception remonte (`pytest.raises(RuntimeError)`) ET
    que le conseiller n'ait pas été convoqué. La fuite avale l'exception :
    plus rien ne remonte, `pytest.raises` échoue.
    """
    return src.replace(
        '        decision: GateDecision = self.gate.evaluate(TableObservation(\n'
        '            hwnd=hwnd, window_title=titre_reel,\n'
        '            play_money_badge_detected=badge_gate))\n',
        '        try:\n'
        '            decision: GateDecision = self.gate.evaluate(TableObservation(\n'
        '                hwnd=hwnd, window_title=titre_reel,\n'
        '                play_money_badge_detected=badge_gate))\n'
        '        except Exception:\n'
        '            from pfs.compliance.gate import Mode, StakeKind\n'
        '            decision = GateDecision(\n'
        '                Mode.LIVE, StakeKind.PLAY_MONEY, (),\n'
        '                "FUITE I : gate en panne — poursuite en LIVE (fail-open)")\n',
    )


def j_badge_menteur(src: str) -> str:
    """`detecter_badge` rend toujours `vu=True` — le badge ment.

    La preuve d'identité PMU PLAY devient inconditionnelle : n'importe quelle
    capture calibrée (n'importe quel client, n'importe quelle table) est
    déclarée porteuse du badge. La corroboration du gate tombe donc pour
    n'importe quoi.

    Mesure du 14 août 2026 : ÉCHOUE (attrapé) —
    `test_arme_sans_badge_reste_en_review` : capture sans badge, fenêtre
    armée ; il exige `badge_vu is False` et `mode == "review"`. La fuite rend
    `badge_vu` vrai et fait basculer le gate en LIVE.
    """
    return src.replace(
        "        vu=filigrane_vu or dos_vu, position=position,\n",
        "        vu=True, position=position,  # FUITE J : badge menteur (toujours vu)\n",
    )


def k_titre_corrobore(src: str) -> str:
    """Le profil PMU PLAY réintègre `WindowTitleSignal` — le titre corrobore.

    La raison d'être du profil est d'IGNORER le titre : sur PMU PLAY les
    tables affichent des euros fictifs et des marqueurs de tournoi, le titre
    ne discrimine rien, et un titre « argent fictif » ne prouve pas plus qu'il
    ne se maquille. Réintroduire le signal, c'est laisser un titre menteur
    tenir lieu de badge.

    Mesure du 14 août 2026 : les tests existants utilisent un titre neutre
    (« 100 - 6-max Turbo Re-buy 424242 »), qu'aucun marqueur play-money ne
    déclenche — la fuite passait donc INAPERÇUE (trou). Comblé le 14 août par
    `test_le_titre_play_money_ne_corrobore_pas_le_badge` (armé sous un titre
    « argent fictif », badge absent) : sur code sain il reste REVIEW ; sous la
    fuite le titre corrobore, le gate passe LIVE, le test tombe. ÉCHOUE
    (attrapé) une fois le test ajouté.
    """
    return src.replace(
        "        gate._signals = [gate._user_armed, PlayMoneyBadgeSignal()]\n",
        "        gate._signals = [gate._user_armed, PlayMoneyBadgeSignal(),\n"
        "                         WindowTitleSignal()]  # FUITE K : le titre corrobore\n",
    )


def l_corroboration_abandonnee(src: str) -> str:
    """La règle #3 du gate tombe : l'armement seul suffit (sans corroboration).

    `evaluate` exige armement manuel ET au moins une corroboration forte.
    Retirer la corroboration, c'est faire passer LIVE une fenêtre armée dont
    la capture ne porte AUCUNE preuve — exactement le cas « un autre client
    affiché dans une fenêtre armée » : badge absent, mais conseil quand même.

    Mesure du 14 août 2026 : ÉCHOUE (attrapé) —
    `test_arme_sans_badge_reste_en_review` : la capture sans badge, fenêtre
    armée, doit rester REVIEW. La fuite la fait basculer en LIVE.
    """
    return src.replace(
        "        if armed and corroboration:\n",
        "        if armed:  # FUITE L : corroboration abandonnée, armement seul suffit\n",
    )


CONTOURNEMENTS = {
    "TEMOIN_import_statique": (LIVE, temoin_import_statique, TEST_ANCIEN),
    "A_import_dynamique": (LIVE, a_import_dynamique, TEST_ANCIEN),
    "B_chaine_libre": (LIVE, b_chaine_libre, TEST_ANCIEN),
    "C_autre_route": (SERVER, c_autre_route, TEST_ANCIEN),
    "D_appel_http": (SERVER, d_appel_http, TEST_ANCIEN),
    "E_code_dans_resume": (LIVE, e_code_dans_resume, TEST_ANCIEN),
    "F_code_dans_statut": (LIVE, f_code_dans_statut, TEST_ANCIEN),
    "G_statut_dans_recogniseur": (RECO, g_statut_dans_le_recogniseur, TEST_ANCIEN),
    "H_route_sans_gate": (SERVER, h_route_sans_gate, TEST_CONSEIL),
    "I_gate_fail_open": (MODE_LIVE, i_gate_fail_open, TEST_CONSEIL),
    "J_badge_menteur": (BADGE, j_badge_menteur, TEST_CONSEIL),
    "K_titre_corrobore": (GATE, k_titre_corrobore, TEST_CONSEIL),
    "L_corroboration_abandonnee": (GATE, l_corroboration_abandonnee, TEST_CONSEIL),
}


# ═══════════════════════════════════════════════════════════════════════════
# HARNAIS — exercer le chemin live sans fenêtre réelle
# ═══════════════════════════════════════════════════════════════════════════

_HARNAIS = '''
import base64, io, json, os, tempfile
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(prefix="pfs_banc_")
from dataclasses import asdict
from pfs.vision import live
from pfs.vision.synth_table import TableSpec, render_table

t = render_table(TableSpec(hero=("Ah", "Kd"),
                           board=("Ts", "4h", "9d", "2c", "Kh"),
                           felt="feutre vert", size=(1280, 800),
                           theme="pmu_solid", card_w=68, card_h=92,
                           decor=True, seed=4242))
tampon = io.BytesIO(); t.image.save(tampon, format="PNG")
png = tampon.getvalue()
live.capturer_fenetre = lambda titre=None, timeout_s=8: png

lecture = live.lire_ecran("table d'essai")
d = asdict(lecture); d.pop("image_b64", None)
d["resume()"] = lecture.resume()
print("lire_ecran   :", json.dumps(d, ensure_ascii=False)[:400])

from pfs.app.server import API
r = API.live_lire({"fenetre": "table d'essai"})
r.pop("image_b64", None)
print("live/lire    :", json.dumps(r, ensure_ascii=False)[:400])
'''


class _PointDeConseil(BaseHTTPRequestHandler):
    """Point d'entrée HTTP qui rend un verdict — tient lieu de /api/advise."""

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        corps = json.dumps({"action": "JAM (tapis)"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)


def _demarrer_point_de_conseil() -> tuple[ThreadingHTTPServer, str]:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), _PointDeConseil)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}/"


# ═══════════════════════════════════════════════════════════════════════════
# MÉCANIQUE
# ═══════════════════════════════════════════════════════════════════════════

def _empreinte(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sauvegarder() -> dict[Path, str]:
    """Copie les sources modifiables et rend leurs empreintes d'origine."""
    empreintes: dict[Path, str] = {}
    for cible in PATCHABLES:
        sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
        if sauvegarde.exists():
            raise SystemExit(
                f"une sauvegarde traîne : {sauvegarde}\n"
                "Une exécution précédente a été interrompue et le dépôt est "
                "peut-être encore patché. Compare les deux fichiers, remets "
                "la source à la main, puis efface la sauvegarde.")
        shutil.copy2(cible, sauvegarde)
        empreintes[cible] = _empreinte(cible)
    return empreintes


def _restaurer(empreintes: dict[Path, str]) -> None:
    for cible in PATCHABLES:
        sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
        if sauvegarde.exists():
            shutil.copy2(sauvegarde, cible)
        if _empreinte(cible) != empreintes[cible]:
            raise SystemExit(f"RESTAURATION RATÉE sur {cible} — vérifie-la !")


def _nettoyer() -> None:
    for cible in PATCHABLES:
        sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
        sauvegarde.unlink(missing_ok=True)


#: basetemp DÉDIÉ au banc — sans lui, pytest partage `pytest-of-<user>` avec
#: les autres exécutions et bute sur un `WinError 5` (accès refusé) quand un
#: `pytest-current` traîne, verrouillé par une exécution concurrente. Piège
#: Windows du dépôt (double bind du temp), à ne pas reproduire.
_BASETEMP = Path(tempfile.gettempdir()) / "pfs_banc_pytest"


def _lancer_pytest(test: str, env_sup: dict | None = None) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": str(PY), **(env_sup or {})}
    r = subprocess.run([sys.executable, "-m", "pytest", test, "-q",
                        "--no-header", "-p", "no:cacheprovider",
                        "--basetemp", str(_BASETEMP)],
                       cwd=str(PY), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _lancer_harnais(env_sup: dict | None = None) -> str:
    env = {**os.environ, "PYTHONPATH": str(PY), **(env_sup or {})}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "harnais.py"
        script.write_text(_HARNAIS, encoding="utf-8")
        r = subprocess.run([sys.executable, str(script)], cwd=str(PY),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
    return (r.stdout or "") + (r.stderr or "")[-800:]


# ── harnais de la NOUVELLE frontière : la route live/table ──────────────────
# Il arme (ou non) une fenêtre, injecte une capture 1920×1361 (badge présent
# ou absent selon l'attaque) et imprime le verdict du gate + le conseil que la
# route fait sortir. Sur code sain la scène ne rend AUCUN conseil ; sous la
# fuite correspondante, un conseil apparaît — c'est ce que `--montrer` affiche.

_PREAMBULE_TABLE = '''
import io, os, tempfile
os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(prefix="pfs_banc_")
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pfs.app.mode_live as mode_live
import pfs.app.server as server
from pfs.app.server import API
from pfs.vision import live as vision_live
from pfs.vision.synth_table import FELTS, TableSpec, render_table
from pfs.vision.digit_ocr import _chemin_police

HWND = 777001
GAB = Path(mode_live.__file__).resolve().parents[2] / "pfs" / "vision" \\
    / "templates" / "pmu_play"
POLICE = _chemin_police("segoeui.ttf") or _chemin_police("arial.ttf")


def frame(avec_badge):
    spec = TableSpec(hero=("Ah", "Kd"), board=("2c", "7d", "Jh"), pot=6.75,
                     hero_stack=42.5, to_call=2.25, card_w=138, card_h=169,
                     size=(1920, 1361), taille_texte=22, decor=False)
    img = render_table(spec).image
    d = ImageDraw.Draw(img)
    d.rectangle([1150, int(1361 * 0.86), 1920, int(1361 * 0.94)],
                fill=FELTS["feutre vert"])
    if POLICE:
        d.text((1400, 1250), "PAYER 2,25 BB", fill=(240, 240, 228),
               font=ImageFont.truetype(str(POLICE), 22))
    if avec_badge:
        dos = Image.open(GAB / "sprite_dos_pmu_play.png").convert("RGB")
        img.paste(dos, (940, 110))
    t = io.BytesIO(); img.save(t, format="PNG"); return t.getvalue()


def montre(titre, r):
    g = r.get("gate", {})
    c = r.get("conseil")
    verdict = c["action"] if c else "(aucun — " + \\
        str(r.get("conseil_refus", "?"))[:60] + ")"
    print("live/table  :", "gate=" + str(g.get("mode")),
          "badge_vu=" + str(g.get("badge_vu")), "-> conseil:", verdict)


mode_live._hwnd_par_titre = lambda titre: HWND
server._MODE_LIVE_INSTANCE = None
'''

#: Scène propre à chaque attaque, ajoutée au préambule.
_SCENES_TABLE = {
    "H_route_sans_gate": (
        'vision_live.capturer_fenetre = lambda *a, **k: frame(True)\n'
        'print("scène : NON armé, badge présent — la garde du gate doit tout '
        'refuser")\n'
        'montre("t", API.live_table({"fenetre": "table"}))\n'
    ),
    "I_gate_fail_open": (
        'vision_live.capturer_fenetre = lambda *a, **k: frame(True)\n'
        'API.live_armer({"fenetre": "table", "confirme": True})\n'
        'from pfs.compliance.gate import ComplianceGate\n'
        'ComplianceGate.evaluate = lambda self, obs: '
        '(_ for _ in ()).throw(RuntimeError("panne simulée"))\n'
        'print("scène : gate en PANNE — le fail-closed doit tout refuser")\n'
        'montre("t", API.live_table({"fenetre": "table"}))\n'
    ),
    "J_badge_menteur": (
        'vision_live.capturer_fenetre = lambda *a, **k: frame(False)\n'
        'API.live_armer({"fenetre": "table", "confirme": True})\n'
        'print("scène : armé, badge ABSENT — sans preuve, aucun conseil")\n'
        'montre("t", API.live_table({"fenetre": "table"}))\n'
    ),
    "K_titre_corrobore": (
        'vision_live.capturer_fenetre = lambda *a, **k: frame(False)\n'
        'titre = "PMU PLAY - argent fictif - table 7"\n'
        'API.live_armer({"fenetre": titre, "confirme": True})\n'
        'print("scène : armé sous titre \\"argent fictif\\", badge ABSENT — '
        'le titre ne doit pas corroborer")\n'
        'montre(titre, API.live_table({"fenetre": titre}))\n'
    ),
    "L_corroboration_abandonnee": (
        'vision_live.capturer_fenetre = lambda *a, **k: frame(False)\n'
        'API.live_armer({"fenetre": "table", "confirme": True})\n'
        'print("scène : armé, badge ABSENT — armement seul ne suffit pas")\n'
        'montre("t", API.live_table({"fenetre": "table"}))\n'
    ),
}


def _lancer_harnais_table(nom: str, env_sup: dict | None = None) -> str:
    scene = _SCENES_TABLE.get(nom)
    if scene is None:
        return "(pas de scène de démonstration pour ce contournement)"
    env = {**os.environ, "PYTHONPATH": str(PY), **(env_sup or {})}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "harnais_table.py"
        script.write_text(_PREAMBULE_TABLE + scene, encoding="utf-8")
        r = subprocess.run([sys.executable, str(script)], cwd=str(PY),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=env)
    return (r.stdout or "") + (r.stderr or "")[-800:]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=None,
                    help="forcer un fichier de test pour TOUS les "
                         "contournements (défaut : chacun le sien — "
                         f"{TEST_ANCIEN} pour A–G, {TEST_CONSEIL} pour H–L)")
    ap.add_argument("--seul", metavar="NOM",
                    help="ne jouer qu'un contournement")
    ap.add_argument("--montrer", action="store_true",
                    help="afficher le verdict que chaque fuite fait sortir")
    a = ap.parse_args()

    a_jouer = {nom: v for nom, v in CONTOURNEMENTS.items()
               if not a.seul or a.seul == nom}
    if not a_jouer:
        print(f"aucun contournement nommé {a.seul!r}")
        return 1

    empreintes = _sauvegarder()
    point, url = _demarrer_point_de_conseil()
    resultats: dict[str, str] = {}
    try:
        # Témoin : chaque fichier de test réellement joué doit être VERT intact.
        fichiers = sorted({a.test or tf for (_, _, tf) in a_jouer.values()})
        for tf in fichiers:
            code, sortie = _lancer_pytest(tf)
            derniere = sortie.strip().splitlines()[-1] if sortie.strip() else ""
            print(f"témoin — {tf} intact : "
                  f"{'VERT' if code == 0 else 'ROUGE'}   {derniere}")
            if code != 0:
                print(sortie[-2000:])
                return 1

        for nom, (cible, patch, tf_defaut) in a_jouer.items():
            test = a.test or tf_defaut
            sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
            origine = sauvegarde.read_text(encoding="utf-8")
            patche = patch(origine)
            if patche == origine:
                resultats[nom] = "PATCH RATÉ (ancre introuvable)"
                print(f"!! {nom} : ancre introuvable — la source a changé, "
                      "le contournement est à réécrire")
                continue

            cible.write_text(patche, encoding="utf-8")
            env = {"PFS_CONSEIL_URL": url} if nom == "D_appel_http" else None
            try:
                if a.montrer:
                    print(f"\n── {nom} : ce que la fuite fait sortir "
                          f"({test}) ──")
                    if tf_defaut == TEST_CONSEIL:
                        print(_lancer_harnais_table(nom, env))
                    else:
                        print(_lancer_harnais(env))
                code, sortie = _lancer_pytest(test, env)
            finally:
                _restaurer(empreintes)

            derniere = sortie.strip().splitlines()[-1] if sortie.strip() else ""
            resultats[nom] = ("PASSE ← TROU" if code == 0
                              else "ÉCHOUE (attrapé)")
            print(f"{nom:<28} {resultats[nom]:<20} {derniere}")
    finally:
        _restaurer(empreintes)
        _nettoyer()
        point.shutdown()

    print("\n──── BILAN ────")
    for nom, v in resultats.items():
        print(f"  {nom:<24} {v}")
    trous = [n for n, v in resultats.items() if v.startswith("PASSE")]
    if trous:
        print(f"\n{len(trous)} contournement(s) NON attrapé(s) : {trous}")
        return 1
    print("\nTous les contournements sont attrapés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
