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

Ce banc écrit sept fuites réelles dans le dépôt, une par une, lance le
fichier de test, puis remet les sources d'origine. Chaque fuite est
FONCTIONNELLE : `--montrer` affiche le verdict qu'elle fait effectivement
sortir du mode live (« CALL (confortable) », « OUVRIR (relance) »,
« JAM (tapis) »…), sur une capture synthétique injectée.

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

Sécurité du banc
----------------
Les sources modifiées (`pfs/vision/live.py`, `pfs/app/server.py`) sont
sauvegardées avant toute écriture et restaurées dans un `finally`, puis
comparées par empreinte SHA-256. Le banc REFUSE de démarrer si une
sauvegarde d'une exécution précédente traîne encore — signe qu'une
exécution a été interrompue et que le dépôt est peut-être encore patché.
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
TEST = "tests/test_live_sans_conseil.py"

#: Les sauvegardes vivent à côté des sources, avec un suffixe reconnaissable :
#: si le banc est tué, le fichier reste visible et le prochain lancement
#: refuse de partir plutôt que d'écraser une source déjà patchée.
SUFFIXE = ".banc_contournements_sauvegarde"


# ═══════════════════════════════════════════════════════════════════════════
# LES SEPT FUITES
#
# Chacune est un patch textuel sur la source d'ORIGINE. Elles sont écrites
# comme les écrirait quelqu'un qui veut vraiment contourner la garantie, pas
# comme des cas d'école : c'est la seule façon de savoir ce que le test tient.
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


CONTOURNEMENTS = {
    "TEMOIN_import_statique": (LIVE, temoin_import_statique),
    "A_import_dynamique": (LIVE, a_import_dynamique),
    "B_chaine_libre": (LIVE, b_chaine_libre),
    "C_autre_route": (SERVER, c_autre_route),
    "D_appel_http": (SERVER, d_appel_http),
    "E_code_dans_resume": (LIVE, e_code_dans_resume),
    "F_code_dans_statut": (LIVE, f_code_dans_statut),
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
    for cible in (LIVE, SERVER):
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
    for cible in (LIVE, SERVER):
        sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
        if sauvegarde.exists():
            shutil.copy2(sauvegarde, cible)
        if _empreinte(cible) != empreintes[cible]:
            raise SystemExit(f"RESTAURATION RATÉE sur {cible} — vérifie-la !")


def _nettoyer() -> None:
    for cible in (LIVE, SERVER):
        sauvegarde = cible.with_suffix(cible.suffix + SUFFIXE)
        sauvegarde.unlink(missing_ok=True)


def _lancer_pytest(test: str, env_sup: dict | None = None) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": str(PY), **(env_sup or {})}
    r = subprocess.run([sys.executable, "-m", "pytest", test, "-q",
                        "--no-header", "-p", "no:cacheprovider"],
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=TEST,
                    help=f"fichier de test à éprouver (défaut : {TEST})")
    ap.add_argument("--seul", metavar="NOM",
                    help="ne jouer qu'un contournement")
    ap.add_argument("--montrer", action="store_true",
                    help="afficher le verdict que chaque fuite fait sortir")
    a = ap.parse_args()

    empreintes = _sauvegarder()
    point, url = _demarrer_point_de_conseil()
    resultats: dict[str, str] = {}
    try:
        code, sortie = _lancer_pytest(a.test)
        derniere = sortie.strip().splitlines()[-1] if sortie.strip() else ""
        print(f"témoin — dépôt intact : "
              f"{'VERT' if code == 0 else 'ROUGE'}   {derniere}")
        if code != 0:
            print(sortie[-2000:])
            return 1

        for nom, (cible, patch) in CONTOURNEMENTS.items():
            if a.seul and a.seul != nom:
                continue
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
                    print(f"\n── {nom} : ce que la fuite fait sortir ──")
                    print(_lancer_harnais(env))
                code, sortie = _lancer_pytest(a.test, env)
            finally:
                _restaurer(empreintes)

            derniere = sortie.strip().splitlines()[-1] if sortie.strip() else ""
            resultats[nom] = ("PASSE ← TROU" if code == 0
                              else "ÉCHOUE (attrapé)")
            print(f"{nom:<24} {resultats[nom]:<20} {derniere}")
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
