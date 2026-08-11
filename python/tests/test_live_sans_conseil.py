"""La calibration en direct lit l'écran, elle ne conseille jamais.

Pourquoi un test et pas seulement une intention
-----------------------------------------------
`pfs.vision.live` capture la fenêtre d'un client poker pendant que la partie
est en cours. C'est légitime pour une seule raison : il n'en sort que le
constat de ce que la machine a *vu*. Dès qu'un verdict, une équité ou un
seuil de bascule franchirait cette frontière, l'outil cesserait d'être un
banc d'essai pour devenir une assistance en temps réel — ce que le projet
refuse.

Une intention ne se relit pas dans six mois, et un refactoring ne la
consulte pas. Ces tests, si.

Pourquoi la version « analyse statique » ne suffisait pas
---------------------------------------------------------
La première version de ce fichier ne regardait que la FORME du code : les
imports écrits en toutes lettres dans `live.py` (analyse AST) et les NOMS
des champs de sortie. Six contournements écrits et exécutés le 11 août 2026
passaient tous au vert alors que la réponse du mode live contenait un
verdict lisible :

  A. `importlib.import_module("pfs.analysis")` appelé DANS `lire_ecran`,
     verdict rangé dans un champ nommé « commentaire » →
     « CALL (confortable) - indicatif » ;
  B. aucun nouveau champ, aucun import statique : le verdict concaténé à la
     fin de `LectureLive.resume()`, une chaîne de texte libre déjà affichée →
     « … (7/9 sûres)   [OUVRIR (relance)] » ;
  C. `live.py` intact : c'est la route `live/lire` du serveur qui appelle
     une AUTRE route du même serveur (`API.advise`) → « OUVRIR (relance) » ;
  D. `live.py` intact : la route fait une requête HTTP sortante et récupère
     le verdict par le réseau → « JAM (tapis) » ;
  E. aucun import, aucun réseau, aucun mot de vocabulaire : une table
     push/fold écrite en dur dans `live.py` et un simple « * » ajouté au
     résumé ;
  F. le champ `statut` — domaine fini, aucun caractère en trop — détourné
     pour signifier « joue / ne joue pas ».

Seul un septième cas — l'import statique naïf en tête de `live.py` — était
attrapé. Une garantie qui n'attrape que le contournement que personne
n'écrirait n'est pas une garantie.

Ce que ces tests vérifient maintenant
-------------------------------------
Le chemin live est **exécuté pour de vrai**, avec une capture synthétique
injectée à la place de la fenêtre (aucune fenêtre réelle n'est nécessaire),
et la réponse est examinée sur quatre plans indépendants :

  1. **le contrat de sortie est CLOS** — l'ensemble exact des champs de
     `LectureLive`, de `CarteLue` et des clés de la route `live/lire` est
     épinglé. Un champ de plus, quel que soit son nom, échoue (A, C, D) ;
  2. **le domaine de chaque valeur est CLOS** — `role`, `statut`, `carte`
     appartiennent à des ensembles finis, `resume()` doit correspondre à une
     forme stricte (E), et chaque valeur rendue doit être EXACTEMENT celle
     que le recogniseur recalcule sur la même découpe (B, F). Rien ne peut
     voyager dans une chaîne libre ni dans un nombre ;
  3. **aucun calculateur de décision n'est joignable à l'exécution** —
     sentinelles sur l'appel des fonctions de décision, sur l'import (y
     compris `importlib`), sur le réseau et sur les sous-processus (A, C, D) ;
  4. **le lexique** — filet de rattrapage : aucun terme d'action, d'équité,
     d'espérance ou de seuil ne doit apparaître dans une valeur ou une clé.

Et la réciproque, qui compte autant : `test_la_calibration_rend_toute_la_
lecture` vérifie que la sortie contient bien la carte, le candidat, le
statut, l'écart, la marge et la boîte de chaque carte. Une garantie qui
viderait la sortie serait inutile.

Limites mesurées, à dire franchement
------------------------------------
* Le lexique attrape du VOCABULAIRE, pas un code arbitraire. Ce n'est pas
  lui qui bloque les contournements E et F mais le point 1 (contrat clos) et
  le point 2 (domaines clos) — d'où l'importance de ne jamais les assouplir
  « juste pour ajouter un champ pratique ».
* Une fuite CONDITIONNELLE ne se voit que si l'on varie la condition. Le
  contournement F, conditionné aux mains hors AKQ, passait au vert tant que
  ce fichier n'exerçait qu'une table (héros Ah Kd). D'où la famille de
  tables ci-dessous — qui reste un échantillon fini : une fuite conditionnée
  à une main absente de la famille passerait encore. Les tests de contrat
  clos, eux, ne dépendent d'aucune main.
* Le recalcul par `identify_card_autour` n'est PAS indépendant : c'est la
  fonction qu'emprunte le chemin live. Il prouve que `live.py` et le serveur
  n'ont rien retouché, pas que la valeur soit vraie. Un conseil codé DANS le
  recogniseur passait au vert tant que la vérité-terrain n'était pas exigée
  (mesuré le 11 août 2026) ; elle l'est désormais sur toute la famille.
* `image_b64` n'est pas passé au lexique : du base64 contient des lettres
  arbitraires. Il est vérifié autrement, et plus fortement : il doit être
  **octet pour octet** la capture reçue. Aucune place pour un ajout.
* Ces tests portent sur le chemin Python. Ils ne disent rien de ce que
  l'interface HTML ferait d'une réponse propre ; `pfs/app/ui.html` n'affiche
  que les champs listés ici, et c'est une lecture de code, pas une mesure.
"""

from __future__ import annotations

import ast
import base64
import builtins
import importlib
import inspect
import io
import re
import socket
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterator

import pytest

from pfs.vision import live

# ═══════════════════════════════════════════════════════════════════════════
# LEXIQUE DES TERMES INTERDITS
#
# Un mot de cette liste dans une réponse du mode live signifie que l'outil a
# cessé de dire ce qu'il a VU pour dire ce qu'il faut FAIRE. La liste est
# volontairement écrite en clair, motif par motif : c'est elle qu'on relira
# dans six mois, et une liste qu'on ne comprend pas ne se maintient pas.
#
# Tous les motifs sont appliqués entre frontières de mot (`\b`) : sans cela
# « remise » déclencherait « mise », et « jamais » déclencherait « jam ».
# ═══════════════════════════════════════════════════════════════════════════

#: Verbes et noms d'ACTION. C'est le cœur de l'interdit : une action nommée,
#: c'est une assistance de jeu, quelle que soit la prudence de la formulation.
#: Les libellés réellement produits par `pfs.analysis.spot_advisor` sont tous
#: couverts : « JAM (tapis) », « FOLD », « CALL (confortable) », « CHECK »,
#: « MISER n (x % du pot) », « OUVRIR (relance) », « MIXTE — ouvrir … »,
#: « MARGINAL — proche de l'indifférence ».
_ACTIONS = (
    r"fold", r"couch(er|ez|é|e)", r"jete[rz]",
    r"call", r"suiv(re|ez|i)", r"pay(er|ez)",
    r"raise", r"relanc\w*", r"surrelanc\w*", r"[34]-?bet",
    r"bet", r"mise[rz]?", r"c-?bet", r"value-?bet",
    r"check", r"parole",
    r"shove", r"jam", r"push", r"all-?in", r"tapis",
    r"limp", r"open", r"ouvrir", r"ouverture",
    r"isoler", r"squeeze", r"steal", r"bluff\w*", r"d[ée]fendre",
    r"abandonner",
)

#: Grandeurs qui ne servent qu'à DÉCIDER. Une équité, une espérance ou un
#: seuil affichés pendant une main sont une assistance, même sans verbe.
#: (« écart » et « marge » sont ABSENTS à dessein : ce sont les deux mesures
#: de confiance de LECTURE, elles doivent rester visibles — voir la réciproque.)
_GRANDEURS = (
    r"[ée]quit[ée]s?", r"equity", r"ev", r"ev_bb", r"ev_icm",
    r"esp[ée]rance", r"mdf", r"cotes?", r"pot-?odds", r"odds",
    r"icm", r"nash", r"gto", r"solveur", r"solver",
    r"ranges?", r"[ée]ventail", r"fr[ée]quences?",
    r"seuils?", r"bascule", r"indiff[ée]ren\w*", r"bubble", r"bulle",
)

#: Mots du VERDICT lui-même — le vocabulaire de la recommandation, sans
#: lequel aucune assistance ne se formule.
_VERDICT = (
    r"verdicts?", r"conseil\w*", r"recommand\w*", r"pr[ée]conis\w*",
    r"d[ée]cisions?", r"optimal\w*", r"devrai[st]", r"il faut",
    r"meilleur (coup|choix|jeu|action)", r"marginal\w*", r"mixte",
    r"exploit\w*", r"profitab\w*", r"rentab\w*", r"[ée]quilibr[ée]e?s?",
)

TERMES_INTERDITS: tuple[str, ...] = _ACTIONS + _GRANDEURS + _VERDICT

_LEXIQUE = re.compile(
    r"\b(?:" + "|".join(TERMES_INTERDITS) + r")\b", re.IGNORECASE)


def conseils_detectes(texte: str) -> list[str]:
    """Termes interdits présents dans un texte, dans l'ordre de lecture."""
    return _LEXIQUE.findall(texte)


# ═══════════════════════════════════════════════════════════════════════════
# CONTRAT DE SORTIE — les ensembles exacts, rien de plus, rien de moins
# ═══════════════════════════════════════════════════════════════════════════

#: Champs de `LectureLive`. FERMÉ : ajouter un champ ici est une décision de
#: conception, pas un détail — c'est par un champ en plus que trois des six
#: contournements ont fait passer un verdict.
CHAMPS_LECTURE = {"fenetre", "largeur", "hauteur", "cartes", "image_b64"}

#: Propriétés calculées de `LectureLive` (elles sortent aussi vers l'UI).
PROPRIETES_LECTURE = {"sures", "taux"}

#: Champs de `CarteLue`. FERMÉ, même raison.
CHAMPS_CARTE = {"role", "boite", "carte", "candidat", "statut",
                "ecart", "marge"}

PROPRIETES_CARTE = {"sure"}

#: Clés de la réponse de la route HTTP `live/lire`, cas nominal. FERMÉ.
CLES_ROUTE = {"fenetre", "largeur", "hauteur", "sures", "total", "taux",
              "resume", "cartes", "image_b64"}

#: Clés de la réponse de la route en cas d'échec de capture. FERMÉ.
CLES_ROUTE_ERREUR = {"erreur"}

#: Domaines finis. Un champ dont le domaine est fini ne peut pas transporter
#: de message : c'est ce qui rend la garantie structurelle et non lexicale.
ROLES = {"hero", "board", "?"}
STATUTS = {"sure", "propose", "refus"}
_CARTE = re.compile(r"^[2-9TJQKA][cdhs]$")

#: Forme EXACTE de `resume()`. C'est la seule chaîne de texte libre de la
#: sortie ; l'épingler ferme le canal que les contournements B et E
#: empruntaient.
_FORME_RESUME = re.compile(
    r"^(?:aucune carte détectée|"
    r"(?:\?|hero|board) (?:[2-9TJQKA][cdhs]|\?)(?: (?:[2-9TJQKA][cdhs]|\?))*"
    r"(?: \| (?:\?|hero|board) (?:[2-9TJQKA][cdhs]|\?)"
    r"(?: (?:[2-9TJQKA][cdhs]|\?))*)*"
    r"  \(\d+/\d+ sûres\))$")


# ═══════════════════════════════════════════════════════════════════════════
# SENTINELLES D'EXÉCUTION
# ═══════════════════════════════════════════════════════════════════════════

#: Les points d'entrée du calcul de décision, dans CHAQUE module qui les
#: expose. Patcher `pfs.analysis.spot_advisor.advise` ne suffit pas :
#: `pfs.analysis.advise` est une AUTRE liaison vers le même objet, et c'est
#: celle qu'un `importlib.import_module("pfs.analysis")` obtient.
CALCULATEURS: tuple[tuple[str, str], ...] = (
    ("pfs.analysis.spot_advisor", "advise"),
    ("pfs.analysis", "advise"),
    ("pfs.analysis.pushfold_review", "review_pushfold_hands"),
    ("pfs.analysis", "review_pushfold_hands"),
    ("pfs.analysis.session_review", "review_hands"),
    ("pfs.analysis", "review_hands"),
    ("pfs.core.equity", "equity_vs_range"),
    ("pfs.core.equity", "equity_multiway"),
    ("pfs.core.icm", "icm_equities"),
    ("pfs.core.icm", "analyse_icm_spot"),
    ("pfs.solver.pushfold", "solve_hu_pushfold"),
    ("pfs.solver.postflop", "PostflopSolver"),
    ("pfs.fusion.arbiter", "arbitrate"),
    ("pfs.train.simulateur", "simuler"),
    ("pfs.engine", "FusionEngine"),
)

#: Modules dont le seul objet est de décider. Aucun n'est atteint par
#: `pfs.vision.*`, qui n'importe que `pfs.vision.phash`, `card_recognizer`,
#: `table_detector` et `archive` (vérifié : aucun import croisé).
_MODULE_DE_CONSEIL = re.compile(
    r"^(?:pfs\.analysis|pfs\.engine|pfs\.solver|pfs\.fusion|pfs\.train"
    r"|pfs\.core\.(?:equity|icm|bluffcatch|bankroll|range_model|bunching|rake)"
    r"|pfs\.bench)\b|spot_advisor|pushfold_review|session_review|simulateur",
    re.IGNORECASE)


@contextmanager
def sentinelles(monkeypatch) -> Iterator[list[str]]:
    """Interdit, le temps du bloc, tout accès à un calculateur de décision.

    Quatre barrières indépendantes, parce qu'un conseil peut arriver par
    quatre chemins :

    * **appel** — les fonctions de décision elles-mêmes sont remplacées ;
      quelle que soit la manière dont on les a atteintes, les appeler échoue ;
    * **import** — `builtins.__import__` (couvre `import X` et `from X
      import Y`) ET `importlib.import_module` (couvre l'import dynamique,
      qu'aucune analyse AST ne voit) ;
    * **réseau** — toute connexion sortante ; c'est ainsi qu'un conseil
      viendrait d'une autre route du même serveur sans le moindre import ;
    * **sous-processus** — un conseil pourrait être calculé ailleurs et lu
      sur la sortie standard.

    La violation est **enregistrée dans la liste rendue AVANT d'être levée**.
    C'est essentiel, et ce n'est pas théorique : le contournement D
    enveloppait son appel réseau dans un `try/except Exception`, ce qui
    avalait l'exception et rendait la sentinelle muette. La trace, elle,
    survit — et c'est elle qui a fait échouer le test.
    """
    violations: list[str] = []

    def interdit(quoi: str):
        violations.append(quoi)
        raise AssertionError(
            f"le chemin live a tenté : {quoi}. La calibration rend ce qui a "
            "été LU, jamais ce qu'il faut FAIRE.")

    with monkeypatch.context() as mp:
        # 1. appel des calculateurs
        for nom_module, nom in CALCULATEURS:
            try:
                module = importlib.import_module(nom_module)
            except ImportError:  # pragma: no cover - dépend de l'installation
                continue
            if not hasattr(module, nom):
                continue

            def piege(*a, _q=f"{nom_module}.{nom}", **k):
                interdit(f"appel de {_q}()")

            mp.setattr(module, nom, piege)

        # 2. import, statique comme dynamique
        vrai_import = builtins.__import__

        def import_surveille(nom, *a, **k):
            if _MODULE_DE_CONSEIL.search(nom):
                interdit(f"import de {nom}")
            return vrai_import(nom, *a, **k)

        mp.setattr(builtins, "__import__", import_surveille)

        vrai_import_module = importlib.import_module

        def import_module_surveille(nom, package=None):
            if _MODULE_DE_CONSEIL.search(nom):
                interdit(f"import dynamique de {nom}")
            return vrai_import_module(nom, package)

        mp.setattr(importlib, "import_module", import_module_surveille)

        # 3. réseau
        def connexion_interdite(self, adresse, *a, **k):
            interdit(f"connexion réseau vers {adresse!r}")

        mp.setattr(socket.socket, "connect", connexion_interdite)
        mp.setattr(socket.socket, "connect_ex", connexion_interdite)
        mp.setattr(socket, "create_connection",
                   lambda adresse, *a, **k: interdit(
                       f"connexion réseau vers {adresse!r}"))
        import urllib.request

        mp.setattr(urllib.request, "urlopen",
                   lambda *a, **k: interdit("requête HTTP sortante"))

        # 4. sous-processus (la capture est injectée : rien ne doit s'exécuter)
        mp.setattr(subprocess, "Popen",
                   lambda *a, **k: interdit(f"sous-processus {a[:1]}"))
        mp.setattr(subprocess, "run",
                   lambda *a, **k: interdit(f"sous-processus {a[:1]}"))

        yield violations


# ═══════════════════════════════════════════════════════════════════════════
# PARCOURS DE LA RÉPONSE
# ═══════════════════════════════════════════════════════════════════════════

def valeurs_textuelles(valeur, chemin: str = "") -> Iterator[tuple[str, str]]:
    """Chaque clé et chaque chaîne de la réponse, avec son chemin.

    Les clés sont inspectées autant que les valeurs : un champ nommé
    « ev_bb » serait un conseil même vide. Les nombres et les `None` ne
    portent pas de texte ; ils sont vérifiés ailleurs (domaines clos et
    recalcul indépendant).
    """
    if is_dataclass(valeur) and not isinstance(valeur, type):
        valeur = asdict(valeur)
    if isinstance(valeur, dict):
        for cle, sous in valeur.items():
            yield f"{chemin}.{cle}", str(cle)
            yield from valeurs_textuelles(sous, f"{chemin}.{cle}")
    elif isinstance(valeur, (list, tuple, set)):
        for i, sous in enumerate(valeur):
            yield from valeurs_textuelles(sous, f"{chemin}[{i}]")
    elif isinstance(valeur, str):
        yield chemin, valeur


def exiger_aucun_conseil(reponse, quoi: str) -> None:
    """Échoue si un terme interdit apparaît où que ce soit dans `reponse`.

    `image_b64` est écarté : du base64 est une suite de lettres arbitraires,
    l'y chercher des mots produirait des échecs au hasard. Il est vérifié
    plus fortement ailleurs — il doit être octet pour octet la capture.
    """
    for chemin, texte in valeurs_textuelles(reponse):
        if chemin.endswith("image_b64"):
            continue
        trouves = conseils_detectes(texte)
        assert not trouves, (
            f"{quoi} : le champ {chemin} contient {trouves} dans {texte!r}. "
            "Un conseil a franchi la frontière lecture / décision.")


# ═══════════════════════════════════════════════════════════════════════════
# CAPTURE INJECTÉE — le chemin live sans fenêtre réelle
# ═══════════════════════════════════════════════════════════════════════════

#: Table de référence. Graine fixée : la lecture attendue est reproductible.
HERO = ("Ah", "Kd")
BOARD = ("Ts", "4h", "9d", "2c", "Kh")
TITRE = "table d'essai"

#: Famille de tables — et pourquoi UNE table ne suffit pas.
#:
#: Le contournement F codait le conseil dans le champ `statut`, mais
#: seulement pour les mains hors AKQ. Sur la seule table de référence, dont
#: le héros tient Ah Kd, il ne se déclenchait jamais : la version de ce
#: fichier qui n'exerçait qu'une table le laissait passer au vert. Une fuite
#: CONDITIONNELLE ne se voit que si l'on varie la condition.
#:
#: La famille couvre donc, côté héros, les rangs A K Q J T 9 7 5 3 2 et les
#: quatre enseignes, avec board complet, partiel et vide (préflop), sur cinq
#: feutres. Ce n'est pas une preuve — c'est un échantillon fini, et une fuite
#: conditionnée à une main absente d'ici passerait encore.
#:
#: Le sixième feutre de `synth_table.FELTS`, « tapis clair » (203/207/214),
#: est ABSENT à dessein : le recogniseur y lit des cartes FAUSSES avec le
#: statut « sure » (mesuré le 11 août 2026 : 3s 2h lus Js Jh, Kc lu 7c). Ce
#: défaut de LECTURE est réel et mérite son propre chantier ; l'ajouter ici
#: rendrait ce test rouge pour une raison qui n'est pas la sienne. À remettre
#: dans la famille le jour où il sera corrigé.
FAMILLE: tuple[tuple[tuple[str, ...], tuple[str, ...], str, int], ...] = (
    (HERO, BOARD, "feutre vert", 4242),
    (("Qs", "Jc"), ("3d", "8s", "Tc"), "bleu nuit", 17),
    (("Th", "9h"), (), "gris moyen", 99),
    (("7c", "5d"), ("6h", "2s", "Ad", "Qd"), "bordeaux", 7),
    (("3s", "2h"), ("Jh", "4c", "Kc"), "noir", 555),
)


def png_de_table(hero: tuple[str, ...], board: tuple[str, ...],
                 feutre: str, graine: int) -> bytes:
    """Une table synthétique encodée en PNG — ce qui tient lieu de capture."""
    from pfs.vision.synth_table import TableSpec, render_table

    table = render_table(TableSpec(
        hero=hero, board=board, felt=feutre, size=(1280, 800),
        theme="pmu_solid", card_w=68, card_h=92, decor=True, seed=graine))
    tampon = io.BytesIO()
    table.image.save(tampon, format="PNG")
    return tampon.getvalue()


@pytest.fixture
def capture_png() -> bytes:
    """La table de référence, encodée en PNG."""
    return png_de_table(HERO, BOARD, "feutre vert", 4242)


@pytest.fixture
def chemin_live(monkeypatch, tmp_path, capture_png) -> bytes:
    """Le chemin live complet, alimenté par l'image plutôt que par l'écran.

    On remplace la seule étape qui a besoin de Windows — la capture — et
    tout le reste (localisation, reconnaissance, archivage, mise en forme)
    s'exécute réellement. L'archive est détournée vers un dossier temporaire
    pour ne pas polluer le banc d'essai de l'utilisateur.
    """
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(
        live, "capturer_fenetre",
        lambda titre=None, timeout_s=live._TIMEOUT_S: capture_png)
    return capture_png


# ═══════════════════════════════════════════════════════════════════════════
# 1. LA FORME DU CODE (première ligne, pas la dernière)
# ═══════════════════════════════════════════════════════════════════════════

#: Modules qui calculent une recommandation. Aucun ne doit être joignable
#: depuis la calibration, ni directement ni par un import transitif écrit
#: dans le fichier.
CONSEIL = (
    "spot_advisor",
    "pushfold_review",
    "session_review",
    "simulateur",
    "solver",
    "equity",
    "icm",
    "nash",
)

#: Termes d'un verdict. Un champ de sortie qui porterait l'un d'eux
#: signalerait que la recommandation a fuité dans la structure de données.
VERDICTS = ("verdict", "action", "conseil", "recommand", "esperance",
            "equite", "equity", "bascule", "ev_", "_ev", "fold", "call",
            "raise", "shove", "jam", "miser", "coucher", "suivre")


def _source() -> str:
    return Path(inspect.getfile(live)).read_text(encoding="utf-8")


def test_aucun_import_de_conseil() -> None:
    """Le module n'importe aucun calculateur de décision."""
    arbre = ast.parse(_source())
    importes: list[str] = []
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            importes += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            importes.append(n.module)
            importes += [f"{n.module}.{a.name}" for a in n.names]

    fautifs = [i for i in importes
               if any(c in i.lower() for c in CONSEIL)]
    assert not fautifs, (
        f"la calibration en direct importe du conseil : {fautifs}. "
        "Elle doit rendre ce qui a été LU, jamais ce qu'il faut FAIRE.")


def test_aucun_import_dynamique_dans_le_module_live() -> None:
    """`live.py` n'utilise aucun mécanisme d'import indirect.

    L'analyse AST du test précédent ne voit que les noeuds `Import` et
    `ImportFrom` : `importlib.import_module("pfs.analysis")` lui est
    invisible — c'est un simple appel de fonction. C'était le contournement A.

    Interdire ces quatre mécanismes dans ce fichier précis ne coûte rien (il
    n'en a aucun besoin) et supprime la seule façon d'atteindre un module
    sans que son nom apparaisse dans un import.
    """
    arbre = ast.parse(_source())
    fautifs: list[str] = []
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            fautifs += [f"import {a.name}" for a in n.names
                        if a.name.split(".")[0] == "importlib"]
        elif isinstance(n, ast.ImportFrom) and n.module:
            if n.module.split(".")[0] == "importlib":
                fautifs.append(f"from {n.module} import …")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in ("__import__", "eval", "exec", "compile"):
                fautifs.append(f"{n.func.id}(…)")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in ("import_module", "load_module"):
                fautifs.append(f"…{n.func.attr}(…)")

    assert not fautifs, (
        f"pfs/vision/live.py utilise un import indirect : {fautifs}. "
        "C'est exactement ce qui rend l'analyse statique aveugle : le nom du "
        "module n'apparaît plus dans un import. Si un besoin légitime "
        "apparaît, il faudra le documenter ici et renforcer les sentinelles "
        "d'exécution en conséquence.")


def test_la_lecture_ne_porte_aucun_verdict() -> None:
    """Aucun champ de `LectureLive` / `CarteLue` ne nomme une décision."""
    for classe in (live.LectureLive, live.CarteLue):
        champs = set(classe.__dataclass_fields__)
        champs |= {n for n, v in vars(classe).items()
                   if isinstance(v, property)}
        fautifs = [c for c in champs
                   if any(v in c.lower() for v in VERDICTS)]
        assert not fautifs, (
            f"{classe.__name__} expose {fautifs} : une recommandation a "
            "franchi la frontière lecture / conseil.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. LE CONTRAT DE SORTIE EST CLOS
# ═══════════════════════════════════════════════════════════════════════════

def test_le_contrat_de_sortie_est_clos() -> None:
    """La sortie a EXACTEMENT ces champs — ni plus (fuite), ni moins (trou).

    Un nom de champ innocent suffit à faire passer un verdict : le
    contournement A s'appelait « commentaire », les contournements C et D
    « note ». Aucune liste de mots interdits ne les aurait attrapés. Une
    liste FERMÉE, si.

    L'égalité est volontairement stricte dans les deux sens : retirer un
    champ casserait la réciproque (la calibration doit tout rendre), en
    ajouter un ouvre un canal. Les deux méritent une décision explicite.
    """
    assert set(live.LectureLive.__dataclass_fields__) == CHAMPS_LECTURE
    assert {n for n, v in vars(live.LectureLive).items()
            if isinstance(v, property)} == PROPRIETES_LECTURE

    assert set(live.CarteLue.__dataclass_fields__) == CHAMPS_CARTE
    assert {n for n, v in vars(live.CarteLue).items()
            if isinstance(v, property)} == PROPRIETES_CARTE


def test_la_route_live_a_un_contrat_clos(chemin_live) -> None:
    """La réponse HTTP de `live/lire` a exactement les clés attendues.

    C'est le test qui manquait : les contournements C et D ne touchaient pas
    `live.py` du tout — ils ajoutaient une clé à la réponse du serveur, que
    l'ancien fichier de test ne regardait jamais.
    """
    from pfs.app.server import API

    reponse = API.live_lire({"fenetre": TITRE})
    assert set(reponse) == CLES_ROUTE, (
        f"clés inattendues : {set(reponse) ^ CLES_ROUTE}")
    for carte in reponse["cartes"]:
        assert set(carte) == CHAMPS_CARTE


def test_la_route_live_a_un_contrat_clos_aussi_en_echec(monkeypatch) -> None:
    """Le chemin d'erreur est un canal comme un autre : il est clos aussi."""
    from pfs.app.server import API

    def capture_ratee(titre=None, timeout_s=live._TIMEOUT_S):
        raise live.CaptureImpossible(
            "la fenêtre n'a rendu aucune image en 8 s.")

    monkeypatch.setattr(live, "capturer_fenetre", capture_ratee)
    reponse = API.live_lire({"fenetre": TITRE})
    assert set(reponse) == CLES_ROUTE_ERREUR
    exiger_aucun_conseil(reponse, "réponse d'erreur de live/lire")


def test_les_domaines_de_valeur_sont_clos(chemin_live) -> None:
    """Chaque champ prend ses valeurs dans un ensemble fini et connu.

    Un champ dont le domaine est fini ne peut pas transporter de message.
    C'est ce qui rend la garantie structurelle plutôt que lexicale : même un
    conseil codé (« statut = A / B / C ») échoue ici, alors que le lexique
    seul le laisserait passer.
    """
    lecture = live.lire_ecran(TITRE)

    assert lecture.fenetre == TITRE
    assert isinstance(lecture.largeur, int) and lecture.largeur > 0
    assert isinstance(lecture.hauteur, int) and lecture.hauteur > 0

    for c in lecture.cartes:
        assert c.role in ROLES, f"rôle inattendu : {c.role!r}"
        assert c.statut in STATUTS, f"statut inattendu : {c.statut!r}"
        assert c.carte is None or _CARTE.match(c.carte), f"carte {c.carte!r}"
        assert c.candidat is None or _CARTE.match(c.candidat), \
            f"candidat {c.candidat!r}"
        assert len(c.boite) == 4 and all(isinstance(v, int) for v in c.boite)
        assert c.ecart is None or isinstance(c.ecart, int)
        assert c.marge is None or isinstance(c.marge, int)


def test_le_resume_a_une_forme_close(chemin_live) -> None:
    """`resume()` est la seule chaîne libre : sa forme exacte est épinglée.

    C'est par elle que passaient les contournements B et E — un verdict
    concaténé à la fin d'un texte déjà affiché, sans nouveau champ ni nouvel
    import. Vérifier « ne contient pas de mot interdit » ne suffit pas : le
    contournement E n'ajoutait qu'une étoile. Vérifier la FORME ferme le
    canal.
    """
    lecture = live.lire_ecran(TITRE)
    texte = lecture.resume()
    assert _FORME_RESUME.match(texte), (
        f"resume() ne respecte plus sa forme : {texte!r}. Si le format a "
        "changé pour une bonne raison, mets à jour _FORME_RESUME — mais "
        "garde-la close : c'est la seule chaîne de texte libre de la sortie.")

    vide = live.LectureLive(fenetre="t", largeur=1, hauteur=1)
    assert _FORME_RESUME.match(vide.resume())


# ═══════════════════════════════════════════════════════════════════════════
# 3. LE CHEMIN LIVE, EXÉCUTÉ POUR DE VRAI
# ═══════════════════════════════════════════════════════════════════════════

def test_lire_ecran_ne_rend_aucun_conseil(monkeypatch, chemin_live) -> None:
    """Le chemin complet s'exécute et ne produit aucune recommandation.

    Les sentinelles couvrent les chemins que l'analyse statique ne voit pas
    (import dynamique, réseau, sous-processus, appel direct d'un calculateur
    déjà chargé), le lexique couvre le contenu.
    """
    with sentinelles(monkeypatch) as violations:
        lecture = live.lire_ecran(TITRE)
    assert not violations, violations

    exiger_aucun_conseil(lecture, "LectureLive")
    exiger_aucun_conseil(lecture.resume(), "resume()")


def test_la_route_live_lire_ne_rend_aucun_conseil(monkeypatch,
                                                  chemin_live) -> None:
    """La route HTTP non plus — sentinelles armées pendant l'appel.

    `pfs.app.server` est importé AVANT d'armer les sentinelles : ce module
    importe légitimement des calculateurs pour ses autres routes. Ce qui est
    interdit, c'est que la route `live/lire` en atteigne un.
    """
    from pfs.app.server import API

    with sentinelles(monkeypatch) as violations:
        reponse = API.live_lire({"fenetre": TITRE})
    assert not violations, violations

    exiger_aucun_conseil(reponse, "réponse de live/lire")


def test_la_sortie_terminal_ne_conseille_rien(monkeypatch, chemin_live,
                                              capsys) -> None:
    """Ce que `calibrer.py` imprime à l'écran est examiné aussi.

    Le terminal est une sortie utilisateur au même titre que l'interface :
    un conseil qui n'apparaîtrait que là serait tout aussi utilisable
    pendant une main.
    """
    import calibrer

    with sentinelles(monkeypatch) as violations:
        assert calibrer._une_lecture(TITRE) is True
    assert not violations, violations

    sortie = capsys.readouterr().out
    assert sortie.strip(), "la calibration n'a rien affiché du tout"
    exiger_aucun_conseil(sortie, "sortie terminal de calibrer.py")


def test_aucune_table_ne_declenche_de_conseil(monkeypatch, tmp_path) -> None:
    """Toute la famille de tables passe par la route, sous sentinelles.

    C'est le test qui attrape les fuites CONDITIONNELLES — celles qui ne se
    déclenchent que sur certaines mains. Le contournement F codait le conseil
    dans `statut` pour les mains hors AKQ : sur la seule table de référence
    (héros Ah Kd) il restait invisible.

    Pour chaque table on vérifie, d'un seul passage : sentinelles muettes,
    clés exactes, forme du résumé, domaines clos, lexique, image identique à
    la capture, que chaque valeur rendue est EXACTEMENT celle que le
    recogniseur recalcule sur la même découpe, et que la lecture est JUSTE.

    Pourquoi la vérité-terrain est exigée ici, et pas seulement le recalcul
    ---------------------------------------------------------------------
    Le recalcul passe par `identify_card_autour`, c'est-à-dire par la MÊME
    fonction que le chemin live : il prouve que `live.py` et le serveur n'ont
    pas retouché la réponse du recogniseur, pas que cette réponse soit
    indépendante. Un conseil codé UN MODULE PLUS BAS — `statut` détourné
    dans `card_recognizer` selon le rang de la carte, exactement le
    contournement F déplacé — serait recalculé à l'identique et resterait
    invisible. Mesuré le 11 août 2026 : cette fuite passait au vert.

    La vérité-terrain de la table synthétique, elle, ne vient pas du
    recogniseur : c'est `render_table` qui l'a dessinée. Les cinq tables de
    la famille sont lues justes, et « sure » sur tous les rôles identifiés
    (mesuré le 11 août 2026). Si cet ensemble d'assertions devient rouge
    après un changement du détecteur ou du recogniseur, c'est une RÉGRESSION
    DE LECTURE à mesurer — pas une exigence à retirer.
    """
    from PIL import Image

    from pfs.app.server import API
    from pfs.vision.card_recognizer import identify_card_autour

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    for hero, board, feutre, graine in FAMILLE:
        png = png_de_table(hero, board, feutre, graine)
        monkeypatch.setattr(
            live, "capturer_fenetre",
            lambda titre=None, timeout_s=live._TIMEOUT_S, _p=png: _p)
        quoi = f"table {'/'.join(hero)} sur {feutre}"

        with sentinelles(monkeypatch) as violations:
            reponse = API.live_lire({"fenetre": TITRE})
        assert not violations, f"{quoi} : {violations}"

        assert set(reponse) == CLES_ROUTE, f"{quoi} : clés {set(reponse)}"
        exiger_aucun_conseil(reponse, quoi)
        assert _FORME_RESUME.match(reponse["resume"]), (
            f"{quoi} : résumé hors forme {reponse['resume']!r}")
        assert base64.b64decode(reponse["image_b64"]) == png, (
            f"{quoi} : l'image rendue n'est plus la capture")

        image = Image.open(io.BytesIO(png)).convert("RGB")
        assert reponse["cartes"], f"{quoi} : plus aucune carte détectée"

        # Vérité-terrain : la seule référence qui ne passe PAS par le
        # recogniseur (voir la docstring). C'est elle qui ferme le canal
        # « conseil codé dans le recogniseur lui-même ».
        assert [c["carte"] for c in reponse["cartes"]
                if c["role"] == "hero"] == list(hero), f"{quoi} : héros faux"
        assert [c["carte"] for c in reponse["cartes"]
                if c["role"] == "board"] == list(board), f"{quoi} : board faux"
        assert all(c["statut"] == "sure" for c in reponse["cartes"]
                   if c["role"] != "?"), f"{quoi} : statut détourné"

        for c in reponse["cartes"]:
            assert set(c) == CHAMPS_CARTE, f"{quoi} : champs {set(c)}"
            assert c["role"] in ROLES and c["statut"] in STATUTS, quoi
            m = identify_card_autour(image, tuple(c["boite"]))
            assert (c["carte"], c["candidat"], c["statut"],
                    c["ecart"], c["marge"]) == \
                (m.card, m.best_guess, m.statut, m.distance, m.margin), (
                    f"{quoi} : la lecture rendue pour {c['boite']} ne "
                    "correspond pas à ce que le recogniseur recalcule")


# ═══════════════════════════════════════════════════════════════════════════
# 4. LE LEXIQUE N'EST PAS TAUTOLOGIQUE
# ═══════════════════════════════════════════════════════════════════════════

def test_le_lexique_attrape_un_vrai_conseil() -> None:
    """Sans ce test, les tests de contenu pourraient être vides de sens.

    Le projet a déjà payé une fois le prix d'un test tautologique : 1057
    tests verts alors que deux gabarits de cartes étaient intervertis, parce
    que chaque gabarit n'était comparé qu'à lui-même. On vérifie donc ici que
    le lexique attrape effectivement ce qu'un vrai conseiller produit — y
    compris les fuites réellement observées lors de l'audit.
    """
    from pfs.analysis import Spot, advise

    vrai = advise(Spot(hero="Ah Kd", board="Ts 4h 9d", pot=10.0, bet=5.0,
                       stack=100.0, big_blind=1.0))
    assert conseils_detectes(vrai.action), (
        f"le lexique ne reconnaît pas l'action réelle {vrai.action!r} : les "
        "tests de contenu ne prouvent alors plus rien.")

    # Les fuites mesurées le 11 août 2026 sur les contournements A à D.
    fuites_observees = (
        "CALL (confortable) - indicatif",
        "? ? ? | board Ts 4h 9d 2c Kh | hero Ah Kd  (7/9 sûres)"
        "   [OUVRIR (relance)]",
        "OUVRIR (relance)",
        "JAM (tapis)",
    )
    for fuite in fuites_observees:
        assert conseils_detectes(fuite), f"fuite non détectée : {fuite!r}"

    # Et tous les libellés que `spot_advisor` sait produire.
    for libelle in ("FOLD", "CHECK", "MISER 7 (70 % du pot)", "MISER (valeur)",
                    "CHECK (ou bluff choisi)", "CALL (juste)",
                    "MARGINAL — proche de l'indifférence",
                    "MIXTE — ouvrir 40 % du temps", "JAM (tapis)",
                    "équité 62 %", "espérance +1,3 bb", "seuil de bascule",
                    "EV ICM négative"):
        assert conseils_detectes(libelle), f"libellé non détecté : {libelle!r}"


def test_le_lexique_ne_condamne_pas_le_vocabulaire_de_lecture() -> None:
    """Réciproque du précédent : le lexique doit rester utilisable.

    Un lexique qui déclencherait sur « marge », « écart » ou « candidat »
    forcerait à vider la sortie pour rester vert — c'est-à-dire à détruire
    la calibration pour sauver le test.
    """
    for legitime in ("hero", "board", "carte", "candidat", "statut",
                     "sure", "propose", "refus", "ecart", "écart",
                     "marge", "boite", "boîte", "largeur", "hauteur",
                     "taux", "resume", "image_b64", "fenetre",
                     "remise à l'échelle", "aucune carte détectée",
                     "7/9 sûres", "jamais", "capture refusée par le client"):
        assert not conseils_detectes(legitime), (
            f"{legitime!r} est du vocabulaire de LECTURE et déclenche le "
            "lexique : corrige le motif, ne vide pas la sortie.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. LA RÉCIPROQUE — la calibration doit tout rendre
# ═══════════════════════════════════════════════════════════════════════════

def test_la_calibration_rend_toute_la_lecture(chemin_live) -> None:
    """Une garantie qui viderait la sortie serait inutile.

    C'est le contrepoids des tests ci-dessus : on vérifie ici que le mode
    calibration rend bien TOUTE l'information de lecture — la carte, le
    candidat, le statut, l'écart, la marge, la boîte — et qu'il lit
    réellement la table.

    Les cartes attendues sont celles de la table synthétique de référence
    (graine 4242). Si ce test échoue après un changement du détecteur ou du
    recogniseur, c'est une RÉGRESSION DE LECTURE à mesurer, pas un test à
    supprimer.
    """
    lecture = live.lire_ecran(TITRE)

    lues: dict[str, list] = {}
    for c in lecture.cartes:
        lues.setdefault(c.role, []).append(c)

    assert [c.carte for c in lues.get("hero", ())] == list(HERO), (
        "les deux cartes du héros ne sont plus lues")
    assert [c.carte for c in lues.get("board", ())] == list(BOARD), (
        "le board n'est plus lu en entier")

    for c in lues["hero"] + lues["board"]:
        assert c.statut == "sure"
        assert c.candidat == c.carte
        assert isinstance(c.ecart, int) and c.ecart >= 0, (
            "l'écart de lecture a disparu de la sortie")
        assert isinstance(c.marge, int) and c.marge > 0, (
            "la marge de confiance a disparu de la sortie")
        x, y, w, h = c.boite
        assert w > 0 and h > 0 and 0 <= x and 0 <= y, (
            "la boîte n'est plus rendue")

    assert lecture.sures == 7
    assert lecture.taux == pytest.approx(lecture.sures / len(lecture.cartes))
    assert "Ah Kd" in lecture.resume() and "7/" in lecture.resume()


def test_les_nombres_sont_ceux_du_recogniseur(chemin_live) -> None:
    """`ecart` et `marge` sont recalculés indépendamment et comparés.

    Les nombres sont le dernier canal qu'un lexique ne couvre pas : une
    équité tiendrait très bien dans un entier. En exigeant que chaque valeur
    soit EXACTEMENT celle que le recogniseur produit sur la même découpe, on
    ferme ce canal — et on vérifie du même coup que la sortie n'est pas
    fabriquée.
    """
    from PIL import Image

    from pfs.vision.card_recognizer import identify_card_autour

    lecture = live.lire_ecran(TITRE)
    image = Image.open(io.BytesIO(chemin_live)).convert("RGB")

    for c in lecture.cartes:
        m = identify_card_autour(image, c.boite)
        assert (c.carte, c.candidat, c.statut, c.ecart, c.marge) == \
            (m.card, m.best_guess, m.statut, m.distance, m.margin), (
                f"la lecture rendue pour {c.boite} ne correspond pas à ce que "
                "le recogniseur calcule sur la même découpe")


def test_l_image_rendue_est_exactement_la_capture(chemin_live) -> None:
    """`image_b64` est la capture, octet pour octet — rien n'y est ajouté.

    C'est ce qui permet d'écarter ce champ du lexique sans laisser de trou :
    une égalité binaire ne laisse aucune place à un message.
    """
    lecture = live.lire_ecran(TITRE)
    assert base64.b64decode(lecture.image_b64) == chemin_live

    from pfs.app.server import API

    reponse = API.live_lire({"fenetre": TITRE})
    assert base64.b64decode(reponse["image_b64"]) == chemin_live


# ═══════════════════════════════════════════════════════════════════════════
# 6. CONSERVÉS DE LA PREMIÈRE VERSION
# ═══════════════════════════════════════════════════════════════════════════

def test_la_lecture_se_resume_sans_rien_recommander() -> None:
    """Le résumé destiné à l'utilisateur ne contient pas de verdict."""
    lecture = live.LectureLive(
        fenetre="table", largeur=1280, hauteur=800,
        cartes=[
            live.CarteLue(role="hero", boite=(0, 0, 40, 60), carte="Ah",
                          candidat="Ah", statut="sure", ecart=12, marge=90),
            live.CarteLue(role="hero", boite=(50, 0, 40, 60), carte=None,
                          candidat="Ks", statut="refus", ecart=940, marge=3),
        ])
    texte = lecture.resume().lower()
    assert not any(v in texte for v in VERDICTS), (
        f"le résumé recommande quelque chose : {texte!r}")
    assert not conseils_detectes(texte)
    assert "1/2" in texte, "le résumé doit dire combien de lectures sont sûres"


def test_taux_de_lecture() -> None:
    """`taux` mesure la part de cartes lues avec certitude."""
    vide = live.LectureLive(fenetre="t", largeur=1, hauteur=1)
    assert vide.taux == 0.0, "une lecture vide ne vaut pas 100 %"

    carte = live.CarteLue(role="board", boite=(0, 0, 1, 1), carte="2c",
                          candidat="2c", statut="sure", ecart=5, marge=80)
    pleine = live.LectureLive(fenetre="t", largeur=1, hauteur=1,
                              cartes=[carte, carte])
    assert pleine.taux == 1.0


def test_sonde_absente_dit_quoi_faire() -> None:
    """Un binaire manquant produit un message actionnable, pas un traceback.

    Le message doit contenir la commande de compilation : sur une machine
    fraîche, c'est la seule chose à faire, autant l'écrire.
    """
    try:
        live.chemin_sonde()
    except live.SondeIntrouvable as e:
        assert "cargo build" in str(e)
    else:
        pytest.skip("la sonde est compilée sur cette machine")
