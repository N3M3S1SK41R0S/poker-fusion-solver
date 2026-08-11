#!/usr/bin/env python
"""Quels modules de `pfs/` aucun `import` ne relie au point d'entrée ?

    python banc_atteignabilite_statique.py             le verdict
    python banc_atteignabilite_statique.py --modules   le détail par module
    python banc_atteignabilite_statique.py --outils    entrées = app + outils CLI
    python banc_atteignabilite_statique.py --json      sortie machine

La question, et seulement elle
------------------------------
Ce banc répond à **une** question, la plus faible des trois que le dépôt
publie : existe-t-il une chaîne d'instructions `import` qui, partant du point
d'entrée de l'application (``pfs/__main__.py``), finit par charger ce module ?

Ce que ce chiffre NE dit PAS, et il faut le lire deux fois :

  * un module **atteint** par ce banc peut n'exécuter que ses `def` au
    chargement et ne jamais rien calculer pour l'utilisateur. L'import prouve
    le chargement, jamais l'exécution ;
  * un module **non atteint** est, lui, une preuve solide : sans import, aucun
    de ses octets ne peut s'exécuter dans l'application. C'est la seule
    direction où ce banc conclut fermement.

C'est pourquoi ce nombre est **structurellement plus petit** que celui de
`banc_couverture_parcours.py` (lignes non traversées) et que celui de
`banc_inertie_causale.py` (calculs traversés mais sans effet). Les trois
mesurent trois choses différentes ; les rapprocher est une faute de
raisonnement, cf. `README.md` § « Trois nombres, trois méthodes ».

Méthode
-------
Analyse syntaxique (`ast`) de chaque fichier de `pfs/`, y compris les imports
**différés** écrits dans le corps des fonctions — le serveur en fait un usage
massif (``from pfs.analysis import advise`` à l'intérieur de la route). Un
parcours en largeur depuis les entrées donne l'ensemble atteignable.

Vérifié : le paquet ne contient **aucun** `importlib.import_module` ni
`__import__` (rejoué par ce banc, qui échoue s'il en apparaît un). Un import
dynamique échapperait à l'analyse et gonflerait à tort le nombre de modules
« morts » ; la garde est donc nécessaire, pas décorative.

Dénominateur en lignes : les *statements* de `coverage.py`, exactement ceux du
banc de couverture, pour que les deux nombres soient comparés sur la même
base. Sans `coverage.py`, repli sur les `co_lines()` du code compilé, et le
rapport le dit.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import types
from collections import deque
from pathlib import Path

RACINE = Path(__file__).resolve().parent
PAQUET = RACINE / "pfs"

#: L'application telle que l'utilisateur la lance : `python -m pfs`.
ENTREES_APP = ("pfs.__main__",)

#: Les outils en ligne de commande livrés à la racine de `python/`. Ils ne
#: sont pas dans `pfs/`, donc jamais comptés au dénominateur ; ils servent
#: uniquement de racines supplémentaires au parcours quand on passe
#: `--outils`. `demo.py` en fait partie : c'est le seul appelant de
#: `FusionEngine.decide()` hors tests, et le distinguer change le verdict sur
#: toute la chaîne de fusion.
OUTILS_CLI = ("analyser_main", "reconnaitre", "recuperer_mains", "calibrer",
              "capturer_session", "demo")


# ═══════════════════════════════════════════════════════════════════════════
# Le graphe d'imports
# ═══════════════════════════════════════════════════════════════════════════


def _nom_module(chemin: Path) -> str:
    """`pfs/core/icm.py` → `pfs.core.icm` ; `pfs/core/__init__.py` → `pfs.core`."""
    rel = chemin.relative_to(RACINE).with_suffix("")
    parties = list(rel.parts)
    if parties[-1] == "__init__":
        parties.pop()
    return ".".join(parties)


def _fichiers_paquet() -> dict[str, Path]:
    """{nom de module: fichier} pour tout `pfs/`."""
    return {_nom_module(f): f for f in sorted(PAQUET.rglob("*.py"))}


def _fichiers_outils() -> dict[str, Path]:
    """{nom: fichier} pour les scripts CLI de la racine, s'ils existent."""
    trouves: dict[str, Path] = {}
    for nom in OUTILS_CLI:
        f = RACINE / f"{nom}.py"
        if f.is_file():
            trouves[nom] = f
    return trouves


def _ancetres(nom: str) -> list[str]:
    """`pfs.core.icm` → [`pfs`, `pfs.core`] : importer un module importe ses paquets."""
    parties = nom.split(".")
    return [".".join(parties[:i]) for i in range(1, len(parties))]


def _resoudre(module_courant: str, est_paquet: bool, noeud: ast.AST,
              connus: set[str]) -> set[str]:
    """Cibles `pfs.*` d'un `import` unique, ancêtres compris."""
    cibles: set[str] = set()

    if isinstance(noeud, ast.Import):
        for alias in noeud.names:
            if alias.name in connus:
                cibles.add(alias.name)
                cibles.update(a for a in _ancetres(alias.name) if a in connus)
        return cibles

    if not isinstance(noeud, ast.ImportFrom):
        return cibles

    if noeud.level:                       # import relatif : `from .x import y`
        base = module_courant if est_paquet else module_courant.rsplit(".", 1)[0]
        for _ in range(noeud.level - 1):
            base = base.rsplit(".", 1)[0] if "." in base else ""
        prefixe = f"{base}.{noeud.module}" if noeud.module else base
    else:
        prefixe = noeud.module or ""

    if prefixe in connus:
        cibles.add(prefixe)
        cibles.update(a for a in _ancetres(prefixe) if a in connus)

    # `from pfs.core import icm` : `icm` est un sous-module, pas un attribut.
    for alias in noeud.names:
        sous = f"{prefixe}.{alias.name}" if prefixe else alias.name
        if sous in connus:
            cibles.add(sous)
            cibles.update(a for a in _ancetres(sous) if a in connus)
    return cibles


def construire_graphe(fichiers: dict[str, Path]) -> dict[str, set[str]]:
    """{module: modules qu'il importe}, imports différés compris.

    Échoue si un import dynamique apparaît : l'analyse statique ne peut pas le
    suivre, et un banc qui l'ignorerait en silence surestimerait le code mort.
    """
    connus = set(fichiers)
    graphe: dict[str, set[str]] = {}
    for nom, chemin in fichiers.items():
        arbre = ast.parse(chemin.read_text(encoding="utf-8"), str(chemin))
        est_paquet = chemin.name == "__init__.py"
        aretes: set[str] = set()
        for noeud in ast.walk(arbre):
            if isinstance(noeud, (ast.Import, ast.ImportFrom)):
                aretes |= _resoudre(nom, est_paquet, noeud, connus)
            elif isinstance(noeud, ast.Call):
                f = noeud.func
                dyn = (isinstance(f, ast.Name) and f.id == "__import__") or (
                    isinstance(f, ast.Attribute) and f.attr == "import_module")
                if dyn:
                    raise SystemExit(
                        f"{chemin}:{noeud.lineno} — import dynamique détecté. "
                        "L'analyse statique ne le suit pas : ce banc doit être "
                        "étendu avant de publier un chiffre.")
        graphe[nom] = aretes
    return graphe


def atteignables(graphe: dict[str, set[str]], entrees: tuple[str, ...]) -> set[str]:
    """Fermeture transitive des imports depuis les entrées."""
    vus: set[str] = set()
    file: deque[str] = deque(e for e in entrees if e in graphe)
    manquantes = [e for e in entrees if e not in graphe]
    if manquantes:
        raise SystemExit(f"entrées introuvables : {manquantes}")
    while file:
        m = file.popleft()
        if m in vus:
            continue
        vus.add(m)
        file.extend(graphe.get(m, ()))
    return vus


# ═══════════════════════════════════════════════════════════════════════════
# Le dénominateur en lignes
# ═══════════════════════════════════════════════════════════════════════════


def _lignes_co(chemin: Path) -> int:
    """Repli sans dépendance : lignes atteignables du code compilé."""
    code = compile(chemin.read_text(encoding="utf-8"), str(chemin), "exec")
    vues: set[int] = set()
    pile: list[types.CodeType] = [code]
    while pile:
        c = pile.pop()
        for _, _, ligne in c.co_lines():
            if ligne:
                vues.add(ligne)
        pile += [k for k in c.co_consts if isinstance(k, types.CodeType)]
    return len(vues)


def compter_lignes(fichiers: dict[str, Path]) -> tuple[dict[str, int], str]:
    """{module: lignes exécutables}, même définition que le banc de couverture."""
    try:
        import coverage
    except ImportError:
        return ({n: _lignes_co(f) for n, f in fichiers.items()},
                f"co_lines (Python {sys.version.split()[0]}) — coverage.py absent")

    # `analysis2` sans collecte préalable : les *statements* sont lus par le
    # parseur, aucun code n'est exécuté. C'est la seule façon d'obtenir le
    # dénominateur EXACT du banc de couverture — un décompte maison divergerait
    # de quelques lignes et rendrait les deux mesures incomparables pour de
    # mauvaises raisons.
    cov = coverage.Coverage(data_file=None)
    compte: dict[str, int] = {}
    try:
        for nom, f in fichiers.items():
            _, statements, _, _, _ = cov.analysis2(str(f))
            compte[nom] = len(statements)
    except Exception as e:              # noqa: BLE001 — API changée : on le dit
        return ({n: _lignes_co(f) for n, f in fichiers.items()},
                f"co_lines (Python {sys.version.split()[0]}) — "
                f"coverage.analysis2 indisponible ({type(e).__name__})")
    return compte, f"coverage.py {coverage.__version__} (statements)"


# ═══════════════════════════════════════════════════════════════════════════


def mesurer(avec_outils: bool) -> dict:
    fichiers = _fichiers_paquet()
    graphe_fichiers = dict(fichiers)
    entrees = ENTREES_APP
    if avec_outils:
        outils = _fichiers_outils()
        graphe_fichiers |= outils
        entrees = ENTREES_APP + tuple(outils)

    graphe = construire_graphe(graphe_fichiers)
    vus = atteignables(graphe, entrees) & set(fichiers)   # on ne compte que pfs/
    lignes, methode_lignes = compter_lignes(fichiers)

    morts = sorted(n for n in fichiers if n not in vus)
    total_lignes = sum(lignes.values())
    lignes_mortes = sum(lignes[n] for n in morts)
    return {
        "entrees": list(entrees),
        "methode_lignes": methode_lignes,
        "modules": len(fichiers),
        "modules_atteints": len(vus),
        "modules_morts": len(morts),
        "liste_morts": morts,
        "lignes": total_lignes,
        "lignes_mortes": lignes_mortes,
        "part_modules": len(morts) / len(fichiers) * 100.0 if fichiers else 0.0,
        "part_lignes": lignes_mortes / total_lignes * 100.0 if total_lignes else 0.0,
        "detail": {n: (n in vus, lignes[n]) for n in sorted(fichiers)},
    }


def rapport(r: dict, detail: bool) -> None:
    print()
    print("═" * 74)
    print("  (a) MODULES DE pfs/ QU'AUCUN IMPORT NE RELIE AU POINT D'ENTRÉE")
    print(f"  (entrées : {', '.join(r['entrees'])})")
    print(f"  (lignes : {r['methode_lignes']})")
    print("═" * 74)

    if detail:
        for nom, (vu, n) in r["detail"].items():
            print(f"  {'atteint  ' if vu else 'INATTEINT'}  {nom:<40s} {n:5d}")
        print("─" * 74)

    for nom in r["liste_morts"]:
        print(f"  mort : {nom:<44s} {r['detail'][nom][1]:5d} lignes")
    print("─" * 74)
    print(f"  Modules du paquet          : {r['modules']}")
    print(f"  Atteints par un import     : {r['modules_atteints']}")
    print(f"  JAMAIS IMPORTÉS            : {r['modules_morts']}"
          f"  ({r['part_modules']:.1f} % des modules)")
    print(f"  Lignes exécutables         : {r['lignes']}")
    print(f"  LIGNES JAMAIS IMPORTÉES    : {r['lignes_mortes']}"
          f"  ({r['part_lignes']:.1f} % des lignes)")
    print("═" * 74)
    print("  Atteint ≠ exécuté : ce banc prouve seulement le chargement.")
    print("  Ce nombre n'est comparable ni à la couverture, ni à l'inertie.")
    print()


def croiser(r: dict, fichier: str) -> dict:
    """Compare l'ensemble (a) « jamais importé » à l'ensemble (b) « jamais traversé ».

    Produit le fichier avec ::

        python banc_couverture_parcours.py --json couverture.json

    L'intérêt n'est pas la somme mais l'**écart** : il montre que les deux
    mesures ne se recouvrent pas, et dans les DEUX sens. Un module importé et
    jamais exécuté est invisible à (a) ; un module que seuls les tests
    importent est compté mort par (a) et couvert par (b).
    """
    charge = json.loads(Path(fichier).read_text(encoding="utf-8"))
    par_module = charge["par_module"]

    def _cle(chemin: str) -> str:
        return _nom_module(RACINE / chemin.replace("\\", "/"))

    # Les modules sans ligne exécutable (paquets vides) sortent du croisement :
    # « jamais traversé » n'a pas de sens pour un fichier qui n'a rien à traverser.
    jamais_traverses = {_cle(c) for c, (vu, tot) in par_module.items()
                        if vu == 0 and tot}
    jamais_importes = {n for n in r["liste_morts"] if r["detail"][n][1]}

    return {
        "fichier_couverture": fichier,
        "methode_couverture": charge.get("methode"),
        "a_jamais_importes": sorted(jamais_importes),
        "b_jamais_traverses": sorted(jamais_traverses),
        "les_deux": sorted(jamais_importes & jamais_traverses),
        "a_seulement": sorted(jamais_importes - jamais_traverses),
        "b_seulement": sorted(jamais_traverses - jamais_importes),
    }


def rapport_croisement(c: dict) -> None:
    print()
    print("═" * 74)
    print("  CROISEMENT (a) JAMAIS IMPORTÉ  ×  (b) JAMAIS TRAVERSÉ")
    print(f"  ({c['methode_couverture']})")
    print("═" * 74)
    print(f"  (a) seul : {len(c['a_seulement'])}"
          "   modules que SEULS LES TESTS importent —")
    for n in c["a_seulement"]:
        print(f"             {n}")
    print("             morts pour l'application, couverts par la mesure.")
    print(f"  (b) seul : {len(c['b_seulement'])}"
          "   modules IMPORTÉS et JAMAIS EXÉCUTÉS —")
    for n in c["b_seulement"]:
        print(f"             {n}")
    print("             charger n'est pas exécuter ; (a) ne peut pas les voir.")
    print(f"  les deux : {len(c['les_deux'])}")
    print("─" * 74)
    print("  Les deux ensembles se recouvrent partiellement et débordent dans")
    print("  les deux sens : aucun des deux nombres ne confirme l'autre.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", action="store_true", help="détail par module")
    ap.add_argument("--outils", action="store_true",
                    help="ajouter les scripts CLI de python/ aux entrées")
    ap.add_argument("--croiser", metavar="COUVERTURE.json",
                    help="croiser avec la sortie de banc_couverture_parcours.py --json")
    ap.add_argument("--json", action="store_true", help="sortie machine")
    args = ap.parse_args()

    r = mesurer(avec_outils=args.outils)
    c = croiser(r, args.croiser) if args.croiser else None
    if args.json:
        r.pop("detail")
        if c:
            r["croisement"] = c
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        rapport(r, args.modules)
        if c:
            rapport_croisement(c)


if __name__ == "__main__":
    main()
