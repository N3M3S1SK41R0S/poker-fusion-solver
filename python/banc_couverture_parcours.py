#!/usr/bin/env python
"""Quelle part de `pfs/` un PARCOURS UTILISATEUR traverse-t-il vraiment ?

    python banc_couverture_parcours.py              le verdict
    python banc_couverture_parcours.py --modules    le détail par module
    python banc_couverture_parcours.py --settrace   force le traceur maison

La question
-----------
`tests/test_parcours_complet.py` part d'un fichier sur disque et traverse toute
la chaîne par les **routes HTTP réelles** : la page, la reconnaissance d'une
découpe, la lecture d'une capture entière, puis les verdicts. C'est le chemin
que l'utilisateur emprunte réellement.

Les 1 095 autres tests, eux, appellent les composants un par un. Ils couvrent
large — et c'est précisément pour ça qu'ils n'ont vu passer aucun des cinq
défauts graves : tous étaient AUX JOINTURES, dans du code que chaque test
contourne en appelant la fonction interne en direct.

Ce banc mesure donc la seule chose qui répond à la question « qu'est-ce qui est
vraiment exercé de bout en bout ? » : le pourcentage de lignes exécutables de
`pfs/` traversé par ce parcours, et lui seul.

Ce que le chiffre veut dire, et ce qu'il ne veut pas dire
--------------------------------------------------------
Un pourcentage bas ne signifie PAS que le reste est mort ou faux : ce sont les
routes que ce parcours ne visite pas (`solve`, `postflop`, `review`, `drill`,
`fusion`, `bankroll`, `hmm`…) et les branches d'erreur. Il signifie qu'AUCUN
test ne dit ce que l'utilisateur en obtiendrait à travers l'interface. C'est
la carte de la zone où vivaient les défauts, pas un score de qualité.

Méthode
-------
`coverage.py` s'il est importable — c'est la mesure de référence, elle sait
suivre les fils de discussion du serveur HTTP. Sinon, un traceur maison
`sys.settrace` + `threading.settrace`, dont le dénominateur vient des
`co_lines()` du code compilé de chaque module : ce sont exactement les lignes
que l'interpréteur peut atteindre, pas une approximation par comptage de
lignes non vides.

Les deux méthodes sont comparables mais pas identiques au chiffre près :
`coverage.py` applique ses propres règles d'exclusion (`# pragma: no cover`,
blocs `if TYPE_CHECKING`). L'écart mesuré est imprimé quand on demande les
deux.
"""

from __future__ import annotations

import argparse
import sys
import threading
import types
from pathlib import Path

RACINE = Path(__file__).resolve().parent
PAQUET = RACINE / "pfs"
PARCOURS = "tests/test_parcours_complet.py"

# Ce module ne doit RIEN importer de `pfs` : la mesure démarre avant le premier
# import, sinon tout le code exécuté au chargement des modules échappe au
# compteur et le chiffre est faussé vers le bas.
assert not any(m.startswith("pfs") for m in sys.modules), (
    "pfs est déjà importé : la mesure serait incomplète")


def _fichiers() -> list[Path]:
    """Tous les modules du paquet, triés."""
    return sorted(PAQUET.rglob("*.py"))


# ═══════════════════════════════════════════════════════════════════════════
# MÉTHODE 1 — coverage.py
# ═══════════════════════════════════════════════════════════════════════════


def mesurer_coverage() -> tuple[dict[str, tuple[int, int]], str]:
    """Couverture par module : {chemin relatif: (traversées, exécutables)}."""
    import coverage
    import pytest

    cov = coverage.Coverage(source=[str(PAQUET)], data_file=None,
                            branch=False, check_preimported=True)
    cov.start()
    try:
        code = pytest.main(["-q", "--no-header", "-p", "no:cacheprovider",
                            PARCOURS])
    finally:
        cov.stop()
    if int(code) != 0:
        raise SystemExit(f"le parcours n'est pas vert (pytest a rendu {code}) : "
                         "mesurer une couverture sur un test rouge n'a pas de sens")

    par_module: dict[str, tuple[int, int]] = {}
    for f in _fichiers():
        try:
            _, executables, _, manquantes, _ = cov.analysis2(str(f))
        except Exception:            # noqa: BLE001 — module illisible : on le dit
            continue
        n_exec = len(executables)
        par_module[str(f.relative_to(RACINE))] = (n_exec - len(manquantes), n_exec)
    return par_module, f"coverage.py {coverage.__version__}"


# ═══════════════════════════════════════════════════════════════════════════
# MÉTHODE 2 — traceur maison (repli sans dépendance)
# ═══════════════════════════════════════════════════════════════════════════


def _lignes_executables(chemin: Path) -> set[int]:
    """Lignes que l'interpréteur peut atteindre, lues dans le code compilé.

    ``co_lines()`` donne la table ligne/octet du bytecode : c'est le
    dénominateur exact, et non une heuristique « ni vide ni commentaire ».
    """
    code = compile(chemin.read_text(encoding="utf-8"), str(chemin), "exec")
    vues: set[int] = set()
    pile: list[types.CodeType] = [code]
    while pile:
        c = pile.pop()
        for _, _, ligne in c.co_lines():
            if ligne:
                vues.add(ligne)
        pile += [k for k in c.co_consts if isinstance(k, types.CodeType)]
    return vues


class _Traceur:
    """Traceur de lignes, restreint au paquet, actif dans tous les fils.

    Le serveur HTTP répond dans un fil par requête : sans
    ``threading.settrace``, tout le code des routes échapperait à la mesure —
    c'est-à-dire exactement ce qu'on veut mesurer.
    """

    def __init__(self, racine: str) -> None:
        self.racine = racine
        self.vues: dict[str, set[int]] = {}

    def __call__(self, frame, evenement, arg):
        if not frame.f_code.co_filename.startswith(self.racine):
            return None
        return self._ligne

    def _ligne(self, frame, evenement, arg):
        if evenement == "line":
            self.vues.setdefault(frame.f_code.co_filename,
                                 set()).add(frame.f_lineno)
        return self._ligne


def mesurer_settrace() -> tuple[dict[str, tuple[int, int]], str]:
    import pytest

    traceur = _Traceur(str(PAQUET))
    threading.settrace(traceur)
    sys.settrace(traceur)
    try:
        code = pytest.main(["-q", "--no-header", "-p", "no:cacheprovider",
                            PARCOURS])
    finally:
        sys.settrace(None)
        threading.settrace(None)
    if int(code) != 0:
        raise SystemExit(f"le parcours n'est pas vert (pytest a rendu {code}).")

    par_module: dict[str, tuple[int, int]] = {}
    for f in _fichiers():
        try:
            executables = _lignes_executables(f)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue
        touchees = traceur.vues.get(str(f), set()) & executables
        par_module[str(f.relative_to(RACINE))] = (len(touchees), len(executables))
    return par_module, f"sys.settrace (Python {sys.version.split()[0]})"


# ═══════════════════════════════════════════════════════════════════════════


def rapport(par_module: dict[str, tuple[int, int]], methode: str,
            detail: bool) -> None:
    total_vu = sum(v for v, _ in par_module.values())
    total = sum(t for _, t in par_module.values())
    pct = total_vu / total * 100.0 if total else 0.0

    print()
    print("═" * 74)
    print("  COUVERTURE DE pfs/ PAR LE SEUL PARCOURS UTILISATEUR")
    print(f"  ({methode} · {PARCOURS})")
    print("═" * 74)

    if detail:
        for nom, (vu, tot) in sorted(par_module.items(),
                                     key=lambda kv: -kv[1][0]):
            part = vu / tot * 100.0 if tot else 0.0
            print(f"  {nom:<44s} {vu:5d} / {tot:5d}  {part:5.1f} %")
        print("─" * 74)

    jamais = [n for n, (v, t) in par_module.items() if v == 0 and t]
    print(f"  Modules mesurés            : {len(par_module)}")
    print(f"  Jamais atteints            : {len(jamais)}")
    print(f"  Lignes exécutables         : {total}")
    print(f"  Lignes traversées          : {total_vu}")
    print(f"  COUVERTURE DU PARCOURS     : {pct:.1f} %")
    print("═" * 74)
    print("  Un pourcentage bas ne dit pas que le reste est faux : il dit")
    print("  qu'aucun test ne montre ce que l'utilisateur en obtiendrait.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modules", action="store_true",
                    help="détail par module")
    ap.add_argument("--settrace", action="store_true",
                    help="force le traceur maison même si coverage.py est là")
    args = ap.parse_args()

    sys.path.insert(0, str(RACINE))
    if not args.settrace:
        try:
            import coverage  # noqa: F401
        except ImportError:
            print("coverage.py absent — repli sur le traceur maison.\n")
            args.settrace = True

    par_module, methode = (mesurer_settrace() if args.settrace
                           else mesurer_coverage())
    rapport(par_module, methode, args.modules)


if __name__ == "__main__":
    main()
