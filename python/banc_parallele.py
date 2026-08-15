"""Banc du PARALLÉLISME du blueprint — combien de classes à la fois ?

Le dimensionnement de ``banc_blueprint.py`` chiffre le coût d'UNE classe et
multiplie par 1 755. Il rend donc des **heures-cœur séquentielles**, pas des
heures d'horloge : les 1 755 classes canoniques sont indépendantes (aucune
n'a besoin du résultat d'une autre), donc calculables en parallèle. Reste à
savoir ce que le parallélisme rend VRAIMENT sur cette machine — la question
n'est pas rhétorique :

* numpy délègue à un BLAS déjà multi-thread ; si un seul solve sature déjà
  plusieurs cœurs, lancer N processus ne divise pas le temps par N, il les
  fait se battre pour les mêmes unités de calcul (et peut RALENTIR le tout) ;
* chaque processus paie sa propre mémoire (arbre + matrices de feuille) ;
* sur Windows, chaque processus fils réimporte numpy/scipy/pfs (~1 s), coût
  amorti seulement si la tâche dure beaucoup plus que ça.

Le banc mesure donc le **débit réel** (classes par heure) à 1, 2, 4, 8 et 16
processus, sur le solve retenu par le dimensionnement (flop à profondeur
limitée, ``leaf_model="rollout"``), et en déduit l'accélération observée et
le temps d'horloge des 1 755 classes. Un chiffre mesuré, pas un facteur
supposé.

Usage ::

    python banc_parallele.py                 # 1, 2, 4, 8, 16 procs
    python banc_parallele.py --procs 1,4,16  # points choisis
    python banc_parallele.py --iters 8       # itérations par classe

Le nombre de classes mesurées par point est choisi pour que chaque processus
en reçoive au moins deux : sinon on mesurerait surtout le coût d'amorçage.

Mesuré le 15 août 2026 (32 threads logiques, 4 itérations/classe, solve à
9,04 s dans un processus seul) ::

    procs   débit          accél.   horloge des 1 755 classes
        1    484,9 cl/h     1,00×      3,6 h
        4   1450,1 cl/h     2,99×      1,2 h
        8   2201,7 cl/h     4,54×      0,8 h
       16   3169,2 cl/h     6,53×      0,6 h

**L'accélération plafonne à ×6,5, pas ×16** : le BLAS de numpy occupe déjà
plusieurs cœurs par solve, donc les processus se disputent les mêmes unités
de calcul (rendement décroissant net dès 8 : ×4,5 pour 8 processus, ×6,5
pour 16). C'est ce facteur MESURÉ — et non le nombre de threads — qu'il
faut appliquer aux heures séquentielles de ``banc_blueprint.py`` pour
obtenir un temps d'horloge. Aux paliers de ce banc (flop à profondeur
limitée) : éco 334 h → **≈ 51 h**, standard 666 h → **≈ 102 h**, fin
1 331 h → **≈ 205 h**.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor

# Les fils réimportent ce module : tout ce qui coûte cher reste dans les
# fonctions, et le travail utile est isolé dans `_solve_une_classe`.

POT, STACK = 6.0, 97.0
BET_FRACS = (0.5, 1.0)
MAX_BETS = 2
OOP_SPEC = "22+, ATs+, KQs, AJo+"
IP_SPEC = "55+, A9s+, KQs, AJo+"


def _solve_une_classe(args: tuple[tuple[int, ...], int]) -> float:
    """Résout une classe et rend son temps — exécuté dans un processus fils."""
    flop, iterations = args
    from pfs.core.range_model import parse_range
    from pfs.solver.postflop import PostflopSolver

    oop, ip = parse_range(OOP_SPEC), parse_range(IP_SPEC)
    t0 = time.perf_counter()
    PostflopSolver(list(flop), oop, ip, pot=POT, stack=STACK,
                   bet_fracs=BET_FRACS, max_bets=MAX_BETS,
                   leaf_model="rollout").solve(iterations)
    return time.perf_counter() - t0


def _classes(n: int) -> list[tuple[int, ...]]:
    """`n` classes canoniques réelles, prises dans l'énumération du blueprint."""
    from pfs.solver.blueprint import enumerate_classes

    return [c.board if hasattr(c, "board") else tuple(c[0])
            for c in list(enumerate_classes())[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--procs", default="1,2,4,8,16",
                    help="nombres de processus à mesurer")
    ap.add_argument("--iters", type=int, default=6,
                    help="itérations CFR par classe")
    a = ap.parse_args()
    points = [int(x) for x in a.procs.split(",") if x.strip()]

    coeurs = os.cpu_count() or 1
    print("═" * 74)
    print("  BANC DU PARALLÉLISME — débit réel du calcul du blueprint")
    print("═" * 74)
    print(f"  Machine : {coeurs} threads logiques — Python "
          f"{platform.python_version()} — {a.iters} itérations/classe")
    print(f"  Solve mesuré : flop profondeur limitée (rollout), "
          f"max_bets={MAX_BETS}, tailles {BET_FRACS}")
    print()

    # Référence séquentielle : une classe, à froid puis à chaud.
    besoin = max(points) * 2
    classes = _classes(besoin)
    t_seq = _solve_une_classe((classes[0], a.iters))
    print(f"  Une classe, dans CE processus : {t_seq:.2f} s")
    print()
    print(f"  {'procs':>5}  {'classes':>7}  {'horloge':>9}  {'débit':>12}  "
          f"{'accél.':>7}  {'1 755 classes':>14}")
    print("  " + "─" * 68)

    base_debit = None
    for p in points:
        n = max(2 * p, 4)
        lot = [(classes[i % len(classes)], a.iters) for i in range(n)]
        t0 = time.perf_counter()
        if p == 1:
            for item in lot:
                _solve_une_classe(item)
        else:
            with ProcessPoolExecutor(max_workers=p) as ex:
                list(ex.map(_solve_une_classe, lot))
        horloge = time.perf_counter() - t0
        debit = n / horloge * 3600.0                    # classes par heure
        if base_debit is None:
            base_debit = debit
        accel = debit / base_debit
        total_h = 1755.0 / debit
        print(f"  {p:>5}  {n:>7}  {horloge:>7.1f} s  {debit:>8.1f} cl/h  "
              f"{accel:>6.2f}×  {total_h:>11.1f} h")

    print()
    print("  Lecture : « 1 755 classes » est le temps d'HORLOGE du blueprint")
    print("  complet à ce réglage. L'accélération plafonne quand les solves se")
    print("  disputent les mêmes unités de calcul (BLAS déjà multi-thread) ou")
    print("  la mémoire — c'est ce plafond, mesuré, qui fixe le palier tenable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
