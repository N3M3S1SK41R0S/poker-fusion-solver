"""Calcule le blueprint flop — les 1 755 classes, en parallèle, reprenable.

C'est le pilote du calcul de la Phase 2. Il ne décide rien : il applique un
palier de précision aux classes qui restent à faire, en parallèle, et écrit
chaque solution dès qu'elle est prête. Coupez-le (Ctrl+C, redémarrage,
coupure de courant) et relancez-le : le manifeste du magasin
(``pfs.solver.blueprint``) sait ce qui est fait, le travail reprend où il en
était. Un palier différent n'hérite de rien et ne détruit rien (chaque
réglage a son propre répertoire, par empreinte).

Paliers (temps d'HORLOGE mesurés le 15 août 2026 sur 32 threads logiques,
16 processus, solve flop à profondeur limitée — voir ``banc_blueprint.py``
pour les heures-cœur et ``banc_parallele.py`` pour l'accélération réelle
×6,53) ::

    palier      itérations   buckets    horloge      stockage
    eco               250        8       ≈  51 h      ~0,1 Go
    standard          500       12       ≈ 102 h      ~0,1 Go
    fin              1000       16       ≈ 205 h      ~0,1 Go

Usage ::

    python calculer_blueprint.py --palier eco          # lance (ou reprend)
    python calculer_blueprint.py --palier eco --etat   # où on en est
    python calculer_blueprint.py --palier eco --procs 8

Architecture, et pourquoi : les classes sont résolues dans des processus
fils (indépendantes par construction), mais **le parent seul écrit**. Le
magasin sérialise chaque solution atomiquement, mais son manifeste est un
fichier unique : seize processus qui le réécrivent en même temps, c'est un
manifeste corrompu et un calcul de deux jours perdu. Le coût de cette
prudence est nul — écrire 56 Ko prend moins d'une milliseconde, les solves
durent des minutes.

Ce qui est ASSUMÉ, et écrit ici plutôt que découvert plus tard :

* le solve est à **profondeur limitée** (``leaf_model="rollout"``) : le flop
  complet au combo exact est hors budget (≈ 125 000 h, contre-épreuve dans
  ``banc_blueprint.py``). Le blueprint approche donc l'équilibre, il ne le
  calcule pas exactement — c'est aussi ce que fait tout blueprint, y compris
  ceux des solveurs commerciaux, avant raffinement à la demande ;
* les ranges de référence sont un spot **CO vs BTN** unique (pot 6 bb, tapis
  97 bb) : un blueprint par contexte de range, pas un blueprint universel.
  Étendre à d'autres spots = relancer avec d'autres ranges, dans un autre
  répertoire ;
* la valeur du blueprint sera **mesurée**, pas supposée : baisse du
  désaccord postflop au banc Pluribus (36,2 % au 14 août 2026) et
  exploitabilité échantillonnée. Tant que ce n'est pas mesuré, rien n'est
  annoncé.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

PALIERS = {
    "eco": (250, 8),
    "standard": (500, 12),
    "fin": (1000, 16),
}

#: Le spot de référence du blueprint — le même que celui du dimensionnement.
POT, STACK = 6.0, 97.0
BET_FRACS = (0.5, 1.0)
MAX_BETS = 2
OOP_SPEC = "22+, ATs+, KQs, AJo+"
IP_SPEC = "55+, A9s+, KQs, AJo+"


def _resoudre(tache: tuple[tuple[int, ...], int]) -> tuple:
    """Résout UNE classe dans un processus fils ; rend les tableaux à écrire.

    Le fils ne touche jamais au magasin : il calcule et rend. C'est le
    parent qui écrit (voir l'en-tête du module).
    """
    board, iterations = tache
    from pfs.core.range_model import parse_range
    from pfs.solver.postflop import PostflopSolver

    t0 = time.perf_counter()
    oop, ip = parse_range(OOP_SPEC), parse_range(IP_SPEC)
    solveur = PostflopSolver(list(board), oop, ip, pot=POT, stack=STACK,
                             bet_fracs=BET_FRACS, max_bets=MAX_BETS,
                             leaf_model="rollout")
    solveur.solve(iterations)
    # La stratégie moyenne de la racine : ce que le blueprint sert ensuite.
    strat = solveur.average_strategy(0)
    actions = [a.label for a in solveur.root_report(top=64)]
    return board, strat, actions, time.perf_counter() - t0


def _humain(secondes: float) -> str:
    h, r = divmod(int(secondes), 3600)
    m, s = divmod(r, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--palier", choices=sorted(PALIERS), default="eco")
    ap.add_argument("--procs", type=int, default=16,
                    help="processus simultanés (16 = plafond mesuré ×6,53)")
    ap.add_argument("--etat", action="store_true",
                    help="afficher l'avancement et sortir")
    ap.add_argument("--racine", default=None,
                    help="dossier du magasin (défaut : blueprints/ du dépôt)")
    a = ap.parse_args()

    from pfs.solver.blueprint import BlueprintSettings, BlueprintStore

    iterations, buckets = PALIERS[a.palier]
    racine = a.racine or os.path.join(os.path.dirname(__file__), "blueprints")
    reglages = BlueprintSettings(iterations=iterations, n_buckets=buckets)
    magasin = BlueprintStore(racine, reglages)

    etat = magasin.progress()
    faites = int(etat.get("n_done", 0))
    total = int(etat.get("n_classes", 1755))
    print("═" * 74)
    print(f"  BLUEPRINT FLOP — palier « {a.palier} » "
          f"({iterations} itérations × {buckets} buckets)")
    print("═" * 74)
    print(f"  Magasin  : {racine}")
    print(f"  Avancement : {faites}/{total} classes "
          f"({100.0 * faites / max(total, 1):.1f} %)")
    if a.etat:
        return 0

    restantes = magasin.pending()
    if not restantes:
        print("  Rien à faire : le blueprint de ce palier est COMPLET.")
        return 0

    procs = max(1, min(a.procs, len(restantes)))
    print(f"  À faire  : {len(restantes)} classes, {procs} processus")
    print(f"  Reprise  : relancer cette commande après une interruption "
          f"reprend ici.")
    print("─" * 74, flush=True)

    taches = [(c.board, iterations) for c in restantes]
    t0 = time.perf_counter()
    n_ok, temps_solves = 0, 0.0
    try:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            futurs = {ex.submit(_resoudre, t): t for t in taches}
            for fut in as_completed(futurs):
                board, strat, actions, dt = fut.result()
                magasin.save_solution(
                    board, {"strategy": strat},
                    meta={"actions": actions, "palier": a.palier,
                          "leaf_model": "rollout", "pot": POT, "stack": STACK,
                          "oop": OOP_SPEC, "ip": IP_SPEC},
                    elapsed_s=dt)
                n_ok += 1
                temps_solves += dt
                if n_ok % 10 == 0 or n_ok == len(taches):
                    ecoule = time.perf_counter() - t0
                    reste = ecoule / n_ok * (len(taches) - n_ok)
                    print(f"  {faites + n_ok:>5}/{total}  "
                          f"écoulé {_humain(ecoule):>12}  "
                          f"reste ≈ {_humain(reste):>12}  "
                          f"({n_ok / ecoule * 3600:.0f} classes/h)",
                          flush=True)
    except KeyboardInterrupt:
        print("\n  Interrompu — les classes déjà écrites sont conservées.")
        print(f"  Relancer : python calculer_blueprint.py --palier {a.palier}")
        return 130

    ecoule = time.perf_counter() - t0
    print("─" * 74)
    print(f"  Terminé : {n_ok} classes en {_humain(ecoule)} d'horloge "
          f"({_humain(temps_solves)} de calcul cumulé, "
          f"accélération {temps_solves / max(ecoule, 1e-9):.2f}×)")
    print(f"  Blueprint : {magasin.progress().get('n_done')}/{total} classes")
    print()
    print("  Prochaine étape — MESURER ce que ça vaut, ne rien annoncer avant :")
    print("    python banc_corpus_pluribus.py    (désaccord postflop, 36,2 % avant)")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
