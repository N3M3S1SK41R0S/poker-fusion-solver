"""Banc du solve FLOP en profondeur complète — la donnée d'entrée du blueprint.

Pourquoi ce fichier existe : « mesurer avant de publier » (trois chiffres
faux ont déjà été publiés depuis un banc jetable — le banc va dans le dépôt).
Il mesure, sur UN flop à ranges réalistes, ce que coûte le solve exact
flop→turn→river de ``pfs.solver.postflop`` : nœuds, mémoire, temps par
itération, et la courbe d'exploitabilité (par ensemble d'information) en
fonction des itérations. C'est le chiffre dont le dimensionnement du
blueprint Phase 2 (1 755 classes isomorphes × buckets) a besoin : il dit ce
qu'un solve exact coûte PAR FLOP, donc ce que l'abstraction doit amortir.

Seuils rapportés : premier passage de l'exploitabilité sous 5 %, 2 % et 1 %
du pot. Le seuil « bon pour un blueprint » retenu est **1 %** : les solveurs
commerciaux visent 0,25–0,5 % pour du GTO de référence, mais un blueprint
est de toute façon dégradé ensuite par l'abstraction en buckets — viser
mieux que 1 % au solve source serait payer de la précision que la
compression détruira.

Usage :
    python banc_flop.py            # courbe jusqu'à 80 itérations (~20 min)
    python banc_flop.py --rapide   # courbe tronquée à 20 itérations
    python banc_flop.py --iters N  # prolonger la courbe (paliers doublés)

Limites ASSUMÉES (mesurées, pas cachées) :
- l'arbre par défaut du solveur (max_bets=2) est ~4,2× plus gros que la
  configuration bancable (max_bets=1) : il est COMPTÉ ici (arbre construit
  à ranges micro — le nombre de nœuds ne dépend pas des ranges) et sa
  mémoire est PROJETÉE par la formule exacte Σ 2·8·n_actions·n_combos,
  mais sa courbe n'est pas mesurée : au rythme mesuré ce serait des heures.
- le temps par itération de max_bets=2 est EXTRAPOLÉ ∝ nœuds (annoncé tel).
- ranges pleines (~1 300 combos) : hors banc — la matrice de feuille
  rollout bascule d'ailleurs en boucle lente au-delà de 250 000 éléments
  (``_LEAF_W_MAX``), et l'arbre complet dépasserait la RAM. C'est
  précisément la raison d'être du blueprint (buckets + isomorphisme).
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import sys
import time

from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.solver.postflop import PostflopSolver

FLOP = "2s 2d 7h"                     # le flop sec de toute la suite de tests
OOP_R = "22+, ATs+, KQs, AJo+"        # défense réaliste (~126 combos ici)
IP_R = "55+, A9s+, KQs, AJo+"         # agresseur réaliste (~117 combos)
POT, STACK = 60.0, 180.0


def _cartes(texte: str) -> list[int]:
    return [RANKS.index(t[0]) * 4 + SUITS.index(t[1].lower())
            for t in texte.split()]


def _rss_mb() -> float:
    """Working set du processus (Windows), en MB — 0.0 si indisponible."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    try:
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p     # pseudo-handle −1 :
        fn = k.K32GetProcessMemoryInfo                    # sans restype/argtypes
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), wt.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)                       # il est tronqué et
        ok = fn(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)  # l'appel échoue
        return pmc.WorkingSetSize / 1e6 if ok else 0.0
    except Exception:                  # pragma: no cover — plateforme non Windows
        return 0.0


def _mem_tableaux_mb(s: PostflopSolver) -> float:
    """Mémoire EXACTE des tableaux CFR (regrets + stratégie moyenne)."""
    return sum(nd.regrets.nbytes + nd.strat_sum.nbytes
               for nd in s._nodes) / 1e6


def _mem_projetee_mb(s: PostflopSolver, n_oop: int, n_ip: int) -> float:
    """Mémoire qu'auraient les tableaux CFR de CET arbre avec d'autres ranges.

    Formule exacte (2 tableaux float64 par nœud, un par action et par combo
    de l'acteur) — permet de chiffrer un arbre sans allouer ses tableaux.
    """
    octets = 0
    for nd in s._nodes:
        n = n_oop if nd.player == 0 else n_ip
        octets += 2 * 8 * len(nd.labels) * n
    return octets / 1e6


def main(argv: list[str] | None = None) -> int:
    # un banc de 20+ minutes écrit ses points AU FIL DE L'EAU : une sortie
    # bufferisée en bloc perd toute la mesure si le processus est tué
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rapide", action="store_true",
                    help="courbe tronquée à 20 itérations")
    ap.add_argument("--iters", type=int, default=80,
                    help="palier maximal de la courbe (défaut 80)")
    a = ap.parse_args(argv)
    plafond = 20 if a.rapide else a.iters

    oop, ip = parse_range(OOP_R), parse_range(IP_R)
    board = _cartes(FLOP)
    print(f"── banc_flop — board {FLOP}, pot {POT:g}, stack {STACK:g}")
    print(f"   OOP «{OOP_R}»  IP «{IP_R}»")

    # ── A. profondeur complète, arbre bancable (1 taille, max_bets=1) ────
    t0 = time.perf_counter()
    s = PostflopSolver(board, oop, ip, pot=POT, stack=STACK,
                       bet_fracs=(0.75,), max_bets=1)
    t_build = time.perf_counter() - t0
    n_oop, n_ip = s.players[0].n, s.players[1].n
    print(f"\nA. FLOP COMPLET (bet 0.75p, max_bets=1)")
    print(f"   combos {n_oop}×{n_ip} · {len(s._nodes)} nœuds · "
          f"construction {t_build:.1f} s")
    print(f"   tableaux CFR {_mem_tableaux_mb(s):.0f} MB · "
          f"RSS {_rss_mb():.0f} MB")

    t0 = time.perf_counter()
    s.solve(2)
    t_iter = (time.perf_counter() - t0) / 2
    print(f"   {t_iter:.1f} s / itération (moyenne sur 2)")

    fait, points, seuils = 2, [], {0.05: None, 0.02: None, 0.01: None}
    palier = 10
    while fait < plafond:
        cible = min(palier, plafond)
        s.solve(cible - fait)
        fait = cible
        t0 = time.perf_counter()
        e = s.exploitability()
        t_expl = time.perf_counter() - t0
        points.append((fait, e))
        for seuil in seuils:
            if seuils[seuil] is None and e < seuil:
                seuils[seuil] = fait
        print(f"   iters {fait:>4} : exploitabilité {e * 100:6.2f} % du pot "
              f"(évaluation {t_expl:.0f} s)")
        palier *= 2
    ev0, ev1 = s.values()
    print(f"   somme des EV = {ev0 + ev1:.6f} (pot {POT:g}) — comptabilité")
    for seuil, it in sorted(seuils.items(), reverse=True):
        if it is not None:
            print(f"   seuil {seuil * 100:g} % atteint à {it} itérations "
                  f"(~{it * t_iter / 60:.1f} min)")
        else:
            print(f"   seuil {seuil * 100:g} % NON atteint en {fait} "
                  f"itérations — extrapoler prudemment, ne pas inventer")

    # ── B. l'arbre par défaut (max_bets=2), compté sans être payé ────────
    micro = PostflopSolver(board, parse_range("AA"), parse_range("KK"),
                           pot=POT, stack=STACK,
                           bet_fracs=(0.75,), max_bets=2)
    proj = _mem_projetee_mb(micro, n_oop, n_ip)
    ratio = len(micro._nodes) / len(s._nodes)
    print(f"\nB. ARBRE PAR DÉFAUT (max_bets=2) — compté, PAS résolu ici")
    print(f"   {len(micro._nodes)} nœuds (×{ratio:.1f}) · tableaux CFR "
          f"projetés {proj:.0f} MB pour {n_oop}×{n_ip} combos")
    print(f"   temps/itération EXTRAPOLÉ ∝ nœuds : ~{t_iter * ratio:.0f} s "
          f"(extrapolation, pas une mesure)")

    # ── C. profondeur limitée (rollout matriciel) — le mode de la route ──
    t0 = time.perf_counter()
    r = PostflopSolver(board, oop, ip, pot=POT, stack=STACK,
                       bet_fracs=(0.75,), max_bets=1, leaf_model="rollout")
    r.solve(2)                         # construit les matrices de feuille
    t_chauffe = time.perf_counter() - t0
    t0 = time.perf_counter()
    r.solve(20)
    t_iter_r = (time.perf_counter() - t0) / 20
    r.solve(128)
    e_r = r.exploitability()
    rv0, rv1 = r.values()
    w = sum(a.nbytes + b.nbytes for a, b in r._leaf_W.values()) / 1e6
    print(f"\nC. FLOP TRONQUÉ AU TURN (rollout, le mode de /api/postflop)")
    print(f"   {len(r._nodes)} nœuds · amorçage {t_chauffe:.1f} s "
          f"(matrices de feuille {w:.0f} MB) · {t_iter_r * 1000:.0f} ms/itération")
    print(f"   exploitabilité {e_r * 100:.2f} % du pot à 150 itérations · "
          f"somme des EV = {rv0 + rv1:.6f}")
    print(f"   ÉCART au solve complet non borné ici : la feuille ignore la "
          f"mise river (cf. tests P3 — c'est le levier que le blueprint "
          f"restituera)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
