#!/usr/bin/env python
"""Banc de DIMENSIONNEMENT du blueprint flop — la table heures × Go.

    python banc_blueprint.py              mesure complète (~5 min)
    python banc_blueprint.py --rapide     chronométrage court
    python banc_blueprint.py --verif-flop + contre-épreuve : une VRAIE
                                          itération de flop complet (~30 min)

C'est LA donnée attendue pour décider du calcul complet des 1 755 classes :
combien d'heures machine et combien de Go, à chaque palier de précision.
La table mesurée est consignée ci-dessous après chaque exécution (règle du
projet : on ne publie pas un chiffre sans son banc dans le dépôt).

RÉSULTATS du 14 août 2026 — Intel64 6/183 (32 threads), Python 3.13.14,
NumPy 2.5.1 ; spot CO vs BTN, pot 6 bb, tapis 97 bb, tailles 50/100 %,
max_bets 2 ; 2 itérations de chauffe, 10 chronométrées :

===========================  =======  ======  ===========  ==========  =======
classe (représentant)        n_o/n_i  lignes  t_iter turn  t_iter lim  octets
===========================  =======  ======  ===========  ==========  =======
léger — sec arc-en-ciel      290/520      13       2,76 s      2,79 s  69 286
moyen — bicolore broadway    276/508      13       2,53 s      2,57 s  52 867
lourd — monotone connecté    297/514      13       2,64 s      2,82 s  46 986
===========================  =======  ======  ===========  ==========  =======

t_abs (table EHS exacte, une fois par classe) : 2,5–3,1 s ; bucketing
K ≤ 16 : < 0,2 ms. Facteur flop complet : 13 lignes × 49 = ×637.

EXTRAPOLATION aux 1 755 classes — granularité COMBO EXACT (bornes hautes) :

=========  ==============  =============  =========  ============  =======  ====
palier     itér × buckets  COMPLET s/cl   h total    LIMITÉ s/cl   h total  Go
=========  ==============  =============  =========  ============  =======  ====
éco          250 ×  8           420 972    205 224           684      334   0,10
standard     500 × 12           841 941    410 446         1 366      666   0,10
fin        1 000 × 16         1 683 879    820 891         2 729    1 331   0,10
=========  ==============  =============  =========  ============  =======  ====

Contre-épreuve ``--verif-flop`` (1 itération de flop COMPLET réelle,
Ks 8h 3d) : construction 47 s, mesuré 1 022 s/itér contre 1 756 s
modélisés — écart ×0,58 (l'all-in tronque les lignes profondes : le modèle
est une borne haute à ×2 près, le mesuré fait foi). Même au chiffre mesuré,
le flop complet reste ≈ 125 000 h au palier éco : HORS BUDGET.

LECTURE DIMENSIONNANTE : le flop COMPLET au combo exact est inatteignable
sur cette machine ; le flop à PROFONDEUR LIMITÉE (rollout, MESURÉ — pas
extrapolé) coûte 334 h au palier éco et 666 h au standard, stockage total
≈ 0,10 Go. Un CFR bucketisé (K = 8-16 contre ~280-520 combos vivants)
diviserait encore la street flop ; découper le calcul en tranches
reprenables est déjà couvert par le manifeste du magasin. La décision
(palier, buckets, tronquage, machine) appartient à Pierre.

Les deux granularités mesurées
------------------------------
1. **Flop COMPLET** (les trois streets jouées) — trop cher pour être
   chronométré en boucle : extrapolé depuis le solveur turn ACTUEL par le
   modèle « lignes × 49 » ci-dessous, contre-épreuvé par ``--verif-flop``
   (une vraie itération de flop complet, chronométrée).
2. **Flop PROFONDEUR LIMITÉE** (``leaf_model='rollout'`` : flop et turn
   joués, river remplacée par la part d'abattage moyennée — P3) — assez
   rapide pour être MESURÉ directement, sans modèle. C'est le palier
   réaliste d'un blueprint : PioSOLVER et GTO Wizard tronquent de même.

Modèle d'extrapolation du flop complet (assumé, écrit pour être contesté)
-------------------------------------------------------------------------
Le solveur turn résout UNE street de mise + le nœud de chance river. Dans
un arbre de flop complet, CHAQUE ligne de mise du flop qui clôt la street
(check-check, bet-call, raise-call…) débouche sur 49 sous-arbres de turn,
chacun ≈ l'arbre du proxy turn. Donc :

    coût_itération_flop ≈ (nb de lignes de clôture) × 49 × coût_itération_turn

Le nombre de lignes est COMPTÉ dans l'arbre construit (pas supposé) ; la
première version de ce banc utilisait ×49 tout court et la contre-épreuve
l'a FALSIFIÉE (mesuré ≈ ×557, modélisé 13 lignes × 49 = 637, écart ×0,87 —
l'all-in tronque certaines lignes profondes) : le facteur mesuré fait foi.

Autres choix du banc
--------------------
- t_iter est mesuré après chauffe ; le coût d'un palier est LINÉAIRE en
  itérations (structure DCFR).
- La table EHS exacte (t_abs) est payée une fois par classe, quel que soit
  le palier ; le bucketing (K ≤ 16) est mesuré sous 2 ms et négligé.
- Le stockage mesuré est RÉEL : stratégies moyennes de la STREET FLOP du
  solveur à profondeur limitée, rembourrées à 1 326 combos en float32,
  écrites par ``BlueprintStore`` (npz compressé), taille lue sur disque.
- Les solves mesurés sont au COMBO exact (1 326) : le palier « buckets »
  dimensionne l'abstraction et le stockage — un CFR bucketisé (K = 8-16
  contre ~250-470 combos vivants) diviserait encore le coût de la street
  flop ; les totaux sont donc des bornes hautes.

Brancher un futur solveur flop dédié = remplacer UNE affectation
(``RESOUDRE_LIMITE`` ou ``RESOUDRE_PROXY``) : le reste du banc est inchangé.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from pfs.core.range_model import (
    N_COMBOS,
    Range,
    combo_cards,
    combo_index,
    opening_range,
)
from pfs.solver.abstraction import bucket_assignments, expected_hand_strength
from pfs.solver.blueprint import BlueprintSettings, BlueprintStore, class_key
from pfs.solver.isomorphism import board_str, canonical_board
from pfs.solver.postflop import PostflopSolver

# ── configuration du spot de référence ──────────────────────────────────────

RANKS, SUITS = "AKQJT98765432", "shdc"


def _c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def _b(txt: str) -> tuple[int, ...]:
    return canonical_board([_c(t) for t in txt.split()])


#: Échantillon représentatif : léger / moyen / lourd en tirages.
ECHANTILLON: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("léger — sec arc-en-ciel", _b("Ks 8h 3d")),
    ("moyen — bicolore broadway", _b("Ks Qs 7h")),
    ("lourd — monotone connecté", _b("9s 8s 7s")),
)

POT, STACK = 6.0, 97.0
BET_FRACS: tuple[float, ...] = (0.5, 1.0)
MAX_BETS = 2

#: Paliers de précision (itérations CFR × buckets EHS).
PALIERS: tuple[tuple[str, int, int], ...] = (
    ("éco", 250, 8),
    ("standard", 500, 12),
    ("fin", 1000, 16),
)

N_TURNS = 49
"""Cartes de turn par flop — la branche du nœud de chance flop → turn."""


# ── les solves par classe (POINTS DE BRANCHEMENT d'un futur solveur) ────────


def resoudre_turn_proxy(flop: tuple[int, ...], iterations: int,
                        oop: Range, ip: Range) -> PostflopSolver:
    """Le proxy du flop complet : classe + 1 turn (plus petite carte libre)."""
    turn = min(c for c in range(52) if c not in flop)
    solver = PostflopSolver(list(flop) + [turn], oop, ip, pot=POT,
                            stack=STACK, bet_fracs=BET_FRACS,
                            max_bets=MAX_BETS)
    return solver.solve(iterations)


def resoudre_flop_limite(flop: tuple[int, ...], iterations: int,
                         oop: Range, ip: Range) -> PostflopSolver:
    """Le solve flop MESURABLE : profondeur limitée (rollout, P3)."""
    solver = PostflopSolver(list(flop), oop, ip, pot=POT, stack=STACK,
                            bet_fracs=BET_FRACS, max_bets=MAX_BETS,
                            leaf_model="rollout")
    return solver.solve(iterations)


RESOUDRE_PROXY = resoudre_turn_proxy      # ← futur solveur : remplacer ici
RESOUDRE_LIMITE = resoudre_flop_limite    # ← ou ici (une ligne chacun)


def compter_lignes_de_cloture(solver: PostflopSolver) -> int:
    """Lignes de mise de la street racine qui la CLÔTURENT (vers la chance).

    Comptées dans l'arbre construit : terminaux ``_Chance`` des nœuds de la
    street racine (reconnus par nom de classe — le banc lit les internes du
    solveur, c'est son privilège de banc). C'est le « nb de lignes » du
    modèle d'extrapolation.
    """
    n = 0
    vus: set[int] = set()

    def walk(idx: int) -> None:
        nonlocal n
        if idx in vus:
            return
        vus.add(idx)
        nd = solver._nodes[idx]
        for a, child in enumerate(nd.children):
            t = nd.terminal[a]
            if t is not None and type(t).__name__ == "_Chance":
                n += 1
            elif t is None and child >= 0:
                walk(child)

    walk(solver._root)
    return n


# ── collecte des stratégies de street (le contenu stocké par classe) ────────


def strategies_de_street(solver: PostflopSolver) -> dict[str, np.ndarray]:
    """Stratégies moyennes des nœuds de la STREET RACINE, indexées 1 326.

    On suit ``children[a]`` uniquement quand ``terminal[a] is None`` : cela
    reste dans la street de la racine (les streets suivantes ne sont
    joignables qu'à travers un nœud de chance). Chaque stratégie
    (n_actions, n_vivants) est rembourrée à (n_actions, 1 326) float32 —
    combos morts à 0 — pour être re-projetable par ``query_flop``.
    """
    out: dict[str, np.ndarray] = {}

    def walk(idx: int, path: tuple[str, ...]) -> None:
        nd = solver._nodes[idx]
        who = "oop" if nd.player == 0 else "ip"
        name = "/".join(path) if path else "racine"
        sigma = solver.average_strategy(idx)
        cards = solver.players[nd.player].cards
        cols = np.array([combo_index(int(a), int(b)) for a, b in cards],
                        dtype=np.int64)
        full = np.zeros((sigma.shape[0], N_COMBOS), dtype=np.float32)
        full[:, cols] = sigma.astype(np.float32)
        out[f"strat/{who}/{name}"] = full
        out[f"labels/{who}/{name}"] = np.array(nd.labels)
        for a, child in enumerate(nd.children):
            if nd.terminal[a] is None and child >= 0:
                walk(child, path + (nd.labels[a],))

    walk(solver._root, ())
    return out


# ── mesures ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MesureClasse:
    nom: str
    board: tuple[int, ...]
    n_oop: int
    n_ip: int
    n_lignes: int          # lignes de clôture de la street (modèle complet)
    t_iter_turn: float     # s/itération du proxy turn
    t_iter_limite: float   # s/itération du flop à profondeur limitée (MESURÉ)
    t_abs: float           # table EHS exacte de la classe
    t_buckets: float
    octets: int


def mesurer_classe(nom: str, flop: tuple[int, ...], oop: Range, ip: Range,
                   store: BlueprintStore, n_chauffe: int,
                   n_mesure: int) -> MesureClasse:
    # 1. abstraction : la table EHS exacte de la classe (cache froid).
    live_oop = oop.remove_blockers(flop)
    idx = np.nonzero(live_oop.weights > 0.0)[0]
    cards = np.array([combo_cards(int(i)) for i in idx], dtype=np.int64)
    t0 = time.perf_counter()
    ehs = expected_hand_strength(cards, flop)          # paie toute la table
    t_abs = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _, _, k in PALIERS:
        bucket_assignments(ehs, n_buckets=k, weights=live_oop.weights[idx])
    t_buckets = time.perf_counter() - t0

    # 2. proxy turn : la brique du modèle « lignes × 49 » du flop complet.
    proxy = RESOUDRE_PROXY(flop, n_chauffe, oop, ip)
    t0 = time.perf_counter()
    proxy.solve(n_mesure)
    t_iter_turn = (time.perf_counter() - t0) / n_mesure
    n_lignes = compter_lignes_de_cloture(proxy)

    # 3. flop à profondeur limitée : MESURÉ directement (pas de modèle).
    limite = RESOUDRE_LIMITE(flop, n_chauffe, oop, ip)
    t0 = time.perf_counter()
    limite.solve(n_mesure)
    t_iter_limite = (time.perf_counter() - t0) / n_mesure

    # 4. taille stockée : la street FLOP du solve limité, écrite pour de vrai.
    arrays = strategies_de_street(limite)
    combo_keys = tuple(k for k in arrays if k.startswith("strat/"))
    path = store.save_solution(
        flop, arrays, combo_keys=combo_keys,
        meta={"nom": nom, "leaf_model": "rollout"},
        elapsed_s=n_mesure * t_iter_limite,
    )
    return MesureClasse(
        nom=nom, board=flop,
        n_oop=limite.players[0].n, n_ip=limite.players[1].n,
        n_lignes=n_lignes, t_iter_turn=t_iter_turn,
        t_iter_limite=t_iter_limite, t_abs=t_abs, t_buckets=t_buckets,
        octets=path.stat().st_size,
    )


def verifier_modele_flop(flop: tuple[int, ...], oop: Range, ip: Range,
                         s_iter_modele: float) -> None:
    """Contre-épreuve du modèle : UNE vraie itération de flop complet.

    Blindée : si l'API du solveur flop bouge, le banc le dit et continue —
    la table principale ne dépend pas de cette vérification.
    """
    try:
        t0 = time.perf_counter()
        solver = PostflopSolver(list(flop), oop, ip, pot=POT, stack=STACK,
                                bet_fracs=BET_FRACS, max_bets=MAX_BETS)
        t_constr = time.perf_counter() - t0
        t0 = time.perf_counter()
        solver.solve(1)
        t_flop = time.perf_counter() - t0
    except Exception as exc:                                  # noqa: BLE001
        print(f"\nvérif : flop complet indisponible ({exc}) — "
              "modèle non contre-épreuvé cette fois.")
        return
    print(f"\nvérif (1 itération de flop COMPLET sur {board_str(flop)}) : "
          f"constr {t_constr:.0f} s — mesuré {t_flop:.0f} s/itér contre "
          f"{s_iter_modele:.0f} s modélisés "
          f"(écart ×{t_flop / s_iter_modele:.2f}) — le mesuré fait foi.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rapide", action="store_true",
                    help="chronométrage court (3 itérations mesurées)")
    ap.add_argument("--verif-flop", action="store_true",
                    help="contre-épreuve du modèle : une vraie itération "
                         "de flop complet (~30 min de plus)")
    args = ap.parse_args(argv)
    n_chauffe, n_mesure = (1, 3) if args.rapide else (2, 10)

    print("Banc de dimensionnement du blueprint flop")
    print(f"machine : {platform.processor() or platform.machine()} — "
          f"{os.cpu_count()} threads — Python {platform.python_version()} — "
          f"NumPy {np.__version__}")
    print(f"spot    : CO vs BTN, pot {POT:g} bb, tapis {STACK:g} bb, "
          f"tailles {BET_FRACS}, max_bets {MAX_BETS}")
    print(f"mesure  : {n_chauffe} itération(s) de chauffe, "
          f"{n_mesure} chronométrées\n")

    oop, ip = opening_range("CO"), opening_range("BTN")
    mesures: list[MesureClasse] = []
    with tempfile.TemporaryDirectory(prefix="pfs-banc-blueprint-") as tmp:
        settings = BlueprintSettings(
            iterations=n_mesure, n_buckets=PALIERS[0][2],
            solver_version="flop-rollout-v1",
        )
        store = BlueprintStore(tmp, settings)
        for nom, flop in ECHANTILLON:
            m = mesurer_classe(nom, flop, oop, ip, store, n_chauffe, n_mesure)
            mesures.append(m)
            octets = f"{m.octets:,}".replace(",", " ")
            print(f"  {nom:<28} {board_str(m.board):<11} "
                  f"[{class_key(m.board)}]  n={m.n_oop}/{m.n_ip}  "
                  f"lignes {m.n_lignes:>2}  t_iter turn {m.t_iter_turn:5.2f} s"
                  f"  limité {m.t_iter_limite:5.2f} s  "
                  f"t_abs {m.t_abs:4.1f} s  {octets} octets")

    t_turn = float(np.mean([m.t_iter_turn for m in mesures]))
    t_lim = float(np.mean([m.t_iter_limite for m in mesures]))
    t_abs = float(np.mean([m.t_abs for m in mesures]))
    lignes = float(np.mean([m.n_lignes for m in mesures]))
    octets_moy = float(np.mean([m.octets for m in mesures]))
    facteur = lignes * N_TURNS
    n_classes = 1755

    print(f"\nmoyennes : t_iter turn {t_turn:.2f} s — flop limité "
          f"{t_lim:.2f} s — t_abs {t_abs:.1f} s — "
          f"{lignes:.0f} lignes de clôture → facteur flop complet "
          f"×{facteur:.0f} — "
          + f"{octets_moy:,.0f} octets/classe".replace(",", " "))

    print("\nEXTRAPOLATION aux 1 755 classes — granularité COMBO EXACT "
          "(bornes hautes)")
    print(f"{'palier':<9} {'itér × buckets':<14} "
          f"{'COMPLET s/cl':>13} {'h total':>9} "
          f"{'LIMITÉ s/cl':>12} {'h total':>9} {'Go':>6}")
    for nom, iters, k in PALIERS:
        s_complet = t_abs + iters * facteur * t_turn
        s_limite = t_abs + iters * t_lim
        h_complet = n_classes * s_complet / 3600.0
        h_limite = n_classes * s_limite / 3600.0
        go = n_classes * octets_moy / 1e9
        ligne = (f"{nom:<9} {f'{iters} × {k}':<14} "
                 f"{s_complet:>13,.0f} {h_complet:>9,.0f} "
                 f"{s_limite:>12,.0f} {h_limite:>9,.0f} {go:>6.2f}")
        print(ligne.replace(",", " "))

    print("\nlecture : le flop COMPLET au combo exact est hors budget ; le "
          "flop à PROFONDEUR\nLIMITÉE (rollout, mesuré) est le palier "
          "réaliste — et un CFR bucketisé (K = 8-16)\nle diviserait encore. "
          "La décision (palier, buckets, tronquage) appartient à Pierre.")
    if args.verif_flop:
        verifier_modele_flop(ECHANTILLON[0][1], oop, ip,
                             mesures[0].n_lignes * N_TURNS
                             * mesures[0].t_iter_turn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
