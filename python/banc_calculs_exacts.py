#!/usr/bin/env python
"""Banc des CALCULS : les quatre noyaux donnent-ils le BON résultat ?

    python banc_calculs_exacts.py                 les quatre sections, ~1 min
    python banc_calculs_exacts.py --icm           ICM seul
    python banc_calculs_exacts.py --equite        équité seule
    python banc_calculs_exacts.py --pushfold      push/fold seul
    python banc_calculs_exacts.py --dcfr          DCFR / Kuhn seul
    python banc_calculs_exacts.py --long          + énumérations exhaustives
                                                  (~15 min : C(48,5) par main,
                                                   cellules exactes 169×169,
                                                   bascules de seuil à 10 bb)

Résultat du mode --long au 14 août 2026 (icm.py corrigé du seuil PKO et du
partage des morts) : **68 vérifications conformes, 0 écart, 831 s.** Le
premier passage de ce mode avait planté à mi-course sur un format de
ndarray dans la borne exact-vs-Monte-Carlo — corrigé dans la section 2, la
borne prend désormais la pire erreur-type et non le tableau entier.

Pourquoi ce fichier existe
--------------------------
Les tests unitaires vérifient que les noyaux **tournent** et respectent des
propriétés internes. Ils ne disent pas s'ils donnent le **bon** nombre. Un
test qui compare un module à lui-même ne prouve rien : deux gabarits de
cartes ont été intervertis pendant des semaines pendant que 1057 tests
passaient.

Ce banc confronte donc chaque noyau à une référence **extérieure au module
testé** :

===========  ==========================================================
noyau        référence indépendante
===========  ==========================================================
``icm``      énumération brute des arrangements de Harville (écrite ici,
             sans mémoïsation ni bitmask), continuité en 0 par tapis
             infinitésimal, et Monte-Carlo sur les mêmes entrées
``equity``   ``phevaluator`` (évaluateur tiers, hors dépôt) + énumération
             EXHAUSTIVE des C(48,5) = 1 712 304 tableaux, comparée aux
             valeurs publiées des matchups classiques
``pushfold`` recalcul indépendant des deux meilleures réponses et de
             l'exploitabilité (formules réécrites ici), puis équités
             exactes par énumération pour mesurer le bruit de la matrice
``dcfr``     équilibre analytique de Kuhn (valeur du jeu = −1/18, famille
             α ∈ [0, ⅓]), meilleure réponse par énumération EXHAUSTIVE
             des 2⁶ stratégies pures, et trois solveurs de référence
             (CFR vanille, CFR+, DCFR fidèle au papier) écrits ici
===========  ==========================================================

Ce que le banc ne prouve pas
----------------------------
* Il ne valide pas les *conventions* de modélisation, seulement les calculs
  qu'elles engendrent. Exemple : l'ICM fait partager à parts égales les
  dernières places entre plusieurs joueurs à zéro — c'est un choix, pas un
  théorème, et le banc se contente de vérifier qu'il conserve la dotation.
* Les valeurs « publiées » des matchups préflop dépendent des couleurs
  exactes ; le banc compare donc main *spécifique* contre main *spécifique*
  et rappelle l'ordre de grandeur attendu, pas une décimale.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.core import icm as I  # noqa: E402
from pfs.core.equity import (  # noqa: E402
    equity_multiway,
    equity_vs_range,
    evaluate7,
)
from pfs.core.range_model import (  # noqa: E402
    COMBO_TO_GROUP,
    N_COMBOS,
    N_GROUPS,
    RANKS,
    SUITS,
    Range,
    combo_cards,
    group_name,
    parse_range,
)
from pfs.solver import pushfold as PF  # noqa: E402
from pfs.solver.dcfr import (  # noqa: E402
    DCFRConfig,
    DCFRSolver,
    KuhnPoker,
    SolveResult,
    hyperparameter_schedule,
)

# ═══════════════════════════════════════════════════════════════════════════
# OUTILS D'AFFICHAGE
# ═══════════════════════════════════════════════════════════════════════════

_ETAT = {"ko": 0, "ok": 0}


def titre(txt: str) -> None:
    print(f"\n── {txt}")


def verdict(nom: str, ok: bool, detail: str) -> None:
    """Une ligne = une affirmation vérifiable, avec son chiffre."""
    _ETAT["ok" if ok else "ko"] += 1
    print(f"  [{'OK ' if ok else 'ÉCART'}] {nom:<44} {detail}")


def carte(txt: str) -> int:
    """'Ah' → carte pfs (rang*4 + couleur)."""
    return RANKS.index(txt[0]) * 4 + SUITS.index(txt[1])


def phe_str(c: int) -> str:
    """Carte pfs → notation phevaluator ('Ah' → 'Ah', couleur minuscule)."""
    return RANKS[c >> 2] + SUITS[c & 3].lower()


# ═══════════════════════════════════════════════════════════════════════════
# 1. ICM — MALMUTH-HARVILLE
# ═══════════════════════════════════════════════════════════════════════════


def harville_brut(stacks, payouts) -> list[float]:
    """Référence : la récurrence de Harville SANS aucune optimisation.

    On énumère tous les arrangements des ``m`` premières places et on
    multiplie les probabilités conditionnelles ``s_i / Σ restants``. C'est la
    définition littérale ; le module, lui, mémoïse sur un bitmask de
    sous-ensembles. Les deux doivent coïncider au bit d'arrondi près.
    """
    n = len(stacks)
    m = min(len(payouts), n)
    eq = [0.0] * n
    for perm in itertools.permutations(range(n), m):
        p = 1.0
        restants = list(range(n))
        for pos in range(m):
            tot = sum(stacks[i] for i in restants)
            if tot <= 0.0:
                p = 0.0
                break
            p *= stacks[perm[pos]] / tot
            restants.remove(perm[pos])
        if p == 0.0:
            continue
        for k in range(m):
            eq[perm[k]] += p * payouts[k]
    return eq


CAS_SOMME = (
    ("tapis égaux", [1000, 1000, 1000], [50, 30, 20]),
    ("un tapis nul", [1000, 1000, 0], [50, 30, 20]),
    ("deux tapis nuls", [1000, 0, 0], [50, 30, 20]),
    ("un seul joueur", [1000], [50]),
    ("un joueur, 3 gains", [1000], [50, 30, 20]),
    ("quatre ex aequo", [500, 500, 500, 500], [50, 30, 20]),
    ("nul hors des places payées", [500, 500, 0, 500], [50, 30, 20]),
    ("deux nuls sur cinq", [500, 0, 500, 0, 500], [50, 30, 20, 10]),
    ("plus de gains que de joueurs", [100, 200, 300], [50, 30, 20, 10, 5]),
    ("plus de joueurs que de gains", [100, 200, 300, 400, 500], [50, 30]),
    ("tous nuls sauf un", [0, 0, 700], [50, 30, 20]),
)


def section_icm(long: bool) -> None:
    print("\n" + "═" * 74)
    print("1. ICM — Malmuth-Harville, bubble factor, PKO")
    print("═" * 74)

    titre("récurrence exacte vs énumération brute des arrangements")
    rng = np.random.default_rng(7)
    pire, cas = 0.0, None
    for n in range(2, 7):
        for _ in range(30):
            st = rng.integers(1, 20000, size=n).astype(float).tolist()
            pay = sorted(rng.integers(1, 100, size=int(rng.integers(1, n + 1)))
                         .astype(float).tolist(), reverse=True)
            a = I.icm_equities(st, pay)
            b = harville_brut(st, pay)
            d = max(abs(float(a[i]) - b[i]) for i in range(n))
            if d > pire:
                pire, cas = d, (st, pay)
    verdict("150 tirages, 2 à 6 joueurs", pire < 1e-9,
            f"écart max {pire:.2e} (pire cas {cas[0]} / {cas[1]})")

    titre("Σ équités = dotation, y compris aux cas dégénérés")
    for nom, st, pay in CAS_SOMME:
        e = I.icm_equities(st, pay)
        dot = sum(pay[:len(st)])          # _validate tronque au nb de joueurs
        verdict(nom, abs(float(e.sum()) - dot) < 1e-9,
                f"Σ={float(e.sum()):.6f} dotation={dot:.2f} "
                f"→ {np.round(e, 4).tolist()}")

    titre("un joueur à 0 jeton touche le dernier gain (pas zéro)")
    e = I.icm_equities([1000, 1000, 0], [50, 30, 20])
    verdict("[1000,1000,0] / 50-30-20", abs(float(e[2]) - 20.0) < 1e-9,
            f"{np.round(e, 4).tolist()} (attendu 40 / 40 / 20)")

    titre("continuité en zéro : le tapis nul est-il la LIMITE du tapis ε ?")
    # C'est le test qui distingue une convention correcte d'un pansement :
    # donner le dernier gain à un busté doit être ce que rend Harville quand
    # son tapis tend vers 0 par valeurs positives.
    for base, pay in (([1000, 1000], [50, 30, 20]),
                      ([1000, 1000], [50, 30]),
                      ([5000, 3000, 2000], [50, 30, 20]),
                      ([5000, 3000, 2000], [50, 30, 20, 10])):
        ref = I.icm_equities(base + [0.0], pay)
        ecarts = [float(np.abs(I.icm_equities(base + [eps], pay) - ref).max())
                  for eps in (1e-3, 1e-6, 1e-9)]
        verdict(f"{base} + ε, gains {pay}",
                ecarts[-1] < 1e-8 and ecarts[0] > ecarts[-1],
                f"écarts ε=1e-3 → 1e-9 : "
                + " → ".join(f"{x:.1e}" for x in ecarts))

    titre("non-linéarité : le gros tapis vaut MOINS que sa part de jetons")
    for st in ([6000, 3000, 1000], [5000, 4000, 1000], [8000, 1000, 1000],
               [3000, 3000, 3000, 1000]):
        pay = [50, 30, 20]
        e = I.icm_equities(st, pay)
        parts = [s / sum(st) * sum(pay) for s in st]
        gros, petit = int(np.argmax(st)), int(np.argmin(st))
        verdict(str(st), e[gros] < parts[gros] and e[petit] > parts[petit],
                f"gros {e[gros]:.2f} < {parts[gros]:.2f} · "
                f"petit {e[petit]:.2f} > {parts[petit]:.2f}")

    titre("exact (≤12 joueurs) vs Monte-Carlo sur les MÊMES entrées")
    for st in ([6000, 3000, 1000], [5000, 3000, 2000, 1000],
               [4000, 3000, 2000, 1000, 500, 500]):
        pay = [50, 30, 20]
        ex = I.icm_equities(st, pay)
        probs, se = I._finish_probs_mc(tuple(float(x) for x in st),
                                       len(pay), 400_000, 12345)
        mc = probs @ np.asarray(pay, dtype=float)
        d = float(np.abs(ex - mc).max())
        # borne : 3 erreurs-types du MC, mise à l'échelle du plus gros gain.
        # se est un tableau (une erreur-type par case) : on prend la pire,
        # sinon la comparaison et le format sont ceux d'un ndarray — c'est le
        # plantage qui a interrompu le premier passage du mode --long.
        seuil = 3.0 * float(np.max(se)) * max(pay)
        verdict(str(st), d < seuil,
                f"écart max {d:.4f} $ (3·erreur-type = {seuil:.4f})")

    titre("invariances")
    e1 = I.icm_equities([6000, 3000, 1000], [50, 30, 20])
    e2 = I.icm_equities([6.0, 3.0, 1.0], [50, 30, 20])
    e3 = I.icm_equities([1000, 3000, 6000], [50, 30, 20])
    verdict("échelle des tapis (×1000)", float(np.abs(e1 - e2).max()) < 1e-12,
            f"écart {float(np.abs(e1 - e2).max()):.1e}")
    verdict("permutation des joueurs", float(np.abs(e1 - e3[::-1]).max()) < 1e-12,
            f"écart {float(np.abs(e1 - e3[::-1]).max()):.1e}")

    titre("bubble factor : les chiffres écrits dans icm.py sont-ils rejouables ?")
    payouts = [176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34]
    stacks = [35.64, 125.74, 27.41, 19.25] + [52.0] * 5
    bf = I.bubble_factor(stacks, payouts, hero=0, villain=1)
    verdict("BF héros vs chip-leader (commentaire : 2,04)",
            abs(bf - 2.04) < 0.005, f"{bf:.4f} → équité exigée {bf / (1 + bf) * 100:.1f} %"
            f" (commentaire : 67,1 %)")

    def eq_zero(st, pay):
        """Variante fautive d'avant correction : le busté touche zéro."""
        out = np.zeros(len(st))
        viv = [i for i, v in enumerate(st) if v > 0]
        if len(viv) == len(st):
            return I.icm_equities(st, pay)
        out[viv] = I.icm_equities([st[i] for i in viv], list(pay)[:len(viv)])
        return out

    def bf_zero(st, pay, h, v):
        s = [float(x) for x in st]
        eff = min(s[h], s[v])
        base = eq_zero(s, pay)[h]
        w = list(s); w[h] += eff; w[v] -= eff
        l = list(s); l[h] -= eff; l[v] += eff
        return (base - eq_zero(l, pay)[h]) / (eq_zero(w, pay)[h] - base)

    bz = bf_zero(stacks, payouts, 0, 1)
    verdict("même BF avec la convention « zéro » (commentaire : 2,42)",
            abs(bz - 2.42) < 0.005,
            f"{bz:.4f} → équité exigée {bz / (1 + bz) * 100:.1f} %"
            f" (commentaire : 70,8 %)")

    titre("cas limites du bubble factor")
    verdict("HU winner-take-all (les jetons sont linéaires)",
            abs(I.bubble_factor([1000, 1000], [100], 0, 1) - 1.0) < 1e-9,
            f"BF = {I.bubble_factor([1000, 1000], [100], 0, 1):.6f} (attendu 1)")
    verdict("HU 60/40 (toujours linéaire en HU)",
            abs(I.bubble_factor([1000, 1000], [60, 40], 0, 1) - 1.0) < 1e-9,
            f"BF = {I.bubble_factor([1000, 1000], [60, 40], 0, 1):.6f}")
    plat = I.bubble_factor([1000, 1000, 1000], [10, 10, 10], 0, 1)
    verdict("gains identiques → aucun enjeu", math.isinf(plat),
            f"BF = {plat} ; convention : équité exigée "
            f"{I.icm_required_equity(100, 75, plat):.1f} (jamais payer)")

    titre("PKO : le seuil recalculé à la main")
    spot = I.PkoSpot(stacks=(100.0, 0.0, 100.0), payouts=(100.0,),
                     bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
                     pot=100.0, bet=100.0)
    a = I.analyse_pko_spot(spot)
    ev_fold, ev_win, ev_lose = 100 / 3, 200 / 3, 0.0
    bv = 50 * (0.5 + 0.5 * 200 / 300)
    r_icm = (ev_fold - ev_lose) / (ev_win - ev_lose)
    r_pko = (ev_fold - ev_lose) / (ev_win + bv - ev_lose)
    verdict("$EV fold / gagne / perd",
            max(abs(a.ev_fold - ev_fold), abs(a.ev_win - ev_win),
                abs(a.ev_lose - ev_lose)) < 1e-9,
            f"{a.ev_fold:.4f} / {a.ev_win:.4f} / {a.ev_lose:.4f}")
    verdict("valeur de prime ½ cash + ½·P(1er)", abs(a.bounty_value - bv) < 1e-9,
            f"{a.bounty_value:.4f} (à la main {bv:.4f})")
    verdict("seuil ICM pur / avec prime",
            abs(a.required_no_bounty - r_icm) < 1e-9
            and abs(a.required_with_bounty - r_pko) < 1e-9,
            f"{a.required_no_bounty:.6f} / {a.required_with_bounty:.6f} "
            f"(à la main {r_icm:.6f} / {r_pko:.6f})")

    titre("FGS : conservation des jetons ET de la dotation")
    r = I.fgs_equities([5000, 3000, 500, 1500], [50, 30, 20],
                       button=1, sb=100, bb=200, n_hands=4)
    d_eq = float(np.abs(r.equities.sum(axis=1) - 100.0).max())
    d_st = float(np.abs(r.stacks_path.sum(axis=1) - 10000.0).max())
    verdict("horizon 4 mains", d_eq < 1e-9 and d_st < 1e-9,
            f"dotation ±{d_eq:.1e} · jetons ±{d_st:.1e} · "
            f"le short perd {r.deltas[1:, 2].min():+.2f} $ de $EV")

    titre("LIMITE CONNUE — non testable, à savoir")
    print("      · plusieurs joueurs à zéro se partagent les dernières places à")
    print("        parts ÉGALES : rien dans les données ne dit qui a sauté en")
    print("        premier. La limite ε le confirme quand les ε sont égaux, et")
    print("        la contredit quand ils diffèrent (ε / 2ε → 13,33 / 16,67 au")
    print("        lieu de 15 / 15). C'est une convention, pas un théorème.")
    print("      · `_finish_probs_exact` appelée DIRECTEMENT avec un tapis nul")
    print("        rend une colonne de places entièrement nulle (somme 0 au lieu")
    print("        de 1). `icm_equities` filtre les zéros en amont, donc le cas")
    print("        n'est pas atteignable par l'API publique.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ÉQUITÉ — ÉVALUATEUR ET ÉNUMÉRATION
# ═══════════════════════════════════════════════════════════════════════════

# Valeurs publiées (tout calculateur d'équité standard). La tolérance est de
# ±0,8 pt parce que ces valeurs sont USUELLEMENT données pour la MAIN, sans
# préciser les couleurs : deux configurations du même matchup s'écartent de
# 1,4 pt (AA vs KK : 81,26 % couleurs disjointes, 82,64 % couleurs partagées).
# Ce banc énumère donc une main SPÉCIFIQUE et confirme le nombre obtenu avec
# un second évaluateur (phevaluator) sur un sous-échantillon déterministe.
TOL_MATCHUP = 0.8

MATCHUPS = (
    (("Ah", "Ad"), ("Ks", "Kc"), 81.9, "AA vs KK, couleurs disjointes"),
    (("As", "Ah"), ("Ks", "Kh"), 81.9, "AA vs KK, couleurs partagées"),
    (("As", "Ks"), ("2h", "2d"), 50.0, "AKs vs 22 — le pile ou face célèbre"),
    (("Ah", "Kd"), ("2h", "2d"), 47.5, "AKo vs 22"),
    (("As", "Ks"), ("Qh", "Qd"), 46.0, "AKs vs QQ"),
    (("Ah", "Kd"), ("Qh", "Qd"), 43.3, "AKo vs QQ"),
    (("Js", "Ts"), ("Ah", "Ad"), 21.7, "JTs vs AA"),
    (("Ah", "Kd"), ("Qs", "Jc"), 64.5, "AKo vs QJo"),
    (("8h", "8d"), ("As", "Kh"), 55.2, "88 vs AKo — paire vs deux surcartes"),
)


def _phe():
    try:
        from phevaluator import evaluate_cards
    except ImportError:                                   # pragma: no cover
        return None
    return evaluate_cards


def section_equite(long: bool) -> None:
    print("\n" + "═" * 74)
    print("2. ÉQUITÉ — évaluateur 7 cartes et énumération")
    print("═" * 74)

    ev = _phe()
    titre("evaluate7 confronté à phevaluator (évaluateur tiers)")
    if ev is None:
        print("      phevaluator absent : section sautée "
              "(pip install phevaluator)")
    else:
        rng = np.random.default_rng(2026)
        n = 60_000
        mains = np.array([rng.permutation(52)[:7] for _ in range(n)])
        mien = evaluate7(mains).astype(np.int64)
        # phevaluator : plus PETIT = plus fort ; pfs : plus GRAND = plus fort
        ref = np.array([ev(*[RANKS[int(c) >> 2] + SUITS[int(c) & 3].lower()
                             for c in m]) for m in mains])
        i = rng.integers(0, n, size=400_000)
        j = rng.integers(0, n, size=400_000)
        d = int((np.sign(mien[i] - mien[j]) != np.sign(ref[j] - ref[i])).sum())
        verdict("mains uniformes : ordre sur 400 000 paires", d == 0,
                f"{d} désaccord(s) sur 400 000 "
                f"({(1 - d / 400_000) * 100:.4f} % d'accord)")

        # Un tirage uniforme de 7 cartes ne contient presque aucune quinte
        # flush : l'accord sur des mains ordinaires ne dit RIEN des catégories
        # hautes, qui sont précisément celles où un évaluateur se trompe. On
        # biaise donc le tirage vers elles.
        durs = []
        deck2 = [r * 4 + s for r in range(13) for s in (0, 1)]   # 2 couleurs
        for _ in range(20_000):
            durs.append(list(rng.permutation(deck2)[:7]))
        for _ in range(20_000):                                  # 7 rangs suivis
            base = int(rng.integers(0, 7))
            dk = [r * 4 + s for r in range(base, base + 7) for s in range(4)]
            durs.append(list(rng.permutation(dk)[:7]))
        for _ in range(20_000):                                  # 4 rangs
            rr = rng.permutation(13)[:4]
            dk = [int(r) * 4 + s for r in rr for s in range(4)]
            durs.append(list(rng.permutation(dk)[:7]))
        H = np.array(durs, dtype=np.int64)
        m2 = evaluate7(H).astype(np.int64)
        r2 = np.array([ev(*[phe_str(int(c)) for c in m]) for m in H])
        i = rng.integers(0, len(H), size=1_000_000)
        j = rng.integers(0, len(H), size=1_000_000)
        d2 = int((np.sign(m2[i] - m2[j]) != np.sign(r2[j] - r2[i])).sum())
        cats = {c: int(((m2 >> 20) == k).sum())
                for k, c in enumerate(("hauteur", "paire", "deux paires",
                                       "brelan", "suite", "couleur", "full",
                                       "carré", "quinte flush"))}
        verdict("mains à catégories hautes : 1 000 000 paires", d2 == 0,
                f"{d2} désaccord(s) · couleurs {cats['couleur']}, "
                f"suites {cats['suite']}, fulls {cats['full']}, "
                f"carrés {cats['carré']}, quintes flush {cats['quinte flush']}")

    titre("équités préflop EXACTES vs valeurs publiées")
    if not long:
        print("      énumération C(48,5) : réservée à --long "
              "(~7 s par matchup)")
    else:
        for h1, h2, attendu, nom in MATCHUPS:
            a = [carte(x) for x in h1]
            b = [carte(x) for x in h2]
            reste = [c for c in range(52) if c not in set(a) | set(b)]
            boards = np.array(list(itertools.combinations(reste, 5)),
                              dtype=np.int64)
            nb = boards.shape[0]
            s1 = evaluate7(np.hstack([np.tile(a, (nb, 1)), boards])
                           ).astype(np.int64)
            s2 = evaluate7(np.hstack([np.tile(b, (nb, 1)), boards])
                           ).astype(np.int64)
            e = (float((s1 > s2).sum()) + 0.5 * float((s1 == s2).sum())) / nb
            # contre-épreuve : le MÊME sous-échantillon de tableaux, arbitré
            # par phevaluator. C'est ce qui distingue « le nombre est stable »
            # de « le nombre est juste ».
            croise = ""
            if ev is not None:
                sub = boards[::37]
                p1 = [phe_str(x) for x in a]
                p2 = [phe_str(x) for x in b]
                gag = eg = 0
                for bd in sub:
                    cb = [phe_str(int(x)) for x in bd]
                    r1, r2 = ev(*p1, *cb), ev(*p2, *cb)
                    gag += r1 < r2
                    eg += r1 == r2
                e_ref = (gag + 0.5 * eg) / len(sub)
                m = (float((s1[::37] > s2[::37]).sum())
                     + 0.5 * float((s1[::37] == s2[::37]).sum())) / len(sub)
                croise = f" · phevaluator {abs(e_ref - m) * 100:.0e} pt d'écart"
            verdict(nom, abs(e * 100 - attendu) < TOL_MATCHUP,
                    f"{e * 100:7.4f} %  (publié ≈ {attendu} %, "
                    f"{nb} tableaux){croise}")

    titre("équité contre une RANGE : exact sur board connu")
    r = equity_vs_range([carte("Ah"), carte("Ad")], parse_range("KK"),
                        [carte(x) for x in ("2s", "7d", "9c", "Jh", "3s")])
    verdict("AA vs KK, river sans amélioration", r.exact and abs(r.equity - 1.0) < 1e-12,
            f"équité {r.equity:.6f} (exacte, {r.n_scenarios} scénarios)")

    titre("équité MULTIWAY : la part espérée du pot, arbitrée par phevaluator")
    board = ["Qs", "Jc", "9h", "4d", "2s"]
    hero = ["Qh", "Qd"]
    r1, r2 = parse_range("TT+,AJs,KJs"), parse_range("KQs,QJs,JTs,AQo,T8s")
    res = equity_multiway([carte(x) for x in hero], [r1, r2],
                          [carte(x) for x in board])
    if ev is None:
        print("      phevaluator absent : contre-épreuve sautée")
    else:
        morts = {carte(x) for x in hero} | {carte(x) for x in board}

        def combos(rg):
            return [(combo_cards(k), float(rg.weights[k]))
                    for k in range(N_COMBOS)
                    if rg.weights[k] > 0 and combo_cards(k)[0] not in morts
                    and combo_cards(k)[1] not in morts]

        bp = [phe_str(carte(x)) for x in board]
        sh = ev(*[phe_str(carte(x)) for x in hero], *bp)
        num = den = 0.0
        for (a1, w1) in combos(r1):
            s1 = ev(phe_str(a1[0]), phe_str(a1[1]), *bp)
            for (a2, w2) in combos(r2):
                if set(a1) & set(a2):
                    continue
                s2 = ev(phe_str(a2[0]), phe_str(a2[1]), *bp)
                meilleur = min(s1, s2)      # phevaluator : plus petit = plus fort
                part = (1.0 if sh < meilleur else 0.0 if sh > meilleur
                        else 1.0 / (1 + (s1 == sh) + (s2 == sh)))
                num += w1 * w2 * part
                den += w1 * w2
        verdict("QQ vs 2 ranges, river énumérée",
                abs(res.equity - num / den) < 1e-12,
                f"pfs {res.equity:.9f} · phevaluator {num / den:.9f} "
                f"(écart {abs(res.equity - num / den):.1e})")
    chop = equity_multiway([carte("2h"), carte("3d")],
                           [parse_range("44"), parse_range("55")],
                           [carte(x) for x in ("As", "Ks", "Qs", "Js", "Ts")])
    verdict("quinte flush royale au tableau → part 1/3",
            abs(chop.equity - 1 / 3) < 1e-12, f"{chop.equity:.9f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PUSH/FOLD — CERTIFICAT DE NASH ET BRUIT DE LA MATRICE
# ═══════════════════════════════════════════════════════════════════════════


def _ev_jam_independant(call, eq, cnt, u_sb):
    """EV(jam) par groupe — formule réécrite ici, boucle explicite.

    Le module la calcule en algèbre matricielle NumPy ; la réécrire en
    boucles est la seule façon de détecter une transposition ou un axe
    inversé, qui ne se voient pas dans un test de forme.
    """
    _, u_nc, u_win, u_lose = u_sb
    out = np.zeros(N_GROUPS)
    for i in range(N_GROUPS):
        acc, tot = 0.0, cnt[i].sum()
        for j in range(N_GROUPS):
            c = cnt[i, j]
            if c == 0:
                continue
            e = eq[i, j]
            acc += c * (call[j] * (e * u_win + (1 - e) * u_lose)
                        + (1 - call[j]) * u_nc)
        out[i] = acc / tot
    return out


def _ev_call_independant(jam, eq, cnt, u_bb):
    """EV(call) par groupe du défenseur — croyance bayésienne réécrite."""
    _, u_win, u_lose = u_bb
    out = np.full(N_GROUPS, -np.inf)
    for j in range(N_GROUPS):
        num = den = 0.0
        for i in range(N_GROUPS):
            w = cnt[j, i] * jam[i]
            if w <= 0:
                continue
            num += w * eq[j, i]
            den += w
        if den > 0:
            e = num / den
            out[j] = e * u_win + (1 - e) * u_lose
    return out


def section_pushfold(long: bool) -> None:
    print("\n" + "═" * 74)
    print("3. PUSH/FOLD — équilibre de Nash jam/fold heads-up")
    print("═" * 74)

    eq = PF.equity_matrix_169()
    cnt = PF._unblocked_counts()

    titre("poids de card-removal recalculés en force brute")
    cartes = [combo_cards(k) for k in range(N_COMBOS)]
    grp = [int(COMBO_TO_GROUP[k]) for k in range(N_COMBOS)]
    pire = 0
    for g in (0, 1, 13, 14, 28, 84, 167):
        a, b = PF._rep_combo(g)
        ref = [0] * N_GROUPS
        for k in range(N_COMBOS):
            x, y = cartes[k]
            if x in (a, b) or y in (a, b):
                continue
            ref[grp[k]] += 1
        pire = max(pire, max(abs(ref[h] - cnt[g, h]) for h in range(N_GROUPS)))
    verdict("7 groupes témoins", pire == 0,
            f"écart max {pire} · combos compatibles avec AA = "
            f"{int(cnt[0].sum())} (attendu 1326−2×50+1 = 1225)")

    titre("certificat : le profil rendu est-il un équilibre ?")
    print(f"      {'cas':>16}{'jam':>9}{'call':>9}{'it':>5}"
          f"{'BR jam ≠':>10}{'BR call ≠':>11}{'exploitabilité':>16}{'/enjeu':>10}")
    # chipEV, puis deux spots ICM (bulle 4 joueurs / 3 payés) : l'ICM rend le
    # jeu NON à somme nulle, donc la convergence du fictitious play n'est plus
    # garantie par Robinson — d'où l'intérêt de certifier chaque cas.
    cas = [(f"{s:.0f} bb chipEV", s, None, None, 0, 1)
           for s in (2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)]
    cas += [("10 bb bulle ICM", 10.0, (50.0, 30.0, 20.0),
             (30.0, 10.0, 10.0, 10.0), 1, 0),
            ("14 bb bulle ICM", 14.0, (50.0, 30.0, 20.0),
             (40.0, 14.0, 10.0, 10.0), 1, 0)]
    for nom, stack, payouts, stacks, hero, vil in cas:
        if payouts is None:
            sol = PF.solve_hu_pushfold(stack, equity=eq)
            sv, h, v = np.array([stack, stack]), 0, 1
        else:
            sol = PF.solve_hu_pushfold(stack, payouts=payouts, stacks=stacks,
                                       hero=hero, villain=vil, equity=eq)
            sv, h, v = np.asarray(stacks, dtype=float), hero, vil
        st = PF._terminal_stacks(sv, h, v, 0.5, 1.0, 0.0)
        u_sb, u_bb = PF._terminal_values(st, payouts, h, v)
        ej = _ev_jam_independant(sol.call_range, eq, cnt, u_sb)
        ec = _ev_call_independant(sol.jam_range, eq, cnt, u_bb)
        br_j = (ej > u_sb[0] + 1e-12 * (1 + abs(u_sb[0]))).astype(float)
        br_c = (ec > u_bb[0] + 1e-12 * (1 + abs(u_bb[0]))).astype(float)
        dj = int(np.abs(br_j - sol.jam_range).sum())
        dc = int(np.abs(br_c - sol.call_range).sum())
        prior = np.array([np.count_nonzero(COMBO_TO_GROUP == g)
                          for g in range(N_GROUPS)], dtype=float) / N_COMBOS
        cur = sol.jam_range * ej + (1 - sol.jam_range) * u_sb[0]
        g_sb = float((prior * (np.maximum(ej, u_sb[0]) - cur)).sum())
        tot = (cnt * sol.jam_range[None, :]).sum(axis=1)
        p_jam = tot / cnt.sum(axis=1)
        ecf = np.where(np.isfinite(ec), ec, u_bb[0])
        cur_b = sol.call_range * ecf + (1 - sol.call_range) * u_bb[0]
        g_bb = float((prior * p_jam * (np.maximum(ecf, u_bb[0]) - cur_b)).sum())
        enjeu = 0.5 * (abs(u_sb[2] - u_sb[3]) + abs(u_bb[1] - u_bb[2]))
        eps = g_sb + g_bb
        print(f"      {nom:>16}{sol.jam_pct * 100:>8.2f}%"
              f"{sol.call_pct * 100:>8.2f}%{sol.n_iter:>5}{dj:>10}{dc:>11}"
              f"{eps:>16.3e}{eps / enjeu:>10.1e}")
    print("      (BR ≠ = nombre de groupes où la meilleure réponse recalculée")
    print("       diffère du profil rendu ; 0 = équilibre pur EXACT du jeu")
    print("       discrétisé. Une valeur non nulle avec ε/enjeu ≈ 1e-4 est un")
    print("       ε-Nash, pas une erreur.)")

    titre("ordres de grandeur publiés (chart Nash HU jam/fold)")
    s10 = PF.solve_hu_pushfold(10.0, equity=eq)
    s15 = PF.solve_hu_pushfold(15.0, equity=eq)
    verdict("10 bb : jam ≈ 57-59 %, call ≈ 37-40 %",
            0.55 <= s10.jam_pct <= 0.61 and 0.35 <= s10.call_pct <= 0.41,
            f"jam {s10.jam_pct * 100:.2f} % · call {s10.call_pct * 100:.2f} %")
    verdict("15 bb : jam ≈ 43-46 %, call ≈ 27-30 %",
            0.42 <= s15.jam_pct <= 0.47 and 0.26 <= s15.call_pct <= 0.31,
            f"jam {s15.jam_pct * 100:.2f} % · call {s15.call_pct * 100:.2f} %")

    titre("bruit de la matrice 169×169 : cellules exactes par énumération")
    if not long:
        print("      énumération C(48,5) par cellule : réservée à --long "
              "(~20 s par cellule)")
    else:
        rng = np.random.default_rng(31)
        cibles = [(0, 14), (1, 28), (13, 28), (94, 3), (95, 3)]
        cibles += [(int(rng.integers(0, N_GROUPS)), int(rng.integers(0, N_GROUPS)))
                   for _ in range(7)]
        ecarts = []
        print(f"      {'i':>6}{'j':>6}{'matrice':>10}{'exact':>10}{'écart':>9}")
        for i, j in cibles:
            if i == j:
                continue
            a, b = PF._rep_combo(i)
            reste = [c for c in range(52) if c not in (a, b)]
            boards = np.array(list(itertools.combinations(reste, 5)),
                              dtype=np.int64)
            hs = evaluate7(np.hstack([np.tile([a, b], (boards.shape[0], 1)),
                                      boards])).astype(np.int64)
            combos = [combo_cards(k) for k in range(N_COMBOS)
                      if COMBO_TO_GROUP[k] == j and a not in combo_cards(k)
                      and b not in combo_cards(k)]
            num, den = 0.0, 0
            for (x, y) in combos:
                keep = ~((boards == x).any(axis=1) | (boards == y).any(axis=1))
                bd = boards[keep]
                vs = evaluate7(np.hstack([np.tile([x, y], (bd.shape[0], 1)), bd])
                               ).astype(np.int64)
                h = hs[keep]
                num += float((h > vs).sum()) + 0.5 * float((h == vs).sum())
                den += bd.shape[0]
            e = num / den
            ecarts.append((eq[i, j] - e) * 100)
            print(f"      {group_name(i):>6}{group_name(j):>6}"
                  f"{eq[i, j] * 100:>9.3f}%{e * 100:>9.3f}%{ecarts[-1]:>+9.3f}")
        ec = np.array(ecarts)
        verdict("écart matrice ↔ exact",
                float(np.abs(ec).mean()) < 1.5,
                f"|moyen| {np.abs(ec).mean():.3f} pt · max "
                f"{np.abs(ec).max():.3f} pt · écart-type {ec.std(ddof=1):.3f} pt"
                f"  (annoncé ±0,91 pt d'erreur-type)")

    titre("ce bruit fait-il BASCULER une décision ? (10 bb)")
    if not long:
        print("      Monte-Carlo 4 000 000 par groupe : réservé à --long "
              "(~15 min)")
    else:
        stack = 10.0
        sol = PF.solve_hu_pushfold(stack, equity=eq)
        st = PF._terminal_stacks(np.array([stack, stack]), 0, 1, 0.5, 1.0, 0.0)
        u_sb, _ = PF._terminal_values(st, None, 0, 1)
        u_fold, u_nc, u_win, u_lose = u_sb
        masque = np.array([sol.call_range[int(COMBO_TO_GROUP[k])]
                           for k in range(N_COMBOS)])
        call_range = Range(masque)
        marginaux = [int(g) for g in np.argsort(np.abs(sol.ev_jam_par_groupe))[:18]]
        bascules = []
        print(f"      {'main':>6}{'EV matrice':>12}{'e matrice':>11}"
              f"{'e MC 4M':>10}{'écart pt':>10}{'EV recalculé':>14}")
        for g in marginaux:
            hero = list(PF._rep_combo(g))
            acc = sum(equity_vs_range(hero, call_range, (), n_sims=200_000,
                                      seed=1000 + k).equity for k in range(20))
            e_mc = acc / 20
            w = cnt[g] * sol.call_range
            e_mat = float((w * eq[g]).sum() / w.sum())
            p_call = float(w.sum() / cnt[g].sum())
            ev_new = (p_call * (e_mc * u_win + (1 - e_mc) * u_lose)
                      + (1 - p_call) * u_nc - u_fold)
            if (ev_new > 0) != (sol.ev_jam_par_groupe[g] > 0):
                bascules.append((group_name(g), sol.ev_jam_par_groupe[g], ev_new))
            print(f"      {group_name(g):>6}{sol.ev_jam_par_groupe[g]:>+12.5f}"
                  f"{e_mat * 100:>10.3f}%{e_mc * 100:>9.3f}%"
                  f"{(e_mat - e_mc) * 100:>+10.3f}{ev_new:>+14.5f}")
        verdict("groupes marginaux dont la décision bascule",
                len(bascules) <= 2,
                f"{len(bascules)} / 18 " + (f"→ {bascules}" if bascules else ""))


# ═══════════════════════════════════════════════════════════════════════════
# 4. DCFR — KUHN POKER
# ═══════════════════════════════════════════════════════════════════════════

DEALS = [(a, b) for a in range(3) for b in range(3) if a != b]
TERM = {"pp", "bp", "bb", "pbp", "pbb"}
KEYS0 = [f"{c}|{h}" for h in ("", "pb") for c in (0, 1, 2)]
KEYS1 = [f"{c}|{h}" for h in ("p", "b") for c in (0, 1, 2)]


def _util(c0: int, c1: int, h: str) -> float:
    win = 1 if c0 > c1 else -1
    if h == "pp":
        return float(win)
    if h == "bp":
        return 1.0
    if h == "pbp":
        return -1.0
    return float(2 * win)


def equilibre_kuhn(alpha: float) -> dict:
    """Équilibre de Nash ANALYTIQUE de Kuhn, famille α ∈ [0, 1/3].

    Joueur 0 mise le valet avec probabilité α et le roi avec 3α ; joueur 1
    paie la dame avec ⅓ et mise le valet après check avec ⅓ ; joueur 0 paie
    la dame avec α + ⅓. Valeur du jeu : −1/18 pour tout α.
    """
    return {
        "0|": np.array([1 - alpha, alpha]),
        "1|": np.array([1.0, 0.0]),
        "2|": np.array([1 - 3 * alpha, 3 * alpha]),
        "0|p": np.array([2 / 3, 1 / 3]),
        "1|p": np.array([1.0, 0.0]),
        "2|p": np.array([0.0, 1.0]),
        "0|b": np.array([1.0, 0.0]),
        "1|b": np.array([2 / 3, 1 / 3]),
        "2|b": np.array([0.0, 1.0]),
        "0|pb": np.array([1.0, 0.0]),
        "1|pb": np.array([1 - (alpha + 1 / 3), alpha + 1 / 3]),
        "2|pb": np.array([0.0, 1.0]),
    }


def br_exhaustive(strategie: dict, br: int) -> float:
    """Meilleure réponse par ÉNUMÉRATION des 2⁶ = 64 stratégies pures.

    C'est l'oracle du calcul d'exploitabilité : aucune récursion savante,
    aucun ordre de traitement des ensembles d'information — on essaie tout.
    """
    def rec(cards, h, p, pur):
        if h in TERM:
            u = _util(cards[0], cards[1], h)
            return (u if br == 0 else -u) * p
        pl = len(h) % 2
        key = f"{cards[pl]}|{h}"
        if pl == br:
            return rec(cards, h + pur[key], p, pur)
        return sum(rec(cards, h + a, p * float(q), pur)
                   for a, q in zip("pb", strategie[key]) if q > 0)

    keys = KEYS0 if br == 0 else KEYS1
    best = -math.inf
    for combo in itertools.product("pb", repeat=len(keys)):
        pur = dict(zip(keys, combo))
        v = sum(rec(c, "", 1 / 6, pur) for c in DEALS)
        best = max(best, v)
    return best


def _rm(R):
    p = np.maximum(R, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.full(2, 0.5)


def solveur_reference(mode: str, T: int) -> dict:
    """CFR vanille / CFR+ / DCFR fidèle au papier — écrits ici, hors dépôt.

    DCFR (Brown & Sandholm 2019) escompte l'ACCUMULATEUR de stratégie
    moyenne : ``S ← S·((t−1)/t)^γ + π σ``. Le poids relatif de l'itération t
    vaut alors (t/T)^γ. C'est cette variante que ``mode="dcfr"`` implémente.
    """
    R, S = {}, {}

    def walk(cards, h, p0, p1, updating, w):
        if h in TERM:
            return _util(cards[0], cards[1], h)
        pl = len(h) % 2
        key = f"{cards[pl]}|{h}"
        if key not in R:
            R[key] = np.zeros(2)
            S[key] = np.zeros(2)
        sig = _rm(R[key])
        u = np.zeros(2)
        for i, a in enumerate("pb"):
            if pl == 0:
                u[i] = walk(cards, h + a, p0 * sig[i], p1, updating, w)
            else:
                u[i] = walk(cards, h + a, p0, p1 * sig[i], updating, w)
        nu = float(sig @ u)
        if updating == pl:
            sign = 1.0 if pl == 0 else -1.0
            cf = p1 if pl == 0 else p0
            R[key] += cf * sign * (u - nu)
            if mode == "cfr+":
                np.maximum(R[key], 0.0, out=R[key])
            S[key] += w * (p0 if pl == 0 else p1) * sig
        return nu

    for t in range(1, T + 1):
        if mode == "dcfr" and t > 1:
            f = ((t - 1.0) / t) ** 2.0
            for s in S.values():
                s *= f
        w = float(t) if mode == "cfr+" else 1.0
        for p in (0, 1):
            for c in DEALS:
                walk(c, "", 1 / 6, 1 / 6, p, w)
        if mode == "dcfr":
            pw = t ** 1.5 / (t ** 1.5 + 1.0)
            for r in R.values():
                np.multiply(r, np.where(r > 0, pw, 0.5), out=r)
    return {k: (S[k] / S[k].sum() if S[k].sum() > 0 else np.full(2, 0.5))
            for k in S}


class _DCFRGammaCorrige(DCFRSolver):
    """Le solveur du dépôt, avec le seul γ remis dans l'ordre du papier."""

    def solve(self, iterations: int = 5000, track_every: int = 0) -> SolveResult:
        g, starts, hist = self.game, self.game.initial_states(), []
        for t in range(1, iterations + 1):
            if self.cfg.use_schedule:
                alpha, beta, gamma = hyperparameter_schedule(t, iterations)
            else:
                alpha, beta, gamma = self.cfg.alpha, self.cfg.beta, self.cfg.gamma
            if t > 1:                       # ← escompte de l'ACCUMULATEUR
                f = ((t - 1.0) / t) ** gamma
                for s in self._strategy_sum.values():
                    s *= f
            for p in ((0, 1) if self.cfg.alternating else (None,)):
                for state, prob in starts:
                    self._walk(state, [prob, prob], p, 1.0)
            pos = (t ** alpha) / (t ** alpha + 1.0)
            neg = (t ** beta) / (t ** beta + 1.0) if beta != 0.0 else 0.5
            for r in self._regrets.values():
                np.multiply(r, np.where(r > 0.0, pos, neg), out=r)
            if track_every and (t % track_every == 0 or t == iterations):
                hist.append((t, self.exploitability()))
        avg = self.average_strategy()
        return SolveResult(avg, dict(self._actions), iterations,
                           self.exploitability(), self.game_value(avg), hist)


def section_dcfr(long: bool) -> None:
    print("\n" + "═" * 74)
    print("4. DCFR — Kuhn poker, dont l'équilibre est connu analytiquement")
    print("═" * 74)

    jeu = KuhnPoker()
    oracle = DCFRSolver(jeu, DCFRConfig())

    titre("la MESURE elle-même est-elle juste ? (avant de mesurer le solveur)")
    for a in (0.0, 1 / 12, 1 / 6, 1 / 4, 1 / 3):
        s = equilibre_kuhn(a)
        v, e = oracle.game_value(s), oracle.exploitability(s)
        verdict(f"équilibre analytique α = {a:.4f}",
                abs(v + 1 / 18) < 1e-12 and abs(e) < 1e-12,
                f"valeur {v:+.12f} (−1/18 = {-1 / 18:+.12f}) · "
                f"exploitabilité {e:.2e}")

    titre("meilleure réponse du dépôt vs énumération des 2⁶ stratégies pures")
    for T in (50, 500, 5000):
        s = DCFRSolver(jeu, DCFRConfig())
        r = s.solve(iterations=T)
        d0 = s.best_response_value(r.strategy, 0) - br_exhaustive(r.strategy, 0)
        d1 = s.best_response_value(r.strategy, 1) - br_exhaustive(r.strategy, 1)
        verdict(f"stratégie moyenne à {T} itérations",
                abs(d0) < 1e-12 and abs(d1) < 1e-12,
                f"écart BR₀ {d0:+.1e} · BR₁ {d1:+.1e}")

    titre("l'exploitabilité décroît-elle et tend-elle vers 0 ?")
    s = DCFRSolver(jeu, DCFRConfig())
    r = s.solve(iterations=2000, track_every=50)
    remontees = sum(1 for k in range(1, len(r.history))
                    if r.history[k][1] > r.history[k - 1][1] + 1e-15)
    verdict("monotonie sur 40 relevés", remontees == 0,
            f"{remontees} remontée(s) · {r.history[0][1]:.3e} → "
            f"{r.history[-1][1]:.3e}")

    print(f"\n      {'it':>7}{'dépôt':>13}{'γ corrigé':>13}{'CFR vanille':>13}"
          f"{'CFR+':>11}{'DCFR papier':>13}")
    iters = (100, 500, 1000, 5000) + ((20000,) if long else ())
    pts = []
    for T in iters:
        a = DCFRSolver(jeu, DCFRConfig()).solve(iterations=T).exploitability
        b = _DCFRGammaCorrige(jeu, DCFRConfig()).solve(iterations=T).exploitability
        c = oracle.exploitability(solveur_reference("vanilla", T))
        d = oracle.exploitability(solveur_reference("cfr+", T))
        e = oracle.exploitability(solveur_reference("dcfr", T))
        pts.append((T, a))
        print(f"      {T:>7}{a:>13.3e}{b:>13.3e}{c:>13.3e}{d:>11.3e}{e:>13.3e}")
    xs = [math.log(t) for t, _ in pts]
    ys = [math.log(v) for _, v in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    pente = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    verdict("pente log-log (ε ≈ T^p)", pente < -0.4,
            f"p = {pente:.3f}  (CFR vanille : −0,5)")

    titre("stratégie moyenne vs équilibre analytique")
    r = DCFRSolver(jeu, DCFRConfig()).solve(iterations=20000 if long else 5000)
    def f(k, a):
        return dict(zip(r.actions[k], r.strategy[k])).get(a, float("nan"))
    aj, ak = f("0|", "b"), f("2|", "b")
    verdict("mise du roi = 3 × mise du valet", abs(ak - 3 * aj) < 0.03,
            f"valet {aj:.4f} · roi {ak:.4f} · 3×valet {3 * aj:.4f} "
            f"(écart {abs(ak - 3 * aj):.1e})")
    verdict("la dame ne mise jamais au premier tour", abs(f("1|", "b")) < 0.01,
            f"{f('1|', 'b'):.4f}")
    verdict("le défenseur paie la dame à ⅓", abs(f("1|b", "b") - 1 / 3) < 0.02,
            f"{f('1|b', 'b'):.4f} (⅓ = 0,3333)")
    verdict("le défenseur mise le valet à ⅓ après check",
            abs(f("0|p", "b") - 1 / 3) < 0.02, f"{f('0|p', 'b'):.4f}")

    titre("LIMITE MESURÉE — non testable ailleurs, à savoir")
    print("      Le γ de DCFR est appliqué à la CONTRIBUTION neuve")
    print("      (strat_sum += (t/(t+1))^γ · π σ) et non à l'ACCUMULATEUR,")
    print("      comme le veut Brown & Sandholm (2019). Le poids relatif de")
    print("      l'itération t devient e^{−γ/t} → 1 (moyenne quasi uniforme)")
    print("      au lieu de (t/T)^γ. Effet mesuré : colonne « γ corrigé »")
    print("      ci-dessus. La conclusion du banc n'en dépend pas — les trois")
    print("      solveurs de référence convergent au même rythme sur Kuhn.")


# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--icm", action="store_true")
    ap.add_argument("--equite", action="store_true")
    ap.add_argument("--pushfold", action="store_true")
    ap.add_argument("--dcfr", action="store_true")
    ap.add_argument("--long", action="store_true",
                    help="active les énumérations exhaustives (~15 min)")
    a = ap.parse_args()
    tout = not (a.icm or a.equite or a.pushfold or a.dcfr)

    t0 = time.perf_counter()
    if tout or a.icm:
        section_icm(a.long)
    if tout or a.equite:
        section_equite(a.long)
    if tout or a.pushfold:
        section_pushfold(a.long)
    if tout or a.dcfr:
        section_dcfr(a.long)

    print("\n" + "═" * 74)
    print(f"{_ETAT['ok']} vérifications conformes · {_ETAT['ko']} écart(s) "
          f"· {time.perf_counter() - t0:.0f} s")
    print("═" * 74)
    return 1 if _ETAT["ko"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
