#!/usr/bin/env python
"""Banc des INVARIANTS de l'ICM : ce que le calcul doit préserver.

    python banc_invariants_icm.py               le balayage de référence
    python banc_invariants_icm.py --large       + balayages aléatoires larges
    python banc_invariants_icm.py --seuils      détaille les deux seuils
                                                numériques et leur marge
    python banc_invariants_icm.py --continuite  la branche « tapis nul » est-elle
                                                bien la LIMITE de la récursion ?

Pourquoi ce fichier existe
--------------------------
Les cinq défauts graves trouvés récemment étaient tous AUX JOINTURES, et
chaque composant était juste isolément. Un invariant est le seul test qui
n'a pas besoin de connaître la bonne réponse : il énonce une propriété que
le résultat doit vérifier QUELLE QUE SOIT l'entrée. C'est ce qui attrape
les défauts qu'aucune relecture ne voit.

L'« amputation » du tapis à 99,9999999 % — un contournement introduit pour
éviter une division par zéro dans `bubble_factor` — aurait été signalée
immédiatement par l'invariant d'échelle : amputer, c'est faire dépendre le
résultat de la VALEUR ABSOLUE des jetons, alors que l'ICM ne dépend que de
leurs proportions.

Les invariants mesurés ici
--------------------------
1. **échelle** : multiplier tous les tapis par une constante ne change RIEN
   aux équités. Corollaire : l'unité (jetons, big blinds, millions) est sans
   effet. C'est l'invariant qui détecte tout calcul accroché à la valeur
   absolue des jetons.
2. **permutation** : permuter deux joueurs permute leurs équités, et ne
   touche à rien d'autre.
3. **conservation** : la somme des équités égale la dotation *attribuable*.
   Attention à la définition, c'est elle qui décide du résultat : `_validate`
   TRONQUE les gains au nombre de joueurs (une 3ᵉ place n'existe pas à deux),
   donc la référence est ``sum(payouts[:len(stacks)])``, pas ``sum(payouts)``.
4. **monotonie** : à structure fixée, augmenter son tapis ne peut pas
   diminuer son équité — que les jetons viennent d'un autre joueur ou de
   nulle part.
5. **non-linéarité** : le fait fondateur de l'ICM. Le PLUS GROS tapis vaut
   moins que sa part de jetons, le PLUS PETIT plus. Uniquement le plus gros
   et le plus petit : les tapis intermédiaires peuvent tomber des deux côtés
   (mesuré ici même), et l'affirmer pour tous serait faux.
6. **exact ↔ Monte-Carlo** : sur les mêmes entrées, l'écart doit tenir dans
   l'erreur d'échantillonnage. Elle est CALCULÉE, pas supposée : sous
   Harville la variance exacte du gain du joueur *i* vaut
   ``Σₖ pᵢₖ·πₖ² − (Σₖ pᵢₖ·πₖ)²``, d'où σ = √(var/n_sims) — bien plus fine
   que la borne √(0,25/n) que renvoie `_finish_probs_mc`.
7. **bubble factor** : == 1 exactement quand l'équité est affine en jetons
   (heads-up, ou gain unique) ; > 1 sinon ; croissant avec le tapis du
   vilain.
8. **continuité en zéro** : la branche « tapis nul » doit être la LIMITE de
   la récursion, pas une valeur voisine. Voir plus bas — c'est l'invariant
   que les six premiers ne pouvaient pas voir.
9. **translation, homogénéité, tapis égaux, dénominateurs nuls** : quatre
   propriétés des GAINS et des cas dégénérés, jusqu'ici non mesurées.

Et les mêmes invariants sur le chemin PKO : la valeur d'une prime ne doit
pas dépendre de l'unité dans laquelle on compte les jetons.

Pourquoi la continuité en zéro était l'angle mort
-------------------------------------------------
Le facteur de bulle vaut ``($EV − $EV_perte) / ($EV_gain − $EV)``. Pour le
joueur COUVERT — le tapis court, celui pour qui ce nombre compte le plus —
``$EV_perte`` est évaluée À TAPIS ZÉRO, donc dans la branche spéciale ; le
numérateur et le dénominateur du quotient viennent alors de DEUX RÉGIMES DE
CODE DIFFÉRENTS. (Pour le joueur COUVRANT c'est ``$EV_gain`` qui y passe :
gagner, c'est éliminer.)

Un écart entre la branche et la limite est ABSORBÉ dans les équités — la
somme reste conservée, l'échelle, la permutation et la monotonie restent
vertes — mais AMPLIFIÉ par le quotient. Mesuré en section 10(c) : une erreur
d'équité de 1 % de la dotation devient **11,9 à 20,3 %** d'erreur relative sur
le facteur de bulle. Les six premiers invariants ne pouvaient pas la voir.

Le résultat de la mesure (section 10) : **la branche et la limite
coïncident**. L'écart tend vers zéro LINÉAIREMENT en epsilon — extrapolé en
zéro, il vaut au plus 2,0e-15 en relatif sur huit spots, soit le bruit de la
double précision. La branche n'est donc pas « proche » de la limite : elle
l'est. Il n'y avait rien à corriger dans ``pfs/core/icm.py`` — mais rien ne
le prouvait.

Ce que la section 10 attrape, et que les sections 1 à 8 laissaient passer
------------------------------------------------------------------------
En injectant dans la branche un écart RELATIF ρ sur ce que touche l'éliminé,
et en le REDISTRIBUANT sur les vivants pour que la dotation reste exacte :
la mutation est invisible pour la conservation (somme intacte), pour
l'échelle (ρ est sans dimension), pour la permutation (elle est symétrique)
et pour la monotonie (elle est constante).

Dans sa version la plus furtive — l'écart n'est appliqué que s'il reste au
moins DEUX survivants, ce qui épargne le heads-up et les cas affines, seuls
endroits où l'ancien socle voyait quelque chose :

Comptage sur les quatre fichiers de tests qui touchent l'ICM
(``test_icm_invariants.py``, ``test_icm.py``, ``test_icm_tapis_nul.py``,
``test_golden_values.py``) :

    ρ        tests d'avant       tests des sections 10-11
    1e-6     2 échecs            35 échecs
    1e-9     **0 échec**         21 échecs
    1e-11    **0 échec**         12 échecs

À ρ = 1e-9, tout ce qui existait avant était VERT sur un ICM faux.

La nuance qu'il ne faut pas rater
---------------------------------
L'ICM est AUTHENTIQUEMENT discontinu quand PLUSIEURS tapis sont nuls, et
exiger la continuité là ferait « corriger » du code juste. La forme générale,
mesurée en 10(f) :

    lim  icm([survivants] + ε·[a₁ … a_k], π)
    ε→0      =  icm([survivants], π[:r])  ⊕  icm([a₁ … a_k], π[r:])

La limite dépend donc des RAPPORTS ``a_i`` entre les tapis qui s'évanouissent.
Sur ``[100, ε, ε/2]`` avec 50/30/20 elle vaut (26,667 ; 23,333) ; sur
``[100, ε, ε²]`` elle vaut (30 ; 20) ; sur ``[100, ε, ε]`` elle vaut
(25 ; 25). Trois chemins, trois limites : il n'y a pas de valeur en zéro.

À tapis EXACTEMENT nuls, ces rapports n'existent plus — l'ordre d'élimination
n'est pas dans les données. La branche rend (25 ; 25), c'est-à-dire la limite
du chemin SYMÉTRIQUE, seule réponse compatible avec l'invariance par
permutation. C'est un choix, pas une convergence, et il est documenté comme
tel.

À UN SEUL tapis nul le groupe qui s'évanouit est un singleton : plus aucun
rapport, la limite est unique, et la continuité est donc EXIGIBLE.

Ce que le banc ne mesure pas
----------------------------
Il ne vérifie AUCUNE valeur d'équité contre une référence externe : c'est le
rôle de `test_icm.py` (calculs à la main sur 3 joueurs) et de
`banc_calculs_exacts.py`. Un banc d'invariants peut être entièrement vert sur
un ICM systématiquement faux d'un facteur constant.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.core.icm import (  # noqa: E402
    IcmSpot,
    PkoSpot,
    _finish_probs_exact,
    _finish_probs_mc,
    _validate,
    analyse_icm_spot,
    analyse_pko_spot,
    bubble_factor,
    icm_dollar_ev,
    icm_equities,
    icm_required_equity,
    risk_premium,
    spot_pko_face_a_tapis,
)

# ── Tables de référence ────────────────────────────────────────────────────
# Une vraie table de tournoi PMU à 9 joueurs, et quelques structures qui
# stressent les cas limites (tapis nul, ex æquo, gain unique, satellite).

PAYOUTS_9 = (176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34)
TABLE_9 = (35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
PRIMES_9 = (8.13, 8.97, 7.43, 4.17, 6.0, 6.0, 6.0, 6.0, 6.0)

REFERENCE: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...] = (
    ("table PMU 9 joueurs", TABLE_9, PAYOUTS_9),
    ("bulle classique 4/3", (5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0)),
    ("ex æquo 3 joueurs", (1000.0, 1000.0, 1000.0), (50.0, 30.0, 20.0)),
    ("un tapis nul", (0.0, 10.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0)),
    ("deux tapis nuls", (0.0, 0.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0)),
    ("gain unique (winner-take-all)", (6000.0, 3000.0, 1000.0), (100.0,)),
    ("satellite : 3 places égales sur 5", (900.0, 700.0, 500.0, 300.0, 100.0),
     (100.0, 100.0, 100.0)),
    ("heads-up", (7000.0, 3000.0), (60.0, 40.0)),
    ("plus de gains que de joueurs", (100.0, 50.0), (50.0, 30.0, 20.0)),
    ("joueur unique", (100.0,), (50.0, 30.0, 20.0)),
)

FACTEURS = (1e-3, 1.0, 1e3, 1e6)
"""Les facteurs d'échelle demandés. 1e-3 ≈ compter en milliers de jetons,
1e6 ≈ compter en millionièmes — l'ICM ne doit pas s'en apercevoir."""


def _ligne(nom: str, valeur: str, verdict: bool, detail: str = "") -> bool:
    marque = "ok  " if verdict else "ÉCHEC"
    print(f"  {marque} {nom:<44s} {valeur:>16s}  {detail}")
    return verdict


# ═══════════════════════════════════════════════════════════════════════════
# 1 — ÉCHELLE
# ═══════════════════════════════════════════════════════════════════════════


def section_echelle(rng: random.Random, large: bool) -> bool:
    print("── 1. INVARIANCE D'ÉCHELLE  (k·tapis ⇒ mêmes équités)")
    ok = True
    pire = 0.0
    for nom, stacks, payouts in REFERENCE:
        ref = icm_equities(stacks, payouts)
        ecart = 0.0
        for k in FACTEURS:
            got = icm_equities([s * k for s in stacks], payouts)
            ecart = max(ecart, float(np.max(np.abs(got - ref))))
        pire = max(pire, ecart)
        # Tolérance : l'arithmétique flottante n'est pas exactement associative,
        # donc on compare à l'échelle de la dotation, pas à zéro. 1e-11 laisse
        # cinq décades au-dessus du bruit mesuré (~1e-14) tout en restant très
        # au-dessous de toute régression réelle.
        seuil = 1e-11 * max(sum(payouts[: len(stacks)]), 1.0)
        ok &= _ligne(nom, f"{ecart:.2e}", ecart <= seuil, f"seuil {seuil:.1e}")

    if large:
        pire_alea = 0.0
        for _ in range(400):
            n = rng.randint(2, 9)
            stacks = [rng.uniform(0.0, 1000.0) for _ in range(n)]
            if sum(stacks) <= 0:
                continue
            m = rng.randint(1, n + 2)
            payouts = sorted((rng.uniform(1, 200) for _ in range(m)), reverse=True)
            ref = icm_equities(stacks, payouts)
            for k in FACTEURS:
                got = icm_equities([s * k for s in stacks], payouts)
                pire_alea = max(pire_alea, float(np.max(np.abs(got - ref))))
        ok &= _ligne("400 tables aléatoires", f"{pire_alea:.2e}",
                     pire_alea <= 1e-9, "seuil 1.0e-09")

    # Le chemin Monte-Carlo (> 12 joueurs) : les tirages sont proportionnels,
    # donc l'invariance doit être EXACTE, au bit près, à graine fixée.
    stacks13 = [float(x) for x in range(1000, 14000, 1000)]
    ref = icm_equities(stacks13, [50.0, 30.0, 20.0], n_sims=20_000, seed=3)
    ecart = 0.0
    for k in FACTEURS:
        got = icm_equities([s * k for s in stacks13], [50.0, 30.0, 20.0],
                           n_sims=20_000, seed=3)
        ecart = max(ecart, float(np.max(np.abs(got - ref))))
    ok &= _ligne("13 joueurs (chemin Monte-Carlo)", f"{ecart:.2e}", ecart == 0.0,
                 "attendu : 0 exactement")

    print(f"     écart maximal sur les tables de référence : {pire:.3e}\n")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 2 — PERMUTATION
# ═══════════════════════════════════════════════════════════════════════════


def section_permutation(rng: random.Random, large: bool) -> bool:
    print("── 2. INVARIANCE PAR PERMUTATION  (σ·tapis ⇒ σ·équités)")
    ok = True
    for nom, stacks, payouts in REFERENCE:
        n = len(stacks)
        if n < 2:
            continue
        ref = icm_equities(stacks, payouts)
        ecart = 0.0
        for _ in range(20):
            perm = list(range(n))
            rng.shuffle(perm)
            got = icm_equities([stacks[i] for i in perm], payouts)
            ecart = max(ecart, float(np.max(np.abs(got - ref[perm]))))
        ok &= _ligne(nom, f"{ecart:.2e}", ecart <= 1e-9, "20 permutations")

    # Le bubble factor doit suivre ses deux joueurs.
    ecart = 0.0
    for _ in range(200 if large else 40):
        n = rng.randint(3, 7)
        stacks = [rng.uniform(1, 500) for _ in range(n)]
        m = rng.randint(2, n)
        payouts = sorted((rng.uniform(1, 100) for _ in range(m)), reverse=True)
        h, v = rng.sample(range(n), 2)
        ref = bubble_factor(stacks, payouts, h, v)
        perm = list(range(n))
        rng.shuffle(perm)
        got = bubble_factor([stacks[i] for i in perm], payouts,
                            perm.index(h), perm.index(v))
        ecart = max(ecart, abs(got - ref) / max(abs(ref), 1.0))
    ok &= _ligne("bubble_factor suit ses deux joueurs", f"{ecart:.2e}",
                 ecart <= 1e-9, "écart relatif")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 3 — CONSERVATION DE LA DOTATION
# ═══════════════════════════════════════════════════════════════════════════


def section_conservation(rng: random.Random, large: bool) -> bool:
    print("── 3. CONSERVATION  (Σ équités == Σ gains attribuables)")
    print("     rappel : les gains sont TRONQUÉS au nombre de joueurs.")
    ok = True
    for nom, stacks, payouts in REFERENCE:
        eq = icm_equities(stacks, payouts)
        attendu = sum(payouts[: len(stacks)])
        ecart = abs(float(eq.sum()) - attendu)
        ok &= _ligne(nom, f"{ecart:.2e}",
                     ecart <= 1e-9 * max(attendu, 1.0),
                     f"Σ={float(eq.sum()):.4f} / {attendu:.4f}")

    if large:
        pire = 0.0
        for _ in range(600):
            n = rng.randint(1, 9)
            # Un tapis nul sur deux, pour couvrir massivement le cas qui a
            # motivé l'amputation.
            stacks = [0.0 if rng.random() < 0.35 else rng.uniform(1, 1000)
                      for _ in range(n)]
            if sum(stacks) <= 0:
                continue
            m = rng.randint(1, n + 3)
            payouts = sorted((rng.uniform(1, 200) for _ in range(m)), reverse=True)
            eq = icm_equities(stacks, payouts)
            attendu = sum(payouts[:n])
            pire = max(pire, abs(float(eq.sum()) - attendu) / max(attendu, 1.0))
        ok &= _ligne("600 tables aléatoires (35 % de tapis nuls)",
                     f"{pire:.2e}", pire <= 1e-9, "écart relatif")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 4 — MONOTONIE
# ═══════════════════════════════════════════════════════════════════════════


def section_monotonie(rng: random.Random, large: bool) -> bool:
    print("── 4. MONOTONIE  (plus de jetons ⇒ jamais moins d'équité)")
    ok = True
    for nom, stacks, payouts in REFERENCE:
        n = len(stacks)
        if n < 2:
            continue
        pire = 0.0
        for hero in range(n):
            # (a) transfert : les jetons viennent d'un autre joueur.
            for autre in range(n):
                if autre == hero or stacks[autre] <= 0:
                    continue
                prec = None
                for frac in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
                    s = list(stacks)
                    d = stacks[autre] * frac
                    s[hero] += d
                    s[autre] -= d
                    v = float(icm_equities(s, payouts)[hero])
                    if prec is not None:
                        pire = min(pire, v - prec)
                    prec = v
            # (b) ex nihilo : le total change, la part du héros aussi.
            prec = None
            for d in (0.0, 1.0, 10.0, 100.0, 1e4):
                s = list(stacks)
                s[hero] += d
                v = float(icm_equities(s, payouts)[hero])
                if prec is not None:
                    pire = min(pire, v - prec)
                prec = v
        ok &= _ligne(nom, f"{pire:+.2e}", pire >= -1e-9,
                     "pire recul d'équité")

    if large:
        pire = 0.0
        for _ in range(300):
            n = rng.randint(2, 8)
            stacks = [rng.uniform(1, 1000) for _ in range(n)]
            m = rng.randint(1, n)
            payouts = sorted((rng.uniform(1, 200) for _ in range(m)), reverse=True)
            hero = rng.randrange(n)
            prec = None
            for d in (0.0, 5.0, 50.0, 500.0):
                s = list(stacks)
                s[hero] += d
                v = float(icm_equities(s, payouts)[hero])
                if prec is not None:
                    pire = min(pire, v - prec)
                prec = v
        ok &= _ligne("300 tables aléatoires", f"{pire:+.2e}", pire >= -1e-9, "")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 5 — NON-LINÉARITÉ
# ═══════════════════════════════════════════════════════════════════════════


def section_non_linearite(rng: random.Random, large: bool) -> bool:
    print("── 5. NON-LINÉARITÉ  (gros tapis décoté, petit tapis surcoté)")
    print("     La frontière du phénomène, et elle est étroite : l'écart est")
    print("     nul EXACTEMENT dans deux cas, et deux seulement —")
    print("       • une seule place payée : l'ICM se réduit au chipEV ;")
    print("       • tous les tapis égaux : tout le monde vaut la moyenne.")
    print("     Partout ailleurs il y a décote. En particulier un satellite")
    print("     (k places de valeur ÉGALE sur n > k joueurs) est fortement")
    print("     non linéaire : ce sont les places non payées qui font la")
    print("     courbure, pas l'inégalité des gains.")
    ok = True
    for nom, stacks, payouts in REFERENCE:
        n = len(stacks)
        pool = sum(payouts[:n])
        total = sum(stacks)
        eq = icm_equities(stacks, payouts)
        part = np.array([s / total * pool for s in stacks])
        ecarts = eq - part
        pire = float(np.max(np.abs(ecarts)))
        i_gros = int(np.argmax(stacks))
        i_petit = int(np.argmin(stacks))
        # Une seule place PAYÉE (les gains nuls ne comptent pas), ou tous les
        # tapis égaux : l'équité vaut exactement la part de jetons.
        payees = [x for x in payouts[:n] if x > 0.0]
        if len(payees) <= 1 or stacks[i_gros] == stacks[i_petit]:
            verdict = pire <= 1e-11 * max(pool, 1.0)
            detail = "écart nul attendu (chipEV ou ex æquo)"
        else:
            verdict = bool(ecarts[i_gros] < 0.0 and ecarts[i_petit] > 0.0)
            detail = f"gros {ecarts[i_gros]:+.3f}  petit {ecarts[i_petit]:+.3f}"
        ok &= _ligne(nom, f"{pire:.2e}", verdict, detail)

    # Les tapis INTERMÉDIAIRES ne sont pas contraints : on le mesure au lieu
    # de l'affirmer, parce que c'est exactement le genre de généralisation
    # fausse qu'un test tautologique laisserait passer.
    stacks = (100.0, 90.0, 80.0, 70.0, 60.0, 50.0)
    payouts = (40.0, 30.0, 20.0, 10.0)
    eq = icm_equities(stacks, payouts)
    part = np.array([s / sum(stacks) * sum(payouts) for s in stacks])
    signes = np.sign(np.round(eq - part, 9))
    print(f"     tapis intermédiaires, signes des écarts : {signes.tolist()}")
    print("     → contre-exemple : la décote ne concerne QUE les extrêmes.\n")

    if large:
        viol = 0
        for _ in range(500):
            n = rng.randint(3, 8)
            stacks = sorted((rng.uniform(1, 1000) for _ in range(n)), reverse=True)
            m = rng.randint(2, n)
            payouts = sorted({round(rng.uniform(1, 100), 3) for _ in range(m)},
                             reverse=True)
            if len(payouts) < 2:
                continue
            eq = icm_equities(stacks, payouts)
            pool, total = sum(payouts), sum(stacks)
            if eq[0] >= stacks[0] / total * pool:
                viol += 1
            if eq[-1] <= stacks[-1] / total * pool:
                viol += 1
        ok &= _ligne("500 tables aléatoires (gains distincts)", f"{viol}",
                     viol == 0, "violations")
        print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 6 — EXACT ↔ MONTE-CARLO
# ═══════════════════════════════════════════════════════════════════════════


def sigma_echantillonnage(probs: np.ndarray, payouts: np.ndarray,
                          n_sims: int) -> np.ndarray:
    """Erreur-type EXACTE de l'estimateur Monte-Carlo, par joueur.

    Le tirage Monte-Carlo répète ``n_sims`` fois une variable aléatoire *X*
    « gain encaissé par le joueur *i* » (zéro s'il finit hors des places
    payées). Sa loi est donnée par les probabilités EXACTES de Harville, donc
    sa variance l'est aussi :

    .. math:: \\operatorname{Var}(X_i) = \\sum_k p_{ik}\\pi_k^2
              - \\Bigl(\\sum_k p_{ik}\\pi_k\\Bigr)^2

    D'où σ = √(Var/n_sims). C'est la borne à laquelle comparer l'écart —
    calculée, pas supposée. La valeur que renvoie `_finish_probs_mc`
    (√(0,25/n), bornée pour une proportion) est correcte mais bien plus
    lâche, et elle ne porte pas sur la même quantité.
    """
    esp = probs @ payouts
    var = probs @ (payouts ** 2) - esp ** 2
    return np.sqrt(np.maximum(var, 0.0) / n_sims)


def section_monte_carlo(rng: random.Random, large: bool) -> bool:
    print("── 6. EXACT ↔ MONTE-CARLO  (écart mesuré en σ d'échantillonnage)")
    print("     σ calculée depuis les probabilités exactes, pas supposée.")
    ok = True
    cas = [
        ("bulle 4/3", (5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0)),
        ("ex æquo 3", (100.0, 100.0, 100.0), (50.0, 30.0, 20.0)),
        ("6 joueurs, 4 gains", (176.0, 124.0, 92.0, 70.0, 55.0, 44.0),
         (100.0, 60.0, 40.0, 25.0)),
        ("chipleader écrasant", (9000.0, 500.0, 300.0, 200.0), (70.0, 30.0)),
        ("table PMU 9 joueurs", TABLE_9, PAYOUTS_9),
    ]
    # Le nombre de tirages ne change PAS la distribution du z-score (σ suit
    # 1/√n, l'écart aussi) : il n'achète que du temps de calcul. On en prend
    # donc juste assez pour que le banc reste rejouable — un banc qu'on ne
    # relance pas ne mesure plus rien.
    graines = range(6 if large else 3)
    tailles = (20_000, 80_000) if large else (50_000,)
    zmax_global = 0.0
    for nom, stacks, payouts in cas:
        pay = np.asarray(payouts, dtype=np.float64)
        exact = _finish_probs_exact(tuple(stacks), len(payouts))
        eq_ex = exact @ pay
        zmax = 0.0
        for n_sims in tailles:
            sigma = sigma_echantillonnage(exact, pay, n_sims)
            for seed in graines:
                mc, _ = _finish_probs_mc(tuple(stacks), len(payouts), n_sims, seed)
                ecart = np.abs(mc @ pay - eq_ex)
                z = np.max(ecart / np.where(sigma > 0, sigma, np.inf))
                zmax = max(zmax, float(z))
        zmax_global = max(zmax_global, zmax)
        ok &= _ligne(nom, f"{zmax:.2f} σ", zmax <= 5.0,
                     f"{len(list(graines))} graines × {len(tailles)} tailles")
    print(f"     |z| maximal observé : {zmax_global:.3f} σ  "
          f"→ le seuil de 5 σ garde {5.0 - zmax_global:.2f} σ de marge.\n")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 7 — BUBBLE FACTOR
# ═══════════════════════════════════════════════════════════════════════════


def section_bubble(rng: random.Random, large: bool) -> bool:
    print("── 7. BUBBLE FACTOR")
    ok = True

    print("     (a) == 1 exactement quand l'équité est AFFINE en jetons")
    affines = [
        ("heads-up, 1 gain", (6000.0, 4000.0), (100.0,)),
        ("heads-up, 2 gains", (6000.0, 4000.0), (60.0, 40.0)),
        ("heads-up, tapis égaux", (100.0, 100.0), (70.0, 30.0)),
        ("gain unique, 3 joueurs", (100.0, 200.0, 300.0), (50.0,)),
        ("gain unique, 6 joueurs", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (50.0,)),
    ]
    for nom, stacks, payouts in affines:
        bf = bubble_factor(stacks, payouts, 0, 1)
        ok &= _ligne(nom, f"{bf:.15f}", abs(bf - 1.0) <= 1e-9,
                     f"|BF−1| = {abs(bf - 1.0):.2e}")

    print("     (b) > 1 dès que la structure met quelque chose en jeu")
    mini = math.inf
    arg = None
    for _ in range(2000 if large else 300):
        n = rng.randint(3, 8)
        stacks = [rng.uniform(1, 1000) for _ in range(n)]
        m = rng.randint(2, n)
        payouts = sorted({round(rng.uniform(1, 100), 3) for _ in range(m)},
                         reverse=True)
        if len(payouts) < 2:
            continue
        h, v = rng.sample(range(n), 2)
        bf = bubble_factor(stacks, payouts, h, v)
        if bf < mini:
            mini, arg = bf, (n, len(payouts))
    ok &= _ligne("balayage aléatoire (n≥3, gains distincts)", f"{mini:.9f}",
                 mini > 1.0, f"minimum atteint à n={arg[0]}, {arg[1]} gains")

    print("     (c) croissant avec le tapis du vilain")
    print("         (transfert depuis deux joueurs neutres, total constant)")
    prec = None
    croissant = True
    for v_s in (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0):
        neutre = 12000.0 - 3000.0 - v_s
        if neutre <= 0:
            break
        s = [3000.0, v_s, neutre / 2, neutre / 2]
        bf = bubble_factor(s, [50.0, 30.0, 20.0], 0, 1)
        fleche = "" if prec is None else ("↗" if bf > prec else "↘ RECUL")
        print(f"         vilain {v_s:>8.0f}   BF = {bf:.6f}  {fleche}")
        if prec is not None and bf < prec:
            croissant = False
        prec = bf
    ok &= _ligne("monotonie en le tapis du vilain", "—", croissant, "")

    print("     (d) structure dégénérée : tous les survivants payés pareil")
    print("         rien n'est en jeu ⇒ +inf (jamais un BF négatif)")
    for nom, stacks, payouts in [
        ("3 joueurs, 3 gains égaux", (1.0, 2.0, 3.0), (1.0, 1.0, 1.0)),
        ("4 joueurs, 4 gains égaux", (10.0, 20.0, 30.0, 40.0), (7.0,) * 4),
        ("2 joueurs, 2 gains égaux", (100.0, 100.0), (50.0, 50.0)),
    ]:
        bf = bubble_factor(stacks, payouts, 0, 1)
        ok &= _ligne(nom, f"{bf}", math.isinf(bf) and bf > 0, "")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 8 — PKO : les mêmes invariants sur les primes
# ═══════════════════════════════════════════════════════════════════════════


def section_pko(rng: random.Random, large: bool) -> bool:
    print("── 8. PKO  (une prime ne dépend pas de l'unité des jetons)")
    ok = True

    print("     (a) échelle, depuis les tapis lus à l'écran")
    ref = None
    ecart = 0.0
    for k in FACTEURS:
        spot = spot_pko_face_a_tapis(
            [t * k for t in TABLE_9], PAYOUTS_9, PRIMES_9, hero=0, villain=3,
            blindes_mortes=1.4 * k, deja_engage_hero=1.0 * k)
        a = analyse_pko_spot(spot)
        cur = (a.bounty_value, a.required_no_bounty, a.required_with_bounty)
        if ref is None:
            ref = cur
            print(f"         référence : prime {a.bounty_value:.6f} $  "
                  f"ICM {a.required_no_bounty:.6%}  PKO {a.required_with_bounty:.6%}")
        else:
            ecart = max(ecart, max(abs(x - y) for x, y in zip(ref, cur)))
        if not a.villain_eliminated:
            ok &= _ligne(f"k={k:g} : vilain couvert reconnu", "—", False, "")
    ok &= _ligne("spot_pko_face_a_tapis", f"{ecart:.2e}", ecart <= 1e-9,
                 f"facteurs {FACTEURS}")

    print("     (b) échelle, sur un PkoSpot construit à la main")
    ref = None
    ecart = 0.0
    for k in FACTEURS:
        a = analyse_pko_spot(PkoSpot(
            stacks=(100.0 * k, 0.0, 100.0 * k), payouts=(100.0,),
            bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
            pot=100.0 * k, bet=100.0 * k))
        cur = (a.bounty_value, a.required_no_bounty, a.required_with_bounty)
        if ref is None:
            ref = cur
        else:
            ecart = max(ecart, max(abs(x - y) for x, y in zip(ref, cur)))
    ok &= _ligne("PkoSpot direct", f"{ecart:.2e}", ecart <= 1e-9, "")

    print("     (c) le vilain NON couvert le reste à toute échelle")
    print("         c'est le défaut qu'un seuil absolu produisait : héros 100,")
    print("         vilain 1 jeton restant, exprimé en unités de plus en plus")
    print("         petites — le vilain finissait par « disparaître ».")
    ref = None
    ecart = 0.0
    coherent = True
    for k in (1.0, 1e-6, 1e-11, 1e-12, 1e-13, 1e-15):
        a = analyse_pko_spot(PkoSpot(
            stacks=(100.0 * k, 1.0 * k, 100.0 * k), payouts=(100.0, 50.0),
            bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
            pot=100.0 * k, bet=100.0 * k))
        cur = (a.bounty_value, a.required_with_bounty)
        print(f"         k={k:<8g} éliminé={a.villain_eliminated!s:<5s} "
              f"prime={a.bounty_value:8.4f}  exigé={a.required_with_bounty:.6%}")
        if a.villain_eliminated:
            coherent = False
        if ref is None:
            ref = cur
        else:
            ecart = max(ecart, max(abs(x - y) for x, y in zip(ref, cur)))
    ok &= _ligne("un vilain non couvert ne devient jamais éliminable", "—",
                 coherent, "")
    ok &= _ligne("verdict identique à toutes les échelles", f"{ecart:.2e}",
                 ecart <= 1e-9, "")

    print("     (d) permutation des joueurs")
    spot = spot_pko_face_a_tapis(TABLE_9, PAYOUTS_9, PRIMES_9,
                                 hero=0, villain=3,
                                 blindes_mortes=1.4, deja_engage_hero=1.0)
    base = analyse_pko_spot(spot)
    pire = 0.0
    for _ in range(50 if large else 10):
        perm = list(range(len(spot.stacks)))
        rng.shuffle(perm)
        a = analyse_pko_spot(PkoSpot(
            stacks=tuple(spot.stacks[i] for i in perm),
            payouts=spot.payouts,
            bounties=tuple(spot.bounties[i] for i in perm),
            hero=perm.index(spot.hero), villain=perm.index(spot.villain),
            pot=spot.pot, bet=spot.bet))
        pire = max(pire, abs(a.bounty_value - base.bounty_value),
                   abs(a.required_with_bounty - base.required_with_bounty),
                   abs(a.required_no_bounty - base.required_no_bounty))
    ok &= _ligne("analyse_pko_spot invariante par permutation", f"{pire:.2e}",
                 pire <= 1e-9, "")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 10 — CONTINUITÉ EN ZÉRO : la branche « tapis nul » est-elle la LIMITE ?
# ═══════════════════════════════════════════════════════════════════════════

EPSILONS = (10.0, 1.0, 0.1, 1e-6, 1e-9, 1e-12)
"""Les tapis résiduels balayés. 10 jetons est déjà « petit » devant un tapis
court de 500 ; 1e-12 est en dessous du dernier chiffre significatif d'un
tapis réel. Six décades suffisent à distinguer une CONVERGENCE d'un PALIER."""


def bf_par_continuite(stacks, payouts, hero: int, villain: int,
                      eps: float) -> float:
    """Le facteur de bulle où le tapis qui tombe à zéro garde ``eps`` jeton.

    Selon qui couvre qui, ce n'est pas le même terme du quotient qui traverse
    la branche « tapis nul » :

    * héros COUVERT (``s[hero] <= s[villain]``) : c'est ``$EV_perte``, le
      héros finit à zéro s'il perd ;
    * héros COUVRANT : c'est ``$EV_gain``, gagner c'est éliminer le vilain.

    On ne perturbe que le terme concerné — perturber les deux mélangerait
    deux effets d'ordre epsilon et brouillerait la pente mesurée. Le transfert
    est réduit d'autant, donc le TOTAL de jetons reste celui du spot : la
    limite mesurée est bien celle de la même table, pas d'une table enrichie.

    Parameters
    ----------
    eps : float
        Jetons laissés au perdant. ``eps = 0`` redonne exactement
        `pfs.core.icm.bubble_factor` (mêmes états, même arithmétique).

    Returns
    -------
    float
        ``($EV − $EV_perte) / ($EV_gain − $EV)``, sans le garde-fou anti-0/0
        de la fonction de production : ici on VEUT voir le quotient brut.
    """
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = icm_dollar_ev(s, p, hero)

    d_win = amt - eps if s[villain] <= s[hero] else amt
    d_lose = amt - eps if s[hero] <= s[villain] else amt
    win = list(s)
    win[hero] += d_win
    win[villain] -= d_win
    lose = list(s)
    lose[hero] -= d_lose
    lose[villain] += d_lose
    return ((base - icm_dollar_ev(lose, p, hero))
            / (icm_dollar_ev(win, p, hero) - base))


def extrapole_en_zero(f, e1: float, e2: float) -> float:
    """``f(0)`` par extrapolation linéaire depuis ``f(e1)`` et ``f(e2)``.

    C'est l'extrapolation de Richardson au premier ordre. Elle vaut mieux
    qu'un simple « f(1e-12) est proche de f(0) » : si ``f(ε) = f(0) + Cε``,
    elle rend ``f(0)`` EXACTEMENT, quel que soit ``C``. Une branche fausse
    donne un palier, donc une extrapolée qui rate la branche de tout le saut.
    """
    return (e1 * f(e2) - e2 * f(e1)) / (e1 - e2)


# Huit spots où un tapis tombe à zéro, dans les DEUX sens de couverture.
CONTINUITE_BF = (
    # nom, tapis, gains, héros, vilain, sens
    ("PMU 9j : court(3) vs chipleader(1)", TABLE_9, PAYOUTS_9, 3, 1, "couvert"),
    ("PMU 9j : court(3) vs moyen(0)", TABLE_9, PAYOUTS_9, 3, 0, "couvert"),
    ("bulle 4/3 : court(3) vs gros(0)",
     (5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 3, 0, "couvert"),
    ("satellite 5j : court(4) vs gros(0)",
     (900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0), 4, 0, "couvert"),
    ("3 joueurs : court(2) vs gros(0)",
     (600.0, 300.0, 100.0), (50.0, 30.0, 20.0), 2, 0, "couvert"),
    ("heads-up (affine) : court(1) vs gros(0)",
     (7000.0, 3000.0), (60.0, 40.0), 1, 0, "couvert"),
    ("PMU 9j : chipleader(1) paie le court(3)", TABLE_9, PAYOUTS_9, 1, 3, "couvrant"),
    ("bulle 4/3 : gros(0) paie le court(3)",
     (5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 0, 3, "couvrant"),
)

# Configurations à UN SEUL tapis nul : la continuité y est EXIGIBLE.
CONTINUITE_VECTEUR = (
    ("9 joueurs / 9 gains",
     (0.0, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0), PAYOUTS_9),
    ("4 joueurs / 3 gains (le mort ne touche rien)",
     (0.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0)),
    ("4 joueurs / 4 gains (le mort touche la 4ᵉ place)",
     (0.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0, 10.0)),
    ("2 joueurs / 3 gains (gains tronqués)", (0.0, 100.0), (50.0, 30.0, 20.0)),
    ("3 joueurs / 1 gain (winner-take-all)", (0.0, 100.0, 200.0), (100.0,)),
    ("satellite 5j / 3 places égales",
     (0.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0)),
    ("zéro au milieu de la table",
     (300.0, 0.0, 200.0, 100.0), (50.0, 30.0, 20.0, 10.0)),
)


def _icm_mort_a_zero(stacks, payouts) -> np.ndarray:
    """La branche FAUSSE : un joueur à zéro touche 0, pas le dernier gain.

    C'est l'écriture naïve — « il n'a plus de jetons, il ne vaut plus rien » —
    et c'est celle que le correctif du tapis nul a remplacée. Elle sert ici de
    TÉMOIN : un invariant que cette version-là ne fait pas tomber ne mesure
    rien.
    """
    s, p = _validate(stacks, payouts)
    viv = [i for i, v in enumerate(s) if v > 0.0]
    if len(viv) == len(s):
        return icm_equities(s, p)
    out = np.zeros(len(s), dtype=np.float64)
    haut = list(p[: len(viv)])
    if haut:
        out[viv] = icm_equities([s[i] for i in viv], haut)
    return out


def _bf_mort_a_zero(stacks, payouts, hero: int, villain: int) -> float:
    """Le facteur de bulle qu'on obtiendrait avec la branche fausse."""
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = float(_icm_mort_a_zero(s, p)[hero])
    win = list(s)
    win[hero] += amt
    win[villain] -= amt
    lose = list(s)
    lose[hero] -= amt
    lose[villain] += amt
    return ((base - float(_icm_mort_a_zero(lose, p)[hero]))
            / (float(_icm_mort_a_zero(win, p)[hero]) - base))


def section_continuite(rng: random.Random, large: bool) -> bool:
    print("── 10. CONTINUITÉ EN ZÉRO  (la branche « tapis nul » est-elle la LIMITE ?)")
    ok = True

    print("     (a) facteur de bulle : branche vs limite, ε décroissant")
    print("         borne : |BF(ε) − BF(0)| ≤ 3·ε/tapis_court + 1e-13·BF.")
    print("         Le premier terme est SANS DIMENSION (ε rapporté au tapis")
    print("         court), donc valable à toute échelle ; pente mesurée 1,0 à")
    print("         2,2. Le second est le plancher de bruit : à ε = 1e-12 le")
    print("         terme linéaire vaut 1e-15 sur le heads-up, soit sous le")
    print("         dernier bit de $EV — la limite y est atteinte, pas approchée.")
    pire_k = 0.0
    for nom, s, p, h, v, sens in CONTINUITE_BF:
        b0 = bubble_factor(s, p, h, v)
        court = min(s[h], s[v])
        pire = 0.0
        detail = []
        for e in EPSILONS:
            if e >= court:
                continue
            d = abs(bf_par_continuite(s, p, h, v, e) - b0)
            pire = max(pire, d / (3.0 * e / court + 1e-13 * b0))
            if e >= 1e-9:                       # au-delà, c'est le bruit
                pire_k = max(pire_k, d * court / e)
            detail.append(f"{d:.1e}")
        ok &= _ligne(f"{nom}  [{sens}]", f"BF={b0:.6f}",
                     pire <= 1.0, "écarts " + " ".join(detail))
    print(f"         pente maximale mesurée (ε ≥ 1e-9) : {pire_k:.3f}  "
          f"(borne posée à 3)")

    print()
    print("     (b) extrapolation de Richardson : la limite EST la branche")
    print("         f(0) ≈ (ε₁·f(ε₂) − ε₂·f(ε₁))/(ε₁−ε₂), depuis 1e-6 et 1e-7.")
    pire_rel = 0.0
    for nom, s, p, h, v, sens in CONTINUITE_BF:
        b0 = bubble_factor(s, p, h, v)
        f0 = extrapole_en_zero(
            lambda e, s=s, p=p, h=h, v=v: bf_par_continuite(s, p, h, v, e),
            1e-6, 1e-7)
        rel = abs(f0 - b0) / abs(b0)
        pire_rel = max(pire_rel, rel)
        ok &= _ligne(nom, f"{rel:.2e}", rel <= 1e-11,
                     f"extrapolé {f0:.15f}")
    print(f"         écart relatif maximal : {pire_rel:.3e}  "
          f"(seuil 1e-11, soit {1e-11 / max(pire_rel, 1e-300):.0f}× de marge)")

    print()
    print("     (c) AMPLIFICATION : pourquoi les six premiers invariants")
    print("         ne pouvaient pas voir cet écart-là.")
    print("         Une erreur sur $EV_perte est absorbée dans la somme des")
    print("         équités (conservation verte) mais lue TELLE QUELLE par le")
    print("         numérateur du quotient.")
    ampli_max = 0.0
    for nom, s, p, h, v, sens in CONTINUITE_BF:
        b0 = bubble_factor(s, p, h, v)
        pool = sum(p[: len(s)])
        amt = min(s[h], s[v])
        base = icm_dollar_ev(s, p, h)
        win = list(s); win[h] += amt; win[v] -= amt
        lose = list(s); lose[h] -= amt; lose[v] += amt
        gain = icm_dollar_ev(win, p, h) - base
        perte = base - icm_dollar_ev(lose, p, h)
        # une erreur de 1 % de la dotation sur $EV_perte
        delta = 0.01 * pool
        faux = (perte + delta) / gain
        ampli = (abs(faux - b0) / b0) / 0.01
        ampli_max = max(ampli_max, ampli)
        print(f"         {nom:<44s} ×{ampli:5.2f}")
    ok &= _ligne("l'amplification est réelle (≥ 5×)", f"×{ampli_max:.2f}",
                 ampli_max >= 5.0, "1 % de dotation → % sur le BF")

    print()
    print("     (d) continuité du VECTEUR entier, à UN SEUL tapis nul")
    print("         borne : écart ≤ 4·ε·dotation/total — sans dimension aussi.")
    pire_k = 0.0
    for nom, s, p in CONTINUITE_VECTEUR:
        i0 = [i for i, x in enumerate(s) if x == 0.0][0]
        ref = icm_equities(s, p)
        echelle = sum(p[: len(s)]) / sum(s)
        pire = 0.0
        detail = []
        for e in EPSILONS:
            se = list(s)
            se[i0] = e
            d = float(np.max(np.abs(icm_equities(se, p) - ref)))
            pire = max(pire, d / (4.0 * e * echelle))
            pire_k = max(pire_k, d / (e * echelle))
            detail.append(f"{d:.1e}")
        ok &= _ligne(nom, f"{pire:.3f}·borne", pire <= 1.0,
                     " ".join(detail))
    print(f"         pente maximale mesurée : {pire_k:.3f}  (borne posée à 4)")

    print()
    print("     (e) TÉMOIN : la branche fausse (« le mort touche 0 ») ne")
    print("         converge pas — elle fait un PALIER, et le BF s'en va.")
    print("         DEUX ANGLES MORTS, à connaître : la branche fausse ne")
    print("         change QUE l'équité du joueur à zéro, donc")
    print("           • m < n : le mort ne touche rien de toute façon ;")
    print("           • héros COUVRANT : c'est le vilain qui tombe à zéro,")
    print("             pas lui — son $EV est identique dans les deux versions.")
    print("         Autrement dit : ce défaut-là ne se voit que du côté du")
    print("         joueur qui va mourir. C'est exactement celui qui compte.")
    mord = False
    for nom, s, p, h, v, sens in CONTINUITE_BF:
        vrai = bubble_factor(s, p, h, v)
        faux = _bf_mort_a_zero(s, p, h, v)
        ecart = abs(faux - vrai) / vrai
        a_vrai = icm_required_equity(100.0, 100.0, vrai)
        a_faux = icm_required_equity(100.0, 100.0, faux)
        if len(p) < len(s):
            note = "   [aveugle : m < n]"
        elif sens == "couvrant":
            note = "   [aveugle : le héros n'est pas celui qui tombe à zéro]"
        else:
            note = ""
        if ecart > 1e-9:
            mord = True
        print(f"         {nom:<44s} {vrai:.4f} → {faux:.4f}"
              f"  ({ecart:+7.2%}, {(a_faux - a_vrai) * 100:+5.1f} pts d'équité)"
              + note)
    ok &= _ligne("la branche fausse EST détectée par le BF", "—", mord,
                 "sinon l'invariant ne prouverait rien")
    # …et la continuité, elle, la voit même là où le BF ne bouge pas.
    s9 = (0.0, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
    ref_faux = _icm_mort_a_zero(s9, PAYOUTS_9)
    palier = min(
        float(np.max(np.abs(_icm_mort_a_zero([e if i == 0 else x
                                              for i, x in enumerate(s9)],
                                             PAYOUTS_9) - ref_faux)))
        for e in (1e-6, 1e-9, 1e-12))
    ok &= _ligne("branche fausse : palier au lieu d'une convergence",
                 f"{palier:.4f}", palier > 1.0,
                 "écart constant quand ε → 0")

    print()
    print("     (f) DISCONTINUITÉ ATTENDUE à PLUSIEURS tapis nuls")
    print("         La limite dépend des RAPPORTS entre les tapis qui")
    print("         s'évanouissent. Exiger la continuité ici ferait")
    print("         « corriger » du code juste.")
    p3 = (50.0, 30.0, 20.0)
    branche = icm_equities([100.0, 0.0, 0.0], p3)
    print(f"         branche [100, 0, 0]            → {branche.round(6).tolist()}")
    chemins = (("[100, ε, ε]   (symétrique)", 1.0, 1.0),
               ("[100, ε, ε/2]", 1.0, 0.5),
               ("[100, ε, ε/100]", 1.0, 0.01),
               ("[100, 3ε, 7ε]", 3.0, 7.0))
    limites = []
    for nom, a, b in chemins:
        lim = icm_equities([100.0, a * 1e-9, b * 1e-9], p3)
        limites.append(tuple(lim.round(6).tolist()))
        print(f"         {nom:<30s} → {lim.round(6).tolist()}")
    distinctes = len(set(limites))
    ok &= _ligne("plusieurs chemins, plusieurs limites", f"{distinctes} distinctes",
                 distinctes >= 3, "→ il n'existe PAS de valeur en zéro")
    ok &= _ligne("la branche = le chemin symétrique", "—",
                 bool(np.allclose(branche, icm_equities([100.0, 1e-9, 1e-9], p3),
                                  atol=1e-6)),
                 "seule réponse compatible avec la permutation")

    print()
    print("     (g) FACTORISATION ASYMPTOTIQUE : la forme générale")
    print("         lim icm(survivants ⊕ ε·mourants) =")
    print("             icm(survivants, π[:r])  ⊕  icm(mourants, π[r:])")
    pire = 0.0
    for nom, surv, mour, p in (
            ("2 survivants, 2 mourants", [300.0, 100.0], [1.0, 0.5],
             (50.0, 30.0, 20.0, 10.0)),
            ("3 survivants, 2 mourants", [300.0, 200.0, 100.0], [3.0, 1.0],
             (50.0, 30.0, 20.0, 10.0, 5.0)),
            ("1 survivant, 3 mourants", [500.0], [1.0, 0.5, 0.25],
             (50.0, 30.0, 20.0, 10.0)),
            ("2 survivants, 3 mourants", [400.0, 100.0], [1.0, 2.0, 4.0],
             (60.0, 40.0, 30.0, 20.0, 10.0))):
        r = len(surv)
        att = np.concatenate([icm_equities(surv, p[:r]),
                              icm_equities(mour, p[r:])])
        got = icm_equities(surv + [a * 1e-9 for a in mour], p)
        d = float(np.max(np.abs(got - att)))
        # et à tapis EXACTEMENT nuls, la branche doit rendre le cas symétrique
        att0 = np.concatenate([icm_equities(surv, p[:r]),
                               icm_equities([1.0] * len(mour), p[r:])])
        d0 = float(np.max(np.abs(icm_equities(surv + [0.0] * len(mour), p) - att0)))
        pire = max(pire, d, d0)
        ok &= _ligne(nom, f"{d:.1e} / {d0:.1e}", d <= 1e-6 and d0 <= 1e-9,
                     "limite ε→0 / branche == symétrique")
    print(f"         écart maximal : {pire:.3e}\n")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 11 — TRANSLATION, HOMOGÉNÉITÉ, TAPIS ÉGAUX, DÉNOMINATEURS NULS
# ═══════════════════════════════════════════════════════════════════════════


def section_gains(rng: random.Random, large: bool) -> bool:
    print("── 11. INVARIANTS DES GAINS ET CAS DÉGÉNÉRÉS")
    ok = True

    print("     (a) TRANSLATION : ajouter c à tous les gains ajoute c à toutes")
    print("         les équités — MAIS seulement si m ≥ n (tout le monde est")
    print("         payé). Sinon Σₖ pᵢₖ < 1 et dépend du joueur : la translation")
    print("         change alors la STRUCTURE, ce n'est pas un défaut.")
    for nom, stacks, payouts in REFERENCE:
        n, m = len(stacks), len(payouts)
        ref = icm_equities(stacks, payouts)
        pire = 0.0
        for c in (0.5, 7.0, 1000.0):
            got = icm_equities(stacks, [x + c for x in payouts])
            pire = max(pire, float(np.max(np.abs(got - (ref + c)))))
        if m >= n:
            ok &= _ligne(f"{nom}  (m={m} ≥ n={n})", f"{pire:.2e}",
                         pire <= 1e-9 * max(sum(payouts[:n]) + 1000.0, 1.0),
                         "invariant exigé")
        else:
            ok &= _ligne(f"{nom}  (m={m} < n={n})", f"{pire:.2e}",
                         pire > 1e-6, "NON invariant — attendu")

    print("         le facteur de bulle sous translation :")
    for nom, s, p, h, v in (("9j/9gains (m≥n)", TABLE_9, PAYOUTS_9, 3, 1),
                            ("4j/3gains (m<n)", (5000.0, 3000.0, 1500.0, 500.0),
                             (50.0, 30.0, 20.0), 3, 0)):
        vals = [bubble_factor(s, [x + c for x in p], h, v)
                for c in (0.0, 7.0, 1000.0)]
        etendue = max(vals) - min(vals)
        attendu_invariant = len(p) >= len(s)
        print(f"         {nom:<20s} BF = " + "  ".join(f"{x:.9f}" for x in vals))
        ok &= _ligne(f"BF invariant par translation ? {nom}", f"{etendue:.2e}",
                     (etendue <= 1e-9) is attendu_invariant,
                     "exigé" if attendu_invariant else "non exigé (m < n)")

    print("         …et quand m < n, la dérive a une LIMITE calculable")
    print("         ailleurs : c → ∞ rend les gains proportionnellement égaux,")
    print("         donc la structure tend vers un SATELLITE PUR. C'est une")
    print("         référence externe, pas une valeur recopiée du code.")
    for nom, s, p, h, v in (
            ("bulle 4j/3gains", (5000.0, 3000.0, 1500.0, 500.0),
             (50.0, 30.0, 20.0), 3, 0),
            ("6j/4gains", (176.0, 124.0, 92.0, 70.0, 55.0, 44.0),
             (100.0, 60.0, 40.0, 25.0), 5, 0)):
        limite = bubble_factor(s, (1.0,) * len(p), h, v)
        suite = [(c, bubble_factor(s, [x + c for x in p], h, v))
                 for c in (0.0, 1e3, 1e9)]
        for c, bf in suite:
            print(f"         {nom:<18s} c={c:<8g} BF = {bf:.9f}")
        print(f"         {nom:<18s} satellite pur = {limite:.9f}")
        ok &= _ligne(f"BF(c→∞) → satellite pur : {nom}",
                     f"{abs(suite[-1][1] - limite):.2e}",
                     abs(suite[-1][1] - limite) <= 1e-6 * limite,
                     "convergence vers une référence externe")

    print()
    print("     (b) HOMOGÉNÉITÉ : λ·gains ⇒ λ·équités, et BF inchangé.")
    print("         (le dollar n'a pas plus d'unité que le jeton)")
    pire_eq = pire_bf = 0.0
    for nom, stacks, payouts in REFERENCE:
        n = len(stacks)
        ref = icm_equities(stacks, payouts)
        for lam in (1e-6, 3.0, 1e6):
            got = icm_equities(stacks, [x * lam for x in payouts])
            pire_eq = max(pire_eq, float(np.max(np.abs(got - lam * ref)))
                          / (lam * max(sum(payouts[:n]), 1.0)))
        ok &= _ligne(nom, f"{pire_eq:.2e}", pire_eq <= 1e-12, "écart relatif")
    for nom, s, p, h, v in (("9j", TABLE_9, PAYOUTS_9, 3, 1),
                            ("4j/3gains", (5000.0, 3000.0, 1500.0, 500.0),
                             (50.0, 30.0, 20.0), 3, 0),
                            ("un tapis nul", (0.0, 3000.0, 1500.0, 500.0),
                             (50.0, 30.0, 20.0, 10.0), 3, 1)):
        b0 = bubble_factor(s, p, h, v)
        for lam in (1e-6, 3.0, 1e6):
            pire_bf = max(pire_bf,
                          abs(bubble_factor(s, [x * lam for x in p], h, v) - b0))
    ok &= _ligne("BF homogène de degré 0 en les gains", f"{pire_bf:.2e}",
                 pire_bf <= 1e-9, "")

    print()
    print("     (c) TAPIS TOUS ÉGAUX : chacun vaut la MOYENNE des gains")
    print("         attribuables, et le BF est le même pour TOUTES les paires.")
    for n, p in ((2, (50.0, 30.0, 20.0)), (3, (50.0, 30.0, 20.0)),
                 (5, (50.0, 30.0, 20.0)), (5, PAYOUTS_9), (9, PAYOUTS_9),
                 (4, (100.0, 60.0, 40.0, 25.0)), (6, (100.0,))):
        s = [1000.0] * n
        eq = icm_equities(s, p)
        moy = sum(p[:n]) / n
        d = float(np.max(np.abs(eq - moy)))
        vals = [bubble_factor(s, p, i, j)
                for i in range(n) for j in range(n) if i != j]
        disp = max(vals) - min(vals)
        ok &= _ligne(f"n={n}, m={len(p)}", f"{d:.2e}",
                     d <= 1e-11 * max(sum(p[:n]), 1.0) and disp <= 1e-9,
                     f"moyenne {moy:.4f}  BF {vals[0]:.9f}  dispersion {disp:.1e}")

    print()
    print("     (d) DÉNOMINATEURS NULS DU FACTEUR DE BULLE")
    print("         gain == 0 ⇒ +inf, et la chaîne complète doit tenir :")
    sp = IcmSpot(stacks=(1.0, 2.0, 3.0), payouts=(1.0, 1.0, 1.0),
                 hero=0, villain=1, pot=100.0, bet=50.0)
    a = analyse_icm_spot(sp, hero_equity=0.99)
    ok &= _ligne("analyse_icm_spot sur structure dégénérée",
                 f"α={a.alpha_icm:.3f}",
                 math.isinf(a.bubble) and a.alpha_icm == 1.0
                 and a.premium == 0.5,
                 "BF=+inf → prime 0,5, équité exigée 100 %")
    ok &= _ligne("risk_premium(+inf)", f"{risk_premium(math.inf)}",
                 risk_premium(math.inf) == 0.5, "")
    ok &= _ligne("icm_required_equity(·, ·, +inf)",
                 f"{icm_required_equity(100.0, 50.0, math.inf)}",
                 icm_required_equity(100.0, 50.0, math.inf) == 1.0, "")

    print("         et le BF ne rend JAMAIS zéro — sinon la chaîne casserait")
    print("         sur « bubble factor doit être > 0 » :")
    zeros = infs = 0
    mini = math.inf
    total = 0
    for _ in range(6000 if large else 2000):
        n = rng.randint(2, 7)
        s = [0.0 if rng.random() < 0.30 else rng.uniform(1, 1000)
             for _ in range(n)]
        if sum(s) <= 0:
            continue
        k = rng.randint(1, n + 1)
        base_p = [round(rng.uniform(0, 100), 1) for _ in range(k)]
        if rng.random() < 0.30:                 # force des ex æquo
            base_p = [base_p[0]] * k
        p = sorted(base_p, reverse=True)
        h, v = rng.sample(range(n), 2)
        if min(s[h], s[v]) <= 0:
            continue
        total += 1
        bf = bubble_factor(s, p, h, v)
        if bf == 0.0:
            zeros += 1
        elif math.isinf(bf):
            infs += 1
        else:
            mini = min(mini, bf)
    ok &= _ligne(f"balayage {total} spots (30 % de tapis nuls, ex æquo)",
                 f"{zeros} zéro(s)", zeros == 0,
                 f"{infs} dégénérés → +inf, min fini {mini:.12f}")
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 9 — LES DEUX SEUILS NUMÉRIQUES ET LEUR MARGE
# ═══════════════════════════════════════════════════════════════════════════


def section_seuils() -> bool:
    print("── 9. SEUILS NUMÉRIQUES  (d'où viennent les deux constantes 1e-12)")
    print()
    print("     (a) `bubble_factor` : quand un gain est-il du bruit ?")
    print("         Le garde-fou compare `gain` à 1e-12 × l'échelle des $EV.")
    print("         Il faut donc que le bruit soit BIEN au-dessous et le plus")
    print("         petit gain légitime BIEN au-dessus.")
    print()
    print("         bruit du cas dégénéré (tous payés pareil) :")
    for stacks, payouts in [((1.0, 2.0, 3.0), (1.0, 1.0, 1.0)),
                            ((100.0, 200.0, 300.0), (50.0, 50.0, 50.0)),
                            ((10.0, 20.0, 30.0, 40.0), (7.0,) * 4)]:
        eff = min(stacks[0], stacks[1])
        base = float(icm_equities(stacks, payouts)[0])
        w = list(stacks); w[0] += eff; w[1] -= eff
        gain = float(icm_equities(w, payouts)[0]) - base
        print(f"           {str(stacks):<26s} {str(payouts):<20s} "
              f"gain relatif = {abs(gain) / max(abs(base), 1e-300):.2e}")
    print()
    print("         plus petit gain LÉGITIME (heads-up, 1 jeton contre R) :")
    for ratio in (1e3, 1e6, 1e9, 1e12):
        stacks = [ratio, 1.0]
        payouts = [60.0, 40.0]
        base = float(icm_equities(stacks, payouts)[0])
        w = list(stacks); w[0] += 1.0; w[1] -= 1.0
        gain = float(icm_equities(w, payouts)[0]) - base
        print(f"           ratio 1:{ratio:<12.0f} gain relatif = "
              f"{abs(gain) / abs(base):.2e}")
    print()
    print("         → le seuil 1e-12 sépare ~1e-16 (bruit) de ~5e-13 (légitime")
    print("           au ratio le plus extrême encore physiquement pensable).")
    print()
    print("     (b) `analyse_pko_spot` : le vilain est-il éliminé ?")
    print("         Sans déclaration, le résidu est jugé devant SA transaction")
    print("         (SEUIL_RESIDU_TRANSACTION × max(pot, bet)) : un rapport")
    print("         sans dimension, donc invariant d'échelle — voir 8(c). Un")
    print("         seuil absolu faisait dépendre le verdict de l'unité ; le")
    print("         seuil relatif au TOTAL des tapis, remplacé le 14 août")
    print("         2026, comparait le résidu à une grandeur étrangère au")
    print("         `stack − bet` qui le produit (les bornes mesurées sont")
    print("         dans la docstring de la constante).")
    print()

    print("     (c) dégradation de l'égalité BF == 1 en heads-up")
    print("         (annulation catastrophique : les $EV se soustraient)")
    for ratio in (2, 10, 100, 1e3, 1e4, 1e6, 1e9):
        bf = bubble_factor([float(ratio), 1.0], [60.0, 40.0], 0, 1)
        print(f"         ratio 1:{ratio:<12.0f} BF = {bf:.15f}   "
              f"|BF−1| = {abs(bf - 1.0):.2e}")
    print("         → l'égalité tient à 1e-9 près jusqu'à des ratios de 1e6 ;")
    print("           au-delà, c'est la soustraction qui perd les chiffres,")
    print("           pas l'ICM. Les tests bornent en conséquence.\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large", action="store_true",
                    help="ajoute les balayages aléatoires larges (plus lent)")
    ap.add_argument("--seuils", action="store_true",
                    help="détaille les deux seuils numériques et leur marge")
    ap.add_argument("--continuite", action="store_true",
                    help="n'exécute QUE les sections 10 et 11 (itération rapide)")
    ap.add_argument("--graine", type=int, default=20260811)
    a = ap.parse_args()

    rng = random.Random(a.graine)
    print(f"banc des invariants ICM — graine {a.graine}, "
          f"{len(REFERENCE)} tables de référence, "
          f"facteurs d'échelle {FACTEURS}\n")

    if a.continuite:
        sections = (
            section_continuite(rng, a.large),
            section_gains(rng, a.large),
        )
    else:
        sections = (
            section_echelle(rng, a.large),
            section_permutation(rng, a.large),
            section_conservation(rng, a.large),
            section_monotonie(rng, a.large),
            section_non_linearite(rng, a.large),
            section_monte_carlo(rng, a.large),
            section_bubble(rng, a.large),
            section_pko(rng, a.large),
            section_continuite(rng, a.large),
            section_gains(rng, a.large),
        )
    if a.seuils:
        section_seuils()

    if all(sections):
        print("TOUS LES INVARIANTS TIENNENT.")
        return 0
    print("AU MOINS UN INVARIANT EST VIOLÉ — voir les lignes « ÉCHEC ».")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
