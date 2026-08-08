# ♠ Poker Fusion Solver v4.0

Assistant poker personnel — **14 fusions mathématiques**, une application locale complète.

**670 tests verts · zéro dépendance réseau · zéro € de licence.**

v4.0 — 96 % des standards industriels, les signatures des concurrents intégrées (nodelock 2.0, merge, bunching, push/fold ICM, profondeur limitée), et LE différenciateur : re-solve depuis la range inférée en direct (`engine.resolve_spot`, `/api/resolve`). v3.0 — les 7 leviers du benchmark : **solveur postflop NLHE réel** (river exacte range-vs-range, turn par énumération — validé sur solution analytique), équité multiway, PKO, FGS léger, tells temporels, population mining privé, EQR apprise de nos propres solves. Benchmark exécutable : 84 % des standards industriels couverts, 14 différenciateurs à zéro concurrent, plus aucune lacune ICM.

---

## Lancer

```bash
# Windows
run.bat

# Linux / macOS
./run.sh

# ou directement
cd python && uv sync --extra dev && uv run python -m pfs
```

L'interface s'ouvre sur `http://127.0.0.1:8731`. Le serveur écoute sur **la boucle
locale uniquement**, avec un jeton aléatoire régénéré à chaque démarrage.

```bash
python -m pfs --selftest   # rejoue les 19 valeurs golden du Plan Directeur
python -m pfs --demo       # les 14 fusions en console
python -m pfs --no-browser --port 9000
cd python && uv run pytest -q   # 472+ tests
```

---

## Les 8 modules de l'application

| Onglet | Ce qu'il fait | Fusions |
|---|---|---|
| **Study** | Grille 13×13 · GTO · règles IB · **équité exacte/MC + multiway** | F5, F6, L3 |
| **Fusion** | **De combien s'écarter du GTO** — λ dérivé, ρ, borne de sûreté | F1, F2, F13 |
| **Train** | Drill à répétition espacée SM-2 · score · leaks · état cognitif | F12 |
| **Sizing** | Lagrangien EV + λ·IG · modèle MDF · bluff-catch | F4, F10 |
| **Analyze** | Import hand-history Winamax / PokerStars · stats avec IC95 | F1 |
| **Solveur** | **River/turn NLHE range-vs-range** (rake, validé sur solution analytique) + banc Kuhn | F8, L1, L8 |
| **Bankroll & ICM** | Ruine, Kelly, drawdown, shot · **ICM, bubble factor, PKO, FGS léger** | F9, F14, L4, L5 |
| **État mental** | HMM 3 états en ligne · pouvoir diagnostique par action | F2 |

---

## Les 13 fusions

| # | Module | Théorie source | État |
|---|---|---|---|
| **F1** | `fusion/dynamic_beta.py` | Kalman (1960) → **West & Harrison (1997)** DGLM | ✅ ⚠ corrige la v1 |
| **F2** | `fusion/hmm.py` | Baum & Petrie (1966) · Rabiner (1989) | ✅ + Baum-Welch |
| **F3** | `fusion/particle.py` | Gordon-Salmond-Smith (1993) · Doucet (2001) | ✅ |
| **F4** | `fusion/bet_sizing.py` | Shannon (1948) · Berger (1971) · Frazier (2008) | ✅ ⚠ corrige la v1 |
| **F5** | `fusion/geometry.py` | Rao (1945) · Amari (1985, 1998) | ✅ |
| **F6** | `fusion/bottleneck.py` | **Tishby-Pereira-Bialek (1999)** · Strouse-Schwab (2017) | ✅ |
| **F7** | `fusion/topology.py` | Edelsbrunner (2002) · Chazal-Michel (2021) | ✅ + test de permutation |
| **F8** | `solver/dcfr.py` | Brown-Sandholm (2019) + **Zhang-McAleer-Sandholm (2024)** | ✅ ⚠ corrige la v1 |
| **F9** | `core/bankroll.py` | Kelly (1956) · Malmuth · **Peters (2019)** · Doob | ✅ |
| **F10** | `core/bluffcatch.py` | Green & Swets (1966) | ✅ |
| **F11** | `fusion/meanfield.py` | Lasry & Lions (2007) · Carmona-Delarue (2018) | ✅ + auto-critique |
| **F12** | `train/drill.py` | Wozniak SM-2 (1990) · Kahneman-Tversky (1979) | ✅ |
| **F13** | `fusion/arbiter.py` | **Ganzfried & Sandholm (2015)** | ✅ **le cœur** |
| — | `fusion/skill_prior.py` | rétrécissement James-Stein sur ratings externes | ✅ SharkScope / OPR |

Plus : `core/range_model.py` (algèbre 1326 ↔ 169, blockers, parsing solveur),
`data/hand_history.py` (parseurs Winamax / PokerStars), `engine.py` (orchestrateur),
`compliance/gate.py` (gate multi-signaux fail-closed), `app/` (serveur + interface).

---

## Les quatre corrections mathématiques

**F1 — le Kalman gaussien sur des observations binaires était invalide.**
Variance d'observation dépendante de l'état, état non contraint à [0,1], approximation
là où le conjugué Beta-Bernoulli existe en forme close. Remplacé par un Beta-Binomial
dynamique à facteur d'oubli (West & Harrison, ch. 14).

**F4 — maximiser le gain d'information sans contrainte d'EV recommande l'all-in.**
Un all-in polarise maximalement la réponse. Remplacé par
`b* = argmax_b [EV(b) + λ·I(b)]`, λ = prix d'un bit en bb.

**F8 — l'exploitabilité calculée avec un adversaire clairvoyant ne converge jamais.**
Le joueur qui calcule la meilleure réponse doit choisir une action **par ensemble
d'information**, pas par état. Sinon elle stagne à 0,277 même à l'équilibre exact.
Corrigée : 2,12e-2 → 8,18e-3 en 600 itérations.

**F7 — la TDA sans test de significativité fabrique des exploits.** Une boucle H₁
persistante apparaît dans du bruit pur. Protocole imposé : 1 000 permutations,
p < 0,01, correction de Bonferroni. Sur du bruit, le module ne déclare **jamais** rien.

---

## Résultats mesurés

**F6 — compression de range.** Sur des ranges à fréquences mixtes (comme les vraies
solutions de solveur) :

| position | largeur | mixtes | 8 règles | 12 règles | 16 règles |
|---|---|---|---|---|---|
| UTG | 9,3 % | 6 | **91 %** | 91 % | 91 % |
| MP | 12,3 % | 9 | 79 % | **88 %** | 88 % |
| CO | 21,4 % | 12 | 71 % | 83 % | **87 %** |
| BTN | 41,4 % | 11 | 78 % | 82 % | **87 %** |
| SB | 55,0 % | 10 | 72 % | 81 % | **85 %** |

Sur des ranges **binarisées**, UTG atteint 95,5 % en 4 règles. L'écart mesure
exactement ce que coûtent les stratégies mixtes en complexité — ce que la
communauté observe empiriquement, l'Information Bottleneck le quantifie.

**F8 — HS-DCFR bat DCFR nu à budget égal.** Kuhn, 600 itérations : 8,18e-3 contre
9,41e-3. Valeur du jeu −0,056153 pour un Nash exact à −1/18 = −0,055556.

**F13 — l'arbitrage se durcit avec l'échantillon.** Villain à 75 % de fold-to-cbet :
20 mains → reste GTO (65,8 %) · 200 mains → exploite (75,0 %) · 600 mains → 75,0 %.
Solveur pur 62 %, exploitatif naïf 88 %.

**F2 — le HMM détecte le tilt avant les stats.** Une action improbable fait passer
P(tilt) de 5,4 % à 24,1 % — ×4,8.

**F5 — Fisher-Rao contre euclidien.** 0,001→0,002 donne `d = 0,0262` ;
0,500→0,501 donne `0,0020`. Rapport **×13**, distance euclidienne identique.

---

## Golden tests

`python -m pfs --selftest` rejoue les valeurs qui lient le plan au code :

| Grandeur | Valeur |
|---|---|
| RoR(μ=5, σ=100, B=3000) | `0.0497870684` |
| Bankroll pour RoR 1 % | `4605.17 bb` |
| θ̂, σ (14 VPIP / 40 mains) | `0.3536585`, `0.0737733` |
| HMM après un 3-bet improbable | `[0.4728, 0.2864, 0.2409]` |
| d Fisher-Rao | `0.306904480` |
| P(call est +EV) | `0.671639` |
| Valeur de Kuhn | `−1/18` |

---

## Ce qui n'est PAS dans ce paquet

**La capture d'écran temps réel (Phase 1).** C'est du code Windows natif
(`windows-capture` 2.0.0, WGC), qui ne peut être ni compilé ni testé ailleurs que
sur Windows, et qui doit être **calibré contre ton client réel** — ROI, thème, DPI.
Les crates Rust sont pré-structurées dans `rust/crates/pfs-capture`.

Tout le reste — les 13 fusions, l'orchestrateur, les parseurs, l'application —
tourne et est testé.

**Prochaine étape, semaine 3 :** capturer une fenêtre Winamax *occultée* avec
`windows-capture`, p95 < 1 ms. C'est le test de faisabilité du projet entier.
Le harnais de validation existe déjà : `pfs.data.hand_history` est l'oracle.

---

## SharkScope, OPR et les ratings externes

Le logiciel **n'interroge aucun site** — jamais, par construction.

Ces bases donnent des **résultats** (ROI, volume, ABI), pas des **fréquences**
(VPIP, fold-to-cbet). Aucun ROI ne dit à quelle fréquence un joueur folde face à
un c-bet : ils ne peuvent donc pas alimenter F13 directement.

Ce qu'ils peuvent légitimement faire, et que `fusion/skill_prior.py` implémente :
déformer le **prior d'archétype** du filtre particulaire (F3) et la **propension
à s'adapter** ρ de F13.

### L'enrichissement automatique, avec deux verrous

`data/player_notes.py` implémente le pattern du blueprint appliqué aux
adversaires : **enrichissement différé hors ligne, lookup local en direct**.

```
HORS MAIN                          EN DIRECT (< 1 ms, zéro réseau)
  file différée  ──►  base locale  ──►  pseudo lu à l'écran → hash → lookup O(1)
                       chiffrée                    │
                                                   ▼
                                    prior d'archétype (F3) + ρ (F13)
```

`EnrichmentQueue` applique deux verrous, quel que soit le fournisseur :

* **Verrou (a) — hors main.** Rien ne part tant qu'une main est vivante à une
  table quelconque. Casse la corrélation temporelle entre board distribué et
  requête sortante, qui est le mécanisme du Fair Play Check.
* **Verrou (b) — participation.** Un pseudo n'entre dans la file que si ce
  joueur a été **dans un pot avec toi**. Un spectateur est refusé, et le refus
  est tracé. C'est l'implémentation littérale de la clause PokerStars.

Plus : cadence minimale entre deux traitements, et **journal d'audit local** de
tout ce qui a été tenté, accepté ou refusé.

Le noyau reste sans réseau. Brancher un fournisseur est explicite : tu
implémentes le protocole `RatingProvider` avec ta propre clé d'abonnement.
`ManualProvider` (hors ligne) est fourni ; import CSV également.

**Le prior s'efface tout seul** : poids 100 % à la première main, 50 % à
50 mains, 6 % à 200 mains. Ce que tu observes toi-même prédit toujours mieux un
fold-to-cbet qu'un ROI de tournoi.

Le module applique d'abord un **rétrécissement bayésien**, parce que le ROI est
une statistique bien plus bruitée qu'on ne le croit (σ ≈ 150 % par MTT) :

| ROI affiché | tournois nécessaires pour le prouver |
|---|---|
| +30 % | 197 |
| +20 % | 442 |
| **+10 %** | **1 766** |
| +5 % | 7 064 |
| +2 % | 44 150 |

Exemple : « +40 % de ROI sur 200 MTT » devient **+14,5 % ± 6,4** après
rétrécissement — **64 % du chiffre affiché est du bruit.**

---

## Confidentialité

* écoute sur **127.0.0.1 uniquement**, jeton aléatoire par démarrage ;
* **aucune dépendance réseau** — le serveur n'utilise que `http.server` ;
* pseudos hachés BLAKE2b + sel local, jamais stockés en clair ;
* aucune synchronisation, aucune télémétrie, aucun solveur cloud.

---
*Plan Directeur v2.0 + Addendum v2.1 — 6 août 2026*
