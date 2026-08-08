# PASSATION — Poker Fusion Solver v4.0

**De : Claude (session Cowork cloud, 6–8 août 2026) → À : Claude Code (fable 5), et à tout agent qui reprend ce projet.**

Salut. Tu reprends un projet en excellent état de marche : **670 tests verts, 19 goldens au selftest, zéro dépendance réseau**. Ce document te donne tout ce qu'il faut pour travailler AVEC le travail déjà fait, pas contre lui. Lis-le en entier avant de toucher au code — les pièges listés en §6 ont chacun coûté une session de débogage.

---

## 0. Le projet en trois phrases

Assistant de poker NL Hold'em **strictement personnel** pour Pierre : application locale (Python, stdlib http.server, 127.0.0.1 + jeton), qui fusionne 14 théories mathématiques établies (F1–F14) pour modéliser l'adversaire individuellement en direct — ce qu'aucun solveur commercial ni IA de recherche ne fait. Le différenciateur final est LIVRÉ : `engine.resolve_spot` re-solve le sous-jeu river contre la range **inférée** par le filtre particulaire, gain mesuré +2,4 à +6,8 jetons contre un vilain biaisé. Il reste deux grandes batailles : la **perception** (Rust/Windows, capture d'écran — le go/no-go du live) et le **flop complet** (les 1 755 classes isomorphes sont prêtes).

## 1. Démarrage en 90 secondes

```bash
cd python
python -m pytest tests/ -q --ignore=tests/test_pushfold.py   # ~2 min, 637 verts
python -m pytest tests/test_pushfold.py -q                   # 1re fois : ~4 min (construit le cache matrice 169×169), ensuite 1,5 s
python -m pfs --selftest                                     # 19 goldens, ~2 s
python -m pfs                                                # l'application → 127.0.0.1:8731
```

Mémoire projet Claude (claude.ai, projet « Poker assistant ») : `claude/Etat_du_projet_et_decisions_ouvertes.md` (point de reprise), `claude/Logiciel_v2.3_livre.md` (état livré), `claude/BENCHMARK_v1_Solveurs_de_reference.md` (positionnement), `claude/PLAN_DIRECTEUR_Poker_Fusion_Solver_v2.md` (plan long terme). **Mets-les à jour à chaque session — c'est la mémoire commune.**

## 2. Carte du code (python/pfs/)

| Module | Rôle | Points d'entrée |
|---|---|---|
| `core/range_model.py` | LA brique : 1326↔169, poids mixtes, blockers, presets | `parse_range`, `Range` |
| `core/equity.py` | Évaluateur 7 cartes vectorisé + équité exacte/MC + multiway | `evaluate7`, `equity_vs_range`, `equity_multiway` |
| `core/icm.py` | F14 : Harville, bubble factor, **PKO**, **FGS léger** | `analyse_icm_spot`, `analyse_pko_spot`, `fgs_equities` |
| `core/rake.py`, `core/bunching.py` | Rake %+cap ; retrait des folds | `RakeModel`, `apply_bunching` |
| `solver/postflop.py` | **Le solveur NLHE** : river exacte, turn, profondeur limitée, nodelock, arbre complet | `PostflopSolver`, `lock_node`, `simplify_report` |
| `solver/pushfold.py` | Push/fold Nash ICM (style HRC), matrice 169×169 en cache | `solve_hu_pushfold` |
| `solver/isomorphism.py`, `solver/abstraction.py` | 1 755 flops (Burnside) ; buckets EHS distribution-aware | — |
| `solver/dcfr.py` | Le banc d'essai Kuhn (oracle −1/18) | `DCFRSolver` |
| `fusion/` | F1 dynamic_beta · F2 hmm · F3 particle · F4 bet_sizing · F5 geometry · F6 bottleneck · F7 topology · F13 arbiter · skill_prior · **timing (L2)** · **eqr (L7)** | — |
| `engine.py` | L'orchestrateur + **`resolve_spot` (LE différenciateur)** | `FusionEngine.resolve_spot` |
| `data/` | Parseurs HH (Winamax/PS, straddle), notes joueurs (2 verrous SharkScope), **population mining** | — |
| `bench/solver_registry.py` | Le benchmark EXÉCUTABLE : 84 paramètres, couverture testée | `coverage_report()` |
| `app/server.py`, `app/ui.html` | 19 routes API + UI 8 onglets | `python -m pfs` |
| `../rust/` | Squelette Phase 1 (perception WGC) — **pas encore construit** | — |

## 3. Les contrats non négociables (le style du projet)

1. **Français partout** (docstrings NumPy, messages, UI). Théories citées (auteur, année). Zéro placeholder.
2. **Toute affirmation est un test.** Le registre du benchmark VÉRIFIE que chaque module cité existe (`test_covered_modules_resolve`). Les valeurs golden sont calculées à la main dans les docstrings de test. Si tu ajoutes une capacité, ajoute son test AVANT de la déclarer au registre.
3. **Zéro réseau dans le cœur** — testé (`test_module_makes_no_network_call`). L'enrichissement SharkScope passe par `EnrichmentQueue` (2 verrous : hors main + participants seulement — décision de Pierre, ne pas défaire).
4. **Honnêteté NEMESIS** : chaque livraison inclut ses limites chiffrées. On écrit « PARTIEL assumé » plutôt qu'un mensonge de couverture. Les approximations sont documentées avec leur erreur mesurée.
5. **Auditer l'UI au navigateur** (Playwright headless, chromium dans /opt/pw-browsers) après tout changement d'`ui.html` : 0 erreur JS exigé.

## 4. Ce qui vient d'être fait (v3.0 → v4.0, 8 août)

- **P1 arbre complet** : `oop_bet_fracs`/`ip_bet_fracs`/`raise_fracs`/`allin_threshold`/`add_allin`.
- **P2 nodelock** : `lock_node(path, strategy, combos=)` — dict uniforme OU tableau (n_actions, n_combos) ; le non-verrouillé re-solve (nodelock 2.0). `unlock_all`.
- **P3 profondeur limitée** : `leaf_model="rollout"` (somme-exacte) / `"eqr"` (valeur apprise L7, DIRECTIONNELLE — la somme des EV dérive, c'est documenté et testé comme tel).
- **P4 LE différenciateur** : `engine.resolve_spot` + route `/api/resolve`. Gain d'inférence = EV(re-solve) − EV(σ_GTO figée via nodelock, vilain inféré re-solvant autour). λ tempère (mélange inférée⊕préset), repli préset sous ESS 0,25.
- **Agents parallèles** : straddle (parseur), abstraction EHS (mémoïsée par classe isomorphe — ×12,6), bunching (MC jointe + pairwise corr 0,995), push/fold ICM (jam 58,7 % @10bb, réf ~58).
- **Registre : 96,25 % des standards industriels.** Restants ASSUMÉS : PLO/Short Deck (autre jeu, hors périmètre), Ressources GPU (Phase 2).

## 5. Les prochaines batailles, dans l'ordre recommandé

1. **Phase 1 — perception (Rust/Windows)** : LE go/no-go du projet. Test de faisabilité : capturer une fenêtre Winamax **occultée** avec `windows-capture` 2.0.0, p95 < 1 ms. Nécessite le PC Windows de Pierre (impossible depuis un conteneur Linux). Le harnais de validation existe déjà : `data/hand_history.py` reconstruit chaque main → mesure gratuite et continue de la précision du scraper. Les timestamps alimenteront `fusion/timing.py` (tout est prêt à consommer).
2. **Boucle live complète** : perception → `engine.observe_action`/`observe_stat` → `resolve_spot` à chaque river. Le pipeline est câblé, il manque les yeux.
3. **Phase 2 — flop complet** : blueprint sur les 1 755 classes isomorphes × buckets EHS (`solver/abstraction.py`), compression, LMDB, GPU wgpu/cudarc. `postflop.py` s'étend au flop en ajoutant une street au constructeur d'arbre (la machinerie chance-node existe).
4. **Calibrations** : GTO_PRESETS sont des charts approximatives (remplace par tes propres solves) ; matrices HMM non calibrées (Baum-Welch dès ≥ 50 k mains réelles) ; EQR : réentraîner sur plus de spots (R² 0,55 sur 32 échantillons river seulement).
5. **Courbes UI** (convergence solveur, trajectoires bankroll — la seule faiblesse esthétique reconnue).

## 6. ⚠️ Les pièges qui ont déjà mordu (ne pas ré-apprendre)

1. **Chance node turn : diviser par 44** (52−4−2−2), pas 46 — sinon la somme des EV ≠ pot. C'est testé ; si tu ajoutes le flop, le même raisonnement donne 45×44/2 par confrontation aux tirages turn+river.
2. **`a[i,j] |= v` NumPy perd des bits sur indices dupliqués** (écriture fantaisie non bufferisée). Toujours passer par réductions one-hot ou `np.logical_or.at`.
3. **Exploitabilité : BR par ENSEMBLE D'INFORMATION**, jamais par état — sinon l'adversaire est clairvoyant et l'exploitabilité stagne (0,277 sur Kuhn).
4. **Le rake fait MONTER les bluffs à l'équilibre** (β* = b/net(P+2b)) et effondrer les calls. L'intuition inverse est fausse — vérifié en forme close.
5. **Kalman sur observations binaires = faux** (variance dépendante de l'état). C'est pour ça que F1 est Beta-Binomiale dynamique.
6. **TDA et tells : toujours un test de significativité** (permutation/Wilson) — sinon machine à exploits fantômes.
7. **`parse_range` ne connaît pas « AK » ni « random »** : écrire « AKs,AKo », `Range.full()`.
8. **Sur le board 2s2d7h8hKc, KK fait un FULL** — vérifie tes boards de test avant d'accuser le solveur (il avait raison, deux fois).
9. **CUSUM (k=0.5, h=5) : ARL₀ ≈ 470** — une alarme occasionnelle sous H₀ est NORMALE, ne « corrige » pas ça.
10. **pushfold : la matrice d'équité se construit une fois** (~4 min, cache `.npy` versionné dans solver/). Ne pas la mettre dans le selftest.
11. **Les conventions de pot diffèrent** : `IcmSpot.pot` = pot AVANT la mise adverse ; `PkoSpot.pot` = mise adverse INCLUSE. La route `/api/icm` fait la conversion (pot+bet pour PKO). Ne pas unifier sans migrer les goldens.
12. **Modes de jeu** : le live argent réel viole les CGU des rooms (décision : privé + argent fictif, stacks en BB). Le gate existe (`compliance/gate`), Pierre a demandé de ne PAS le sur-investir.

## 7. Comment on travaille avec Pierre

Réponses en français, denses, structurées NEMESIS (modélisation → calculs explicites → validation/limites → implémentation → critique). Il veut le maximum d'un coup (« chef d'œuvre »), les alternatives chiffrées, zéro question inutile — mais il tranche vite quand on lui présente des options claires (D1–D6 dans l'état projet). Livre les fichiers à chaque itération (zip versionné + benchmark HTML), sauvegarde tout en mémoire projet, et mets à jour l'artefact desktop `pfs-benchmark-solveurs`.

## 8. État Git

Ce dépôt est initialisé avec l'historique de la v4.0 (commit initial complet). Le remote GitHub de Pierre reste à configurer :

```bash
git remote add origin https://github.com/<pierre>/poker-fusion-solver.git
git push -u origin main
```

Bon jeu. Le plus dur (le cœur mathématique, le solveur, le différenciateur) est fait et verrouillé par les tests — ta mission est de lui donner des yeux (Phase 1) et de la profondeur (Phase 2).

— Claude, 8 août 2026
