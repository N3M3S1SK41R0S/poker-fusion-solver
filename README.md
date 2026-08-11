# ♠ Poker Fusion Solver

Assistant poker personnel : **analyse a posteriori** de ses propres mains,
**conseil sur une main terminée** à partir d'une capture d'écran, et
**entraînement**. Application 100 % locale, en français.

**1 095 tests verts · zéro dépendance réseau · zéro € de licence.**

> ## État réel — à lire avant tout
>
> Ce qui suit est vérifié au 11 août 2026, et ce paragraphe existe parce que
> la version précédente de ce README promettait davantage que ce que le
> logiciel exécute.
>
> **Ce qui marche et se mesure :** la lecture des historiques PMU (721 mains
> réelles récupérées sur 62 tournois), la revue de session avec séparation
> variance / jeu, la revue push/fold contre Nash, le simulateur, les drills.
>
> **Ce qui marche À MOITIÉ, et le chiffre honnête :** la lecture des cartes
> sur capture réelle lit **76,7 % des cartes réellement présentes**
> (198/258 sur 57 captures PMU annotées à la main), dont **65,1 % avec le bon
> rôle**. Ce que le recogniseur affirme est juste — précision 199/199, zéro
> lecture fausse affirmée — mais **60 cartes ne sont jamais vues**, dont les
> deux cartes du héros sur presque toute une des deux tables. Le README a
> longtemps annoncé « 95 % » : ce chiffre était calculé sur les seules cartes
> que le détecteur avait bien voulu trouver, sans vérité-terrain. Voir
> [Reconnaissance de cartes](#reconnaissance-de-cartes-pfsvision).
>
> **Ce qui existe mais ne s'exécute jamais :** **17 modules sur 62 ne sont
> importés par aucune chaîne partant de `python -m pfs`** (2 313 lignes sur
> 9 428), et **21 modules ne sont traversés par aucun parcours utilisateur**.
> Les deux nombres sont différents parce qu'ils mesurent deux choses
> différentes — le détail et les bancs sont dans
> [« Trois nombres, trois méthodes »](#trois-nombres-trois-méthodes).
> En particulier :
> * la **chaîne de fusion** (`pfs/fusion/`, 13 modules) —
>   `FusionEngine.decide()` n'est appelée que par `demo.py` et les tests.
>   Les fusions ne participent à **aucune** décision rendue à l'utilisateur ;
> * l'**OCR des montants** (`pfs/vision/digit_ocr.py`), qui sait lire les
>   tapis et le pot à 99,5 % sur son banc, mais n'est branché nulle part —
>   d'où la saisie manuelle des montants après un collage.
>
> Le détail, les mesures et les défauts corrigés récemment sont dans
> [`BRIEF_CONSEIL_DES_MODELES.md`](BRIEF_CONSEIL_DES_MODELES.md), rédigé pour
> une revue critique externe. C'est le document le plus à jour du dépôt.
>
> **Ce que le logiciel ne fait pas, et ne fera pas :** aucune assistance en
> temps réel pendant une partie d'argent réel. La limite est inscrite dans le
> code (`pfs/vision/live.py`) et vérifiée par
> `tests/test_live_sans_conseil.py`.

v4.2 — **reconnaissance de cartes** (`pfs/vision`) : à partir d'une capture,
le recogniseur identifie les cartes par hash perceptuel (pHash DCT) sur le deck
du client PMU, extrait et étiqueté (52 cartes vérifiées). Robuste à l'échelle
(un gabarit 15×20 reconnaît une carte affichée bien plus grande — testé jusqu'à
×8), robuste au bruit, séparation de 30 bits entre cartes distinctes. Le flux
complet image → cartes → conseil est câblé (`reconnaitre.py`). Re-calibrable sur
un autre thème/room en une commande (`build_templates`).

v4.1 — la suite d'analyse post-partie, adossée aux solveurs du cœur :

* **Parseur iPoker/PMU** (`data/hand_history.parse_ipoker`) — le format XML de PMU,
  partypoker, Betclic, Unibet ; décodé et prouvé sur mains réelles (conservation
  des jetons vérifiée).
* **Revue de session** (`analysis/session_review`) — profil statistique exact
  (VPIP, PFR, 3-bet, fold-to-cbet, WTSD, net en bb) + équité des tapis
  « all-in adjusted » qui sépare le niveau de jeu de la variance.
* **Revue shove/fold** (`analysis/pushfold_review`) — chaque décision préflop de
  tapis court confrontée à l'équilibre de Nash, écart chiffré en bb.
* **Conseiller de spot** (`analysis/spot_advisor`, `analyser_main.py`) — « qu'est-ce
  qu'il fallait faire ? » à partir de ce qu'une capture montre : tes cartes, le
  board, le pot, la mise. Nash exact en tapis court, équité exacte + cotes du pot
  en postflop, seuil de bascule toujours donné.
* **Perception Phase 1 — GO** : la faisabilité de la capture d'écran est **tranchée
  et mesurée** (fenêtre occultée, lecture ROI p95 < 1 ms) — voir `rust/` et
  `scripts/perception/`.

v4.0 — 96 % des standards industriels, les signatures des concurrents intégrées (nodelock 2.0, merge, bunching, push/fold ICM, profondeur limitée), et LE différenciateur : re-solve depuis la range inférée en direct (`engine.resolve_spot`, `/api/resolve`). v3.0 — les 7 leviers du benchmark : **solveur postflop NLHE réel** (river exacte range-vs-range, turn par énumération — validé sur solution analytique), équité multiway, PKO, FGS léger, tells temporels, population mining privé, EQR apprise de nos propres solves.

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
cd python && pytest -q     # 718 tests
```

### Analyser tes mains déjà jouées (sans le serveur)

```bash
cd python
# une main, comme sur une capture :
python analyser_main.py --hero "Ah Kd" --board "Qs 7d 2c" --pot 100 --bet 75 --bb 10
# en rafale (une main par ligne, format court) :
python analyser_main.py
#   Ah Kd | Qs 7d 2c | pot 100 | bet 75 | bb 10
python analyser_main.py --recap        # synthèse des mains analysées

# revue complète d'un dossier d'historiques PMU :
python -c "from pfs.analysis import review_folder; \
  print(review_folder(r'%LocalAppData%\PMU PLAY 100%% Poker\data\<pseudo>\History').explain())"
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
`data/hand_history.py` (parseurs Winamax / PokerStars / **iPoker-PMU XML**),
`engine.py` (orchestrateur), `compliance/gate.py` (gate multi-signaux fail-closed),
`app/` (serveur + interface).

---

## La suite d'analyse post-partie (v4.1)

Étude a posteriori de tes propres mains — jamais en direct pendant le jeu. Tout est
adossé aux solveurs du cœur, donc chaque verdict est calculé, pas énoncé.

| Module | Ce qu'il fait | Route API |
|---|---|---|
| `data/hand_history.parse_ipoker` | parse le XML PMU/iPoker (N mains/session), all-in exact via `@bet`, conservation des jetons vérifiée | — |
| `analysis/session_review` | profil VPIP/PFR/3-bet/fold-cbet/WTSD + équité des tapis « all-in adjusted » | `/api/review` |
| `analysis/pushfold_review` | décisions préflop tapis court vs Nash jam/fold, écart en bb | `/api/review/pushfold` |
| `analysis/spot_advisor` | « que fallait-il faire ? » depuis un spot (cartes, board, pot, mise) | `/api/advise` |
| `vision/card_recognizer` | reconnaît les cartes d'une image (pHash sur le deck PMU) | `/api/recognize` |
| `analyser_main.py` | outil console : une main, en rafale, ou `--recap` du journal | — |
| `reconnaitre.py` | outil console : image → cartes → conseil | — |

**Deux niveaux de certitude, toujours annoncés.** *Certain* quand l'équilibre de Nash
push/fold tranche (tapis court — vérité de théorie des jeux). *Indicatif* en postflop,
où l'équité est **exacte** mais la range adverse est une hypothèse : le conseiller
donne alors le **seuil de bascule** (à partir de quelle lecture la décision
s'inverse) plutôt qu'un verdict péremptoire.

Latence (serveur chaud) : préflop 6 ms · river 5 ms · turn 17 ms · flop ~300 ms.

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

## Trois nombres, trois méthodes

Une version antérieure de ce README rapprochait deux chiffres : 31,7 % de code
mort par recensement, 31,7 % de couverture par `coverage.py`, « obtenus par
deux méthodes entièrement différentes ». **Ce rapprochement était faux, et il
est retiré.**

Il l'était pour deux raisons, l'une arithmétique et l'autre factuelle.

* **Arithmétique.** 31,7 % de lignes *traversées* implique 68,3 % de lignes
  *non traversées*. Pour qu'une couverture confirme un taux de mortalité de
  31,7 %, il aurait fallu lire **68,3 %** de couverture. Le chiffre obtenu
  était l'exact **complément** de celui qu'il était censé confirmer : les deux
  31,7 % ne pointaient pas le même ensemble de lignes, ils pointaient les deux
  moitiés complémentaires du fichier.
* **Factuelle.** Le banc de couverture trouve **20 modules jamais atteints sur
  61** (21 sur 62 à la dernière mesure) là où le recensement à la main en
  annonçait 12. La couverture ne confirmait pas le recensement : elle le
  **révisait à la hausse de 67 %**.

C'est le même travers que le « 97 % des pertes viennent des décisions » déjà
corrigé ici : une conclusion plus forte que la mesure qui la porte. À la
place, trois nombres, trois bancs, trois questions distinctes.

| | question | banc rejouable | mesure du 11 août 2026 |
|---|---|---|---|
| **(a)** | quels modules **aucun `import`** ne relie au point d'entrée ? | `python banc_atteignabilite_statique.py` | **17 modules sur 62** · 2 313 lignes sur 9 428 (**24,5 %**) |
| **(b)** | quelles lignes un **parcours utilisateur réel** ne traverse pas ? | `python banc_couverture_parcours.py` | **6 515 lignes non traversées sur 9 428 (69,1 %)** · 21 modules à zéro |
| **(c)** | quels calculs sont **traversés mais sans effet** sur la sortie ? | `python banc_inertie_causale.py` | **0 symbole inerte sur 13** (réponse entière) · **7 sur 13** si l'on n'observe que le verdict |

**Ces trois nombres ne sont ni égaux, ni comparables, et aucun ne valide les
autres.** Ils ne portent même pas sur les mêmes objets : (a) et (b) comptent
des lignes, (c) compte des symboles. Chacun est un minorant du gaspillage,
mesuré sous une hypothèse différente.

Les valeurs absolues bougent à chaque commit — le dépôt a gagné un module
pendant la rédaction de cette section, et les trois nombres ont bougé avec.
**C'est le banc qui fait foi, pas ce tableau** : rejouez-le.

**(a) n'est pas contenu dans (b), et l'écart se mesure.** On croise les deux
ensembles avec :

```bash
cd python
python banc_couverture_parcours.py --json couverture.json
python banc_atteignabilite_statique.py --croiser couverture.json
```

Croisement du 11 août 2026 : **13 modules dans les deux ensembles**, **1 dans
(a) seulement**, **8 dans (b) seulement**. Chacun déborde de l'autre, **dans
les deux sens** — ce qu'aucune confirmation mutuelle ne pourrait produire.

* **(b) seulement — chargé, jamais exécuté.** `pfs/engine.py` : le serveur
  l'importe dans la route `resolve`, donc le banc statique le déclare vivant ;
  mais l'import est **différé** dans le corps de la route, et rien ne le charge
  tant que personne n'appelle `/api/resolve`. Le parcours ne le traverse pas
  d'une ligne (0 sur 200). Charger n'est pas exécuter, et (a) ne sait pas faire
  la différence.
* **(a) seulement — mort pour l'application, vivant pour la mesure.**
  `pfs/vision/synth_table.py` : aucune chaîne d'import partant de
  `python -m pfs` ne le charge, et il ressort couvert à 95,0 %, parce que ce
  sont les **tests** qui l'importent pour fabriquer des tables synthétiques.

Chaque banc a raison sur sa question ; c'est le rapprochement qui aurait tort.

**(c) est la catégorie que ni (a) ni (b) ne voient.** Un calcul peut être
importé, traversé, couvert — et son résultat jeté. C'était le cas de
`/api/advise`, qui construisait `stacks` et `payouts` puis les abandonnait :
le régime ICM devenait inatteignable et le verdict rendu sur une bulle était
l'**inverse** du bon, sous une couverture verte. Ni l'analyse statique ni la
couverture ne peuvent voir ce défaut : elles mesurent le passage, pas la
causalité. `banc_inertie_causale.py` le mesure par **perturbation** — on
modifie une entrée, et si la sortie ne bouge pas, le calcul qui la consommait
est inerte.

Preuve que ce banc n'est pas décoratif : `--defaut stacks` réinstalle le
défaut d'origine, et le banc classe alors **3 champs sur 12** causalement
inertes (`stacks`, `payouts`, `ante`) au lieu de 0. Sans le défaut, aucun
champ du formulaire n'est inerte.

**La sortie observée fait partie de la mesure.** `--sortie action` ne compare
que le verdict, pas l'explication : sous cette lecture, **7 symboles sur 13**
sont inertes — perturber le bubble factor déplace le texte du conseil sans
renverser la décision sur ces six spots. Ce n'est pas un défaut du code, c'est
la distance des spots à leur seuil de bascule ; le nombre mesure la batterie
autant que le logiciel, et se cite avec elle.

**Ce que (c) ne dit pas :** il ne rend aucun pourcentage de lignes, ne couvre
qu'une route sur 32, et son verdict « inerte » vaut pour la batterie de six
spots utilisée — une absence d'effet observée n'est pas une preuve d'absence
d'effet. Il a d'ailleurs commencé par se tromper lui-même : la mémoïsation
`lru_cache` masquait deux symboles, rapportés « jamais appelés » alors qu'ils
sont au cœur du chemin. Les limites complètes sont dans le docstring du banc.

`python/tests/test_trois_nombres_distincts.py` échoue si ce README réaffirme
une équivalence entre ces trois nombres.

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

## Reconnaissance de cartes (`pfs/vision`)

```bash
cd python
python reconnaitre.py --card carte.png            # une carte
python reconnaitre.py --image table.png --rois "120,300,60,80; 190,300,60,80"
# image -> cartes -> conseil (ROI héros puis board) :
python reconnaitre.py --image table.png \
    --hero-rois "..." --board-rois "..." --pot 100 --bet 75 --bb 10
```

pHash DCT 256 bits sur le deck PMU étiqueté (`vision/templates/pmu_deck`, 52 cartes
vérifiées). Mesuré : auto-reconnaissance 52/52, séparation ≥ 30 bits entre cartes
distinctes, robuste à l'échelle (×3/×5/×8 : 52/52) et au bruit. Re-calibrage sur un
autre thème : `build_templates(dossier)` sur des `<carte>.png`.

### Ce que la chaîne lit vraiment, sur de vraies captures

Les chiffres ci-dessus sont ceux du **recogniseur seul**, sur des découpes déjà
cadrées. Ils ne disent rien de la chaîne complète. Celle-ci est mesurée contre une
**vérité-terrain relevée à la main** sur 57 captures de deux tables PMU
(`python/tests/donnees/verite_captures.json`, 263 cartes) :

```bash
cd python
python banc_verite_captures.py                  # les vrais chiffres
python banc_verite_captures.py --detail         # une ligne par capture
python banc_verite_captures.py --quiet-sides 2  # ablation, voir plus bas
python banc_mutations_verite.py                 # les tests tombent-ils ?
```

| mesure | valeur | ce qu'elle veut dire |
|---|---|---|
| cartes réellement présentes | **258** | 112 héros + 146 board (+ 5 en transition) |
| **rappel de localisation** | **76,7 %** (198/258) | ce que le détecteur voit |
| **rappel de lecture** | **76,7 %** (198/258) | lu, juste, et affirmé — **le taux à annoncer** |
| dont **bon rôle** | **65,1 %** (168/258) | héros/board correctement distingués |
| **précision** | **100 %** (199/199) | ce qui est affirmé est juste |
| abstention | 38,8 % (126/325 boîtes) | refus assumés |
| **lectures fausses affirmées** | **0** | le chiffre qui doit rester nul |
| cartes jamais localisées | **60** | l'information qui manquait |
| **rôles faux affirmés** | **30** | défaut ouvert (voir ci-dessous) |

**Le README a annoncé « 199 cartes lues sur 209, soit 95 % ». Ce chiffre était
faux deux fois** : le dénominateur ne comptait que les cartes que le détecteur
avait trouvées (le rappel n'existait pas), et « lue avec certitude » voulait dire
« non refusée » (une lecture fausse et affirmée comptait comme un succès).

**Où sont les 60 cartes perdues**, mesuré et non supposé :

* **45 cartes du héros**, presque toutes sur la table 7-max. Ce n'est ni une
  occultation ni un siège vide : la carte est parfaitement visible, ses deux
  arêtes verticales sont trouvées, le recalage donne le bon rapport (1,017). C'est
  la règle « **3 abords calmes sur 4** » (`table_detector.QUIET_SIDES`) qui les
  rejette — l'habillage « KO » de cette table pose un rail lumineux juste à côté du
  siège, et une pastille de prime « 2,25 € » à cheval sur la carte de gauche. Cette
  règle avait été calibrée sur des tables **synthétiques**, au feutre uni ;
* **15 cartes du board** (5♠ ×11, 9♣ ×4) : la pile de jetons du pot est posée sur
  leur bas, le recalage part sur une mauvaise arête horizontale et le rapport sort
  des bornes.

**Les 30 rôles faux en découlent** : `read_table` prend la rangée la plus basse
pour la main du héros ; quand le siège n'est pas détecté, c'est le board qui hérite
du rôle. Pour un solveur, une carte du board présentée comme carte du héros vaut
une carte fausse.

**L'ablation qui chiffre l'enjeu**, `--quiet-sides 2` sur les mêmes captures :
rappel **76,7 % → 96,9 %**, rôles justes **65,1 % → 96,9 %**, **zéro carte
inventée**, 8 cartes encore perdues (celles sous les jetons). Sur les 144 tables
synthétiques de `test_table_detector.py`, la même ablation coûtait 1,9 % de boîtes
fantômes. Le desserrement n'est **pas** appliqué : il demande son propre chantier,
mesuré sur les deux bancs à la fois.

**Ce qui reste à caler** : les **coordonnées des régions d'intérêt** pour le mode
`reconnaitre.py --rois` dépendent de la room et de la résolution. Le détecteur
automatique (`read_table`) s'en passe, au prix mesuré ci-dessus.

## Phase 1 — perception (capture temps réel)

La **faisabilité** de la capture d'écran est **tranchée et mesurée** sur la machine
cible (Windows 11, DPI 150 %) : la sonde `rust/crates/pfs-capture/src/bin/probe.rs`
(WGC, `windows-capture` 2.0.0) capture une fenêtre **occultée** et lit une région
d'intérêt 200×100 en **p95 445 µs** (proxy) / **245 µs** (client PMU réel) — sous le
budget 1 ms. Harnais : `scripts/perception/probe_occulte.ps1` ; `probe.exe --auto`
détecte le client (PMU, Winamax, PokerStars, partypoker…).

**Ce que ce paquet ne fait PAS** : aucune assistance en direct pendant une partie
d'argent réel. La suite d'analyse et la reconnaissance ne servent que des mains
**terminées** (captures a posteriori) — c'est un choix assumé, pas une limite
technique.

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
