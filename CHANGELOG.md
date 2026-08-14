# Journal des versions — Poker Fusion Solver

## Non publié — 14 août 2026 (suite)

### Onglet Live — le conseil en direct, sur argent fictif prouvé uniquement

Le mode Live lit la table en continu (~1,2 s par cycle) et affiche le
verdict PENDANT la main — mais seulement derrière une **preuve positive**
d'argent fictif, et la frontière est câblée, pas déclarée.

- `pfs/app/mode_live.py` (nouveau) fait tout ce qui précède le conseil —
  résolution de la fenêtre (EnumWindows), capture, badge « PMU PLAY »
  (`badge_pmu`), décision du gate, lecture des cartes et des montants — et
  **n'importe aucun module de recommandation** (vérifiable au grep). Le
  conseiller n'est convoqué qu'APRÈS un verdict ``mode == "live"`` ET une
  lecture complète (2 cartes héros sûres + les quatre montants) ; sinon un
  refus explicite, jamais un conseil sur des chiffres devinés.
- Gate `ComplianceGate.profil_pmu_play()` : sur PMU PLAY, les tables
  affichent des euros fictifs et des marqueurs de tournoi — le jeu de
  signaux générique les mettrait toutes en REVIEW à perpétuité. La preuve
  devient donc l'**identité du client** (badge pixel) TOUJOURS combinée à
  l'armement manuel confirmé ; le titre et la devise sont journalisés, pas
  votants. Mémoire du badge de 60 s liée au couple (fenêtre, titre),
  purgée au moindre changement — fail-closed.
- Trois routes nouvelles (`live/armer` avec confirmation explicite exigée,
  `live/desarmer`, `live/table` à contrat clos) ; les routes historiques
  `live/fenetres` et `live/lire` restent SANS conseil, leur verrou
  (`test_live_sans_conseil.py`) intact.
- Onglet « Live (fictif) » : bandeau du gate toujours visible (vert
  « table fictive prouvée » / rouge « non prouvé »), sélection et armement
  de fenêtre, cartes en vignettes, montants lus ou refusés, verdict en
  tuiles avec infobulles — le même composant que ♠ Ma main.
- **`tests/test_live_conseil_gate.py`** (nouveau, 14 tests) verrouille le
  miroir inversé du verrou existant : non armé, sans badge, gate en panne,
  désarmé, titre changé, souvenir expiré, lecture insuffisante ⇒ jamais de
  conseil ; armé + badge ⇒ le conseil est EXACTEMENT ce que le conseiller
  recalcule (anti-fabrication).

### Sécurité — banc adverse de la porte Live/conseil

- `python/banc_contournements_live.py` étendu de 5 attaques (H–L) contre la
  frontière « `live/table` ne conseille que prouvé LIVE », chacune écrite
  pour de vrai, mesurée, puis retirée (sauvegarde SHA-256) : route sans
  garde de gate, `try/except` fail-open, badge menteur, titre réintégré
  comme corroboration, corroboration abandonnée. L'attaque du titre
  (renommer sa fenêtre « argent fictif ») était un **trou** — aucun test ne
  l'exerçait ; comblé, puis attrapé. Bilan : **13/13 contournements
  attrapés** (TÉMOIN, A–L), les deux fichiers de tests verts.

### Le tapis rayé « Twister Flash » se lit : source de candidates par couleur de fond

Une capture PMU PLAY « Twister Flash » (3-max, tapis vert rayé, 2194 × 1660)
rendait 0/3 : aucune boîte sur les deux cartes du héros, et les trois dos de
cartes adverses du haut de table promus « hero ». L'instrumentation étape par
étape (même méthodologie que le chantier QUIET_SIDES) a montré que le tapis
rayé était INNOCENT à la détection d'arêtes — ses diagonales restent sous le
seuil (densité d'arêtes du feutre nu : 0,000) et les quatre bords des cartes
portent chacun 4 à 6 segments verticaux. Le vrai coupable : la plaque
« Temps : 14 / 10 BB » coupe les DEUX arêtes de chaque carte à la même
hauteur, chaque paire est donc jugée « carte entière » et son rapport recalé
(0,830–0,908 sur les 60 paires rejouées) tombe dans le trou entre
MAX_RATIO = 0,82 et CUT_MIN_RATIO = 0,92 — un trou que les avatars de siège
(0,846–0,872) interdisent de refermer, et le chemin « carte coupée » ne
s'ouvre jamais faute d'arête qui se prolonge (contrairement à
`pmu_hero_tronque`, où une arête suit le bandeau).

La parade est colorimétrique, pas géométrique : sur le deck `pmu_solid`,
l'aplat EST la signature. `table_detector._solid_background_boxes` rend
candidate toute composante connexe d'une teinte EXACTE de `FAMILLES`
(tolérance 6 — celle de l'annotation des 57 captures ; aplats Twister mesurés
à ≤ 2 de la référence), aux proportions d'une carte entière ou tronquée par
le bas ([0,60 ; 1,10]), dans la moitié basse, remplissant ≥ 0,55 de sa boîte
(vraies cartes : 0,71–0,73, amas d'interface ≥ 400 px² : ≤ 0,41), de petit
côté ≥ 26 px (fragments de barre d'équité du crop KO : ≤ 13 ; plus petite
vraie carte du corpus : 52), et portant l'encre blanche d'un glyphe —
part de blanc dans [0,09 ; 0,40], vraies cartes mesurées 0,179–0,242,
poches d'aplat détachées par le symbole d'enseigne et fragments JPEG ≤ 0,004
ou ≥ 0,556. Cette dernière garde rend au module sa propriété « le JPEG perd
des cartes, il n'en invente pas » (sans elle : 7 fantômes à q75, 14 à q60 ;
avec : 0, et la tranche q75 revient exactement à 53/84). Chaque seuil est
posé au milieu d'un vide mesuré, chiffres dans les docstrings.

Rôles : « les cartes du héros sont les plus basses » n'autorise pas la
réciproque. `read_table` ne promeut plus la rangée la plus basse en « hero »
que si son bas dépasse HERO_BOTTOM_MIN = 0,60 × la hauteur de l'image —
seuil posé au milieu du vide mesuré sur tous les actifs du dépôt (plus haut
non-héros : 0,543, board synthétique et dos du siège bas ; plus bas héros :
0,658, la capture Twister elle-même). Sous le seuil, les cartes restent dans
« autres » et le board n'est plus contraint d'être au-dessus d'un héros.

Résultats, tous bancs rejoués :

    capture Twister (2194 × 1660)        avant         après
      cartes du héros localisées          0/2      2/2 (IoU 0,945 et 0,938)
      lues (chaîne complète)               —       Kd sure (écart 90,
                                                   marge 307), 8d sure
                                                   (écart 125, marge 67)
      dos adverses promus « hero »         3/3           0/3
    57 captures réelles (banc vérité)
      rappel de lecture                 243/258       243/258 (94,2 %)
      précision                          100 %         100 %
      cartes inventées                     0             0
      boîtes rendues                      370           370  (aucune nouvelle)
    banc synthétique (144 tables)
      localisation                      664/672       664/672
      boîtes fantômes                    0/986         0/986
      dos promus                         0/720         0/720
      rôles justes                      659/672       662/672  (le garde-fou
                                        de promotion rend son rôle au board
                                        des tables où la main est manquée)

Coût, meilleur de passes alternées : la capture Twister passe de 1,64 s à
1,78 s (+8,5 % : la source couleur coûte ~0,25 s sur 3,6 Mpx, partiellement
compensée par une borne d'élagage pur sur la fenêtre d'appariement — une
arête d'habillage de 1 355 px l'ouvrait à 1 492 colonnes pour 1 228
segments, or aucune boîte plus large que √(1,10 × 0,10 × aire) ne peut
survivre aux bornes d'aire et de rapport : 1,64 → 1,53 s à résultat
identique par construction). La table 900 × 560 de référence passe de
0,164 à 0,217 s. Le fond rayé, lui, n'explosait rien : le coût venait de
cette arête de cadre.

Actifs : découpe `tests/donnees/pmu_twister_feutre_raye.png` (540 × 400,
78 Ko, cartes + plaque + jeton donneur + rayures) et classe
`TestTwisterStripedFelt` (6 tests : localisation, rôles, lecture Kd/8d
affirmée, rien d'autre annoncé, ablation de la source, ablation du seuil de
promotion). Pas d'entrée dans `verite_captures.json` : ses `emplacements`
sont globaux et liés à la fenêtre 1899 × 1348 des 57 captures, la frame
Twister (2194 × 1660) n'y a pas sa place — le test dédié la couvre.

## v4.5.0 — 14 août 2026

Session « faire marcher le logiciel » : sept agents en parallèle sur les
calculs, la vision, le serveur et l'interface. L'état de référence a été
vérifié avant (suite verte + 19/19 goldens) et après (voir chaque section).

### La capture collée se lit en entier : cartes ET montants

`pfs/vision/zones_montants.py` (nouveau) vise les montants pour `digit_ocr`,
qui lit mais ne localise pas : cadres géométriques relatifs aux cartes
détectées, détection des lignes de texte par énergie de gradient, lecture de
lignes ENTIÈRES uniquement — mesuré ici même, une ligne coupée à mi-hauteur
se lit de travers avec une confiance parfois maximale (« 24,87 BB » tronqué
→ 87 à 0,66) ; les lignes qui touchent un bord de cadre ne sont jamais lues.
Pot contre-vérifié par l'étiquette du tas de jetons (désaccord = refus),
mise lue sur les boutons (« PAYER X » → X ; « CHECK » → 0 : un zéro LU, pas
supposé), tapis exigeant le suffixe « BB », blinde = 1 quand l'affichage est
en blindes. Sur les 57 captures réelles : pot 29, mise 22, tapis 50,
blinde 54, **zéro valeur inventée, zéro désaccord**, 71 ms médian. La route
`lire_capture` rend ces montants, l'onglet ♠ Ma main les préremplit
(marqués « image », jamais par-dessus une frappe) et déclenche le verdict
quand les quatre champs sont couverts : le flux « coller → verdict sans
saisie » existe. 16 tests dédiés + parcours réel au navigateur.

### Le verdict se lit d'un coup d'œil

L'onglet ♠ Ma main rend désormais la table en images : vignettes de cartes
(héros, board groupé et étiqueté Flop · Turn · River, dos de cartes pour
l'adversaire avec son profil), position rappelée (pastille BTN/SB/BB/UTG/
MP/CO + jeton « D » du donneur), verdict en tuiles (EV, équité, équité
requise, marge, cote du pot, MDF) et **frise d'évolution par street** —
l'équité recalculée par la même route `advise` sur les boards tronqués,
jamais recopiée en JavaScript. Chaque chiffre et chaque position porte une
infobulle en français simple tirée du lexique (une seule source). Le
simulateur réutilise les mêmes vignettes, y compris pour les cartes
adverses révélées. 0 erreur JS (audit navigateur).

### Le JPEG casse les garanties « zéro faux » — mesuré, puis refusé

Le chiffre de la passation (98,8 % → 88,1 % à q=75) venait du banc
synthétique. Rejoué sur les 57 captures réelles avec vérité-terrain
(recompression PIL) ::

    qualité   rappel lecture   précision   lectures fausses   rôles faux
    PNG           76,7 %        100,0 %           0                30
    q90           67,1 %        100,0 %           0                70
    q75           64,3 %         99,4 %           1                64
    q60           50,0 %        100,0 %           0                64

À q75, un 3h en retournement sort « 8h » affirmé, et les montants font
pire : deux tapis faux (3,79 BB lu 11,50 à confiance 0,79). L'écart
introduit (0,31–0,51 px de flou gaussien équivalent) reste SOUS la
frontière de 0,7 px de digit_ocr, qui ne se transpose donc pas au JPEG —
le ringing frappe les glyphes, pas le feutre. Conséquence câblée :
`lire_capture` REFUSE les JPEG (octets magiques FF D8 FF, avant tout
décodage) avec une erreur qui explique le recollage en PNG. Ces mesures
étaient le point 4 de la passation, resté sans commit jusqu'ici.

### Serveur : nodelock, rake et EQR atteignables ; routes documentées

- `/api/postflop` accepte `locks: [{path, strategy, combos?}]` (signature
  Pio, appliqué avant le solve — le non-verrouillé re-solve autour,
  vérifié à travers la route), `rake: {pct, cap}` (défaut aucun ;
  `expected_rake` publié ; bluffs ↑ et calls ↓ conformes aux formes
  closes), `leaf_model: "rollout"|"eqr"` et `nodes: [[path]]` (fréquences
  moyennes de n'importe quel nœud).
- `fusion/eqr` est BRANCHÉ (recommandation « S » de l'audit) :
  `train_eqr()` une fois, mémoïsé ; la réponse republie le R² et le n
  MESURÉS du modèle entraîné avec sa limite (valeur directionnelle, sans
  garantie d'écart au solve complet, somme des EV dérivante). Registre :
  `wired=True`, `library_only` 8 → 7.
- `/api/presets` et `/api/drill/next` : conservées avec statut documenté
  (outillage/CLI, aucune UI) ; `GET /api/health` documentée comme sonde
  manuelle sans jeton.

### Drills depuis les fuites, et l'interface qui se souvient

- **S'entraîner sur ses propres fuites.** L'onglet Mes sessions propose,
  après chaque revue, « M'entraîner sur mes fuites » : les erreurs préflop
  chiffrées par le solveur deviennent des exercices JAM/FOLD rejoués tels
  qu'ils ont été joués (cartes réelles, pot, ante), planifiés par la
  répétition espacée SM-2 — un spot raté revient immédiatement. Trois
  routes : `drill/fuites` (génération depuis un dossier d'historiques +
  résumé honnête du corpus, part non mesurée des limps comprise),
  `drill/fuites/next`, `drill/fuites/answer` (corrigé complet après la
  réponse ; l'énoncé, lui, ne souffle jamais rien).
  `pfs/train/leak_drills.py`, jusqu'ici orphelin, est branché sans
  modification. Vérifié sur le dossier PMU réel : 641 mains, 7 drills.
- **Détection des historiques PMU partagée.** `pfs/data/emplacements.py`
  (nouveau), importé par `recuperer_mains.py` ET la route `emplacements` —
  le champ « dossier d'historiques » se préremplit au premier lancement.
- **Persistance locale minimale.** Le dossier d'historiques et la fenêtre
  de calibration survivent au rechargement (localStorage). Les montants du
  spot ne sont JAMAIS persistés : des chiffres d'une vieille session
  rejoués en silence seraient un mensonge — un test structurel verrouille
  que rien d'autre ne s'écrit.
- Tests : `tests/test_parcours_fuites.py` (16 tests par les routes
  réelles) ; le garde « marque de saisie » reconnaît la lecture d'image
  comme deuxième source légitime à côté de la frappe.

### Détecteur de badge « PMU PLAY » (préparation du mode Live)

`pfs/vision/badge_pmu.py` : `detecter_badge(image)` lit deux marqueurs sur
une capture 1920×1361 — filigrane central ZNCC à position fixe par thème
(seuil 0,25 : positifs 0,417–1,000, occulté ≤ 0,086, négatifs ≤ 0,050) et
dos de carte orange ZNCC plein cadre sur le canal R−B, plancher de
variance σ ≥ 5 (seuil 0,80 : positifs ≥ 0,975, négatifs ≤ 0,569). Sur les
57 captures réelles : 50/57 frames portent une preuve immédiate, les 7
restantes sont des fins de main, 0 faux positif. Toute autre taille de
fenêtre est REFUSÉE (fail-closed). Le module ne décide rien d'éthique : il
rend une lecture ; le branchement au gate de conformité (mode Live sur
argent fictif uniquement) est le chantier suivant, conçu et documenté.
Sprites + métadonnées versionnés (`pfs/vision/templates/pmu_play/`),
`banc_badge_pmu.py` rejouable, 12 tests.

### Lexique : clés stables et vocabulaire du verdict (39 → 49 termes)

Chaque terme porte une clé stable et des alias (`PAR_CLE`,
`definir_par_cle()`, rétrocompatible — route et onglet inchangés). Dix
termes ajoutés : les six positions une par une (bouton/BTN, petite et
grosse blinde, UTG, MP/lojack/hijack, cutoff) dans une catégorie
« positions », plus mise à payer, cote du pot, marge et outs, chacun avec
un exemple chiffré en français simple. Termes clarifiés : équité, cote du
pot / équité requise scindées, prime de risque, street, range. 21 tests.

### Audit des orphelins : le registre ne ment plus par omission

- Docstrings « Statut » en tête de 10 modules testés mais jamais branchés
  (pourquoi, ce qui existe déjà, point d'accroche précis) : fusion/timing,
  topology, meanfield, eqr (branché depuis), core/bunching,
  solver/abstraction, isomorphism, analysis/inference_check,
  data/population, compliance/gate.
- `bench/solver_registry` : nouvel axe `wired` — `coverage` dit « le code
  existe et il est testé », `wired` dit « le joueur en profite ».
  `library_only()` liste les paramètres implémentés non branchés ; liste
  figée par test (TestWiring) : tout branchement futur doit mettre le
  registre à jour. La faiblesse périmée « Perception non encore livrée »
  du profil PFS est remplacée par le rappel mesuré.
- Suppression du paquet vide `pfs/perception/`.

### Plus aucun chemin personnel dans le code versionné

- `recuperer_mains.py` : dossiers PMU dérivés de `%LOCALAPPDATA%` (repli
  `~/AppData/Local`) ; iso-fonctionnalité mesurée : 789 mains / 65
  tournois avant ET après.
- `pfs/analysis/reperes.py` : `racine_corpus()` — variable `PFS_CORPUS`,
  sinon `<parent du dépôt>/corpus` ; `CORPUS_PLURIBUS` et
  `CORPUS_WSOP_2023` en dérivent. Table gelée et route `/api/reperes`
  inchangées (`--verifier` passe chiffre par chiffre).
- `banc_corpus_pluribus.py` et `tests/test_phh.py` : même résolution ;
  les tests sur corpus réel deviennent des SKIP documentés s'il est absent.

### Les bancs longs sont rejoués et leurs chiffres consignés

- `banc_calculs_exacts.py --long` : **68 vérifications conformes, 0 écart,
  831 s** (le premier passage a révélé et corrigé un plantage de format
  ndarray dans la borne exact-vs-Monte-Carlo du banc lui-même).
- `banc_corpus_pluribus.py`, rejeu complet des 10 000 mains : **accord
  78,0 % tous régimes (15 169 spots)**, 86,3 % préflop profond, 63–64 %
  postflop — chiffres désormais dans la docstring du banc avec leur
  lecture honnête (la famille dominante du désaccord postflop est la
  défense face à une ouverture, à laquelle la chart d'ouverture répond
  hors de son domaine ; cash 6-max 100 bb, rien ne valide le tournoi).
- Le selftest survit aux consoles Windows cp1252 (reconfiguration UTF-8
  avec remplacement) : il plantait AVANT de vérifier quoi que ce soit, ce
  qui ressemblait à un échec des calculs.


### Les deux défauts mathématiques du conseil des modèles (tour 4) sont clos

**1. Le seuil d'élimination PKO comparait le résidu du vilain aux tapis des
autres joueurs.** Le repli d'inférence de `_vilain_elimine`
(`python/pfs/core/icm.py`) jugeait « éliminé » un tapis résiduel sous
``1e-12 · Σ tapis`` — or la somme des tapis n'entre jamais dans le
``stack − bet`` qui produit ce résidu. Conséquence mesurée (exemple du
tour 4) : un jeton de saisie résiduel devant un all-in de 288 000, sur un
tournoi à 16,7 M de jetons, restait « vivant », la prime tombait à zéro en
silence et l'équité exigée montait de 36,6 % à 42,2 % — un chiffre faux et
plausible, du côté du fold. Le repli compare désormais le résidu à la
transaction qui l'a produit : ``SEUIL_RESIDU_TRANSACTION (1e-5) ·
max(pot, bet)``, un rapport sans dimension donc invariant d'échelle. Le
seuil est encadré par deux bornes mesurées — au-dessous, le bruit flottant
(~1e-12 de la transaction) et les artefacts d'un jeton de départ
(≤ 3,3e-6) ; au-dessus, le plus petit vrai jeton en circulation
(≥ 1/(100 bb⁻¹ · 300 bb) = 3,3e-5) — et 1e-5 en est la moyenne géométrique
(1,04e-5). Goldens à la main : winner-take-all 3-way à tapis égaux,
r* = 4/13 exactement quand la prime est en jeu, 1/2 exactement quand le
vilain garde un vrai jeton (`test_icm_ordre_et_elimination.py`). Sur la
table MTT réelle : résidu de 1e-3 ou 1 jeton → 49,95 % (comme le spot
propre) ; vrai tapis de 1 000 jetons → 53,57 %, sans prime, comme il se
doit. `villain_all_in` et `unite_jeton` priment toujours sur ce repli, et
la borne grise entre 3,3e-6 et 3,3e-5 reste indécidable sans eux — c'est
documenté dans la constante, pas caché.

**2. Les quatre goldens « avant » de l'ordre d'élimination étaient faux, pas
le code.** Les tests interrompus du chantier (écarts de 1,3e-5 à 3,9e-5 hors
tolérance) opposaient des valeurs enregistrées à mi-chantier au témoin
actuel du partage égal. Conformément à l'avertissement du chantier — ne
jamais élargir une tolérance — les deux candidats ont été départagés par un
recalcul INDÉPENDANT : Malmuth-Harville par énumération complète des
8! = 40 320 ordres d'arrivée, sans aucun import de `pfs.core.icm`.
L'énumération reproduit le code actuel à mieux que 1e-9 en relatif sur les
quatre cas (p. ex. héros 3 contre vilain 0 : 1,7140596405 des deux côtés,
contre 1,714123 enregistré à mi-chantier). Les goldens ont été corrigés
(1,9002683 ; 2,2613563 ; 1,9392982 ; 1,7140596 — et côté production
1,6118303 ; 2,0334389 ; 1,7084969 ; 1,4538859), et la convention est
ancrée par un calcul à la main vérifiable sans machine : sur
``[100, 100, 0, 0]`` avec 50/30/20/10, BF = (40−30)/(50−40) = 1 exactement
avec l'ordre déclaré, 2 exactement sous le partage égal.

### Déplacé

- **`chantiers_interrompus/test_icm_ordre_et_elimination.py` →
  `python/tests/`** — 51 tests, tous verts, protocole de mutation re-mesuré
  (A : 11 échecs ; B : 9 ; B′ : 12 ; C : 3 ; D : 3). Batterie ICM +
  conseiller + serveur : 700 tests, 0 échec, 2 sautés (préexistants).
  Selftest : 19/19 goldens conformes.

### Ce qui reste ouvert (et n'a pas été touché)

- La **marche aléatoire absorbante** — seul juge externe du biais de
  Harville ; le banc d'invariants ne vérifie que l'appartenance à la
  famille de modèles.
- La **dégénérescence DCFR** (`α = β = γ = 1` n'est pas CFR standard mais
  Linear CFR).
- Le **banc WSOP** et ses deux tests de tolérance préflop
  (`chantiers_interrompus/`).

## v4.5.0 · travaux du 11 août 2026

### Les « zones saines » de la revue de session étaient de la littérature

L'interface affichait « zone saine 28–50 % » pour le VPIP, « 8–24 % » pour le
3-bet, « 25–40 % » pour le WTSD. Aucun de ces chiffres n'était mesuré : ni
corpus, ni dénominateur, ni effectif. Ils sont remplacés par des valeurs
**recalculées** sur `phh-dataset` (licence MIT, université de Toronto) avec le
**même code de comptage** que celui qui mesure l'utilisateur
(`ParsedHand.stat_observations`) — c'est ce partage qui rend la comparaison
licite.

Sur les 10 000 mains de Pluribus, restreintes aux positions d'une table à
trois (30 000 mains-joueurs) :

| statistique | repère mesuré | occasions | Q1 · méd · Q3 (14 joueurs) |
|---|---|---|---|
| VPIP | **32,19 %** | 30 000 | 29,1 · 32,5 · 34,1 |
| PFR | **15,67 %** | 30 000 | 15,2 · 16,1 · 16,7 |
| écart VPIP−PFR | **16,52 pt** | 30 000 | 12,5 · 16,7 · 17,9 |
| 3-bet | **6,73 %** | 25 611 | 6,1 · 7,0 · 7,6 |
| fold to c-bet | **47,25 %** | 2 823 | 41,4 · 47,4 · 52,5 |
| WTSD | **14,69 %** | 16 014 | 13,5 · 14,8 · 15,6 |
| AF postflop | **1,95** | 2 443 suivis | 1,8 · 1,9 · 2,2 |

Sur la table à six complète (60 000 mains-joueurs) : VPIP 26,32 %, PFR
17,55 %, écart 8,77 pt, 3-bet 4,37 %, fold-to-cbet 46,87 %, WTSD 10,62 %,
AF 2,27.

**Ce que ces chiffres ne sont pas.** Pluribus joue du **cash 6-max à 100 bb
sans ante** : ce sont des repères d'**équilibre**, pas une moyenne de
population, et surtout **pas des repères de tournoi**. Le seul tournoi du
corpus — WSOP 2023 event #43 — ne contient que **18 mains de hold'em** sur 83
(championnat mixte : les 65 autres sont refusées par le lecteur PHH, pas lues
de travers), soit 90 mains-joueurs : l'intervalle de Wilson de son VPIP couvre
23,5–42,4 %. Il est publié avec ses effectifs et marqué non concluant ; rien
n'en est déduit.

**Le WTSD n'est pas comparable d'un format à l'autre** et l'API le dit
désormais champ par champ (`comparable: false` + la raison) : son dénominateur
compte tous les joueurs assis dès qu'un flop tombe, couchés préflop compris —
il baisse donc mécaniquement quand la table s'agrandit.

### Ajouté

- **`python/pfs/analysis/reperes.py`** — les repères mesurés, gelés dans le
  code (le corpus vit hors du dépôt) et recalculables à la commande.
  `situer()` place un profil face à un jeu et marque ce qui n'est pas
  comparable ; `jeu_par_defaut()` choisit le mélange de positions d'après la
  taille de table réellement jouée.
- **`python/banc_reperes_corpus.py`** — le banc rejouable : `--verifier`
  compare la table gelée au corpus champ par champ et sort en code 1 au
  premier écart, `--situer` place le profil de l'utilisateur, `--geler`
  réimprime le littéral. 17 s pour le rapport complet.
- **`python/tests/test_reperes_corpus.py`** — 25 tests : le calcul sur des
  mains PHH écrites à la main, la cohérence interne de la table gelée, le
  refus de conclure sur les WSOP, et le parcours HTTP réel module → route
  → page.
- **Route `/api/reperes`** et son bouton « Voir les repères mesurés » dans
  l'onglet *Mes sessions* : les repères sont consultables sans historiques,
  avec leur source, leurs effectifs et leurs limites.
- **AF postflop du héros** dans `HeroProfile` — la seule statistique de style
  que les repères donnaient et que le profil ne mesurait pas.

### Corrigé

- `pfs/lexique.py` — les entrées VPIP, PFR et WTSD citaient des fourchettes de
  manuel ; elles citent maintenant les repères mesurés et, pour le WTSD, la
  réserve sur son dénominateur.

### Le « 95 % » de lecture des cartes était faux : le vrai taux est 76,7 %

Le dépôt annonçait « 199 cartes lues sur 209, soit 95 % » sur 57 captures
réelles. Une revue externe a démonté la métrique, et elle avait raison sur les
deux points : le dénominateur ne comptait que les cartes que le détecteur
avait bien voulu trouver — le **rappel n'était pas mesuré** — et « lue avec
certitude » voulait dire « non refusée », faute de **vérité-terrain** : une
lecture fausse et affirmée comptait comme un succès.

Les 57 captures ont été annotées à la main, carte par carte. Contre ce relevé,
la chaîne de production rend :

| mesure | valeur |
|---|---|
| cartes réellement présentes | **258** (et non 209) |
| rappel de lecture | **76,7 %** (198/258) |
| dont bon rôle | **65,1 %** (168/258) |
| précision | **100 %** (199/199) |
| lectures fausses affirmées | **0** |
| cartes jamais localisées | **60** |
| rôles faux affirmés | **30** |

### Ajouté

- **`python/tests/donnees/verite_captures.json`** — la vérité-terrain :
  57 frames, 263 cartes relevées à l'œil, emplacements mesurés au pixel. Le
  relevé visuel et un masque de couleur exact sur les quatre aplats du jeu
  concordent sur les 263 cartes.
- **`python/banc_verite_captures.py`** — rejoue la chaîne de production sur
  les captures et rend rappel, précision, abstention, lectures fausses
  affirmées, cartes inventées et rôles faux. Option `--quiet-sides N` pour
  l'ablation qui chiffre la cause des cartes perdues.
- **`python/banc_mutations_verite.py`** — casse une par une les onze choses
  que les nouveaux tests protègent et exige qu'ils tombent. La première
  mutation refabrique exactement le « 95 % ».
- **`python/tests/test_verite_captures.py`** — cohérence du relevé
  (notation, doublons, board monotone, totaux) et arithmétique du banc.
- **`pfs.vision.live.lire_image()`** — le coeur de `lire_ecran` sans la
  capture d'écran, pour que le banc rejoue le code de production et non une
  copie.

### Corrigé

- **Une carte en cours de retournement pouvait être AFFIRMÉE, et fausse.**
  Le 6♣ du flop de `300_7-max_KO/0003` sortait « Kc », statut « sure », à un
  écart de 616 pour une marge de 33. Le contrôle de dispersion du lecteur à
  fond plein existait et refusait bien la découpe — mais son refus était
  **muet** : `identify_card_autour` passait la main au hachage, dont les 40
  cadrages finissaient par en trouver un sous le seuil. Le refus est
  désormais **franc** quand la teinte est celle d'une famille du jeu (donc :
  c'est une carte de ce jeu, partiellement recouverte), et la main n'est plus
  passée. Un garde-fou contournable en changeant de chemin n'en est pas un.

### Le rappel réel passe de 76,7 % à 94,2 % — le décor se déduit des abords (14 août)

Les 45 cartes du héros rejetées par la règle « 3 abords calmes sur 4 » sont
récupérées, **sans toucher à `QUIET_SIDES = 3`** : la densité d'un abord se
mesure désormais **décor déduit** (`table_detector._quiet_density`). La règle
est locale, et chacune de ses constantes est posée sur une mesure des deux
populations qu'elle sépare :

- un pixel d'abord est du **décor** si son run d'arêtes rectiligne traverse
  la bande de bout en bout (halo « KO » mesuré : runs de 105-127 px pour des
  bandes de 93-94 ; glyphes : p50 = 4 px) **et** porte une signature de
  rail : ruban couvrant ≥ 0,45 de la bande (mesuré 0,50 et 0,75, jugeable
  seulement à ≥ 8 px d'épaisseur) **ou** dépassement d'au moins 20 px
  au-delà de la bande (rail : 34 px ; arête d'une carte voisine : ≤ 12 px) ;
- planchers mesurés : bande d'au moins 40 px de long (l'arête du jeton
  « 1K », 38 px, fabriquait sinon une carte 16 × 21) et 4 px d'épaisseur
  (à 2 px, une seule ligne « remplit » la bande — 3 dos adverses entraient).

Mesures avant → après, les **deux bancs rejoués ensemble** cette fois :
rappel réel 76,7 % → **94,2 %** (243/258), bon rôle 65,1 % → **91,9 %**,
précision **100 %** inchangée, **0** carte inventée, **0** lecture fausse,
rôles faux 30 → 6 ; banc synthétique **rigoureusement identique** (664/672,
0 fantôme, 986 boîtes — aucun rail à y déduire). Effet de bord mesuré : un
libellé collé à 6 px sous les cartes ne coûte plus une seule carte
(664 → 652 avant, 664 → 664 après), le test du banc verrouille le nouveau
comportement. Le cas guéri est dans le dépôt
(`tests/donnees/pmu_ko_hero_rail.png`, découpe de `300_7-max_KO/0014`) avec
son test et son ablation. Trois pistes essayées et **annulées** sont
documentées dans `_quiet_density` (composantes connexes : 73 % des fantômes
candidats passaient ; « continue au-delà de la boîte » : le halo épouse la
carte ; veto du tiers central : la colonne de pips pontée traverse une vraie
carte).

### Diagnostiqué, non corrigé

- **15 cartes du board** disparaissent sous la pile de jetons du pot : le
  recalage horizontal accroche l'arête des jetons et le rapport (0,594) sort
  des bornes. Sur 3 captures, ce 9♣ manquant réduit le board visible à
  2 cartes : les 6♠/A♥ trouvés restent sans rôle — les **6 rôles faux
  résiduels** du banc vérité-terrain viennent de là.

## v4.4.0 — 10 août 2026

Calibration en direct : lire une vraie table à l'écran, sans jamais conseiller.

La boucle « je colle une capture → ça rate → je corrige → je recolle » coûtait
un aller-retour complet par image. Le logiciel lit maintenant la fenêtre du
client à l'instant présent et rend **ce qu'il y a vu**, carte par carte, avec
sa confiance. Une session de calibration donne des dizaines de lectures
réelles en quelques minutes.

### La frontière, inscrite dans le code

`pfs/vision/live.py` ne produit **aucune recommandation** : ni verdict, ni
équité, ni seuil de bascule. Lire son propre écran pour vérifier qu'un
programme reconnaît des images ne retire rien à personne ; recevoir une
recommandation calculée pendant une main d'argent réel la retire aux
adversaires, qui ignorent qu'ils affrontent une machine. Le module n'importe
donc aucun calculateur de décision, aucun champ de sortie ne nomme une
action, et `tests/test_live_sans_conseil.py` le vérifie à chaque exécution.
Le conseil reste disponible sur les mains **terminées** et à l'entraînement.

### Ajouté

- **`pfs/vision/live.py`** — capture (sonde Rust) → localisation →
  reconnaissance → archivage des échecs, en un appel. `lire_ecran()`,
  `capturer_fenetre()`, `fenetres_disponibles()`.
- **`calibrer.py`** — banc en console, avec mode `--boucle`.
- **Onglet « Calibration »** dans l'interface, avec un composant de couleur
  distinct de celui des verdicts : `.lecture` dit une **confiance de
  lecture**, jamais une action. Partager le composant aurait suggéré qu'une
  lecture sûre est un feu vert pour jouer.
- Routes `POST /api/live/fenetres` et `POST /api/live/lire`.

### Corrigé

- **Deux serveurs sur le même port.** `HTTPServer` active `allow_reuse_address`
  par défaut ; sur Windows cette option laisse un **second** processus se lier
  à un port déjà en écoute — vérifié en faisant écouter *trois* processus sur
  le même port, `netstat` affichant les trois. Le partage n'est pas
  équitable : le listener **le plus ancien capte la totalité** des connexions
  (12/12 puis 6/6), le suivant ne prenant le relais qu'à sa mort. Après chaque
  modification du code, un nouveau serveur démarrait donc pendant que l'ancien
  répondait à *tout*, et les routes fraîchement ajoutées renvoyaient « route
  inconnue » systématiquement — pas par intermittence, ce qui rendait le
  diagnostic trompeur. Le serveur utilise désormais `SO_EXCLUSIVEADDRUSE` et
  refuse de démarrer, avec un message actionnable, si le port est pris.
- **Cartes inventées avec aplomb.** La première lecture en direct a produit
  une carte « 4h » avec le statut **sure** sur une découpe de *décor*. La
  règle de confiance ne regardait que la **marge** — l'avance du meilleur
  gabarit sur le deuxième — sans exiger que ce gabarit *ressemble* à l'image.
  Sur une découpe qui n'est pas une carte, le classement des gabarits est
  arbitraire, et l'écart entre le premier et le deuxième l'est autant. Mesure
  sur 552 échantillons : les vraies cartes cadrées sur feutre s'échelonnent
  de 251 à 599, les non-cartes (bruit, feutre, dos, jetons) de 658 à 790 —
  un vide franc où ne tombe aucun des deux nuages. `DISTANCE_SURE = 625` s'y
  place : 100 % des vraies cartes conservées, 100 % des fausses rejetées. Un
  premier essai à 520 a été mesuré puis abandonné, il coupait au milieu des
  vraies cartes (40/52 → 12/52 sur feutre vert). Une carte masquée au tiers
  par le HUD tombe à 688, donc côté « non-carte » — c'est le bon
  comportement : masquée, elle ne doit pas être affirmée.
- **Chemin d'archive mensonger.** Le logiciel annonçait
  `%LOCALAPPDATA%\PokerFusionSolver\captures`, un dossier qui **n'existe pas**
  sur le disque. L'interpréteur du projet dérive d'un Python Microsoft Store
  (`sys.base_prefix` sous `C:\Program Files\WindowsApps\...`) : Windows
  redirige silencieusement les écritures vers le `LocalCache` du paquet, tout
  en laissant `os.environ["LOCALAPPDATA"]` et `os.path.abspath` afficher le
  chemin d'origine. `Test-Path` répondait `False` là où Python voyait ses
  fichiers. `dossier_archive()` résout désormais le chemin réel, copiable dans
  l'explorateur. Aucun test ne pouvait le détecter : tous remplacent
  `LOCALAPPDATA` par un dossier temporaire, ce qui court-circuite la
  virtualisation — d'où `tests/test_archive_chemin_reel.py`, qui travaille
  dans l'environnement réel.
- **Échecs d'archive qui s'écrasaient.** Les noms de fichiers étant horodatés
  à la seconde, les huit découpes d'une même lecture de table se recouvraient :
  l'archive ne gardait qu'une découpe par seconde et par statut, en silence.
- **Fenêtre figée jamais capturée.** Windows Graphics Capture n'émet une image
  que lorsque la fenêtre se redessine — une table entre deux actions restait
  muette jusqu'au timeout. La cible est maintenant réveillée par
  `RedrawWindow`, sans être ni déplacée ni activée.

### Mesuré

Banc rejouable : `python banc_localisation.py --large` (54 tables décorées
2560×1529 par configuration). Il n'existait pas dans une première version de
cette entrée, et les chiffres publiés étaient donc invérifiables — deux
d'entre eux étaient faux.

- **La localisation se fait à pleine échelle.** C'est la seule largeur qui
  tient partout : **100 % de localisation et de rôles dans les cinq
  configurations** (habillage plein, deck classique, images bruitées, cartes
  de 52×70 et de 80×108). Réduire à 1280 avant de chercher ne coûte rien sur
  la famille de tables qui avait servi à la première mesure, mais fait tomber
  la localisation à **56,0 % sur le deck classique**, 70,2 % en 52×70,
  79,8 % avec du bruit. 1280 n'est même pas monotone : 960 y fait mieux
  (65,5 %). Le réglage reste accessible par paramètre pour le seul cas qui le
  justifie — une fenêtre de bureau très chargée.
- Contrepartie assumée : la pleine échelle produit **~0,83 boîte fantôme par
  table** (45 sur 54), contre 0 à 1280. Toutes tombent dans `others`, aucune
  n'est promue carte du héros ou du board, et les échecs de `others` ne sont
  plus archivés — sinon le banc se remplirait de découpes de feutre et de dos
  d'adversaires, illisibles par nature.
- Coût : ~700 ms pour localiser une table à pleine échelle, ~230 ms à 1280.
  Hors capture, la chaîne complète coûte ~300 ms sur une table 2030×1271
  (décodage 10 ms, localisation ~220 ms, reconnaissance de 6 boîtes ~45 ms).

### Corrigé après coup dans cette même entrée

Trois chiffres publiés ici étaient faux, et une décision de conception en
découlait :

- « 48 tables, 100 % de localisation à 1280 comme à pleine échelle » — le
  banc en comptait 18, et le 100 % à 1280 ne valait que pour une seule
  famille de tables.
- « 32 boîtes fantômes à pleine échelle » — le compte est proportionnel au
  nombre de tables (~0,83/table) : 15 sur 18 tables, 45 sur 54. 32 ne
  correspond à aucun banc.
- « 1280 est un optimum, pas un compromis » — faux, voir ci-dessus.
- La justification « en dessous, les arêtes de carte passent sous le plancher
  de 14 px » était fausse aussi : à 640 px la carte fait encore 23 px de
  haut. La cause de l'effondrement n'est pas isolée.

### Non mesuré, assumé

La précision de reconnaissance sur une **vraie table de room**. La chaîne a
été validée de bout en bout, mais sur une table synthétique dont les cartes
ne sont pas celles du deck PMU — le taux de lecture observé (0/8) ne dit rien
de la performance réelle. C'est la première chose à faire à la prochaine
session.

## v4.3 — 9-10 août 2026

Version intermédiaire restée sans entrée de journal : l'état complet, les
pièges et les priorités de l'époque sont consignés dans `PASSATION_v43.md`
et `PLAN_v43.md` à la racine du dépôt. (Entrée ajoutée le 14 août 2026 pour
que le journal soit continu.)

## v4.2.0 — 8 août 2026

Reconnaissance de cartes — la limite annoncée en v4.1 est levée.

### Ajouté

- **`pfs/vision`** — reconnaissance de cartes à partir d'images.
  - `phash.py` : hash perceptuel DCT 256 bits (bloc 16×16), robuste à
    l'échelle et au bruit. Le bloc 8×8 initial ne séparait pique et trèfle
    de même rang que de 2 bits ; 16×16 porte la séparation à 30 bits.
  - `card_recognizer.py` : `identify_card`, `recognize_cards` (par ROI),
    `build_templates` (re-calibrage sur un autre thème/room en une commande),
    seuils de confiance calés sur les distances mesurées.
  - `templates/pmu_deck/` : les 52 cartes du client PMU, extraites de
    `PokerCommonWidgetsQRC.rcc` et **étiquetées + vérifiées visuellement**
    (complétude 13×4 + planche de contrôle) ; `templates/pmu_phash.json`
    signatures pré-calculées.
  - Route `POST /api/recognize` ; outil console `reconnaitre.py`
    (image → cartes, et image → cartes → conseil avec le contexte du spot).
- Dépendance `pillow>=10` (module vision).

### Mesuré

- Auto-reconnaissance 52/52 ; séparation ≥ 30 bits entre cartes distinctes ;
  échelle ×3/×5/×8 : 52/52 ; bruit (flou + σ10) : 52/52, pire distance
  correcte 18 (seuil d'acceptation 55). Flux complet image → cartes → conseil
  validé de bout en bout.

### Tests

- 718 → **731 tests verts** (+13, `test_vision.py`), 19 goldens inchangés.

### À caler encore

- Coordonnées des régions d'intérêt (position des cartes sur la table) :
  dépendent de la room et de la résolution, à mesurer une fois sur une vraie
  capture. Le recogniseur, lui, est prêt.

---

## v4.1.0 — 8 août 2026

Session Claude Code (Fable 5) sur le PC Windows de Pierre. Reprise de la
passation v4.0, installation vérifiée, puis livraison de la **suite d'analyse
post-partie** et du **GO de faisabilité Phase 1**.

### Ajouté

- **Parseur iPoker/PMU** (`pfs/data/hand_history.parse_ipoker`) — format XML de
  PMU, partypoker, Betclic, Unibet. Codes d'action décodés empiriquement et
  vérifiés contre l'attribut de contrôle `player@bet` ; all-in détecté par
  `@bet ≥ tapis` ; pot = somme des gains (exact). Conservation des jetons
  283/283 sur le corpus réel de Pierre. Correction de `went_to_showdown`
  (basé sur les folds, pas la présence de cartes — sinon WTSD=100 % en iPoker).
- **Revue de session** (`pfs/analysis/session_review`) — profil exact
  (VPIP, PFR, 3-bet, fold-to-cbet, WTSD, net en bb) et équité des tapis
  « all-in adjusted » (équité au moment du tapis vs cartes adverses connues,
  réalisé − espéré = variance). Route `/api/review`.
- **Revue shove/fold** (`pfs/analysis/pushfold_review`) — décisions préflop de
  tapis court confrontées à l'équilibre de Nash du solveur, écart chiffré en bb.
  Détection du spot heads-up au moment de la décision (pas sur les sièges
  distribués). Route `/api/review/pushfold`.
- **Conseiller de spot** (`pfs/analysis/spot_advisor`) — `advise(Spot)` : Nash
  exact en tapis court, chart d'ouverture en profond, équité exacte + cotes du
  pot en postflop avec seuil de bascule et hypothèse de range déclarée.
  `parse_cards` tolérant (« AhKd », « A♠ K♦ », « 10h »). Route `/api/advise`.
- **Outil console** `analyser_main.py` — une main (`--hero`/`--board`/…), mode
  rafale (une main par ligne), journal des mains analysées + `--recap`.
- **Perception Phase 1** — sonde de faisabilité `rust/crates/pfs-capture`
  (`probe.exe`, `--auto` multi-rooms, `--snap`, `--timeout`) et harnais
  d'occultation `scripts/perception/`. GO mesuré : capture d'une fenêtre
  occultée, ROI 200×100 en p95 < 1 ms, validé sur le client PMU réel.

### Corrigé

- `rust/Cargo.toml` — `optional = true` interdit dans `[workspace.dependencies]`
  (le workspace n'avait jamais compilé). `run.bat` — priorité au `.venv` du
  dépôt (le `python` du PATH de la machine est un venv tiers sans numpy).

### Performance

- **Équité exacte ~2× plus rapide** : la main du héros ne dépend que du runout,
  elle est désormais évaluée une fois par runout (990 au flop) au lieu d'une
  fois par couple (runout, combo) (169 290). Le tableau one-hot de `evaluate7`
  n'est plus construit deux fois. Flop 0,58 s → 0,30 s, **résultats identiques
  au bit près** (42 cas de référence vérifiés champ par champ).
- Mémoïsation du solve Nash (par 0,1 bb) et des ranges adverses dans le
  conseiller. Préflop 0,01 s → 0,00 s.
- Test de non-régression `tests/test_equity_perf.py` : correction vs oracle
  naïf (1e-12) + garde-fou anti-retour du travail redondant.

### Tests

- 670 → **718 tests verts**, 19 goldens au selftest inchangés.
- Nouveaux : `test_ipoker_hh`, `test_session_review`, `test_pushfold_review`,
  `test_spot_advisor`, `test_equity_perf`.

### Éthique / périmètre (inchangé, réaffirmé)

- Refus maintenu de toute assistance en direct sur argent réel (préjudice aux
  autres joueurs par la tromperie). La suite livrée n'analyse que des mains
  **terminées** — étude, comme un tracker adossé à un solveur.

---

## v4.0.0 — 8 août 2026 (passation)

14 fusions, solveur NLHE réel (river exacte, turn par énumération), 96 % des
standards industriels, nodelock 2.0, push/fold ICM, bunching, et le
différenciateur `engine.resolve_spot` (re-solve depuis la range inférée).
670 tests. Voir `PASSATION.md`.
