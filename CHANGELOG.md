# Journal des versions — Poker Fusion Solver

## Non publié — 11 août 2026

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

### Diagnostiqué, non corrigé

- **45 des 60 cartes perdues sont rejetées par `QUIET_SIDES = 3`** — la règle
  « 3 abords calmes sur 4 », calibrée sur des tables synthétiques au feutre
  uni. L'habillage « KO » de la table 7-max borde le siège d'un rail lumineux
  et pose une pastille de prime sur la carte de gauche. À 2, le rappel réel
  passe à 96,9 % sans une seule carte inventée, mais fait entrer 1,9 % de
  fantômes sur le banc synthétique. Chantier suivant.
- **15 cartes du board** disparaissent sous la pile de jetons du pot : le
  recalage horizontal accroche l'arête des jetons et le rapport sort des
  bornes.

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
