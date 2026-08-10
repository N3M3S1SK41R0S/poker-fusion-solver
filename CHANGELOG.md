# Journal des versions — Poker Fusion Solver

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
  à un port déjà en écoute. Après chaque modification du code, un nouveau
  serveur démarrait pendant que l'ancien continuait de répondre : les routes
  fraîchement ajoutées renvoyaient « route inconnue » une fois sur deux. Le
  serveur utilise désormais `SO_EXCLUSIVEADDRUSE` et refuse de démarrer, avec
  un message actionnable, si le port est pris.
- **Échecs d'archive qui s'écrasaient.** Les noms de fichiers étant horodatés
  à la seconde, les huit découpes d'une même lecture de table se recouvraient :
  l'archive ne gardait qu'une découpe par seconde et par statut, en silence.
- **Fenêtre figée jamais capturée.** Windows Graphics Capture n'émet une image
  que lorsque la fenêtre se redessine — une table entre deux actions restait
  muette jusqu'au timeout. La cible est maintenant réveillée par
  `RedrawWindow`, sans être ni déplacée ni activée.

### Mesuré

- Localisation sur image réduite à 1280 px avant reconnaissance sur
  l'originale : sur 48 tables synthétiques 2560×1529, **100 % de localisation
  et de rôles** dans les deux cas, mais **32 boîtes fantômes à pleine échelle
  contre 0 à 1280** — la réduction lisse les petits décors pris pour des
  cartes. Balayage complet des largeurs dans `live.py` : 1280 est un optimum,
  pas un compromis.
- Boucle live complète : **750 ms** sur une table, 5,6 s sur une fenêtre de
  bureau chargée (cas le pire, milliers d'arêtes verticales).
- **Non mesuré, assumé** : la précision de reconnaissance sur une vraie table
  de room. La chaîne a été validée de bout en bout, mais sur une table
  synthétique dont les cartes ne sont pas celles du deck PMU — le taux de
  lecture observé (0/8) ne dit rien de la performance réelle. C'est la
  première chose à faire à la prochaine session.

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
