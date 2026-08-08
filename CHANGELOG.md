# Journal des versions — Poker Fusion Solver

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
