# Plan v4.3 — « coller une capture → verdict », ICM, drills

Feuille de route suivie à la lettre. Chaque phase se clôt par un **auto-audit
chiffré** (critère de sortie), puis un commit. Tant que le critère n'est pas
atteint : mesurer → diagnostiquer → corriger → re-mesurer.

Rappel honnêteté (NEMESIS) : la validation utilise des captures **synthétiques**
que je compose ; la validation sur de vraies captures de Pierre viendra ensuite
et pourra faire bouger les seuils. Tout est écrit pour que ce ré-ajustement soit
un réglage, pas une réécriture.

## Phase A — Détection automatique des cartes
Trouver les cartes (héros + board) dans une capture, sans ROI codées en dur.
`pfs/vision/table_detector.py` : segmentation par écart au feutre → composantes
au ratio carte → tri héros (bas) / board (rangée médiane).
**Critère** : sur tables synthétiques (habillages × fonds × dispositions),
≥ 95 % des cartes localisées, 0 fausse localisation grossière.

> **Le critère est atteint sur synthétique (98,8 %) et RATÉ sur réel
> (76,7 %).** Mesuré le 11 août 2026 par `python/banc_verite_captures.py`
> contre la vérité-terrain des 57 captures PMU : 198 cartes localisées sur
> 258 réellement présentes. L'avertissement d'honnêteté ci-dessus s'est
> vérifié au pire endroit — 60 cartes ne sont jamais vues, dont les deux
> cartes du héros sur presque toute une des deux tables, et la cause est une
> constante calée sur le synthétique (`QUIET_SIDES`, voir le README). La
> phase A n'est donc **pas close** : son critère doit être re-libellé sur du
> réel, et c'est le premier chantier de vision à ouvrir.

### Phase A — diagnostic mesuré du cas qui bloque
Carte BLANCHE (deck classique) sur tapis CLAIR : corps de carte `[254,254,254]`,
ovale de table `[229,233,240]` → distance RGB **36,1** (donc séparables), mais la
quantification à 6 niveaux (largeur de bin ≈ 43) les range dans le **même bin**
(code 215) : la carte devient invisible. Sur feutre vert : distance 313, bins
distincts (215 vs 49) → détection correcte.
Second constat : sur feutre vert, 9 boîtes trouvées pour 5 cartes → il y a aussi
de la **sur-détection**, que le dédoublonnage par IoU ne suffit pas à corriger.
Piste : segmentation **multi-échelle** (rejouer à 6, 10, 14 niveaux puis fusionner),
plus un filtre de validation « card-like » contre les fausses boîtes.
Script : `scratchpad/diag_clair.py`.

## Phase B — Lecture des montants (OCR à gabarits)
`pfs/vision/digit_ocr.py` : gabarits de chiffres extraits du client → lecture de
« Pot: 1,50 BB », « 10 BB », etc.
**Critère** : ≥ 98 % de chaînes de montants synthétiques lues exactement.

## Phase C — Flux « coller → verdict »
Route `/api/read_table` (image → cartes + montants) ; UI : coller → pré-remplir
→ `advise`, avec correction manuelle possible.
**Critère** : bout en bout au navigateur, 0 erreur JS, verdict correct sur une
table synthétique complète.

## Phase D — ICM 3-max dans le conseiller
`advise()` gère le short-stack à 3 joueurs et l'effet bulle (Harville, bubble
factor déjà dans `core/icm`). Verdicts SB-vs-2 et BB-vs-jam, EV corrigée ICM.
**Critère** : goldens ICM à la main ; l'ICM resserre bien les jams vs chipEV.

## Phase E — Solveur postflop dans le conseiller
Brancher le solveur river/turn exact : verdicts de **mise** (taille + fréquence)
au lieu d'un « MISER » générique.
**Critère** : sur un spot de polarisation, retrouve bluff = b/(P+2b), taille et
fréquence cohérentes ; EV ≥ ligne passive.

## Phase F — Boucle analyse → drills
Générer des drills depuis les fuites détectées (revue de session/shove-fold) et
les rejouer (SM-2 existant).
**Critère** : depuis un historique, produit des drills ciblés jouables, réponse
correcte = celle du solveur.

## Phase G — Finalisation
Régression complète + selftest, README/CHANGELOG v4.3, paquet zip+bundle vérifié,
push GitHub.
**Critère** : suite verte, 19 goldens, paquet ré-extrait fonctionnel.
