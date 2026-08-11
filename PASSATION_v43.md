# PASSATION — session du 8-10 août 2026

**De : Claude (Claude Code, Fable 5) → à la prochaine session.**
Suite de `PASSATION.md` (v4.0). Lis les deux.

---

## 0. Où en est le projet

**986 tests verts · 19 goldens selftest conformes · dépôt GitHub privé
synchronisé** : https://github.com/N3M3S1SK41R0S/poker-fusion-solver

Environnement (voir aussi la mémoire projet) :
- dépôt : `C:\Users\pierr\Documents\POKERFUSIONSOLVER\poker-fusion-solver`
- **Python : TOUJOURS `.venv\Scripts\python.exe` du dépôt.** Le `python` du
  PATH est un venv tiers sans numpy.
- lancer : raccourci bureau **PKS**, ou `run.bat`. URL stable désormais
  (jeton persistant) : `http://127.0.0.1:8731/?t=...`
- ⚠️ **Vérifier que le serveur a bien redémarré** avant de conclure qu'une
  route manque : un processus non tué a déjà fait perdre du temps
  (« route inconnue » alors que le code était juste).

---

## 1. Ce que cette session a ajouté

| Domaine | Livré |
|---|---|
| Perception | Sonde WGC (GO mesuré, ROI p95 245 µs sur client PMU réel), recogniseur de cartes multi-habillages, détecteur de cartes dans une table |
| Analyse | Parseur iPoker/PMU (XML), revue de session, revue shove/fold, conseiller de spot, OCR des montants |
| Théorie | ICM 3-max + solve river dans le conseiller, validation prédictive de l'inférence adverse |
| Entraînement | Drills depuis les fuites, **simulateur de main**, **lexique (39 termes)** |
| Interface | Onglets par tâche, code couleur des actions, capture collable, mobile |

---

## 2. LE point à retenir : la thèse du projet est validée (partiellement)

`pfs/analysis/inference_check.py` — 286 mains réelles, 106 adversaires,
validation prédictive **sans fuite temporelle** (à k=0 le gain vaut
+0,00000000 exactement, preuve structurelle).

| Stat | Gain log-vrais. | p | Verdict |
|---|---|---|---|
| vpip | +0,0113 | 0,002 | **bat le prior** |
| pfr | +0,0059 | 0,002 | **bat le prior** |
| wtsd | +0,0147 | 0,012 | **bat le prior** |
| three_bet | +0,0008 | 0,075 | non concluant (718 obs) |
| fold_to_cbet | — | 0,130 | sous-alimenté (94 obs) |

**Résultat gênant conservé** : avec un prior de force 1, le modèle est
**pire** que le prior. Le gain vient du **rétrécissement empirical-Bayes**,
pas de l'estimation brute par joueur. Ne pas l'oublier en communiquant.

---

## 3. Le profil de Pierre, mesuré (ne pas re-mesurer, c'est fait)

317 mains, 2 comptes. **Sa fuite est unique et massive : il joue trop de
mains, passivement.**

| | compteA | compteB |
|---|---|---|
| Net | −375 bb | −364 bb |
| VPIP | 63,1 % | **73,8 %** (sain : 28-50) |
| Écart VPIP−PFR | 30,6 pts | **39,8 pts** (sain : <18) |
| 3-bet | 4,9 % | 3,0 % (sain : 8-24) |

**La variance n'explique rien** : +52,9 bb et −86,5 bb, soit ~−34 bb net
pour −739 bb de pertes. Ses **tapis préflop sont bons** (7 écarts sur 37
décisions, ~1,5 bb) — surtout des *limps*. La perte est postflop et dans
les mains non-tapis.

---

## 4. ⚠️ Les pièges de CETTE session (ne pas les revivre)

1. **Je me suis trompé 4 fois** sur l'échec de reconnaissance avant de
   trouver. Cause réelle : **le cadrage à la souris ne découpait pas la
   carte** — `getBoundingClientRect` donne des pixels d'AFFICHAGE, le code
   divisait par l'échelle du canvas. Le CSS `max-width:100%` rend les deux
   différentes. Preuve : cible en `[492,192,94,120]`, ancienne formule
   `[1,1]`. **Leçon : mesurer sur le fichier réel, pas raisonner sur une
   capture re-rendue.**
2. **Le jeton changeait à chaque démarrage** → 403 silencieux sur tout
   onglet ouvert. Corrigé (jeton persistant), mais c'est ce qui faisait
   croire à un défaut de reconnaissance.
3. **Les seuils calibrés sur du synthétique ne tiennent pas sur du réel** :
   « 0,06 % de lectures fausses » devient **4,5 %** sur images dégradées.
4. Une piste mesurée puis **retirée** : reconnaître sur le seul coin
   d'index pour résister au HUD. Le HUD masque bien le bas des cartes du
   héros (15 % masqué → écart 347 → 688, reconnaissance 0/5), mais une
   découpe *relative* ne s'aligne plus sur une carte tronquée. **Problème
   ouvert.**

---

## 5. Ce qui reste à faire, par priorité

1. **VALIDER SUR DE VRAIES CAPTURES.** Tout est mesuré sur du synthétique.
   Il faut 20-50 captures PMU **enregistrées en PNG sur disque** et
   annotées. Rien ne remplace ça — c'est ce qui a fait perdre le plus de
   temps cette session.
2. **Occultation par le HUD** (cf. piège 4) : les cartes du héros restent
   illisibles. Piste non essayée : aligner la découpe sur le bord DÉTECTÉ
   de la carte plutôt qu'en fractions relatives.
3. **Brancher vision + OCR dans l'application** : `table_detector` et
   `digit_ocr` sont testés mais **importés par personne**. Le flux
   « coller → verdict sans saisie » n'existe pas encore.
4. **JPEG** : la détection tombe de 98,8 % (PNG) à 88,1 % (q=75).
5. **Solveur 3-max** : l'ICM à 3+ joueurs reste « indicatif » (le solveur
   est heads-up). Chantier lourd.
6. **Périmètre des drills** : 6 drills sur 317 mains. L'élargir demande un
   solveur préflop profond ou un modèle de limp.
7. **TempData** : les mains des tournois EN COURS sont en protobuf binaire
   (`History\TempData\...`), pas en XML. Elles échappent à l'analyse
   jusqu'à la fin du tournoi. Les captures d'écran de Pierre venaient de
   là — d'où l'impossibilité de valider ses conseils contre le résultat.

---

## 6. Contrats non négociables (rappel, ils ont tenu)

- **Français partout**, docstrings NumPy, commentaires qui expliquent le
  *pourquoi*.
- **Toute affirmation est un test.** Mesurer avant d'optimiser ET avant
  d'affirmer. Un seuil sans mesure est un bug en attente.
- **Honnêteté NEMESIS** : écrire les limites chiffrées. Cette session a
  corrigé 3 affirmations fausses grâce à une relecture adversariale.
- **Périmètre éthique** : le logiciel n'analyse que des mains **terminées**.
  Pierre a demandé une assistance en direct sur argent réel ; j'ai refusé
  et maintenu le refus. Il a accepté le recentrage post-partie. **Ne pas
  revenir dessus.**

---

*Poker Fusion Solver v4.2+ — 986 tests · 10 août 2026*
