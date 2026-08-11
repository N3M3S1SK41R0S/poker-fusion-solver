# Mission — revue critique de Poker Fusion Solver

**Destinataires :** Gemini 3.5 Pro, ChatGPT 5.6, Claude Opus 5 (via Perplexity)
**Commanditaire :** Pierre, auteur et unique utilisateur du logiciel
**Date d'émission :** 11 août 2026
**Dépôt :** `github.com/N3M3S1SK41R0S/poker-fusion-solver` (branche `master`, 32 commits)

---

## 0. Avertissement préalable — ce qu'on n'attend pas de vous

Ce logiciel a été écrit par un modèle (Claude Opus 5) sur plusieurs sessions.
Il contient des erreurs, certaines découvertes le jour même de la rédaction de
ce brief. Une revue complaisante n'a **aucune valeur** pour nous.

Trois travers à éviter :

1. **Ne validez rien que vous n'ayez vérifié.** Si vous n'avez pas pu lire le
   code d'un module, dites « non vérifié » plutôt que « semble correct ».
2. **Ne confondez pas la présence d'un test avec une garantie.** Ce projet a
   vécu six semaines avec deux gabarits de cartes intervertis, sous
   1 057 tests verts, parce que les tests concernés comparaient chaque objet
   **à lui-même**. Cherchez ce genre de tautologie.
3. **Ne proposez pas ce qui existe déjà.** La section 5 liste ce qui est
   implémenté. Une recommandation « ajoutez de l'ICM » sera comptée comme un
   signe que vous n'avez pas lu le dépôt.

Nous préférons **cinq objections précises et vérifiables** à trente
suggestions génériques.

---

## 1. Ce qu'est ce logiciel, et à qui il sert

Poker Fusion Solver est un assistant de poker **personnel**, en français,
destiné à un seul joueur. Il ne se vend pas, ne s'héberge pas, ne s'exécute
que sur la machine de son utilisateur (serveur local, boucle 127.0.0.1
uniquement).

Il poursuit trois fonctions :

1. **Analyse post-partie.** Lire les historiques de mains du client PMU
   (XML iPoker), produire un profil statistique, séparer ce qui vient de la
   variance de ce qui vient du jeu, confronter les décisions préflop à
   l'équilibre de Nash.
2. **Conseil sur une main terminée.** À partir d'une capture d'écran collée,
   reconnaître les cartes et dire ce qu'il aurait fallu faire, avec le seuil
   de bascule et les hypothèses explicites.
3. **Entraînement.** Simulateur de mains, drills construits depuis les fuites
   mesurées du joueur, lexique.

### Ligne éthique — contrainte non négociable

Le logiciel **ne fournit aucune assistance en temps réel pendant une partie
d'argent réel**, et cette limite ne se négocie pas. Elle est inscrite dans le
code : `pfs/vision/live.py` n'importe aucun calculateur de décision, aucun
champ de sortie ne nomme une action, et `tests/test_live_sans_conseil.py` le
vérifie à chaque exécution. Un mode « calibration » lit l'écran pour vérifier
la reconnaissance d'images, sans jamais conseiller.

**Toute proposition d'assistance live en argent réel sera écartée sans
discussion.** Ce n'est pas une question de faisabilité : les adversaires
perdent de l'argent réel sans savoir qu'ils affrontent une machine. En
revanche, l'observation en argent fictif, l'étude, les drills et l'analyse
post-partie sont pleinement dans le périmètre.

---

## 2. Accès au code — à régler avant de commencer

⚠️ **Le dépôt est PRIVÉ et n'est accessible qu'en SSH.** Aucun des modèles
destinataires ne pourra le cloner en l'état. Pierre doit choisir l'une de ces
options avant de vous transmettre ce brief :

| Option | Ce qu'elle donne | Contrepartie |
|---|---|---|
| **A.** Rendre le dépôt public temporairement | Lecture complète, la meilleure revue possible | Le code devient visible de tous ; aucun secret n'y figure, mais des historiques de mains anonymisés y sont versionnés |
| **B.** Inviter des comptes GitHub en lecture | Contrôle fin | Les modèles via Perplexity n'ont pas de compte GitHub ; ne marche que pour des humains |
| **C.** Fournir une archive (zip) du dépôt | Lecture complète, hors ligne | À rejouer à chaque itération |
| **D.** Ne fournir que ce brief | Revue possible mais limitée aux points documentés ici | Vous ne pourrez pas vérifier les calculs vous-mêmes |

**Recommandation : option A ou C.** Ce brief est rédigé pour rester utile en
option D, mais vos réponses seront alors nécessairement plus faibles, et nous
vous demandons de le signaler explicitement.

### Repères pour se déplacer dans le dépôt

```
python/pfs/core/        ICM, équité, ranges, MDF, bluffcatch, bankroll, rake
python/pfs/solver/      push/fold Nash, DCFR, postflop, isomorphisme, abstraction
python/pfs/fusion/      13 « fusions » : HMM, filtre particulaire, Fisher-Rao,
                        goulot d'information, beta dynamique, arbitre…
python/pfs/analysis/    revue de session, revue push/fold, conseiller de spot
python/pfs/vision/      capture, détection de cartes, pHash, lecteur fond plein
python/pfs/data/        parseur d'historiques iPoker/PMU + PokerStars
python/pfs/train/       simulateur, drills, détecteur de fuites
python/pfs/app/         serveur HTTP local + interface (un seul fichier HTML)
python/banc_*.py        bancs de mesure rejouables — commencez par eux
python/tests/           1 095 tests
rust/crates/pfs-capture sonde de capture d'écran (Windows Graphics Capture)
```

**Par où commencer si votre temps est limité :** les fichiers `banc_*.py` et
les docstrings de `pfs/core/icm.py`, `pfs/vision/card_recognizer.py`,
`pfs/solver/pushfold.py`. Les docstrings de ce projet documentent les
approches **abandonnées** et leurs mesures, pas seulement le code retenu.

---

## 3. L'état réel, sans complaisance

Chiffres au 11 août 2026, mesurés :

| | |
|---|---|
| Modules de production | 61 fichiers, **19 373 lignes** |
| Tests | 44 fichiers, 10 124 lignes, **1 095 tests verts** |
| Sonde Rust | 4 fichiers, 348 lignes |
| Routes serveur exposées | 32, dont **4 jamais appelées** par l'interface |
| Modules exécutés lors d'une décision complète | **14 sur 61** |
| Modules inatteignables par toute action utilisateur | **12, soit 6 124 lignes = 31,7 % du code** |

### Ce qui est mesuré et tient

- **Perception.** Localisation des cartes à pleine échelle : 100 % de
  localisation et de rôles sur cinq configurations de bancs synthétiques
  (54 tables chacune). Sur une capture réelle de table PMU, les deux cartes du
  héros sont lues avec certitude (`5c` écart 116, `3h` écart 90).
- **Refus des faux positifs.** Sur 240 découpes qui ne sont pas des cartes
  (bruit, feutre, dos, jetons), **0 est lue comme une carte**.
- **Analyse de session.** 721 mains réelles récupérées sur 62 tournois. Le
  joueur perd 1 160,8 bb, dont **31,5 bb seulement de variance aux tapis** —
  97 % des pertes viennent des décisions.

### Ce qui ne tient pas, et qu'il faut que vous sachiez

Ces défauts ont été trouvés **le jour de la rédaction de ce brief**, ce qui
donne une idée du taux de défauts restants :

1. **`/api/advise` jetait les tapis et les gains de tournoi**, rendant le
   régime ICM inatteignable depuis l'interface. Mesuré sur un spot de bulle :
   le verdict était **inversé** (« OUVRIR » au lieu de « FOLD »). Corrigé.
   *Cause de non-détection : tous les tests du régime ICM appelaient la
   fonction en direct, jamais la route.*
2. **Le verdict se déclenchait sur des montants inventés.** La capture ne
   donne que les cartes ; le pot, la mise et les tapis restaient les valeurs
   par défaut du formulaire. Corrigé par un garde-fou explicite.
3. **Deux gabarits de cartes étaient intervertis** (`7h` / `7d`) depuis leur
   extraction. Le vrai 7 de cœur était lu « 7♦ » avec une confiance maximale.
   Corrigé, avec un contrôle géométrique non tautologique.
4. **Le parseur d'historiques mourait sur un montant en devise** (« 120€ »),
   emportant en silence toutes les mains suivantes du fichier. 427 mains
   récupérées avant correction, **721 après**.
5. **L'ICM plantait sur un tapis nul**, contourné jusque-là par une
   « amputation » (évaluer la perte sur 99,9999999 % du tapis). Corrigé
   exactement ; les valeurs de référence sont retrouvées au dernier chiffre.

### Les 12 modules morts

Aucun n'est atteignable par une action de l'utilisateur ; chacun a pourtant
ses tests au vert.

`digit_ocr` (lecture des montants sur capture, 99,5 % d'exactitude sur son
banc — jamais appelé), `eqr` (modèle de feuille postflop), `bunching`
(repondération multiway), `abstraction` + `isomorphism` (accélération du
solveur), `inference_check`, `solver_registry`, `gate`, `meanfield`,
`timing`, `topology`, `leak_drills`.

**Cas particulier — la chaîne de fusion.** `pfs/engine.py::FusionEngine.decide()`,
dont la docstring annonce « là où les 13 fusions deviennent une seule
décision », n'est appelée que par `demo.py` et les tests. **Les 13 fusions ne
s'exécutent jamais** dans le chemin utilisateur. Exécutée à la main, la route
correspondante met 221 secondes et, faute d'observations adverses qu'aucune
interface ne permet de fournir, dégénère en GTO pur.

---

## 4. Ce que nous vous demandons — trois axes

Répondez aux trois. Si votre budget est limité, privilégiez **l'axe A**.

### Axe A — Les calculs sont-ils justes ?

C'est le cœur. Un solveur qui se trompe est pire qu'un solveur absent.

Vérifiez, **contre des références extérieures au dépôt** :

- **ICM (`pfs/core/icm.py`).** Malmuth-Harville exact jusqu'à 12 joueurs,
  Monte-Carlo au-delà ; `bubble_factor`, prime de risque, variante PKO
  (tournois à primes), FGS léger (érosion des blindes futures).
  Points de contrôle attendus : conservation de la dotation en toutes
  circonstances (tapis nuls, ex æquo, joueur unique) ; un gros tapis vaut
  moins que sa part de jetons ; convergence exact ↔ Monte-Carlo. **Comparez
  aux valeurs d'ICMIZER ou HRC si vous les connaissez.**
- **Nash push/fold heads-up (`pfs/solver/pushfold.py`).** Confrontez quelques
  cellules aux tables publiques (SAGE, équilibres HU classiques).
- **Équité (`pfs/core/equity.py`).** AA vs KK, AKs vs 22, paire vs deux
  surcartes.
- **DCFR (`pfs/solver/dcfr.py`).** L'exploitabilité doit décroître et tendre
  vers zéro sur Kuhn poker, dont l'équilibre est connu analytiquement.
- **Le parseur (`pfs/data/hand_history.py`).** Sémantique iPoker décodée
  empiriquement : `sum` d'une mise est cumulatif pour la rue, incrémental pour
  un call. Le pot est reconstruit par somme des gains. **Cherchez les cas où
  cette reconstruction se trompe.**
- **La revue de session.** L'équité « all-in adjusted » est calculée au moment
  où les jetons entrent. Vérifiez qu'aucune fuite temporelle ne s'y glisse.

Pour chaque point : **confirmé avec chiffres**, **réfuté avec le contre-exemple**,
ou **non vérifié**. Pas de quatrième catégorie.

### Axe B — Qu'est-ce qui est calculé mais jamais utilisé ?

L'audit interne a trouvé 31,7 % de code mort. **La décision est prise : tout
ce qui sert la décision de jeu sera branché, rien ne sera supprimé pour
simplifier.** Ce que nous attendons de vous, c'est le *comment* et l'*ordre*.

- Ce recensement est-il juste ? En avons-nous manqué ? Des modules que nous
  croyons vivants sont-ils en réalité contournés ?
- Pour chacun des 12 modules morts : **par où le brancher**, quel maillon
  manque, et qu'est-ce que l'utilisateur y gagne concrètement. Exemple connu :
  `digit_ocr` sait lire les montants, mais le détecteur ne sait localiser que
  des cartes — il manque la détection des bandeaux de texte.
- **La chaîne de fusion (13 modules) doit être branchée.** Comment ? Elle met
  aujourd'hui 221 secondes sur sa seule route, et dégénère en GTO pur faute
  d'observations adverses qu'aucune interface ne permet de fournir. Il faut
  donc : un chemin de saisie ou de capture des observations, et un budget de
  calcul compatible avec une interface. Dites-nous comment vous vous y
  prendriez, et ce que chacune des 13 fusions apporte réellement à la
  décision finale — les inutiles doivent être identifiées comme telles, mais
  par leur apport mesuré, pas par principe d'économie.
- Les 4 routes fantômes (`resolve`, `presets`, `drill/next`, `next`) :
  comment les exposer utilement dans l'interface ?

### Axe C — Comment être loin devant la concurrence ?

Concurrents de référence : **GTO Wizard**, **PioSOLVER**, **Simple Postflop**,
**MonkerSolver** (multiway), **HRC** et **ICMIZER** (tournois), **Flopzilla**
et **Equilab** (équité), **PokerSnowie** (heuristique neuronale), et la
littérature académique (DeepStack, Libratus, Pluribus, ReBeL).

**L'objectif est explicite et il n'est pas négociable : l'outil le plus abouti
qui ait existé pour un joueur seul.** Pas un outil de plus, pas un sous-ensemble
raisonnable. Nous savons qu'un solveur postflop sur un PC personnel ne battra
pas PioSOLVER en force brute — mais la force brute n'est qu'un des axes, et
c'est le seul sur lequel nous concédons quelque chose. Partout ailleurs, la
question est **comment faire mieux**, pas s'il faut essayer.

**Les questions qui nous intéressent :**

1. **Quel est l'angle mort commun à tous ces outils ?** Ce projet parie sur
   un angle : ils sont tous *normatifs* (voici l'équilibre) et aucun n'est
   *personnel* (voici **ta** fuite, mesurée sur **tes** mains, et l'exercice
   qui la corrige). Ce pari tient-il ? Quel autre angle mort voyez-vous ?
2. **Que manque-t-il à ce logiciel pour dépasser chacun d'eux sur son propre
   terrain, un par un ?** Prenez-les nommément — GTO Wizard, Pio, Monker, HRC,
   ICMIZER, Snowie — et pour chacun : sur quoi sommes-nous déjà devant, sur
   quoi sommes-nous derrière, et qu'est-ce qui comblerait l'écart. Le profil
   mesuré du joueur (VPIP 63 %, PFR 31 %, 3-bet 5 %, WTSD 45 %) sert à
   **prioriser** vos réponses, pas à en écarter.
3. **Qu'est-ce qui, dans la littérature récente, n'est encore dans aucun
   produit commercial ?** DeepStack, Libratus, Pluribus, ReBeL, et ce qui a
   suivi. Nous acceptons les réponses ambitieuses dès lors qu'elles tournent
   sur une machine personnelle. C'est là que se gagne l'avance, et c'est la
   partie de vos réponses qui nous intéresse le plus.
4. **Quelle mesure de progrès proposeriez-vous ?** Le logiciel sait mesurer
   les fuites ; il ne sait pas encore mesurer si le joueur s'améliore. Comment
   établiriez-vous qu'un entraînement a un effet, avec quelques centaines de
   mains par semaine et une variance énorme ?
5. **Qu'est-ce qui manque que personne n'a encore nommé ?** Si votre meilleure
   idée n'entre dans aucune des questions ci-dessus, c'est probablement la
   plus intéressante : mettez-la en premier.

---

## 5. Ce qui est déjà implémenté — ne le proposez pas

Pour éviter les redites. Tout ceci existe et est testé :

**Théorie du jeu :** équilibre de Nash push/fold heads-up (tabulé), DCFR
(discounted CFR) sur jeux jouets, solveur postflop avec abstraction de mises,
exploitabilité, isomorphisme des flops (1 755 classes), nodelock.

**Tournois :** ICM Malmuth-Harville exact et Monte-Carlo, facteur de bulle,
prime de risque, ICM 3-max, PKO (valeur de capture d'une prime, ½ cash +
½ sur sa propre tête), FGS léger, bankroll et critère de Kelly.

**Théorie de la décision :** MDF, cotes du pot, équité requise, solution de
polarisation (bluff optimal b/(P+2b)), EQR, bluffcatch, dimensionnement de
mise, prix de la connaissance.

**Modélisation de l'adversaire :** rétrécissement empirical-Bayes des stats,
HMM d'état mental (tilt), filtre particulaire sur les ranges, distance de
Fisher-Rao entre distributions, goulot d'information, a priori de niveau de
jeu, arbitre de fusion.

**Analyse :** parseur iPoker/PMU et PokerStars, revue de session, équité
all-in adjusted, revue push/fold contre Nash, validation prédictive sans fuite
temporelle (empirical-Bayes vs estimation brute, p ≤ 0,012 sur vpip/pfr/wtsd).

**Perception :** capture d'écran de fenêtre occultée (Windows Graphics
Capture, Rust), détection de cartes par géométrie d'arêtes, reconnaissance par
hachage perceptuel avec gabarits amputés (bandeau d'interface), lecture
directe des jeux à fond plein (la couleur donne l'enseigne), OCR de montants
(écrit, non branché).

**Entraînement :** simulateur de mains à tirage réellement aléatoire, drills
construits depuis les fuites mesurées, lexique de 39 termes.

---

## 6. Format de réponse attendu

Structurez ainsi. La longueur est libre, la précision ne l'est pas.

```
## Ce que j'ai réellement lu
   (fichiers ou sections consultés ; si vous n'avez pas eu accès au dépôt,
    dites-le ici, en premier)

## Axe A — validation des calculs
   Un bloc par cible :
   CIBLE : <module ou fonction>
   VERDICT : CONFIRMÉ | RÉFUTÉ | NON VÉRIFIÉ
   PREUVE : <valeur de référence, source, écart mesuré>
   GRAVITÉ : bloquant | important | mineur

## Axe B — code calculé mais inutilisé
   Pour chaque module : brancher / archiver / supprimer, et POURQUOI,
   en termes de ce que l'utilisateur y gagne.

## Axe C — propositions
   Classées par (valeur pour CE joueur) ÷ (coût de réalisation).
   Pour chacune : le problème résolu, comment on saurait que ça marche,
   ce que la concurrence fait déjà, et l'effort estimé.

## Ce que je n'ai pas pu vérifier
   Section obligatoire. Une réponse sans cette section sera considérée
   comme incomplète.

## Mon désaccord principal avec le brief
   Section obligatoire également. Si le brief vous paraît juste sur tout,
   dites ce qui vous semble le plus fragile dans son raisonnement.
```

---

## 7. Trois questions dont la réponse nous importe particulièrement

1. **Le joueur a une fuite mesurée — écart VPIP−PFR de 32 points, constant sur
   62 tournois — et il a tranché : il ne veut pas d'un outil réduit à cette
   fuite, il veut le solveur le plus abouti possible, et il apprendra avec.
   Cette décision n'est pas à rediscuter.** La question qui nous intéresse est
   donc l'inverse de celle du périmètre : **comment un outil complet peut-il
   aussi corriger une fuite élémentaire, sans se brider ?** Autrement dit, que
   faut-il ajouter pour que la sophistication *serve* l'apprentissage au lieu
   de le contourner — hiérarchie des conseils, ordre d'exposition, boucle
   drill → mesure → drill ? Répondez à cette question-là, pas à celle de
   savoir s'il faut en faire moins.

2. **La reconnaissance d'images a coûté six diagnostics erronés successifs
   avant d'aboutir.** Chaque fois, une cause plausible était affirmée sans
   mesure, puis démentie. Quel protocole imposeriez-vous à un modèle qui
   travaille seul sur un problème de perception, pour que cela ne se
   reproduise pas ?

3. **Le logiciel n'a aucun moyen de savoir s'il rend le joueur meilleur.**
   Comment le mesureriez-vous, sachant que la variance au poker exige des
   dizaines de milliers de mains pour distinguer un gain de compétence, et que
   le joueur en produit quelques centaines par semaine ?

---

## 8. Itération

Pierre collectera vos trois réponses, les confrontera, et reviendra vers
Claude Opus 5 pour la mise en œuvre. Les désaccords entre vous nous
intéressent plus que vos consensus : dites clairement quand vous pensez
qu'une autre lecture est défendable, et pourquoi vous ne la retenez pas.

**Contrainte de mise en œuvre, à garder en tête dans vos propositions :** tout
ce que vous proposerez devra tourner sur un PC Windows personnel, sans GPU
dédié, sans service externe, et rester lisible par une seule personne. Une
proposition qui exige un cluster ou une API payante sera écartée quelle que
soit sa valeur théorique.
