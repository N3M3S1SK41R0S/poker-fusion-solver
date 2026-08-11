# Réponse au conseil des modèles — tour 1

*(À envoyer aux trois : GPT-5.6 Sol Thinking, Claude Opus 5 Thinking, Gemini 3.1 Pro Thinking)*

---

Merci à vous trois. Cette revue a été utile malgré son échec principal, et
elle l'a été surtout là où elle nous contredit. Ce message fait quatre
choses : régler l'accès, vous dire ce que nous avons accepté et corrigé,
trancher les points où nous ne vous suivons pas, et poser le tour 2.

---

## 1. L'accès — c'était chronologique, pas technique

Vous avez tous les trois échoué à lire le dépôt, et vous avez eu raison de
vous placer en option D plutôt que d'inventer des verdicts. L'explication est
bête : **le brief vous a été transmis pendant que le dépôt était encore
privé.** Il a été rendu public juste après.

Vérifié à l'instant, en anonyme et sans jeton :

```
api.github.com/repos/N3M3S1SK41R0S/poker-fusion-solver    → 200
raw.githubusercontent.com/.../master/README.md            → 200
codeload.github.com/.../zip/refs/heads/master             → 200
```

**https://github.com/N3M3S1SK41R0S/poker-fusion-solver**

Une **archive zip** (1,1 Mo, 284 fichiers, brief et message inclus) est
également disponible et vous sera jointe si votre outillage ne suit pas les
redirections GitHub. Elle ne dépend d'aucune propagation d'index.

**Conséquence : l'axe A n'a pas été traité, et c'est désormais la priorité
absolue du tour 2.** Toute la partie « les calculs sont-ils justes ? » reste
entièrement ouverte.

---

## 2. Ce que nous avons accepté, et déjà corrigé

### 2.1 Le chiffre « 97 % des pertes viennent des décisions » était faux

GPT-5.6 et Claude Opus 5 ont raison, Gemini ne l'a pas relevé. Nous avons
vérifié et le reproche tient entièrement.

Ce que le logiciel mesure, c'est **uniquement** la variance des tapis où les
deux mains sont connues : −31,5 bb sur 71 spots. Trois sources lui échappent
totalement :

- **25 tapis sur 96 (26 %)** sans cartes adverses connues — non mesurés ;
- la variance des pots **hors tapis** — jamais calculée ;
- la variance de **distribution** (recevoir de mauvaises mains) — jamais
  calculée.

Annoncer 97 % revenait à traiter tout ce qui n'est pas mesuré comme valant
zéro. La formulation honnête est : *la variance que l'outil sait mesurer
explique 2,7 % du résultat ; le reste n'est pas décomposé.* La fuite reste
massivement établie — l'écart VPIP−PFR de 32 points, constant sur
62 tournois, ne dépend d'aucun de ces calculs — mais la part de
responsabilité ne devait pas être chiffrée. Corrigé partout.

C'est exactement le travers que ce projet combat : une conclusion plus forte
que la mesure. Merci de l'avoir attrapé.

### 2.2 Les défauts sont aux jointures — c'est la critique la plus juste

Claude Opus 5 conteste la structure même du brief, et il a raison. Relisez nos
cinq défauts : `/api/advise` qui jette les tapis, le verdict sur montants par
défaut, les gabarits intervertis, le parseur mort sur « 120€ », l'ICM sur
tapis nul. **Tous aux jointures**, chaque composant étant correct isolément.
Aucune relecture de code ne les attrape de façon fiable, et notre brief le
diagnostiquait lui-même sans en tirer la conclusion.

Trois chantiers sont lancés, chacun suivi d'un agent chargé de le démolir :

1. **`test_parcours_complet.py`** — part d'un fichier sur disque, traverse les
   **routes HTTP réelles** (serveur monté sur port libre, dialogue par
   `urllib`), jamais les fonctions internes. Une douzaine de verdicts écrits à
   la main couvrant les quatre régimes, **un cas où le refus est obligatoire**,
   et la vérification que les montants non saisis ne produisent pas de verdict.
   Exigence explicite : **prouver que le test échoue** si l'on réintroduit deux
   des défauts. Mesure de couverture d'un parcours utilisateur réel incluse —
   nous nous attendons à un chiffre bas et nous le publierons tel quel.
2. **Invariants ICM** — invariance d'échelle et par permutation, absentes de
   nos tests. La première attrape directement le bug de « l'amputation à
   99,9999999 % ». Étendues au chemin PKO.
3. **Robustesse de la garantie éthique** — voir §3.4.

### 2.3 SAGE n'est pas Nash

Accepté sans réserve, et c'est un piège que nous aurions mordu : notre brief
citait SAGE comme référence de validation. Un écart avec notre solveur
n'aurait donc **pas** été une réfutation. La remarque associée est encore plus
utile : si notre interface rend un verdict push/fold à 25 bb, le calcul peut
être exact et le conseil mauvais — même famille de bug que le défaut n° 2.
Nous en faisons un test.

### 2.4 Le biais structurel de Harville

Accepté, et c'est la contribution la plus intéressante de la revue. Concorder
avec ICMIZER ou HRC ne prouverait que la reproduction fidèle d'un modèle
biaisé, hérité de la modélisation hippique. Nous retenons la suggestion :
rendre la **sensibilité du verdict au modèle d'équité** plutôt qu'un binaire.
Un verdict qui bascule entre Harville et Malmuth-Weitzman doit se déclarer
fragile.

### 2.5 `digit_ocr` et `leak_drills`

Consensus des trois, accepté. `digit_ocr` referme le défaut n° 2 ;
`leak_drills` était mal classé parmi les morts. La réserve d'Opus 5 est
retenue : **99,5 % sur banc synthétique n'est pas 99,5 % sur capture PMU
réelle**, et un OCR lisant « 1200 » au lieu de « 120 » inverserait un verdict.
Branchement avec canal de confiance et **refus explicite** — la philosophie
qui a déjà produit 0 faux positif sur 240 découpes.

### 2.6 La mesure de progrès

La chaîne à quatre étages est adoptée telle quelle, avec l'étage 2 (rétention
à J+7 / J+28) comme pivot, pour la raison qu'Opus 5 donne : c'est un **taux
binomial sur des dizaines d'essais**, à variance faible, contrairement à un
winrate. Le plan multiple-baseline A/B/C de GPT-5.6 est retenu pour
l'attribution causale.

### 2.7 Le protocole de perception

Adopté, dans la formulation la plus stricte des trois — celle de Gemini :
**aucune modification de code avant qu'un script d'introspection n'ait
sauvegardé sur disque ce que le code croit voir** (masque de pixels, cartes
d'arêtes, boîte englobante). Nos six diagnostics erronés venaient tous de
raisonnement sans observation. La règle complémentaire est retenue aussi :
pas d'hypothèse sans, dans le même message, la mesure qui la départage d'une
hypothèse rivale.

---

## 3. Ce que nous ne retenons pas, ou pas ainsi

### 3.1 L'équité 22 vs AKs — Gemini se trompe

Gemini annonce 47,3 / 52,7, chiffre **erroné et inversé**. Les sources
concordent autour de 50,1 % pour la paire de 2 contre AK assorti, et ~52 %
contre AK non assorti. Nous le signalons sans malice : c'est précisément le
type d'assertion que notre brief demandait de ne pas produire de mémoire. Il
illustre pourquoi l'axe A doit être traité **avec le code sous les yeux et un
banc qui tourne**, pas par rappel.

### 3.2 Le périmètre n'est pas à rediscuter

Les trois recommandent de suspendre la sophistication. Le commanditaire a
tranché et cette décision n'est pas soumise à révision : il veut l'outil le
plus abouti possible et apprendra avec. Ce n'est pas une réponse d'orgueil,
c'est un choix assumé sur ce qu'il attend d'un logiciel personnel.

Nous retenons en revanche la reformulation d'Opus 5, qui est compatible et
que nous adoptons comme règle permanente :

> **La chose à arrêter n'est pas la sophistication, c'est l'ajout de
> capacités non branchées.** Aucun module fusionné dans `master` sans une
> route qui l'appelle, un élément d'interface qui déclenche la route, et un
> test de bout en bout traversant les trois.

Cette règle seule aurait empêché douze modules morts, deux gabarits
intervertis pendant six semaines, et un régime ICM inatteignable. Elle est
désormais la règle du projet.

Nous retenons aussi le refus de GPT-5.6 du dogme « ne limpe jamais » : un
écart de 32 points signale une fuite probable, pas la preuve que chaque call
est faux. La cible est **l'entrée passive non justifiée par la position, la
profondeur et l'action antérieure**.

### 3.3 Les 13 fusions — nous suivons, avec une réserve

Votre triple convergence, par trois raisonnements indépendants, nous a
convaincus de ne pas les brancher en l'état. L'argument décisif n'est pas la
lenteur mais l'identifiabilité : **721 mains sur 62 tournois font ~12 mains
par adversaire**. Il n'y a pas de population à modéliser, et la dégénérescence
en GTO pur est le comportement *correct* d'un estimateur devant un échantillon
vide.

Mais nous ne les archivons pas telles quelles. Nous retenons la proposition
d'Opus 5 de **retourner Fisher-Rao et l'empirical-Bayes vers le joueur** : une
distance à l'équilibre par session, scalaire, sans variance de résultat. Le
code existe, il est validé (p ≤ 0,012 sur vpip/pfr/wtsd), et là il dispose
d'un échantillon suffisant. C'est le seul cas où un module « mort » devient
un instrument de mesure central.

**Question ouverte pour le tour 2 :** quelles autres fusions se retournent
ainsi vers le joueur plutôt que vers l'adversaire ? Le HMM d'état mental
appliqué à *ses propres* sessions a-t-il un sens, avec 721 mains ?

### 3.4 La robustesse du test éthique — noté, mais reporté

Opus 5 signale que notre test de non-assistance porte sur les imports
statiques et les noms de champs, pas sur le contenu de la réponse. La remarque
est juste et elle est enregistrée.

Elle est cependant **hors périmètre du tour 2**. Le comportement du logiciel
est inchangé — il ne fournit aucune assistance en temps réel sur argent réel,
et rien dans nos travaux ne va dans cette direction. Ce qui est reporté, c'est
le durcissement du test qui le garde. Le commanditaire a fixé la priorité :
faire marcher le logiciel d'abord. Nous y reviendrons quand la chaîne
principale fonctionnera de bout en bout ; inutile d'y consacrer du temps au
tour 2.

### 3.5 Le pari « personnel vs normatif » — nous demandons l'arbitrage au tour 2

C'est votre désaccord le plus conséquent. Opus 5 et GPT-5.6 soutiennent que
l'angle mort s'est refermé (GTO Wizard dispose d'un mode Analyze, mesure l'EV
loss, va « de la fuite au drill ») ; Gemini soutient que le pari tient. Vous
avez tous les trois raisonné **sans voir le code**.

Nous ne tranchons pas maintenant. La différenciation reformulée qui nous
paraît la plus défendable est celle de la synthèse : **local, privé,
francophone, mono-joueur, et surtout le seul à fermer la boucle diagnostic →
drill → retest espacé → preuve de correction.** Ce que les outils commerciaux
documentent comme un protocole que l'humain exécute à la main, un logiciel
personnel peut l'automatiser.

Confirmez ou infirmez au tour 2, code en main.

---

## 4. Le cadrage du tour 2 — d'abord, faire marcher

Avant les priorités, un cadrage que nous vous devons, parce qu'il détermine ce
qui nous est utile.

**En l'état, la chaîne principale ne fonctionne pas de bout en bout.** Un
utilisateur colle une capture d'écran : les cartes sont lues correctement
(mesuré : `5c` et `3h` sur une vraie table PMU, 2/2 affirmées, 1,1 s), puis
**tout s'arrête**. Le pot, la mise, le tapis et la blinde ne sont pas lus sur
l'image — `digit_ocr` existe mais n'est branché nulle part — donc l'utilisateur
doit les saisir à la main, et jusqu'à hier le logiciel rendait un verdict
calculé sur les valeurs par défaut du formulaire sans le signaler.

Le commanditaire l'exprime sans détour : *« ça ne fonctionne pas, alors
d'abord on veut un logiciel qui marche »*. Il a raison, et c'est ce qui
gouverne le tour 2.

**Ce que cela implique pour vous :** privilégiez, à valeur égale, ce qui rend
la chaîne opérante sur ce qui l'enrichit. Une objection qui nous empêche de
livrer un verdict faux nous sert plus qu'une proposition d'architecture, aussi
juste soit-elle. Et l'axe A garde la première place parce qu'un logiciel qui
marche mais qui calcule faux ne marche pas.

### Priorité 1 — Les calculs sont-ils justes ? (axe A, à refaire entièrement)

Avec le code sous les yeux cette fois. Cibles, dans l'ordre :

1. **`pfs/core/icm.py`** — Malmuth-Harville exact et Monte-Carlo,
   `bubble_factor`, PKO, FGS léger. Un correctif récent y a changé le
   traitement du tapis nul : un joueur à zéro touche désormais le dernier
   gain au lieu de zéro. **Vérifiez ce correctif et cherchez ce qu'il a pu
   casser.** Les valeurs de référence de l'ancienne implémentation sont
   retrouvées au dernier chiffre (5,021764845 ; heads-up exactement 1,0) —
   confirmez ou réfutez.
2. **`pfs/solver/pushfold.py`** — contre la table Nash HU de HoldemResources,
   **pas** contre SAGE.
3. **`pfs/core/equity.py`** — les valeurs archi-connues, avec vos sources.
4. **`pfs/solver/dcfr.py`** — sur Kuhn : valeur du jeu −1/18, famille continue
   d'équilibres α ∈ [0, 1/3], ratio K = 3α. Testez les dégénérescences
   DCFR → CFR / CFR+. Ajoutez Leduc si vous le pouvez.
5. **`pfs/data/hand_history.py`** — l'identité proposée par GPT-5.6
   (contributions − retours − rake = gains, **testée par main**) nous paraît
   la meilleure idée de la revue sur ce module. Cherchez les cas où notre
   reconstruction du pot se trompe.
6. **`pfs/analysis/session_review.py`** — l'équité all-in adjusted est
   calculée au moment où les jetons entrent. Cherchez une fuite temporelle.

Format : **CONFIRMÉ / RÉFUTÉ / NON VÉRIFIÉ**, avec valeur de référence,
source, écart mesuré. Toujours pas de quatrième catégorie.

### Priorité 2 — Comment brancher `digit_ocr`, concrètement

C'est **le maillon qui manque pour que la chaîne fonctionne**, et le maillon
absent est bien identifié : `TableRead` ne porte que des `CardBox`. La
perception n'a aucune notion de « où est l'étiquette du pot ».

Nous attendons de vous une méthode, pas un principe :

- Comment localiser les **bandeaux de texte** (pot, tapis, blinde) sur une
  capture de table, avec la même exigence de refus explicite que pour les
  cartes ? Nos captures réelles sont dans le dépôt.
- Quelle **structure de confiance** pour un montant ? Pour une carte nous
  avons distance et marge, et un vide mesuré entre vraies cartes (≤ 599) et
  non-cartes (≥ 658). Quel est l'équivalent pour un nombre ?
- Un OCR lisant « 1200 » au lieu de « 120 » **inverse un verdict**. Quelles
  vérifications de cohérence imposeriez-vous — le pot doit être cohérent avec
  les mises visibles, les tapis avec la blinde annoncée dans le titre de la
  fenêtre, etc. ?
- Quel **taux de refus** est acceptable ? Nous préférons dix refus à une
  lecture fausse, mais un outil qui refuse tout ne marche pas non plus.

### Priorité 3 — Attaquez ce que nous venons d'écrire

Deux artefacts neufs, à casser :

- `python/tests/test_parcours_complet.py` — prétend traverser les routes
  réelles. Vérifiez qu'il n'appelle aucune fonction interne en se prétendant
  de bout en bout, et **cassez le code qu'il prétend protéger** pour voir s'il
  tombe vraiment.
- `python/banc_invariants_icm.py` et `tests/test_icm_invariants.py`.

### Priorité 4 — Le recensement de code mort

GPT-5.6 signale à juste titre que notre chiffre de 31,7 % n'est pas prouvé
sans **couverture de branche instrumentée** : imports dynamiques, handlers,
appels JS, code exécuté à l'import. Nous publierons la couverture réelle
mesurée par le parcours complet. Dites-nous si notre méthode de comptage vous
paraît sonner juste, et ce qu'elle manque.

*Suite donnée le 11 août 2026.* Nous avons mesuré, et nous en avons d'abord
tiré une conclusion fausse — « le même 31,7 %, obtenu par une méthode
entièrement différente » — retirée depuis : un taux de lignes traversées ne
peut pas valider un taux de code mort, il en est le complément. Trois mesures
séparées les remplacent, chacune avec son banc rejouable (README,
§ « Trois nombres, trois méthodes ») : `banc_atteignabilite_statique.py`,
`banc_couverture_parcours.py`, `banc_inertie_causale.py`. La troisième cherche
ce que les deux autres ne peuvent pas voir — le code traversé dont le résultat
n'atteint jamais la sortie, la catégorie où vivait le défaut n°1.

### Priorité 5 — Aller devant, concurrent par concurrent

Question inchangée, mais désormais avec le code : **GTO Wizard, PioSOLVER,
MonkerSolver, HRC, ICMIZER, PokerSnowie** — nommément, un par un. Sur quoi
sommes-nous déjà devant, sur quoi derrière, qu'est-ce qui comble l'écart.

Et la question qui nous intéresse le plus, reposée telle quelle :
**qu'est-ce qui, dans la littérature récente, n'est encore dans aucun produit
commercial et tournerait sur un PC personnel ?**

---

## 5. Deux propositions retenues pour exécution immédiate

Elles ne demandent pas d'arbitrage, nous les mettons en œuvre :

- **Drill sans bouton CALL** (Gemini) — seuls FOLD et RAISE disponibles.
  Correction comportementale directe d'un tic mesuré, coût quasi nul.
- **Heatmap 13×13** superposant le jeu réel et la recommandation Nash, cases
  perdantes en rouge. Rend visuellement insupportable le bas de la grille.

---

## 6. Mode de travail pour la suite

Nous continuerons à vous solliciter, et nous vous devons deux choses en
retour.

**Ce que vous aurez à chaque tour :** le dépôt à jour, les mesures brutes des
bancs, et un compte rendu de ce que nous avons fait de vos recommandations —
y compris celles que nous n'avons pas suivies, avec la raison.

**Ce que nous vous demandons :** continuez à répondre séparément, sans vous
concerter. Vos désaccords nous ont plus appris que vos consensus. Trois fois
sur cette revue, la bonne réponse était celle d'un seul d'entre vous — le
biais de Harville, le piège SAGE, l'erreur d'équité — et nous ne l'aurions pas
eue avec un avis unique.

Gardez les deux sections obligatoires : « ce que je n'ai pas pu vérifier » et
« mon désaccord principal ». Et si vous pensez que nous nous trompons sur
l'un des points du §3, dites-le franchement : nous avons corrigé le chiffre
des 97 % parce que deux d'entre vous ont insisté.

Merci.

