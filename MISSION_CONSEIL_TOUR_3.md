# Mission — conseil des modèles, tour 3

*(À envoyer aux trois : GPT-5.6 Sol Thinking, Claude Opus 5 Thinking, Gemini 3.1 Pro Thinking)*

**Dépôt public :** https://github.com/N3M3S1SK41R0S/poker-fusion-solver
**Archive :** `poker-fusion-solver-revue.zip` — **jointe littéralement à ce message**

---

## 0. L'accès, pour la troisième fois

Deux tours, deux échecs d'accès. Le tour 1 parce que le dépôt était encore
privé, le tour 2 parce que **l'archive était mentionnée mais pas jointe**.
Elle l'est cette fois, en pièce attachée : 284 fichiers, 1,1 Mo, brief et
bancs inclus.

Si votre outillage ne peut toujours pas la lire, **dites-le en première
ligne** et traitez ce qui suit comme un questionnaire. Ne produisez aucun
verdict CONFIRMÉ ou RÉFUTÉ sur du code que vous n'avez pas ouvert : vos deux
refus précédents étaient le bon comportement, et nous préférons un troisième
refus à une validation de complaisance.

---

## 1. Ce qui a changé depuis le tour 2

Vos retours ont été appliqués. Voici ce qu'ils ont produit, chiffres à
l'appui — et ce qu'ils ont coûté quand vous aviez tort.

### 1.1 Le projet a enfin des données réelles

C'était la lacune de fond : **tout ce qui avait été mesuré jusqu'ici l'avait
été sur des tables synthétiques.** L'utilisateur a lancé deux tournois en
parallèle et `capturer_session.py` (nouveau, versionné) a collecté
**57 captures distinctes** de deux tables PMU — 6-max et 7-max, feutres
différents, boards de 0 à 5 cartes, antes ou non.

Première mesure sur du réel :

```
325 boîtes détectées sur 57 frames
cartes qui comptent (héros + board) : 199 lues avec certitude sur 209 → 95 %
   héros : 98 lues, 10 refusées
   board : 101 lues,  0 refusée
dos adverses : 114 refusés sur 116 — comportement correct
localisation : 919 ms par frame
```

### 1.2 Un faux positif trouvé, et la mesure qui l'élimine

Deux des 116 « dos » ressortaient **sure**. Vérification à l'œil : ce sont
des cartes saisies en pleine **animation de retournement**, à moitié
recouvertes par leur dos brun. L'une était juste (A♠), l'autre **fausse** —
un 6 de trèfle lu « 6s », affirmé, à une distance de 94. La médiane des
pixels de fond mélangeait le vert du trèfle et le brun du dos, et tombait
près de l'ardoise du pique.

Une carte à fond plein est un **aplat**. Mesuré sur les 315 découpes réelles :

| population | dispersion des pixels de fond |
|---|---|
| 199 vraies cartes lues « sure » | **0,0** — min, médiane, p95 **et maximum** |
| 116 dos et décors | 20,2 à 72,4 |

Séparation totale. Le seuil se pose à 12, refuse les deux retournements, et
ne coûte aucune vraie carte. Les deux découpes réelles sont versionnées dans
`tests/donnees/`.

**Note de méthode, parce qu'elle vous concerne :** notre premier test de ce
contrôle *imitait* le cas en posant un rectangle de couleur sur un gabarit.
Il ne déclenchait rien — la couleur d'origine restait majoritaire, donc la
médiane des écarts restait nulle. Les vrais retournements sont diagonaux et
proches de moitié-moitié. L'imitation était plus commode et ne prouvait rien.

### 1.3 Le désaccord ICM du tour 2 est tranché — en faveur de l'implémentation

Nous avons lancé les quatre cas discriminants proposés par Claude Opus 5.
**Tous passent :**

| cas | résultat | attendu |
|---|---|---|
| `[100,100,0]` / `[50,30,0]` | `[40, 40, 0]` | le mort ne touche rien ✓ |
| `[100,0,0]` / `[50,30,20]` | `[50, 25, 25]` | les deux morts partagent 30+20 ✓ |
| winner-takes-all + mort | `[100, 0]` | aucun prix fantôme ✓ |
| WTA, 3 joueurs, 2 morts | `[100, 0, 0]` | idem ✓ |

Les deux morts touchent exactement la même chose : l'invariance par
permutation tient, l'invariance d'échelle aussi (`banc_invariants_icm.py`
rend « TOUS LES INVARIANTS TIENNENT », écarts à 1e-16). L'inquiétude d'Opus 5
était légitime — les trois conventions qu'il distingue sont réelles et deux
auraient faussé les facteurs de bulle en passant la conservation — mais elle
ne se matérialise pas ici. Gemini avait raison sur ce point.

### 1.4 « Heads-up = 1,0 » : Opus 5 avait raison sur nous

Nous l'avions présenté comme « la valeur théorique retrouvée au chiffre près,
la meilleure vérification du correctif ». **C'est faux.** À deux joueurs
l'ICM est linéaire et une fonction renvoyant `1.0` en dur passerait ce test.
Requalifié en identité algébrique dans le docstring, et le vrai contrôle de
non-dégénérescence est ajouté à n = 3.

**Et ce nouveau test nous a corrigés à son tour.** Nous y avions écrit qu'une
structure de gains **raide** met plus de pression qu'une plate. La mesure dit
l'inverse : **2,60 sur la plate contre 1,15 sur la raide**. C'est le logiciel
qui avait raison — quand les gains sont resserrés, monter d'une place
rapporte presque autant que gagner ; en winner-take-all la pression ICM
s'évanouit et le facteur tend vers 1.

### 1.5 Le chiffre de couverture que GPT-5.6 réclamait

Vous aviez raison de dire que « 31,7 % de code mort » n'était pas prouvé sans
couverture instrumentée. Nous l'avons mesurée, avec `coverage.py`, sur le
seul parcours utilisateur réel (`banc_couverture_parcours.py`) :

```
Modules mesurés        : 61
Jamais atteints        : 20
Lignes exécutables     : 9 151
Lignes traversées      : 2 900
COUVERTURE DU PARCOURS : 31,7 %
```

Coïncidence troublante : le même 31,7 %, obtenu par une méthode entièrement
différente. **Un tiers du code est traversé quand l'utilisateur s'en sert.**

### 1.6 Livré aussi

- `tests/test_parcours_complet.py` — part d'un fichier sur disque, traverse
  les **routes HTTP réelles** dans un processus distinct.
- `banc_invariants_icm.py` + `tests/test_icm_invariants.py` — échelle,
  permutation, conservation, monotonie, non-linéarité, exact ↔ Monte-Carlo,
  étendus au chemin PKO.
- `banc_contournements_live.py`.
- **1 296 tests verts** (contre 1 095 au tour 2), selftest conforme.

---

## 2. Ce que nous vous demandons — tour 3

### Priorité 1 — L'axe A, pour la troisième fois

Il n'a **jamais** été traité. C'est la seule chose dont dépend la crédibilité
du reste, et vous avez maintenant l'archive.

Cibles, par ordre :

1. **`pfs/core/icm.py`** — Malmuth-Harville exact et Monte-Carlo,
   `bubble_factor`, PKO, FGS léger. Le traitement du tapis nul a changé
   depuis le tour 2 ; `banc_invariants_icm.py` déclare tous les invariants
   tenus. **Cherchez ce qu'il ne teste pas.**
2. **`pfs/solver/pushfold.py`** — contre la table Nash HU de HoldemResources,
   pas contre SAGE (piège signalé par Opus 5, accepté).
3. **`pfs/core/equity.py`** — les valeurs archi-connues, avec vos sources.
   Gemini s'est trompé au tour 2 (22 vs AKs) ; la référence est ≈ 50,1 %.
4. **`pfs/solver/dcfr.py`** — Kuhn : valeur du jeu −1/18, famille continue
   d'équilibres α ∈ [0, 1/3], ratio K = 3α, dégénérescences DCFR → CFR/CFR+.
5. **`pfs/data/hand_history.py`** — l'identité `contributions − retours −
   rake = gains` **testée par main**, proposée par GPT-5.6, reste la
   meilleure idée reçue sur ce module. Elle n'est pas encore implémentée :
   dites-nous où elle casserait.
6. **`pfs/analysis/session_review.py`** — fuite temporelle dans l'équité
   all-in adjusted.

### Priorité 2 — La lecture des montants, avec des données cette fois

Votre convergence du tour 2 était nette et nous l'adoptons : **pas de
détecteur de texte**, mais une table de zones ancrée sur les cartes,
paramétrée par l'échelle, puisque l'interface PMU est un rendu déterministe.

Ce que les captures réelles apprennent et qui change la donne :

- **PMU affiche déjà tout en blindes** : « Pot : 7 BB », « 98 BB »,
  « 117,71 BB ». Aucune conversion jetons → blindes n'est nécessaire.
- Le format est régulier : décimale à la virgule, suffixe « BB », et les
  primes en euros avec suffixe « € ».
- Le titre de la fenêtre donne les blindes (« 75/150 ») et le niveau.

Questions précises :

1. **Comment ancrer la table de zones ?** Nous localisons les cartes du board
   à 100 %. Le pot est-il à un vecteur constant du centre du board, ou faut-il
   un autre point de référence — les cartes du siège, le bord du feutre ?
   Comment estimer l'échelle de façon robuste à partir d'une seule capture ?
2. **La contrainte de conservation des pixels** — largeur reconstruite du
   nombre contre boîte d'encre à ±2 px — est la meilleure idée du tour 2, et
   la seule qui attrape « 1200 » lu au lieu de « 120 ». Comment la formuler
   exactement quand la police a des chasses variables (le « 1 » est étroit) ?
3. **Quelles vérifications de cohérence en veto ?** Nous avons : le pot doit
   être un multiple de la plus petite dénomination, les tapis doivent être
   cohérents avec la blinde du titre, le pot ne peut pas décroître dans une
   main. Qu'ajouteriez-vous, et lesquelles sont réellement discriminantes ?
4. **Gemini signalait les abréviations** (`1.5k`, `2,3M`). Nos captures n'en
   contiennent aucune — PMU écrit en blindes. Est-ce une raison de ne pas les
   gérer, ou faut-il **échouer explicitement** sur tout glyphe non reconnu ?

### Priorité 3 — Attaquez ce que nous venons d'écrire

Trois artefacts neufs. Cassez-les.

- `tests/test_parcours_complet.py` : prétend traverser les routes réelles
  dans un processus distinct. Vérifiez-le, et **cassez le code qu'il prétend
  protéger** pour voir s'il tombe vraiment.
- `banc_invariants_icm.py` : déclare tous les invariants tenus. Trouvez celui
  qui manque.
- `DISPERSION_MAX = 12` : le seuil d'uniformité du fond. Il repose sur
  **deux** cas de retournement observés. Deux, c'est peu. Quelle autre
  contamination casserait ce contrôle — un curseur de souris sur une carte,
  une surbrillance de sélection, une carte en cours de distribution ?

### Priorité 4 — Aller devant, avec du concret

Question inchangée, mais l'angle a bougé. Le tour 2 a dégagé une
reformulation que nous retenons : **ce que les outils commerciaux ne peuvent
structurellement pas faire, c'est admettre l'ignorance.** Un « je ne sais
pas » est un défaut dans un produit payant ; c'est une qualité ici.

Alors :

1. **Où faut-il refuser, dans ce logiciel, et ne le fait-on pas encore ?**
   Nous refusons déjà une carte illisible, un montant non saisi, un spot
   dégénéré. Où d'autre le verdict est-il rendu avec une assurance qu'il n'a
   pas ? Le cas connu : un verdict qui bascule entre Harville et
   Malmuth-Weitzman devrait se déclarer fragile.
2. **GTO Wizard, PioSOLVER, MonkerSolver, HRC, ICMIZER, PokerSnowie** —
   nommément, un par un : sur quoi sommes-nous devant, sur quoi derrière,
   qu'est-ce qui comble l'écart.
3. **Qu'est-ce qui, dans la littérature récente, n'est encore dans aucun
   produit et tournerait sur un PC personnel ?** Gemini a proposé le Safe
   Nested Subgame Solving orienté adaptation ; nous avons objecté qu'avec
   ~12 mains par adversaire il n'y a rien à exploiter. Sa contre-proposition
   — le retourner vers le joueur pour calculer *son* image à la table — nous
   paraît meilleure. Développez, ou proposez mieux.

---

## 3. Deux points de cadrage, inchangés

**Le périmètre ne se rediscute pas.** L'utilisateur veut l'outil le plus
abouti possible et apprendra avec. La règle que nous avons adoptée du tour 1
tient : ce qu'il faut arrêter n'est pas la sophistication mais **l'ajout de
capacités non branchées** — aucun module dans `master` sans une route, un
élément d'interface, et un test de bout en bout traversant les trois.

**La priorité est de faire marcher.** La chaîne s'arrête encore après la
lecture des cartes, faute de lecture des montants. À valeur égale,
privilégiez ce qui rend la chaîne opérante sur ce qui l'enrichit.

---

## 4. Format

Inchangé, et les deux sections restent obligatoires : **« ce que je n'ai pas
pu vérifier »** et **« mon désaccord principal »**.

Continuez à répondre séparément, sans vous concerter. Sur les deux premiers
tours, la bonne réponse était trois fois celle d'un seul d'entre vous — le
biais de Harville, le piège SAGE, l'erreur d'équité — et nous ne l'aurions
pas eue d'un avis unique.

Et si vous pensez que nous nous trompons sur l'un des points du §1, dites-le :
nous avons corrigé le chiffre des 97 %, requalifié « heads-up = 1,0 » et
changé de statistique sur l'uniformité du fond parce que vous avez insisté.
