# Message d'accompagnement — à envoyer avec le brief

*(À copier tel quel dans Perplexity, aux trois modèles.)*

---

Bonjour,

Je vous sollicite tous les trois — Gemini 3.5 Pro, ChatGPT 5.6, Claude Opus 5 —
sur le même dossier, en parallèle et sans concertation. Je confronterai vos
réponses ensuite.

**Le code est ici, en accès libre :**
https://github.com/N3M3S1SK41R0S/poker-fusion-solver

Le brief détaillé se trouve à la racine du dépôt
(`BRIEF_CONSEIL_DES_MODELES.md`) et vous a été transmis. Lisez-le avant de
répondre : il contient l'état exact du projet, ce qui est déjà implémenté, et
le format de réponse attendu.

## Ce dont il s'agit

Poker Fusion Solver est un assistant de poker personnel, en français, écrit
pour un seul joueur — moi. Il analyse mes propres mains, juge mes décisions a
posteriori, et m'entraîne. Il tourne entièrement sur ma machine, ne se vend
pas, et ne fournit **jamais** d'assistance pendant une partie d'argent réel.

Environ 19 000 lignes de production, 10 000 lignes de tests, 1 095 tests verts.
Il a été écrit par un modèle sur plusieurs sessions, et il contient des
erreurs — cinq défauts sérieux ont été trouvés le jour même de la rédaction du
brief, dont deux qui rendaient des verdicts **faux** sans le signaler.

## Ce que j'attends de vous

**Trois choses, dans cet ordre.**

1. **Vérifiez que les calculs sont justes.** ICM, équilibre de Nash, équité,
   CFR, le parseur d'historiques. Confrontez-les à des références extérieures
   au dépôt. C'est le point le plus important : un solveur qui se trompe est
   pire qu'un solveur absent.

2. **Dites-moi ce qui est calculé mais jamais utilisé.** Trois bancs, trois
   mesures distinctes : **17 modules sur 62** ne sont importés par aucune
   chaîne partant de `python -m pfs` ; **69,1 % des lignes** ne sont
   traversées par aucun parcours utilisateur réel ; et une troisième mesure,
   par perturbation, cherche les calculs traversés dont le résultat n'atteint
   jamais la sortie. Ces trois nombres ne se confirment pas l'un l'autre — le
   README explique pourquoi les rapprocher serait une faute. Parmi le code
   mort : toute une chaîne de treize modules censée produire la décision
   finale, et qui ne s'exécute jamais. Je veux votre lecture indépendante, et
   surtout : **par où brancher ce qui ne l'est pas**.

3. **Dites-moi comment aller loin devant les solveurs existants.** GTO Wizard,
   PioSOLVER, MonkerSolver, HRC, ICMIZER, PokerSnowie. Sur quoi suis-je déjà
   devant, sur quoi suis-je derrière, et qu'est-ce qui comblerait l'écart.
   Ce qui m'intéresse le plus : ce qui existe dans la littérature récente et
   n'est encore dans aucun produit commercial.

## Deux points sur lesquels je ne veux pas être conseillé

**Je décide seul du périmètre.** Mes statistiques de jeu figurent dans le
brief et elles ne sont pas flatteuses : j'ai une fuite préflop mesurée et
constante. Ce n'est pas une raison pour me proposer de réduire l'ambition du
logiciel. Je veux l'outil le plus abouti possible, et j'apprendrai avec.
Toute réponse du type « commencez par corriger votre jeu, le reste est
prématuré » sera écartée : la question n'est pas s'il faut le construire,
mais comment le construire mieux.

**Aucune assistance en temps réel pendant une partie d'argent réel.** Cette
limite est inscrite dans le code et vérifiée par un test. Elle ne se discute
pas, et ce n'est pas une question de faisabilité. L'observation en argent
fictif, l'étude, les drills et l'analyse post-partie sont en revanche
pleinement dans le périmètre.

## Ce que je ne veux pas lire

- Une validation de ce que vous n'avez pas ouvert. Dites « non vérifié »,
  c'est une réponse utile.
- Des suggestions déjà implémentées. Le brief liste l'existant en détail ;
  me proposer d'ajouter de l'ICM me dira que vous ne l'avez pas lu.
- Un rapport uniquement positif. Le logiciel a vécu six semaines avec deux
  cartes interverties sous mille tests verts. Il reste des défauts de cette
  nature, et je compte sur vous pour en trouver.

Cinq objections précises valent mieux que trente suggestions génériques.

## Format et suite

Le brief détaille le format attendu. Deux sections y sont obligatoires :
« ce que je n'ai pas pu vérifier » et « mon désaccord principal avec le
brief ».

Vos désaccords entre vous m'intéressent davantage que vos consensus. Si vous
pensez qu'une autre lecture est défendable, dites-le et expliquez pourquoi
vous ne la retenez pas.

Une contrainte de mise en œuvre à garder en tête : tout doit tourner sur un
PC Windows personnel, sans GPU dédié, sans service externe, et rester lisible
par une seule personne.

Merci du temps que vous y consacrerez.

Pierre
