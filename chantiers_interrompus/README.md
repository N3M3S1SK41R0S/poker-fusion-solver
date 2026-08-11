# Chantiers interrompus — à reprendre

Ces fichiers ont été écrits le 11 août 2026 par des agents dont l'exécution a
été **coupée en cours de route** par l'épuisement d'un quota hebdomadaire.
Ils ne sont donc ni finis ni vérifiés, et ils ne sont pas dans le chemin des
tests : les y laisser aurait rendu la suite rouge pour une raison qui n'a
rien à voir avec le logiciel.

Ils ne sont pas jetés pour autant — chacun porte du travail réel.

## `banc_wsop.py` et `test_banc_wsop.py`

Banc destiné à parcourir les 83 mains des World Series 2023 du corpus PHH et
à classer chaque décision en trois catégories :

* **erreur prouvable sans le recul** — la décision est mauvaise contre toute
  range plausible, donc avec la seule information disponible au moment de
  jouer. C'est la seule catégorie où l'on peut dire qu'un professionnel s'est
  trompé ;
* **écart avec le recul seul** — notre conseiller aurait joué autrement et,
  les cartes étant connues, cela aurait rapporté davantage. **Ce n'est pas une
  erreur** : c'est le sophisme du résultat, et le banc doit le nommer comme
  tel ;
* **désaccord sans preuve**.

Deux tests échouent (`TestTolerancePreflop`), sur la tolérance appliquée aux
décisions préflop. À reprendre en même temps que le banc.

## `test_icm_ordre_et_elimination.py`

Tests du correctif d'ordre d'élimination signalé par la revue externe : quand
plusieurs joueurs sont à zéro, `icm_equities` leur donne la moyenne des gains
restants, alors que dans le chemin `bubble_factor` le héros est **par
construction** le dernier éliminé et devrait toucher le gain de la place
immédiatement inférieure aux vivants. L'écart mesuré est de 25 % sur
`ev_lose`, numérateur du facteur de bulle.

Quatre tests échouent, et l'écart est **minuscule** : 1,7140596 obtenu contre
1,714123 attendu, soit 6e-5 hors d'une tolérance de 1,7e-5. Ce sont des
valeurs de référence enregistrées à mi-chantier, contre une implémentation
qui a bougé ensuite — ou qui n'a jamais été terminée.

**Ne pas se contenter d'élargir la tolérance pour faire passer les tests.**
Il faut d'abord établir laquelle des deux valeurs est juste. Un golden qu'on
ajuste jusqu'à ce qu'il passe ne prouve plus rien, et c'est précisément le
travers que ce projet combat.

## Ce qu'il reste à faire, par ordre d'urgence

1. **Le seuil d'élimination PKO**, partiellement corrigé. La déclaration
   explicite (`villain_all_in`) fonctionne et refuse les contradictions, mais
   l'inférence sur un tapis résiduel se trompe encore : sur un tournoi à
   16,7 M de jetons, un vilain gardant **un jeton** n'est pas vu comme
   éliminé, la prime tombe à zéro et l'équité exigée passe de 36,6 % à
   42,2 %. Un chiffre faux et plausible, dans le régime où l'utilisateur
   joue.
2. **L'ordre d'élimination** ci-dessus.
3. **La marche aléatoire absorbante** — la seule chose qui puisse valider un
   modèle d'ICM, puisqu'un banc d'invariants ne peut vérifier que
   l'appartenance à la famille, jamais le choix du membre. L'écart mesuré
   entre elle et Malmuth-Harville **est** le biais de Harville.
4. **La dégénérescence DCFR** : le test affirme que `α = β = γ = 1` donne
   CFR standard, ce qui est faux — cela donne Linear CFR. Vanilla CFR est la
   limite infinie, CFR+ est `(∞, 0, 2)`.
