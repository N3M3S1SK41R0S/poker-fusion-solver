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

## ~~`test_icm_ordre_et_elimination.py`~~ — TERMINÉ le 14 août 2026

Chantier clos, fichier déplacé dans `python/tests/`. Les deux défauts sont
corrigés :

* **Seuil d'élimination PKO** : le repli d'inférence de `_vilain_elimine`
  (`pfs/core/icm.py`) ne compare plus le tapis résiduel à
  ``1e-12 · Σ tapis`` — la somme des tapis des AUTRES joueurs, étrangère au
  ``stack − bet`` qui produit le résidu — mais à l'échelle de la transaction
  elle-même : ``SEUIL_RESIDU_TRANSACTION (1e-5) · max(pot, bet)``, seuil
  encadré par deux bornes mesurées (bruit flottant et artefacts de saisie
  ≤ 3,3e-6 de la transaction ; plus petit vrai jeton ≥ 3,3e-5). Le jeton
  résiduel du tour 4 est maintenant lu comme l'artefact qu'il est, et un
  vrai tapis de 1 000 jetons reste un joueur vivant. `villain_all_in` et
  `unite_jeton` priment toujours.
* **Goldens de l'ordre d'élimination** : les quatre valeurs « avant »
  enregistrées à mi-chantier étaient les valeurs FAUSSES. Départagées par
  recalcul indépendant — Malmuth-Harville par énumération complète des
  8! = 40 320 ordres d'arrivée, sans import de `pfs.core.icm` — qui
  reproduit le code actuel à mieux que 1e-9 en relatif. Goldens corrigés,
  convention ancrée par un calcul à la main (BF exactement 1 contre 2 sur
  ``[100, 100, 0, 0]``, gains 50/30/20/10). Aucune tolérance n'a été
  élargie.

## Ce qu'il reste à faire, par ordre d'urgence

1. **La marche aléatoire absorbante** — la seule chose qui puisse valider un
   modèle d'ICM, puisqu'un banc d'invariants ne peut vérifier que
   l'appartenance à la famille, jamais le choix du membre. L'écart mesuré
   entre elle et Malmuth-Harville **est** le biais de Harville.
2. **La dégénérescence DCFR** : le test affirme que `α = β = γ = 1` donne
   CFR standard, ce qui est faux — cela donne Linear CFR. Vanilla CFR est la
   limite infinie, CFR+ est `(∞, 0, 2)`.
3. **Le banc WSOP** ci-dessus, avec ses deux tests de tolérance préflop.
