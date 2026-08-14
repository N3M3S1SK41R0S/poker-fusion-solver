# Chantiers interrompus — à reprendre

Ces fichiers ont été écrits le 11 août 2026 par des agents dont l'exécution a
été **coupée en cours de route** par l'épuisement d'un quota hebdomadaire.
Ils ne sont donc ni finis ni vérifiés, et ils ne sont pas dans le chemin des
tests : les y laisser aurait rendu la suite rouge pour une raison qui n'a
rien à voir avec le logiciel.

Ils ne sont pas jetés pour autant — chacun porte du travail réel.

## ~~`banc_wsop.py` et `test_banc_wsop.py`~~ — TERMINÉ le 14 août 2026

Chantier clos, fichiers déplacés dans `python/` et `python/tests/`. Le banc
parcourt les 83 mains des World Series 2023 du corpus PHH et classe chaque
décision en trois catégories (erreur prouvable sans le recul / écart avec le
recul seul, nommé sophisme du résultat / désaccord sans preuve).

Les deux tests rouges de `TestTolerancePreflop` sont verts SANS toucher à la
tolérance. Départage par calcul indépendant : les deux tests inversaient la
cote :math:`\alpha = c/(P+c)` en :math:`c = \alpha P/(1-\alpha)` — inversion
juste quand :math:`P` est le pot GAGNABLE — puis passaient
``pot_gagnable = P + c``, comptant la mise adverse deux fois. La cote
produite valait :math:`\alpha/(1+\alpha)` au lieu de :math:`\alpha`, et
l'échec mesuré (0,26155 pour 0,35420 visé, or 0,35420/1,35420 = 0,26155) en
est la signature exacte. Les cibles étaient justes, la construction de la
décision synthétique était fausse ; `TOLERANCE_PREFLOP` n'a pas bougé d'un
point.

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

## ~~Ce qu'il reste à faire~~ — TOUT EST CLOS le 14 août 2026

1. ~~**La marche aléatoire absorbante**~~ — construite :
   `python/banc_marche_absorbante.py` + `python/tests/test_marche_absorbante.py`.
   Le biais de Harville est quantifié contre la marche à pas unitaires
   (politique documentée, ancres martingale et bistochastique mesurées) :
   Harville sous-estime les places intermédiaires du gros tapis (−2,3 pt sur
   « le leader finit 2ᵉ »), surestime celles du petit et sous-estime nettement
   sa dernière place (−5,4 pt à 4 joueurs, −6,3 à 6) ; en $EV, il survalorise
   les petits tapis de 0,2 à 0,8 % de la dotation. Le biais change de signe
   sous la politique « allin » : c'est une fonction de la dynamique supposée,
   pas un scalaire. Aucun défaut d'implémentation dans `pfs/core/icm.py` —
   c'est un biais de MODÈLE, documenté dans la docstring du banc.
2. ~~**La dégénérescence DCFR**~~ — le test avait déjà été réécrit au tour 5
   (`python/tests/test_dcfr_degenerescences.py`, 40 tests) : il MESURE les
   dégénérescences contre des réimplémentations de référence au lieu
   d'affirmer des noms. Vérifié le 14 août : `(1,1)` = Linear CFR (exact à
   1e-12), `(1,1)` ≠ Vanilla, `β = 0` = division par deux (pas CFR+), aucun
   paramètre fini ne donne CFR+ ni Vanilla (le poids de t = 1 vaut ½ pour
   tout exposant fini), γ n'implémente pas la loi du papier. Les docstrings
   de `pfs/solver/dcfr.py` disent désormais ce que les défauts font VRAIMENT
   (schedule actif par défaut qui ignore α, β, γ ; γ sur la contribution et
   non l'accumulateur).
3. ~~**Le banc WSOP**~~ ci-dessus.

Ce dossier ne contient plus de chantier ouvert.
