"""Détection automatique des cartes dans une capture de table.

Aucune coordonnée codée en dur : les positions changent avec la room, le
thème, la résolution et le nombre de joueurs. On cherche donc les cartes par
leur FORME et leur COULEUR, puis on les range par géométrie.

Méthode retenue : **une carte est un rectangle dont les quatre côtés sont des
arêtes droites**. On repère les segments verticaux longs, on les apparie, on
recale le haut et le bas sur les arêtes horizontales, puis on trie. Une seule
exception, et elle est chiffrée plus bas : une carte que l'interface RECOUVRE
par le bas n'a plus qu'un rectangle VISIBLE, plus large que haut — on le rend
tel quel, sans jamais deviner ce qui est caché.

Ce qui a échoué, et pourquoi (gardé ici pour ne pas y revenir) :
  · « tout ce qui n'est pas le feutre » — une table a au moins deux tons
    (feutre + ovale central) : l'ovale et les cartes fusionnent en une seule
    tache. Mesuré : 0 % ;
  · remplissage des contours — le bord de l'ovale est lui aussi une boucle
    fermée, la table entière se solidifie ; et les cartes du héros, à cheval
    sur ce bord, voient leur contour fusionner avec lui. Mesuré : 58 % ;
  · régions délimitées par les contours — les glyphes découpent l'intérieur
    des cartes en fragments. Mesuré : 14 % ;
  · segmentation par couleurs quantifiées (6 niveaux/canal, soit des paliers
    de 43) — une carte BLANCHE (250) et l'ovale d'un tapis clair (229) tombent
    dans le même palier et fusionnent. Mesuré : 81,5 %, dont 49 % sur tapis
    clair, avec 50 % de boîtes fantômes ;
  · régions UNIES délimitées par le gradient — sur le deck classique, les pips
    de la colonne centrale coupent la marge blanche en deux demi-cartes de
    rapport 0,37, hors gabarit. Mesuré : 23 %, et 0,4 % sur le deck classique.

Ce qui a fait la différence, dans l'ordre où ça a été mesuré :
  1. le gradient LOCAL sépare là où la couleur ne sépare pas : entre un carton
     blanc et l'ovale d'un tapis clair, l'écart RGB n'est que de 37 (même
     palier de quantification) mais le gradient au bord vaut 26 contre 0 à
     l'intérieur ;
  2. les COINS ARRONDIS raccourcissent l'arête verticale de ~4 px : sans
     recalage sur les arêtes horizontales, la vérification tombait à côté du
     vrai bord (pmu_solid sur gris moyen : 0/52 → 52/52) ;
  3. une boîte à cheval sur deux cartes voisines est plus GRANDE que chacune,
     donc elle gagnait la suppression des doublons et les éliminait toutes les
     deux ; on exige que trois des quatre abords soient calmes (peu d'arêtes),
     ce qui la disqualifie. Le mécanisme n'est PAS un tri des candidates :
     instrumenté sur les 68 893 candidates qui atteignent ce test, exiger
     trois abords calmes en garde 26,4 % parmi les bonnes (IoU ≥ 0,5) et
     28,0 % parmi les fantômes (IoU < 0,3) — sélectivité nulle, et même
     légèrement à l'envers. Ce que ce test change, c'est ce qui SURVIT au
     dédoublonnage. L'ablation le montre — QUIET_SIDES
     → localisation / fantômes, sur les 144 tables du fichier de tests :
     0 → 98,4 % / 5,0 % ; 2 → 98,2 % / 1,9 % ; 3 → 98,8 % / 0,0 % ;
     4 → 76,6 % / 0,0 % (à 4, le simple voisinage de la carte d'à côté
     disqualifie une vraie carte, et la localisation s'effondre) ;
  4. sous bruit, l'arête ondule d'un pixel : mesurer la couverture sur une
     bande de trois lignes plutôt que sur une seule récupère les dernières
     pertes (98,6 % → 99,1 %), mais fait entrer les avatars de siège ;
  5. les distracteurs d'une vraie table sont des piles de jetons (striées) et
     des avatars de siège (carrés). Le plafond de densité intérieure écarte
     les premières, la borne de rapport les seconds, mais les deux bornes ne
     se valent pas : celle du rapport sépare vraiment sur ce banc (0,82 contre
     0,846 pour le premier avatar, soit 2,6 points de marge et zéro carte
     perdue), celle de la densité tranche DANS une seule population et coûte
     des cartes. Le prix des deux est chiffré juste en dessous ;
  6. une carte du héros peut être RECOUVERTE par le bas — bandeau d'équité,
     pavé du joueur — et c'est le cas de la première capture PMU réelle du
     dépôt (`tests/donnees/pmu_hero_tronque.png`). Ce n'est PAS un problème
     de contraste, contrairement à ce que suggère l'intuition « carte verte
     sur feutre vert » : au bord gauche de cette carte-là le gradient vaut 76
     en médiane et 54 au p10 contre un seuil de 13 (bruit résiduel nul, PNG),
     et 98,1 % des lignes du bord le dépassent — le liseré clair du pourtour
     en porte la moitié. Les segments verticaux sont trouvés, aux bonnes
     colonnes. Le défaut est GÉOMÉTRIQUE : le bandeau coupe la carte à 68 %
     de sa hauteur, sa boîte visible mesure 121 × 115 soit un rapport de 1,05
     là où une carte vaut 0,716, et l'arête verticale d'un seul de ses deux
     bords se prolonge le long du bandeau (126 px contre 100). L'appariement
     était donc rejeté AVANT même le test de rapport, par la règle « les deux
     côtés couvrent la même hauteur ». Ce qu'on rend alors n'est pas la
     carte : c'est sa partie VISIBLE.

Le prix des deux bornes, mesuré sur les 144 tables du fichier de tests
(672 cartes) :

  · INNER_MAX = 0,72. Les densités d'arêtes se RECOUVRENT, il n'y a rien à
    séparer. Les 672 vraies cartes mesurent 0,463 en médiane, 0,661 au p90,
    0,777 au p99 et 0,819 au maximum ; les piles de jetons, telles que le
    détecteur les cadre, lisent 0,769 et 0,775 — donc 11 vraies cartes sur
    672 (1,6 %) sont PLUS striées que le plus calme des tas de jetons. Aucun
    plafond ne garde toutes les cartes en rejetant tous les jetons. Celui-ci
    se place sous toute la zone de recouvrement et coûte les 4,8 % de cartes
    qui le dépassent. Balayage (INNER_MAX → localisation / fantômes) :
    0,72 → 98,8 % / 0 ; 0,75 → 99,3 % / 12 ; 0,78 → 99,7 % / 24 ;
    0,80 → 99,7 % / 132 ; 0,82 → 99,7 % / 180. Chacun de ces fantômes est
    une pile de jetons ; on paie 0,9 point de localisation pour les zéro ;
  · MAX_RATIO = 0,82. Les fantômes qui apparaissent quand on desserre cette
    borne sont TOUS des avatars de siège (25/25 à 0,85, 49/49 à 0,88) : leur
    boîte recalée mesure 0,846 ou 0,868–0,872 selon l'arête sur laquelle le
    recalage tombe. Balayage : 0,88 → 4,7 % de fantômes, 0,85 → 2,5 %,
    0,82 → 0,0 %, à localisation inchangée (98,8 %). Celle-là ne coûte rien
    sur ce banc.

Le chemin « carte coupée par le bas » et le prix de ses trois conditions,
mesurés sur les mêmes 144 tables — les trois se rejouent en remettant une
constante à sa valeur neutre :

  · CUT_MIN_COVER = 0,60 (neutre : 0,85, c'est-à-dire chemin fermé). Fermé,
    le banc donne 664/672 localisées, 659/672 rôles et 0/986 fantômes, et la
    capture PMU ne rend AUCUNE boîte. Ouvert, le banc donne exactement les
    mêmes chiffres. Il n'est pas inerte pour autant : sur 5 tables des 144
    (habillage classique, tapis « bleu nuit » ou « noir », où le libellé de
    tapis sous les cartes du héros fait office de recouvrement), une boîte
    devient plus trapue — rapports 0,960 à 1,028 au lieu de ≤ 0,82, cadrage
    élargi au vide entre les deux cartes. Chacune reste posée sur une vraie
    carte du héros (IoU 0,66 à 0,72), le nombre de boîtes est identique, et
    aucun compte ne bouge. C'est le prix mesuré du chemin sur ce banc ;
  · CUT_TOP_ALIGN = 0,06 (neutre : 1,0, aucun alignement exigé). La coupure
    vient du bas, donc les deux arêtes partent du même bord haut. Sans cette
    condition, le chemin apparie un bord de plaque de siège avec une arête de
    glyphe : le banc passe à 665/672 mais fait entrer **39 fantômes** sur
    1 026 boîtes. C'est de loin la plus utile des trois. Le cas d'école : une
    boîte 40 × 37 de rapport 1,08 sur le tapis noir, dont les deux arêtes sont
    décalées de 5 px EN HAUT pour une hauteur de 32 ;
  · CUT_BACK_MIN = 32. Une carte entière a le même fond au-dessus et en
    dessous ; une carte coupée a la table au-dessus et l'élément qui la
    recouvre en dessous (43,7 sur la capture PMU). Sans cette condition le
    banc passe à 665/672 mais fait entrer 3 fantômes, tous des avatars de
    siège (boîtes 33 × 33 et 33 × 34, rapports 0,971 à 1,000). On paie donc
    une carte sur 672 pour ces trois-là.

Les deux bornes de rapport du chemin coupé, elles, se lisent sur le banc
`banc_localisation.py --large` (2560 × 1529, cinq configurations) :

  · CUT_MIN_RATIO = 0,92. Sans plancher — c'est-à-dire en acceptant tout ce
    qui dépasse MAX_RATIO = 0,82 — la configuration bruitée à pleine échelle
    gagnait UN fantôme (83 boîtes fantômes au lieu de 82 sur 335), une boîte
    58 × 66 de rapport 0,879 posée sur rien. Le plancher se place dans le vide
    mesuré entre ce 0,879 et la plus trapue des boîtes conservées (0,960) ;
    au-dessus, on croit à une coupure, en dessous on lit un cadrage élargi ;
  · CUT_MAX_RATIO = 1,10 ne se justifie PAS par le banc : desserrée à 1,30
    elle donne les mêmes 664/672 et 0/986 fantômes. Ce n'est pas une borne
    anti-fantôme, c'est une borne de SENS — au-delà, la partie cachée dépasse
    la partie visible. Elle est posée juste au-dessus du seul cas réel mesuré
    (1,052), avec 5 % de marge.

Ces deux bornes, prises ensemble, disent quelle PART d'une carte peut être
recouverte pour qu'on la retrouve encore : de 22 % à 35 % de sa hauteur, en
supposant la carte au rapport standard 0,716. En dessous de 22 % le chemin ne
s'ouvre pas (la carte reste vue comme entière et son rapport la fait rejeter),
au-dessus de 35 % elle est perdue. La capture PMU est à 32 % — dans la bande,
mais près du haut. C'est la limite la plus étroite de ce module.

Dernier effet du chemin coupé, sur l'attribution des rôles cette fois :
comparer les échelles sur la HAUTEUR faisait perdre son rôle au board dès que
la main du héros était recouverte (0/18 sur les six tapis du banc, rapport de
hauteurs 1,315 pour une tolérance de 1,30). La comparaison porte donc sur la
LARGEUR, qu'une coupure par le bas ne touche pas : 18/18, et le banc entier
est inchangé au chiffre près (664/672 localisées, 659/672 rôles, 0/986
fantômes, 0/720 dos promus, avant comme après).

Taux mesurés sur le banc du fichier de tests (`tests/test_table_detector.py`,
144 tables = 2 habillages × 6 tapis × 3 tailles de board × 2 échelles de carte
× 2 niveaux de bruit, avec sièges, jetons, boutons, montants et cartes face
cachée). Ce banc-là est DANS le dépôt : chaque chiffre ci-dessous se rejoue.

  · **98,8 %** des cartes du héros et du board localisées (664/672, IoU ≥ 0,5) ;
  · **98,1 %** de rôles héros/board corrects (659/672), et aucun dos de carte
    adverse promu en carte du héros ou du board (0/720) ;
  · **0,0 %** de boîtes fantômes (aucune carte inventée sur 986 boîtes).

Sur la SEULE capture réelle dont on dispose (PMU PLAY, 1899 × 1348, tournoi
6-max, habillage « quatre couleurs à fond plein » absent des gabarits), les
deux cartes du héros — recouvertes par le bandeau d'équité — sont localisées
et étiquetées « hero », et les deux dos adverses restent dans « autres ».
Avant le chemin « carte coupée », `read_table` ne rendait que ces deux dos,
et les étiquetait « hero » faute de rien trouver de plus bas. Une capture
n'est pas un banc : ce résultat dit que le cas est traité, il ne dit rien
d'un taux.

Quel cas est le plus dur dépend du banc, ce n'est pas une propriété du
détecteur : sur celui-ci, l'habillage plein est à 100 % sur les six tapis et
tout l'écart vient du deck classique — gris moyen 94,6 %, tapis clair 96,4 %,
les quatre autres tapis 98,2 % ou plus. Coût mesuré sur une table 900 × 560 :
0,17 s pour cinq cartes, 0,24 à 0,35 s en moyenne sur le banc — le coût est
dominé par l'appariement des segments, donc par leur NOMBRE. Le chemin
« carte coupée » élargit la fenêtre d'appariement de MAX_RATIO à
CUT_MAX_RATIO, soit +34 % de colonnes candidates : mesuré sur cette même
table (meilleur de 9 passes, chemin ouvert puis fermé puis rouvert pour
neutraliser la charge machine), 0,217 s contre 0,202 s, soit +7 %.

Limites assumées, chiffrées :
  · **compression JPEG** — le banc est en PNG, une capture d'écran réelle ne
    l'est presque jamais. Recompressé, il donne q = 95 → 96,6 %, q = 90 →
    95,5 %, q = 85 → 93,3 %, q = 75 → 88,1 %, q = 60 → 75,0 % de
    localisation ; les fantômes, eux, restent à 0,0 % à toutes les qualités.
    Le critère « ≥ 95 % des cartes localisées » de la phase A tient jusqu'à
    q = 90 et tombe dès q = 85. Sur la tranche la plus fragile du banc
    (habillage plein, cartes de 52 px, sans bruit) la chute est bien plus
    brutale : 100 % en PNG, 95,2 % à q = 90, 63,1 % à q = 75. C'est le
    premier chantier de la validation sur vraies captures ;
  · un libellé collé sous une carte la fait perdre — le banc écrit le tapis du
    héros 16 px sous ses cartes ; à 6 px, ses glyphes tombent dans la bande
    d'abord basse et la localisation passe de 98,8 % à 97,0 % ;
  · deux cartes qui se chevauchent au point de masquer une arête verticale
    entière ne donnent qu'une seule boîte — c'est signalé par le nombre de
    cartes trouvées, jamais deviné ;
  · le rapport accepté est 0,60–0,82 pour une carte entière (la carte à jouer
    vaut 0,716, soit ±15 %), et 0,92–1,10 pour une carte coupée par le bas.
    Un habillage hors de ces plages serait ignoré : c'est un réglage, pas une
    réécriture ;
  · une carte coupée par le bas est rendue par sa partie VISIBLE. Sa boîte
    n'est donc PAS la carte : sur la capture PMU elle mesure 126 × 115 pour
    une carte de 121 px de large, décalée de 8 px sur la carte de droite —
    le dédoublonnage garde la plus GRANDE des boîtes concurrentes, et la
    plus grande est ici celle qui déborde sur la voisine (recouvrement 1,00
    et 0,98 de la vérité-terrain, IoU 0,92 et 0,90). Pour la reconnaissance
    du rang et de la couleur, qui lit le coin haut-gauche, c'est sans effet ;
    pour un futur découpage plein cadre, ça ne l'est pas ;
  · la part recouverte doit tomber entre 22 % et 35 % de la hauteur de la
    carte — en supposant celle-ci au rapport standard 0,716, car sa vraie
    hauteur n'est pas observable : le bas est caché. Sous cette hypothèse la
    capture PMU est à 32 %, près du haut de la bande. Une carte à peine
    entamée (moins de 22 %) est vue comme entière et rejetée sur son rapport ;
    une carte plus qu'à moitié cachée est perdue. C'est la première chose à
    re-chiffrer sur une deuxième capture réelle ;
  · l'élément qui recouvre doit TRANCHER sur la table (CUT_BACK_MIN = 32).
    Un bandeau de la teinte du tapis rendrait la coupure invisible et la
    carte serait perdue — mesuré en peignant un bandeau (22, 38, 64) sur les
    six tapis du banc : il échoue sur « bleu nuit » et « noir » ;
  · un rectangle uni au rapport ET à la taille d'une carte reste indiscernable
    d'une carte à ce stade ; c'est la reconnaissance (`card_recognizer`) qui
    tranche ensuite ;
  · **SUR DE VRAIES CAPTURES, LA LOCALISATION TOMBE À 76,7 %.** Tous les taux
    ci-dessus sont mesurés sur des tables SYNTHÉTIQUES : pas de dégradé de
    feutre, pas d'ombre portée, pas de décor lumineux, un seul jeu de
    gabarits. Mesuré le 11 août 2026 contre la vérité-terrain de 57 captures
    PMU (`banc_verite_captures.py`, `tests/donnees/verite_captures.json`) :
    **198 cartes localisées sur 258 réellement présentes**, et 30 cartes du
    board promues « hero » faute d'avoir trouvé le siège. Les 60 cartes
    perdues se répartissent ainsi :

      - **45 cartes du HÉROS**, dont les 42 de la table 7-max (24 captures
        sur 25). La carte est parfaitement visible, ses deux arêtes
        verticales sont trouvées aux bonnes colonnes, `_snap_to_edges` rend
        (839, 924, 117, 115) soit un rapport de 1,017 — dans la plage
        « carte coupée » — et `_covered_from_below` répond True. C'est
        `_looks_like_a_card` qui refuse, et précisément **QUIET_SIDES** :
        seuls 2 abords sur 4 sont calmes (haut 0,00 / bas 0,22 / gauche 0,52
        / droite 0,77). L'habillage « KO » de cette table pose un rail
        lumineux horizontal juste à côté du siège, et une pastille de prime
        « 2,25 € » à cheval sur la carte de gauche. Instrumenté sur la
        capture `300_7-max_KO/0014` : 8 candidates bien cadrées meurent sur
        ce seul test, contre 19 GARDÉES pour la même carte sur la table
        6-max ;
      - **15 cartes du BOARD** (5♠ ×11, 9♣ ×4) : la pile de jetons du pot est
        posée sur leur bas ; le recalage horizontal part sur l'arête des
        jetons, la boîte passe de 160 à 197 px de haut et son rapport (0,594)
        tombe juste sous MIN_RATIO.

    Ablation mesurée sur ces mêmes captures
    (`banc_verite_captures.py --quiet-sides 2`) : localisation 76,7 → 96,9 %,
    rôles justes 65,1 → 96,9 %, **0 carte inventée**, 8 cartes encore perdues
    (celles sous les jetons). Sur les 144 tables synthétiques, la même
    ablation coûtait 1,9 % de boîtes fantômes — d'où QUIET_SIDES = 3. Les
    deux mesures ne pointent pas dans le même sens, et c'est **la mesure sur
    le réel qui n'existait pas** quand la constante a été posée. Le
    desserrement n'est pas appliqué ici : il demande un chantier qui
    re-mesure les deux bancs ensemble ;
  · rien n'appelle encore ce module hors de son banc et de ses tests : la
    fonctionnalité n'est PAS disponible dans l'application.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pfs.vision.phash import _rgb_array

__all__ = ["CardBox", "TableRead", "detect_card_boxes", "read_table"]

# Une carte à jouer vaut 63/88 = 0,716. La borne haute est fixée par les
# avatars de siège, et par eux seuls : desserrée à 0,85 puis 0,88, elle fait
# entrer 25 puis 49 fantômes, tous des avatars (boîtes recalées à 0,846 et
# 0,868–0,872), sans gagner une seule carte.
MIN_RATIO, MAX_RATIO = 0.60, 0.82
MIN_AREA_FRAC = 0.0006   # part de l'image (rejette les pastilles et icônes)
MAX_AREA_FRAC = 0.10

# Seuil d'arête. 13 est le plancher : le contraste le plus faible mesuré au
# bord d'une carte (carton blanc sur ovale de tapis clair) vaut 26.
EDGE_MIN = 13.0
NOISE_K = 5.0            # multiple du bruit résiduel, quand il domine 13
SMOOTH_ABOVE = 1.5       # au-delà, on lisse avant de dériver
BRIDGE = 9               # comble les coupures d'arête (liseré clair traversant)

SIDE_COVER = 0.72        # part de la largeur couverte par l'arête haute/basse
SIDE_CONTRAST = 16.0     # écart de couleur dedans/dehors, par côté
QUIET_MAX = 0.25         # densité d'arêtes tolérée dans un abord « calme »
# Nombre d'abords calmes exigés sur 4.
#
# C'était LA constante qui coûtait le plus cher sur de vraies captures. À 3,
# elle rejetait 45 des 60 cartes perdues du banc de vérité-terrain — dont les
# deux cartes du héros sur presque toute la table 7-max, parce qu'un décor
# lumineux borde le siège et qu'une pastille de prime se pose sur la carte de
# gauche. Deux abords bruyants suffisaient à écarter la carte.
#
# LE PASSAGE À 2 A ÉTÉ TENTÉ, MESURÉ, ET ANNULÉ. L'épisode mérite d'être
# écrit, parce que l'erreur de méthode est plus instructive que le réglage.
#
# Le gain sur les captures réelles est considérable et il est réel :
#
#     QUIET_SIDES                3           2
#     RÉEL (57 captures annotées, 258 cartes)
#       rappel de lecture      76,7 %      96,9 %
#       dont bon rôle          65,1 %      96,9 %
#       précision             100,0 %     100,0 %
#
# Un banc maison (`banc_localisation.py --large`, 5 configurations à pleine
# échelle) annonçait un coût de DEUX boîtes fantômes sur 262. Sur cette base,
# la valeur a été passée à 2 — puis la suite de tests a rejeté le changement
# avec DOUZE échecs, dont `test_no_phantom_box` : **8 boîtes fantômes pour
# 56 cartes** là où l'on en attend zéro.
#
# Les deux mesures ne se contredisaient pas, elles ne comptaient pas la même
# chose : le banc maison comparait les boîtes à TOUTES les cartes du tableau,
# dos adverses compris, donc une boîte posée sur un dos n'y était pas un
# fantôme. La suite de tests, elle, mesure sur des tables SANS adversaire, où
# toute boîte surnuméraire en est un.
#
# La leçon, qui est celle de tout ce module : deux chiffres obtenus sur des
# bancs différents ne se comparent pas, et un gain mesuré sur un banc ne
# justifie pas un réglage tant que l'autre n'a pas été rejoué. Le rappel de
# 76,7 % reste un vrai défaut ; sa correction passe par une règle qui
# distingue un décor lumineux d'un vrai bord de carte, pas par un
# assouplissement global de celle-ci.
QUIET_SIDES = 3
INNER_MAX = 0.72         # densité d'arêtes tolérée dans le carton (jetons : 0,83)
NMS_COVER = 0.55         # recouvrement relatif au-delà duquel on ne garde qu'une

# Une carte peut être RECOUVERTE par un élément d'interface posé dessus
# (bandeau d'équité, pavé du joueur) : seule sa partie haute reste visible. Sa
# boîte est alors plus LARGE que haute, et une seule de ses deux arêtes
# verticales garde la bonne longueur — l'autre se prolonge le long du bord de
# l'élément qui la recouvre. Valeurs mesurées sur la capture PMU du dépôt
# (`tests/donnees/pmu_hero_tronque.png`) : rapport recalé 1,052 et boîte
# couvrant 0,79 de la plus longue des deux arêtes.
CUT_MIN_RATIO = 0.92     # en deçà, c'est un cadrage élargi, pas une coupure
CUT_MAX_RATIO = 1.10     # rapport apparent maximal d'une carte coupée par le bas
CUT_MIN_COVER = 0.60     # part de la plus longue arête que la boîte doit couvrir
CUT_TOP_ALIGN = 0.06     # décalage toléré entre les deux arêtes, EN HAUT
CUT_BACK_MIN = 32.0      # écart de couleur exigé entre le fond du haut et du bas


@dataclass(frozen=True, slots=True)
class CardBox:
    """Une carte localisée : boîte (x, y, w, h) et son rôle présumé."""

    x: int
    y: int
    w: int
    h: int
    role: str = "?"          # « hero » | « board » | « ? »

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass(slots=True)
class TableRead:
    """Ce que la détection a compris de la table."""

    hero: list[CardBox]
    board: list[CardBox]
    others: list[CardBox]

    @property
    def all(self) -> list[CardBox]:
        return self.hero + self.board + self.others


def _noise_sigma(a: np.ndarray) -> float:
    """Écart-type du bruit, par MAD des différences horizontales.

    La MAD ignore les vraies transitions (rares et fortes) et ne retient que
    la dispersion de fond : c'est elle qui doit fixer le seuil d'arête, sinon
    une capture bruitée le fait exploser.
    """
    d = np.diff(a[..., 1], axis=1).ravel()
    mad = float(np.median(np.abs(d - np.median(d))))
    return mad / 0.6745 / np.sqrt(2.0)


def _run_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longueur du run vertical de chaque pixel, et le masque des débuts."""
    n, w = mask.shape
    idx = np.arange(n, dtype=np.int32)[:, None]
    prev = np.vstack([np.zeros((1, w), bool), mask[:-1]])
    nxt = np.vstack([mask[1:], np.zeros((1, w), bool)])
    start = np.maximum.accumulate(np.where(mask & ~prev, idx, 0), axis=0)
    end = np.minimum.accumulate(
        np.where(mask & ~nxt, idx, n - 1)[::-1], axis=0)[::-1]
    return np.where(mask, end - start + 1, 0).astype(np.int32), prev


def _vertical_segments(mask: np.ndarray, lmin: int) -> np.ndarray:
    """Segments verticaux (colonne, y_début, y_fin) d'au moins `lmin` pixels."""
    lengths, prev = _run_lengths(mask)
    ys, xs = np.where(mask & ~prev & (lengths >= lmin))
    if not ys.size:
        return np.zeros((0, 3), dtype=np.int32)
    return np.stack([xs, ys, ys + lengths[ys, xs] - 1], axis=1).astype(np.int32)


def _edge_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masques des arêtes verticales et horizontales, seuil calé sur le bruit."""
    H, W = rgb.shape[:2]
    a = rgb
    if _noise_sigma(rgb) >= SMOOTH_ABOVE:
        a = ndimage.gaussian_filter(rgb, sigma=(1.0, 1.0, 0))
    t = max(EDGE_MIN, NOISE_K * _noise_sigma(a))

    gx = np.abs(a[:, 2:] - a[:, :-2]).max(axis=2)
    gy = np.abs(a[2:, :] - a[:-2, :]).max(axis=2)
    vm = np.zeros((H, W), bool)
    hm = np.zeros((H, W), bool)
    vm[:, 1:-1] = gx > t
    hm[1:-1, :] = gy > t
    # Un liseré clair qui traverse le bord d'une carte (l'ovale de la table)
    # coupe l'arête en deux : sans ce pontage, les cartes du héros posées sur
    # ce liseré étaient perdues.
    vm = ndimage.binary_closing(vm, np.ones((BRIDGE, 1)))
    hm = ndimage.binary_closing(hm, np.ones((1, BRIDGE)))
    return vm, hm


def _snap_to_edges(hm: np.ndarray, box: tuple[int, int, int, int]
                   ) -> tuple[int, int, int, int] | None:
    """Recale le haut et le bas sur les vraies arêtes horizontales.

    Les coins ARRONDIS raccourcissent l'arête verticale de quelques pixels :
    le segment s'arrête avant le bord réel. On cherche donc l'arête horizontale
    dans un voisinage et on retient la plus EXTÉRIEURE qui couvre la largeur.

    La couverture se mesure sur une bande de trois lignes, pas sur une seule :
    sous bruit, l'arête ondule d'un pixel et aucune ligne seule n'atteignait le
    seuil — toutes les pertes résiduelles venaient de là.
    """
    H = hm.shape[0]
    x, y, w, h = box
    reach = max(3, int(round(h * 0.12)))
    found: list[int | None] = []
    for anchor in (y, y + h - 1):
        edge = None
        for d in range(reach, -reach - 1, -1):
            yy = anchor + (-d if anchor == y else d)
            if not (0 <= yy < H):
                continue
            band = hm[max(0, yy - 1):yy + 2, x:x + w]
            if band.size and band.any(axis=0).mean() >= SIDE_COVER:
                edge = yy
                break
        found.append(edge)
    top, bottom = found
    if top is None or bottom is None or bottom - top < 14:
        return None
    return (x, top, w, bottom - top + 1)


def _rect(a: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Sous-tableau borné à l'image (vide si la fenêtre sort du cadre)."""
    H, W = a.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return a[y0:y1, x0:x1] if (x1 > x0 and y1 > y0) else a[0:0, 0:0]


def _side_windows(box: tuple[int, int, int, int]
                  ) -> tuple[list[tuple[int, int, int, int]],
                             list[tuple[int, int, int, int]]]:
    """Les quatre bandes juste dedans et les quatre juste dehors."""
    x, y, w, h = box
    k = max(2, min(w, h) // 14)
    q = k + 2
    inside = [(x + q, y + 2, w - 2 * q, k), (x + q, y + h - 2 - k, w - 2 * q, k),
              (x + 2, y + q, k, h - 2 * q), (x + w - 2 - k, y + q, k, h - 2 * q)]
    outside = [(x + q, y - 1 - k, w - 2 * q, k), (x + q, y + h + 1, w - 2 * q, k),
               (x - 1 - k, y + q, k, h - 2 * q), (x + w + 1, y + q, k, h - 2 * q)]
    return inside, outside


def _looks_like_a_card(rgb: np.ndarray, edges: np.ndarray,
                       box: tuple[int, int, int, int]) -> bool:
    """Trois vérifications sur ce qui entoure et remplit la boîte.

    · contraste : le haut ET le bas doivent trancher sur ce qu'il y a autour,
      et au moins un des deux côtés (l'autre peut toucher la carte voisine) ;
    · abords calmes : le tapis n'a pas d'arêtes. Une boîte à cheval sur deux
      cartes a du glyphe des deux côtés — c'est ce qui la disqualifie, et sans
      ce test elle éliminait les deux vraies cartes lors du dédoublonnage ;
    · intérieur : un carton est plus calme qu'une pile de jetons EN MÉDIANE
      (0,463 contre 0,77), mais les deux distributions se recouvrent — 1,6 %
      des vraies cartes sont plus striées que le plus calme des tas de jetons.
      Le plafond est un arbitrage placé sous le recouvrement, pas une
      frontière.
    """
    inside, outside = _side_windows(box)
    contrasts = []
    for ins, outs in zip(inside, outside):
        a, b = _rect(rgb, *ins), _rect(rgb, *outs)
        if a.size == 0 or b.size == 0:
            contrasts.append(0.0)
            continue
        mi = np.median(a.reshape(-1, 3), axis=0)
        mo = np.median(b.reshape(-1, 3), axis=0)
        contrasts.append(float(np.sqrt(((mi - mo) ** 2).sum())))
    top, bottom, left, right = contrasts
    if min(top, bottom) < SIDE_CONTRAST or max(left, right) < SIDE_CONTRAST:
        return False

    quiet = 0
    for outs in outside:
        band = _rect(edges, *outs)
        if band.size and float(band.mean()) < QUIET_MAX:
            quiet += 1
    if quiet < QUIET_SIDES:
        return False

    x, y, w, h = box
    m = max(2, min(w, h) // 8)
    core = _rect(edges, x + m, y + m, w - 2 * m, h - 2 * m)
    return bool(core.size) and float(core.mean()) <= INNER_MAX


def _covered_from_below(rgb: np.ndarray,
                        box: tuple[int, int, int, int]) -> bool:
    """Le bord BAS de la boîte est-il un élément posé dessus, pas la table ?

    Une carte entière a le MÊME fond au-dessus et en dessous — la table — et
    ses deux abords se ressemblent. Une carte coupée a la table au-dessus et
    l'élément qui la recouvre en dessous. C'est la seule marque de la coupure
    qui se mesure sans rien supposer de la position de la carte dans l'image,
    et c'est elle qui écarte les avatars de siège : posés sur le feutre, ils
    ont le même fond des deux côtés.
    """
    _, outside = _side_windows(box)
    haut, bas = _rect(rgb, *outside[0]), _rect(rgb, *outside[1])
    if haut.size == 0 or bas.size == 0:
        return False
    mh = np.median(haut.reshape(-1, 3), axis=0)
    mb = np.median(bas.reshape(-1, 3), axis=0)
    return float(np.sqrt(((mh - mb) ** 2).sum())) >= CUT_BACK_MIN


def detect_card_boxes(image) -> list[CardBox]:
    """Localise les cartes d'une capture. Renvoie les boîtes, non triées.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Capture d'une table déjà jouée.

    Returns
    -------
    list[CardBox]
        Une boîte par carte trouvée, rôle non encore attribué (« ? »).
    """
    rgb = _rgb_array(image)
    H, W = rgb.shape[:2]
    area_img = float(H * W)
    vm, hm = _edge_maps(rgb)
    edges = vm | hm

    hmin = max(14, int(np.sqrt(MIN_AREA_FRAC * area_img / MAX_RATIO)))
    seg = _vertical_segments(vm, hmin)
    if not len(seg):
        return []
    seg = seg[np.argsort(seg[:, 0])]
    cols = seg[:, 0]
    # La plus large des cartes possibles, rapportée à la plus haute arête.
    # C'est CUT_MAX_RATIO et non MAX_RATIO qui borne ici : une carte coupée
    # par le bas est plus large que haute, et sur une image où l'arête la plus
    # longue est justement celle d'une carte coupée, la borne à MAX_RATIO
    # excluait la bonne paire de colonnes avant même de l'examiner (mesuré :
    # 121 px de large pour une fenêtre de 106).
    width_max = int(CUT_MAX_RATIO * (seg[:, 2] - seg[:, 1] + 1).max()) + 2

    kept: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int, bool]] = set()
    for i in range(len(seg)):
        xl, top_l, bot_l = (int(v) for v in seg[i])
        hi = int(np.searchsorted(cols, xl + width_max, side="right"))
        for j in range(i + 1, hi):
            xr, top_r, bot_r = (int(v) for v in seg[j])
            w = xr - xl
            if w < 10:
                continue
            y0, y1 = max(top_l, top_r), min(bot_l, bot_r)
            h = y1 - y0 + 1
            if h < 14:
                continue
            span = max(bot_l - top_l + 1, bot_r - top_r + 1)
            # Deux côtés qui couvrent la MÊME hauteur : carte ENTIÈRE. Sinon
            # on apparie soit un bord de carte avec une arête de glyphe (à
            # jeter), soit une carte COUPÉE par un élément d'interface, dont
            # une seule arête se prolonge le long du bord de cet élément — ce
            # cas-là continue, mais il devra se montrer trop trapu pour une
            # carte entière (voir CUT_MAX_RATIO) pour être retenu.
            entiere = h >= 0.85 * span
            if not entiere:
                # La coupure vient du BAS : le haut de la carte, lui, n'est
                # pas recouvert, donc les deux arêtes partent du même bord et
                # ne divergent que par le bas. Sans cette condition, le chemin
                # appariait un bord de plaque avec une arête de glyphe et
                # inventait une carte (mesuré : boîte 40 × 37 sur le tapis
                # noir du banc, arêtes décalées de 5 px EN HAUT).
                if abs(top_l - top_r) > max(2, round(CUT_TOP_ALIGN * h)):
                    continue
                if h < CUT_MIN_COVER * span:
                    continue
            # Fenêtre volontairement large ici : la hauteur mesurée est celle
            # des segments, amputée du rayon des coins arrondis (~12 %). Le
            # rapport définitif se juge après recalage, pas avant. Une carte
            # coupée est plus trapue, sa fenêtre d'avant recalage l'est aussi.
            if not (0.45 <= w / h <= (1.05 if entiere else 1.40)):
                continue
            box = _snap_to_edges(hm, (xl, y0, w + 1, h))
            if box is None or (*box, entiere) in seen:
                continue
            seen.add((*box, entiere))
            bw, bh = box[2], box[3]
            ratio = bw / bh
            if entiere:
                if not (MIN_RATIO <= ratio <= MAX_RATIO):
                    continue
            # Un appariement bancal ne fabrique JAMAIS une carte entière : il
            # ne sert qu'aux boîtes FRANCHEMENT trop trapues pour en être une
            # (CUT_MIN_RATIO, pas MAX_RATIO — juste au-dessus, une boîte trop
            # large est un cadrage élargi, pas une coupure), et encore faut-il
            # que quelque chose les recouvre par le bas. Sans cette dernière
            # condition, trois avatars de siège du banc entraient (boîtes
            # 33 × 33 et 33 × 34, rapports 0,971 à 1,000).
            elif not (CUT_MIN_RATIO <= ratio <= CUT_MAX_RATIO
                      and _covered_from_below(rgb, box)):
                continue
            if not (MIN_AREA_FRAC * area_img <= bw * bh <= MAX_AREA_FRAC * area_img):
                continue
            if _looks_like_a_card(rgb, edges, box):
                kept.append(box)

    return [CardBox(*b) for b in _dedupe(kept)]


def _overlap_min(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Recouvrement rapporté à la PLUS PETITE des deux boîtes.

    L'IoU ne voit pas qu'une petite boîte est entièrement dans une grande :
    deux cartes ne s'emboîtent jamais, ce rapport-là le dit.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / min(aw * ah, bw * bh)


def _dedupe(boxes: list[tuple[int, int, int, int]]
            ) -> list[tuple[int, int, int, int]]:
    """Une même carte sort de plusieurs appariements : on garde la plus grande."""
    kept: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda c: -(c[2] * c[3])):
        if all(_overlap_min(b, k) < NMS_COVER for k in kept):
            kept.append(b)
    return kept


# Échelle relative au-delà de laquelle une carte n'appartient plus au héros ni
# au board. Un client dessine les cartes des adversaires nettement plus petites
# (0,62 sur le banc) ; sans ce tri, deux dos de sièges opposés se retrouvaient
# dans la même « rangée » que le flop, à ordonnée voisine.
SAME_SCALE = (0.78, 1.30)
ROW_TOL = 0.5            # écart d'ordonnée toléré, en hauteurs de carte
SPREAD_MAX = 1.6         # étalement d'un board, en largeurs de carte cumulées


def _rows(boxes: list[CardBox], tol: float = ROW_TOL) -> list[list[CardBox]]:
    """Regroupe en rangées horizontales des cartes de MÊME échelle."""
    rows: list[list[CardBox]] = []
    for b in sorted(boxes, key=lambda c: c.cy):
        for r in rows:
            if abs(r[0].cy - b.cy) <= tol * b.h and _same_scale(r[0].h, b.h):
                r.append(b)
                break
        else:
            rows.append([b])
    for r in rows:
        r.sort(key=lambda c: c.x)
    return rows


def _same_scale(a: int, b: int) -> bool:
    return SAME_SCALE[0] <= b / a <= SAME_SCALE[1]


def _is_contiguous(row: list[CardBox]) -> bool:
    """Les cartes d'un board se touchent presque ; deux sièges opposés non."""
    span = row[-1].x + row[-1].w - row[0].x
    return span <= SPREAD_MAX * sum(c.w for c in row)


def read_table(image) -> TableRead:
    """Localise les cartes ET leur attribue un rôle (héros / board).

    Trois faits de géométrie, vrais sur toutes les tables en ligne, suffisent :
    les cartes du héros sont les plus BASSES ; le board est une rangée
    CONTIGUË de 3 à 5 cartes proche du centre horizontal ; les cartes des
    adversaires sont dessinées PLUS PETITES que celles du héros.

    C'est ce dernier point qui manquait. Un dos de carte adverse et une carte
    du board peuvent être à la MÊME ordonnée : regroupées naïvement, elles
    formaient une rangée étalée sur toute la largeur, rejetée comme non
    contiguë — et le board entier partait alors dans « autres ». Le
    regroupement n'accepte donc que des cartes de même échelle.

    L'échelle du BOARD se compare à la LARGEUR du héros, pas à sa hauteur :
    une carte du héros recouverte par le bas est plus courte que les autres
    sans être plus petite. Sur les six tapis du banc, avec la main du héros
    recouverte au quart, la comparaison sur la hauteur faisait tomber le board
    entier dans « autres » (0/18) — le rapport des hauteurs vaut alors 1,315
    pour une tolérance de 1,30.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Capture d'une table déjà jouée.

    Returns
    -------
    TableRead
        Cartes du héros, du board, et le reste (dos adverses notamment).
    """
    boxes = detect_card_boxes(image)
    if not boxes:
        return TableRead([], [], [])

    lowest = max(boxes, key=lambda c: c.y + c.h)
    hero_row = sorted((b for b in boxes
                       if abs(b.cy - lowest.cy) <= ROW_TOL * lowest.h
                       and _same_scale(lowest.h, b.h)),
                      key=lambda c: c.x)
    hero_bottom = max(b.y + b.h for b in hero_row)

    cx_img = np.median([b.cx for b in boxes])
    board_row: list[CardBox] = []
    for r in _rows([b for b in boxes if b not in hero_row]):
        if not (3 <= len(r) <= 5) or not _is_contiguous(r):
            continue
        if max(c.y + c.h for c in r) >= hero_bottom:
            continue
        # L'échelle se compare sur la LARGEUR, pas sur la hauteur : la hauteur
        # d'une carte du héros recouverte par le bas est tronquée, la largeur
        # non. Comparer les hauteurs faisait perdre son rôle au board dès que
        # la main du héros était recouverte (mesuré : 0/18 sur les six tapis).
        if not _same_scale(lowest.w, float(np.median([c.w for c in r]))):
            continue
        if not board_row or (len(r), -abs(_centre(r) - cx_img)) > (
                len(board_row), -abs(_centre(board_row) - cx_img)):
            board_row = r

    used = {b.box for b in hero_row} | {b.box for b in board_row}
    return TableRead(
        hero=[CardBox(b.x, b.y, b.w, b.h, "hero") for b in hero_row],
        board=[CardBox(b.x, b.y, b.w, b.h, "board") for b in board_row],
        others=[CardBox(b.x, b.y, b.w, b.h, "?") for b in boxes
                if b.box not in used],
    )


def _centre(row: list[CardBox]) -> float:
    return (row[0].cx + row[-1].cx) / 2
