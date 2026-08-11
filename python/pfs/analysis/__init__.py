"""Analyse post-partie — revue des mains DÉJÀ jouées (jamais en direct).

Ce sous-paquet ne lit que des historiques de mains terminées : il ne
regarde aucune partie en cours et ne produit aucune recommandation
pendant le jeu. C'est l'équivalent d'un tracker (PT4/HM3) adossé au
solveur — l'étude a posteriori de ses propres décisions, seul levier de
progression qui ne touche pas à l'équité des parties en cours.
"""

from pfs.analysis.reperes import (
    REPERES,
    Ecart,
    JeuDeReperes,
    Repere,
    ReperesError,
    calculer_jeu,
    jeu_par_defaut,
    situer,
)
from pfs.analysis.pushfold_review import (
    PushFoldReview,
    ShoveSpot,
    review_pushfold_folder,
    review_pushfold_hands,
)
from pfs.analysis.spot_advisor import Advice, Spot, advise, parse_cards
from pfs.analysis.session_review import (
    AllInSpot,
    HeroProfile,
    SessionReport,
    review_folder,
    review_hands,
)

__all__ = [
    "AllInSpot",
    "HeroProfile",
    "SessionReport",
    "review_hands",
    "review_folder",
    "PushFoldReview",
    "ShoveSpot",
    "review_pushfold_hands",
    "review_pushfold_folder",
    "Spot",
    "Advice",
    "advise",
    "parse_cards",
    "REPERES",
    "Ecart",
    "JeuDeReperes",
    "Repere",
    "ReperesError",
    "calculer_jeu",
    "jeu_par_defaut",
    "situer",
]
