"""Analyse post-partie — revue des mains DÉJÀ jouées (jamais en direct).

Ce sous-paquet ne lit que des historiques de mains terminées : il ne
regarde aucune partie en cours et ne produit aucune recommandation
pendant le jeu. C'est l'équivalent d'un tracker (PT4/HM3) adossé au
solveur — l'étude a posteriori de ses propres décisions, seul levier de
progression qui ne touche pas à l'équité des parties en cours.
"""

from pfs.analysis.session_review import (
    AllInSpot,
    HeroProfile,
    SessionReport,
    review_hands,
    review_folder,
)

__all__ = [
    "AllInSpot",
    "HeroProfile",
    "SessionReport",
    "review_hands",
    "review_folder",
]
