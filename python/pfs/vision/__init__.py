"""Perception — reconnaissance de cartes à partir d'images.

Ce paquet lit des IMAGES (captures d'écran, crops) et en extrait des cartes,
pour alimenter le conseiller de spot sans ressaisie manuelle. Il ne capture
rien lui-même (ça, c'est la sonde Rust `rust/crates/pfs-capture`, Phase 1) :
il reconnaît ce qu'on lui donne.

Méthode : hash perceptuel (pHash DCT) sur un jeu de gabarits — le deck du
client PMU, extrait de ses ressources et étiqueté (52 cartes vérifiées). Le
pHash normalise l'échelle : un gabarit 15×20 reconnaît une carte affichée
bien plus grande à l'écran.

Limite assumée (NEMESIS) : le recogniseur est CALIBRÉ sur le deck par défaut
de PMU. Un autre thème de cartes, ou une autre room, demande de reconstruire
les gabarits (`build_templates` sur un dossier `<carte>.png`) — une commande,
pas une réécriture. Le mapping des régions d'intérêt (où sont les cartes sur
la table) dépend de la room et de la résolution : fourni en best-effort, à
caler sur une vraie capture.
"""

from pfs.vision.card_recognizer import (
    CardMatch,
    build_templates,
    identify_card,
    load_templates,
    recognize_cards,
)
from pfs.vision.phash import hamming, phash

__all__ = [
    "CardMatch",
    "build_templates",
    "identify_card",
    "load_templates",
    "recognize_cards",
    "phash",
    "hamming",
]
