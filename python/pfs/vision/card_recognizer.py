"""Reconnaissance de cartes par gabarits pHash.

Chaque carte du deck de référence a une signature pHash. Reconnaître une
image de carte = trouver le gabarit de plus petite distance de Hamming. La
confiance combine la distance absolue (proche de 0 = sûr) et la MARGE avec
le second meilleur (une grande marge = pas d'ambiguïté).

Le deck par défaut (`templates/pmu_deck/`, 52 cartes du client PMU) et ses
signatures pré-calculées (`templates/pmu_phash.json`) sont livrés. Pour une
autre room ou un autre thème : `build_templates(dossier)` sur un dossier de
`<carte>.png` régénère les signatures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from pfs.vision.phash import (
    HASH_BITS,
    autocrop_card,
    colour_distance,
    colour_signature,
    corner_phash,
    hamming,
    phash,
)

# Signature : hash du carton entier + hash du coin (le rang) + couleur moyenne.
Signature = tuple[int, int, tuple[float, float, float]]

# Poids de la couleur : un écart d'enseigne du thème « fond plein »
# (rouge vs vert, ~150 en RGB) pèse ~50 — l'ordre de grandeur de la
# séparation de forme, sans l'écraser.
COLOUR_WEIGHT = 0.34
# Poids du coin : c'est là que se lit le RANG. Sans lui, deux cartes de même
# enseigne mais de rang différent ne se séparaient que de 0 à 2 bits.
CORNER_WEIGHT = 1.6

__all__ = [
    "CardMatch",
    "build_templates",
    "load_templates",
    "identify_card",
    "recognize_cards",
    "DEFAULT_TEMPLATES",
]

_HERE = Path(__file__).parent
_TEMPLATE_ROOT = _HERE / "templates"
DEFAULT_TEMPLATES = _TEMPLATE_ROOT / "pmu_phash.json"
_DECK_DIR = _TEMPLATE_ROOT / "pmu_deck"

# Thèmes livrés. Le client PMU change complètement l'habillage des cartes
# selon le réglage : « pmu_deck » = deck classique (fond blanc, symboles
# rouges/noirs) ; « pmu_solid » = fond plein saturé (rouge=cœur, bleu=carreau,
# vert=trèfle, noir=pique) avec glyphes blancs. On ne DEVINE pas le thème :
# toutes les signatures sont dans la même banque, et c'est la plus proche qui
# gagne. Ajouter un thème = déposer un dossier de `<carte>.png` ici.
_THEMES = ("pmu_deck", "pmu_solid")

# Seuils calés sur 1 560 essais (7 tapis × 2 habillages × cadrages serré et
# large). Deux constats dictent la règle :
#   · les distances des bonnes et des mauvaises réponses SE CHEVAUCHENT —
#     la distance seule ne peut donc pas trancher ;
#   · c'est la MARGE avec le second candidat qui sépare proprement.
# Compromis mesuré sur images NETTES (lues / fausses) : marge 12 → 96,8 % /
# 1,67 % ; marge 25 → 95,0 % / 0,45 % ; marge 32 → 94,0 % / 0,06 %.
#
# MAIS ces marges s'effondrent sur une image réelle. Reproduction des
# conditions d'une vraie capture (habillage à fond plein, cartes 78×104 sur
# tapis clair) : image ré-échelonnée à 0,66 → **0/8 acceptées**, alors que le
# bon carton était en tête dans 5 cas sur 8 (« 9s » lu 9s avec 11 de marge,
# « Jh » lu Jh avec 8). Un couperet unique jetait donc des lectures JUSTES.
#
# D'où trois niveaux plutôt qu'un seuil : au-dessus de MARGE_SURE la lecture
# s'impose ; entre MARGE_PROPOSE et MARGE_SURE elle est PROPOSÉE et attend un
# clic — l'œil humain tranche en une seconde et ne se trompe pas sur une carte
# qu'il voit ; en dessous, on refuse.
#
# CORRECTION (10 août 2026). Cette construction affirmait tout de même des
# cartes à tort, et la première lecture en direct l'a montré : une découpe de
# DÉCOR a été lue « 4h », statut « sure », à un écart de 703 — parce que la
# marge, elle, valait 44. La marge seule ne suffit pas : elle dit qu'un
# gabarit devance ses concurrents, pas qu'il ressemble à l'image.
#
# Mesure du plancher de bruit sur 240 découpes qui ne sont PAS des cartes
# (bruit uniforme, aplats de feutre, dos de cartes, jetons chiffrés) ::
#
#     famille   écart min   p5    médiane      sure   propose
#     bruit          658   684        718         1        21
#     feutre         700   713        747         0        25
#     dos            686   695        730         1        20
#     jeton          670   699        717         0        28
#
# Le plancher global est 658, très en dessous de MAX_ACCEPT_DISTANCE : 39 %
# des non-cartes atteignaient « propose » et 0,8 % « sure ».
#
# Une lecture affirmée exige donc désormais AUSSI d'être PROCHE. Les deux
# populations se séparent franchement — 312 cartes réellement cadrées sur
# feutre (2 habillages × 3 feutres × 52 cartes) contre les 240 non-cartes ::
#
#                              n     min   p5   médiane   p95   max
#     cartes bien identifiées  312   251   268     406    565   599
#     non-cartes               240   658   695     724    767   790
#
# Il existe donc un vide réel entre 599 et 658, où AUCUN des 552 échantillons
# ne tombe. Le seuil s'y place au milieu : il garde 100 % des vraies cartes
# et rejette 100 % des fausses. Un premier essai à 520 avait été tenté et
# rejeté — il coupait au milieu de la population des vraies cartes et faisait
# tomber la lecture de 40/52 à 12/52 sur feutre vert.
#
# Repère utile : une carte masquée au tiers par le HUD monte à 688, donc dans
# la population des non-cartes — et c'est le bon comportement, une carte
# masquée ne doit pas être affirmée.
MAX_ACCEPT_DISTANCE = 900   # au-delà : refus pur et simple
DISTANCE_SURE = 625         # au-delà : au mieux « propose », jamais « sure »
MARGE_SURE = 32
MARGE_PROPOSE = 8
MIN_MARGIN = MARGE_SURE     # compatibilité : ancien nom du seuil d'acceptation


@dataclass(frozen=True, slots=True)
class CardMatch:
    """Résultat d'une reconnaissance de carte."""

    card: str | None       # « Ah », « Ts »… ou None si sous le seuil de confiance
    distance: int          # Hamming au meilleur gabarit
    margin: int            # écart avec le 2e meilleur (grand = sans ambiguïté)
    confidence: float      # dans [0, 1]
    runner_up: str | None  # 2e meilleur candidat (diagnostic)
    best_guess: str | None = None
    """Meilleur candidat, MÊME quand la lecture est refusée.

    Un refus sans candidat est une impasse : impossible de distinguer un
    cadrage à reprendre d'un habillage absent de la banque. En exposant ce que
    le recogniseur a failli lire, et de combien il s'en est fallu, l'appelant
    peut diagnostiquer — et l'utilisateur trancher lui-même, son œil restant
    la référence.
    """

    statut: str = "refus"
    """« sure » (lecture qui s'impose) · « propose » (à confirmer d'un clic) ·
    « refus » (rien de plausible). Voir MARGE_SURE / MARGE_PROPOSE."""

    @property
    def accepted(self) -> bool:
        return self.card is not None

    @property
    def a_confirmer(self) -> bool:
        """Lecture plausible mais pas certaine : l'appelant doit demander."""
        return self.statut == "propose"


def build_templates(deck_dir: str | Path = _DECK_DIR,
                    save_to: str | Path | None = None) -> dict[str, int]:
    """Calcule les signatures pHash d'un dossier de gabarits `<carte>.png`.

    Parameters
    ----------
    deck_dir : chemin
        Dossier contenant `Ah.png`, `Ks.png`, … (52 cartes attendues).
    save_to : chemin, optionnel
        Si fourni, écrit le JSON `{carte: hash}` à cet endroit.

    Returns
    -------
    dict[str, int]
        Signature par carte.
    """
    deck_dir = Path(deck_dir)
    templates: dict[str, Signature] = {}
    for f in sorted(deck_dir.glob("*.png")):
        templates[f.stem] = (phash(f), corner_phash(f), colour_signature(f))
    if not templates:
        raise FileNotFoundError(f"aucun gabarit `<carte>.png` dans {deck_dir}")
    if save_to is not None:
        Path(save_to).write_text(_dumps(templates), encoding="utf-8")
    return templates


def _dumps(templates: dict[str, "Signature"]) -> str:
    """Sérialise : hash 256 bits en hexadécimal (portable) + couleur moyenne."""
    return json.dumps(
        {k: {"h": format(h, "x"), "k": format(kh, "x"),
             "c": [round(x, 2) for x in c]}
         for k, (h, kh, c) in templates.items()},
        indent=0, sort_keys=True)


def build_all_themes(
    save_to: str | Path | None = DEFAULT_TEMPLATES,
) -> dict[str, Signature]:
    """Signatures de TOUS les thèmes présents, clés « carte@thème ».

    Un même `Ah` existe dans plusieurs habillages : les clés sont donc
    suffixées par le thème pour qu'aucune signature n'en écrase une autre.
    """
    out: dict[str, Signature] = {}
    for theme in _THEMES:
        d = _TEMPLATE_ROOT / theme
        if not d.is_dir():
            continue
        for card, sig in build_templates(d).items():
            out[f"{card}@{theme}"] = sig
    if not out:
        raise FileNotFoundError(f"aucun thème de cartes sous {_TEMPLATE_ROOT}")
    if save_to is not None:
        Path(save_to).write_text(_dumps(out), encoding="utf-8")
    return out


@lru_cache(maxsize=4)
def load_templates(path: str | Path = DEFAULT_TEMPLATES) -> dict[str, Signature]:
    """Charge les signatures pré-calculées (mémoïsé). Repli : les reconstruit.

    Si le JSON pré-calculé est absent mais que les dossiers de gabarits
    existent, on recalcule à la volée — le paquet reste fonctionnel même
    sans le JSON.
    """
    p = Path(path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: (int(v["h"], 16), int(v["k"], 16), tuple(v["c"]))
                for k, v in raw.items()}
    return build_all_themes(save_to=None)


def _card_of(key: str) -> str:
    """« Ah@pmu_solid » → « Ah » (le thème n'intéresse que le diagnostic)."""
    return key.split("@", 1)[0]


def _rank(image, templates: dict[str, Signature]) -> tuple[int, str, int, str | None]:
    """(distance, carte, marge, dauphin) pour une image donnée.

    Distance = forme (Hamming sur le pHash) + couleur pondérée. La marge se
    calcule contre la meilleure AUTRE carte, pas contre le même carton dans un
    autre habillage : deux thèmes qui proposent tous deux « Ah » ne créent
    aucune ambiguïté et ne doivent pas écraser la marge.
    """
    h = phash(image)
    kh = corner_phash(image)
    c = colour_signature(image)
    ranked = sorted(
        (hamming(h, sig) + CORNER_WEIGHT * hamming(kh, ksig)
         + COLOUR_WEIGHT * colour_distance(c, col), key)
        for key, (sig, ksig, col) in templates.items()
    )
    best_d, best_key = ranked[0]
    best_card = _card_of(best_key)
    second_d, second_card = float(HASH_BITS), None
    for d, key in ranked[1:]:
        if _card_of(key) != best_card:
            second_d, second_card = d, _card_of(key)
            break
    return round(best_d), best_card, round(second_d - best_d), second_card


def identify_card(image, templates: dict[str, int] | None = None,
                  autocrop: bool = True) -> CardMatch:
    """Reconnaît UNE carte à partir de son image.

    Le cadrage n'a pas besoin d'être précis : par défaut, on cherche aussi
    le bord de la carte dans la sélection (``autocrop``) et on garde la
    meilleure des deux lectures. Sans ce recadrage, une sélection élargie
    ou rognée de quelques pixels fait échouer la reconnaissance — mesuré.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Image d'une carte, éventuellement avec du décor autour.
    templates : dict, optionnel
        Signatures à utiliser (défaut : le deck PMU livré).
    autocrop : bool
        Tenter le recadrage automatique (mettre ``False`` pour un crop
        déjà exact, par exemple un gabarit).

    Returns
    -------
    CardMatch
        Carte reconnue (ou ``None`` si sous le seuil), distance, marge,
        confiance.
    """
    templates = templates if templates is not None else load_templates()
    best = _rank(image, templates)
    if autocrop:
        # deux polarités : carte CLAIRE sur tapis sombre (deck classique) et
        # carte SOMBRE sur fond clair (thème à fond plein). On garde la
        # meilleure lecture des trois — un cadrage déjà juste n'est jamais
        # dégradé, puisque l'original participe à la comparaison.
        for dark in (False, True):
            try:
                alt = _rank(autocrop_card(image, dark_card=dark), templates)
                if alt[0] < best[0]:
                    best = alt
            except Exception:
                pass

    best_d, best_c, margin, second_c = best
    # La confiance suit la MARGE : c'est elle qui sépare une lecture nette
    # d'une hésitation entre deux gabarits (cf. calibration ci-dessus).
    conf = max(0.0, min(1.0, margin / 80.0))
    plausible = best_d <= MAX_ACCEPT_DISTANCE
    # Mais AFFIRMER exige les deux : devancer les concurrents (marge) ET
    # ressembler au gabarit (distance). Sans la seconde condition, une
    # découpe de décor ou un dos de carte peut sortir « sure » sur une marge
    # chanceuse — c'est arrivé, et c'est le pire mode d'échec de cet outil.
    if plausible and margin >= MARGE_SURE and best_d <= DISTANCE_SURE:
        statut = "sure"
    elif plausible and margin >= MARGE_PROPOSE:
        statut = "propose"
    else:
        statut = "refus"
    return CardMatch(
        card=best_c if statut == "sure" else None,
        distance=best_d, margin=margin,
        confidence=round(max(0.0, min(1.0, conf)), 3),
        runner_up=second_c,
        best_guess=best_c if statut != "refus" else None,
        statut=statut,
    )


def _crop(image, roi: Sequence[int]):
    """Découpe une région (x, y, w, h) d'une image PIL/chemin/ndarray."""
    from PIL import Image

    import numpy as np
    if isinstance(image, np.ndarray):
        im = Image.fromarray(image.astype("uint8"))
    elif isinstance(image, (str,)) or hasattr(image, "__fspath__"):
        im = Image.open(image)
    else:
        im = image
    x, y, w, h = roi
    return im.crop((x, y, x + w, y + h))


def recognize_cards(
    image,
    rois: Iterable[Sequence[int]],
    templates: dict[str, int] | None = None,
) -> list[CardMatch]:
    """Reconnaît plusieurs cartes, une par région d'intérêt (x, y, w, h).

    Les ROI dépendent de la room et de la résolution : à caler sur une vraie
    capture. Une fois calées, cette fonction lit tout le tableau d'un coup
    (cartes du héros + board).
    """
    templates = templates if templates is not None else load_templates()
    return [identify_card(_crop(image, roi), templates) for roi in rois]
