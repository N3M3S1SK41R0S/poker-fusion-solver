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
    "identify_card_autour",
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
#
# PROVENANCE, retrouvée le 10 août 2026 (`scripts/extraire_jeux_pmu.py`, qui
# lit les archives Qt du client sans aucun outil Qt) :
#   · `pmu_solid` est, AU BIT PRÈS, le jeu « 5 » (= « 4c_5 ») de
#     `PokerCommonWidgetsQRC.rcc`, en 65 × 88 — la taille exacte à laquelle la
#     table dessine une carte (`BaseCardDelegate.qml`). C'est l'habillage de la
#     capture PMU PLAY du 10 août ;
#   · `pmu_deck` vient de `/Images/HHCards/hh_*`, en 15 × 20 : la vignette de
#     l'historique des mains, pas le carton de la table.
#
# COUVERTURE ASSUMÉE. Le client propose **douze** jeux complets (« 0 » à « 5 »
# et leurs variantes quatre couleurs « 4c_0 » à « 4c_5 ») ; deux sont ici.
# L'extraction des dix autres est mesurée et sans danger — 676 gabarits, tous
# se relisent juste, le plancher de bruit ne bouge pas (672, seuil 625) — mais
# elle n'a été validée sur AUCUNE capture réelle et coûte 8,2 → 20,5 ms par
# carte. Elle reste donc une commande à passer le jour où un joueur change de
# jeu, pas une couverture annoncée d'avance :
#     python scripts/extraire_jeux_pmu.py --extraire
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
# la population des non-cartes. Tant qu'on la compare à un gabarit ENTIER,
# c'est le bon comportement. Depuis `_truncated_templates`, une découpe dont
# la FORME dit qu'elle est amputée est comparée à des gabarits amputés
# d'autant, et redescend à 212–269 : le refus ne portait pas sur la carte mais
# sur la comparaison.
#
# CONFIRMATION SUR CAPTURE RÉELLE (10 août 2026, table PMU PLAY). Le
# plafond de 900 laissait passer bien pire qu'un « sure » de trop : une carte
# du héros a été PROPOSÉE comme « Ts » à un écart de 714, et l'interface
# offrait de l'accepter d'un clic. Au-dessus du plancher de bruit, le
# classement des gabarits ne porte aucune information : le premier n'est pas
# « le plus ressemblant », il est le moins éloigné d'un ensemble qui ne
# contient pas la réponse. Nommer un candidat dans ces conditions fabrique de
# la confiance à partir de rien.
#
# Le plafond de plausibilité descend donc au même point que le seuil
# d'affirmation : dans le vide mesuré entre les deux populations. Au-delà, le
# recogniseur dit « je ne connais pas cette carte » — ce qui est la vérité.
#
# RECTIFICATION DU DIAGNOSTIC (même journée, `banc_carte_recouverte.py`).
# Ce refus avait été attribué à un habillage ABSENT des gabarits. C'était
# faux, et le banc le montre : l'habillage est `pmu_solid`, déjà livré, et le
# même carton cadré sur TOUTE sa hauteur se lit 5♣ à 490 et 3♥ à 369. Ce qui
# manquait n'était pas un jeu de cartes mais la prise en compte du bandeau
# d'équité du client, qui recouvre le tiers bas des cartes du siège — voir
# `_truncated_templates` juste en dessous. La valeur du seuil, elle, reste
# celle qu'il faut : une carte dont on ne voit pas assez ne doit pas être
# affirmée.
MAX_ACCEPT_DISTANCE = 625   # au-delà : refus, et AUCUN candidat nommé
DISTANCE_SURE = 625         # au-delà : au mieux « propose », jamais « sure »
MARGE_SURE = 32
MARGE_PROPOSE = 8
MIN_MARGIN = MARGE_SURE     # compatibilité : ancien nom du seuil d'acceptation

# --- cartes RECOUVERTES par un élément d'interface -------------------------
# Le client PMU PLAY pose un bandeau d'équité sous chaque siège ; il mange le
# bas des cartes. La carte reste parfaitement lisible à l'œil — rang et
# enseigne sont en haut à gauche — mais le hachage, lui, compare une image
# amputée à un gabarit entier. Mesuré sur la capture du 10 août 2026 (découpes
# exactes 122 × 109, cartons de 122 × 165) : 5♣ lu « Qc » à 705, 3♥ lu « 5h »
# à 731, donc refusés, alors que la même carte cadrée entière se lit à 490 et
# 369. C'est la cause du refus, PAS un habillage manquant.
#
# Le rapport de la découpe suffit à retrouver ce qui manque : un carton entier
# vaut 65/88 = 0,739 ; la découpe visible vaut 122/109 = 1,119 ; il en reste
# donc 0,739 × 109/122 = 0,66. Recoupés à 0,66, les gabarits donnent 5♣ à 212
# (marge 198) et 3♥ à 269 (marge 166) — franchement sous le seuil, et sans la
# moindre constante calée sur cette capture.
CARD_RATIO = 65 / 88        # largeur / hauteur d'un carton entier (gabarits livrés)
# Au-dessous, la découpe passe pour un carton entier et rien n'est tenté. La
# borne haute des cartes entières vaut 0,82 dans `table_detector` ; on laisse
# une marge avant de soupçonner une amputation.
TRUNC_MIN_RATIO = 0.85
# Plage de fractions tentées. En dessous de 0,45 il ne reste plus l'index
# complet (rang + enseigne occupent les 82 % supérieurs de la colonne gauche,
# cf. `corner_phash`) ; au-dessus de 0,92 l'amputation ne change plus rien.
TRUNC_MIN_FRACTION = 0.45
TRUNC_MAX_FRACTION = 0.92


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


@lru_cache(maxsize=64)
def _truncated_templates(frac: float) -> dict[str, Signature]:
    """Signatures des gabarits AMPUTÉS de leur bas, à la fraction `frac`.

    Les gabarits livrés sont des cartons ENTIERS. À l'écran, le client pose son
    bandeau d'équité (« 0 % 0 % 12 % ») sur le bas des cartes du siège : le
    tiers inférieur du carton n'est jamais visible. Comparer une carte amputée
    à un gabarit entier revient à comparer deux images différentes — mesuré sur
    la capture PMU PLAY du 10 août : écart 705 pour le 5♣ et 731 pour le 3♥,
    au-dessus du plancher de bruit, donc refus.

    On ne devine pas la fraction manquante : elle se DÉDUIT du rapport de la
    découpe (voir `_visible_fraction`), et on recoupe les gabarits d'autant.

    Le calcul est mémoïsé par fraction arrondie au centième — les fractions
    utiles tiennent dans une trentaine de valeurs, et une banque coûte
    104 hachages.
    """
    from PIL import Image

    out: dict[str, Signature] = {}
    for theme in _THEMES:
        d = _TEMPLATE_ROOT / theme
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.png")):
            im = Image.open(f)
            w, h = im.size
            keep = max(4, int(round(h * frac)))
            if keep >= h:
                continue
            cut = im.crop((0, 0, w, keep))
            out[f"{f.stem}@{theme}"] = (phash(cut), corner_phash(cut),
                                        colour_signature(cut))
    return out


def _visible_fraction(image) -> float | None:
    """Part de carte encore visible, déduite du RAPPORT de la découpe.

    Un carton entier vaut ``CARD_RATIO`` (largeur / hauteur). Une découpe
    nettement plus trapue ne peut pas être un carton entier : soit ce n'est pas
    une carte, soit son bas est recouvert. Dans le second cas la hauteur
    manquante se lit directement — ``frac = CARD_RATIO × h / w`` — sans aucune
    constante calée sur une capture particulière.

    Renvoie ``None`` quand la découpe a la forme d'une carte entière, ou quand
    la fraction déduite sort de la plage où l'index (rang + enseigne, en haut à
    gauche) reste lisible.
    """
    from PIL import Image

    im = image if isinstance(image, Image.Image) else _as_pil_for_ratio(image)
    w, h = im.size
    if h <= 0 or w / h <= TRUNC_MIN_RATIO:
        return None
    frac = round(CARD_RATIO * h / w, 2)
    if not (TRUNC_MIN_FRACTION <= frac <= TRUNC_MAX_FRACTION):
        return None
    return frac


def _as_pil_for_ratio(image):
    """PIL.Image à partir d'un chemin ou d'un ndarray (sans conversion de mode)."""
    from PIL import Image

    import numpy as np
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8"))
    if isinstance(image, str) or hasattr(image, "__fspath__"):
        return Image.open(image)
    return image


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


def _rank_best(image, templates: dict[str, Signature]
               ) -> tuple[int, str, int, str | None]:
    """Meilleure des deux lectures : gabarits entiers, puis gabarits amputés.

    On n'essaie les gabarits amputés que si la découpe est TROP TRAPUE pour un
    carton entier — c'est la seule situation où l'amputation explique la forme.
    Sur une découpe de forme normale, rien ne change.
    """
    best = _rank(image, templates)
    frac = _visible_fraction(image)
    if frac is not None:
        banque = _truncated_templates(frac)
        if banque:
            alt = _rank(image, banque)
            if alt[0] < best[0]:
                best = alt
    return best


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
    best = _rank_best(image, templates)
    if autocrop:
        # deux polarités : carte CLAIRE sur tapis sombre (deck classique) et
        # carte SOMBRE sur fond clair (thème à fond plein). On garde la
        # meilleure lecture des trois — un cadrage déjà juste n'est jamais
        # dégradé, puisque l'original participe à la comparaison.
        for dark in (False, True):
            try:
                alt = _rank_best(autocrop_card(image, dark_card=dark), templates)
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


#: Cadrages essayés autour d'une boîte imprécise, et budget d'essais.
#:
#: Le détecteur rend une boîte « à peu près juste » ; le hachage exige une
#: boîte exacte. Mesuré sur la première capture réelle : la boîte rendue fait
#: 126 px de large pour une carte de 121, décalée de 7 px — et cet écart-là
#: suffit à faire passer une lecture parfaite au refus (3♥ : refus à 681 avec
#: la boîte du détecteur, « sure » à 209 avec la boîte exacte).
#:
#: Essayer plusieurs cadrages et garder le meilleur, c'est se donner N chances
#: de tomber par hasard sous le seuil : le problème des comparaisons
#: multiples, qui se paie en cartes inventées. Il fallait donc le MESURER
#: avant de l'adopter — `banc_cadrage.py`, grille de 225 cadrages ::
#:
#:     vraies cartes   5♣ : refus/propose 575 → « sure » 362, marge  97
#:                     3♥ : refus         681 → « sure » 222, marge 165
#:     240 non-cartes  écart minimal 644 (seuil 625) ; marge au minimum :
#:                     médiane 12, p95 32, MAXIMUM 46
#:     → 0 non-carte affirmée sur 240.
#:
#: C'est la MARGE qui protège, plus encore que la distance : au minimum de
#: bruit elle plafonne à 46, contre 97 et 165 pour les vraies cartes.
#:
#: Le parcours va du centre vers l'extérieur et s'arrête à la première
#: lecture sûre. Une carte lisible est donc trouvée en quelques essais
#: (~20 ms l'essai), et seul un échec paie le budget entier. L'arrêt
#: anticipé réduit aussi le nombre de comparaisons effectives, donc le
#: risque mesuré ci-dessus est un majorant.
PAS_CADRAGE = 3
BUDGET_CADRAGE = 40


#: Marge de confiance exigée du lecteur à fond plein pour AFFIRMER un rang.
#:
#: Mesuré sur les 52 gabarits `pmu_solid` et sur les deux cartes de la
#: capture réelle : les lectures justes ont une marge de 0,088 à 0,180, quand
#: une lecture hésitante tombe sous 0,02. Le seuil se place entre les deux.
MARGE_RANG_MIN = 0.04


def _lecture_fond_plein(image, boite) -> "CardMatch | None":
    """Lecture directe d'une carte à fond plein, ou ``None`` si inapplicable.

    Renvoie ``None`` — et non un refus — quand l'habillage n'est pas à fond
    plein : l'appelant enchaîne alors sur le hachage, qui couvre les jeux
    classiques. Un refus franc du lecteur (teinte d'aucune famille) rend
    aussi ``None``, pour la même raison : ce n'est pas à lui de conclure sur
    une carte qui n'est pas de son ressort.
    """
    from pfs.vision.lecteur_fond_plein import lire_carte_fond_plein

    x, y, w, h = boite
    lu = lire_carte_fond_plein(image.crop((x, y, x + w, y + h)))
    if lu.carte is None or lu.marge_rang < MARGE_RANG_MIN:
        return None
    # Les échelles diffèrent : l'écart de forme vaut entre 0 et 1, là où le
    # hachage compte en bits. On les ramène au millième pour que les seuils
    # et l'affichage restent comparables d'un chemin à l'autre.
    return CardMatch(
        card=lu.carte,
        distance=int(round(lu.ecart_rang * 1000)),
        margin=int(round(lu.marge_rang * 1000)),
        confidence=round(min(1.0, lu.marge_rang / 0.15), 3),
        runner_up=None,
        best_guess=lu.carte,
        statut="sure",
    )


def _cadrages(pas: int, budget: int) -> list[tuple[int, int, int, int]]:
    """Décalages (dx, dy, dw, dh), du plus proche du centre au plus lointain."""
    d = (-2 * pas, -pas, 0, pas, 2 * pas)
    t = (-2 * pas, 0, 2 * pas)
    grille = [(dx, dy, dw, dh) for dx in d for dy in d for dw in t for dh in t]
    grille.sort(key=lambda o: (abs(o[0]) + abs(o[1]) + abs(o[2]) + abs(o[3])))
    return grille[:budget]


def identify_card_autour(
    image,
    boite: Sequence[int],
    templates: dict[str, int] | None = None,
    pas: int = PAS_CADRAGE,
    budget: int = BUDGET_CADRAGE,
) -> CardMatch:
    """Reconnaît une carte dont la boîte n'est juste qu'à quelques pixels.

    À utiliser quand la boîte vient d'une détection automatique ou d'un
    cadrage à la souris — jamais nécessaire sur un gabarit découpé au pixel.

    Parameters
    ----------
    image : PIL.Image
        La capture entière.
    boite : sequence of int
        ``(x, y, largeur, hauteur)``, approximative.
    pas : int
        Écart en pixels entre deux cadrages essayés.
    budget : int
        Nombre maximal de cadrages. Le parcours partant du centre, un budget
        court reste efficace : c'est le cas d'échec qui le consomme en entier.

    Returns
    -------
    CardMatch
        La première lecture sûre rencontrée ; à défaut, la plus proche
        obtenue. Les seuils de confiance sont ceux d'`identify_card` : cette
        fonction élargit la recherche, elle n'abaisse aucune exigence.
    """
    x, y, w, h = (int(v) for v in boite)
    largeur, hauteur = image.width, image.height

    # D'abord la lecture DIRECTE, quand l'habillage s'y prête : sur un jeu à
    # fond plein, la teinte donne la famille et il ne reste qu'un chiffre
    # blanc à reconnaître. Aucun besoin de gabarit entier, donc ni le bandeau
    # qui masque le bas ni un cadrage approximatif ne la gênent — mesuré
    # 9 cadrages sur 10 justes sur la capture réelle, contre 0 sans elle, et
    # 0 non-carte lue à tort sur 240. Elle coûte 1 ms au lieu de 20.
    direct = _lecture_fond_plein(image, (x, y, w, h))
    if direct is not None:
        return direct

    templates = templates if templates is not None else load_templates()
    meilleur: CardMatch | None = None

    for dx, dy, dw, dh in _cadrages(pas, budget):
        xx, yy, ww, hh = x + dx, y + dy, w + dw, h + dh
        if ww < 20 or hh < 20 or xx < 0 or yy < 0:
            continue
        if xx + ww > largeur or yy + hh > hauteur:
            continue
        m = identify_card(image.crop((xx, yy, xx + ww, yy + hh)), templates)
        if m.statut == "sure":
            return m
        if meilleur is None or m.distance < meilleur.distance:
            meilleur = m

    if meilleur is not None:
        return meilleur
    # Aucun cadrage n'a même pu être essayé (boîte hors image) : on rend la
    # lecture brute plutôt qu'une exception, l'appelant saura la refuser.
    return identify_card(image.crop((x, y, x + w, y + h)), templates)


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
