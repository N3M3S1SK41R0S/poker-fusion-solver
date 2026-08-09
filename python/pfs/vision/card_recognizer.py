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
# Compromis mesuré (lues / fausses) : marge 12 → 96,8 % / 1,67 % ;
# marge 25 → 95,0 % / 0,45 % ; marge 32 → 94,0 % / 0,06 %.
# On retient 32 : une carte fausse annoncée avec aplomb fausse silencieusement
# tout le conseil qui suit, alors qu'un refus demande juste de recadrer.
MAX_ACCEPT_DISTANCE = 900
MIN_MARGIN = 32


@dataclass(frozen=True, slots=True)
class CardMatch:
    """Résultat d'une reconnaissance de carte."""

    card: str | None       # « Ah », « Ts »… ou None si sous le seuil de confiance
    distance: int          # Hamming au meilleur gabarit
    margin: int            # écart avec le 2e meilleur (grand = sans ambiguïté)
    confidence: float      # dans [0, 1]
    runner_up: str | None  # 2e meilleur candidat (diagnostic)

    @property
    def accepted(self) -> bool:
        return self.card is not None


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
    # La confiance suit la MARGE, pas la distance absolue : c'est elle qui
    # sépare une lecture sûre d'une hésitation (cf. calibration ci-dessus).
    conf = max(0.0, min(1.0, margin / 80.0))
    accepted = best_d <= MAX_ACCEPT_DISTANCE and margin >= MIN_MARGIN
    return CardMatch(
        card=best_c if accepted else None,
        distance=best_d, margin=margin,
        confidence=round(max(0.0, min(1.0, conf)), 3),
        runner_up=second_c,
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
