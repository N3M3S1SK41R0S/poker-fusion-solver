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

from pfs.vision.phash import HASH_BITS, hamming, phash

__all__ = [
    "CardMatch",
    "build_templates",
    "load_templates",
    "identify_card",
    "recognize_cards",
    "DEFAULT_TEMPLATES",
]

_HERE = Path(__file__).parent
DEFAULT_TEMPLATES = _HERE / "templates" / "pmu_phash.json"
_DECK_DIR = _HERE / "templates" / "pmu_deck"

# distance de Hamming au-delà de laquelle on refuse de trancher (sur 256 bits).
# Calée sur les mesures : les vraies reconnaissances (même deck, échelle et
# bruit variés) restent < 40, la séparation entre cartes distinctes est ≥ 30.
MAX_ACCEPT_DISTANCE = 55
MIN_MARGIN = 8


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
    templates: dict[str, int] = {}
    for f in sorted(deck_dir.glob("*.png")):
        templates[f.stem] = phash(f)
    if not templates:
        raise FileNotFoundError(f"aucun gabarit `<carte>.png` dans {deck_dir}")
    if save_to is not None:
        # signatures 256 bits stockées en hexadécimal (portable, sans risque
        # de dépassement d'entier côté parseurs JSON tiers)
        Path(save_to).write_text(
            json.dumps({k: format(v, "x") for k, v in templates.items()},
                       indent=0, sort_keys=True),
            encoding="utf-8")
    return templates


@lru_cache(maxsize=4)
def load_templates(path: str | Path = DEFAULT_TEMPLATES) -> dict[str, int]:
    """Charge les signatures pré-calculées (mémoïsé). Repli : les reconstruit.

    Si le JSON pré-calculé est absent mais que le dossier de gabarits existe,
    on recalcule à la volée — le paquet reste fonctionnel même sans le JSON.
    """
    p = Path(path)
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        # signatures en hexadécimal (cf. build_templates)
        return {k: int(v, 16) for k, v in raw.items()}
    return build_templates()


def identify_card(image, templates: dict[str, int] | None = None) -> CardMatch:
    """Reconnaît UNE carte à partir de son image (crop).

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Image d'une seule carte.
    templates : dict, optionnel
        Signatures à utiliser (défaut : le deck PMU livré).

    Returns
    -------
    CardMatch
        Carte reconnue (ou ``None`` si sous le seuil), distance, marge,
        confiance.
    """
    templates = templates if templates is not None else load_templates()
    h = phash(image)
    ranked = sorted(((hamming(h, sig), card) for card, sig in templates.items()))
    best_d, best_c = ranked[0]
    second_d, second_c = ranked[1] if len(ranked) > 1 else (HASH_BITS, None)
    margin = second_d - best_d
    # confiance : proximité absolue temperée par la marge
    conf = max(0.0, 1.0 - best_d / HASH_BITS) * (1.0 if margin >= MIN_MARGIN
                                                 else margin / MIN_MARGIN)
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
