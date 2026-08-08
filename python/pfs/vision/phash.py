"""Hash perceptuel (pHash DCT) — signature d'image robuste à l'échelle.

Le pHash de Zauner (2010) : niveaux de gris → 32×32 → DCT-II 2D → bloc
basse fréquence → seuil à la médiane → hash binaire. Deux images
visuellement proches (même carte à des tailles différentes, léger bruit,
recompression) ont un petit nombre de bits différents ; la distance de
Hamming les sépare des images distinctes.

Pourquoi la DCT et pas un simple redimensionnement : les basses fréquences
capturent la STRUCTURE (glyphe du rang, forme du pip, disposition) et
ignorent les détails fins et le niveau global de luminosité — exactement ce
qu'il faut pour reconnaître une carte quel que soit son rendu à l'écran.

Taille du bloc : 16×16 (256 bits). Un bloc 8×8 ne distinguait pas assez le
pique du trèfle de même rang (2 bits de séparation seulement — glyphe de
rang identique, forme du pip perdue à basse fréquence) ; 16×16 porte cette
séparation à 30 bits, sans quoi une capture réelle légèrement différente
confondrait ces cartes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.fft import dct

__all__ = ["phash", "hamming", "HASH_BITS"]

_SIZE = 32          # image normalisée avant DCT
_LOW = 16           # bloc basse fréquence conservé (16×16 = 256 bits)
HASH_BITS = _LOW * _LOW


def _to_gray_array(image) -> npt.NDArray[np.float64]:
    """Convertit une entrée image en tableau (H, W) niveaux de gris [0, 255].

    Accepte un chemin, un objet PIL.Image, ou un ndarray (H,W) / (H,W,3/4).
    Un canal alpha est composé sur blanc (les cartes ont un fond
    transparent dans les ressources du client).
    """
    from PIL import Image

    if isinstance(image, (str,)) or hasattr(image, "__fspath__"):
        im = Image.open(image)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            return arr.astype(np.float64)
        im = Image.fromarray(arr.astype(np.uint8))
    else:
        im = image  # suppose un PIL.Image

    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("L"), dtype=np.float64)


def phash(image) -> int:
    """Signature perceptuelle 64 bits d'une image.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Chemin, image PIL, ou tableau (niveaux de gris ou couleur).

    Returns
    -------
    int
        Entier 64 bits (déterministe pour une image donnée).

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((20, 15), dtype=np.uint8)
    >>> a[5:15, 5:10] = 255
    >>> isinstance(phash(a), int)
    True
    >>> phash(a) == phash(a)          # déterministe
    True
    """
    from PIL import Image

    gray = _to_gray_array(image)
    im = Image.fromarray(gray.astype(np.uint8)).resize((_SIZE, _SIZE), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)

    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:_LOW, :_LOW]
    # médiane hors terme continu [0,0] (la luminosité moyenne ne porte pas
    # d'information de forme et fausserait le seuil)
    flat = low.flatten()
    med = float(np.median(flat[1:]))
    bits = (low > med).flatten()

    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    """Nombre de bits différents entre deux hashes (0 = identiques)."""
    return int(bin(a ^ b).count("1"))
