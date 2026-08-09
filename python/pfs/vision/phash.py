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

__all__ = ["phash", "hamming", "autocrop_card", "HASH_BITS"]

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


def _otsu(gray: npt.NDArray[np.float64]) -> float:
    """Seuil d'Otsu (1979) : sépare deux populations en maximisant la
    variance inter-classes. Aucun réglage à la main — c'est l'image qui
    décide où est la frontière clair/sombre."""
    hist, edges = np.histogram(gray, bins=64, range=(0.0, 255.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centres) / np.maximum(w0, 1e-12)
    mt = float((p * centres).sum())
    m1 = (mt - np.cumsum(p * centres)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centres[int(np.argmax(between))])


def autocrop_card(image, pad: int = 1):
    """Recadre sur la carte contenue dans une sélection APPROXIMATIVE.

    Sans ça, la reconnaissance exige un cadrage au pixel près : une
    sélection élargie de 6 px (du feutre autour) ou rognée de 5 px fait
    chuter le taux à zéro — mesuré. Personne ne cadre au pixel près à la
    souris, donc on retrouve nous-mêmes le bord de la carte.

    Principe : une carte est une grande tache CLAIRE sur un fond plus
    sombre. Seuil d'Otsu, plus grande composante connexe claire, boîte
    englobante. Si rien de probant n'est trouvé, l'image est renvoyée telle
    quelle — le recadrage ne doit jamais dégrader un cadrage déjà correct.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Sélection contenant la carte (et un peu de décor autour).
    pad : int
        Marge conservée autour du bord détecté, en pixels.

    Returns
    -------
    PIL.Image
        La carte recadrée (ou l'image d'origine si la détection échoue).
    """
    from PIL import Image
    from scipy import ndimage

    gray = _to_gray_array(image)
    h, w = gray.shape
    if h < 8 or w < 8:
        return _as_pil(image)

    mask = gray > _otsu(gray)
    if mask.mean() > 0.97 or mask.mean() < 0.02:
        return _as_pil(image)          # image uniforme : rien à recadrer

    lab, n = ndimage.label(mask)
    if n == 0:
        return _as_pil(image)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(lab == biggest)
    if ys.size < 0.05 * h * w:         # tache trop petite pour être une carte
        return _as_pil(image)

    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    if (y1 - y0) < 6 or (x1 - x0) < 5:
        return _as_pil(image)
    return _as_pil(image).crop((int(x0), int(y0), int(x1), int(y1)))


def _as_pil(image):
    """Normalise une entrée image en PIL.Image (sans conversion de mode)."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    if isinstance(image, (str,)) or hasattr(image, "__fspath__"):
        return Image.open(image)
    return image
