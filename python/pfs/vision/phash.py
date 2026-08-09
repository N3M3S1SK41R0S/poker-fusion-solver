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

__all__ = ["phash", "hamming", "autocrop_card", "colour_signature",
           "colour_distance", "HASH_BITS"]

_SIZE = 64          # image normalisée avant DCT
_LOW = 24           # bloc basse fréquence conservé (24×24 = 576 bits)
_INSET = 0.08       # frange écartée (coins arrondis transparents)
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
    # On écarte une frange : les coins des cartes sont ARRONDIS et
    # transparents. Sur un gabarit ils sont composés sur blanc, à l'écran ils
    # laissent voir le tapis — leur couleur dépend donc du fond et pollue la
    # signature. En hachant le cœur de la carte, la comparaison redevient
    # valable quel que soit le tapis.
    h, w = gray.shape
    iy, ix = int(h * _INSET), int(w * _INSET)
    if h - 2 * iy >= 8 and w - 2 * ix >= 8:
        gray = gray[iy:h - iy, ix:w - ix]
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


def corner_phash(image) -> int:
    """Signature de la COLONNE D'INDEX (gauche) : le rang ET l'enseigne.

    Deux raisons de la hacher à part. Sur un habillage à fond plein, le rang
    est un petit glyphe noyé dans un large aplat : ramené à la taille du
    carton entier il ne pèse presque rien, et « 7♦ » devenait indistinguable
    de « J♦ » (marge mesurée : 0 bit). Symétriquement, la zone doit descendre
    assez bas pour inclure l'ENSEIGNE : limitée au rang seul, elle confondait
    une enseigne sur deux (26/52 — cœur avec carreau, pique avec trèfle).

    Tous les habillages placent rang au-dessus, enseigne en dessous, dans une
    colonne à gauche : une même fraction convient donc aux deux.
    """
    im = _as_pil(image)
    w, h = im.size
    box = im.crop((0, 0, max(4, int(w * 0.46)), max(4, int(h * 0.82))))
    return phash(box)


def colour_signature(image) -> tuple[float, float, float]:
    """Couleur moyenne du cœur de la carte (R, G, B).

    Indispensable : certains habillages codent l'ENSEIGNE par la couleur du
    carton (rouge = cœur, bleu = carreau, vert = trèfle, noir = pique). En
    niveaux de gris, rouge et vert sont presque identiques — le hash seul
    confondait donc une couleur sur deux (26/52 mesuré). La couleur moyenne
    lève cette ambiguïté sans rien coûter.
    """
    rgb = _rgb_array(image)
    h, w = rgb.shape[:2]
    iy, ix = int(h * _INSET), int(w * _INSET)
    core = rgb[iy:h - iy, ix:w - ix] if (h - 2 * iy >= 4 and w - 2 * ix >= 4) else rgb
    m = core.reshape(-1, 3).mean(axis=0)
    return (float(m[0]), float(m[1]), float(m[2]))


def colour_distance(a: tuple[float, float, float],
                    b: tuple[float, float, float]) -> float:
    """Distance euclidienne RGB entre deux signatures de couleur."""
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


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


def _rgb_array(image) -> npt.NDArray[np.float64]:
    """Tableau (H, W, 3) RGB, alpha composé sur blanc."""
    from PIL import Image

    im = _as_pil(image)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"), dtype=np.float64)


def autocrop_card(image, pad: int = 1, dark_card: bool = False):
    """Recadre sur la carte contenue dans une sélection APPROXIMATIVE.

    Sans ça, la reconnaissance exige un cadrage au pixel près : une
    sélection élargie de 6 px (du feutre autour) ou rognée de 5 px fait
    chuter le taux à zéro — mesuré. Personne ne cadre au pixel près à la
    souris, donc on retrouve nous-mêmes le bord de la carte.

    Principe : la carte est la grande région dont la COULEUR s'écarte du
    tapis. On estime le tapis sur le pourtour de la sélection (vrai dès que
    le cadrage est généreux, ce que l'interface demande), puis on seuille la
    distance couleur par Otsu et on garde la plus grande composante connexe.

    Travailler sur la couleur et non sur la luminosité est indispensable :
    un carton rouge sur feutre vert a quasiment la MÊME luminosité — mesuré,
    l'ancienne version tombait à 0/52 dans ce cas. La distance au fond, elle,
    ne suppose ni carte claire ni carte sombre, et fonctionne donc pour tout
    habillage de cartes sur tout tapis.

    Parameters
    ----------
    image : str | PIL.Image | numpy.ndarray
        Sélection contenant la carte (et un peu de décor autour).
    pad : int
        Marge conservée autour du bord détecté, en pixels.
    dark_card : bool
        Conservé pour compatibilité ; sans effet (la détection par écart de
        couleur n'a plus besoin de connaître la polarité).

    Returns
    -------
    PIL.Image
        La carte recadrée (ou l'image d'origine si la détection échoue).
    """
    from scipy import ndimage

    rgb = _rgb_array(image)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return _as_pil(image)

    # couleur du tapis : médiane du pourtour (robuste à un coin de carte qui
    # dépasserait dans le cadre)
    b = max(1, int(min(h, w) * 0.06))
    ring = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                           rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    felt = np.median(ring, axis=0)

    dist = np.sqrt(((rgb - felt) ** 2).sum(axis=2))
    if float(dist.max()) < 24.0:       # sélection uniforme : rien à isoler
        return _as_pil(image)

    # Seuil calé sur le BRUIT PROPRE du tapis, pas sur une séparation en deux
    # classes : un seuil d'Otsu opposait les glyphes (très loin du tapis) au
    # RESTE, et rangeait le corps d'une carte noire du côté du feutre vert —
    # le recadrage ne gardait alors qu'un glyphe. En prenant « tout ce qui
    # dépasse nettement la dispersion du tapis », le carton entier est retenu,
    # quelle que soit sa couleur.
    ring_d = np.sqrt(((ring - felt) ** 2).sum(axis=1))
    thr = max(float(np.percentile(ring_d, 98)) * 1.8, 18.0)
    mask = dist > thr
    if mask.mean() > 0.97 or mask.mean() < 0.02:
        return _as_pil(image)

    # les trous internes (glyphes de la couleur du tapis) ne doivent pas
    # scinder la carte en morceaux
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    mask = ndimage.binary_fill_holes(mask)

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
