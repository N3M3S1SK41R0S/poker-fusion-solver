"""Lire une carte à fond plein : la couleur donne la famille, le glyphe le rang.

Pourquoi ce module existe
-------------------------
La reconnaissance par hachage perceptuel compare la carte ENTIÈRE à des
gabarits. Elle marche, mais elle paie cher trois choses qui n'ont rien à voir
avec l'identité de la carte :

* le **bandeau d'équité** du client recouvre le tiers bas — il fallait
  fabriquer des gabarits amputés pour y répondre ;
* un **cadrage à 7 px près** faisait basculer une lecture parfaite en refus —
  il fallait essayer 40 cadrages voisins pour la retrouver ;
* l'**échelle** et l'habillage démultiplient le nombre de gabarits à tenir.

Sur un jeu « 4 couleurs à fond plein », rien de tout cela n'est nécessaire.
La carte est un aplat uni dont la teinte EST la famille, et le rang est un
gros glyphe blanc en haut à gauche. Mesuré sur les 52 gabarits ``pmu_solid`` ::

    trèfle   RGB (31, 127, 44)      dispersion interne 0,0
    cœur     RGB (207, 22, 19)      dispersion interne 0,0
    carreau  RGB (2, 28, 195)       dispersion interne 0,0
    pique    RGB (53, 59, 66)       dispersion interne 0,0

La teinte est rigoureusement identique d'un rang à l'autre, et les familles
les plus proches (trèfle et pique) sont séparées de 74,8 en distance RGB.
La famille se lit donc sans ambiguïté possible.

Reste le rang : treize formes blanches sur fond uni. On les compare sur le
seul MASQUE du glyphe, normalisé à taille fixe — ce qui rend la lecture
indifférente à l'échelle, au cadrage et au bas de la carte, puisque le rang
est en haut à gauche et que le bandeau, lui, mange le bas.

Ce module ne remplace pas `card_recognizer` : il le complète là où il est
applicable, c'est-à-dire sur les habillages à fond plein. Les jeux classiques
(carte blanche, symboles rouges et noirs) restent du ressort du hachage.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "CarteFondPlein",
    "FAMILLES",
    "lire_carte_fond_plein",
    "couleur_dominante",
    "famille_de_couleur",
]

_HERE = Path(__file__).parent
_THEME_SOLID = _HERE / "templates" / "pmu_solid"

RANGS = "23456789TJQKA"

#: Teintes de fond mesurées sur les 52 gabarits `pmu_solid`, dispersion 0,0.
FAMILLES: dict[str, tuple[int, int, int]] = {
    "c": (31, 127, 44),     # trèfle  — vert
    "h": (207, 22, 19),     # cœur    — rouge
    "d": (2, 28, 195),      # carreau — bleu
    "s": (53, 59, 66),      # pique   — ardoise
}

#: Au-delà, la teinte n'est celle d'aucune famille : ce n'est pas une carte
#: de ce jeu. La famille la plus proche d'une autre est à 74,8 ; on autorise
#: un peu moins de la moitié, pour absorber l'anti-crénelage et la
#: compression sans jamais confondre deux familles.
ECART_COULEUR_MAX = 34.0

#: Un glyphe de rang est blanc. Seuil sur le canal minimal : au-dessus, le
#: pixel est de l'encre blanche ; en dessous, c'est le fond coloré.
SEUIL_BLANC = 190

#: Le rang occupe le coin haut-gauche. La zone où on le cherche est définie
#: en fraction de la LARGEUR, jamais de la hauteur.
#:
#: C'est le point qui décide de tout. Défini en fraction de la hauteur, le
#: cadre change de sens dès que le bas de la carte manque : sur un carton
#: entier de 122×165 il couvre 96 px, sur la même carte tronquée à 115 il
#: n'en couvre que 67 — et englobe alors le symbole d'enseigne, qui fusionne
#: avec le chiffre dans la boîte englobante. Mesuré : le 5♣ se lit
#: correctement sur le carton entier (écart 0,071) et devient « Kc » sur la
#: carte tronquée (0,458). Rapporté à la largeur, le cadre garde la même
#: taille réelle quelle que soit la part visible.
LARGEUR_ZONE_RANG = 0.60    # part de la largeur, en x comme en y

#: Taille de normalisation du masque de rang. 24×32 conserve la distinction
#: entre glyphes voisins (6/8, 3/9) sans amplifier le bruit de bord.
_NORM = (24, 32)

#: Part maximale de pixels blancs tolérée dans la découpe. Au-delà, ce n'est
#: pas une carte à fond plein mais une zone claire (bandeau, texte, jeton).
BLANC_MAX = 0.45

#: Dispersion maximale des pixels de fond autour de leur médiane.
#:
#: Une carte à fond plein est un APLAT : le client la rend sans dégradé ni
#: anti-crénelage à l'intérieur. Mesuré sur 315 découpes issues de 57 captures
#: réelles de deux tables ::
#:
#:     199 vraies cartes lues « sure » : dispersion 0,0 — min, médiane,
#:                                       p95 et MAXIMUM tous à zéro
#:     116 dos et décors               : 20,2 à 72,4
#:
#: La séparation est totale, et c'est ce qui rend ce contrôle décisif.
#:
#: Il répond à un faux positif trouvé sur une vraie table : une carte saisie
#: en pleine ANIMATION DE RETOURNEMENT, à moitié recouverte par son dos brun.
#: La médiane des pixels de fond mélangeait alors deux populations — le vert
#: du trèfle et le brun du dos — et tombait près de l'ardoise du pique. Le
#: 6♣ était lu « 6s », affirmé, à une distance de 94. Une carte à moitié
#: retournée ne doit pas être lue du tout ; c'est ce que la dispersion dit,
#: là où la médiane seule ne pouvait pas le voir.
#:
#: Le seuil est posé bas dans le vide mesuré [0 ; 20], avec de la marge pour
#: une éventuelle compression JPEG que ce banc ne contient pas.
DISPERSION_MAX = 12.0


@dataclass(frozen=True, slots=True)
class CarteFondPlein:
    """Lecture d'une carte à fond plein.

    Attributes
    ----------
    carte : str or None
        Notation courte (« 5c »), ou ``None`` si la lecture est refusée.
    famille : str or None
        Enseigne déduite de la teinte du fond.
    rang : str or None
        Rang déduit de la forme du glyphe blanc.
    ecart_couleur : float
        Distance RGB à la teinte de référence de la famille retenue.
    ecart_rang : float
        Écart au meilleur gabarit de rang, entre 0 et 1.
    marge_rang : float
        Écart au deuxième rang candidat : c'est lui qui dit si la forme
        tranche vraiment ou si elle hésite entre deux chiffres.
    motif : str
        Pourquoi la lecture a été refusée, le cas échéant.
    """

    carte: str | None
    famille: str | None
    rang: str | None
    ecart_couleur: float
    ecart_rang: float
    marge_rang: float
    motif: str = ""

    @property
    def sure(self) -> bool:
        return self.carte is not None


def _rgb(image) -> np.ndarray:
    from PIL import Image

    if isinstance(image, np.ndarray):
        a = image
    elif isinstance(image, (str, Path)) or hasattr(image, "__fspath__"):
        a = np.asarray(Image.open(image).convert("RGB"))
    else:
        a = np.asarray(image.convert("RGB"))
    return a.astype(np.float64)


def couleur_dominante(image) -> tuple[np.ndarray, float, float]:
    """Teinte du fond, part de pixels blancs, et dispersion autour de la teinte.

    La médiane des pixels NON blancs est prise plutôt que la moyenne : un
    bord de carte, un liseré ou un morceau de feutre entrant dans la découpe
    déplacent une moyenne, pas une médiane.

    Mais la médiane seule ne dit pas si la découpe est HOMOGÈNE. Une carte à
    moitié recouverte — retournement en cours, jeton posé dessus — mélange
    deux populations de couleurs, et sa médiane tombe entre les deux, sur une
    teinte que la carte n'a nulle part. D'où la troisième valeur rendue.

    Returns
    -------
    tuple
        ``(teinte, part_blanche, dispersion)`` — la dispersion est l'écart
        médian des pixels de fond à leur médiane, en distance RGB.
    """
    a = _rgb(image)
    pix = a.reshape(-1, 3)
    blanc = pix.min(axis=1) >= SEUIL_BLANC
    fond = pix[~blanc]
    if not len(fond):
        return np.array([255.0, 255.0, 255.0]), 1.0, 0.0
    med = np.median(fond, axis=0)
    dispersion = float(np.median(np.linalg.norm(fond - med, axis=1)))
    return med, float(blanc.mean()), dispersion


def famille_de_couleur(couleur: np.ndarray) -> tuple[str | None, float]:
    """Famille la plus proche d'une teinte, et l'écart correspondant."""
    meilleure, ecart = None, float("inf")
    for s, ref in FAMILLES.items():
        d = float(np.linalg.norm(couleur - np.array(ref, dtype=float)))
        if d < ecart:
            meilleure, ecart = s, d
    return (meilleure if ecart <= ECART_COULEUR_MAX else None), ecart


def _masque_rang(image) -> np.ndarray | None:
    """Masque normalisé du glyphe de rang, isolé dans le coin haut-gauche.

    Le glyphe est cherché par sa boîte englobante plutôt qu'à une position
    fixe : c'est ce qui rend la lecture indifférente au cadrage. Le bas de la
    carte n'entre jamais dans la zone, donc le bandeau d'équité ne gêne pas.
    """
    from PIL import Image

    a = _rgb(image)
    h, w = a.shape[:2]
    cote = max(4, int(round(w * LARGEUR_ZONE_RANG)))
    coin = a[:min(h, cote), :min(w, cote)]
    if coin.size == 0:
        return None
    blanc = coin.min(axis=2) >= SEUIL_BLANC
    if blanc.sum() < 8:
        return None

    # Le coin contient DEUX glyphes blancs : le rang, puis le symbole
    # d'enseigne en dessous. Prendre la boîte englobante de tout le blanc
    # les réunit, et la comparaison finit dominée par le symbole — qui est
    # identique pour les treize rangs d'une famille. Mesuré : le 3♥ était
    # alors lu « Jh ». On ne garde donc que la PREMIÈRE bande horizontale
    # d'encre, séparée de la suivante par la ligne vide qui court entre les
    # deux glyphes.
    lignes = blanc.any(axis=1)
    if not lignes.any():
        return None
    debut = int(np.argmax(lignes))
    fin = debut
    while fin + 1 < len(lignes) and lignes[fin + 1]:
        fin += 1
    bande = blanc[debut:fin + 1]
    if bande.sum() < 8:
        return None
    xs = np.where(bande.any(axis=0))[0]
    g = bande[:, xs.min():xs.max() + 1]
    # Un glyphe de rang n'est ni un trait ni une tache : bornes de forme
    # larges, seulement là pour écarter l'aberrant.
    hh, ww = g.shape
    if hh < 4 or ww < 3 or ww / hh > 2.5:
        return None
    img = Image.fromarray((g * 255).astype(np.uint8)).resize(
        _NORM, Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


@lru_cache(maxsize=1)
def _gabarits_rang() -> dict[str, np.ndarray]:
    """Un masque de rang par rang, moyenné sur les quatre familles.

    Moyenner les quatre exemplaires efface le bruit propre à une famille et
    ne garde que la forme du chiffre — ce qui est précisément l'information
    cherchée.
    """
    out: dict[str, np.ndarray] = {}
    for r in RANGS:
        masques = []
        for s in FAMILLES:
            p = _THEME_SOLID / f"{r}{s}.png"
            if not p.exists():
                continue
            m = _masque_rang(p)
            if m is not None:
                masques.append(m)
        if masques:
            out[r] = np.mean(masques, axis=0)
    return out


def lire_carte_fond_plein(image) -> CarteFondPlein:
    """Lit une carte d'un jeu à fond plein.

    Parameters
    ----------
    image : str | Path | PIL.Image | numpy.ndarray
        Découpe contenant la carte. Le cadrage peut être approximatif : le
        glyphe de rang est cherché dans le coin, pas supposé à une position.
        Le bas de la carte peut manquer (bandeau d'équité) sans conséquence.

    Returns
    -------
    CarteFondPlein
        La carte lue, ou un refus motivé. Aucune carte n'est rendue sans que
        la teinte ET la forme du rang aient tranché.
    """
    couleur, part_blanche, dispersion = couleur_dominante(image)
    if part_blanche > BLANC_MAX:
        return CarteFondPlein(None, None, None, 0.0, 1.0, 0.0,
                              "trop clair pour une carte à fond plein")

    # Le fond doit être un APLAT. Une carte à moitié retournée ou masquée
    # mélange deux couleurs, et sa médiane désigne une famille qu'elle n'a
    # pas — c'est ainsi qu'un 6♣ en cours de retournement a été lu « 6s »,
    # affirmé. Mieux vaut ne rien lire qu'une carte à moitié visible.
    if dispersion > DISPERSION_MAX:
        return CarteFondPlein(
            None, None, None, 0.0, 1.0, 0.0,
            f"fond non uniforme (dispersion {dispersion:.0f}) — carte "
            "partiellement recouverte ou en cours de retournement")

    famille, ecart_c = famille_de_couleur(couleur)
    if famille is None:
        return CarteFondPlein(
            None, None, None, ecart_c, 1.0, 0.0,
            f"teinte ({couleur[0]:.0f},{couleur[1]:.0f},{couleur[2]:.0f}) "
            "d'aucune famille connue")

    masque = _masque_rang(image)
    if masque is None:
        return CarteFondPlein(None, famille, None, ecart_c, 1.0, 0.0,
                              "aucun glyphe de rang isolable")

    gabarits = _gabarits_rang()
    scores = sorted(
        ((float(np.abs(masque - g).mean()), r) for r, g in gabarits.items()))
    if len(scores) < 2:
        return CarteFondPlein(None, famille, None, ecart_c, 1.0, 0.0,
                              "gabarits de rang indisponibles")
    (e1, r1), (e2, _) = scores[0], scores[1]
    return CarteFondPlein(f"{r1}{famille}", famille, r1, ecart_c,
                          e1, e2 - e1)
