"""Lecture des montants affichés sur une table — OCR à gabarits.

Même philosophie que le recogniseur de cartes (`card_recognizer`) : une banque
de gabarits, des seuils calés sur mesure, et un REFUS explicite plutôt qu'une
réponse fausse. Un pot mal lu fausse silencieusement tout le conseil qui suit ;
un refus demande simplement de recadrer.

Chaîne
------
1. **Encre** : distance couleur au fond (médiane du pourtour). Travailler sur
   l'écart au fond et non sur la luminosité rend la lecture indépendante de la
   polarité — texte clair sur feutre sombre comme texte sombre sur tapis clair.
2. **Segmentation** : composantes connexes, fusion de celles qui se superposent
   en x (les deux points du « : », le point du « i »), redécoupe des glyphes
   collés, regroupement en lignes puis en mots. Les écarts inter-glyphes sont
   corrigés de la chasse des voisins étroits avant d'être comparés au seuil
   d'espace : sinon un « 1 » laisse derrière lui un blanc d'espace.
3. **Reconnaissance** : corrélation croisée normalisée sur un canevas 10×14,
   plus trois mesures géométriques (largeur, hauteur, position du sommet par
   rapport à la ligne de base), toutes rapportées à la hauteur des chiffres de
   la ligne. Sans ces mesures, « 0 » et « o », « . » et « , » ne se séparent
   pas : le canevas normalisé efface justement la taille et la position.
4. **Assemblage** : le montant est le mot ENTIÈREMENT numérique de la ligne.

Gabarits
--------
Rendus avec PIL à partir des polices d'interface du système (Segoe UI, Arial,
Tahoma, Verdana, en romain et en gras) à douze corps de 9 à 40 px.

Pourquoi pas les glyphes extraits des ressources du client PMU : la planche
qui avait servi d'essai (19 glyphes rouges et en relief, avec les suffixes
K/M/B et les symboles $/€/£) est une police de compteur de jackpot, pas celle
des libellés de table, et elle ne fait pas partie du dépôt — aucun chiffre la
concernant n'est donc rejouable et aucun n'est publié ici. Faute d'une capture
réelle de table pour trancher, la banque est construite sur les polices
système ; `construire_gabarits` permet de la reconstruire sur la fonte exacte
du client, exactement comme `build_templates` pour les cartes.

Ce qui est mesuré
-----------------
Banc synthétique, rejouable par ``pytest tests/test_digit_ocr.py`` : 648 bandes
= 18 montants × 6 formats (« Pot: {} BB », « {} BB », « CALL {} BB »,
« RAISE {} BB », « {} EUR », nombre nu) × 6 corps de 14 à 28 px, tirés sur
6 fonds (feutre vert, bleu nuit, noir, tapis clair, bordeaux, gris sombre),
avec flou gaussien 0 à 0,7 px et bruit gaussien σ 0 à 6. Sauf mention, graine 1
pour les polices de la banque et graine 3 pour les autres.

· polices présentes dans la banque : **99,5 % de montants lus exactement,
  0,00 % de montants faux** ; sur six graines (1, 2, 5, 7, 11, 23), 99,4 à
  99,8 % lus et 0,00 % de faux à chaque fois ;
· `lire_texte` sur la même série (graine 7) : 91,0 % de chaînes rendues à
  l'identique — 590/648 ; les écarts portent sur le libellé, pas sur le
  nombre ;
· polices ABSENTES de la banque (Trebuchet, Calibri, Consolas, Franklin) :
  84,9 % lus, 0,00 % faux ; 84,3 à 85,5 % sur six graines. Le manque vient
  presque entièrement du corps 14, où le garde-fou de hauteur mord (17,6 % lus
  seulement) : sur les corps 16 à 28, ces mêmes polices donnent 99,2 %
  (756 bandes) ;
· coût : 4,0 ms par bande (médiane de trois passes sur les 648 bandes, banque
  déjà construite ; dépend de la machine). Banque de 4 824 gabarits construite
  en 0,61 s dans un processus neuf (0,61 / 0,65 / 0,61 s), 1,08 s si l'on
  compte l'import de numpy, scipy et PIL ; puis mémoïsée.

Le banc reste SYNTHÉTIQUE : pas de compression, pas d'ombre portée, pas de
dégradé de feutre. Il valide la chaîne de lecture, pas la tolérance au bruit
d'un vrai client — d'où le flou et le bruit ajoutés, et le fait que le taux
sur police inconnue soit publié à côté du taux calibré.

Limites assumées (NEMESIS)
--------------------------
· **Deux montants sur la même bande : la valeur rendue, quand il y en a une,
  n'est pas fiable.** Banc de 2 268 bandes portant deux montants voisins
  (18 paires × 3 formats × 6 corps × 7 graines) : avec les polices de la
  banque, 2 rendent une valeur (0,09 %) ; hors banque, 89 (3,92 %). Dans les
  2 268 + 2 268 bandes, aucune valeur INVENTÉE : la valeur rendue est toujours
  l'un des deux montants affichés, l'autre ayant été mal segmenté. C'est la
  correction du défaut le plus grave du module : la règle de recollage
  précédente soudait les deux nombres et fabriquait un montant qui n'était pas
  à l'écran (« 10 1 » → 101,0 à confiance 1,0 ; « 1 2 » → 12,0 sur 42 des
  42 couples police × corps ; « 9,50 4,50 » → 9504,5). Sur les 11 textes qui
  déclenchaient ce défaut × 6 polices × 7 corps, 148/462 rendaient une valeur,
  0/462 aujourd'hui. **Le champ `confiance` ne protège pas de ce reste** : les
  valeurs concernées sortent avec des confiances allant jusqu'à 0,94. À
  l'appelant de cadrer sur UN montant.
· **Hauteur de chiffre minimale : 10 px.** En dessous, la lecture est refusée
  sans être tentée. Mesuré sur 540 bandes de 9 à 13 px : sans ce garde-fou,
  2,96 % de montants FAUX (la virgule est avalée par le chiffre voisin —
  « 4,50 » lu 450, « 6,75 » lu 675) ; avec, 0,00 % de faux pour 91,5 % de
  refus. Le garde-fou ne coûte rien dans le domaine (99,5 % inchangé).
· **Ce module lit une bande de texte, il ne DÉTECTE pas le texte.** Sur
  400 images non textuelles (générateur `_images_non_textuelles` du fichier de
  test, graine 2026), 25 rendent quand même un nombre (6,2 %) : 13/100 pour
  les compositions de formes pleines (un disque ressemble à un « 0 »), 12/100
  pour les dégradés teintés, 0/100 pour le bruit pur et 0/100 pour les barres
  verticales. À l'appelant de viser une zone de texte.
· Le remède à une fonte inconnue est d'AJOUTER la police du client à la
  banque, pas de la remplacer : Calibri passe ainsi de 99,2 % à 100,0 %
  (corps 16-28, 756 bandes), alors qu'une banque réduite à Calibri seule
  retombe à 99,7 % et reste moins robuste que la banque complète.
· **Le « 0,00 % de faux » a une frontière nette : le flou.** Balayé sur
  324 bandes par point (flou 0 à 1,3 px × bruit σ 0 à 18) : jusqu'à 0,7 px de
  flou, 0,00 % de montants faux quel que soit le bruit, polices de la banque
  comme inconnues ; à 1,0 px, 0,62 % (banque) et 3,40 % (inconnues) ; à 1,3 px,
  4,63 % et 8,02 %. Le bruit seul, lui, ne fabrique aucun faux même à σ = 18.
  Une capture plus floue que ~0,7 px sur des chiffres de 16 à 26 px sort du
  domaine où le module tient sa promesse.
· Un séparateur suivi d'exactement trois chiffres est lu comme séparateur de
  milliers (« 1,500 » → 1500) ; un client qui afficherait trois décimales
  serait mal lu — aucun ne le fait pour des blindes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from pfs.vision.phash import _rgb_array

__all__ = [
    "Gabarits",
    "LectureMontant",
    "construire_gabarits",
    "charger_gabarits",
    "lire_ligne",
    "lire_montant",
    "lire_texte",
    "ALPHABET",
    "HAUTEUR_MIN",
]

# Alphabet complet, et pas seulement les chiffres : c'est ce qui permet à
# « BB », « EUR » ou « Pot: » de ne PAS être pris pour des chiffres. Sans les
# lettres, « B » deviendrait « 8 » et « 1,50 BB » se lirait 15088.
ALPHABET = ("0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            ".,:%$")
CHIFFRES = "0123456789.,"

# Polices d'interface livrées avec Windows, en romain et en gras : elles
# couvrent l'essentiel de ce qu'un client de poker affiche. Le nombre compte
# autant que la ressemblance — balayé sur le banc : 1 police → 83,0 % de
# lectures pour 5,25 % de montants FAUX, 2 → 85,8 % / 4,32 %, 4 → 95,1 % /
# 0,00 %, 6 → 99,5 % / 0,00 %. Une banque trop pauvre ne refuse pas : elle
# invente, parce que le second candidat est loin et la marge donc large.
POLICES = ("segoeui.ttf", "segoeuib.ttf", "arial.ttf", "arialbd.ttf",
           "tahoma.ttf", "verdanab.ttf")
# Douze corps. Le crénelage d'un glyphe de 8 px et celui du même glyphe à 40 px
# n'ont rien à voir ; un seul corps ne peut pas les couvrir. Balayé sur le banc
# (lectures justes / montants faux, polices de la banque) : 1 corps → 81,0 % /
# 0,93 % ; 3 corps → 98,0 % / 0,00 % ; 6 corps → 99,2 % ; 12 corps → 99,5 %.
CORPS = (9, 10, 11, 12, 13, 15, 17, 20, 24, 28, 34, 40)

_DOSSIERS_POLICES = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
)

# Canevas de comparaison. Balayé (banque / inconnues / coût) : 10×14 →
# 99,5 % / 84,9 % / 4,4 ms ; 12×18 → 99,5 % / 84,7 % / 4,1 ms ; 16×24 →
# 99,5 % / 83,5 % / 4,6 ms ; 20×28 → 99,5 % / 84,7 % / 5,9 ms. La finesse
# n'apporte rien : on garde le plus petit.
CANEVAS = (10, 14)

# Seuil d'encre, en fraction de l'écart maximal au fond. Balayé (polices de la
# banque / polices inconnues) : 0,35 → 99,8 % / 85,0 % ; 0,40 → 99,5 % /
# 84,9 % ; 0,45 → 99,7 % / 84,9 % ; 0,52 → 98,9 % / 84,4 % ; 0,60 → 93,8 % /
# 83,0 % avec 0,15 % de montants faux (les traits fins disparaissent et les
# glyphes se cassent). Le palier 0,35-0,45 tient en 2 bandes sur 648 : la
# valeur retenue en est le centre, elle n'est pas un optimum tranché.
SEUIL_ENCRE = 0.40

# Poids des mesures géométriques dans la distance. Les mettre à zéro, c'est-à-
# dire ne garder que la forme normalisée, ramène 0,15 % de montants FAUX sur
# les polices de la banque et 8,95 % sur les inconnues (« 0 » lu « o », « , »
# lu « . ») : la taille et la position sont de l'information, pas du bruit.
# Balayés (banque / inconnues) : (0 ; 0 ; 0) → 96,8 % / 59,6 % ;
# (0,3 ; 0,5 ; 0,5) → 99,2 % / 83,3 % ; (0,55 ; 0,9 ; 0,9) → 99,5 % / 84,9 % ;
# (0,8 ; 1,4 ; 1,4) → 99,4 % / 84,9 % ; (1,2 ; 2,0 ; 2,0) → 99,2 % / 83,5 %.
POIDS_LARGEUR, POIDS_HAUTEUR, POIDS_SOMMET = 0.55, 0.90, 0.90

# Au-delà de cette largeur (en hauteur de chiffre), une composante est
# suspectée d'être deux glyphes collés et on tente de la redécouper. Balayé
# (banque / inconnues) : ne jamais redécouper → 98,0 % / 79,5 %, soit 1,5 et
# 5,4 points perdus ; redécouper dès 0,80 → 97,8 % / 82,9 % et fait
# réapparaître 0,31 % de montants faux hors banque (les chiffres larges de
# Verdana Bold se cassent) ; 0,95 et 1,10 → 99,5 % / 84,9 % l'un comme l'autre.
LARGEUR_DECOUPE = 0.95

# Espaces. Distributions mesurées sur 7 597 écarts corrigés (1 262 bandes du
# banc où la segmentation rend exactement un glyphe par caractère, polices de
# la banque et polices inconnues) : vrais espaces p1 = 0,368 · p5 = 0,430 ·
# médiane = 0,545 hauteur de chiffre ; simples interlettres p95 = 0,246 ·
# p99 = 0,396 · médiane = 0,131. Les deux populations se chevauchent encore, et
# le réglage est asymétrique EXPRÈS : 0,18 % d'espaces manqués (3/1 672) pour
# 2,46 % de faux espaces (146/5 925). Un espace manqué entre deux montants
# fabrique un nombre qui n'est pas à l'écran ; un faux espace ne coûte qu'un
# refus, ou rien du tout quand la jonction porte un séparateur.
# Balayés (lus banque / bandes à deux montants rendant une valeur sur 324 /
# textes qui fabriquaient un montant sur 462) :
# ABS 0,26 → 97,4 % / 0 / 2 ; 0,30 → 98,6 % / 0 / 0 ; 0,32 → 99,5 % / 0 / 0 ;
#     0,34 → 99,4 % / 1 / 0 ; 0,38 → 98,5 % / 3 / 5 ;
# REL 0,05 et 0,09 → 99,5 % / 0 / 0 ; 0,11 → 99,5 % / 0 / 0 ;
#     0,14 → 99,5 % / 0 / 2 ; 0,20 → 99,7 % / 0 / 6.
ESPACE_ABS, ESPACE_REL = 0.32, 0.11
# Quand AUCUN écart de la ligne n'est sous le seuil absolu, l'interlettre local
# n'est pas estimable et le critère relatif n'a plus de sens : « 1 2 » n'offre
# qu'un seul écart, dont la médiane vaut... lui-même, donc jamais un espace.
# C'est ce trou qui faisait lire « 1 2 » 12,0 à confiance 1,0 (42/42 polices ×
# corps). Dans ce cas le seuil absolu décide seul, mais relevé. Balayé (mêmes
# colonnes) : 0,38 / 0,42 / 0,45 → 99,5 % / 0 / 0, indiscernables ; 0,50 →
# 99,5 % / 0 / 12 ; 0,60 → 53 ; 0,75 → 85, tous des « 10 1 » resoudés.
ESPACE_SEUL = 0.45
# Correction de chasse. L'écart est mesuré d'encre à encre : après un glyphe
# étroit (« 1 »), il est gonflé des approches que la fonte lui réserve, et un
# « 10 » se fend en « 1 » + « 0 ». On retranche donc une fraction du déficit de
# chasse de chaque voisin PLEINE HAUTEUR (les « , » et « : », courts et étroits,
# sont exclus : les compenser mangeait les vrais espaces de « Pot: 1,50 BB »).
# Largeur d'encre d'un chiffre autre que « 1 » : médiane 0,733 hauteur de
# chiffre (p5 0,648) sur la banque, 0,750 hors banque ; « 1 » : 0,500.
# Balayés (mêmes colonnes) : COMP 0,00 → 97,4 % / 0 / 2 ; 0,10 → 98,8 % / 0 / 1 ;
# 0,15 → 99,5 % / 0 / 0 ; 0,20 → 99,5 % / 1 / 0 ; 0,25 → 99,2 % / 3 / 3 ;
# 0,50 → 92,3 % / 27 / 42. PLEINE 0,60 → 97,5 % ; 0,66 → 98,8 % ; 0,73 →
# 99,5 % ; 0,80 → 99,4 % / 1 / 3 ; 0,90 → 98,9 % / 2 / 12. Trop corriger soude
# les nombres voisins, pas assez fend les « 10 ».
COMP_CHASSE, CHASSE_PLEINE = 0.15, 0.73

# Un glyphe est « numérique » si le meilleur gabarit chiffre n'est pas plus
# mauvais que le meilleur gabarit toutes classes confondues. Mesuré : pour un
# vrai chiffre cet écart vaut exactement 0 ; pour une lettre il vaut 0,37 en
# médiane, et seulement 0,11 au 5e centile. Une tolérance large fait donc
# passer les lettres pour des chiffres et rend toute la ligne ambiguë —
# balayé (banque / inconnues) : 0,00 → 99,5 % / 83,5 % ; 0,02 → 99,5 % /
# 84,9 % ; 0,10 → 98,1 % / 80,7 % ; 0,30 → 48,3 % / 34,7 %, avec le retour de
# montants faux hors banque (0,62 %).
TOL_NUMERIQUE = 0.02

# Marge minimale avec le deuxième chiffre candidat, et distance absolue
# maximale. Balayés conjointement contre les 400 images non textuelles du
# fichier de test (banque / inconnues / images non textuelles acceptées) :
# (0,00 ; 1,00) → 99,5 % / 84,9 % / 37 ; (0,02 ; 0,80) → 99,5 % / 84,9 % / 25 ;
# (0,05 ; 0,70) → 98,1 % / 84,4 % / 20 ; (0,10 ; 0,65) → 96,9 % / 80,4 % / 17 ;
# (0,20 ; 0,60) → 79,3 % / 51,4 % / 15 ; (0,30 ; 0,55) → 39,0 % / 16,4 % / 4.
# Le couple retenu est au coude : serrer davantage coûte bien plus de lectures
# que ça n'écarte de formes qui ne sont pas du texte.
MARGE_MIN = 0.02
DISTANCE_MAX = 0.80

# Hauteur de chiffre en dessous de laquelle on refuse de lire (cf. docstring).
HAUTEUR_MIN = 10.0


@dataclass(frozen=True, slots=True)
class Gabarits:
    """Banque de gabarits de caractères prête à comparer.

    Attributes
    ----------
    caracteres : ndarray of str
        Le caractère de chaque gabarit.
    vecteurs : ndarray, shape (n, l*h)
        Imagette normalisée (moyenne nulle, norme 1) — la corrélation croisée
        se réduit alors à un produit scalaire.
    geometrie : ndarray, shape (n, 3)
        Largeur, hauteur et position du sommet, rapportées à la hauteur des
        chiffres de la police.
    classes : ndarray of int
        Indice de classe (un même caractère rendu par plusieurs polices partage
        sa classe) — sert à calculer la marge avec un caractère DIFFÉRENT.
    numerique : ndarray of bool
        Gabarits dont le caractère est un chiffre ou un séparateur.
    """

    caracteres: npt.NDArray[np.str_]
    vecteurs: npt.NDArray[np.float64]
    geometrie: npt.NDArray[np.float64]
    classes: npt.NDArray[np.int64]
    numerique: npt.NDArray[np.bool_]

    def __len__(self) -> int:
        return int(self.caracteres.size)


@dataclass(frozen=True, slots=True)
class LectureMontant:
    """Ce qu'une bande de texte a donné.

    Attributes
    ----------
    valeur : float or None
        Le montant, ou ``None`` si la lecture est refusée.
    texte : str
        Le texte reconnu, lignes séparées par des retours à la ligne.
    hauteur : float
        Hauteur des chiffres en pixels (0 si rien n'a été segmenté).
    confiance : float
        Dans [0, 1], tirée de la marge du glyphe le moins sûr du nombre.
    refus : str or None
        Motif du refus, en clair, pour le diagnostic.
    """

    valeur: float | None
    texte: str
    hauteur: float
    confiance: float
    refus: str | None

    @property
    def acceptee(self) -> bool:
        return self.valeur is not None


# ------------------------------------------------------------------ gabarits
def _chemin_police(nom: str) -> Path | None:
    """Retrouve un fichier de police par son nom de fichier."""
    p = Path(nom)
    if p.is_file():
        return p
    for d in _DOSSIERS_POLICES:
        if not d.is_dir():
            continue
        direct = d / nom
        if direct.is_file():
            return direct
        for f in d.rglob(nom):
            return f
    return None


def _rendre_glyphe(police, caractere: str
                   ) -> tuple[npt.NDArray[np.float64], int, int, int, int] | None:
    """Rend un caractère seul ; renvoie l'imagette d'encre et sa boîte.

    La boîte est donnée par rapport à la LIGNE DE BASE, pas au coin de
    l'image : c'est ce qui distingue « o » de « 0 » et « . » de « ' ».
    """
    from PIL import Image, ImageDraw

    cote = max(24, police.size * 3)
    im = Image.new("L", (cote, cote), 0)
    base = cote // 2
    ImageDraw.Draw(im).text((cote // 3, base), caractere, fill=255,
                            font=police, anchor="ls")
    a = np.asarray(im, dtype=np.float64)
    ys, xs = np.nonzero(a > 8)
    if ys.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return a[y0:y1, x0:x1] / 255.0, x1 - x0, y1 - y0, y0 - base, y1 - base


def _vecteur(imagette: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Imagette → vecteur centré de norme 1 sur le canevas de comparaison.

    Centrer et normer ramène la corrélation croisée normalisée à un simple
    produit scalaire. Une imagette UNIFORME (« I » et « l », qui remplissent
    tout leur cadre) donne un vecteur nul : ces glyphes n'ont aucune forme
    propre, ce sont les mesures géométriques qui les départagent, et un
    vecteur nul les met à égalité de forme avec tout le reste — ce qui est
    exactement l'information disponible.
    """
    from PIL import Image

    im = Image.fromarray((np.clip(imagette, 0.0, 1.0) * 255).astype(np.uint8))
    v = np.asarray(im.resize(CANEVAS, Image.LANCZOS), dtype=np.float64).ravel()
    v = v / 255.0
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def construire_gabarits(polices: tuple[str, ...] = POLICES,
                        corps: tuple[int, ...] = CORPS) -> Gabarits:
    """Rend l'alphabet avec chaque police à chaque corps demandé.

    Parameters
    ----------
    polices : tuple of str
        Noms de fichiers (« arial.ttf ») ou chemins complets. Les polices
        introuvables sont ignorées ; il en faut au moins une.
    corps : tuple of int
        Corps de rendu en pixels.

    Returns
    -------
    Gabarits
        Banque prête à comparer.

    Raises
    ------
    FileNotFoundError
        Si aucune des polices demandées n'est installée.
    """
    from PIL import ImageFont

    chemins = [c for c in (_chemin_police(n) for n in polices) if c is not None]
    if not chemins:
        raise FileNotFoundError(
            "aucune police système trouvée parmi "
            f"{', '.join(polices)} — passer un chemin explicite à "
            "construire_gabarits(polices=...)")

    cars: list[str] = []
    vecs: list[npt.NDArray[np.float64]] = []
    geos: list[tuple[float, float, float]] = []
    for chemin in chemins:
        for taille in corps:
            police = ImageFont.truetype(str(chemin), taille)
            rendus = {c: _rendre_glyphe(police, c) for c in ALPHABET}
            chiffres = [r for c, r in rendus.items() if c.isdigit() and r]
            if not chiffres:
                continue
            # Référence de la police : hauteur et ligne de base des CHIFFRES.
            # C'est la seule grandeur stable d'une ligne à l'autre — une ligne
            # sans majuscule ni jambage garde ses chiffres.
            ref = float(np.median([r[2] for r in chiffres]))
            base = float(np.median([r[4] for r in chiffres]))
            for c, r in rendus.items():
                if r is None:
                    continue
                imagette, larg, haut, sommet, _ = r
                cars.append(c)
                vecs.append(_vecteur(imagette))
                geos.append((larg / ref, haut / ref, (base - sommet) / ref))

    caracteres = np.array(cars)
    _, classes = np.unique(caracteres, return_inverse=True)
    return Gabarits(
        caracteres=caracteres,
        vecteurs=np.array(vecs, dtype=np.float64),
        geometrie=np.array(geos, dtype=np.float64),
        classes=classes.astype(np.int64),
        numerique=np.isin(caracteres, list(CHIFFRES)),
    )


@lru_cache(maxsize=1)
def charger_gabarits() -> Gabarits:
    """Banque par défaut, construite une fois par processus (0,61 s mesuré)."""
    return construire_gabarits()


# -------------------------------------------------------------- segmentation
def _encre(image) -> npt.NDArray[np.float64] | None:
    """Carte d'encre dans [0, 1] : écart au fond, normalisé par son maximum.

    Le fond est estimé par la médiane du pourtour. Passer par l'écart de
    COULEUR et non par la luminosité permet de lire aussi bien un texte clair
    sur feutre sombre qu'un texte sombre sur tapis clair, sans savoir lequel
    des deux on regarde. ``None`` si la bande est trop uniforme pour contenir
    du texte.
    """
    rgb = _rgb_array(image)
    h, w = rgb.shape[:2]
    b = max(1, int(min(h, w) * 0.08))
    pourtour = np.concatenate([
        rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    fond = np.median(pourtour, axis=0)
    d = np.sqrt(((rgb - fond) ** 2).sum(axis=2))
    haut = float(d.max())
    if haut < 30.0:
        return None
    return d / haut


def _composantes(encre: npt.NDArray[np.float64]) -> list[list[int]]:
    """Boîtes [x0, y0, x1, y1] des taches d'encre, non triées."""
    masque = encre > SEUIL_ENCRE
    if masque.mean() > 0.60 or masque.sum() < 4:
        return []
    etiquettes, n = ndimage.label(masque, structure=np.ones((3, 3)))
    if n == 0:
        return []
    boites = []
    for sy, sx in ndimage.find_objects(etiquettes):
        x0, y0, x1, y1 = int(sx.start), int(sy.start), int(sx.stop), int(sy.stop)
        if (x1 - x0) * (y1 - y0) >= 2:
            boites.append([x0, y0, x1, y1])
    return boites


def _fusionner_colonnes(boites: list[list[int]]) -> list[list[int]]:
    """Recolle les taches qui occupent la même colonne D'UNE MÊME LIGNE.

    Un « : » fait deux taches, un « i » aussi, un « % » jusqu'à trois : elles
    se superposent en x et appartiennent au même caractère. À n'appliquer
    qu'après le regroupement en lignes — sinon deux libellés empilés dans la
    même capture se soudent en une seule tache haute comme le cadre, et le
    montant lu n'a plus rien à voir avec ce qui est affiché.
    """
    sortie: list[list[int]] = []
    for b in sorted(boites, key=lambda k: k[0]):
        for k in sortie:
            chevauche = min(b[2], k[2]) - max(b[0], k[0])
            if chevauche > 0 and chevauche / min(b[2] - b[0], k[2] - k[0]) > 0.6:
                k[0], k[1] = min(k[0], b[0]), min(k[1], b[1])
                k[2], k[3] = max(k[2], b[2]), max(k[3], b[3])
                break
        else:
            sortie.append(list(b))
    return sorted(sortie, key=lambda k: k[0])


def _grouper_lignes(boites: list[list[int]]) -> list[list[list[int]]]:
    """Range les taches en lignes de texte, chacune triée de gauche à droite."""
    lignes: list[list[list[int]]] = []
    for b in sorted(boites, key=lambda k: (k[1] + k[3]) / 2):
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        for ligne in lignes:
            ref_cy = float(np.mean([(k[1] + k[3]) / 2 for k in ligne]))
            ref_h = max(k[3] - k[1] for k in ligne)
            if abs(ref_cy - cy) <= 0.6 * max(h, ref_h):
                ligne.append(b)
                break
        else:
            lignes.append([b])
    return [sorted(ligne, key=lambda k: k[0]) for ligne in lignes]


def _reference(ligne: list[list[int]]) -> tuple[float, float]:
    """(hauteur des chiffres, ordonnée de la ligne de base) d'une ligne.

    Estimées sur les taches les plus hautes : chiffres et capitales partagent
    la même hauteur et la même assise, les médianes résistent à un jambage
    isolé qui descendrait sous la ligne.
    """
    hmax = max(b[3] - b[1] for b in ligne)
    hautes = [b for b in ligne if (b[3] - b[1]) >= 0.75 * hmax]
    return (float(np.median([b[3] - b[1] for b in hautes])),
            float(np.median([b[3] for b in hautes])))


def _redecouper(encre: npt.NDArray[np.float64], boite: list[int],
                ref_h: float) -> list[list[int]]:
    """Sépare une tache trop large en glyphes distincts.

    Deux glyphes voisins ne se touchent que par un pont d'antialiasing : ce
    pont casse à un seuil d'encre plus élevé, bien avant le corps des traits.
    On monte donc le seuil jusqu'à voir apparaître une colonne vide.
    """
    x0, y0, x1, y1 = boite
    if (x1 - x0) <= LARGEUR_DECOUPE * ref_h:
        return [boite]
    sous = encre[y0:y1, x0:x1]
    marge = max(1, int(0.16 * ref_h))
    if sous.shape[1] <= 2 * marge:
        return [boite]

    for seuil in (SEUIL_ENCRE, 0.60, 0.72, 0.84):
        profil = (sous > seuil).sum(axis=0)
        creux = [i for i in range(marge, len(profil) - marge) if profil[i] == 0]
        if not creux:
            continue
        coupes, groupe = [], [creux[0]]
        for i in creux[1:]:
            if i == groupe[-1] + 1:
                groupe.append(i)
            else:
                coupes.append(int(np.mean(groupe)))
                groupe = [i]
        coupes.append(int(np.mean(groupe)))
        bornes = [0, *coupes, x1 - x0]
        largeur_mini = max(2, 0.13 * ref_h)
        if any(b - a < largeur_mini for a, b in zip(bornes[:-1], bornes[1:])):
            continue
        morceaux = []
        for a, b in zip(bornes[:-1], bornes[1:]):
            ys, xs = np.nonzero(sous[:, a:b] > SEUIL_ENCRE)
            if ys.size == 0:
                morceaux = []
                break
            morceaux.append([x0 + a + int(xs.min()), y0 + int(ys.min()),
                             x0 + a + int(xs.max()) + 1, y0 + int(ys.max()) + 1])
        if len(morceaux) >= 2:
            return morceaux
    return [boite]


def _espaces(ligne: list[list[int]], ref_h: float) -> list[bool]:
    """Quels écarts inter-glyphes sont de vrais espaces (distributions en tête).

    La référence interlettre est prise sur les seuls écarts DÉJÀ sous le seuil
    absolu : la médiane de tous les écarts est tirée vers le haut par l'espace
    lui-même dès que la ligne en compte peu, et « 10 2 » (deux écarts, dont un
    espace) ne franchissait alors plus le critère relatif.
    """
    if len(ligne) < 2:
        return []
    deficits = [
        COMP_CHASSE * max(0.0, CHASSE_PLEINE * ref_h - (b[2] - b[0]))
        if (b[3] - b[1]) >= 0.75 * ref_h else 0.0
        for b in ligne]
    ecarts = [max(0.0, ligne[i + 1][0] - ligne[i][2]
                  - deficits[i] - deficits[i + 1])
              for i in range(len(ligne) - 1)]
    interlettres = [e for e in ecarts if e < ESPACE_ABS * ref_h]
    if not interlettres:
        return [e >= ESPACE_SEUL * ref_h for e in ecarts]
    inter = float(np.median(interlettres))
    return [e >= ESPACE_ABS * ref_h and e - inter >= ESPACE_REL * ref_h
            for e in ecarts]


# ------------------------------------------------------------ reconnaissance
@dataclass(frozen=True, slots=True)
class _Glyphe:
    """Un caractère segmenté et ses deux lectures : libre et forcée numérique."""

    caractere: str
    distance: float
    espace: bool
    car_num: str
    distance_num: float
    marge_num: float


def _lire_glyphe(encre: npt.NDArray[np.float64], boite: list[int],
                 ref_h: float, base: float, g: Gabarits
                 ) -> tuple[str, float, str, float, float]:
    """Compare un glyphe à toute la banque, librement puis en forçant un chiffre."""
    x0, y0, x1, y1 = boite
    v = _vecteur(encre[y0:y1, x0:x1])
    mesures = np.array([(x1 - x0) / ref_h, (y1 - y0) / ref_h, (base - y0) / ref_h])
    d = (1.0 - g.vecteurs @ v) \
        + POIDS_LARGEUR * np.abs(g.geometrie[:, 0] - mesures[0]) \
        + POIDS_HAUTEUR * np.abs(g.geometrie[:, 1] - mesures[1]) \
        + POIDS_SOMMET * np.abs(g.geometrie[:, 2] - mesures[2])

    i = int(np.argmin(d))
    dn = d[g.numerique]
    j = int(np.argmin(dn))
    classes_num = g.classes[g.numerique]
    autres = classes_num != classes_num[j]
    # La marge se prend contre le meilleur AUTRE caractère : le même chiffre
    # rendu par une seconde police n'est pas une ambiguïté.
    marge = float(dn[autres].min() - dn[j]) if autres.any() else 1.0
    return (str(g.caracteres[i]), float(d[i]),
            str(g.caracteres[g.numerique][j]), float(dn[j]), marge)


def _analyser(image, g: Gabarits) -> list[tuple[float, list[_Glyphe]]]:
    """Segmente et reconnaît : (hauteur des chiffres, glyphes) par ligne."""
    encre = _encre(image)
    if encre is None:
        return []
    boites = _composantes(encre)
    if not boites:
        return []

    lignes: list[tuple[float, list[_Glyphe]]] = []
    for brute in _grouper_lignes(boites):
        ligne = _fusionner_colonnes(brute)
        ref_h, _ = _reference(ligne)
        decoupee: list[list[int]] = []
        for b in ligne:
            decoupee.extend(_redecouper(encre, b, ref_h))
        ligne = sorted(decoupee, key=lambda k: k[0])
        ref_h, base = _reference(ligne)

        est_espace = _espaces(ligne, ref_h)

        glyphes: list[_Glyphe] = []
        for i, b in enumerate(ligne):
            c, d, cn, dn, marge = _lire_glyphe(encre, b, ref_h, base, g)
            glyphes.append(_Glyphe(
                caractere=c, distance=d, espace=i > 0 and est_espace[i - 1],
                car_num=cn, distance_num=dn, marge_num=marge))
        lignes.append((ref_h, glyphes))
    return lignes


def _rendu(lignes: list[tuple[float, list[_Glyphe]]]) -> str:
    """Texte reconstitué, une chaîne par ligne."""
    return "\n".join(
        "".join((" " if gl.espace else "") + gl.caractere for gl in ligne)
        for _, ligne in lignes)


def _mots_numeriques(lignes: list[tuple[float, list[_Glyphe]]]
                     ) -> tuple[list[list[_Glyphe]], bool]:
    """Mots dont TOUS les glyphes s'expliquent par un chiffre ou un séparateur.

    On ne tronque jamais un mot : « 4B » (un « 48 » dont le 8 a été mal lu) ou
    « 1% » (deux chiffres restés collés) sont rejetés en bloc. Tronquer
    rendrait 4 au lieu de 48, c'est-à-dire un montant faux annoncé avec aplomb.

    Deux mots numériques voisins ne sont recollés que si la JONCTION porte un
    séparateur décimal (« 1, » + « 50 », ou « 1 » + « ,50 ») : un nombre ne se
    termine ni ne commence par une virgule, c'est donc la signature d'un faux
    espace, alors que « 10 » + « 1 » est celle de deux montants affichés côte à
    côte. Un critère d'écart le remplaçait, et soudait « 10 1 » en 101,0.

    Returns
    -------
    mots : list of list of _Glyphe
        Les mots entièrement numériques de la capture.
    douteux : bool
        Vrai si un mot NON numérique commence par un séparateur décimal, ce
        qui ne se produit dans aucun libellé : c'est la signature d'une
        segmentation ratée (« 1 ,fi0 BB » pour « 1,50 BB »). Il ne se
        déclenche jamais dans le domaine documenté (0 sur les 2 884 bandes des
        bancs) ; sur 3 000 bandes volontairement dégradées au-delà (flou
        jusqu'à 1,3 px, bruit σ jusqu'à 16), il se déclenche 2 fois et retire
        1 montant faux sur 25, sans coûter une lecture juste (82,1 % avec
        comme sans).
    """
    sortie: list[list[_Glyphe]] = []
    douteux = False
    for _, ligne in lignes:
        mots: list[list[_Glyphe]] = []
        for gl in ligne:
            if gl.espace or not mots:
                mots.append([])
            mots[-1].append(gl)
        numerique = [all(g.distance_num - g.distance <= TOL_NUMERIQUE for g in m)
                     for m in mots]
        i = 0
        while i + 1 < len(mots):
            if numerique[i] and numerique[i + 1] \
                    and (mots[i][-1].car_num in ".,"
                         or mots[i + 1][0].car_num in ".,"):
                mots[i] = mots[i] + mots[i + 1]
                del mots[i + 1], numerique[i + 1]
            else:
                i += 1
        for m, n in zip(mots, numerique):
            if n and any(g.car_num.isdigit() for g in m):
                sortie.append(m)
            elif not n and len(m) > 1 and m[0].caractere in ".,":
                douteux = True
    return sortie, douteux


def _valeur(chaine: str) -> float | None:
    """« 1,50 » → 1.5, « 1.250,75 » → 1250.75, sinon ``None``.

    Le dernier séparateur est le séparateur décimal, sauf s'il est suivi
    d'exactement trois chiffres : dans ce cas c'est un séparateur de milliers
    (voir la limite documentée en tête de module). Un zéro de tête devant un
    autre chiffre fait refuser : « 000 » n'est pas un montant, c'est une
    lecture ratée.

    Un montant qui porte plusieurs séparateurs n'écrit JAMAIS les milliers et
    la décimale avec le même caractère : « 1.250,75 » et « 1,250.75 » existent,
    « 1,250,75 » non. Exiger ce contraste écarte les chaînes issues de deux
    montants voisins dont l'espace a été manqué (« 6,75 » et « 9,50 » soudés en
    « 6,759,50 », qui se lisait 6759,50), sans rien coûter aux formats réels.
    """
    if not re.fullmatch(r"(0|[1-9][0-9]*)([.,][0-9]+)*", chaine):
        return None
    groupes = re.split(r"[.,]", chaine)
    if len(groupes) == 1:
        return float(chaine)
    separateurs = re.findall(r"[.,]", chaine)
    if len(groupes[-1]) == 3:
        if len(groupes[0]) > 3 or not all(len(g) == 3 for g in groupes[1:]):
            return None
        if len(set(separateurs)) > 1:
            return None
        return float("".join(groupes))
    if len(groupes) > 2:
        if len(groupes[0]) > 3 or not all(len(g) == 3 for g in groupes[1:-1]):
            return None
        if len(set(separateurs[:-1])) > 1 or separateurs[-1] == separateurs[0]:
            return None
    return float("".join(groupes[:-1]) + "." + groupes[-1])


def lire_ligne(image, gabarits: Gabarits | None = None) -> LectureMontant:
    """Lit une bande de texte : le texte, le montant, et le motif de refus.

    Parameters
    ----------
    image : str, PIL.Image or numpy.ndarray
        Bande de texte découpée dans une capture (le libellé et l'unité
        peuvent y figurer : « Pot: 1,50 BB »).
    gabarits : Gabarits, optional
        Banque à utiliser (défaut : celle construite sur les polices système).

    Returns
    -------
    LectureMontant
        ``valeur`` vaut ``None`` dès qu'un doute subsiste ; ``refus`` dit
        lequel.
    """
    g = gabarits if gabarits is not None else charger_gabarits()
    lignes = _analyser(image, g)
    if not lignes:
        return LectureMontant(None, "", 0.0, 0.0, "aucun texte")

    texte = _rendu(lignes)
    hauteur = max(h for h, _ in lignes)
    if hauteur < HAUTEUR_MIN:
        return LectureMontant(None, texte, hauteur, 0.0,
                              f"chiffres de {hauteur:.0f} px "
                              f"(minimum {HAUTEUR_MIN:.0f})")

    mots, douteux = _mots_numeriques(lignes)
    if not mots:
        return LectureMontant(None, texte, hauteur, 0.0, "aucun nombre")
    if len(mots) > 1:
        return LectureMontant(None, texte, hauteur, 0.0,
                              f"{len(mots)} nombres — lequel ?")
    if douteux:
        return LectureMontant(None, texte, hauteur, 0.0, "découpage douteux")

    mot = mots[0]
    marge = min(g.marge_num for g in mot)
    # Échelle de confiance : la marge médiane d'une lecture juste vaut 0,28 sur
    # le banc, donc 0,30 place une lecture ordinaire à 1,0 et ne laisse une
    # confiance basse qu'aux chiffres réellement serrés.
    confiance = round(max(0.0, min(1.0, marge / 0.30)), 3)
    if marge < MARGE_MIN:
        return LectureMontant(None, texte, hauteur, confiance, "chiffre ambigu")
    if max(g.distance_num for g in mot) > DISTANCE_MAX:
        return LectureMontant(None, texte, hauteur, confiance,
                              "glyphe trop éloigné des gabarits")

    valeur = _valeur("".join(g.car_num for g in mot))
    if valeur is None:
        return LectureMontant(None, texte, hauteur, confiance,
                              "nombre non interprétable")
    return LectureMontant(valeur, texte, hauteur, confiance, None)


def lire_montant(image, gabarits: Gabarits | None = None) -> float | None:
    """Le montant affiché dans une bande de texte, ou ``None``.

    Parameters
    ----------
    image : str, PIL.Image or numpy.ndarray
        Bande de texte découpée dans une capture (« Pot: 1,50 BB »,
        « 9,50 BB », « CALL 1 BB »…).
    gabarits : Gabarits, optional
        Banque à utiliser (défaut : celle construite sur les polices système).

    Returns
    -------
    float or None
        ``None`` dès qu'un doute subsiste — un pot faux fausse tout le conseil,
        un refus demande juste de recadrer. `lire_ligne` donne le motif.
    """
    return lire_ligne(image, gabarits).valeur


def lire_texte(image, gabarits: Gabarits | None = None) -> str:
    """Le texte reconnu dans une bande, lignes séparées par « \\n ».

    Contrairement à `lire_montant`, aucun refus : c'est une aide au diagnostic
    et à la saisie, pas une source de vérité. Mesuré sur le banc (graine 7) :
    91,0 % de chaînes rendues à l'identique, les écarts portant sur le libellé
    et non sur le nombre.
    """
    g = gabarits if gabarits is not None else charger_gabarits()
    return _rendu(_analyser(image, g))
