"""Localisation des montants d'une capture — le viseur de `digit_ocr`.

`digit_ocr` lit une bande de texte mais ne la DÉTECTE pas : « à l'appelant
de viser une zone de texte » (sa limite assumée). Ce module est cet appelant.
Il construit des cadres géométriques relatifs aux cartes détectées
(`TableRead`), y détecte les lignes de texte par leur énergie de gradient,
lit chaque ligne ENTIÈRE une seule fois, et ne retient que celles qui
passent un filtre de forme strict. Le refus de `digit_ocr` reste le seul
juge de la lecture ; ici on ne fait que viser.

Pourquoi des cadres géométriques et pas une détection de pastilles sombres :
mesuré sur les captures réelles, le thème de table varie (feutre clair de la
6-max, fond sombre de la 7-max) et la pastille du pot fusionne avec le fond
sombre en une composante de 620×186 px — la géométrie relative aux cartes,
elle, est la même sur toutes les tables observées.

Zones (disposition PMU, vérifiée sur les 57 captures des sessions des
10-11 août 2026) :

- **pot** : pastille « Pot: X BB » au-dessus du board. L'étiquette du tas de
  jetons sous le board porte LA MÊME valeur : quand les deux se lisent, elles
  se contre-vérifient ; un désaccord est un refus, pas un choix.
- **tapis** : ligne « X BB » de la plaque sous les cartes du héros. Les
  lignes « Temps: 20 » (deux-points) et « 72% 5% 48% » (pour-cents) sont
  écartées par le filtre de forme, le pseudo par ses lettres.
- **mise à payer** : les boutons d'action en bas à droite. « PAYER X » (ou
  CALL/SUIVRE) donne X ; « CHECK » (ou PAROLE) sans montant signifie qu'il
  n'y a RIEN à payer : mise = 0. « RELANCER »/« RAISE » est ignoré — le
  montant d'une relance n'est pas le montant à payer. Pas de bouton lisible :
  refus (héros hors du coup, ou capture tronquée).
- **blinde** : si le pot ou le tapis retenus portent le suffixe « BB »,
  l'affichage est en blindes et la blinde vaut 1 par définition. Sinon,
  refus : la barre de titre (« 1 000/2 000 ») est trop petite et son « / »
  n'est pas dans l'alphabet de `digit_ocr` — mieux vaut une saisie qu'une
  lecture douteuse.

Ce qui est mesuré (les 57 captures réelles, ce module tel quel) : pot lu sur
29 captures — soit toutes celles où le board est détecté sauf trois —, tapis
50, mise 22, affichage BB confirmé 54 ; **zéro désaccord** entre les deux
lectures du pot (12 contre-vérifications), **zéro valeur inventée** (valeurs
distinctes recoupées à l'œil sur les images). Coût : 71 ms par capture en
médiane (158 max), banque de gabarits déjà construite. Le passage du
balayage en bandelettes aux lignes entières a fait gagner 14 tapis et
28 points de blinde — et supprimé la seule source de fausses valeurs
observée (les lignes coupées). Aucun seuil de confiance n'est ajouté
par-dessus le refus de `digit_ocr` : on n'a observé aucune lecture fausse à
écarter, et un seuil sans mesure est un bug en attente.

Les manques ont des causes connues, pas mystérieuses : board ou cartes du
héros non détectés (règle QUIET_SIDES du détecteur, chantier ouvert), bouton
absent de l'écran (héros hors du coup), élément sous un autre habillage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pfs.vision.digit_ocr import Gabarits, LectureMontant, charger_gabarits, lire_ligne
from pfs.vision.table_detector import TableRead

__all__ = ["LectureZone", "lire_montants"]

# ── Cadres, en unités de hauteur de carte (h) ou de taille d'image ──────────
# Marges relevées sur les captures réelles : la pastille du pot déborde peu
# du board (« Pot: 10,60 BB » ≈ 200 px pour un board de 390), la plaque du
# héros est plus large que ses cartes, les boutons occupent le quart bas-droit.
_POT_MARGE_X = 1.2       # h, de part et d'autre du board
_POT_HAUT = 0.70         # h, au-dessus du bord haut du board
_JETONS_MARGE_X = 1.0    # h, cadre de l'étiquette du tas de jetons
_JETONS_BAS = 1.1        # h, sous le bord bas du board
_TAPIS_MARGE_X = 0.9     # h, de part et d'autre des cartes du héros
_TAPIS_BAS = 1.4         # h, sous le bord bas des cartes du héros
_BOUTONS_X0 = 0.42       # fraction de la largeur de l'image
_BOUTONS_Y0 = 0.84       # fraction de la hauteur de l'image

# Découpage en lignes de texte : une ligne est un groupe de rangées à forte
# énergie de gradient horizontal. Pourquoi pas un balayage en bandelettes
# aveugles : mesuré ici même, une bandelette qui COUPE une ligne fait mal
# lire les glyphes tronqués avec une confiance parfois maximale (« Pot: 1500 »
# coupé à mi-hauteur → « Pnt. 1 RDD », valeur 1,0 à confiance 1,0 ;
# « 24,87 BB » coupé → 87). Le garde-fou géométrique de digit_ocr rapporte
# tout à la hauteur des chiffres DE LA LIGNE : une ligne uniformément
# tronquée reste cohérente avec elle-même et passe. Il faut donc viser des
# lignes ENTIÈRES, jamais des tranches.
_GRADIENT_MIN = 24.0     # niveaux de gris — sous quoi une transition est du fond
_PART_ACTIVE = 0.015     # part de colonnes en transition pour qu'une rangée compte
_LIGNE_MIN = 6           # px — plus courte, ce n'est pas une ligne de texte
# 5 px et pas 3 : le haut antialiasé des chiffres passe sous le seuil de
# gradient et sort de la bande détectée — à 3 px de marge, « 42,50 » se
# lisait « 42,so » (le 5 et le 0 décapités deviennent des minuscules).
_MARGE_LIGNE = 5         # px ajoutés au-dessus et en dessous avant lecture

_RE_NUMERIQUE = re.compile(r"^\d[\d.,]*$")
_RE_PAYER = re.compile(r"\b(payer|call|suivre)\b", re.IGNORECASE)
_RE_PAROLE = re.compile(r"\b(check|parole)\b", re.IGNORECASE)
_RE_RELANCE = re.compile(r"relance|raise", re.IGNORECASE)


def _est_montant(ligne: str) -> bool:
    """La ligne porte-t-elle UN montant lisible comme tel ?

    Règles, dans l'ordre de ce qu'elles écartent :

    - pas de « : » ni de « % » — c'est ce qui distingue un tapis des lignes
      « Temps: 16 » et « 72% 5% 48% » de la même plaque ;
    - un mot entièrement numérique, immédiatement suivi du mot « BB ».

    Exiger le suffixe « BB » n'est pas du zèle : le voisinage d'une zone
    (coin d'un bouton, arête de jetons) peut se segmenter en mots-bruit, et
    il suffirait qu'UN de ces mots soit pris pour un nombre pour inventer un
    tapis. Le bruit ne fabrique pas « BB » juste après un nombre. Revers
    assumé : en affichage jetons (« 12 450 », sans BB), le tapis est refusé
    et reste à saisir — l'affichage en blindes est le mode recommandé, et
    c'est celui des captures réelles.
    """
    if ":" in ligne or "%" in ligne:
        return False
    mots = ligne.split()
    return any(_RE_NUMERIQUE.match(m) and i + 1 < len(mots)
               and mots[i + 1].rstrip(".,").upper() == "BB"
               for i, m in enumerate(mots))


@dataclass(frozen=True, slots=True)
class LectureZone:
    """Ce qu'une zone de la capture a donné.

    Attributes
    ----------
    valeur : float or None
        Le montant, ou ``None`` si la zone est refusée.
    confiance : float
        Confiance de `digit_ocr` pour la ligne retenue (0 si refus).
    texte : str
        La ligne de texte retenue, telle que lue.
    refus : str or None
        Motif du refus, en clair — c'est lui qui dit à l'utilisateur ce
        qu'il reste à saisir à la main, et pourquoi.
    """

    valeur: float | None
    confiance: float
    texte: str
    refus: str | None

    def as_dict(self) -> dict:
        return {"valeur": self.valeur, "confiance": round(self.confiance, 3),
                "texte": self.texte, "refus": self.refus}


def _refus(motif: str) -> LectureZone:
    return LectureZone(valeur=None, confiance=0.0, texte="", refus=motif)


def _lignes_texte(crop: Image.Image) -> list[tuple[int, int]]:
    """Bandes verticales [y0, y1) des lignes de texte du crop.

    Une rangée de texte est une rangée où assez de colonnes présentent une
    transition horizontale franche. L'énergie de gradient est insensible au
    fond (plaque sombre, feutre clair, dégradé) là où une distance à la
    couleur médiane du pourtour ne l'est pas — c'est ce qui avait fait
    échouer la première visée du tapis, le pourtour mêlant plaque et feutre.
    """
    a = np.asarray(crop.convert("L"), dtype=np.float64)
    if a.shape[0] < _LIGNE_MIN or a.shape[1] < 12:
        return []
    transitions = np.abs(np.diff(a, axis=1)) > _GRADIENT_MIN
    actif = transitions.mean(axis=1) > _PART_ACTIVE
    lignes: list[tuple[int, int]] = []
    debut: int | None = None
    for i, v in enumerate(actif):
        if v and debut is None:
            debut = i
        elif not v and debut is not None:
            if i - debut >= _LIGNE_MIN:
                lignes.append((debut, i))
            debut = None
    if debut is not None and len(actif) - debut >= _LIGNE_MIN:
        lignes.append((debut, len(actif)))
    return lignes


def _balayer(image: Image.Image, cadre: tuple[float, float, float, float],
             g: Gabarits) -> list[LectureMontant]:
    """Lit chaque ligne de texte ENTIÈRE trouvée dans le cadre.

    Jamais de tranche : une ligne coupée à mi-hauteur fait mal lire les
    glyphes tronqués avec une confiance qui peut être maximale (cf. le
    commentaire des constantes). On découpe donc aux frontières mesurées
    des lignes, avec une petite marge, et on lit chacune une seule fois.
    """
    x0, y0, x1, y1 = cadre
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(image.width, int(x1)), min(image.height, int(y1))
    if x1 - x0 < 12 or y1 - y0 < _LIGNE_MIN:
        return []
    crop = image.crop((x0, y0, x1, y1))
    lectures: list[LectureMontant] = []
    for ly0, ly1 in _lignes_texte(crop):
        if ly0 <= 0 or ly1 >= crop.height:
            # Une ligne qui touche le bord du cadre continue peut-être
            # au-delà : la lire serait lire une ligne coupée — mesuré ici,
            # « 24,87 BB » tronqué par le bas se lit 87 à confiance 0,66.
            continue
        bande = crop.crop((0, max(0, ly0 - _MARGE_LIGNE), crop.width,
                           min(crop.height, ly1 + _MARGE_LIGNE)))
        lectures.append(lire_ligne(bande, g))
    return lectures


def _une_ligne(texte: str) -> str | None:
    """La ligne unique du texte, ou ``None`` s'il y en a zéro ou plusieurs."""
    lignes = [li.strip() for li in texte.splitlines() if li.strip()]
    return lignes[0] if len(lignes) == 1 else None


def _meilleure(lectures: list[LectureMontant], admissible) -> LectureMontant | None:
    """La lecture admise la plus confiante, dédoublonnée par valeur."""
    admises = [le for le in lectures
               if le.valeur is not None
               and (ligne := _une_ligne(le.texte)) is not None
               and admissible(ligne)]
    if not admises:
        return None
    return max(admises, key=lambda le: le.confiance)


def lire_montants(image: Image.Image, table: TableRead,
                  gabarits: Gabarits | None = None) -> dict[str, dict]:
    """Lit pot, mise à payer, tapis du héros et blinde sur une capture.

    Parameters
    ----------
    image : PIL.Image.Image
        La capture entière, telle que collée.
    table : TableRead
        Les cartes localisées par `read_table` — les cadres de visée en
        dérivent ; sans board le pot est refusé, sans cartes du héros le
        tapis l'est.
    gabarits : Gabarits, optional
        Banque de gabarits de `digit_ocr` ; construite (et mémoïsée) au
        premier appel si absente.

    Returns
    -------
    dict of {str : dict}
        Clés ``pot``, ``mise``, ``tapis``, ``blinde`` ; chaque valeur est le
        ``as_dict`` d'une `LectureZone`. Un montant refusé porte son motif —
        jamais de valeur inventée, c'est le contrat de toute la vision.
    """
    g = gabarits if gabarits is not None else charger_gabarits()
    zones: dict[str, LectureZone] = {
        "pot": _refus("board non détecté — pas de cadre pour viser le pot"),
        "mise": _refus("aucun bouton d'action lisible (héros hors du coup, "
                       "ou capture tronquée)"),
        "tapis": _refus("cartes du héros non détectées — pas de cadre pour "
                        "viser son tapis"),
        "blinde": _refus("affichage en BB non confirmé — saisis la blinde"),
    }
    if table.all:
        h = float(np.median([b.h for b in table.all]))

        if table.board:
            bx0 = min(b.x for b in table.board)
            bx1 = max(b.x + b.w for b in table.board)
            by0 = min(b.y for b in table.board)
            by1 = max(b.y + b.h for b in table.board)

            haut = _meilleure(
                _balayer(image, (bx0 - _POT_MARGE_X * h, by0 - _POT_HAUT * h,
                                 bx1 + _POT_MARGE_X * h, by0 - 2), g),
                lambda li: "pot" in li.lower() and ":" in li)
            jetons = _meilleure(
                _balayer(image, (bx0 - _JETONS_MARGE_X * h, by1 + 2,
                                 bx1 + _JETONS_MARGE_X * h, by1 + _JETONS_BAS * h),
                         g),
                _est_montant)
            if haut is None:
                zones["pot"] = _refus("pastille « Pot: … » illisible au-dessus "
                                      "du board")
            elif jetons is not None and abs(haut.valeur - jetons.valeur) > 0.005:
                # Les deux affichages portent le même nombre sur le client ;
                # s'ils divergent, l'un des deux est mal lu — on ne devine pas.
                zones["pot"] = _refus(
                    f"pastille du pot ({haut.valeur:g}) et étiquette des "
                    f"jetons ({jetons.valeur:g}) en désaccord")
            else:
                conf = haut.confiance if jetons is None else max(
                    haut.confiance, jetons.confiance)
                zones["pot"] = LectureZone(haut.valeur, conf,
                                           _une_ligne(haut.texte) or "", None)

        if table.hero:
            hx0 = min(b.x for b in table.hero)
            hx1 = max(b.x + b.w for b in table.hero)
            hy1 = max(b.y + b.h for b in table.hero)
            tapis = _meilleure(
                _balayer(image, (hx0 - _TAPIS_MARGE_X * h, hy1 - 4,
                                 hx1 + _TAPIS_MARGE_X * h, hy1 + _TAPIS_BAS * h),
                         g),
                _est_montant)
            if tapis is None:
                zones["tapis"] = _refus("aucune ligne « montant » lisible sous "
                                        "les cartes du héros")
            else:
                zones["tapis"] = LectureZone(tapis.valeur, tapis.confiance,
                                             _une_ligne(tapis.texte) or "", None)

    # ── boutons d'action : indépendants des cartes détectées ────────────────
    W, H = image.width, image.height
    boutons = _balayer(image, (_BOUTONS_X0 * W, _BOUTONS_Y0 * H, W, H), g)
    payer = [le for le in boutons
             if le.valeur is not None and _RE_PAYER.search(le.texte)
             and not _RE_RELANCE.search(le.texte)]
    parole = [le for le in boutons if _RE_PAROLE.search(le.texte)
              and not _RE_PAYER.search(le.texte)]
    if payer:
        best = max(payer, key=lambda le: le.confiance)
        zones["mise"] = LectureZone(best.valeur, best.confiance,
                                    " ".join(best.texte.split()), None)
    elif parole:
        # « Check »/« Parole » à l'écran : il n'y a rien à payer. Le zéro est
        # une LECTURE, pas un défaut — c'est ce que dit le client.
        best = max(parole, key=lambda le: le.confiance)
        zones["mise"] = LectureZone(0.0, best.confiance,
                                    " ".join(best.texte.split()), None)

    # ── blinde : 1 par définition quand l'affichage est en BB ───────────────
    en_bb = any(z.valeur is not None and z.texte.lower().rstrip().endswith("bb")
                for z in (zones["pot"], zones["tapis"]))
    if en_bb:
        zones["blinde"] = LectureZone(1.0, 1.0, "affichage en BB", None)

    return {nom: z.as_dict() for nom, z in zones.items()}
