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

« La famille se lit donc sans ambiguïté possible », disait la suite de cette
phrase. C'était trop dire, et deux fois. Ces « dispersion interne 0,0 » sont
mesurés sur une chaîne PNG SANS PERTE, et une dispersion nulle sur toute une
population n'est pas une marge mais une saturation — voir ``DISPERSION_MAX``.
Et 74,8 d'écart entre deux familles n'interdit pas de passer de l'une à
l'autre : un voile coloré uniforme y suffisait — voir ``MARGE_COULEUR_MIN``.
``banc_robustesse_fond.py`` chiffre les deux.

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
    "part_etrangere",
    "MARGE_COULEUR_MIN",
    "ECART_RANG_MAX",
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
#: compression.
#:
#: « Sans jamais confondre deux familles » disait ce commentaire. C'était
#: faux, et ``banc_robustesse_fond.py`` le montre : 34 vaut 45 % de 74,8,
#: donc un déplacement de 34 DANS LA BONNE DIRECTION laisse la teinte plus
#: près de la voisine que de la sienne. Le balayage de 216 teintes de voile ×
#: 8 opacités trouve le cas : un voile vert (0,255,0) à 20 % amène le pique
#: (53,59,66) sur (42,98,52), à 32,0 du trèfle contre 42,9 du pique. Le 2♠
#: était lu « 2c », affirmé — teinte acceptée, fond parfaitement uniforme,
#: dispersion 0,00. C'est ``MARGE_COULEUR_MIN`` qui ferme ce cas.
ECART_COULEUR_MAX = 34.0

#: Écart minimal entre la DEUXIÈME famille et la première.
#:
#: Un plus proche voisin sur quatre références ne doit pas se juger à une
#: distance absolue mais à une marge : la bonne question n'est pas « la
#: teinte est-elle assez près de cette famille » mais « est-elle nettement
#: plus près d'elle que de toutes les autres ». C'est la même logique que
#: ``marge_rang`` pour le glyphe, et elle manquait pour la couleur.
#:
#: C'est aussi la forme utile du contrôle demandé en revue — « comparer la
#: valeur du fond à la médiane du gabarit de la famille retenue ». Comparée
#: dans l'absolu, cette valeur laissait passer le voile vert ci-dessus ;
#: comparée aux QUATRE médianes de gabarit, elle le refuse.
#:
#: Mesuré par ``banc_robustesse_fond.py``, section 4 :
#:
#:     vraies cartes (52 gabarits + 2 découpes versionnées, PNG puis
#:     JPEG q95→q30, n=378)                      : marge minimale 66,8
#:     idem sur 20 découpes réelles (n=140)      : marge minimale 69,9
#:     lectures d'une autre famille, seuil neutralisé : 10,8 à 29,4
#:     38 non-cartes distinctes des captures réelles  : médiane 15,6,
#:                                sous le seuil pour 95 % d'entre elles
#:
#: Vide mesuré [29,4 ; 66,8]. Le seuil est posé dedans : 0 vraie carte
#: perdue, 23 lectures fausses sur 23 rattrapées (3 sur 3 sur les découpes
#: réelles). Le coût se paie sur les découpes ARTIFICIELLEMENT voilées, où
#: 14 à 20 % des lectures affirmées deviennent des refus — ce qui est le
#: comportement voulu : une carte dont on ne sait plus dire l'enseigne ne
#: doit pas être nommée.
MARGE_COULEUR_MIN = 45.0

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
#: anti-crénelage à l'intérieur. Ce contrôle répond à un faux positif trouvé
#: sur une vraie table : une carte saisie en pleine ANIMATION DE
#: RETOURNEMENT, à moitié recouverte par son dos brun. La médiane des pixels
#: de fond mélangeait alors deux populations — le vert du trèfle et le brun
#: du dos — et tombait près de l'ardoise du pique. Le 6♣ était lu « 6s »,
#: affirmé. C'est ce que la dispersion voit, là où la médiane seule ne le
#: pouvait pas.
#:
#: CE QUE CE SEUIL VAUT, ET CE QU'IL NE VAUT PAS
#: ---------------------------------------------
#: La mesure d'origine — « 199 vraies cartes à dispersion 0,0, min, médiane,
#: p95 ET maximum tous à zéro » — n'était pas une marge : c'était une
#: SATURATION. La dispersion est la MÉDIANE des écarts à la médiane ; sur une
#: découpe majoritairement plate elle vaut exactement 0 quoi qu'il arrive au
#: reste. Deux conséquences, toutes deux mesurées par
#: ``banc_robustesse_fond.py`` :
#:
#: 1. **Sous compression, la marge existe et se referme avec la TAILLE.**
#:    Ré-encodage JPEG 4:2:0 (le sous-échantillonnage par défaut des
#:    encodeurs, donc le pire cas pour une carte dont l'identité est une
#:    couleur) — dispersion MAXIMALE :
#:
#:    ===============================  ====  ====  =====  =====  =====  =====
#:    jeu                               q95   q85    q70    q50    q40    q30
#:    ===============================  ====  ====  =====  =====  =====  =====
#:    20 découpes réelles, 126–138 px  3,00  5,00   8,60  10,05  10,82  13,60
#:    52 gabarits `pmu_solid`, 65 px   4,12  8,51  12,12  15,39  16,16  17,35
#:    ===============================  ====  ====  =====  =====  =====  =====
#:
#:    Le seuil tient jusqu'à q40 sur une découpe de 130 px (1 refus sur 20
#:    seulement à q30), mais dès q70 il refuse 1 gabarit sur 52, et 14 sur 52
#:    à q50. Autrement dit : la marge dépend de la taille de la carte, pas
#:    de la qualité seule. Aucune compression, à aucune qualité testée
#:    jusqu'à q30, n'a en revanche produit une lecture FAUSSE — le seuil
#:    échoue en refusant, jamais en affirmant à tort.
#:
#: 2. **Le contrôle ne voit PAS une occultation minoritaire.** Recouvrir le
#:    haut de la carte sur 8 à 20 % de sa hauteur laisse la dispersion à
#:    0,00 — exactement zéro — parce que la couleur de la carte reste
#:    majoritaire. Le glyphe de rang, lui, est mutilé, et la lecture devient
#:    FAUSSE : un 6♥ lu « 5h », un 3♥ lu « Jh », affirmés. Mesure A/B du
#:    banc, seuil de forme neutralisé : 615 lectures fausses sur les
#:    gabarits, 166 sur les découpes réelles — le contrôle de fond en voyait
#:    **0**. La formule de refus parle de « carte partiellement recouverte » :
#:    elle ne couvre en réalité que le voisinage de 50 %. C'est
#:    ``ECART_RANG_MAX``, plus bas, qui rattrape ce cas.
#:
#: Le seuil reste à 12 : au-dessus du pire vrai positif jusqu'à q40 sur les
#: découpes réelles, et loin sous les intrus réels (42,2 et 48,5 sur les deux
#: découpes de cartes retournées versionnées).
DISPERSION_MAX = 12.0

#: Écart de forme maximal toléré pour AFFIRMER un rang.
#:
#: Sans ce seuil, la lecture rendait toujours le plus proche des treize
#: gabarits, si mauvais soit le rapprochement : treize voisins, aucun refus.
#: C'est le défaut que le contrôle de dispersion croyait couvrir et ne
#: couvrait pas (voir ci-dessus).
#:
#: Mesuré par ``banc_robustesse_fond.py`` :
#:
#: * lectures JUSTES — 20 découpes réelles distinctes, en PNG puis JPEG
#:   q95/85/70/50 : écart maximal 0,137 (0,145 à q30) ; 52 gabarits
#:   `pmu_solid` : 0,033 en PNG, et jusqu'à 0,1745 sous JPEG (l'As de trèfle
#:   à q50 ; 0,161 à q70) ; les deux découpes versionnées : 0,116 et 0,090 ;
#: * lectures FAUSSES fabriquées par occultation partielle du haut
#:   (5 couleurs d'occultant × 6 hauteurs) : écart minimal 0,1324 sur les
#:   gabarits (0,1386 sur les découpes réelles), médiane 0,296.
#:
#: Les deux nuages SE CHEVAUCHENT sur [0,1324 ; 0,1745] : il n'y a pas de
#: vide cette fois, et le dire est le point. Une première rédaction annonçait
#: [0,139 ; 0,145] ; c'était le chevauchement des seules découpes réelles,
#: pris pour celui de l'ensemble — le chevauchement mesuré est 3,4 fois plus
#: large. Le seuil est posé à 0,18, au-dessus du pire vrai positif mesuré
#: toutes qualités confondues, mais de 0,0055 seulement : la marge est
#: MINCE, et un habillage plus compressé la mangerait.
#:
#: A/B du banc, seuil neutralisé puis actif :
#:
#:     52 gabarits + 2 découpes versionnées : 615 lectures fausses → 20
#:     20 découpes réelles de session       : 166 lectures fausses → 10
#:
#: Ce qui subsiste est un seul motif : un 6 dont on rogne 8 % du haut
#: ressemble vraiment à un 5, et son écart (0,132 à 0,147) tombe dans la
#: plage des vraies lectures compressées. Aucun seuil sur cette statistique
#: ne les sépare ; le banc l'imprime comme limite assumée plutôt que de
#: choisir un seuil qui coûterait des vraies cartes.
ECART_RANG_MAX = 0.18


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

    @property
    def du_jeu(self) -> bool:
        """La teinte du fond est celle d'une famille du jeu à fond plein.

        Sert à distinguer deux refus que rien d'autre ne sépare :

        * « je ne sais pas lire ça » — teinte d'aucune famille : ce peut être
          un jeu classique, et le hachage doit prendre la suite ;
        * « ceci est une carte de CE jeu, mais elle n'est pas lisible » —
          teinte reconnue, fond hétérogène : la carte est partiellement
          recouverte, et personne ne peut honnêtement la nommer. Enchaîner
          sur le hachage ne peut alors produire qu'une affirmation fausse.

        Mesuré le 11 août 2026 sur `banc_verite_captures.py` : sans cette
        distinction, un 6♣ saisi en pleine animation de retournement était
        affirmé « Kc » par le hachage, à un écart de 616 pour une marge de 33.
        """
        return self.famille is not None


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

    La dispersion rendue est une MÉDIANE : elle ne bouge pas tant que la
    contamination reste minoritaire. C'est une propriété, pas un défaut de
    mise en œuvre — mais elle borne ce que ce contrôle peut voir. Voir la
    note de ``DISPERSION_MAX``.

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


def famille_de_couleur(couleur: np.ndarray) -> tuple[str | None, float, float]:
    """Famille d'une teinte, écart à cette famille, et marge sur la suivante.

    C'est ici que se joue la « comparaison à la médiane du gabarit » :
    ``FAMILLES`` contient les quatre teintes mesurées sur les 52 gabarits.
    Une contamination HOMOGÈNE — voile, surbrillance, alpha global — laisse
    la dispersion à zéro et déplace cette médiane : elle ne peut se faire
    prendre que là.

    Deux conditions, et il en fallait bien deux. L'écart absolu dit « cette
    teinte est plausible pour ce jeu de cartes ». La MARGE dit « et elle
    l'est pour une seule famille ». Sans la seconde, un voile vert à 20 %
    faisait lire un pique comme un trèfle (voir ``MARGE_COULEUR_MIN``).

    Returns
    -------
    tuple
        ``(famille, écart, marge)``. ``famille`` vaut ``None`` si l'écart
        dépasse ``ECART_COULEUR_MAX`` ou si la marge est sous
        ``MARGE_COULEUR_MIN`` ; l'écart et la marge sont rendus dans tous
        les cas, pour que l'appelant puisse motiver son refus.
    """
    ecarts = sorted(
        (float(np.linalg.norm(couleur - np.array(ref, dtype=float))), s)
        for s, ref in FAMILLES.items())
    (ecart, meilleure), (suivant, _) = ecarts[0], ecarts[1]
    marge = suivant - ecart
    if ecart > ECART_COULEUR_MAX or marge < MARGE_COULEUR_MIN:
        return None, ecart, marge
    return meilleure, ecart, marge


def part_etrangere(image, tolerance: float = ECART_COULEUR_MAX) -> float | None:
    """Part des pixels de fond dont la VALEUR n'est pas celle de la famille.

    Contrôle proposé en revue : ne pas vérifier seulement que le fond est
    uniforme, mais que sa valeur est la bonne, pixel par pixel et non plus
    seulement en médiane. La distance n'est pas prise à la teinte de
    référence mais au SEGMENT qui la relie au blanc : l'anti-crénelage du
    glyphe vit exactement sur ce segment, et le compter comme étranger
    ferait monter les vraies cartes à 23 % au lieu de 0 %.

    **Mesuré, non retenu comme filtre.** ``banc_robustesse_fond.py``,
    section 4, forme B. Cette part n'est pas sans pouvoir : seuil de forme
    neutralisé, un seuil qui ne coûte aucune vraie carte rattrape une bonne
    moitié des lectures fausses par occultation. Ce n'est pas ce qui décide.

    Ce qui décide est l'apport MARGINAL, une fois ``ECART_RANG_MAX`` en
    place. Sur le résidu que celui-ci laisse — le 6 rogné de 8 % lu « 5 » —
    ``part_etrangere`` rattrape **0 cas sur 20**, parce que ce résidu a un
    fond parfaitement propre : c'est le glyphe, et lui seul, qui est abîmé.
    Sur les deux cartes retournées versionnées elle vaut 0,46 et 0,83, mais
    celles-là la dispersion les refuse déjà.

    « Zéro » vaut pour les données DU DÉPÔT, et le mot était trop large.
    Rejoué avec ``--captures`` sur les 20 découpes de session, ce même
    contrôle rattrape **4 cas sur 10** du résidu, au seuil qui ne coûte
    aucune vraie carte (0,2879). L'apport marginal n'est donc pas nul
    partout : il est nul sur les gabarits et non nul sur le seul jeu de
    données réelles dont on dispose — jeu de 20 découpes et 11 étiquettes
    distinctes, trop petit pour trancher dans l'autre sens. La décision
    d'écarter tient toujours, mais elle tient sur un échantillon, pas sur
    une propriété.

    Une constante de plus pour un cas gagné incertain n'a pas sa place
    aujourd'hui. Elle reste donc une MESURE — un test épingle le zéro du
    dépôt pour qu'on ne la promeuve pas sans refaire le calcul — et non un
    refus. La forme utile du contrôle de valeur, celle qui rattrape
    vraiment, est ``MARGE_COULEUR_MIN``.

    Returns
    -------
    float or None
        La part, entre 0 et 1, ou ``None`` si la découpe n'a pas de fond
        exploitable ou pas de famille identifiable.
    """
    a = _rgb(image)
    pix = a.reshape(-1, 3)
    fond = pix[pix.min(axis=1) < SEUIL_BLANC]
    if not len(fond):
        return None
    famille, _, _ = famille_de_couleur(np.median(fond, axis=0))
    if famille is None:
        return None
    ref = np.array(FAMILLES[famille], dtype=float)
    axe = np.array([255.0, 255.0, 255.0]) - ref
    t = np.clip((fond - ref) @ axe / float(axe @ axe), 0.0, 1.0)
    ecarts = np.linalg.norm(fond - (ref + t[:, None] * axe), axis=1)
    return float((ecarts > tolerance).mean())


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

    famille, ecart_c, marge_c = famille_de_couleur(couleur)

    # Le fond doit être un APLAT. Une carte à moitié retournée ou masquée
    # mélange deux couleurs, et sa médiane désigne une famille qu'elle n'a
    # pas — c'est ainsi qu'un 6♣ en cours de retournement a été lu « 6s »,
    # affirmé. Mieux vaut ne rien lire qu'une carte à moitié visible.
    #
    # La FAMILLE est calculée avant ce refus, et rendue avec lui : c'est elle
    # qui dit si la découpe relève de ce jeu (voir `CarteFondPlein.du_jeu`).
    # Un refus muet sur ce point laissait l'appelant enchaîner sur le hachage,
    # qui affirmait alors une carte fausse.
    if dispersion > DISPERSION_MAX:
        return CarteFondPlein(
            None, famille, None, ecart_c, 1.0, 0.0,
            f"fond non uniforme (dispersion {dispersion:.0f}) — carte "
            "partiellement recouverte ou en cours de retournement")

    if famille is None:
        # Deux refus différents sous un même « None » : la teinte n'est
        # d'aucune famille, ou elle est à peu près à égale distance de deux.
        # Le second cas est celui du voile coloré, et le dire aide à le
        # reconnaître dans une trace.
        pourquoi = ("d'aucune famille connue" if ecart_c > ECART_COULEUR_MAX
                    else f"entre deux familles (marge {marge_c:.0f}, il en "
                         f"faut {MARGE_COULEUR_MIN:.0f})")
        return CarteFondPlein(
            None, None, None, ecart_c, 1.0, 0.0,
            f"teinte ({couleur[0]:.0f},{couleur[1]:.0f},{couleur[2]:.0f}) "
            + pourquoi)

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

    # Treize gabarits et un plus proche voisin : sans seuil, la comparaison
    # rend TOUJOURS un rang, si mauvais soit-il. Une carte dont le haut est
    # rogné de 8 à 20 % garde un fond parfaitement uniforme — la dispersion
    # reste à 0,00 — mais son glyphe est mutilé, et le plus proche voisin
    # devient un autre chiffre : 6♥ lu « 5h », 3♥ lu « Jh ». C'est ici, sur
    # la FORME, que ce cas se refuse, pas sur le fond.
    if e1 > ECART_RANG_MAX:
        return CarteFondPlein(
            None, famille, None, ecart_c, e1, e2 - e1,
            f"glyphe de rang trop loin du gabarit ({e1:.3f}) — carte "
            "rognée, recouverte ou d'un autre habillage")

    return CarteFondPlein(f"{r1}{famille}", famille, r1, ecart_c,
                          e1, e2 - e1)
