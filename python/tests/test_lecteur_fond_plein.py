"""Lire une carte à fond plein : la couleur donne la famille, le chiffre le rang.

L'idée vient de Pierre, et elle est meilleure que le hachage perceptuel sur
cet habillage. Le hachage compare la carte ENTIÈRE : il paie donc le bandeau
d'équité qui masque le bas, un cadrage à sept pixels près, et l'échelle. Un
jeu « 4 couleurs à fond plein » ne demande rien de tout cela — la teinte est
l'enseigne, et il ne reste qu'un chiffre blanc à reconnaître.

Ces tests épinglent les trois propriétés qui font la valeur de la méthode :
elle lit juste, elle est indifférente au cadrage, et elle refuse ce qui n'est
pas une carte.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pfs.vision import lecteur_fond_plein as _lecteur
from pfs.vision.card_recognizer import _TEMPLATE_ROOT
from pfs.vision.lecteur_fond_plein import (
    DISPERSION_MAX,
    ECART_COULEUR_MAX,
    ECART_RANG_MAX,
    FAMILLES,
    MARGE_COULEUR_MIN,
    RANGS,
    couleur_dominante,
    famille_de_couleur,
    lire_carte_fond_plein,
    part_etrangere,
)

_SOLID = _TEMPLATE_ROOT / "pmu_solid"
FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70),
           (92, 26, 38), (84, 84, 90), (16, 16, 20))


@pytest.fixture(scope="module")
def gabarits():
    if not _SOLID.is_dir():
        pytest.skip("habillage pmu_solid absent")
    return _SOLID


def test_les_52_cartes_se_lisent(gabarits) -> None:
    """Aucune carte manquée, aucune confondue."""
    faux, refus = [], []
    for r in RANGS:
        for s in FAMILLES:
            p = gabarits / f"{r}{s}.png"
            if not p.exists():
                continue
            m = lire_carte_fond_plein(p)
            if m.carte is None:
                refus.append((f"{r}{s}", m.motif))
            elif m.carte != f"{r}{s}":
                faux.append((f"{r}{s}", m.carte))
    assert not faux, f"cartes confondues : {faux}"
    assert not refus, f"cartes refusées : {refus}"


def test_les_quatre_familles_sont_separees(gabarits) -> None:
    """La teinte de fond identifie l'enseigne sans ambiguïté.

    C'est l'hypothèse qui fonde toute la méthode : si deux familles se
    rapprochaient, le seuil de couleur cesserait de protéger.
    """
    teintes = {}
    for s in FAMILLES:
        vues = []
        for r in RANGS:
            p = gabarits / f"{r}{s}.png"
            if p.exists():
                vues.append(couleur_dominante(p)[0])
        assert vues
        teintes[s] = np.mean(vues, axis=0)
        dispersion = float(np.max(np.std(vues, axis=0)))
        assert dispersion < 5.0, (
            f"la teinte de {s} varie d'un rang à l'autre ({dispersion:.1f})")

    familles = list(teintes)
    for i, a in enumerate(familles):
        for b in familles[i + 1:]:
            d = float(np.linalg.norm(teintes[a] - teintes[b]))
            assert d > 2 * ECART_COULEUR_MAX, (
                f"{a} et {b} ne sont séparées que de {d:.1f}, pour un seuil "
                f"de {ECART_COULEUR_MAX} : une confusion devient possible")


def test_une_teinte_quelconque_n_est_aucune_famille() -> None:
    """Le filtre de couleur est ce qui protège des non-cartes."""
    rng = np.random.default_rng(7)
    acceptees = 0
    for _ in range(400):
        c = rng.integers(0, 256, 3).astype(float)
        if famille_de_couleur(c)[0] is not None:
            acceptees += 1
    assert acceptees / 400 < 0.05, (
        f"{acceptees}/400 teintes au hasard passent pour une famille : le "
        "seuil de couleur est trop large")


def _non_cartes(n: int = 60):
    rng = np.random.default_rng(20260810)
    for i in range(n):
        h, w = int(rng.integers(90, 140)), int(rng.integers(90, 140))
        if i % 3 == 0:
            yield Image.fromarray(
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
        elif i % 3 == 1:
            c = FEUTRES[i % len(FEUTRES)]
            a = np.tile(np.array(c, dtype=np.uint8), (h, w, 1))
            a = np.clip(a.astype(int) + rng.normal(0, 5, a.shape), 0, 255)
            yield Image.fromarray(a.astype(np.uint8))
        else:
            img = Image.new("RGB", (w, h),
                            tuple(int(v) for v in rng.integers(60, 200, 3)))
            d = ImageDraw.Draw(img)
            for k in range(0, w + h, 7):
                d.line([(k, 0), (0, k)],
                       fill=tuple(int(v) for v in rng.integers(0, 120, 3)),
                       width=2)
            yield img


def test_rien_qui_ne_soit_une_carte_n_est_lu() -> None:
    """Bruit, feutre, dos de cartes : tout doit être refusé."""
    lues = [m for m in (lire_carte_fond_plein(i) for i in _non_cartes())
            if m.carte is not None]
    assert not lues, (
        f"{len(lues)} non-carte(s) lue(s) : "
        f"{[(m.carte, round(m.ecart_couleur, 1)) for m in lues[:5]]}")


def test_une_carte_a_moitie_recouverte_est_refusee() -> None:
    """Le fond d'une carte à fond plein est un APLAT — sinon on ne lit pas.

    Défaut trouvé sur 57 captures réelles de deux tables : une carte saisie
    en pleine animation de retournement, à moitié recouverte par son dos
    brun. La médiane des pixels de fond mélangeait le vert du trèfle et le
    brun du dos, et tombait près de l'ardoise du pique : le 6♣ était lu
    « 6s », **affirmé**, à une distance de 94.

    La mesure qui tranche, sur 315 découpes réelles :
      * 199 découpes lues « sure » : dispersion 0,0 — y compris au maximum.
        Le client rend un aplat parfait ;
      * 116 dos et décors : 20,2 à 72,4.
    Séparation totale. Le seuil est posé à 12, dans le vide.

    Deux réserves sur ces deux lignes, parce qu'elles ont été présentées
    comme plus fortes qu'elles ne sont :

    * « 199 vraies cartes » était faux. Ces 199 découpes étaient celles que
      la chaîne n'avait pas REFUSÉES, sans vérité-terrain — et l'une d'elles
      était précisément une lecture fausse (`banc_verite_captures.py`, 11 août
      2026). Le nombre de vraies cartes lues juste est 198 sur 258 présentes ;
    * une dispersion nulle jusqu'au maximum n'est pas une marge, c'est une
      SATURATION de la statistique — voir `banc_robustesse_fond.py`, qui
      cherche où elle casse.

    Le contrôle, lui, reste juste et nécessaire : c'est sa PORTÉE qui avait
    été surestimée. Il refuse la découpe, mais son refus était muet, et la
    chaîne complète affirmait quand même — voir
    `tests/test_verite_captures.py::test_une_carte_en_retournement_n_est_
    jamais_affirmee_par_la_chaine`, qui emprunte le chemin de la chaîne.

    Le cas est testé sur les DEUX DÉCOUPES RÉELLES, versionnées dans
    `tests/donnees/`. Une première version de ce test imitait le cas en
    posant un rectangle de couleur sur un gabarit : il ne déclenchait rien,
    parce que la couleur d'origine restait majoritaire et que la médiane des
    écarts restait donc nulle. Les vrais retournements sont diagonaux et
    proches de moitié-moitié. L'imitation était plus commode et ne prouvait
    rien — d'où le passage aux vraies images.
    """
    dossier = Path(__file__).parent / "donnees"
    cas = ("pmu_carte_retournee_6c.png", "pmu_carte_retournee_as.png")
    presents = [dossier / n for n in cas if (dossier / n).exists()]
    if not presents:
        pytest.skip("découpes de cartes retournées absentes")

    for chemin in presents:
        lu = lire_carte_fond_plein(chemin)
        assert lu.carte is None, (
            f"{chemin.name} : carte à moitié retournée lue « {lu.carte} ». "
            "C'est le faux positif que ce contrôle doit empêcher — le 6♣ "
            "était lu « 6s », affirmé.")
        assert "uniforme" in lu.motif, (
            f"{chemin.name} : refusée, mais pas pour la bonne raison "
            f"({lu.motif})")


def test_la_dispersion_separe_les_deux_populations(gabarits) -> None:
    """Le seuil doit rester dans un vide, comme celui des cartes.

    Si les deux nuages se rapprochaient — autre habillage, capture
    compressée — ce test le dirait avant que le précédent ne devienne un
    tirage au sort.
    """
    from pfs.vision.lecteur_fond_plein import DISPERSION_MAX, couleur_dominante

    dispersions = [couleur_dominante(p)[2]
                   for p in sorted(gabarits.glob("*.png"))]
    assert dispersions
    assert max(dispersions) < DISPERSION_MAX, (
        f"une vraie carte atteint une dispersion de {max(dispersions):.1f}, "
        f"pour un seuil de {DISPERSION_MAX} : le seuil n'est plus protecteur")


def test_la_lecture_survit_a_un_cadrage_approximatif(gabarits) -> None:
    """Élargir, resserrer ou décaler la découpe ne doit rien changer.

    C'est l'avantage décisif sur le hachage, qui perdait la carte pour sept
    pixels : le chiffre est cherché dans le coin, pas supposé à une place.
    """
    from PIL import Image as PILImage

    p = gabarits / "5c.png"
    if not p.exists():
        pytest.skip("gabarit 5c absent")
    src = PILImage.open(p).convert("RGB")
    w, h = src.size

    # Découpe posée sur du feutre, puis recadrée de plusieurs façons.
    marge = 14
    fond = PILImage.new("RGB", (w + 2 * marge, h + 2 * marge), FEUTRES[0])
    fond.paste(src, (marge, marge))

    cadrages = {
        "exact": (marge, marge, w, h),
        "large": (marge - 8, marge - 8, w + 16, h + 16),
        "serré": (marge + 4, marge + 3, w - 8, h - 6),
        "décalé": (marge + 5, marge + 4, w, h),
        "bas coupé": (marge, marge, w, int(h * 0.66)),
    }
    ratés = []
    for nom, (x, y, ww, hh) in cadrages.items():
        m = lire_carte_fond_plein(fond.crop((x, y, x + ww, y + hh)))
        if m.carte != "5c":
            ratés.append((nom, m.carte, m.motif))
    assert not ratés, f"cadrages perdus : {ratés}"


# ==========================================================================
# Robustesse du contrôle de fond — chantier du 11 août 2026.
#
# La mesure qui justifiait le seuil de dispersion — « 199 vraies cartes,
# min, médiane, p95 ET maximum tous à 0,0 » — n'était pas une marge mais une
# SATURATION : la dispersion est une médiane, elle ne bouge pas tant que la
# contamination reste minoritaire. Les tests qui suivent épinglent ce que ce
# contrôle voit, ce qu'il ne voit pas, et les deux seuils ajoutés pour
# couvrir ce qu'il ne voyait pas. Les chiffres viennent de
# `banc_robustesse_fond.py`, rejouable.
# ==========================================================================

#: Couleurs plausibles pour un élément qui recouvre le haut d'une carte :
#: le dos brun d'une carte en cours de distribution, le feutre, un pavé
#: d'interface, une autre carte posée dessus.
OCCULTANTS = ((54, 34, 7), (24, 86, 52), (16, 16, 20), (53, 59, 66))


def _jpeg(img: Image.Image, qualite: int, sous_ech: int = 2) -> Image.Image:
    """Ré-encodage JPEG en mémoire, 4:2:0 par défaut."""
    tampon = io.BytesIO()
    img.convert("RGB").save(tampon, format="JPEG", quality=qualite,
                            subsampling=sous_ech)
    tampon.seek(0)
    return Image.open(tampon).convert("RGB")


def _rogne_le_haut(img: Image.Image, part: float, couleur) -> Image.Image:
    """Remplace la bande HAUTE — celle qui porte le glyphe de rang."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64).copy()
    a[:int(round(a.shape[0] * part))] = np.array(couleur, dtype=float)
    return Image.fromarray(a.astype(np.uint8))


def _voile(img: Image.Image, couleur, opacite: float) -> Image.Image:
    """Contamination HOMOGÈNE : tous les pixels bougent ensemble."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    c = np.array(couleur, dtype=float)
    return Image.fromarray(
        np.clip(a * (1 - opacite) + c * opacite, 0, 255).astype(np.uint8))


def _cartes_du_depot(gabarits) -> list[tuple[Image.Image, str]]:
    """Les 52 gabarits, plus les deux découpes réelles versionnées."""
    out = [(Image.open(gabarits / f"{r}{s}.png").convert("RGB"), f"{r}{s}")
           for r in RANGS for s in FAMILLES
           if (gabarits / f"{r}{s}.png").exists()]
    donnees = Path(__file__).parent / "donnees"
    for nom, carte in (("pmu_play_hero_5c.png", "5c"),
                       ("pmu_play_hero_3h.png", "3h")):
        if (donnees / nom).exists():
            out.append((Image.open(donnees / nom).convert("RGB"), carte))
    return out


def test_une_carte_rognee_du_haut_n_est_pas_lue_comme_une_autre(gabarits) -> None:
    """Le défaut central : occultation minoritaire, lecture fausse affirmée.

    Recouvrir le haut d'une carte sur 10 à 20 % de sa hauteur ne trouble en
    rien son fond : la couleur d'origine reste largement majoritaire, la
    dispersion vaut EXACTEMENT 0,00 et le contrôle « fond non uniforme »
    laisse passer. Mais le glyphe de rang, lui, est mutilé, et la comparaison
    aux treize gabarits — un plus proche voisin sans seuil de refus — rend
    alors un AUTRE chiffre, affirmé : 2♣ lu « Kc », 3♥ lu « Jh », 5♣ lu
    « 8c ».

    Mesuré par `banc_robustesse_fond.py` : sans `ECART_RANG_MAX`, 615
    lectures fausses sur les données du dépôt et 166 sur 20 découpes de
    session — dont le contrôle de fond voyait ZÉRO.

    Ce test tombe si `ECART_RANG_MAX` est retiré, relevé au-dessus de 0,22,
    ou si la comparaison de rang redevient un plus proche voisin sans refus.
    """
    cartes = _cartes_du_depot(gabarits)
    assert cartes, "aucune carte de référence"
    faux = []
    for img, attendu in cartes:
        for part in (0.10, 0.12, 0.15, 0.20):
            for couleur in OCCULTANTS:
                lu = lire_carte_fond_plein(_rogne_le_haut(img, part, couleur))
                if lu.sure and lu.carte != attendu:
                    faux.append((attendu, lu.carte, f"{part:.0%}", couleur,
                                 round(lu.ecart_rang, 4)))
    assert not faux, (
        f"{len(faux)} carte(s) rognée(s) lue(s) comme une AUTRE carte, "
        f"affirmée(s) : {faux[:6]}")

    # L'autre bord du seuil : trop BAS, il perdrait des vraies cartes en
    # silence, parce qu'un refus ne fait pas de bruit. On le borne donc par la
    # mesure, pas par une constante écrite.
    #
    # Le seuil est NEUTRALISÉ pendant cette mesure, et c'est tout le sujet.
    # Mesurer `ecart_rang` sur les lectures encore justes ALORS QUE le seuil
    # agit ne prouve rien : une lecture dont l'écart dépasse le seuil est
    # refusée, donc n'est plus « juste », donc sort de la population. Le
    # maximum se retrouve borné par le seuil lui-même et la comparaison
    # devient vraie d'office. Vérifié le 11 août 2026 : sous cette forme,
    # l'assertion passait encore à ECART_RANG_MAX = 0,02, seuil qui perd
    # pourtant 100 lectures justes sur 265.
    ancien = _lecteur.ECART_RANG_MAX
    _lecteur.ECART_RANG_MAX = 1.0        # 1.0 : aucun masque ne l'atteint
    try:
        pires = [lu.ecart_rang for img, attendu in cartes
                 for r in [img] + [_jpeg(img, q) for q in (95, 85, 70, 50, 30)]
                 for lu in [lire_carte_fond_plein(r)]
                 if lu.carte == attendu]
    finally:
        _lecteur.ECART_RANG_MAX = ancien
    assert pires
    assert ECART_RANG_MAX > max(pires), (
        f"ECART_RANG_MAX={ECART_RANG_MAX} est sous le pire écart d'une "
        f"lecture JUSTE ({max(pires):.4f}, JPEG compris) : le seuil se met à "
        "refuser des cartes lisibles, et un refus ne fait pas de bruit")


def test_la_dispersion_est_aveugle_a_une_occultation_minoritaire(gabarits) -> None:
    """La LIMITE du contrôle de fond, épinglée pour qu'on cesse de la nier.

    Le motif de refus dit « carte partiellement recouverte ». C'est faux au
    sens large : la dispersion étant une médiane, elle ne réagit qu'au
    voisinage de 50 % de recouvrement. Sous ce seuil, elle vaut exactement
    zéro — pas « presque zéro », zéro.

    Ce test le prouve en neutralisant le seuil de forme : les lectures
    redeviennent alors fausses, et leur dispersion reste à 0,00. Il tombera
    le jour où le contrôle de fond verra enfin ces cas — ce serait une bonne
    nouvelle, et il faudrait alors réécrire la note de `DISPERSION_MAX`
    plutôt que ce test.
    """
    cartes = _cartes_du_depot(gabarits)
    ancien = _lecteur.ECART_RANG_MAX
    _lecteur.ECART_RANG_MAX = 1.0        # 1.0 : aucun masque ne l'atteint
    try:
        fausses = [(couleur_dominante(r)[2], attendu, lu.carte)
                   for img, attendu in cartes
                   for part in (0.10, 0.15, 0.20)
                   for couleur in OCCULTANTS
                   for r in [_rogne_le_haut(img, part, couleur)]
                   for lu in [lire_carte_fond_plein(r)]
                   if lu.sure and lu.carte != attendu]
    finally:
        _lecteur.ECART_RANG_MAX = ancien

    assert fausses, (
        "aucune lecture fausse fabriquée : le montage ne prouve plus rien")
    pire = max(d for d, _, _ in fausses)
    assert pire == 0.0, (
        f"la dispersion des {len(fausses)} lectures fausses atteint "
        f"{pire:.2f} : elle en voit désormais quelque chose. Bonne nouvelle, "
        "mais la note de DISPERSION_MAX ne le dit pas encore.")
    assert pire < DISPERSION_MAX


def test_un_voile_colore_ne_fait_pas_changer_une_carte_de_famille(gabarits) -> None:
    """Contamination HOMOGÈNE : la dispersion ne peut rien, la teinte doit.

    Un voile uniforme laisse un aplat parfait — dispersion 0,00 — et déplace
    toutes les teintes ensemble. Le seul contrôle capable de le voir est
    celui de la couleur, et sa forme absolue ne suffisait pas : avec quatre
    familles séparées de 74,8 et une tolérance de 34, un déplacement de 34
    dans la bonne direction laisse la teinte PLUS PRÈS de la voisine que de
    la sienne.

    Le balayage systématique du banc trouve le cas : un voile vert (0,255,0)
    à 20 % amène le pique (53,59,66) sur (42,98,52), à 32,0 du trèfle contre
    42,9 du pique. Le 2♠ était lu « 2c », affirmé.

    Ce test tombe si `MARGE_COULEUR_MIN` est retiré ou descendu sous 30.
    """
    cartes = _cartes_du_depot(gabarits)
    # Le vert pur à 20 % et son voisin : les deux couples que le banc isole
    # sur 216 teintes × 8 opacités. Les garder nommés rend le test rapide
    # sans rien lui retirer — le balayage complet vit dans le banc.
    voiles = (((0, 255, 0), 0.20), ((51, 255, 51), 0.30),
              ((0, 255, 0), 0.25), ((26, 255, 26), 0.25))
    faux = []
    for img, attendu in cartes:
        for couleur, opacite in voiles:
            lu = lire_carte_fond_plein(_voile(img, couleur, opacite))
            if lu.sure and lu.carte != attendu:
                faux.append((attendu, lu.carte, couleur, opacite))
    assert not faux, (
        f"{len(faux)} carte(s) changée(s) de famille par un voile homogène, "
        f"affirmée(s) : {faux[:6]}")


def test_la_marge_de_couleur_separe_vraies_cartes_et_teintes_ambigues(
        gabarits) -> None:
    """Le seuil de marge doit rester dans un vide, et le vide doit être large.

    Mesuré par le banc : vraies cartes (PNG et JPEG jusqu'à q30) marge
    minimale 66,8 ; lectures d'une autre famille sous voile, 10,8 à 29,4.
    Le seuil est à 45, dans le vide [29,4 ; 66,8].

    Ce test tombe si un habillage futur rapproche deux familles, ou si la
    compression met à mal la médiane de teinte — c'est-à-dire AVANT que le
    test précédent ne devienne un tirage au sort.
    """
    cartes = _cartes_du_depot(gabarits)
    marges = [famille_de_couleur(couleur_dominante(r)[0])[2]
              for img, _ in cartes
              for r in [img] + [_jpeg(img, q) for q in (95, 85, 70, 50, 30)]]
    assert marges
    pire = min(marges)
    # Borne écrite en dur, et non rapportée au seuil : rapportée au seuil,
    # elle deviendrait vraie d'office si quelqu'un mettait le seuil à zéro.
    assert pire > 60.0, (
        f"la marge de couleur d'une vraie carte descend à {pire:.1f} : le "
        "vide mesuré [29,4 ; 66,8] s'est refermé, et le seuil ne protège plus")
    # Le seuil doit rester DANS ce vide : ni sous le nuage des teintes
    # ambiguës mesurées (29,4), ni au-dessus du pire vrai positif.
    assert 30.0 <= MARGE_COULEUR_MIN <= pire, (
        f"MARGE_COULEUR_MIN={MARGE_COULEUR_MIN} n'est plus dans le vide "
        f"[29,4 ; {pire:.1f}] : sous 30 il laisse repasser le voile vert, "
        "au-dessus il coûte des vraies cartes")


def test_la_lecture_survit_a_une_compression_jpeg(gabarits) -> None:
    """Le seuil de dispersion ne doit pas rendre le lecteur aveugle au JPEG.

    Les 0,0 mesurés à l'origine venaient d'une chaîne PNG sans perte. Sous
    JPEG 4:2:0 — le sous-échantillonnage par défaut, donc le pire cas pour
    une carte dont l'identité EST une couleur — la dispersion monte : 4,1 à
    q95 et 8,5 à q85 sur les gabarits de 65 px, 3,0 et 5,0 sur les découpes
    réelles de 130 px.

    Jusqu'à q85 inclus, aucune carte ne doit être perdue NI confondue. Ce
    test tombe si `DISPERSION_MAX` descend sous 9, et il tombe aussi si une
    compression se met à produire une lecture fausse — ce qu'aucune qualité
    jusqu'à q30 n'a fait au banc.
    """
    cartes = _cartes_du_depot(gabarits)
    perdues, fausses = [], []
    for img, attendu in cartes:
        for q in (95, 85):
            for sous_ech in (0, 2):
                lu = lire_carte_fond_plein(_jpeg(img, q, sous_ech))
                if lu.carte is None:
                    perdues.append((attendu, q, sous_ech, lu.motif))
                elif lu.carte != attendu:
                    fausses.append((attendu, lu.carte, q, sous_ech))
    assert not fausses, f"lectures fausses sous JPEG : {fausses[:6]}"
    assert not perdues, (
        f"{len(perdues)} carte(s) perdue(s) sous JPEG q85 ou mieux : "
        f"{perdues[:6]}")


def test_part_etrangere_n_ajoute_rien_par_dessus_le_seuil_de_forme(
        gabarits) -> None:
    """Le contrôle de valeur PIXEL À PIXEL, mesuré et écarté — pas oublié.

    La revue proposait de vérifier que la valeur du fond est la bonne, et non
    seulement qu'elle est uniforme. Pris pixel à pixel — part des pixels de
    fond loin du segment [teinte de la famille ; blanc] — ce contrôle n'est
    pas sans pouvoir : seuil de forme neutralisé, il rattrape une bonne part
    des lectures fausses par occultation, sans coûter de vraie carte.

    Mais ce n'est pas la question. La question est : APPORTE-T-IL QUELQUE
    CHOSE que `ECART_RANG_MAX` ne fait pas déjà ? Sur le résidu que le seuil
    de forme laisse passer (le 6 rogné lu « 5 »), il rattrape zéro — sur les
    données DU DÉPÔT, qui sont les seules que ce test peut atteindre.

    La portée s'arrête là, et il faut le dire ici plutôt que le laisser
    croire : rejoué avec `--captures`, le banc mesure 4 cas rattrapés sur 10
    du résidu, à coût nul, sur les 20 découpes de session. Ce test épingle
    donc le zéro DU DÉPÔT, pas une propriété. Le jour où l'on reprendra la
    décision, c'est le banc avec captures qui tranchera, pas ce test.

    Ce test épingle ce zéro. Il tombera si le résidu change de nature, ou si
    quelqu'un modifie `part_etrangere` — et il faudra alors reprendre la
    décision, chiffres à l'appui, plutôt que la subir. Il TOMBE bien : la
    variante « quart haut de la découpe », essayée, rattrape 18 cas sur 20 et
    fait tomber ce test. Elle a pourtant été écartée elle aussi, pour une
    raison que seule la mesure donne — sous cadrage approximatif, une découpe
    élargie de 8 px fait entrer le feutre dans le quart haut et la statistique
    monte à 0,55 sur des cartes parfaitement lisibles, contre 0,31 au cadrage
    exact. Voir `banc_robustesse_fond.py`, section 4 formes C et D.
    """
    cartes = _cartes_du_depot(gabarits)
    vraies = [p for img, _ in cartes
              for r in [img] + [_jpeg(img, q) for q in (95, 85, 70, 50, 30)]
              for p in [part_etrangere(r)] if p is not None]
    # Le résidu : lectures encore FAUSSES alors que le seuil de forme est
    # actif. C'est le seul gain qu'un contrôle supplémentaire pourrait faire.
    residu = [p for img, attendu in cartes
              for part in (0.05, 0.08, 0.10, 0.15, 0.20)
              for couleur in OCCULTANTS
              for r in [_rogne_le_haut(img, part, couleur)]
              if lire_carte_fond_plein(r).carte not in (None, attendu)
              for p in [part_etrangere(r)] if p is not None]

    assert vraies, "aucune vraie carte mesurée"
    assert residu, (
        "plus aucune lecture fausse ne subsiste : le résidu a disparu, ce "
        "test ne mesure plus rien et la note de ECART_RANG_MAX est à refaire")
    # Seuil le plus généreux qui ne coûte AUCUNE vraie carte.
    gratuit = max(vraies)
    rattrapees = sum(1 for p in residu if p > gratuit)
    assert rattrapees == 0, (
        f"part_etrangere rattrape désormais {rattrapees}/{len(residu)} du "
        f"résidu sans coûter une vraie carte (seuil {gratuit:.4f}) : elle "
        "mérite d'être reconsidérée comme refus, chiffres à l'appui")

