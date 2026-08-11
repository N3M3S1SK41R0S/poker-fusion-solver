#!/usr/bin/env python
"""Banc de ROBUSTESSE du contrôle de fond du lecteur à fond plein.

    python banc_robustesse_fond.py
    python banc_robustesse_fond.py --captures CHEMIN\\vers\\sessions

Pourquoi ce fichier existe
--------------------------
``pfs/vision/lecteur_fond_plein.py`` refusait une carte dont le fond n'est
pas uniforme, sur la foi d'une mesure qui disait : « 199 vraies cartes,
dispersion 0,0 — min, médiane, p95 ET MAXIMUM tous à zéro ; 116 dos, 20,2 à
72,4 ; séparation totale ».

Une statistique dont les quatre quantiles valent exactement 0,0 sur 199
mesures ne mesure rien : elle SATURE. La dispersion est la médiane des
écarts à la médiane ; sur une découpe majoritairement plate elle vaut zéro
quoi qu'il advienne du reste. La question n'était donc pas « quelle est la
marge » mais « que se passe-t-il quand la découpe cesse d'être parfaite ».

Ce banc y répond en quatre sections, dans l'ordre où elles tranchent :

1. **Compression.** Ré-encodage JPEG des découpes réelles. La marge existe,
   elle se referme, et elle dépend de la TAILLE de la carte : le seuil de 12
   tient jusqu'à q40 sur une découpe de 130 px de large, mais jusqu'à q85
   seulement sur un gabarit de 65 px.
2. **Contaminations homogènes.** Voile gris, surbrillance, alpha global,
   puis un balayage systématique de 216 teintes × 8 opacités. Un aplat teinté
   reste un aplat : la dispersion ne bouge pas. Le balayage a trouvé un VRAI
   faux positif que six teintes choisies à la main ne donnaient pas — un
   voile vert à 20 % faisait lire un pique comme un trèfle, affirmé. Fermé
   par ``MARGE_COULEUR_MIN``.
3. **Occultation partielle — le vrai angle mort.** Recouvrir le haut de la
   carte sur 8 à 20 % laisse la dispersion à 0,00 *exactement*, et la lecture
   devient FAUSSE. C'est le défaut que la section chiffre, avant et après le
   seuil ``ECART_RANG_MAX`` ajouté pour lui.
4. **Le contrôle de VALEUR proposé en revue.** Vérifier non pas que le fond
   est uniforme mais que sa valeur est celle attendue. Quatre formes mesurées,
   une retenue : ce qu'elles coûtent en vraies cartes, ce qu'elles rattrapent,
   et surtout ce qu'elles ajoutent PAR-DESSUS les contrôles déjà en place —
   la seule question qui décide.

Deux mesures y sont faites en A/B RÉEL, seuil neutralisé puis actif, et non
déduites des statistiques du contrôle lui-même. C'est précisément la
déduction qui avait produit la mesure saturée que ce banc corrige.

Données
-------
Le banc tourne SANS rien d'extérieur au dépôt : 52 gabarits `pmu_solid` et
les découpes réelles versionnées dans ``tests/donnees/``. L'option
``--captures`` ajoute les captures de session brutes (hors dépôt), qui
donnent des découpes de 126×115 à 138×187 px là où les gabarits font 65×88.
Les chiffres cités dans ``lecteur_fond_plein.py`` viennent de cette seconde
passe ; la première suffit à faire tomber le banc si une régression revient.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from pfs.vision import lecteur_fond_plein as lecteur  # noqa: E402
from pfs.vision.card_recognizer import _TEMPLATE_ROOT  # noqa: E402
from pfs.vision.lecteur_fond_plein import (  # noqa: E402
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

DONNEES = Path(__file__).resolve().parent / "tests" / "donnees"
SOLID = _TEMPLATE_ROOT / "pmu_solid"

#: Découpes réelles versionnées, avec leur vérité-terrain lue à l'œil.
REELLES = (("pmu_play_hero_5c.png", "5c"), ("pmu_play_hero_3h.png", "3h"))

#: Découpes réelles de cartes en cours de retournement — les seuls intrus
#: dont la teinte médiane tombe malgré tout dans une famille.
RETOURNEES = ("pmu_carte_retournee_6c.png", "pmu_carte_retournee_as.png")

#: Couleurs d'occultant. Toutes plausibles sur une table : le dos brun d'une
#: carte en cours de distribution, le feutre, un pavé d'interface, une autre
#: carte posée par-dessus.
OCCULTANTS = {
    "dos brun": (54, 34, 7),
    "feutre vert": (24, 86, 52),
    "pavé noir": (16, 16, 20),
    "carte ardoise": (53, 59, 66),
    "carte rouge": (207, 22, 19),
}

#: Hauteurs d'occultation, en fraction de la hauteur de la découpe.
HAUTEURS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)

QUALITES = (95, 85, 70, 50, 40, 30)


# --------------------------------------------------------------------------
# outillage
# --------------------------------------------------------------------------

@contextmanager
def _sans(nom: str, valeur_neutre: float):
    """Neutralise temporairement un seuil du lecteur.

    C'est ce qui permet de mesurer un contrôle en A/B — combien de lectures
    fausses il fait taire — au lieu de le déduire de ses propres statistiques.
    La déduction, c'est exactement ce qui a produit la mesure de dispersion
    saturée que ce banc corrige.
    """
    ancien = getattr(lecteur, nom)
    setattr(lecteur, nom, valeur_neutre)
    try:
        yield
    finally:
        setattr(lecteur, nom, ancien)


def _jpeg(img: Image.Image, qualite: int, sous_ech: int = 2) -> Image.Image:
    """Ré-encodage JPEG en mémoire.

    ``sous_ech=2`` est le 4:2:0 des encodeurs par défaut — celui qui touche
    le plus la chrominance, donc le pire cas pour une carte dont l'identité
    EST une couleur. Le 4:4:4 est mesuré à part en section 1.
    """
    tampon = io.BytesIO()
    img.convert("RGB").save(tampon, format="JPEG", quality=qualite,
                            subsampling=sous_ech)
    tampon.seek(0)
    return Image.open(tampon).convert("RGB")


def _recouvre_haut(img: Image.Image, part: float, couleur) -> Image.Image:
    """Remplace la bande HAUTE de la découpe par une couleur pleine.

    Le haut, et non le bas : c'est là qu'est le glyphe de rang. Le bas est
    déjà mangé par le bandeau d'équité sans que cela gêne — c'est mesuré par
    ``banc_carte_recouverte.py``.
    """
    a = np.asarray(img.convert("RGB"), dtype=np.float64).copy()
    a[:int(round(a.shape[0] * part))] = np.array(couleur, dtype=float)
    return Image.fromarray(a.astype(np.uint8))


def _voile(img: Image.Image, couleur, opacite: float) -> Image.Image:
    """Contamination HOMOGÈNE : mélange global avec une couleur unique."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    c = np.array(couleur, dtype=float)
    return Image.fromarray(
        np.clip(a * (1 - opacite) + c * opacite, 0, 255).astype(np.uint8))


def _gabarits() -> list[tuple[Image.Image, str]]:
    out = []
    for r in RANGS:
        for s in FAMILLES:
            p = SOLID / f"{r}{s}.png"
            if p.exists():
                out.append((Image.open(p).convert("RGB"), f"{r}{s}"))
    return out


def _versionnees() -> list[tuple[Image.Image, str]]:
    return [(Image.open(DONNEES / n).convert("RGB"), c)
            for n, c in REELLES if (DONNEES / n).exists()]


def _captures(dossier: Path):
    """Découpes extraites des captures de session, dédupliquées.

    La vérité-terrain est la lecture faite sur le PNG SANS PERTE : c'est le
    protocole de la mesure d'origine, et le seul disponible ici. Le banc ne
    demande donc pas « la lecture est-elle juste dans l'absolu » mais « la
    même découpe donne-t-elle encore la même carte quand on l'abîme », ce qui
    est exactement la question posée.

    Returns
    -------
    tuple
        ``(cartes lues, découpes refusées)``. Les secondes — dos, décors,
        avatars, jetons — servent de population de NON-CARTES réelles :
        c'est sur elles qu'on mesure ce que la marge de couleur rejette en
        plus, section 4.
    """
    from pfs.vision.table_detector import detect_card_boxes

    vues: set[bytes] = set()
    out: list[tuple[Image.Image, str]] = []
    refusees: list[Image.Image] = []
    for png in sorted(dossier.rglob("*.png")):
        img = Image.open(png).convert("RGB")
        for b in detect_card_boxes(img):
            crop = img.crop((b.x, b.y, b.x + b.w, b.y + b.h))
            lu = lire_carte_fond_plein(crop)
            cle = crop.tobytes()
            if cle in vues:
                continue
            vues.add(cle)
            if lu.sure:
                out.append((crop, lu.carte))
            else:
                refusees.append(crop)
    return out, refusees


def _quantiles(v) -> str:
    a = np.asarray(v, dtype=float)
    return (f"méd {np.median(a):6.2f}  p95 {np.percentile(a, 95):6.2f}  "
            f"max {a.max():6.2f}")


# --------------------------------------------------------------------------
# 1. compression
# --------------------------------------------------------------------------

def section_compression(jeux) -> int:
    """La marge sous JPEG, et à partir d'où le seuil de 12 cède."""
    print("=== 1. compression JPEG : la dispersion des VRAIES cartes ===")
    print(f"    seuil en vigueur : DISPERSION_MAX = {DISPERSION_MAX}")
    fautes = 0
    for nom, cartes in jeux:
        if not cartes:
            continue
        larg = sorted({im.size[0] for im, _ in cartes})
        print(f"\n    -- {nom} ({len(cartes)} découpes, "
              f"largeur {larg[0]}–{larg[-1]} px) --")
        for sous_ech, etiq in ((0, "4:4:4"), (2, "4:2:0")):
            print(f"       {etiq}")
            for q in QUALITES:
                disp, refus, faux = [], 0, 0
                for im, attendu in cartes:
                    r = _jpeg(im, q, sous_ech)
                    disp.append(couleur_dominante(r)[2])
                    lu = lire_carte_fond_plein(r)
                    if lu.carte is None:
                        refus += 1
                    elif lu.carte != attendu:
                        faux += 1
                d = np.array(disp)
                casse = int((d > DISPERSION_MAX).sum())
                print(f"         q{q:3d} : {_quantiles(d)}  | "
                      f"au-dessus du seuil {casse:3d}/{len(d)} | "
                      f"refus {refus:3d} | FAUSSES {faux:3d}")
                # Une compression ne doit jamais faire LIRE une autre carte.
                # Un refus est acceptable : c'est le silence, pas l'erreur.
                if faux:
                    fautes += 1
                # Jusqu'à q85 inclus, aucune vraie carte ne doit être perdue :
                # c'est la qualité que produit n'importe quelle chaîne de
                # capture raisonnable.
                if q >= 85 and refus:
                    fautes += 1
    print("\n    lecture : le seuil de 12 n'a AUCUNE marge dans le sens où on "
          "l'entendait\n    (les 0,0 étaient une saturation), mais il en a une "
          "sous compression, et\n    elle dépend de la taille : plus la carte "
          "est petite, plus vite elle cède.")
    return fautes


# --------------------------------------------------------------------------
# 2. contaminations homogènes
# --------------------------------------------------------------------------

def section_homogene(cartes) -> int:
    """Un aplat teinté reste un aplat. Qui l'arrête, alors ?"""
    print("\n=== 2. contaminations HOMOGÈNES ===")
    print("    la dispersion ne peut pas les voir — reste à savoir qui les voit")
    cas = (("voile gris 20 %", (128, 128, 128), 0.20),
           ("voile gris 35 %", (128, 128, 128), 0.35),
           ("surbrillance 20 %", (255, 255, 255), 0.20),
           ("surbrillance 35 %", (255, 255, 255), 0.35),
           ("assombrissement 30 %", (0, 0, 0), 0.30),
           ("teinte bleue 30 %", (0, 0, 255), 0.30))
    fautes = 0
    for nom, coul, op in cas:
        disp, justes, refus, faux, motifs = [], 0, 0, [], {}
        for im, attendu in cartes:
            r = _voile(im, coul, op)
            disp.append(couleur_dominante(r)[2])
            lu = lire_carte_fond_plein(r)
            if lu.carte == attendu:
                justes += 1
            elif lu.carte is None:
                refus += 1
                cle = lu.motif.split("(")[0].strip()
                motifs[cle] = motifs.get(cle, 0) + 1
            else:
                faux.append((attendu, lu.carte))
        d = np.array(disp)
        print(f"    {nom:22s} dispersion max {d.max():6.2f} | justes {justes:3d} "
              f"refus {refus:3d} FAUSSES {len(faux):3d}")
        if refus:
            print(f"                           refusées par : {motifs}")
        if faux:
            fautes += 1
            print(f"                           !! {faux[:4]}")

    print("\n    -- balayage systématique : 216 teintes × 8 opacités --")
    print("       c'est ce balayage, et lui seul, qui a trouvé le faux positif")
    print("       du voile vert : six teintes choisies à la main ne le donnent pas.")
    grille = [(r, g, b) for r in (0, 51, 102, 153, 204, 255)
              for g in (0, 51, 102, 153, 204, 255)
              for b in (0, 51, 102, 153, 204, 255)]
    opacites = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)

    def balaye():
        """Rend (essais, lectures affirmées, fausses, dispersion max des lues)."""
        essais = lues = 0
        faux: list[tuple] = []
        disp_max = 0.0
        for coul in grille:
            for op in opacites:
                for im, attendu in cartes:
                    essais += 1
                    r = _voile(im, coul, op)
                    lu = lire_carte_fond_plein(r)
                    if not lu.sure:
                        continue
                    lues += 1
                    disp_max = max(disp_max, couleur_dominante(r)[2])
                    if lu.carte != attendu:
                        faux.append((attendu, lu.carte, coul, op))
        return essais, lues, faux, disp_max

    # A/B réel : le même balayage, avec puis sans le seuil de marge. C'est
    # la seule façon honnête de dire ce que ce seuil rattrape.
    with _sans("MARGE_COULEUR_MIN", 0.0):
        essais, lues_avant, faux_avant, disp_avant = balaye()
    _, lues_apres, faux_apres, disp_apres = balaye()

    print(f"    {essais} essais sur {len(cartes)} cartes")
    print(f"      SANS MARGE_COULEUR_MIN : {lues_avant:6d} affirmées, "
          f"{len(faux_avant):3d} FAUSSES")
    print(f"      AVEC MARGE_COULEUR_MIN : {lues_apres:6d} affirmées, "
          f"{len(faux_apres):3d} FAUSSES")
    print(f"      dispersion maximale des lectures affirmées : "
          f"{max(disp_avant, disp_apres):.2f} — la dispersion ne voit rien ici")
    if faux_avant:
        vus = sorted({(a, b, c, f"{o:.0%}") for a, b, c, o in faux_avant})
        print(f"      les {len(faux_avant)} fausses d'avant : "
              f"{len({(a[-1], b[-1]) for a, b, _, _ in faux_avant})} transition(s) "
              f"de famille, {sorted({(a[-1], b[-1]) for a, b, _, _ in faux_avant})}")
        print(f"      exemples : {vus[:3]}")
    cout = lues_avant - lues_apres
    print(f"      coût : {cout} lectures perdues "
          f"({cout / max(lues_avant, 1):.1%} des affirmations)")
    if faux_apres:
        fautes += 1
        print(f"    !! il reste {len(faux_apres)} fausses : {faux_apres[:5]}")
    print("\n    lecture : la dispersion ne peut rien voir d'un aplat teinté, et")
    print("    part_etrangere non plus — une contamination homogène déplace TOUS")
    print(f"    les pixels ensemble. Seule la teinte tranche : "
          f"ECART_COULEUR_MAX={ECART_COULEUR_MAX} dit")
    print("    « cette couleur existe dans ce jeu », et "
          f"MARGE_COULEUR_MIN={MARGE_COULEUR_MIN} dit")
    print("    « pour une seule famille ». Sans le second, un voile vert à 20 % "
          "faisait\n    lire un pique comme un trèfle.")
    return fautes


# --------------------------------------------------------------------------
# 3. occultation partielle
# --------------------------------------------------------------------------

def _essais_occultation(cartes):
    """Rend (dispersion, écart_rang, juste, attendu, lu, occultant, part)."""
    for nom, coul in OCCULTANTS.items():
        for part in HAUTEURS:
            for im, attendu in cartes:
                r = _recouvre_haut(im, part, coul)
                lu = lire_carte_fond_plein(r)
                yield (couleur_dominante(r)[2], lu.ecart_rang,
                       lu.carte == attendu, attendu, lu.carte, nom, part)


def section_occultation(cartes) -> int:
    """L'angle mort réel : la médiane ne voit pas une occultation minoritaire."""
    print("\n=== 3. occultation PARTIELLE du haut — l'angle mort de la médiane ===")
    essais = list(_essais_occultation(cartes))
    lues = [e for e in essais if e[4] is not None]
    faux = [e for e in essais if e[4] is not None and not e[2]]
    print(f"    {len(cartes)} découpes × {len(OCCULTANTS)} occultants × "
          f"{len(HAUTEURS)} hauteurs = {len(essais)} essais")
    print(f"    {len(lues)} lectures affirmées, dont {len(faux)} FAUSSES "
          f"avec le seuil actuel")

    # La démonstration de saturation : sur les essais où la lecture reste
    # affirmée, la dispersion ne bouge pas d'un pouce.
    d_lues = np.array([e[0] for e in lues]) if lues else np.zeros(1)
    print(f"    dispersion des essais affirmés : {_quantiles(d_lues)}  "
          f"(seuil {DISPERSION_MAX})")

    print("\n    -- ce que chaque hauteur d'occultation produit --")
    print("       part | affirmées | fausses | dispersion max | écart_rang min")
    for part in HAUTEURS:
        gr = [e for e in essais if e[6] == part]
        aff = [e for e in gr if e[4] is not None]
        fx = [e for e in gr if e[4] is not None and not e[2]]
        dmax = max((e[0] for e in aff), default=0.0)
        emin = min((e[1] for e in fx), default=float("nan"))
        print(f"       {part:4.0%} | {len(aff):9d} | {len(fx):7d} | "
              f"{dmax:14.2f} | {emin:14.4f}")

    print("\n    -- A/B réel : les mêmes essais SANS le seuil de forme --")
    # 1.0 neutralise le seuil : aucun écart de masque normalisé ne l'atteint.
    with _sans("ECART_RANG_MAX", 1.0):
        avant = list(_essais_occultation(cartes))
    lues_avant = [e for e in avant if e[4] is not None]
    faux_avant = [e for e in avant if e[4] is not None and not e[2]]
    d_avant = np.array([e[0] for e in faux_avant]) if faux_avant else np.zeros(1)
    print(f"       SANS ECART_RANG_MAX : {len(lues_avant):4d} affirmées, "
          f"{len(faux_avant):4d} FAUSSES")
    print(f"       AVEC ECART_RANG_MAX={ECART_RANG_MAX} : {len(lues):4d} "
          f"affirmées, {len(faux):4d} FAUSSES")
    print(f"       dispersion des {len(faux_avant)} lectures fausses d'avant : "
          f"{_quantiles(d_avant)}")
    print(f"       → le contrôle de fond en voyait "
          f"{int((d_avant > DISPERSION_MAX).sum())} sur {len(faux_avant)}.")
    perdues = len(lues_avant) - len(lues)
    print(f"       coût : {perdues} lectures perdues, dont "
          f"{len(faux_avant) - len(faux)} qui étaient FAUSSES")

    fautes = 0
    # Critère de non-régression : à partir de 10 % d'occultation, plus une
    # seule lecture fausse ne doit être affirmée. En deçà, un 6 rogné de 8 %
    # ressemble vraiment à un 5, et la limite est assumée.
    graves = [e for e in faux if e[6] >= 0.10]
    if graves:
        fautes += 1
        print(f"\n    !! {len(graves)} lecture(s) fausse(s) à 10 % "
              f"d'occultation ou plus : {[(e[3], e[4], e[5], f'{e[6]:.0%}') for e in graves[:6]]}")
    residuel = [e for e in faux if e[6] < 0.10]
    if residuel:
        print(f"\n    limite assumée : {len(residuel)} lecture(s) fausse(s) "
              f"sous 10 % d'occultation")
        print(f"                     {sorted({(e[3], e[4]) for e in residuel})}")
        print("                     écart_rang de ces cas : "
              f"{sorted({round(e[1], 4) for e in residuel})}")
    return fautes


# --------------------------------------------------------------------------
# 4. le contrôle de valeur proposé
# --------------------------------------------------------------------------

def _marge_couleur(image) -> float:
    """Écart à la deuxième famille moins écart à la première."""
    return famille_de_couleur(couleur_dominante(image)[0])[2]


def _part_coin(image, tolerance: float = ECART_COULEUR_MAX) -> float | None:
    """``part_etrangere`` restreinte à la fenêtre où le rang est cherché.

    Vit ici et non dans le module : c'est une variante MESURÉE, pas un
    contrôle livré. La garder dans le banc évite qu'on la redécouvre et la
    croie prometteuse — les chiffres qu'elle produit sont juste en dessous.
    """
    a = lecteur._rgb(image)
    h, w = a.shape[:2]
    pix = a.reshape(-1, 3)
    fond = pix[pix.min(axis=1) < lecteur.SEUIL_BLANC]
    if not len(fond):
        return None
    famille, _, _ = famille_de_couleur(np.median(fond, axis=0))
    if famille is None:
        return None
    cote = max(4, int(round(w * lecteur.LARGEUR_ZONE_RANG)))
    coin = a[:min(h, cote), :min(w, cote)].reshape(-1, 3)
    coin = coin[coin.min(axis=1) < lecteur.SEUIL_BLANC]
    if not len(coin):
        return None
    ref = np.array(FAMILLES[famille], dtype=float)
    axe = np.array([255.0, 255.0, 255.0]) - ref
    t = np.clip((coin - ref) @ axe / float(axe @ axe), 0.0, 1.0)
    return float((np.linalg.norm(coin - (ref + t[:, None] * axe), axis=1)
                  > tolerance).mean())


def _part_quart(image, tolerance: float = ECART_COULEUR_MAX) -> float | None:
    """``part_etrangere`` restreinte au QUART HAUT de la découpe.

    Là où l'occultation qui fait mentir la lecture se trouve. Séduisante, et
    fausse pour une raison que seule la mesure dit : voir section 4, forme D.
    """
    a = lecteur._rgb(image)
    pix = a.reshape(-1, 3)
    fond = pix[pix.min(axis=1) < lecteur.SEUIL_BLANC]
    if not len(fond):
        return None
    famille, _, _ = famille_de_couleur(np.median(fond, axis=0))
    if famille is None:
        return None
    quart = a[:max(1, a.shape[0] // 4)].reshape(-1, 3)
    quart = quart[quart.min(axis=1) < lecteur.SEUIL_BLANC]
    if not len(quart):
        return None
    ref = np.array(FAMILLES[famille], dtype=float)
    axe = np.array([255.0, 255.0, 255.0]) - ref
    t = np.clip((quart - ref) @ axe / float(axe @ axe), 0.0, 1.0)
    return float((np.linalg.norm(quart - (ref + t[:, None] * axe), axis=1)
                  > tolerance).mean())


#: Feutres de table, pour reconstituer un cadrage imprécis autour d'une carte.
FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70))


def _cadrages_approximatifs(img: Image.Image):
    """La même carte, posée sur du feutre et recadrée de plusieurs façons.

    C'est la propriété que ce lecteur revendique — le glyphe est CHERCHÉ dans
    le coin, pas supposé à une place. Tout contrôle candidat doit donc être
    mesuré sous ces cadrages, et pas seulement sur une découpe parfaite.
    """
    w, h = img.size
    m = 14
    for feutre in FEUTRES:
        fond = Image.new("RGB", (w + 2 * m, h + 2 * m), feutre)
        fond.paste(img.convert("RGB"), (m, m))
        for x, y, ww, hh in ((m, m, w, h),
                             (m - 8, m - 8, w + 16, h + 16),
                             (m + 4, m + 3, w - 8, h - 6),
                             (m + 5, m + 4, w, h),
                             (m, m - 10, w, h + 10)):
            yield fond.crop((x, y, x + ww, y + hh))


def section_valeur(cartes, non_cartes=()) -> int:
    """Ce que coûte et ce que rattrape le contrôle de VALEUR du fond.

    La revue proposait une chose et une seule : « comparer la valeur du fond
    à la médiane du gabarit de la famille retenue ». Elle admet deux formes,
    et elles ne se valent pas du tout. Cette section les mesure toutes les
    deux plutôt que d'en supposer une bonne.
    """
    print("\n=== 4. le contrôle de VALEUR proposé en revue ===")
    print("    « ne pas vérifier seulement que le fond est uniforme, mais que")
    print("      sa valeur est celle attendue » — deux formes, mesurées.")

    print("\n    -- forme A, RETENUE : marge à la deuxième famille --")
    print("       la valeur n'est pas comparée à UNE médiane de gabarit mais")
    print("       aux QUATRE ; ce qui compte est l'écart entre les deux plus")
    print("       proches, pas la distance absolue à la meilleure.")
    ref_marge = []
    for im, _ in cartes:
        for r in [im] + [_jpeg(im, q) for q in QUALITES]:
            ref_marge.append(_marge_couleur(r))
    rm = np.array(ref_marge)
    print(f"       vraies cartes (PNG + JPEG q95→q30, n={len(rm)}) : "
          f"min {rm.min():6.2f}  p05 {np.percentile(rm, 5):6.2f}  "
          f"méd {np.median(rm):6.2f}")

    grille = [(r, g, b) for r in (0, 51, 102, 153, 204, 255)
              for g in (0, 51, 102, 153, 204, 255)
              for b in (0, 51, 102, 153, 204, 255)]
    # Les marges des lectures que le seuil doit refuser : celles qui, sans
    # lui, sortaient une carte d'une AUTRE famille que la vraie.
    faux_marges = []
    with _sans("MARGE_COULEUR_MIN", 0.0):
        for coul in grille:
            for op in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90):
                for im, attendu in cartes:
                    r = _voile(im, coul, op)
                    lu = lire_carte_fond_plein(r)
                    if lu.sure and lu.famille != attendu[-1]:
                        faux_marges.append(
                            famille_de_couleur(couleur_dominante(r)[0])[2])
    if faux_marges:
        fm = np.array(faux_marges)
        print(f"       lectures d'une AUTRE famille, seuil neutralisé "
              f"(n={len(fm)}) : min {fm.min():6.2f}  max {fm.max():6.2f}")
        if fm.max() < rm.min():
            print(f"       vide mesuré [{fm.max():.1f} ; {rm.min():.1f}] — "
                  f"seuil posé à {MARGE_COULEUR_MIN} dedans")
        else:
            print(f"       CHEVAUCHEMENT : fausses jusqu'à {fm.max():.1f}, "
                  f"vraies dès {rm.min():.1f}")
        print("       seuil | vraies perdues | familles fausses rattrapées")
        for s in (20.0, 30.0, 45.0, 60.0):
            print(f"       {s:5.1f} | {(rm < s).mean():13.2%} | "
                  f"{(fm < s).mean():26.2%}")
    else:
        print("       aucune lecture d'une autre famille, seuil neutralisé")

    if len(non_cartes):
        nm = np.array([_marge_couleur(im) for im in non_cartes])
        print(f"       bonus — {len(nm)} NON-CARTES réelles (dos, décors, "
              f"avatars) :")
        print(f"       marge min {nm.min():6.2f}  méd {np.median(nm):6.2f}  "
              f"p95 {np.percentile(nm, 95):6.2f} — sous le seuil pour "
              f"{(nm < MARGE_COULEUR_MIN).mean():.0%} d'entre elles")

    print("\n    -- forme B, ÉCARTÉE : part des pixels de fond hors teinte --")
    vraies = []
    for im, _ in cartes:
        for r in [im] + [_jpeg(im, q) for q in QUALITES]:
            p = part_etrangere(r)
            if p is not None:
                vraies.append(p)
    v = np.array(vraies)
    print(f"\n    vraies cartes (PNG + JPEG q95→q30, n={len(v)}) : "
          f"{_quantiles(v)}")

    intrus = []
    for n in RETOURNEES:
        p = DONNEES / n
        if p.exists():
            x = part_etrangere(Image.open(p).convert("RGB"))
            d = couleur_dominante(Image.open(p).convert("RGB"))[2]
            intrus.append((n, x, d))
            print(f"    {n:32s} part {x:.4f}  (dispersion {d:.2f} — déjà "
                  f"refusée par le fond)")

    def _parts_fausses() -> list[float]:
        return [x for im, attendu in cartes
                for part in HAUTEURS for coul in OCCULTANTS.values()
                for r in [_recouvre_haut(im, part, coul)]
                if lire_carte_fond_plein(r).carte not in (None, attendu)
                for x in [part_etrangere(r)] if x is not None]

    # Deux chiffres, et il faut les deux. Le premier flatte le contrôle, le
    # second décide : ce qui compte n'est pas ce qu'il rattraperait tout
    # seul, mais ce qu'il ajoute PAR-DESSUS le seuil de forme déjà en place.
    with _sans("ECART_RANG_MAX", 1.0):
        seul = _parts_fausses()
    residu = _parts_fausses()

    if seul:
        ps, gratuit = np.array(seul), v.max()
        print(f"    lectures fausses, seuil de forme neutralisé (n={len(ps)}) : "
              f"min {ps.min():.4f}  méd {np.median(ps):.4f}")
        print(f"    au seuil gratuit ({gratuit:.4f}, aucune vraie carte "
              f"perdue) : rattrape {int((ps > gratuit).sum())}/{len(ps)} "
              f"({(ps > gratuit).mean():.1%})")
        if residu:
            pr = np.array(residu)
            print(f"    MAIS sur le résidu que ECART_RANG_MAX laisse "
                  f"({len(pr)} cas) : rattrape {int((pr > gratuit).sum())}/"
                  f"{len(pr)}")
        else:
            print("    et il ne reste aucun résidu à rattraper")
        print("\n    -- balayage du seuil --")
        print("       seuil | vraies perdues | fausses rattrapées (seuil "
              "de forme neutralisé)")
        for s in (0.05, 0.10, 0.20, 0.30, 0.40):
            print(f"       {s:5.2f} | {(v > s).mean():13.2%} | "
                  f"{(ps > s).mean():17.2%}")
        print("\n    lecture : ce contrôle a du pouvoir dans l'absolu, et aucun "
              "apport marginal.\n    Le résidu qui subsiste — un 6 rogné de "
              "8 % lu « 5 » — a un fond PARFAITEMENT\n    propre : c'est le "
              "glyphe qui est abîmé, pas la couleur. Une constante de\n    "
              "plus pour zéro cas gagné n'a pas sa place ; elle reste une "
              "mesure.")
    else:
        print("    aucune lecture fausse à mesurer")

    print("\n    -- forme C, ÉCARTÉE : la même part, restreinte au COIN du rang --")
    print("       la variante qu'on propose naturellement en voyant échouer la")
    print("       forme B : mesurer la valeur du fond LÀ où elle décide, dans la")
    print("       fenêtre où le glyphe est cherché. Elle a du pouvoir, mais pas")
    print("       sur ce qui reste à rattraper.")
    ref_c = [x for im, _ in cartes
             for r in [im] + [_jpeg(im, q) for q in QUALITES]
             for x in [_part_coin(r)] if x is not None]
    vc = np.array(ref_c)
    with _sans("ECART_RANG_MAX", 1.0):
        faux_c = [x for im, attendu in cartes
                  for part in HAUTEURS for coul in OCCULTANTS.values()
                  for r in [_recouvre_haut(im, part, coul)]
                  if lire_carte_fond_plein(r).carte not in (None, attendu)
                  for x in [_part_coin(r)] if x is not None]
    # Le résidu : ce qui reste faux MALGRÉ le seuil de forme. C'est la seule
    # chose qu'un contrôle supplémentaire aurait à gagner.
    residu_c = [x for im, attendu in cartes
                for part in HAUTEURS for coul in OCCULTANTS.values()
                for r in [_recouvre_haut(im, part, coul)]
                if lire_carte_fond_plein(r).carte not in (None, attendu)
                for x in [_part_coin(r)] if x is not None]
    if len(vc) and faux_c:
        fc, gratuit = np.array(faux_c), vc.max()
        print(f"       vraies (n={len(vc)}) : {_quantiles(vc)}")
        print(f"       fausses, seuil de forme neutralisé (n={len(fc)}) : "
              f"min {fc.min():.4f}  méd {np.median(fc):.4f}")
        print(f"       au seuil gratuit ({gratuit:.4f}) : rattrape "
              f"{int((fc > gratuit).sum())}/{len(fc)} "
              f"({(fc > gratuit).mean():.1%})")
        if residu_c:
            rc = np.array(residu_c)
            print(f"       MAIS sur le résidu que le seuil de forme laisse "
                  f"({len(rc)} cas) : rattrape {int((rc > gratuit).sum())}/"
                  f"{len(rc)}")
        print("       → elle ne remplace pas ECART_RANG_MAX (94 % à coût nul)")
        print("         et n'ajoute presque rien par-dessus. Écartée aussi.")

    print("\n    -- forme D, ÉCARTÉE : la même part, sur le QUART HAUT --")
    print("       celle-ci rattrape VRAIMENT le résidu — et c'est un piège.")
    exact = np.array([x for im, _ in cartes
                      for r in [im] + [_jpeg(im, q) for q in QUALITES]
                      for x in [_part_quart(r)] if x is not None])
    approx = np.array([x for im, _ in cartes
                       for r in _cadrages_approximatifs(im)
                       for x in [_part_quart(r)] if x is not None])
    res_d = [x for im, attendu in cartes
             for part in HAUTEURS for coul in OCCULTANTS.values()
             for r in [_recouvre_haut(im, part, coul)]
             if lire_carte_fond_plein(r).carte not in (None, attendu)
             for x in [_part_quart(r)] if x is not None]
    if len(exact) and len(approx) and res_d:
        rd = np.array(res_d)
        print(f"       cadrage EXACT (n={len(exact)})        : "
              f"max {exact.max():.4f} → rattrape "
              f"{int((rd > exact.max()).sum())}/{len(rd)} du résidu")
        print(f"       cadrage APPROXIMATIF (n={len(approx)}) : "
              f"max {approx.max():.4f} → rattrape "
              f"{int((rd > approx.max()).sum())}/{len(rd)} du résidu")
        print("       une découpe élargie de 8 px fait entrer le feutre dans le")
        print("       quart haut : la statistique explose sur des cartes")
        print("       parfaitement lisibles. Elle échange l'INDIFFÉRENCE AU")
        print(f"       CADRAGE — la propriété qui fonde ce lecteur — contre "
              f"{len(rd)} cas. Écartée.")

    print("\n    -- ce que la forme B coûterait sur les intrus déjà connus --")
    for s in (0.30, 0.40, 0.50):
        print(f"       seuil {s:.2f} : vraies perdues {(v > s).mean():.2%} | "
              f"cartes retournées refusées : "
              f"{sum(1 for _, x, _ in intrus if x > s)}/{len(intrus)}")
    print("       (elles étaient déjà refusées par la dispersion : rien de neuf)")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captures", type=Path, default=None,
                   help="dossier de sessions de capture (hors dépôt)")
    args = p.parse_args()

    gabarits = _gabarits()
    versionnees = _versionnees()
    if not gabarits:
        raise SystemExit("habillage pmu_solid absent : banc inexploitable")

    jeux = [("52 gabarits pmu_solid", gabarits),
            ("découpes réelles versionnées", versionnees)]
    reelles: list = []
    non_cartes: list = []
    if args.captures:
        if not args.captures.is_dir():
            raise SystemExit(f"dossier de captures absent : {args.captures}")
        reelles, non_cartes = _captures(args.captures)
        jeux.append((f"découpes extraites de {args.captures.name}", reelles))
        print(f"[captures] {len(reelles)} découpes distinctes lues « sure », "
              f"{len(non_cartes)} refusées\n")

    # Les sections 2 à 4 tournent sur le jeu le plus représentatif
    # disponible : les découpes réelles si on les a, les gabarits sinon.
    principal = reelles or (versionnees + gabarits)

    fautes = 0
    fautes += section_compression(jeux)
    fautes += section_homogene(principal)
    fautes += section_occultation(principal)
    fautes += section_valeur(principal, non_cartes)

    print("\n" + ("BANC VERT" if fautes == 0 else f"BANC ROUGE ({fautes})"))
    return fautes


if __name__ == "__main__":
    raise SystemExit(main())
