"""Tests de la lecture des montants (OCR à gabarits).

**Garantie visée : ne jamais annoncer un montant faux.** Un pot mal lu fausse
silencieusement tout le conseil qui suit, alors qu'un refus demande juste de
recadrer. Les tests verrouillent, dans l'ordre d'importance :

1. aucun montant faux — sur le banc synthétique, sur du bruit, sur des aplats ;
2. aucun montant INVENTÉ quand la bande porte deux montants : c'était le trou
   du module (« 10 1 » lu 101,0 à confiance 1,0, « 1 2 » lu 12,0 sur 42 des
   42 couples police × corps). Verrouillé sur les cas nommés et sur un banc ;
3. le garde-fou de hauteur : sous 10 px de chiffre, on refuse au lieu de
   deviner (sans lui : 2,96 % de montants faux mesurés sur 540 bandes de
   9 à 13 px) ;
4. les taux de lecture publiés dans la docstring du module — 99,5 % dans le
   domaine, 84,9 % hors calibration, 91,0 % pour `lire_texte` — pour qu'une
   régression qui les ferait chuter ne passe pas inaperçue ;
5. la frontière du « 0,00 % de faux » : elle tient jusqu'à 0,7 px de flou et
   casse à 1,3 px ;
6. la virgule décimale, le séparateur de milliers, et les libellés/unités qui
   entourent le nombre.

Les bancs sont reconstruits ici même (`_banc`, `_banc_paires`,
`_images_non_textuelles`) : les chiffres publiés dans la docstring du module
sortent de ces mêmes cas, pas d'une mesure d'à-côté.
"""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from pfs.vision.digit_ocr import (
    ALPHABET,
    CORPS,
    HAUTEUR_MIN,
    POLICES,
    Gabarits,
    _chemin_police,
    _valeur,
    charger_gabarits,
    construire_gabarits,
    lire_ligne,
    lire_montant,
    lire_texte,
)

FEUTRE = (20, 83, 45)
BLANC = (255, 255, 255)

# Fonds variés : le module s'appuie sur l'ÉCART au fond, jamais sur la
# luminosité — il doit lire aussi bien du clair sur sombre que l'inverse.
FONDS = [
    ((20, 83, 45), (255, 255, 255)),
    ((16, 24, 56), (232, 236, 245)),
    ((12, 12, 14), (245, 214, 96)),
    ((203, 207, 214), (24, 24, 28)),
    ((92, 18, 30), (255, 255, 255)),
    ((44, 46, 52), (190, 195, 205)),
]

MONTANTS = [
    ("1,50", 1.5), ("10", 10.0), ("9,50", 9.5), ("1", 1.0), ("2", 2.0),
    ("2,50", 2.5), ("0,50", 0.5), ("3", 3.0), ("4,50", 4.5), ("12,50", 12.5),
    ("25", 25.0), ("100", 100.0), ("7,25", 7.25), ("0,25", 0.25),
    ("48", 48.0), ("6,75", 6.75), ("20", 20.0), ("1,25", 1.25),
]
FORMATS = ["Pot: {} BB", "{} BB", "CALL {} BB", "{}", "{} EUR", "RAISE {} BB"]
FORMATS_PAIRE = ["{} {}", "{} {} BB", "Pot: {} {} BB"]
CORPS_DOMAINE = (14, 16, 18, 20, 24, 28)
INCONNUES = ("trebuc.ttf", "calibri.ttf", "consola.ttf", "framd.ttf")

# Les onze textes qui fabriquaient un montant absent de l'écran avant la
# correction du recollage : 148 des 462 rendus (11 × 6 polices × 7 corps)
# donnaient une valeur, dont « 1 2 » → 12,0 sur les 42 rendus.
TEXTES_A_DEUX_MONTANTS = ["10 1", "10 2", "9,50 4,50", "1,50 48", "1 2",
                          "2 3", "0,50 1 BB", "20 1", "100 20", "1,25 2",
                          "12,50 48"]


def _rendre(texte: str, police: str = "arial.ttf", corps: int = 20,
            fond=FEUTRE, couleur=BLANC, flou: float = 0.0,
            bruit: float = 0.0, graine: int = 0) -> Image.Image:
    """Rend une bande de texte comme un client l'afficherait."""
    f = ImageFont.truetype(str(_chemin_police(police)), corps)
    boite = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), texte, font=f)
    marge = max(4, corps // 2)
    im = Image.new("RGB", (boite[2] - boite[0] + 2 * marge,
                           boite[3] - boite[1] + 2 * marge), fond)
    ImageDraw.Draw(im).text((marge - boite[0], marge - boite[1]), texte,
                            font=f, fill=couleur)
    if flou:
        im = im.filter(ImageFilter.GaussianBlur(flou))
    if bruit:
        a = np.asarray(im).astype(np.int16)
        a = np.clip(a + np.random.default_rng(graine).normal(0, bruit, a.shape),
                    0, 255)
        im = Image.fromarray(a.astype(np.uint8))
    return im


def _banc(polices=POLICES, corps=CORPS_DOMAINE, graine=1):
    """Le banc synthétique : (texte, valeur attendue, image)."""
    rng = np.random.default_rng(graine)
    for fmt in FORMATS:
        for chaine, valeur in MONTANTS:
            texte = fmt.format(chaine)
            for c in corps:
                fond, couleur = FONDS[int(rng.integers(len(FONDS)))]
                yield texte, valeur, _rendre(
                    texte, polices[int(rng.integers(len(polices)))], c,
                    fond, couleur,
                    float(rng.choice([0.0, 0.0, 0.4, 0.7])),
                    float(rng.choice([0.0, 0.0, 3.0, 6.0])),
                    int(rng.integers(1 << 30)))


def _banc_paires(polices=POLICES, corps=CORPS_DOMAINE, graine=13):
    """324 bandes portant DEUX montants : (texte, valeurs affichées, image)."""
    rng = np.random.default_rng(graine)
    for fmt in FORMATS_PAIRE:
        for i, (a, va) in enumerate(MONTANTS):
            b, vb = MONTANTS[(i + 5) % len(MONTANTS)]
            texte = fmt.format(a, b)
            for c in corps:
                fond, couleur = FONDS[int(rng.integers(len(FONDS)))]
                yield texte, {va, vb}, _rendre(
                    texte, polices[int(rng.integers(len(polices)))], c,
                    fond, couleur,
                    float(rng.choice([0.0, 0.0, 0.4, 0.7])),
                    float(rng.choice([0.0, 0.0, 3.0, 6.0])),
                    int(rng.integers(1 << 30)))


def _images_non_textuelles(n: int = 400, graine: int = 2026):
    """Bruit, dégradés, formes pleines et barres — jamais du texte.

    Quatre familles alternées : c'est le banc qui chiffre la limite « ce
    module ne DÉTECTE pas le texte » et qui justifie MARGE_MIN/DISTANCE_MAX.
    """
    rng = np.random.default_rng(graine)
    for i in range(n):
        h, w = int(rng.integers(24, 48)), int(rng.integers(80, 220))
        genre = i % 4
        if genre == 0:
            yield Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
            continue
        if genre == 1:
            a = np.linspace(0, 255, w, dtype=np.uint8)
            if rng.random() < 0.5:
                a = a[::-1]
            im = np.repeat(np.repeat(a[None, :], h, axis=0)[..., None], 3, axis=2)
            teinte = rng.integers(0, 256, 3)
            yield Image.fromarray((im * (teinte / 255.0)).astype(np.uint8))
            continue
        fond = tuple(int(v) for v in rng.integers(0, 120, 3))
        encre = tuple(int(v) for v in rng.integers(140, 256, 3))
        im = Image.new("RGB", (w, h), fond)
        d = ImageDraw.Draw(im)
        if genre == 2:
            for _ in range(int(rng.integers(1, 6))):
                x = int(rng.integers(0, w - 12))
                y = int(rng.integers(0, max(1, h - 12)))
                dx = int(rng.integers(8, 26))
                dy = int(rng.integers(8, max(9, h - y)))
                forme = int(rng.integers(0, 3))
                if forme == 0:
                    d.ellipse([x, y, x + dx, y + dy], fill=encre)
                elif forme == 1:
                    d.rectangle([x, y, x + dx, y + dy], fill=encre)
                else:
                    d.polygon([(x, y + dy), (x + dx // 2, y), (x + dx, y + dy)],
                              fill=encre)
        else:
            for k in range(int(rng.integers(2, 9))):
                x = 4 + k * int(rng.integers(6, 16))
                if x + 4 >= w:
                    break
                d.rectangle([x, int(rng.integers(2, h // 2)), x + 4, h - 3],
                            fill=encre)
        yield im


class TestGabarits(unittest.TestCase):
    def test_banque_par_defaut_couvre_l_alphabet(self) -> None:
        g = charger_gabarits()
        self.assertEqual(set(g.caracteres), set(ALPHABET))
        self.assertEqual(len(g), g.vecteurs.shape[0])
        self.assertEqual(g.geometrie.shape, (len(g), 3))

    def test_vecteurs_centres_et_normes(self) -> None:
        # La corrélation croisée se réduit à un produit scalaire : c'est vrai
        # seulement si chaque vecteur est centré et de norme 1. Seule exception
        # admise : les glyphes qui remplissent tout leur cadre (« I », « l »)
        # n'ont aucune forme propre et donnent un vecteur nul.
        g = charger_gabarits()
        self.assertTrue(np.allclose(g.vecteurs.mean(axis=1), 0.0, atol=1e-12))
        normes = np.linalg.norm(g.vecteurs, axis=1)
        nulles = normes < 1e-9
        self.assertTrue(np.allclose(normes[~nulles], 1.0))
        self.assertLessEqual(set(g.caracteres[nulles]), {"I", "l"})

    def test_banque_memoisee(self) -> None:
        self.assertIs(charger_gabarits(), charger_gabarits())

    def test_construction_sur_une_seule_police(self) -> None:
        g = construire_gabarits(polices=("arial.ttf",), corps=(20, 32))
        self.assertIsInstance(g, Gabarits)
        self.assertEqual(set(g.caracteres), set(ALPHABET))
        self.assertEqual(len(g), 2 * len(ALPHABET))

    def test_police_absente_refusee_clairement(self) -> None:
        with self.assertRaises(FileNotFoundError):
            construire_gabarits(polices=("police-qui-n-existe-pas.ttf",))

    def test_chiffres_marques_numeriques(self) -> None:
        g = charger_gabarits()
        for c, num in zip(g.caracteres, g.numerique):
            self.assertEqual(bool(num), c in "0123456789.,", c)


class TestMontantsTypes(unittest.TestCase):
    """Les libellés exacts qu'affiche une table."""

    def test_libelles_de_table(self) -> None:
        attendu = {
            "Pot: 1,50 BB": 1.5,
            "10 BB": 10.0,
            "9,50 BB": 9.5,
            "CALL 1 BB": 1.0,
            "2 BB": 2.0,
            "RAISE 4,50 BB": 4.5,
            "Pot: 12,50 BB": 12.5,
            "0,25 EUR": 0.25,
            "100": 100.0,
        }
        for texte, valeur in attendu.items():
            for corps in (16, 22, 30):
                lu = lire_montant(_rendre(texte, corps=corps))
                self.assertEqual(lu, valeur, f"{texte!r} à {corps} px → {lu}")

    def test_virgule_decimale_et_point_donnent_la_meme_valeur(self) -> None:
        # Un client francophone écrit « 1,50 », un autre « 1.50 » : même montant.
        self.assertEqual(lire_montant(_rendre("Pot: 1,50 BB", corps=24)), 1.5)
        self.assertEqual(lire_montant(_rendre("Pot: 1.50 BB", corps=24)), 1.5)

    def test_separateur_de_milliers(self) -> None:
        self.assertEqual(lire_montant(_rendre("12.500 BB", corps=26)), 12500.0)
        self.assertEqual(lire_montant(_rendre("1.250,75 EUR", corps=26)), 1250.75)

    def test_le_libelle_et_l_unite_ne_polluent_pas_le_nombre(self) -> None:
        # « BB » ne doit pas devenir « 88 », « Pot » ne doit pas devenir « P0t ».
        for texte, valeur in (("Pot: 8 BB", 8.0), ("CALL 88 BB", 88.0),
                              ("Pot: 0,50 BB", 0.5)):
            self.assertEqual(lire_montant(_rendre(texte, corps=24)), valeur, texte)

    def test_lire_texte_rend_le_libelle(self) -> None:
        for texte in ("Pot: 1,50 BB", "CALL 1 BB", "9,50 BB"):
            self.assertEqual(lire_texte(_rendre(texte, corps=26)), texte)

    def test_lecture_sur_tous_les_fonds(self) -> None:
        for fond, couleur in FONDS:
            im = _rendre("Pot: 9,50 BB", corps=22, fond=fond, couleur=couleur)
            self.assertEqual(lire_montant(im), 9.5, f"fond {fond}")

    def test_diagnostic_complet(self) -> None:
        r = lire_ligne(_rendre("Pot: 2,50 BB", corps=24))
        self.assertTrue(r.acceptee)
        self.assertEqual(r.valeur, 2.5)
        self.assertEqual(r.texte, "Pot: 2,50 BB")
        self.assertIsNone(r.refus)
        self.assertGreaterEqual(r.hauteur, HAUTEUR_MIN)
        self.assertGreater(r.confiance, 0.0)


class TestBanc(unittest.TestCase):
    """Les taux chiffrés publiés en docstring, verrouillés."""

    def test_taux_de_lecture_dans_le_domaine(self) -> None:
        # Mesuré : 99,5 % (99,4 à 99,8 % sur les graines 1, 2, 5, 7, 11, 23),
        # 0,00 % de faux à chaque fois.
        exact = faux = total = 0
        erreurs = []
        for texte, valeur, im in _banc():
            total += 1
            lu = lire_montant(im)
            if lu is None:
                continue
            if abs(lu - valeur) < 1e-9:
                exact += 1
            else:
                faux += 1
                erreurs.append((texte, lu))
        self.assertGreaterEqual(exact, 0.99 * total,
                                f"{exact}/{total} montants lus exactement")
        self.assertEqual(faux, 0, f"{faux}/{total} montants FAUX : {erreurs[:6]}")

    def test_texte_exact_dans_le_domaine(self) -> None:
        # Mesuré : 91,0 % (590/648). Le seuil reste sous la mesure — `lire_texte`
        # est une aide au diagnostic, la garantie porte sur `lire_montant`.
        exact = total = 0
        for texte, _, im in _banc(graine=7):
            total += 1
            exact += lire_texte(im) == texte
        self.assertGreaterEqual(exact, 0.88 * total, f"{exact}/{total} chaînes")


class TestDeuxMontants(unittest.TestCase):
    """Le défaut corrigé : une bande à deux montants n'en fabrique pas un tiers.

    Avant correction, le recollage des mots numériques voisins soudait les deux
    nombres : « 10 1 » → 101,0 à confiance 1,0, « 1 2 » → 12,0 sur les
    42 couples police × corps, « 9,50 4,50 » → 9504,5.
    """

    def test_cas_qui_fabriquaient_un_montant(self) -> None:
        # 148/462 rendaient une valeur avant correction, 0/462 après.
        rendus = []
        for texte in TEXTES_A_DEUX_MONTANTS:
            for police in POLICES:
                for corps in (14, 16, 18, 20, 24, 28, 32):
                    r = lire_ligne(_rendre(texte, police=police, corps=corps))
                    if r.valeur is not None:
                        rendus.append((texte, police, corps, r.valeur, r.confiance))
        self.assertEqual(rendus, [], f"valeurs rendues : {rendus[:8]}")

    def test_banc_de_paires_ne_fabrique_aucun_montant(self) -> None:
        # 324 bandes, polices de la banque. Mesuré : 0 valeur rendue ici, et sur
        # 2 268 bandes (7 graines) 2 valeurs rendues (0,09 %), aucune inventée.
        rendus, inventes, total = [], [], 0
        for texte, affichees, im in _banc_paires():
            total += 1
            r = lire_ligne(im)
            if r.valeur is None:
                continue
            rendus.append((texte, r.valeur))
            if r.valeur not in affichees:
                inventes.append((texte, r.valeur, r.confiance))
        self.assertEqual(inventes, [], f"montants INVENTÉS : {inventes[:6]}")
        self.assertLessEqual(len(rendus), 0.01 * total,
                             f"{len(rendus)}/{total} rendent une valeur")

    def test_banc_de_paires_hors_banque_ne_fabrique_aucun_montant(self) -> None:
        # Hors calibration, une bande à deux montants en rend un plus souvent
        # (mesuré 11/324 ici, 3,92 % sur 2 268 bandes) mais jamais un montant
        # absent de l'écran : c'est l'autre nombre qui a été mal segmenté.
        rendus, inventes, total = [], [], 0
        for texte, affichees, im in _banc_paires(polices=INCONNUES, graine=17):
            total += 1
            r = lire_ligne(im)
            if r.valeur is None:
                continue
            rendus.append((texte, r.valeur, r.confiance))
            if r.valeur not in affichees:
                inventes.append((texte, r.valeur, r.confiance))
        self.assertEqual(inventes, [], f"montants INVENTÉS : {inventes[:6]}")
        self.assertLessEqual(len(rendus), 0.08 * total,
                             f"{len(rendus)}/{total} rendent une valeur")
        # La limite documentée dit que `confiance` ne trie pas ce reste : la
        # vérifier plutôt que de laisser croire qu'un seuil suffirait.
        self.assertTrue(any(c > 0.9 for _, _, c in rendus),
                        f"confiances observées : {[c for _, _, c in rendus]}")

    def test_deux_nombres_bien_separes_sont_ambigus(self) -> None:
        for texte in ("0,50 1 BB", "10 1", "9,50 4,50", "1 2"):
            r = lire_ligne(_rendre(texte, corps=26))
            self.assertIsNone(r.valeur, f"{texte!r} → {r.valeur}")
            self.assertIn("nombres", r.refus or "", texte)

    def test_un_faux_espace_dans_un_nombre_est_recolle(self) -> None:
        # La contrepartie : quand la jonction porte un séparateur, c'est un
        # nombre fendu et non deux montants — « 1, » + « 50 » vaut 1,50.
        for police in ("arial.ttf", "consola.ttf", "calibri.ttf"):
            for corps in (18, 24, 28):
                lu = lire_montant(_rendre("1,50 BB", police=police, corps=corps))
                self.assertEqual(lu, 1.5, f"{police} à {corps} px → {lu}")


class TestGardeFouHauteur(unittest.TestCase):
    """Sous 10 px de chiffre, refuser plutôt que deviner."""

    def test_texte_trop_petit_toujours_refuse(self) -> None:
        declenche = 0
        for corps in (7, 8, 9, 10, 11):
            for texte in ("Pot: 4,50 BB", "12,50", "100 BB"):
                r = lire_ligne(_rendre(texte, corps=corps))
                if r.hauteur >= HAUTEUR_MIN:
                    continue
                declenche += 1
                self.assertIsNone(r.valeur, f"{texte!r} à {corps} px")
                self.assertIn("px", r.refus or "")
        # Sans cette borne, le test passerait en ne vérifiant rien du tout.
        self.assertGreaterEqual(declenche, 6, f"{declenche} cas sous le seuil")

    def test_aucun_montant_faux_sur_du_texte_minuscule(self) -> None:
        faux = []
        for texte, valeur, im in _banc(corps=(9, 10, 11, 12, 13), graine=11):
            lu = lire_montant(im)
            if lu is not None and abs(lu - valeur) > 1e-9:
                faux.append((texte, lu))
        self.assertEqual(faux, [], f"montants FAUX sous le seuil : {faux[:6]}")


class TestRefus(unittest.TestCase):
    """Un refus est toujours préférable à un nombre faux."""

    def test_image_vide_refusee(self) -> None:
        for couleur in ((255, 255, 255), FEUTRE, (0, 0, 0)):
            a = np.full((40, 160, 3), couleur, dtype=np.uint8)
            self.assertIsNone(lire_montant(a), f"aplat {couleur}")
            self.assertEqual(lire_texte(a), "")

    def test_bruit_pur_refuse(self) -> None:
        rng = np.random.default_rng(20260809)
        for _ in range(40):
            a = rng.integers(0, 256, (40, 180, 3), dtype=np.uint8)
            self.assertIsNone(lire_montant(a))

    def test_degrade_refuse(self) -> None:
        g = np.linspace(0, 255, 200, dtype=np.uint8)
        a = np.repeat(np.repeat(g[None, :], 40, axis=0)[..., None], 3, axis=2)
        self.assertIsNone(lire_montant(a))

    def test_ligne_sans_chiffre_refusee(self) -> None:
        r = lire_ligne(_rendre("FOLD", corps=26))
        self.assertIsNone(r.valeur)
        self.assertEqual(r.refus, "aucun nombre")

    def test_lettre_collee_au_nombre_fait_refuser(self) -> None:
        # Un « 48 » dont le 8 serait lu « B » ne doit pas rendre 4 : le mot
        # partiellement numérique est rejeté en bloc, jamais tronqué.
        r = lire_ligne(_rendre("4B", corps=26))
        self.assertIsNone(r.valeur)

    def test_decoupage_douteux(self) -> None:
        # Un mot NON numérique qui commence par un séparateur trahit une
        # segmentation ratée. Cas exact tiré d'un banc volontairement dégradé :
        # « 2,50 BB » se segmente en « 2 » + « ,filJ » + « B8 », et sans ce
        # garde-fou le module rendrait 2,0 au lieu de 2,50.
        im = _rendre("2,50 BB", police="consola.ttf", corps=19,
                     fond=(12, 12, 14), couleur=(245, 214, 96),
                     flou=0.978, bruit=6.828, graine=786061504)
        r = lire_ligne(im)
        self.assertIsNone(r.valeur)
        self.assertEqual(r.refus, "découpage douteux")


class TestNonTextuel(unittest.TestCase):
    """Ce module lit une bande de texte, il ne DÉTECTE pas le texte."""

    def test_taux_d_acceptation_des_images_non_textuelles(self) -> None:
        # Mesuré : 25/400 (6,2 %). Le test borne le taux dans les deux sens —
        # au-dessus c'est une régression, très en dessous c'est que le banc
        # ne mord plus et ne justifie plus MARGE_MIN/DISTANCE_MAX.
        acceptees = sum(lire_montant(im) is not None
                        for im in _images_non_textuelles())
        self.assertLessEqual(acceptees, 40, f"{acceptees}/400 acceptées")
        self.assertGreaterEqual(acceptees, 10, f"{acceptees}/400 acceptées")

    def test_bruit_et_barres_integralement_refuses(self) -> None:
        # Les deux familles où le refus est total : 0/100 et 0/100.
        for genre, nom in ((0, "bruit"), (3, "barres")):
            acceptees = [lire_montant(im)
                         for i, im in enumerate(_images_non_textuelles())
                         if i % 4 == genre]
            restants = [v for v in acceptees if v is not None]
            self.assertEqual(restants, [], f"{nom} : {restants[:6]}")


class TestInterpretation(unittest.TestCase):
    """Le passage de la chaîne de chiffres à la valeur."""

    def test_decimales(self) -> None:
        self.assertEqual(_valeur("1,50"), 1.5)
        self.assertEqual(_valeur("1.50"), 1.5)
        self.assertEqual(_valeur("0,25"), 0.25)
        self.assertEqual(_valeur("12"), 12.0)

    def test_milliers(self) -> None:
        self.assertEqual(_valeur("1.250"), 1250.0)
        self.assertEqual(_valeur("12.500"), 12500.0)
        self.assertEqual(_valeur("1.250,75"), 1250.75)
        self.assertEqual(_valeur("1,250.75"), 1250.75)

    def test_formes_refusees(self) -> None:
        # Un zéro de tête devant un autre chiffre trahit une lecture ratée
        # (« 000 »), pas un montant ; idem pour un groupe de milliers bancal.
        for mauvais in ("", ",50", "1,", "007", "12.34.56", "1..5", "12345.678"):
            self.assertIsNone(_valeur(mauvais), mauvais)

    def test_milliers_et_decimale_doivent_differer(self) -> None:
        # « 6,759,50 » est la soudure de « 6,75 » et « 9,50 », pas un montant :
        # aucun format n'écrit les milliers et la décimale du même caractère.
        for mauvais in ("6,759,50", "1,250,75", "1.250.75"):
            self.assertIsNone(_valeur(mauvais), mauvais)
        self.assertEqual(_valeur("1.250.750"), 1250750.0)

    def test_decimales_multiples_acceptees(self) -> None:
        # Quatre décimales ne sont pas ambiguës : seul un groupe de TROIS
        # chiffres après le dernier séparateur peut être des milliers.
        self.assertEqual(_valeur("1.2500"), 1.25)


class TestPolicesInconnues(unittest.TestCase):
    """Limite documentée : la banque est calibrée sur SES polices."""

    def test_taux_hors_calibration(self) -> None:
        # Mesuré : 84,9 % lus, 0,00 % faux (84,3 à 85,5 % sur six graines). Le
        # taux est verrouillé, pas seulement l'absence de faux : une régression
        # qui le ferait tomber à 30 % passerait sinon inaperçue.
        exact = faux = total = 0
        for _, valeur, im in _banc(polices=INCONNUES, graine=3):
            total += 1
            lu = lire_montant(im)
            if lu is None:
                continue
            if abs(lu - valeur) < 1e-9:
                exact += 1
            else:
                faux += 1
        self.assertEqual(faux, 0, f"{faux}/{total} FAUX hors banque")
        self.assertGreaterEqual(exact, 0.82 * total,
                                f"{exact}/{total} lus hors banque")

    def test_le_manque_vient_du_corps_14(self) -> None:
        # Le trou hors calibration n'est pas réparti : à 14 px les chiffres
        # touchent le garde-fou de hauteur (17,6 % lus), alors que de 16 à
        # 28 px ces mêmes polices donnent 99,2 %.
        exact = total = 0
        for _, valeur, im in _banc(polices=INCONNUES,
                                   corps=(16, 18, 20, 22, 24, 26, 28), graine=3):
            total += 1
            lu = lire_montant(im)
            exact += lu is not None and abs(lu - valeur) < 1e-9
        self.assertGreaterEqual(exact, 0.97 * total, f"{exact}/{total} en 16-28")

    def test_ajouter_la_police_du_client_ameliore(self) -> None:
        # Le remède documenté : AJOUTER la fonte du client à la banque.
        # Mesuré sur Calibri, corps 16-28 : 99,2 % → 100,0 %.
        enrichie = construire_gabarits(polices=(*POLICES, "calibri.ttf"),
                                       corps=CORPS)
        avant = apres = total = 0
        for _, valeur, im in _banc(polices=("calibri.ttf",),
                                   corps=(16, 20, 26), graine=9):
            total += 1
            a, b = lire_montant(im), lire_montant(im, enrichie)
            avant += a is not None and abs(a - valeur) < 1e-9
            apres += b is not None and abs(b - valeur) < 1e-9
        self.assertGreaterEqual(apres, avant, f"{avant} → {apres} sur {total}")
        self.assertGreaterEqual(apres, 0.99 * total, f"{apres}/{total}")

    def test_chasse_fixe_lue_exactement(self) -> None:
        # Consolas était l'échec documenté du module (« 1,50 » segmenté en
        # « 1, » + « 50 »). La correction de chasse le résout : les écarts
        # interlettres d'une fonte à chasse fixe redescendent sous le seuil
        # d'espace une fois corrigés du déficit des glyphes étroits.
        for texte, valeur in (("Pot: 1,50 BB", 1.5), ("9,50 BB", 9.5),
                              ("CALL 1 BB", 1.0), ("12,50", 12.5),
                              ("0,25 EUR", 0.25)):
            for corps in (16, 20, 24, 28):
                im = _rendre(texte, police="consola.ttf", corps=corps)
                self.assertEqual(lire_montant(im), valeur,
                                 f"{texte!r} à {corps} px")


class TestRobustesse(unittest.TestCase):
    def test_flou_et_bruit_ne_produisent_jamais_de_faux(self) -> None:
        for flou in (0.0, 0.4, 0.7):
            for bruit in (0.0, 6.0, 12.0):
                im = _rendre("Pot: 9,50 BB", corps=24, flou=flou,
                             bruit=bruit, graine=5)
                lu = lire_montant(im)
                self.assertIn(lu, (9.5, None),
                              f"flou {flou} bruit {bruit} → {lu}")

    def test_frontiere_du_flou(self) -> None:
        # La garantie « 0,00 % de faux » a une frontière mesurée : elle tient
        # jusqu'à 0,7 px de flou (0/108 ici, 0,00 % sur 324 bandes par point du
        # balayage complet) et casse au-delà — 4,63 % de faux à 1,3 px. Les
        # deux côtés sont vérifiés : sans le second, la limite publiée pourrait
        # être une prudence gratuite.
        def faux(flou):
            n = 0
            for chaine, valeur in MONTANTS:
                for fmt in ("Pot: {} BB", "{} BB", "{}"):
                    im = _rendre(fmt.format(chaine), corps=20, flou=flou,
                                 bruit=3.0, graine=77)
                    lu = lire_montant(im)
                    n += lu is not None and abs(lu - valeur) > 1e-9
            return n
        self.assertEqual(faux(0.7), 0, "des faux apparaissent dès 0,7 px")
        self.assertGreater(faux(1.4), 0, "aucun faux à 1,4 px : limite trop molle")

    def test_marge_de_cadrage_genereuse(self) -> None:
        # Personne ne recadre au pixel près sur le libellé du pot.
        base = _rendre("Pot: 2,50 BB", corps=24)
        for marge in (0, 8, 20):
            im = Image.new("RGB", (base.width + 2 * marge,
                                   base.height + 2 * marge), FEUTRE)
            im.paste(base, (marge, marge))
            self.assertEqual(lire_montant(im), 2.5, f"marge {marge} px")

    def test_deux_lignes_sont_ambigues(self) -> None:
        # Une capture qui attrape deux libellés : on refuse au lieu de choisir.
        haut = _rendre("Pot: 1,50 BB", corps=22)
        bas = _rendre("9,50 BB", corps=22)
        im = Image.new("RGB", (max(haut.width, bas.width),
                               haut.height + bas.height), FEUTRE)
        im.paste(haut, (0, 0))
        im.paste(bas, (0, haut.height))
        r = lire_ligne(im)
        self.assertIsNone(r.valeur)
        self.assertIn("\n", r.texte)

    def test_montant_seul_sur_une_ligne_de_deux(self) -> None:
        # En revanche, un seul nombre réparti sur deux lignes reste lisible.
        haut = _rendre("POT", corps=22)
        bas = _rendre("9,50 BB", corps=22)
        im = Image.new("RGB", (max(haut.width, bas.width),
                               haut.height + bas.height), FEUTRE)
        im.paste(haut, (0, 0))
        im.paste(bas, (0, haut.height))
        self.assertEqual(lire_montant(im), 9.5)


if __name__ == "__main__":
    unittest.main()
