"""Le viseur de montants : lit ce qui est là, refuse ce qui ne l'est pas.

Deux étages de vérification :

1. **Ciblage seul** : des images fabriquées ici, avec un `TableRead` construit
   à la main — on contrôle exactement où est chaque texte, donc on sait ce que
   chaque zone DOIT rendre. La chaîne de lecture elle-même (segmentation,
   gabarits) a son propre banc dans `test_digit_ocr.py` : ici on teste la
   visée et les règles de sélection.
2. **Bout en bout** : `render_table` (générateur synthétique) → `read_table`
   (détection réelle) → `lire_montants` — le chemin exact de la route
   ``lire_capture``.

Le contrat central, hérité de toute la vision : **jamais de valeur inventée**.
Une image sans texte, un désaccord entre les deux affichages du pot, une
relance sans bouton PAYER : autant de refus motivés, pas de conjectures.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw, ImageFont

from pfs.vision.digit_ocr import _chemin_police, charger_gabarits
from pfs.vision.table_detector import CardBox, TableRead
from pfs.vision.zones_montants import lire_montants

FOND = (32, 78, 46)          # feutre vert sombre
ENCRE = (240, 240, 228)

# La police d'interface doit exister pour rendre du texte lisible par les
# gabarits ; sur une machine sans Segoe ni Arial, ces tests n'ont pas de sens.
_POLICE = _chemin_police("segoeui.ttf") or _chemin_police("arial.ttf")
pytestmark = pytest.mark.skipif(
    _POLICE is None, reason="aucune police d'interface (segoeui/arial) — "
    "le rendu de texte de test serait illisible par construction")


@pytest.fixture(scope="module")
def gabarits():
    return charger_gabarits()


def _image(w: int = 900, h: int = 560) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), FOND)
    return img, ImageDraw.Draw(img)


def _police(taille: int = 20) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(_POLICE), taille)


def _table(hero: bool = True, board: bool = True) -> TableRead:
    """Une table plausible : board centré, héros en bas — cartes de 68×92.

    Les boîtes sont la seule chose que `lire_montants` regarde : pas besoin
    de dessiner les cartes elles-mêmes pour tester la visée.
    """
    cartes_board = [CardBox(340 + i * 76, 213, 68, 92, "board") for i in range(3)]
    cartes_hero = [CardBox(382 + i * 72, 403, 68, 92, "hero") for i in range(2)]
    return TableRead(hero=cartes_hero if hero else [],
                     board=cartes_board if board else [], others=[])


class TestPot:
    def test_pot_lu_au_dessus_du_board(self, gabarits):
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["pot"]["refus"] is None
        assert m["pot"]["valeur"] == 12.5

    def test_contre_verification_en_accord(self, gabarits):
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        d.text((420, 315), "12,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["pot"]["valeur"] == 12.5

    def test_desaccord_entre_les_deux_affichages_refuse(self, gabarits):
        # Sur le client, la pastille « Pot: » et l'étiquette du tas de jetons
        # portent le MÊME nombre. Si la lecture les voit différents, l'une
        # des deux est fausse — deviner laquelle serait inventer un pot.
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        d.text((420, 315), "13,00 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["pot"]["valeur"] is None
        assert "désaccord" in m["pot"]["refus"]

    def test_sans_board_le_pot_est_refuse_avec_motif(self, gabarits):
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(board=False), gabarits)
        assert m["pot"]["valeur"] is None
        assert "board" in m["pot"]["refus"]


class TestTapis:
    def test_tapis_lu_sous_les_cartes_du_heros(self, gabarits):
        img, d = _image()
        d.text((410, 500), "34,20 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["tapis"]["refus"] is None
        assert m["tapis"]["valeur"] == 34.2

    def test_le_pseudo_et_le_compte_a_rebours_sont_ecartes(self, gabarits):
        # La plaque réelle empile « Temps: 16 » (deux-points), le pseudo
        # (lettres) et le tapis : seule la ligne « montant » doit compter.
        # L'image est plus haute que la table : une ligne collée au bord
        # serait ignorée par principe (ligne peut-être coupée), et c'est
        # testé plus bas — ici on teste la sélection, pas le garde-fou.
        img, d = _image(900, 640)
        d.text((400, 500), "Temps: 16", fill=ENCRE, font=_police(18))
        d.text((408, 524), "sailorwaterfall5", fill=ENCRE, font=_police(18))
        d.text((414, 548), "24,87 BB", fill=ENCRE, font=_police(18))
        m = lire_montants(img, _table(), gabarits)
        assert m["tapis"]["valeur"] == 24.87

    def test_une_ligne_coupee_par_le_bord_est_ignoree(self, gabarits):
        # « 24,87 BB » tronqué par le bas se lisait 87 à confiance 0,66 —
        # une ligne qui touche le bord du cadre n'est jamais lue.
        img, d = _image(900, 560)
        d.text((414, 548), "24,87 BB", fill=ENCRE, font=_police(18))
        m = lire_montants(img, _table(), gabarits)
        assert m["tapis"]["valeur"] is None


class TestMise:
    def test_bouton_payer_donne_la_mise(self, gabarits):
        img, d = _image()
        d.text((620, 505), "PAYER 3,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["mise"]["refus"] is None
        assert m["mise"]["valeur"] == 3.5

    def test_check_signifie_rien_a_payer(self, gabarits):
        # « Check » à l'écran est une INFORMATION : la mise à payer est
        # nulle. Ce zéro-là est lu, pas supposé.
        img, d = _image()
        d.text((640, 505), "Check", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["mise"]["valeur"] == 0.0

    def test_une_relance_seule_ne_donne_pas_la_mise(self, gabarits):
        # « RELANCER À 8 BB » n'est pas le montant à payer — le confondre
        # gonflerait la mise et fausserait toute la suite du conseil.
        img, d = _image()
        d.text((600, 505), "RAISE 8,00 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["mise"]["valeur"] is None

    def test_sans_bouton_la_mise_est_refusee_avec_motif(self, gabarits):
        img, _ = _image()
        m = lire_montants(img, _table(), gabarits)
        assert m["mise"]["valeur"] is None
        assert "bouton" in m["mise"]["refus"]


class TestBlinde:
    def test_le_suffixe_bb_confirme_l_affichage_en_blindes(self, gabarits):
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["blinde"]["valeur"] == 1.0

    def test_sans_suffixe_bb_la_blinde_reste_a_saisir(self, gabarits):
        # Affichage en jetons (« Pot: 950 ») : la blinde vaut peut-être
        # 2 000, peut-être 50 — l'affirmer serait pire que la demander.
        # (950 et pas 1500 : un « 1 » de tête se segmente en « 1 500 »,
        # deux mots numériques, que digit_ocr refuse — c'est assumé, les
        # séparateurs de milliers à l'écran produisent le même refus.
        # L'affichage en BB est le mode recommandé.)
        img, d = _image()
        d.text((400, 180), "Pot: 950", fill=ENCRE, font=_police())
        m = lire_montants(img, _table(), gabarits)
        assert m["pot"]["valeur"] == 950
        assert m["blinde"]["valeur"] is None
        assert "saisis" in m["blinde"]["refus"]


class TestRienNestInvente:
    def test_image_sans_texte_quatre_refus_motives(self, gabarits):
        img, _ = _image()
        m = lire_montants(img, _table(), gabarits)
        for zone in ("pot", "mise", "tapis", "blinde"):
            assert m[zone]["valeur"] is None, zone
            assert m[zone]["refus"], zone

    def test_sans_aucune_carte_detectee_rien_ne_sort(self, gabarits):
        img, d = _image()
        d.text((400, 180), "Pot: 12,50 BB", fill=ENCRE, font=_police())
        m = lire_montants(img, TableRead(hero=[], board=[], others=[]), gabarits)
        assert m["pot"]["valeur"] is None
        assert m["tapis"]["valeur"] is None


class TestBoutEnBout:
    """render_table → read_table → lire_montants : le chemin de la route."""

    def test_la_table_synthetique_est_lue_de_bout_en_bout(self, gabarits):
        from pfs.vision.synth_table import TableSpec, render_table
        from pfs.vision.table_detector import read_table

        # decor=False : la mise en page du générateur est compressée
        # verticalement et fait croiser la rangée du tapis avec celle des
        # boutons FOLD/CALL/RAISE — le coin d'un bouton sur la même ligne
        # corrompt la référence de hauteur des chiffres et fait REFUSER la
        # ligne (échec sûr, mesuré : « 42,50 » → « 42,so » → refus). Sur le
        # client réel les deux rangées sont disjointes (57 captures). Le cas
        # « bruit sur la ligne » est couvert par les tests de sélection.
        spec = TableSpec(hero=("Ah", "Kd"), board=("2c", "7d", "Jh"),
                         pot=6.75, hero_stack=42.5, to_call=2.25,
                         taille_texte=18, decor=False)
        synth = render_table(spec)
        table = read_table(synth.image)
        assert len(table.board) == 3 and len(table.hero) == 2, (
            "la détection doit voir la table synthétique — sinon ce test "
            "mesure autre chose que la lecture des montants")
        m = lire_montants(synth.image, table, gabarits)
        assert m["pot"]["valeur"] == 6.75
        assert m["tapis"]["valeur"] == 42.5
        assert m["mise"]["valeur"] == 2.25
        assert m["blinde"]["valeur"] == 1.0
