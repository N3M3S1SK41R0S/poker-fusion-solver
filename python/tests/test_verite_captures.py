"""La vérité-terrain des 57 captures, et l'arithmétique qui la mesure.

Pourquoi ce fichier existe
-------------------------
Le dépôt a annoncé « 199 cartes lues sur 209, soit 95 % » sur ces captures.
Le chiffre était faux deux fois :

* **le dénominateur était conditionnel à la détection** — 209 = le nombre de
  boîtes que le détecteur a bien voulu rendre. Une carte jamais localisée ne
  figurait ni au numérateur ni au dénominateur, et le rappel n'existait pas ;
* **il n'y avait aucune vérité-terrain** — « lue avec certitude » voulait dire
  « non refusée ». Une lecture fausse et affirmée comptait comme un succès.

Le vrai dénominateur est dans `donnees/verite_captures.json`, relevé à l'œil
sur les 57 captures : **258 cartes pleinement visibles** (112 héros, 146
board) et 5 en transition.

Ce que ces tests garantissent
-----------------------------
1. **La vérité-terrain est cohérente** — cartes bien formées, aucune carte en
   double dans une même frame (un jeu n'en contient qu'un exemplaire), board
   monotone au sein d'une main, totaux épinglés. Une faute de saisie dans le
   relevé rendrait toute mesure ultérieure fausse en silence : c'est le
   premier endroit où il faut regarder.
2. **L'arithmétique du banc ne peut pas re-fabriquer le « 95 % »** — les
   tests d'imputation vérifient qu'une carte NON DÉTECTÉE reste au
   dénominateur, qu'une lecture affirmée fausse est comptée comme fausse, et
   qu'une carte affirmée avec le mauvais rôle est signalée. C'est exactement
   ce que l'ancienne métrique ne faisait pas.
3. **Le garde-fou de la carte en cours de retournement tient dans la
   CHAÎNE**, pas seulement dans le lecteur à fond plein pris isolément.

Ce que ces tests ne garantissent pas
------------------------------------
Ils ne rejouent pas les 57 captures : les PNG (~108 Mo) ne sont pas
versionnés. Le test qui les rejoue existe, mais il est SAUTÉ quand les
images sont absentes — et il ne faut pas le confondre avec une garantie
permanente. Les chiffres, eux, se rejouent avec `banc_verite_captures.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DONNEES = Path(__file__).parent / "donnees"
VERITE = DONNEES / "verite_captures.json"

_CARTE = re.compile(r"^[2-9TJQKA][cdhs]$")

#: Totaux du relevé. Épinglés parce que ce sont eux qui remplacent le « 209 »
#: conditionnel : si l'un bouge, c'est le relevé qui a changé, et il faut
#: alors savoir laquelle des 57 captures a été revue et pourquoi.
CARTES_HERO = 112
CARTES_BOARD = 146
CARTES_TRANSITION = 5
FRAMES = 57


@pytest.fixture(scope="module")
def verite() -> dict:
    if not VERITE.exists():                  # pragma: no cover - dépôt incomplet
        pytest.fail(f"vérité-terrain absente : {VERITE}")
    return json.loads(VERITE.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════
# 1. LE RELEVÉ EST COHÉRENT
# ═══════════════════════════════════════════════════════════════════════════

def test_le_releve_couvre_les_57_captures(verite) -> None:
    """57 frames, deux sessions, aucun doublon."""
    frames = verite["frames"]
    assert len(frames) == FRAMES
    cles = {(f["session"], f["frame"]) for f in frames}
    assert len(cles) == FRAMES, "deux entrées pour la même capture"
    assert len({f["session"] for f in frames}) == 2


def test_les_totaux_sont_ceux_annonces(verite) -> None:
    """Le dénominateur du rappel, épinglé.

    C'est LA valeur que l'ancienne métrique n'avait pas : 258 cartes
    réellement présentes, contre 209 boîtes détectées.
    """
    frames = verite["frames"]
    hero = sum(len(f["hero"]) for f in frames)
    board = sum(len(f["board"]) for f in frames)
    transition = sum(len(f["hero_transition"]) + len(f["board_transition"])
                     for f in frames)
    assert (hero, board, transition) == (CARTES_HERO, CARTES_BOARD,
                                         CARTES_TRANSITION)
    assert hero + board == 258, (
        "le dénominateur du rappel a changé : reprends le relevé capture par "
        "capture avant de toucher au moindre chiffre annoncé.")


def test_chaque_carte_est_bien_formee(verite) -> None:
    """Notation courte valide partout — sinon le banc apparie dans le vide."""
    for f in verite["frames"]:
        for cle in ("hero", "board", "hero_transition", "board_transition"):
            for carte in f[cle]:
                assert _CARTE.match(carte), (
                    f"{f['session']}/{f['frame']} : carte {carte!r} mal formée")


def test_aucune_carte_en_double_dans_une_frame(verite) -> None:
    """Un jeu ne contient qu'un exemplaire de chaque carte.

    C'est le contrôle qui attrape une faute de recopie : si le relevé disait
    deux fois « 2c » sur la même table, la vérité-terrain serait impossible
    et toute mesure faite dessus serait fausse.
    """
    for f in verite["frames"]:
        toutes = (f["hero"] + f["board"]
                  + f["hero_transition"] + f["board_transition"])
        assert len(toutes) == len(set(toutes)), (
            f"{f['session']}/{f['frame']} : carte en double dans {toutes}")


def test_le_board_ne_recule_jamais_dans_une_main(verite) -> None:
    """Flop, turn, river : le board s'allonge, il ne se réécrit pas.

    Invariant du jeu, et donc contrôle de saisie : si le relevé faisait
    apparaître « 2c 6h 2s » puis « 2c 6h 3s », l'une des deux frames aurait
    été lue de travers. Une main se reconnaît à ses cartes du héros ; le
    board d'une même main doit être un PRÉFIXE du board de la frame suivante,
    ou l'inverse quand une main nouvelle commence.
    """
    for session in {f["session"] for f in verite["frames"]}:
        frames = [f for f in verite["frames"] if f["session"] == session]
        frames.sort(key=lambda f: f["frame"])
        precedent, main = [], None
        for f in frames:
            hero = tuple(f["hero"]) or tuple(f["hero_transition"])
            if hero != main:                 # nouvelle main : le board repart
                main, precedent = hero, []
            board = f["board"] or f["board_transition"]
            if board:
                court, long = sorted((precedent, list(board)), key=len)
                assert long[:len(court)] == court, (
                    f"{session}/{f['frame']} : le board passe de {precedent} "
                    f"à {board} — l'un des deux relevés est faux")
                precedent = list(board)


def test_les_emplacements_suffisent_a_toutes_les_frames(verite) -> None:
    """Chaque carte du relevé a un emplacement où être cherchée."""
    emplacements = verite["emplacements"]
    assert len(emplacements["board"]) == 5 and len(emplacements["hero"]) == 2
    for boite in emplacements["board"] + emplacements["hero"]:
        x, y, w, h = boite
        assert w > 0 and h > 0 and x >= 0 and y >= 0
    for f in verite["frames"]:
        assert len(f["board"]) + len(f["board_transition"]) <= 5
        assert len(f["hero"]) + len(f["hero_transition"]) <= 2


# ═══════════════════════════════════════════════════════════════════════════
# 2. L'ARITHMÉTIQUE DU BANC — celle qui a menti la première fois
# ═══════════════════════════════════════════════════════════════════════════

def _banc():
    import sys

    racine = Path(__file__).resolve().parents[1]
    if str(racine) not in sys.path:
        sys.path.insert(0, str(racine))
    import banc_verite_captures as b

    return b


class _CarteLue:
    """Ce que la chaîne rend pour une boîte — juste ce que le banc lit."""

    def __init__(self, role, boite, carte, statut="sure"):
        self.role, self.boite, self.carte, self.statut = (role, boite, carte,
                                                          statut)
        self.candidat, self.ecart, self.marge = carte, 100, 100


class _Lecture:
    def __init__(self, cartes):
        self.cartes = cartes

    @property
    def sures(self):
        return sum(1 for c in self.cartes if c.statut == "sure")


def _mesure(frame, lues, emplacements=None):
    """Impute une lecture fabriquée, sans toucher au disque."""
    b = _banc()
    emplacements = emplacements or {
        "board": [[644, 514, 117, 160], [777, 514, 117, 160],
                  [910, 514, 117, 160], [1043, 514, 117, 160],
                  [1176, 514, 117, 160]],
        "hero": [[839, 929, 117, 106], [965, 929, 117, 106]],
    }
    bilan = b.Bilan()
    from unittest.mock import patch

    class _Im:
        @staticmethod
        def open(_):
            class _X:
                def convert(self, _m):
                    return self
            return _X()

    with patch.object(b, "lire_image", lambda *a, **k: _Lecture(lues)), \
            patch.dict("sys.modules", {"PIL": type(
                "m", (), {"Image": _Im})()}):
        b.mesure_frame(Path("s/0000.png"), frame, emplacements, bilan)
    return bilan


def _frame(hero=(), board=(), hero_tr=(), board_tr=()):
    return {"session": "s", "frame": "0000", "hero": list(hero),
            "board": list(board), "hero_transition": list(hero_tr),
            "board_transition": list(board_tr), "note": ""}


def test_une_carte_jamais_detectee_reste_au_denominateur() -> None:
    """LE défaut du « 95 % », épinglé.

    Deux cartes présentes, une seule boîte rendue et lue juste. L'ancienne
    métrique aurait annoncé 1/1 = 100 %. Le rappel dit 1/2 = 50 %, et nomme
    la carte perdue.
    """
    bilan = _mesure(
        _frame(hero=("Kc", "9c")),
        [_CarteLue("hero", (839, 929, 117, 106), "Kc")])
    assert bilan.cartes_reelles == 2, (
        "une carte non détectée a disparu du dénominateur — c'est exactement "
        "l'erreur que ce banc existe pour ne plus commettre")
    assert bilan.lues_justes.n == 1
    assert bilan.non_localisees.n == 1
    assert "9c" in bilan.non_localisees.cas[0]


def test_une_lecture_affirmee_fausse_est_comptee_fausse() -> None:
    """« Non refusée » n'est pas « juste » : la carte lue est comparée."""
    bilan = _mesure(
        _frame(board=("2c", "6h", "2s")),
        [_CarteLue("board", (644, 514, 117, 160), "Kc")])
    assert bilan.fausses_affirmees.n == 1
    assert bilan.lues_justes.n == 0
    assert bilan.localisees.n == 1, (
        "la carte est bien localisée : c'est la LECTURE qui est fausse")


def test_une_affirmation_sur_du_vide_est_une_carte_inventee() -> None:
    """Une boîte posée là où il n'y a aucune carte, et pourtant affirmée."""
    bilan = _mesure(
        _frame(hero=("Kc", "9c")),
        [_CarteLue("hero", (839, 929, 117, 106), "Kc"),
         _CarteLue("hero", (965, 929, 117, 106), "9c"),
         _CarteLue("?", (20, 20, 100, 140), "As")])
    assert bilan.inventees.n == 1
    assert bilan.lues_justes.n == 2


def test_un_role_faux_est_signale_meme_quand_la_carte_est_juste() -> None:
    """Une carte du board présentée comme carte du héros reste une faute.

    Pour un solveur, confondre les deux est aussi grave que lire une carte
    fausse : la main du héros et le tableau ne jouent pas le même rôle. C'est
    ce qui arrive réellement sur la table 7-max, où le détecteur ne trouve
    aucune carte au siège et promeut le board à sa place.
    """
    bilan = _mesure(
        _frame(board=("6s", "Ah", "5s")),
        [_CarteLue("hero", (644, 514, 117, 160), "6s")])
    assert bilan.lues_justes.n == 1
    assert bilan.lues_justes_bon_role.n == 0
    assert bilan.roles_faux.n == 1


def test_une_carte_en_transition_ne_gonfle_pas_le_rappel() -> None:
    """Elle est comptée à part : refuser une carte à moitié retournée est sain.

    Mais si elle est AFFIRMÉE, elle est jugée : juste ou fausse, le banc le
    dit. C'est le seul traitement honnête — l'exclure du dénominateur sans
    juger les affirmations recréerait le biais du « 95 % ».
    """
    bilan = _mesure(
        _frame(board_tr=("6c", "As", "Js")),
        [_CarteLue("board", (644, 514, 117, 160), "Kc"),
         _CarteLue("board", (777, 514, 117, 160), "As")])
    assert bilan.cartes_reelles == 0, "une carte en transition n'est pas au rappel"
    assert bilan.transitoires == 3
    assert bilan.transitoires_affirmees.n == 2
    assert bilan.fausses_affirmees.n == 1, (
        "le 6♣ affirmé « Kc » doit être compté comme une lecture fausse, même "
        "en transition")
    assert bilan.affirmations_justes == 1, "le As affirmé juste compte pour la précision"


def test_une_boite_qui_rate_la_carte_ne_la_valide_pas() -> None:
    """Le recouvrement est exigé : une boîte à côté ne « trouve » rien."""
    bilan = _mesure(
        _frame(hero=("Kc",)),
        [_CarteLue("hero", (400, 200, 117, 106), "Kc")])
    assert bilan.non_localisees.n == 1
    assert bilan.inventees.n == 1, (
        "affirmer « Kc » à 500 px de la carte est une carte inventée")


# ═══════════════════════════════════════════════════════════════════════════
# 3. LE GARDE-FOU TIENT DANS LA CHAÎNE, PAS SEULEMENT DANS LE LECTEUR
# ═══════════════════════════════════════════════════════════════════════════

CARTES_RETOURNEES = ("pmu_carte_retournee_6c.png", "pmu_carte_retournee_as.png")


def test_une_carte_en_retournement_n_est_jamais_affirmee_par_la_chaine() -> None:
    """Le refus du lecteur à fond plein doit être FRANC, pas contournable.

    `test_lecteur_fond_plein.py` vérifie déjà que `lire_carte_fond_plein`
    refuse ces deux découpes. Ce n'était pas suffisant : dans la chaîne
    complète, le refus était muet, `identify_card_autour` passait la main au
    hachage, et les 40 cadrages essayés finissaient par en trouver un sous le
    seuil. Mesuré le 11 août 2026 par `banc_verite_captures.py` sur la
    capture `300_7-max_KO/0003` : le 6♣ ressortait **« Kc », statut
    « sure »**, écart 616, marge 33.

    Un garde-fou qu'on contourne en changeant de chemin n'en est pas un.
    Ce test emprunte le chemin de la chaîne, pas celui du lecteur.
    """
    from PIL import Image

    from pfs.vision.card_recognizer import identify_card_autour

    presents = [DONNEES / n for n in CARTES_RETOURNEES
                if (DONNEES / n).exists()]
    if not presents:                         # pragma: no cover - dépôt incomplet
        pytest.skip("découpes de cartes retournées absentes")

    for chemin in presents:
        decoupe = Image.open(chemin).convert("RGB")
        # La découpe telle qu'elle sort du détecteur, posée dans une image un
        # peu plus grande : c'est la situation réelle, celle qui laisse la
        # boucle des 40 cadrages s'exercer.
        fond = Image.new("RGB", (decoupe.width + 40, decoupe.height + 40),
                         (28, 22, 46))
        fond.paste(decoupe, (20, 20))
        m = identify_card_autour(fond, (20, 20, decoupe.width, decoupe.height))
        assert m.statut != "sure", (
            f"{chemin.name} : la chaîne AFFIRME « {m.card} » sur une carte en "
            f"cours de retournement (écart {m.distance}, marge {m.margin}). "
            "C'est le pire mode d'échec de cet outil : une carte nommée sans "
            "réserve, et fausse.")
        assert m.card is None


def test_le_refus_franc_ne_s_applique_qu_au_jeu_a_fond_plein() -> None:
    """Réciproque : une découpe qui n'est PAS de ce jeu passe au hachage.

    Sans cette réciproque, le correctif ci-dessus pourrait être « refuser
    toute découpe non uniforme », ce qui condamnerait le deck classique —
    dont le fond blanc, une fois les pixels blancs écartés, ne laisse que des
    pips très dispersés.
    """
    from PIL import Image

    from pfs.vision.card_recognizer import _TEMPLATE_ROOT, _lecture_fond_plein

    p = _TEMPLATE_ROOT / "pmu_deck" / "Ah.png"
    if not p.exists():                       # pragma: no cover - dépôt incomplet
        pytest.skip("habillage pmu_deck absent")
    carte = Image.open(p).convert("RGB").resize((120, 165), Image.LANCZOS)
    assert _lecture_fond_plein(carte, (0, 0, 120, 165)) is None, (
        "une carte du deck CLASSIQUE ne relève pas du lecteur à fond plein : "
        "il doit passer la main, pas refuser.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. LE BANC COMPLET, QUAND LES CAPTURES SONT LÀ
# ═══════════════════════════════════════════════════════════════════════════

def test_le_banc_complet_ne_produit_aucune_affirmation_fausse(verite) -> None:
    """Rejoue les 57 captures si elles sont sur cette machine.

    SAUTÉ sans les images — elles pèsent ~108 Mo et ne sont pas versionnées.
    Ce test ne remplace donc pas `banc_verite_captures.py`, il le double là
    où c'est possible. Les seuils vérifiés sont ceux qui ne doivent JAMAIS se
    dégrader : aucune carte inventée, aucune lecture affirmée fausse.

    Et le RAPPEL, épinglé à sa valeur exacte. Sans lui, ce fichier vérifiait
    tout de la mesure sauf le nombre qu'elle annonce : une régression qui
    ferait tomber la localisation de 198 à 20 laisserait la suite verte,
    puisque moins on lit, moins on peut se tromper. C'est exactement le
    travers que ce chantier reproche au « 95 % ». La valeur est déterministe
    (mêmes images, même chaîne, aucun tirage) et elle TOMBE quand on touche
    au détecteur : `QUIET_SIDES = 2` la porte à 250.
    """
    b = _banc()
    racine = b.captures_par_defaut()
    if not racine.is_dir():
        pytest.skip(f"captures absentes ({racine})")

    bilan = b.Bilan()
    vues = 0
    for frame in verite["frames"]:
        chemin = racine / frame["session"] / f"{frame['frame']}.png"
        if not chemin.exists():
            continue
        b.mesure_frame(chemin, frame, verite["emplacements"], bilan)
        vues += 1
    if vues < FRAMES:
        pytest.skip(f"{vues}/{FRAMES} captures présentes seulement")

    assert bilan.fausses_affirmees.n == 0, bilan.fausses_affirmees.cas
    assert bilan.inventees.n == 0, bilan.inventees.cas
    assert bilan.cartes_reelles == 258
    assert (bilan.localisees.n, bilan.lues_justes.n,
            bilan.lues_justes_bon_role.n) == (198, 198, 168), (
        f"le rappel a bougé : {bilan.localisees.n} localisées, "
        f"{bilan.lues_justes.n} lues justes, "
        f"{bilan.lues_justes_bon_role.n} avec le bon rôle, sur 258 cartes "
        "réellement présentes (attendu 198/198/168, soit 76,7 % / 65,1 %). "
        "Vers le HAUT, c'est un progrès à mesurer et à réannoncer partout "
        "(README, CHANGELOG, docstring de table_detector) ; vers le BAS, "
        "c'est une régression silencieuse de la perception.")
