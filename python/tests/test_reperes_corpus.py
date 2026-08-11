r"""Les repères mesurés : du corpus au module, du module à la route, à la page.

Ce que ces tests protègent, et comment on l'a vérifié
-----------------------------------------------------
La revue de session comparait le joueur à « 28–50 % », un chiffre de manuel.
Il est remplacé par des valeurs recalculées sur ``phh-dataset``. Trois choses
peuvent casser, et chacune a son test :

1. **Le calcul** — ``calculer_jeu`` doit compter comme
   :meth:`~pfs.data.hand_history.ParsedHand.stat_observations`, sinon le
   repère et la mesure de l'utilisateur ne parlent plus de la même chose.
   Vérifié sur des mains PHH écrites à la main, dont on connaît la réponse.
2. **Le branchement** — un repère qui n'est pas atteignable par une route et
   déclenché par un élément de page est un module mort. Le parcours HTTP réel
   est traversé ici, sur le vrai serveur.
3. **L'honnêteté** — le WTSD a un dénominateur qui dépend du nombre de sièges,
   et un jeu de 90 mains-joueurs ne conclut rien. Ces deux réserves doivent
   SORTIR de l'API, pas rester dans un commentaire.

Preuve que ces tests tombent quand on casse ce qu'ils protègent (mutations
appliquées à la main, une par une, puis annulées) :

* dans ``reperes.py``, vider ``_DEPEND_DU_NOMBRE_DE_SIEGES`` (le WTSD devient
  comparable) → ``test_le_wtsd_est_marque_non_comparable`` et
  ``test_la_route_review_sort_la_reserve_sur_le_wtsd`` échouent ;
* remplacer ``N_MIN_CONCLUANT = 1000`` par ``0`` → ``test_wsop_non_concluant``
  échoue ;
* faire rendre ``"pluribus_6max"`` à ``jeu_par_defaut`` pour toute taille de
  table → ``test_table_courte_prend_les_positions_tardives`` échoue ;
* retirer ``"reperes"`` du dictionnaire de la réponse de ``API.review`` →
  ``test_la_route_review_porte_les_reperes`` échoue ;
* retirer l'appel ``reperesRun()`` de ``ui.html`` →
  ``test_la_page_declenche_la_route_reperes`` échoue.

Ce qui n'est PAS testé ici
--------------------------
Les valeurs gelées elles-mêmes (VPIP 32,19 % sur les positions tardives…) ne
sont pas revérifiées contre le corpus : celui-ci vit hors du dépôt, 10 000
fichiers, et une machine sans corpus ne pourrait pas exécuter la suite. C'est
le rôle de ``python banc_reperes_corpus.py --verifier``, qui compare la table
gelée au corpus chiffre par chiffre et sort en code 1 au moindre écart. Les
tests d'ici vérifient la mécanique et la cohérence interne de la table.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from pfs.analysis import review_hands
from pfs.analysis.reperes import (
    N_MIN_CONCLUANT,
    POSITIONS_TABLE_COURTE,
    REPERES,
    ReperesError,
    calculer_jeu,
    compter_agression,
    jeu_par_defaut,
    situer,
    wilson,
)
from pfs.app.server import create_server
from pfs.data.hand_history import parse_ipoker, player_key
from pfs.data.phh import parse_phh

JETON = "jeton-des-reperes"

# Deux mains PHH minimales dont on connaît les comptes à la main.
# Main A : p1 (SB/BTN en heads-up) relance, p2 se couche. Pas de flop.
MAIN_A = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [50, 100]
starting_stacks = [10000, 10000]
actions = ["d dh p1 AsAh", "d dh p2 7d2c", "p1 cbr 300", "p2 f"]
finishing_stacks = [10100, 9900]
"""

# Main B : p1 suit, p2 checke, flop, p2 mise, p1 suit, turn, river, abattage.
MAIN_B = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [50, 100]
starting_stacks = [10000, 10000]
actions = ["d dh p1 AsAh", "d dh p2 7d2c", "p1 cc", "p2 cc",
           "d db Jc3d5c", "p2 cbr 100", "p1 cc",
           "d db 9h", "p2 cc", "p1 cc",
           "d db 2s", "p2 cc", "p1 cc"]
finishing_stacks = [10200, 9800]
"""


# ═══════════════════════════════════════════════════════════════════════════
# 1. LE CALCUL
# ═══════════════════════════════════════════════════════════════════════════


def _corpus(tmp_path, mains):
    for i, texte in enumerate(mains, start=1):
        (tmp_path / f"{i}.phh").write_text(texte, encoding="utf-8")
    return tmp_path


def test_calcul_compte_les_occasions_a_la_main(tmp_path):
    """Deux mains connues → des comptes vérifiables sans exécuter le code.

    Main A : p1 relance (VPIP + PFR), p2 se couche (ni l'un ni l'autre).
    Main B : p1 suit (VPIP, pas PFR), p2 checke la BB (ni l'un ni l'autre).
    Donc VPIP = 2/4, PFR = 1/4.
    """
    jeu = calculer_jeu(_corpus(tmp_path, [MAIN_A, MAIN_B]), "essai", "src", ())
    assert jeu.n_mains == 2
    assert jeu.n_mains_joueurs == 4
    assert (jeu["vpip"].succes, jeu["vpip"].occasions) == (2, 4)
    assert (jeu["pfr"].succes, jeu["pfr"].occasions) == (1, 4)


def test_le_wtsd_ne_compte_que_les_mains_avec_flop(tmp_path):
    """Occasion de WTSD = un flop est tombé, pour TOUS les joueurs assis.

    Seule la main B a un board : 2 occasions, et les deux joueurs vont à
    l'abattage. La main A n'en crée aucune.
    """
    jeu = calculer_jeu(_corpus(tmp_path, [MAIN_A, MAIN_B]), "essai", "src", ())
    assert (jeu["wtsd"].succes, jeu["wtsd"].occasions) == (2, 2)


def test_agression_ignore_le_preflop(tmp_path):
    """L'AF ne compte que flop, turn et river.

    Main B : p2 mise une fois au flop (1 agressive), p1 suit une fois
    (1 suivi). La relance préflop de la main A ne doit RIEN ajouter, sans
    quoi PFR et AF diraient deux fois la même chose.
    """
    main_b = parse_phh(MAIN_B)
    joueurs = [s.player for s in main_b.seats]
    assert compter_agression(main_b, joueurs[0]) == (0, 1)
    assert compter_agression(main_b, joueurs[1]) == (1, 0)

    main_a = parse_phh(MAIN_A)
    for j in (s.player for s in main_a.seats):
        assert compter_agression(main_a, j) == (0, 0)


def test_restriction_aux_positions_change_le_denominateur(tmp_path):
    """Le filtre de positions retire bien des mains-joueurs du comptage.

    En heads-up les positions sont SB et BB : le filtre « table courte »
    (BTN/SB/BB) les garde toutes les deux, un filtre sur la seule BB n'en
    garde qu'une, et un filtre sur UTG — position inexistante à deux — n'en
    garde aucune et ne fabrique aucun repère plutôt qu'un repère vide.
    """
    racine = _corpus(tmp_path, [MAIN_A, MAIN_B])
    complet = calculer_jeu(racine, "e", "src", ())
    assert complet.n_mains_joueurs == 4
    tardives = calculer_jeu(racine, "e", "src", (),
                            positions=POSITIONS_TABLE_COURTE)
    assert tardives.n_mains_joueurs == 4      # SB et BB sont toutes deux dedans
    # p1 est la SB (= le bouton en heads-up) et entre dans les deux mains ;
    # p2 est la BB et n'entre dans aucune. Le filtre doit donc changer le
    # NUMÉRATEUR autant que le dénominateur.
    sb_seule = calculer_jeu(racine, "e", "src", (), positions=("SB",))
    assert (sb_seule["vpip"].succes, sb_seule["vpip"].occasions) == (2, 2)
    bb_seule = calculer_jeu(racine, "e", "src", (), positions=("BB",))
    assert bb_seule.n_mains_joueurs == 2
    assert (bb_seule["vpip"].succes, bb_seule["vpip"].occasions) == (0, 2)
    hors = calculer_jeu(racine, "e", "src", (), positions=("UTG",))
    assert hors.n_mains_joueurs == 0
    assert hors.reperes == {}


def test_corpus_absent_leve_une_erreur(tmp_path):
    with pytest.raises(ReperesError):
        calculer_jeu(tmp_path / "nulle-part", "e", "src", ())


def test_wilson_encadre_le_taux():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == (0.0, 1.0)
    # Un petit effectif doit donner un intervalle LARGE : c'est tout l'objet.
    lo_petit, hi_petit = wilson(29, 90)
    assert hi_petit - lo_petit > 0.15


# ═══════════════════════════════════════════════════════════════════════════
# 2. L'HONNÊTETÉ DE LA TABLE GELÉE
# ═══════════════════════════════════════════════════════════════════════════


def test_la_table_gelee_est_coherente():
    """Chaque repère doit pouvoir être recalculé depuis ses deux entiers."""
    for cle, jeu in REPERES.items():
        assert jeu.cle == cle
        assert jeu.n_mains > 0 and jeu.n_mains_joueurs > 0
        assert jeu.limites, f"{cle} sans limites énoncées"
        for stat, r in jeu.reperes.items():
            assert r.occasions > 0, f"{cle}.{stat} sans dénominateur"
            attendu = (r.succes / r.occasions
                       * (1.0 if stat == "af_postflop" else 100.0))
            assert abs(r.valeur - attendu) < 1e-3, (
                f"{cle}.{stat} : {r.valeur} ≠ {r.succes}/{r.occasions}")
            assert r.ic_bas <= r.valeur <= r.ic_haut


def test_wsop_non_concluant():
    """Les 18 mains de hold'em des WSOP ne peuvent RIEN conclure."""
    jeu = REPERES["wsop2023_holdem"]
    assert jeu.n_mains == 18
    assert jeu.n_mains_joueurs == 90
    assert jeu.n_mains_joueurs < N_MIN_CONCLUANT
    assert jeu.concluant is False
    # L'IC du VPIP doit être si large qu'il interdit toute lecture fine.
    vpip = jeu["vpip"]
    assert vpip.ic_haut - vpip.ic_bas > 15.0
    # Et aucune dispersion inter-joueurs n'est publiée : 18 mains par joueur.
    assert all(r.n_joueurs == 0 for r in jeu.reperes.values())


def test_pluribus_est_concluant_et_dispersé():
    for cle in ("pluribus_6max", "pluribus_tardives"):
        jeu = REPERES[cle]
        assert jeu.concluant is True
        assert jeu["vpip"].n_joueurs >= 10


def test_table_courte_prend_les_positions_tardives():
    assert jeu_par_defaut(2) == "pluribus_tardives"
    assert jeu_par_defaut(3) == "pluribus_tardives"
    assert jeu_par_defaut(6) == "pluribus_6max"
    assert REPERES[jeu_par_defaut(3)].positions == POSITIONS_TABLE_COURTE


def test_le_melange_de_positions_change_le_repere():
    """L'agrégat 6-max est plus serré que les seules positions tardives.

    C'est la raison d'être du jeu « tardives » : comparer un joueur de table
    à trois à l'agrégat 6-max lui reproche l'UTG et le MP qu'il ne joue pas.
    Si l'écart disparaissait, le second jeu n'aurait plus d'objet.
    """
    six = REPERES["pluribus_6max"]["vpip"].valeur
    trois = REPERES["pluribus_tardives"]["vpip"].valeur
    assert trois - six > 4.0


def test_le_wtsd_est_marque_non_comparable():
    """Le dénominateur du WTSD dépend du nombre de sièges : dit, pas caché."""
    ecarts = situer({"vpip": 63.1, "wtsd": 44.9}, REPERES["pluribus_tardives"])
    par_stat = {e.stat: e for e in ecarts}
    assert par_stat["vpip"].comparable is True
    assert par_stat["wtsd"].comparable is False
    assert "dénominateur" in par_stat["wtsd"].pourquoi
    # Les comparables passent devant : le premier écart de la liste doit être
    # lisible, pas un artefact de dénominateur.
    assert ecarts[0].comparable is True


def test_situer_classe_le_plus_grand_ecart_en_tete():
    """Le profil réel de l'utilisateur : son plus grand écart est le VPIP."""
    profil = {"vpip": 63.1, "pfr": 31.3, "ecart_vpip_pfr": 31.8,
              "three_bet": 5.4, "wtsd": 44.9}
    ecarts = situer(profil, REPERES["pluribus_tardives"])
    comparables = [e for e in ecarts if e.comparable]
    assert comparables[0].stat == "vpip"
    assert comparables[0].ecart > 30.0
    assert comparables[0].ecart_relatif > 1.9
    # le 3-bet est SOUS le repère : le sens du signe doit être conservé
    trois_bet = next(e for e in ecarts if e.stat == "three_bet")
    assert trois_bet.ecart < 0


def test_situer_ignore_les_stats_absentes_du_jeu():
    assert situer({"inexistante": 1.0}, REPERES["pluribus_6max"]) == []


# ═══════════════════════════════════════════════════════════════════════════
# 3. LA REVUE DE SESSION SE SERT DES REPÈRES
# ═══════════════════════════════════════════════════════════════════════════

HEADS_UP = """<?xml version="1.0"?>
<session sessioncode="9">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <nickname>H</nickname><tournamentcode>7</tournamentcode></general>
 <game gamecode="9">
  <general><smallblind>10</smallblind><bigblind>20</bigblind>
   <players>
    <player bet="60" chips="1000" dealer="1" name="H" seat="1" win="120"/>
    <player bet="60" chips="1000" dealer="0" name="V" seat="2" win="0"/>
   </players></general>
  <round no="0">
   <action no="1" player="H" sum="10" type="1"/>
   <action no="2" player="V" sum="20" type="2"/></round>
  <round no="1">
   <cards player="H" type="Pocket">SA HA</cards>
   <action no="3" player="H" sum="20" type="3"/>
   <action no="4" player="V" sum="0" type="4"/></round>
  <round no="2"><cards type="Flop">C2 D7 S9</cards>
   <action no="5" player="V" sum="20" type="5"/>
   <action no="6" player="H" sum="20" type="3"/></round>
 </game>
</session>"""


def test_la_revue_choisit_le_repere_de_table_courte():
    rep = review_hands(parse_ipoker(HEADS_UP), hero=player_key("H"))
    assert rep.profile.sieges_median == 2
    assert rep.jeu_de_reperes().cle == "pluribus_tardives"


def test_la_revue_mesure_l_agression_postflop():
    """Le héros suit une mise au flop : 0 agressive, 1 suivi → AF = 0."""
    rep = review_hands(parse_ipoker(HEADS_UP), hero=player_key("H"))
    assert rep.profile.calls_postflop == 1
    assert rep.profile.aggr_postflop == 0
    assert rep.profile.af_postflop == 0.0


def test_l_explication_cite_le_corpus_et_ses_limites():
    rep = review_hands(parse_ipoker(HEADS_UP), hero=player_key("H"))
    texte = rep.explain()
    assert "pluribus_tardives" in texte
    assert "mains-joueurs" in texte
    assert "non comparable" in texte        # la réserve WTSD est imprimée
    assert "BORNE BASSE" in texte           # la limite du jeu est imprimée


def test_profil_vide_ne_casse_pas():
    """Aucune main : pas de division par zéro, et un jeu par défaut quand même."""
    rep = review_hands([], hero=None)
    assert rep.profile.sieges_median == 0
    assert rep.jeu_de_reperes().cle == "pluribus_6max"
    assert rep.explain()


# ═══════════════════════════════════════════════════════════════════════════
# 4. DE BOUT EN BOUT : LA ROUTE RÉELLE ET LA PAGE RÉELLE
# ═══════════════════════════════════════════════════════════════════════════


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def base() -> str:
    srv, _ = create_server(_port_libre(), token=JETON)
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        fil.join(timeout=5)
        srv.server_close()


def _poster(base: str, route: str, charge: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{base}/api/{route}",
        data=json.dumps(charge).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-PFS-Token": JETON},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as rep:
            return rep.status, json.loads(rep.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_la_route_reperes_sert_les_trois_jeux(base):
    code, corps = _poster(base, "reperes", {})
    assert code == 200
    cles = {j["cle"] for j in corps["jeux"]}
    assert cles == set(REPERES)
    par_cle = {j["cle"]: j for j in corps["jeux"]}
    # chaque jeu sort avec ses effectifs, ses limites et ses dénominateurs
    for jeu in par_cle.values():
        assert jeu["n_mains_joueurs"] > 0
        assert jeu["limites"]
        assert all(s["occasions"] > 0 for s in jeu["stats"].values())
    assert par_cle["wsop2023_holdem"]["concluant"] is False


def test_la_route_reperes_selectionne_par_nombre_de_sieges(base):
    code, corps = _poster(base, "reperes", {"sieges": 3})
    assert code == 200
    assert [j["cle"] for j in corps["jeux"]] == ["pluribus_tardives"]


def test_la_route_reperes_refuse_un_jeu_inconnu(base):
    code, corps = _poster(base, "reperes", {"jeu": "gto-wizard"})
    assert code >= 400
    assert "gto-wizard" in corps["error"]


def test_la_route_review_porte_les_reperes(base, tmp_path):
    """La revue d'un dossier réel rapporte les repères ET les écarts."""
    (tmp_path / "main.xml").write_text(HEADS_UP, encoding="utf-8")
    code, corps = _poster(base, "review", {"path": str(tmp_path)})
    assert code == 200
    assert corps["profile"]["n_hands"] == 1
    rp = corps["reperes"]
    assert rp["cle"] == "pluribus_tardives"
    assert rp["stats"]["vpip"]["occasions"] == 30000
    assert rp["ecarts"], "aucun écart calculé"
    # l'AF du héros doit être mesurée et transmise
    assert "af_postflop" in corps["profile"]


def test_la_route_review_sort_la_reserve_sur_le_wtsd(base, tmp_path):
    """La réserve de comparabilité doit traverser l'API, pas rester en Python."""
    (tmp_path / "main.xml").write_text(HEADS_UP, encoding="utf-8")
    _, corps = _poster(base, "review", {"path": str(tmp_path)})
    wtsd = next(e for e in corps["reperes"]["ecarts"] if e["stat"] == "wtsd")
    assert wtsd["comparable"] is False
    assert "dénominateur" in wtsd["pourquoi"]


def test_la_page_declenche_la_route_reperes(base):
    """Sur les octets RÉELLEMENT servis par GET / : le bouton et son appel.

    Vérification structurelle, pas comportementale — aucun moteur JS n'est
    disponible ici, et le test le dit plutôt que de prétendre le contraire.
    """
    with urllib.request.urlopen(f"{base}/", timeout=30) as rep:
        page = rep.read().decode("utf-8")
    assert 'onclick="reperesRun()"' in page
    assert 'id="rp-out"' in page
    assert 'api("reperes"' in page
    # la revue lit bien le bloc repères de la réponse, et non des constantes
    assert "rp=r.reperes" in page
    assert "repère mesuré" in page


def test_la_page_ne_contient_plus_la_fourchette_de_manuel(base):
    """« zone saine 28–50 % » ne doit plus exister nulle part dans la page.

    C'est le défaut d'origine : une borne sans corpus ni dénominateur. Le
    test échoue si elle revient, sous cette forme ou avec ses bornes en dur.
    """
    with urllib.request.urlopen(f"{base}/", timeout=30) as rep:
        page = rep.read().decode("utf-8")
    assert "zone saine" not in page
    assert "statRow(\"VPIP" not in page
