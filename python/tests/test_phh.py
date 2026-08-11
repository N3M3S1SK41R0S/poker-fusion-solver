"""Tests du lecteur PHH — ancrés sur une main dont le déroulé est public.

Le golden est **Dwan-Ivey 2009**, le premier pot télévisé à un million de
dollars (Full Tilt Million Dollar Cash Game S4E12). Son déroulé est vérifiable
à la main, jeton par jeton, depuis le fichier ``dwan-ivey-2009.phh`` :

    antes 500 × 3 · blindes 1000/2000 · tapis 1 125 600 / 2 000 000 / 553 500
    p3 monte à 7 000, p1 à 23 000, p2 passe, p3 suit
    flop Jc3d5c   p1 mise 35 000, p3 suit
    turn 4h       p1 mise 90 000, p3 monte à 232 600,
                  p1 met ses 977 100 restants (« monte à 1 067 100 »),
                  p3 suit pour ses 262 400 derniers jetons

D'où, à la main :

* Ivey engage 500 + 1 000 + 22 000 + 35 000 + 90 000 + 977 100 = 1 125 600,
  soit exactement son tapis ;
* Dwan engage 500 + 7 000 + 16 000 + 35 000 + 232 600 + 262 400 = 553 500,
  soit exactement le sien ;
* Antonius laisse son ante et sa big blind, 2 500 ;
* Ivey a surmisé de 1 067 100 − 495 000 = **572 100** que personne ne pouvait
  payer, et qui lui reviennent ;
* le pot vaut donc 1 125 600 + 553 500 + 2 500 − 572 100 = **1 109 500**.

Chaque test ci-dessous a été vérifié TOMBANT. Ce n'est pas une intention :
quatorze mutations du lecteur ont été jouées une à une, et **aucune ne
survit** à ce fichier —

    cbr lu comme un incrément · rendu de la surmise supprimé · pot gagnable
    non plafonné · rejeu montrant le board final · préflop non ouverte par la
    big blind · all-in non détecté · bouton heads-up décalé · règle du couché
    désactivée · to_call non plafonné par le tapis · ante comptée dans la
    baseline de la rue · cartes espacées tronquées au premier jeton ·
    variante non vérifiée · position inventée sur une table inconnue ·
    compteur de relances non remis à zéro par rue

— chacune fait échouer au moins un test nommé dans la note qui l'annonce.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from pfs.data.hand_history import (
    ActionType,
    Room,
    Street,
    detect_room,
    parse_text,
    player_key,
)
from pfs.data.phh import (
    Decision,
    PhhError,
    iter_decisions,
    parse_phh,
    parse_phh_file,
    phh_players,
    positions_par_siege,
)

CORPUS = Path(r"C:\Users\pierr\Documents\POKERFUSIONSOLVER\corpus\phh-dataset\data")
DWAN = CORPUS / "dwan-ivey-2009.phh"

#: Le golden est recopié ici plutôt que lu sur le disque : le corpus est un
#: dépôt externe, et un test qui se met en SKIP quand le corpus manque ne
#: prouve plus rien le jour où il manque.
DWAN_PHH = """
variant = "NT"
ante_trimming_status = true
antes = [500, 500, 500]
blinds_or_straddles = [1000, 2000, 0]
min_bet = 2000
starting_stacks = [1125600, 2000000, 553500]
actions = [
  "d dh p1 Ac2d",
  "d dh p2 ????",
  "d dh p3 7h6h",
  "p3 cbr 7000",
  "p1 cbr 23000",
  "p2 f",
  "p3 cc",
  "d db Jc3d5c",
  "p1 cbr 35000",
  "p3 cc",
  "d db 4h",
  "p1 cbr 90000",
  "p3 cbr 232600",
  "p1 cbr 1067100",
  "p3 cc",
  "p1 sm Ac2d",
  "p3 sm 7h6h",
  "d db Jh",
]
players = ["Phil Ivey", "Patrik Antonius", "Tom Dwan"]
currency = "USD"
"""

IVEY, ANTONIUS, DWAN_J = (player_key(n) for n in
                          ("Phil Ivey", "Patrik Antonius", "Tom Dwan"))


class TestDwanIvey(unittest.TestCase):
    """Le déroulé public, jeton par jeton."""

    def setUp(self) -> None:
        self.h = parse_phh(DWAN_PHH, hero="Tom Dwan", hand_id="dwan-ivey-2009")

    def test_entete(self) -> None:
        self.assertIs(self.h.room, Room.PHH)
        self.assertEqual(self.h.big_blind, 2000.0)
        self.assertEqual(self.h.ante, 500.0)
        self.assertEqual(len(self.h.seats), 3)
        self.assertTrue(self.h.is_real_money)     # clé « currency » présente
        self.assertFalse(self.h.is_tournament)    # PHH ne le déclare pas

    def test_cartes_de_tous_les_joueurs(self) -> None:
        """PHH donne les cartes des couchés — ce qu'aucun HH de room ne fait.

        Tombe si ``d dh`` n'alimentait que le héros : Ivey s'est couché… non,
        Antonius s'est couché, et ses cartes sont inconnues (« ???? ») donc
        vides ; Ivey, lui, doit avoir Ac2d bien qu'il ne soit pas le héros.
        """
        par_siege = {s.seat: s.cards for s in self.h.seats}
        self.assertEqual(par_siege[1], ("Ac", "2d"))    # Ivey, pas le héros
        self.assertEqual(par_siege[2], ())              # « ???? » → rien
        self.assertEqual(par_siege[3], ("7h", "6h"))
        self.assertEqual(self.h.hero_cards, ("7h", "6h"))

    def test_board_complet_dans_l_ordre(self) -> None:
        self.assertEqual(self.h.board, ("Jc", "3d", "5c", "4h", "Jh"))

    def test_cbr_est_un_montant_cumule_pas_un_increment(self) -> None:
        """Le cœur du format : « cbr 23000 » = « monte à 23 000 ».

        Ivey avait déjà 1 000 de small blind : l'incrément vaut 22 000. Lire
        ``cbr`` comme un incrément donnerait 23 000 ici — c'est la mutation
        qui fait tomber ce test, et avec lui toute la comptabilité.
        """
        relances = [a for a in self.h.actions
                    if a.player == IVEY and a.street is Street.PREFLOP
                    and a.action is ActionType.RAISE]
        self.assertEqual(len(relances), 1)
        self.assertEqual(relances[0].amount, 22000.0)     # incrément payé
        self.assertEqual(relances[0].total_bet, 23000.0)  # cumul de la rue

    def test_engagement_de_chaque_joueur(self) -> None:
        """Chacun engage exactement le montant calculé à la main plus haut."""
        engage: dict[str, float] = {}
        for a in self.h.actions:
            engage[a.player] = engage.get(a.player, 0.0) + a.amount
        self.assertEqual(engage[IVEY], 1125600.0)      # tapis entier
        self.assertEqual(engage[DWAN_J], 553500.0)     # tapis entier
        self.assertEqual(engage[ANTONIUS], 2500.0)     # ante + big blind

    def test_pot_ampute_de_la_surmise_non_payable(self) -> None:
        """1 109 500, pas 1 681 600.

        Ivey monte à 1 067 100 alors que Dwan ne peut mettre que 495 000 sur
        la rue : 572 100 lui reviennent. Retirer l'appel à
        ``_rendre_non_paye`` gonfle le pot de ces 572 100 — c'est la mutation
        vérifiée.
        """
        self.assertEqual(self.h.pot, 1109500.0)

    def test_l_allin_est_l_action_agressive_qui_vide_le_tapis(self) -> None:
        """Ivey passe ALLIN au turn ; Dwan, qui SUIT à tapis, reste CALL.

        Même convention que ``parse_ipoker`` : un suivi qui met à tapis n'est
        pas une relance, et le compter comme telle fausserait PFR et 3-bet.
        """
        turn = [a for a in self.h.actions if a.street is Street.TURN]
        self.assertEqual([(a.player == IVEY, a.action) for a in turn], [
            (True, ActionType.BET),
            (False, ActionType.RAISE),
            (True, ActionType.ALLIN),
            (False, ActionType.CALL),
        ])

    def test_positions_trois_joueurs(self) -> None:
        self.assertEqual(positions_par_siege(self.h), {1: "SB", 2: "BB", 3: "BTN"})

    def test_premiere_action_preflop_est_une_relance_pas_un_bet(self) -> None:
        """La big blind ouvre la rue : Dwan RELANCE à 7 000.

        Si la rue préflop n'était pas marquée ouverte, ce serait un BET et
        ``preflop_raise`` (la stat PFR) manquerait tous les open-raises.
        """
        prem = next(a for a in self.h.actions
                    if a.street is Street.PREFLOP
                    and a.action is not ActionType.POST)
        self.assertIs(prem.action, ActionType.RAISE)
        self.assertTrue(self.h.preflop_raise(DWAN_J))

    def test_showdown_et_stats(self) -> None:
        self.assertTrue(self.h.went_to_showdown(DWAN_J))
        self.assertFalse(self.h.went_to_showdown(ANTONIUS))
        self.assertTrue(self.h.voluntarily_put_in_pot(IVEY))
        self.assertFalse(self.h.voluntarily_put_in_pot(ANTONIUS))  # blinde seule


class TestDecisionsDwanIvey(unittest.TestCase):
    """Le rejeu : pot, cote et tapis à chaque prise de parole."""

    def setUp(self) -> None:
        self.d: list[Decision] = list(iter_decisions(
            parse_phh(DWAN_PHH, hero="Tom Dwan", hand_id="dwan")))

    def test_dix_decisions_les_posts_n_en_sont_pas(self) -> None:
        """15 actions, dont 5 postes d'ante/blinde : 10 décisions."""
        self.assertEqual(len(self.d), 10)
        self.assertTrue(all(x.action is not ActionType.POST for x in self.d))

    def test_premier_pot_ce_sont_les_antes_et_les_blindes(self) -> None:
        """500×3 + 1 000 + 2 000 = 4 500, et Dwan doit 2 000 pour suivre."""
        p = self.d[0]
        self.assertEqual((p.player, p.position), (DWAN_J, "BTN"))
        self.assertEqual(p.pot, 4500.0)
        self.assertEqual(p.to_call, 2000.0)
        self.assertEqual(p.stack, 553000.0)      # tapis moins l'ante
        self.assertEqual(p.relances_avant, 0)
        self.assertEqual(p.actifs, 3)

    def test_le_board_est_celui_du_moment_pas_celui_de_la_fin(self) -> None:
        """Au flop on voit 3 cartes, au turn 4 — jamais les 5 de la main.

        Tombe si le rejeu recopiait ``hand.board`` : le conseiller
        calculerait alors l'équité au flop en connaissant la river.
        """
        vus = {(x.street, x.board) for x in self.d}
        self.assertIn((Street.PREFLOP, ()), vus)
        self.assertIn((Street.FLOP, ("Jc", "3d", "5c")), vus)
        self.assertIn((Street.TURN, ("Jc", "3d", "5c", "4h")), vus)
        self.assertTrue(all(len(x.board) < 5 for x in self.d))

    def test_dernier_call_le_pot_gagnable_est_plafonne(self) -> None:
        """Dwan risque 262 400 pour gagner 847 100, pas 1 419 200.

        Les 572 100 qu'Ivey a surmisés ne sont pas gagnables : ils lui
        reviendront. Confondre les deux donne une cote de 16 % au lieu de
        24 % — l'erreur qui fait payer ce qu'il fallait coucher.
        """
        p = self.d[-1]
        self.assertEqual((p.player, p.action), (DWAN_J, ActionType.CALL))
        self.assertEqual(p.pot, 1419200.0)
        self.assertEqual(p.pot_gagnable, 847100.0)
        self.assertEqual(p.to_call, 262400.0)
        # cote exacte : 262 400 / (847 100 + 262 400)
        self.assertAlmostEqual(p.to_call / (p.pot_gagnable + p.to_call),
                               0.2365029, places=6)

    def test_le_pot_gagnable_suit_le_pot_hors_spot_a_tapis(self) -> None:
        """Tant que personne n'est à tapis pour moins, les deux coïncident."""
        for p in self.d[:-1]:
            self.assertEqual(p.pot, p.pot_gagnable, msg=f"décision #{p.ordre}")

    def test_les_relances_de_la_rue_sont_comptees_par_rue(self) -> None:
        """Le compteur repart à zéro au flop : c'est ce qui distingue une
        ouverture d'une réponse à une relance."""
        flop = [x for x in self.d if x.street is Street.FLOP]
        self.assertEqual([x.relances_avant for x in flop], [0, 1])

    def test_tapis_effectif_borne_par_l_adversaire(self) -> None:
        """Antonius a 2 000 000 mais ne joue que 1 102 100 : Ivey l'a couvert."""
        p = self.d[2]
        self.assertEqual((p.player, p.action), (ANTONIUS, ActionType.FOLD))
        self.assertEqual(p.stack, 1997500.0)
        self.assertEqual(p.stack_effectif, 1102100.0)


class TestComptabiliteVerifieeParLesTapisDeFin(unittest.TestCase):
    """``finishing_stacks`` transforme le lecteur en son propre oracle."""

    BASE = """
variant = "NT"
antes = [0, 0]
blinds_or_straddles = [50, 100]
starting_stacks = [10000, 10000]
actions = ["d dh p1 AsAh", "d dh p2 7d2c", "p1 cbr 300", "p2 f"]
finishing_stacks = [10100, 9900]
"""

    def test_relance_non_payee_le_pot_est_de_200(self) -> None:
        """SB 50 + BB 100 + les 50 de complément : 200. La relance à 300 n'a
        jamais été suivie, 200 reviennent à p1 (qui gagne 100 net)."""
        h = parse_phh(self.BASE)
        self.assertEqual(h.pot, 200.0)
        self.assertEqual(h.winners, {player_key("p1"): 200.0})

    def test_tapis_de_fin_incoherents_refuses(self) -> None:
        """Un encaissement négatif = lecture fausse : on refuse la main."""
        faux = self.BASE.replace("[10100, 9900]", "[9000, 9900]")
        with self.assertRaises(PhhError) as ctx:
            parse_phh(faux)
        self.assertIn("comptabilité", str(ctx.exception))

    #: Main construite pour que les deux lectures possibles de ``cbr``
    #: divergent OBSERVABLEMENT — ce que ``BASE`` ne fait pas.
    #:
    #: Déroulé (blindes 50/100, trois joueurs) : p3 ouvre à 300, p1 passe,
    #: p2 (big blind, déjà 100 en jeu) monte à 900, p3 **paie**, puis p2 mise
    #: 500 au flop et p3 se couche.
    #:
    #: * lecture juste (« monte à ») : p2 ajoute 800, p3 ajoute 600 → chacun
    #:   900 préflop ; les 500 du flop reviennent à p2 ; pot 1 850 ;
    #: * lecture fausse (« ajoute ») : p2 arrive à 1 000, p3 doit suivre à
    #:   1 000 → p3 aurait engagé 1 000 au lieu de 900.
    #:
    #: Quand tout le monde passe, le rendu de la surmise gomme cet écart et
    #: les deux lectures donnent le même pot. Ici p3 PAIE puis se couche :
    #: son encaissement passe de 0 (juste) à +100 (faux), et la règle « un
    #: couché n'encaisse rien » le voit.
    RELANCE_PAYEE = """
variant = "NT"
antes = [0, 0, 0]
blinds_or_straddles = [50, 100, 0]
starting_stacks = [10000, 10000, 10000]
actions = [
  "d dh p1 2c3d", "d dh p2 AsAh", "d dh p3 KsKd",
  "p3 cbr 300", "p1 f", "p2 cbr 900", "p3 cc",
  "d db 7c8d9h",
  "p2 cbr 500", "p3 f",
]
finishing_stacks = [9950, 10950, 9100]
"""

    def test_relance_payee_puis_fold_le_pot_est_de_1850(self) -> None:
        h = parse_phh(self.RELANCE_PAYEE)
        self.assertEqual(h.pot, 1850.0)
        self.assertEqual(h.winners, {player_key("p2"): 1850.0})

    def test_le_garde_fou_tombe_si_cbr_est_lu_comme_un_increment(self) -> None:
        """Preuve que le garde-fou n'est pas tautologique.

        Les tapis de fin sont remplacés par ceux qu'une lecture « incrément »
        rendrait cohérents (p3 engage 1 000, p2 gagne 2 050). Le lecteur
        correct doit refuser ce fichier : p3 s'est couché après avoir engagé
        900, il ne peut pas finir avec 100 de plus que prévu.
        """
        incoherent = self.RELANCE_PAYEE.replace("[9950, 10950, 9100]",
                                                "[9950, 11050, 9000]")
        with self.assertRaises(PhhError) as ctx:
            parse_phh(incoherent)
        self.assertIn("COUCHÉ", str(ctx.exception))

    def test_un_couche_qui_encaisse_est_refuse_meme_en_positif(self) -> None:
        """La règle « un couché n'encaisse rien » a sa propre force.

        Le test précédent serait passé sans elle : l'encaissement y était
        négatif, ce que le premier garde-fou attrape déjà. Ici p3 finit
        100 jetons TROP HAUT — encaissement +100 alors qu'il s'est couché.
        Seule la règle du couché voit ce cas, et elle seule voit la lecture
        « cbr = incrément » sur les mains où la surmise est payée.
        """
        trop = self.RELANCE_PAYEE.replace("[9950, 10950, 9100]",
                                          "[9950, 10850, 9200]")
        with self.assertRaises(PhhError) as ctx:
            parse_phh(trop)
        self.assertIn("COUCHÉ", str(ctx.exception))


class TestRefusExplicites(unittest.TestCase):
    """Ce que le lecteur refuse — plutôt que de le lire de travers."""

    def test_variante_hors_holdem(self) -> None:
        with self.assertRaises(PhhError) as ctx:
            parse_phh('variant = "FB"\nantes = [0, 0]\n'
                      'blinds_or_straddles = [1, 2]\n'
                      'starting_stacks = [100, 100]\nactions = []\n')
        self.assertIn("FB", str(ctx.exception))

    def test_action_inconnue(self) -> None:
        """« sd » (stand pat, variantes de draw) n'a pas de sens en hold'em."""
        with self.assertRaises(PhhError) as ctx:
            parse_phh('variant = "NT"\nantes = [0, 0]\n'
                      'blinds_or_straddles = [1, 2]\n'
                      'starting_stacks = [100, 100]\nactions = ["p1 sd"]\n')
        self.assertIn("sd", str(ctx.exception))

    def test_tableaux_de_longueurs_incoherentes(self) -> None:
        with self.assertRaises(PhhError):
            parse_phh('variant = "NT"\nantes = [0, 0, 0]\n'
                      'blinds_or_straddles = [1, 2]\n'
                      'starting_stacks = [100, 100]\nactions = []\n')

    def test_heros_absent_de_la_table(self) -> None:
        with self.assertRaises(PhhError):
            parse_phh(DWAN_PHH, hero="Daniel Negreanu")

    def test_toml_illisible(self) -> None:
        with self.assertRaises(PhhError):
            parse_phh("ceci n'est pas du TOML [[[")

    def test_cartes_espacees_ne_sont_pas_perdues(self) -> None:
        """« d db Jc 3d 5c » vaut « d db Jc3d5c ».

        Ne lire que le premier jeton perdrait deux cartes du flop en
        silence — et l'équité serait calculée sur un board tronqué, sans
        qu'aucune exception ne prévienne.
        """
        compact = parse_phh(DWAN_PHH)
        espace = parse_phh(DWAN_PHH
                           .replace('"d db Jc3d5c"', '"d db Jc 3d 5c"')
                           .replace('"d dh p1 Ac2d"', '"d dh p1 Ac 2d"'))
        self.assertEqual(espace.board, compact.board)
        self.assertEqual([s.cards for s in espace.seats],
                         [s.cards for s in compact.seats])


class TestIntegrationDispatch(unittest.TestCase):
    """Le PHH entre par la porte commune des hand histories."""

    def test_detection(self) -> None:
        self.assertIs(detect_room(DWAN_PHH), Room.PHH)

    def test_les_autres_formats_ne_sont_pas_confondus(self) -> None:
        """Un fichier TOML quelconque n'est pas une main : il faut « actions ».

        Sans la double signature, tout ``pyproject.toml`` du disque
        deviendrait une main de poker.
        """
        self.assertIs(detect_room('variant = "NT"\nfoo = 1\n'), Room.UNKNOWN)
        self.assertIs(detect_room("blah"), Room.UNKNOWN)

    def test_parse_text_aiguille(self) -> None:
        h = parse_text(DWAN_PHH)
        self.assertIs(h.room, Room.PHH)
        self.assertEqual(h.pot, 1109500.0)


class TestPositions(unittest.TestCase):
    """Les noms doivent être ceux des charts du conseiller (GTO_PRESETS)."""

    def _table(self, n: int) -> dict[int, str]:
        h = parse_phh(
            'variant = "NT"\n'
            f"antes = [{', '.join('0' for _ in range(n))}]\n"
            f"blinds_or_straddles = [50, 100"
            f"{''.join(', 0' for _ in range(n - 2))}]\n"
            f"starting_stacks = [{', '.join('10000' for _ in range(n))}]\n"
            "actions = []\n")
        return positions_par_siege(h)

    def test_six_max_couvre_les_charts(self) -> None:
        from pfs.core.range_model import GTO_PRESETS
        t = self._table(6)
        self.assertEqual(t, {1: "SB", 2: "BB", 3: "UTG", 4: "MP",
                            5: "CO", 6: "BTN"})
        # toutes sauf la big blind ont une chart d'ouverture
        self.assertEqual({p for p in t.values() if p not in GTO_PRESETS}, {"BB"})

    def test_heads_up_le_bouton_est_la_small_blind(self) -> None:
        self.assertEqual(self._table(2), {1: "SB", 2: "BB"})

    def test_table_inconnue_rend_vide_plutot_qu_un_nom_invente(self) -> None:
        """À dix joueurs, aucune convention n'est posée : ne rien dire.

        Un nom inventé enverrait le conseiller chercher une chart qui ne
        correspond pas à la position réelle.
        """
        self.assertEqual(self._table(10), {})


class TestCorpusReel(unittest.TestCase):
    """Contrôles sur le corpus téléchargé, s'il est présent."""

    def setUp(self) -> None:
        if not DWAN.exists():
            self.skipTest(f"corpus PHH absent ({CORPUS})")

    def test_fichier_disque_identique_au_golden_recopie(self) -> None:
        """Le golden en dur ci-dessus dit bien ce que le fichier dit.

        Sans ce test, une divergence entre le corpus et la copie rendrait
        toute la classe précédente creuse.
        """
        disque = parse_phh_file(DWAN, hero="Tom Dwan")
        copie = parse_phh(DWAN_PHH, hero="Tom Dwan")
        self.assertEqual(disque.pot, copie.pot)
        self.assertEqual(disque.board, copie.board)
        self.assertEqual([(a.player, a.action, a.amount) for a in disque.actions],
                         [(a.player, a.action, a.amount) for a in copie.actions])
        self.assertEqual(phh_players(DWAN.read_text(encoding="utf-8")),
                         ("Phil Ivey", "Patrik Antonius", "Tom Dwan"))

    def test_main_pluribus_verifiee_a_la_main(self) -> None:
        """``pluribus/100/30.phh`` : relance à 737, tout le monde passe.

        Pluribus (p6, bouton) monte à 737 face à l'ouverture à 250 de
        MrBrown (p5). Les quatre autres passent. Pluribus récupère
        737 − 250 = 487 non payés ; il encaisse 50 + 100 + 250 + 250 = 650 et
        finit à 10 400, soit +400 — exactement ce qu'annonce le fichier.
        """
        f = CORPUS / "pluribus" / "100" / "30.phh"
        if not f.exists():
            self.skipTest("corpus Pluribus absent")
        h = parse_phh_file(f, hero="Pluribus")
        self.assertEqual(h.pot, 650.0)
        self.assertEqual(h.winners, {player_key("Pluribus"): 650.0})
        self.assertEqual(h.hero_cards, ("Tc", "As"))
        d = list(iter_decisions(h))
        self.assertEqual([x.position for x in d],
                         ["UTG", "MP", "CO", "BTN", "SB", "BB", "CO"])
        pluribus = next(x for x in d if x.position == "BTN")
        self.assertEqual(pluribus.to_call, 250.0)
        self.assertEqual(pluribus.pot, 400.0)      # 50 + 100 + 250
        self.assertEqual(pluribus.relances_avant, 1)


if __name__ == "__main__":
    unittest.main()
