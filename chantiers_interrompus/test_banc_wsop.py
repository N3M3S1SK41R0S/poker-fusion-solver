"""Tests du banc WSOP — la machinerie qui décide « erreur prouvable » ou non.

Ce banc produit une phrase lourde de conséquences : « le professionnel s'est
trompé, et voici de combien ». Chaque endroit où une erreur silencieuse
transformerait un jeu correct en accusation — ou l'inverse — est épinglé ici :

1. **la cote** (:func:`alpha_requise`) : la confondre avec ``b/(P+2b)``
   sous-estime l'équité requise et innocente des suivis à tort ;
2. **les bornes d'équité** (:func:`equites_par_combo`), exactes par
   énumération : ce sont elles, et rien d'autre, qui portent la preuve ;
3. **la condition de fermeture** (:func:`ferme_le_coup`) : sans elle, un
   couché serait « réfuté » alors que les rues suivantes n'ont pas été
   comptées ;
4. **le classement** (:func:`prouver`) : chaque critère doit se déclencher
   quand il faut ET se taire quand il ne faut pas ;
5. **le sens de la tolérance préflop** : appliquée à l'envers, elle
   accuserait plus facilement au lieu de protéger ;
6. **le pot transmis au conseiller** (:func:`construire_spot`) ;
7. **la lecture des tapis** (:func:`tapis_avant_chaque_decision`), qui doit
   coïncider avec celle de ``iter_decisions`` sous peine de rendre le volet
   ICM faux sans prévenir.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import banc_wsop as banc  # noqa: E402
from pfs.analysis.spot_advisor import parse_cards  # noqa: E402
from pfs.core.bluffcatch import required_equity  # noqa: E402
from pfs.core.icm import icm_required_equity  # noqa: E402
from pfs.data.hand_history import ActionType, Street  # noqa: E402
from pfs.data.phh import Decision, iter_decisions, parse_phh  # noqa: E402

CORPUS = Path(
    r"C:\Users\pierr\Documents\POKERFUSIONSOLVER\corpus\phh-dataset\data\wsop\2023")


def _d(**kw) -> Decision:
    """Décision synthétique, avec des valeurs par défaut neutres."""
    base = dict(
        hand_id="t", ordre=0, street=Street.RIVER, player="p", seat=1,
        position="BB", cards=(), board=(), pot=1000.0, pot_gagnable=1000.0,
        to_call=100.0, stack=5000.0, stack_effectif=5000.0, engage=0.0,
        actifs=2, relances_avant=1, action=ActionType.CALL, montant=100.0)
    base.update(kw)
    return Decision(**base)  # type: ignore[arg-type]


class TestCote(unittest.TestCase):
    """α = c/(P+c), et la borne ½ dont dépend tout le raisonnement sur A1."""

    def test_valeur_a_la_main(self) -> None:
        """Payer 75 dans un pot gagnable de 175 : 75/250 = 0,30."""
        self.assertAlmostEqual(banc.alpha_requise(175.0, 75.0), 0.30, places=12)

    def test_ce_n_est_pas_la_formule_de_bluffcatch(self) -> None:
        """``required_equity`` attend le pot AVANT la mise ; leur donner le
        même nombre donne deux réponses, et c'est exactement le piège.

        Sur ce spot, la confusion ferait passer l'équité requise de 30 % à
        23,1 % — de quoi innocenter un suivi perdant de 7 points.
        """
        self.assertAlmostEqual(banc.alpha_requise(175.0, 75.0), 0.30, places=12)
        self.assertAlmostEqual(required_equity(175.0, 75.0), 0.2307692, places=6)
        self.assertNotAlmostEqual(banc.alpha_requise(175.0, 75.0),
                                  required_equity(175.0, 75.0), places=3)

    def test_les_deux_coincident_avec_le_bon_pot(self) -> None:
        """C'est la garantie que :func:`construire_spot` peut être juste :
        ``required_equity(P − c, c)`` redonne bien ``alpha_requise(P, c)``."""
        for pot, c in ((250.0, 75.0), (1000.0, 900.0), (2825.0, 2200.0)):
            self.assertAlmostEqual(banc.alpha_requise(pot, c),
                                   required_equity(pot - c, c), places=12)

    def test_la_cote_ne_depasse_jamais_un_demi(self) -> None:
        """Fait structurel qui rend A1 inatteignable : le pot gagnable vaut au
        moins ce qu'il reste à payer, donc α ≤ ½. Le cas d'égalité est la
        limite, jamais atteinte à une vraie table."""
        self.assertAlmostEqual(banc.alpha_requise(75.0, 75.0), 0.5, places=12)
        for pot in (75.0, 100.0, 500.0, 1e6):
            self.assertLessEqual(banc.alpha_requise(pot, 75.0), 0.5 + 1e-12)

    def test_refus_des_entrees_absurdes(self) -> None:
        with self.assertRaises(ValueError):
            banc.alpha_requise(100.0, 0.0)
        with self.assertRaises(ValueError):
            banc.alpha_requise(-1.0, 10.0)


class TestEvSuivre(unittest.TestCase):
    """EV(suivre) − EV(coucher). Un signe inversé retournerait chaque verdict."""

    def test_valeur_a_la_main(self) -> None:
        """e = ½, P = 175, c = 75 : 0,5×175 − 0,5×75 = 50."""
        self.assertAlmostEqual(banc.ev_suivre(0.5, 175.0, 75.0), 50.0, places=12)

    def test_s_annule_exactement_a_la_cote(self) -> None:
        """Le zéro de l'EV EST la cote : les deux formules doivent se croiser
        au même point, sinon le banc accuserait dans une bande fantôme."""
        for pot, c in ((1000.0, 100.0), (2825.0, 2200.0), (375.0, 175.0)):
            a = banc.alpha_requise(pot, c)
            self.assertAlmostEqual(banc.ev_suivre(a, pot, c), 0.0, places=9)
            self.assertGreater(banc.ev_suivre(a + 1e-3, pot, c), 0.0)
            self.assertLess(banc.ev_suivre(a - 1e-3, pot, c), 0.0)


class TestFermeture(unittest.TestCase):
    """Sans cette condition, un couché serait « réfuté » à crédit."""

    def test_suivi_a_tapis(self) -> None:
        self.assertTrue(banc.ferme_le_coup(
            _d(street=Street.FLOP, actifs=3, to_call=500.0, stack=500.0)))

    def test_plus_personne_n_a_de_jetons_derriere(self) -> None:
        self.assertTrue(banc.ferme_le_coup(
            _d(street=Street.FLOP, actifs=3, to_call=100.0, stack=5000.0,
               stack_effectif=0.0)))

    def test_river_en_tete_a_tete(self) -> None:
        self.assertTrue(banc.ferme_le_coup(
            _d(street=Street.RIVER, actifs=2, to_call=100.0)))

    def test_river_multiway_ne_ferme_pas(self) -> None:
        """Un troisième joueur peut encore relancer derrière : le coup n'est
        pas clos, et le banc doit refuser de conclure."""
        self.assertFalse(banc.ferme_le_coup(
            _d(street=Street.RIVER, actifs=3, to_call=100.0)))

    def test_flop_profond_ne_ferme_pas(self) -> None:
        self.assertFalse(banc.ferme_le_coup(
            _d(street=Street.FLOP, actifs=2, to_call=100.0, stack=5000.0,
               stack_effectif=5000.0)))


class TestBornesExactes(unittest.TestCase):
    """Les bornes portent la preuve : elles doivent être exactes, pas proches."""

    def test_quatre_as_battent_tout(self) -> None:
        """A♥A♦ sur A♣ A♠ K♦ 7♣ 2♥ : carré d'as, aucune quinte flush possible.
        Les 990 mains adverses valent toutes 0 pour l'adversaire."""
        combos, eq = banc.equites_par_combo(parse_cards("Ah Ad"),
                                            parse_cards("Ac As Kd 7c 2h"))
        self.assertEqual(int(combos.shape[0]), 990)
        self.assertEqual(float(eq.min()), 1.0)
        self.assertEqual(float(eq.max()), 1.0)

    def test_quinte_royale_au_tableau_partage_avec_tout_le_monde(self) -> None:
        """A♠K♠Q♠J♠T♠ : personne ne peut faire mieux, tout le monde partage.
        C'est le cas qui montre que la borne HAUTE ne descend pas sous ½."""
        _, eq = banc.equites_par_combo(parse_cards("3d 2c"),
                                       parse_cards("As Ks Qs Js Ts"))
        self.assertEqual(float(eq.min()), 0.5)
        self.assertEqual(float(eq.max()), 0.5)

    def test_quinte_maximale_perd_contre_personne_et_partage_avec_elle_meme(self) -> None:
        """6♦5♣ sur 2♣ 7♦ 9♠ 3♥ 4♣ : quinte 7-6-5-4-3, la plus haute possible
        (aucune couleur : deux trèfles au tableau seulement). Un adversaire à
        6-5 partage, tous les autres perdent."""
        _, eq = banc.equites_par_combo(parse_cards("6d 5c"),
                                       parse_cards("2c 7d 9s 3h 4c"))
        self.assertEqual(float(eq.min()), 0.5)
        self.assertEqual(float(eq.max()), 1.0)

    def test_nombre_de_mains_adverses_par_rue(self) -> None:
        """C(47,2) au flop, C(46,2) au turn, C(45,2) à la river. Un compte
        faux fausserait la moyenne uniforme, donc le critère A3."""
        for board, attendu in ((" ".join(["2c", "7d", "9s"]), 1081),
                               ("2c 7d 9s 3h", 1035),
                               ("2c 7d 9s 3h 4c", 990)):
            combos, _ = banc.equites_par_combo(parse_cards("Ah Kd"),
                                               parse_cards(board))
            self.assertEqual(int(combos.shape[0]), attendu, board)

    def test_l_uniforme_est_la_moyenne_des_mains_possibles(self) -> None:
        """``Bornes.uniforme`` doit être la moyenne simple, pas une pondération
        maison : c'est ce qui autorise à l'appeler « range la plus large »."""
        _, eq = banc.equites_par_combo(parse_cards("Ah Kd"),
                                       parse_cards("2c 7d 9s 3h 4c"))
        b = banc.bornes_equite(parse_cards("Ah Kd"), parse_cards("2c 7d 9s 3h 4c"))
        self.assertAlmostEqual(b.uniforme, float(eq.mean()), places=12)
        self.assertAlmostEqual(b.basse, float(eq.min()), places=12)
        self.assertAlmostEqual(b.haute, float(eq.max()), places=12)
        self.assertTrue(b.exacte)
        self.assertEqual(b.tolerance, 0.0)

    def test_preflop_refuse_l_enumeration(self) -> None:
        """Préflop, l'énumération coûterait 1,7 million de boards PAR main
        adverse : la fonction doit refuser plutôt que de ramer une heure."""
        with self.assertRaises(ValueError):
            banc.equites_par_combo(parse_cards("Ah Kd"), [])


class TestEquiteVsMainConnue(unittest.TestCase):
    """Le chiffre du recul (catégorie B) doit être exact, lui aussi."""

    def test_valeur_a_la_main(self) -> None:
        """A♥A♦ contre K♥K♦ sur 2♣ 7♦ 9♠ 3♥, une carte à venir : le héros ne
        perd que si un roi tombe (2 sur 44 cartes restantes), et aucune quinte
        ni couleur n'est possible. Équité = 42/44."""
        eq = banc.equite_vs_main(parse_cards("Ah Ad"), parse_cards("Kh Kd"),
                                 parse_cards("2c 7d 9s 3h"))
        self.assertAlmostEqual(eq, 42.0 / 44.0, places=12)

    def test_le_recul_utilise_les_cartes_reellement_tombees(self) -> None:
        """Avec la river connue, il ne reste plus rien à énumérer : l'équité
        vaut 0 ou 1. C'est ce qui fait de la catégorie B un chiffre de recul
        et non une espérance."""
        eq = banc.equite_vs_main(parse_cards("Ah Ad"), parse_cards("Kh Kd"),
                                 parse_cards("2c 7d 9s 3h"),
                                 futures=tuple(parse_cards("Kc")))
        self.assertEqual(eq, 0.0)
        eq = banc.equite_vs_main(parse_cards("Ah Ad"), parse_cards("Kh Kd"),
                                 parse_cards("2c 7d 9s 3h"),
                                 futures=tuple(parse_cards("4c")))
        self.assertEqual(eq, 1.0)

    def test_collision_refusee(self) -> None:
        with self.assertRaises(ValueError):
            banc.equite_vs_main(parse_cards("Ah Ad"), parse_cards("Ah Kd"),
                                parse_cards("2c 7d 9s"))


class TestProuver(unittest.TestCase):
    """Chaque critère doit crier quand il faut, et se taire sinon."""

    def test_les_cas_de_demonstration_rendent_ce_qu_ils_annoncent(self) -> None:
        """Le rapport publie ces six spots comme preuve que les détecteurs
        sont vivants. Si l'un d'eux cessait de rendre son critère, le rapport
        mentirait — d'où ce test, qui lit la MÊME table que le rapport."""
        for titre, attendu, preuve in banc.demonstration():
            obtenu = preuve.critere if preuve is not None else ""
            self.assertEqual(obtenu, attendu, titre)

    def test_a3_se_tait_quand_l_equite_repasse_au_dessus_de_la_cote(self) -> None:
        """Même main, même board, cote abaissée : le détecteur doit lâcher.

        7♣2♦ sur A♥K♦Q♣J♠ vaut ≈ 20 % contre une range uniforme. À 900 dans
        1 000 (47,4 % requis) le suivi est réfuté ; à 100 dans 1 000 (9,1 %
        requis) il ne l'est plus.
        """
        b = banc.bornes_equite(parse_cards("7c 2d"), parse_cards("Ah Kd Qc Js"))
        cher = _d(street=Street.TURN, cards=("7c", "2d"),
                  board=("Ah", "Kd", "Qc", "Js"), pot_gagnable=1000.0,
                  to_call=900.0, montant=900.0)
        pas_cher = _d(street=Street.TURN, cards=("7c", "2d"),
                      board=("Ah", "Kd", "Qc", "Js"), pot_gagnable=1000.0,
                      to_call=100.0, montant=100.0)
        self.assertEqual(banc.prouver(cher, b, 100.0).critere, "A3")
        self.assertIsNone(banc.prouver(pas_cher, b, 100.0))

    def test_a2_exige_que_le_coup_soit_ferme(self) -> None:
        """Coucher la quinte maximale pour 1 jeton est réfutable — mais
        seulement si le suivi ferme le coup. Le même couché au flop, tapis
        profonds, ne doit RIEN déclencher : l'adversaire pourrait encore
        miser deux rues."""
        b = banc.bornes_equite(parse_cards("6d 5c"), parse_cards("2c 7d 9s 3h 4c"))
        clos = _d(action=ActionType.FOLD, montant=0.0, cards=("6d", "5c"),
                  board=("2c", "7d", "9s", "3h", "4c"), pot_gagnable=10001.0,
                  to_call=1.0, stack=1.0, stack_effectif=1.0)
        self.assertEqual(banc.prouver(clos, b, 100.0).critere, "A2")
        ouvert = _d(action=ActionType.FOLD, montant=0.0, cards=("6d", "5c"),
                    board=("2c", "7d", "9s", "3h", "4c"), pot_gagnable=10001.0,
                    to_call=1.0, stack=99999.0, stack_effectif=99999.0,
                    actifs=3)
        self.assertIsNone(banc.prouver(ouvert, b, 100.0))

    def test_a4_ne_depend_d_aucune_borne(self) -> None:
        """Se coucher gratuitement est dominé quoi qu'il arrive : le critère
        doit conclure même sans bornes d'équité."""
        gratuit = _d(action=ActionType.FOLD, montant=0.0, to_call=0.0,
                     cards=("Ah", "Kd"), board=("2c", "7d", "9s"),
                     street=Street.FLOP, pot_gagnable=500.0)
        p = banc.prouver(gratuit, None, 100.0)
        self.assertIsNotNone(p)
        self.assertEqual(p.critere, "A4")
        self.assertEqual(p.action_correcte, "CHECKER")

    def test_a1_prime_sur_a3(self) -> None:
        """Un spot réfuté par les deux doit être classé sous le critère le plus
        fort : sinon le rapport annoncerait une hypothèse là où il n'y en a
        pas."""
        b = banc.bornes_equite(parse_cards("3d 2c"), parse_cards("As Ks Qs Js Ts"))
        d = _d(cards=("3d", "2c"), board=("As", "Ks", "Qs", "Js", "Ts"),
               pot_gagnable=100.0, to_call=900.0, montant=900.0)
        p = banc.prouver(d, b, 100.0)
        self.assertEqual(p.critere, "A1")
        self.assertEqual(p.hypothese, "")

    def test_seul_a3_porte_une_hypothese(self) -> None:
        """A1, A2 et A4 ne supposent rien ; A3 suppose une range non plus
        faible que le hasard, et doit le DIRE."""
        for _, attendu, preuve in banc.demonstration():
            if preuve is None:
                continue
            if attendu == "A3":
                self.assertTrue(preuve.hypothese)
            else:
                self.assertEqual(preuve.hypothese, "")

    def test_le_cout_est_l_ev_perdue(self) -> None:
        """Le coût publié doit être exactement |EV(suivre) − EV(coucher)|
        calculée avec la borne qui porte la preuve — pas une approximation."""
        b = banc.bornes_equite(parse_cards("7c 2d"), parse_cards("Ah Kd Qc Js"))
        d = _d(street=Street.TURN, cards=("7c", "2d"),
               board=("Ah", "Kd", "Qc", "Js"), pot_gagnable=1000.0,
               to_call=900.0, montant=900.0)
        p = banc.prouver(d, b, 100.0)
        attendu = -banc.ev_suivre(b.uniforme, 1000.0, 900.0)
        self.assertAlmostEqual(p.cout_jetons, attendu, places=9)
        self.assertAlmostEqual(p.cout_bb, attendu / 100.0, places=9)


class TestTolerancePreflop(unittest.TestCase):
    """La tolérance doit protéger le joueur, jamais l'accuser plus vite."""

    def test_sens_de_la_tolerance(self) -> None:
        """Borne haute relevée, borne basse abaissée, uniforme relevée : dans
        les trois cas, cela rend une accusation PLUS difficile. L'appliquer à
        l'envers ferait accuser à tort dans la bande de bruit Monte-Carlo."""
        b = banc.bornes_equite(parse_cards("7c 2d"), [])
        brute_b, brute_h, brute_u = b.brutes
        self.assertFalse(b.exacte)
        self.assertAlmostEqual(b.tolerance, banc.TOLERANCE_PREFLOP, places=12)
        self.assertAlmostEqual(b.haute, min(1.0, brute_h + b.tolerance), places=12)
        self.assertAlmostEqual(b.basse, max(0.0, brute_b - b.tolerance), places=12)
        self.assertAlmostEqual(b.uniforme, min(1.0, brute_u + b.tolerance),
                               places=12)
        self.assertGreater(b.haute, brute_h)
        self.assertLess(b.basse, brute_b)

    def test_la_tolerance_est_reellement_dissuasive(self) -> None:
        """Un suivi dont l'équité uniforme brute est juste sous la cote ne doit
        PAS être accusé si l'écart tient dans la tolérance."""
        b = banc.bornes_equite(parse_cards("7c 2d"), [])
        brute_u = b.brutes[2]
        # Cote calée 1 point au-dessus de l'équité brute : dans la tolérance.
        alpha_vise = brute_u + 0.01
        pot, c = 1000.0, alpha_vise * 1000.0 / (1.0 - alpha_vise)
        d = _d(street=Street.PREFLOP, cards=("7c", "2d"), board=(),
               pot_gagnable=pot + c, to_call=c, montant=c)
        self.assertAlmostEqual(banc.alpha_requise(pot + c, c), alpha_vise, places=9)
        self.assertIsNone(banc.prouver(d, b, 100.0))

    def test_hors_tolerance_le_detecteur_conclut_quand_meme(self) -> None:
        """Sinon la tolérance serait un bâillon, pas une marge."""
        b = banc.bornes_equite(parse_cards("7c 2d"), [])
        alpha_vise = b.brutes[2] + banc.TOLERANCE_PREFLOP + 0.05
        pot = 1000.0
        c = alpha_vise * pot / (1.0 - alpha_vise)
        d = _d(street=Street.PREFLOP, cards=("7c", "2d"), board=(),
               pot_gagnable=pot + c, to_call=c, montant=c)
        p = banc.prouver(d, b, 100.0)
        self.assertIsNotNone(p)
        self.assertEqual(p.critere, "A3")


class TestSeuilBubbleFactor(unittest.TestCase):
    """Le seuil publié sous chaque A2 doit inverser exactement le verdict."""

    def test_valeur_a_la_main(self) -> None:
        """P = 250 (dont 75 de mise), c = 75, e = ½ :
        BF* = 0,5 × 250 / (75 × 0,5) = 3,3333."""
        self.assertAlmostEqual(banc.bubble_factor_seuil(250.0, 75.0, 0.5),
                               10.0 / 3.0, places=9)

    def test_le_seuil_annule_la_marge(self) -> None:
        """Au seuil, l'équité requise ICM égale l'équité du héros : c'est la
        définition, et elle doit tenir contre le module ICM du projet."""
        for pot, c, e in ((250.0, 75.0, 0.5), (2825.0, 2200.0, 0.6),
                          (10001.0, 1.0, 0.5)):
            bf = banc.bubble_factor_seuil(pot, c, e)
            self.assertAlmostEqual(icm_required_equity(pot - c, c, bf), e,
                                   places=9)

    def test_equite_certaine_rend_le_couche_inexcusable(self) -> None:
        self.assertTrue(math.isinf(banc.bubble_factor_seuil(250.0, 75.0, 1.0)))


class TestDomaineDuConseiller(unittest.TestCase):
    """Un refus mal compté transformerait un manque du logiciel en désaccord."""

    def test_tous_les_motifs_sont_rendus_pas_seulement_le_premier(self) -> None:
        """Un préflop relancé FACE À UN TAPIS cumule deux violations. N'en
        rendre qu'une ferait disparaître « payer un jam » du tableau des
        motifs — c'est exactement ce que ce test protège."""
        d = _d(street=Street.PREFLOP, relances_avant=2, to_call=2200.0,
               stack=2200.0, stack_effectif=0.0)
        motifs = banc.hors_domaine(d)
        self.assertEqual(len(motifs), 2)
        self.assertTrue(any("relance" in m for m in motifs))
        self.assertTrue(any("tapis adverse" in m for m in motifs))

    def test_postflop_multiway_est_hors_domaine(self) -> None:
        d = _d(street=Street.FLOP, actifs=3, relances_avant=0)
        self.assertIn("multiway", " ".join(banc.hors_domaine(d)))

    def test_un_spot_couvert_ne_produit_aucun_motif(self) -> None:
        d = _d(street=Street.FLOP, actifs=2, relances_avant=0, to_call=100.0,
               stack=5000.0, stack_effectif=5000.0)
        self.assertEqual(banc.hors_domaine(d), ())


class TestSpotTransmisAuConseiller(unittest.TestCase):
    """Le conseiller doit recevoir le pot que son contrat annonce."""

    def test_le_pot_exclut_la_mise_adverse(self) -> None:
        """``Spot.pot`` est documenté « pot AVANT la mise adverse ». Lui passer
        le pot mise comprise ferait calculer au conseiller une équité requise
        plus faible que la vraie, et innocenterait des suivis perdants."""
        main = parse_phh(
            "variant = 'NT'\n"
            "antes = [0, 0]\n"
            "blinds_or_straddles = [50, 100]\n"
            "starting_stacks = [10000, 10000]\n"
            "actions = ['d dh p1 AsAh', 'd dh p2 7d2c', 'p1 cbr 300', 'p2 f']\n")
        d = _d(cards=("As", "Ah"), board=("2c", "7d", "9s", "3h", "4c"),
               pot_gagnable=1000.0, to_call=250.0)
        spot = banc.construire_spot(d, main, "moyenne")
        self.assertAlmostEqual(spot.pot, 750.0, places=12)
        self.assertAlmostEqual(spot.bet, 250.0, places=12)
        self.assertAlmostEqual(required_equity(spot.pot, spot.bet),
                               banc.alpha_requise(1000.0, 250.0), places=12)

    def test_le_solveur_river_est_desactive(self) -> None:
        """Un solve CFR par spot rendrait le banc non rejouable en temps et
        n'apporterait rien à la catégorie A, qui ne dépend pas du conseiller."""
        main = parse_phh(
            "variant = 'NT'\nantes = [0, 0]\nblinds_or_straddles = [50, 100]\n"
            "starting_stacks = [10000, 10000]\n"
            "actions = ['d dh p1 AsAh', 'd dh p2 7d2c', 'p1 cbr 300', 'p2 f']\n")
        spot = banc.construire_spot(_d(cards=("As", "Ah")), main, "moyenne")
        self.assertEqual(spot.solver_budget_s, 0.0)


class TestLectureDesTapis(unittest.TestCase):
    """Deux lectures des tapis coexistent : elles doivent coïncider."""

    MAIN = (
        "variant = 'NT'\n"
        "antes = [100, 100, 100]\n"
        "blinds_or_straddles = [50, 100, 0]\n"
        "starting_stacks = [3000, 5000, 4000]\n"
        "actions = ['d dh p1 AsAh', 'd dh p2 7d2c', 'd dh p3 KsKd',"
        " 'p3 cbr 300', 'p1 f', 'p2 cbr 900', 'p3 cc',"
        " 'd db 2h5s9c', 'p2 cbr 400', 'p3 f']\n")

    def test_le_tapis_du_joueur_qui_parle_coincide(self) -> None:
        """``tapis_avant_chaque_decision`` rejoue la comptabilité en parallèle
        de ``iter_decisions``. Si les deux divergeaient, le volet ICM du banc
        serait faux sans que rien ne le signale."""
        main = parse_phh(self.MAIN, is_tournament=True)
        tables = banc.tapis_avant_chaque_decision(main)
        decisions = list(iter_decisions(main))
        self.assertEqual(len(tables), len(decisions))
        for d, table in zip(decisions, tables, strict=True):
            self.assertAlmostEqual(table[d.player], d.stack, places=9,
                                   msg=f"décision #{d.ordre}")

    def test_les_antes_et_blindes_sont_debitees_avant_la_premiere_decision(self) -> None:
        """Sinon les tapis seraient ceux du début de main, pas ceux du moment."""
        main = parse_phh(self.MAIN, is_tournament=True)
        premiere = banc.tapis_avant_chaque_decision(main)[0]
        cles = [s.player for s in main.seats]
        self.assertAlmostEqual(premiere[cles[0]], 3000 - 100 - 50, places=9)
        self.assertAlmostEqual(premiere[cles[1]], 5000 - 100 - 100, places=9)
        self.assertAlmostEqual(premiere[cles[2]], 4000 - 100, places=9)


class TestValeurAvecCartesConnues(unittest.TestCase):
    """Le chiffre de la catégorie B, et les cas où il faut refuser de le donner."""

    def test_refuse_les_pots_multiway(self) -> None:
        """À trois, « la » main adverse n'existe pas : le banc doit rendre None
        plutôt qu'un chiffre construit sur un adversaire choisi au hasard."""
        main = parse_phh(TestLectureDesTapis.MAIN, is_tournament=True)
        d = _d(actifs=3, cards=("As", "Ah"), board=("2h", "5s", "9c"))
        self.assertIsNone(banc.valeur_avec_cartes_connues(d, main))

    def test_refuse_quand_il_n_y_a_rien_a_payer(self) -> None:
        main = parse_phh(TestLectureDesTapis.MAIN, is_tournament=True)
        d = _d(to_call=0.0, cards=("As", "Ah"))
        self.assertIsNone(banc.valeur_avec_cartes_connues(d, main))


class TestVariante(unittest.TestCase):
    """Le rapport annonce 11 no-limit et 7 limit : la lecture doit être juste."""

    def test_lit_la_variante_declaree(self) -> None:
        for code in ("NT", "FT"):
            main = parse_phh(
                f"variant = '{code}'\nantes = [0, 0]\n"
                "blinds_or_straddles = [50, 100]\n"
                "starting_stacks = [10000, 10000]\n"
                "actions = ['d dh p1 AsAh', 'd dh p2 7d2c', 'p1 cbr 300', 'p2 f']\n")
            self.assertEqual(banc.variante(main), code)


class TestAxeDAccord(unittest.TestCase):
    """Compter un accord pour un désaccord gonflerait la catégorie C."""

    def test_preflop_l_axe_est_jouer_ou_passer(self) -> None:
        """Le conseiller ne sait pas dire « suivre » : opposer son OUVRIR au
        limp du joueur mélangerait deux questions dont la seconde est hors
        modèle."""
        d = _d(street=Street.PREFLOP)
        self.assertTrue(banc.accord(d, "AGRESSER", "CONTINUER"))
        self.assertFalse(banc.accord(d, "AGRESSER", "PASSER"))
        self.assertTrue(banc.accord(d, "PASSER", "PASSER"))

    def test_sans_mise_l_axe_est_miser_ou_checker(self) -> None:
        d = _d(street=Street.FLOP, to_call=0.0)
        self.assertTrue(banc.accord(d, "AGRESSER", "AGRESSER"))
        self.assertFalse(banc.accord(d, "CHECK", "AGRESSER"))
        self.assertTrue(banc.accord(d, "CHECK", "CHECK"))

    def test_les_indecis_ne_tranchent_pas(self) -> None:
        d = _d(street=Street.PREFLOP)
        for vu in ("MIXTE", "INDECIS", "REFUS", "?bizarre"):
            self.assertIsNone(banc.accord(d, vu, "PASSER"), vu)


class TestBubbleFactorSurLeCorpus(unittest.TestCase):
    """Sans les gains, tout le volet tournoi repose sur « BF ≥ 1 »."""

    def test_le_minimum_vaut_un_sur_des_tapis_reels(self) -> None:
        """Winner-take-all rend les jetons linéaires (BF = 1 exactement) ; toute
        structure décroissante ne peut qu'ajouter de la pression. Si ce
        minimum passait sous 1, un suivi réfuté en jetons ne le serait plus
        forcément en $EV, et la catégorie A cesserait de tenir en tournoi."""
        tapis = [(7380000.0, 2500000.0, 5110000.0, 10170000.0, 4545000.0),
                 (2200000.0, 2575000.0, 3125000.0, 2375000.0, 19425000.0)]
        bmin, bmax, n = banc.bubble_factors_mesures(tapis)
        self.assertGreater(n, 0)
        self.assertGreaterEqual(bmin, 1.0 - 1e-9)
        self.assertGreater(bmax, 1.0)


@unittest.skipUnless(CORPUS.exists(), "corpus WSOP absent de cette machine")
class TestCorpusReel(unittest.TestCase):
    """Les chiffres de tête du rapport, épinglés sur le corpus réel."""

    def test_dix_huit_mains_de_holdem_sur_quatre_vingt_trois(self) -> None:
        """Le championnat est MIXTE : 65 mains ne sont pas du hold'em et sont
        refusées par le lecteur, pas mal lues."""
        from pfs.data.phh import PhhError, iter_phh_files, parse_phh_file

        lues = refusees = 0
        variantes: dict[str, int] = {}
        for f in iter_phh_files(CORPUS):
            try:
                main = parse_phh_file(f, is_tournament=True)
            except PhhError:
                refusees += 1
                continue
            lues += 1
            v = banc.variante(main)
            variantes[v] = variantes.get(v, 0) + 1
        self.assertEqual(lues, 18)
        self.assertEqual(refusees, 65)
        self.assertEqual(variantes, {"NT": 11, "FT": 7})


if __name__ == "__main__":
    unittest.main()
