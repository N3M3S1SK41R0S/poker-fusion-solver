"""Goldens du conseiller en tournoi (ICM) et du solve river.

Trois familles de vérités, toutes vérifiables :

1. **Linéarité.** Une structure winner-take-all rend les jetons linéaires —
   l'ICM doit y coïncider EXACTEMENT avec le chipEV. C'est vrai aussi d'une
   table à deux joueurs, quelle que soit la structure : le $EV de Harville y
   est affine en jetons, donc BF = 1 exactement.
2. **Sens de l'ICM.** Le seul universel qui survit à la mesure porte sur le
   **call** : sur les 36 configurations balayées par
   ``test_le_call_se_resserre_toujours_le_jam_non``, la range de call
   ICM ne dépasse jamais la range chipEV — zéro contre-exemple. Le **jam**
   n'a pas de sens fixe : il s'élargit dans 19 de ces 36 configurations, y
   compris héros couvert ([15, 16, 1] bb, gains 65/35 : 44,5 % → 60,8 %).
   Le golden 10/40/1 (58,7 % → 8,9 %) est un cas, pas une règle.
3. **Solveur river.** Le spot de polarisation nuts/air contre bluff-catchers
   a une solution analytique : dans la range de mise, la part de bluff vaut
   b/(P+2b) et le vilain défend 1 − b/(P+b) = MDF. Le conseiller doit la
   retrouver sur le combo du héros.
"""

from __future__ import annotations

import math
import time
import unittest

import numpy as np

from pfs.analysis.spot_advisor import (
    _DEFAULT_VILLAIN,
    _INDIFFERENCE_BB,
    _INDIFFERENCE_CAGNOTTE,
    _MIXTE_MIN,
    RIVER_SOLVE_ITERATIONS,
    Spot,
    advise,
    parse_cards,
)
from pfs.core.icm import bubble_factor, icm_equities
from pfs.core.range_model import Range, combo_index, parse_range
from pfs.solver.postflop import PostflopSolver
from pfs.solver.pushfold import (
    PushFoldSolution,
    equity_matrix_169,
    solve_hu_pushfold,
)

# Le héros est couvert (10 bb contre 40) et un joueur est à 1 bb : la bulle
# est immédiate. C'est le spot de Twister où le chipEV ment le plus.
BULLE = dict(stacks="10, 40, 1", payouts="65, 35", players=3)

# Balayage de référence du régime ICM : héros de 5 à 20 bb, tables de 3 et 4
# joueurs, deux structures de gains. Sert au sens de l'ICM et à la bande
# d'indifférence — les deux affirmations que la revue a prises en défaut.
_TAPIS_BALAYES = (
    (10, 40, 1), (10, 10, 1.5), (12, 12, 12), (15, 16, 1),
    (8, 30, 30), (20, 20, 20), (10, 25, 5), (5, 40, 40),
    (10, 40, 1, 1), (12, 30, 8), (10, 10, 10, 10), (6, 6, 25),
    (15, 15, 2), (9, 50, 3), (11, 11, 11), (20, 5, 5),
    (10, 40, 20), (14, 14, 7),
)
_GAINS_BALAYES = ((65.0, 35.0), (50.0, 30.0, 20.0))

# Plafond du solve de référence des tests de bande d'indifférence. Mesuré :
# porter ce plafond à 4000 ne change ni le nombre de bascules ni leur |EV|
# maximal sur les deux configurations épinglées ci-dessous.
_PLAFOND_REFERENCE = 1000


class TestIcmContreChipEV(unittest.TestCase):
    """Le conseiller ICM contre le même équilibre résolu en jetons."""

    def test_jam_chipev_devient_fold_icm(self) -> None:
        # K9o : +0,77 bb en chipEV à 10 bb, −0,17 unité de gain sous ICM.
        chip = advise(Spot(hero="Kh 9d", stack=10, big_blind=1))
        icm = advise(Spot(hero="Kh 9d", big_blind=1, **BULLE))
        self.assertTrue(chip.action.startswith("JAM"))
        self.assertEqual(icm.action, "FOLD")
        self.assertGreater(icm.ev_bb, 0.0)      # le chipEV du même spot
        self.assertLess(icm.ev_icm, 0.0)
        self.assertEqual(icm.hand, "K9o")

    def test_les_as_jamment_quand_meme(self) -> None:
        # L'ICM resserre, il ne paralyse pas : AA reste un jam.
        a = advise(Spot(hero="Ah Ad", big_blind=1, **BULLE))
        self.assertTrue(a.action.startswith("JAM"))
        self.assertGreater(a.ev_icm, 0.0)

    def test_bubble_factor_et_prime_declares(self) -> None:
        a = advise(Spot(hero="Kh 9d", big_blind=1, **BULLE))
        self.assertGreater(a.bubble, 1.0)       # mesuré : 5,02
        self.assertIn("ICM", a.regime)
        self.assertTrue(any("hipEV" in r for r in a.reasons))

    def test_a_trois_joueurs_le_verdict_est_indicatif(self) -> None:
        """Le solve suppose le troisième joueur couché : c'est une hypothèse.

        Le critère du module réserve « certain » aux verdicts qui ne
        supposent rien du jeu adverse. Ici le repli N-way de
        ``solver.pushfold`` prête une action au joueur à 1 bb — le label doit
        le refléter, pas seulement les hypothèses.
        """
        a = advise(Spot(hero="Kh 9d", big_blind=1, **BULLE))
        self.assertEqual(a.confidence, "indicatif")
        self.assertTrue(any("supposés déjà couchés" in x for x in a.assumptions))
        self.assertTrue(any("suppose les autres déjà couchés" in r
                            for r in a.reasons))

    def test_winner_take_all_identique_au_chipev(self) -> None:
        """Gains winner-take-all : $EV = payout × tapis/total, donc linéaire.

        Le bubble factor vaut alors exactement 1 et l'équilibre ICM est
        l'équilibre chipEV — c'est la propriété qui rend le Twister
        winner-take-all analysable en jetons sans rien perdre.
        """
        a = advise(Spot(hero="Kh 9d", stacks="10, 10, 5", payouts="100",
                        players=3, big_blind=1))
        self.assertAlmostEqual(a.bubble, 1.0, places=6)
        chip = advise(Spot(hero="Kh 9d", stack=10, big_blind=1))
        self.assertEqual(a.action, chip.action)
        self.assertAlmostEqual(a.ev_bb, chip.ev_bb, places=9)

    def test_ranges_dequilibre_se_resserrent(self) -> None:
        """Sur CE spot, les deux ranges se resserrent — et rien n'est ajouté.

        Le seuil de 0,5 est la convention de ``pushfold.chart`` : elle est
        nécessaire ici parce que les stratégies moyennes du fictitious play
        ne sont pas pures. Mesuré sur ces deux solves, la distance maximale à
        {0, 1} vaut 0 exactement en chipEV (équilibre pur, certificat n°1)
        mais 0,446 sur la range de jam ICM et 0,425 sur sa range de call :
        c'est le transitoire du moyennage, pas de la poussière.
        """
        e = equity_matrix_169()
        chip = solve_hu_pushfold(10.0, sb=0.5, bb=1.0, equity=e)
        icm = solve_hu_pushfold(10.0, payouts=[65.0, 35.0],
                                stacks=[10.0, 40.0, 1.0], sb=0.5, bb=1.0,
                                equity=e)
        self.assertLess(icm.call_pct, chip.call_pct)     # 6,2 % vs 37,6 %
        self.assertLess(icm.jam_pct, chip.jam_pct)       # 8,9 % vs 58,7 %
        for chp, ic in ((chip.jam_range, icm.jam_range),
                        (chip.call_range, icm.call_range)):
            ajoutes = np.nonzero((ic >= 0.5) & (chp < 0.5))[0]
            self.assertEqual(ajoutes.size, 0)
        # les chiffres de la docstring, épinglés
        pur = max(float(np.max(np.minimum(v, 1.0 - v)))
                  for v in (chip.jam_range, chip.call_range))
        self.assertEqual(pur, 0.0)
        self.assertAlmostEqual(
            float(np.max(np.minimum(icm.jam_range, 1.0 - icm.jam_range))),
            0.446, places=3)
        self.assertAlmostEqual(
            float(np.max(np.minimum(icm.call_range, 1.0 - icm.call_range))),
            0.425, places=3)

    def test_le_call_se_resserre_toujours_le_jam_non(self) -> None:
        """Le seul universel qui tient : l'ICM ne paie jamais plus large.

        Balayage de 36 configurations (18 tables × 2 structures de gains).
        Le call ICM ne dépasse jamais le call chipEV — zéro contre-exemple.
        Le jam, lui, s'élargit dans 19 cas sur 36 : la direction annoncée
        dans les versions précédentes de ce fichier était fausse.
        """
        e = equity_matrix_169()
        chips: dict[float, PushFoldSolution] = {}
        elargit_le_jam: list[tuple] = []
        elargit_le_call: list[tuple] = []
        for tapis in _TAPIS_BALAYES:
            eff = float(min(tapis[0], tapis[1]))
            if eff not in chips:
                chips[eff] = solve_hu_pushfold(eff, sb=0.5, bb=1.0, equity=e)
            chip = chips[eff]
            for gains in _GAINS_BALAYES:
                icm = solve_hu_pushfold(
                    float(tapis[0]), payouts=list(gains),
                    stacks=[float(s) for s in tapis], sb=0.5, bb=1.0,
                    equity=e)
                if icm.call_pct > chip.call_pct + 1e-12:
                    elargit_le_call.append((tapis, gains))
                if icm.jam_pct > chip.jam_pct + 1e-12:
                    elargit_le_jam.append((tapis, gains))
        self.assertEqual(elargit_le_call, [], "l'ICM a élargi un call")
        self.assertEqual(len(elargit_le_jam), 19)
        # le contre-exemple héros couvert, cité par la docstring du module
        self.assertIn(((15, 16, 1), (65.0, 35.0)), elargit_le_jam)

    def test_jam_elargi_alors_que_le_heros_est_couvert(self) -> None:
        """Le contre-exemple, mesuré à travers le conseiller lui-même.

        Héros 15 bb couvert par 16 bb, un joueur à 1 bb : le call du vilain
        s'effondre (28,6 % → 5,0 %) et voler les blindes devient si bon marché
        que le jam s'élargit (44,5 % → 60,8 %).
        """
        e = equity_matrix_169()
        chip = solve_hu_pushfold(15.0, sb=0.5, bb=1.0, equity=e)
        icm = solve_hu_pushfold(15.0, payouts=[65.0, 35.0],
                                stacks=[15.0, 16.0, 1.0], sb=0.5, bb=1.0,
                                equity=e)
        self.assertLess(icm.call_pct, chip.call_pct)
        self.assertGreater(icm.jam_pct, chip.jam_pct)
        self.assertAlmostEqual(chip.jam_pct, 0.445, places=3)
        self.assertAlmostEqual(icm.jam_pct, 0.608, places=3)

    def test_bande_dindifference_icm_couvre_les_bascules(self) -> None:
        """``_INDIFFERENCE_CAGNOTTE`` borne bien ce qu'un solve plus long change.

        Le solve livré s'arrête à 20 itérations sur ce spot. Rejoué avec un
        plafond de 1000, cinq groupes changent de verdict jam/fold ; le plus
        « tranché » d'entre eux n'affichait que 0,032 % de la cagnotte. Le
        seuil du module (0,1 %) est donc au-dessus de la bande, d'un facteur
        3 — c'est la mesure qui le justifie, pas un chiffre au juger.
        """
        e = equity_matrix_169()
        kw = dict(hero_stack_bb=10.0, payouts=[65.0, 35.0],
                  stacks=[10.0, 40.0, 1.0], sb=0.5, bb=1.0)
        livre = solve_hu_pushfold(equity=e, **kw)
        pousse = solve_hu_pushfold(equity=e, max_iter=_PLAFOND_REFERENCE,
                                   tol=1e-10, **kw)
        bascules = np.nonzero(np.sign(pousse.ev_jam_par_groupe)
                              != np.sign(livre.ev_jam_par_groupe))[0]
        self.assertEqual(bascules.size, 5)
        pire = float(np.abs(livre.ev_jam_par_groupe[bascules]).max()) / 100.0
        self.assertAlmostEqual(pire, 0.000318, places=6)
        self.assertLess(pire, _INDIFFERENCE_CAGNOTTE)

    def test_bande_dindifference_chipev_couvre_les_bascules(self) -> None:
        """Même mesure en chipEV, au tapis le plus instable du domaine.

        À 22,1 bb, quatre groupes changent de verdict entre le solve livré et
        un solve à 1000 itérations ; le plus tranché affichait 0,029 bb.
        ``_INDIFFERENCE_BB`` (0,05) est 1,7 fois au-dessus.
        """
        e = equity_matrix_169()
        livre = solve_hu_pushfold(22.1, sb=0.5, bb=1.0, equity=e)
        pousse = solve_hu_pushfold(22.1, sb=0.5, bb=1.0, equity=e,
                                   max_iter=_PLAFOND_REFERENCE, tol=1e-10)
        bascules = np.nonzero(np.sign(pousse.ev_jam_par_groupe)
                              != np.sign(livre.ev_jam_par_groupe))[0]
        self.assertEqual(bascules.size, 4)
        pire = float(np.abs(livre.ev_jam_par_groupe[bascules]).max())
        self.assertAlmostEqual(pire, 0.0288, places=4)
        self.assertLess(pire, _INDIFFERENCE_BB)

    def test_chipev_converge_sur_tout_le_domaine_push_fold(self) -> None:
        """Le label « certain » du régime chipEV n'est jamais un vœu.

        ``_advise_preflop_short`` ne dit « certain » que si le solve est
        certifié. Balayage des 240 tapis que le conseiller peut atteindre
        (arrondi au dixième de bb, de 1,1 à 25,0) : tous certifiés.
        """
        e = equity_matrix_169()
        non_certifies = [d / 10.0 for d in range(11, 251)
                         if not solve_hu_pushfold(d / 10.0, sb=0.5, bb=1.0,
                                                  equity=e).converged]
        self.assertEqual(non_certifies, [])

    def test_main_de_frontiere_chiffre_un_ecart_pas_une_frequence(self) -> None:
        """KTs sur le golden : +0,0047 unité de gain, 0,005 % de la cagnotte.

        La fréquence moyenne du fictitious play vaut 0,446 sur ce groupe —
        elle ne certifie rien (voir ``test_ranges_dequilibre_se_resserrent``).
        Le conseiller doit donc parler de l'écart d'EV, jamais d'un « X % du
        temps à l'équilibre ».
        """
        a = advise(Spot(hero="Kh Th", big_blind=1, **BULLE))
        frontiere = [r for r in a.reasons if "frontière" in r]
        self.assertEqual(len(frontiere), 1)
        self.assertIn("% de la cagnotte", frontiere[0])
        self.assertFalse(any("du temps" in r for r in a.reasons))
        self.assertLess(abs(a.ev_icm), _INDIFFERENCE_CAGNOTTE * 100.0)

    def test_ante_elargit_le_jam(self) -> None:
        """L'ante est de l'argent mort : elle paie la prise de risque.

        Mesuré à 10/40/1 bb, gains 65/35 : la range de jam passe de 8,9 %
        sans ante à 11,8 % avec 0,2 bb d'ante, et A5s bascule fold → jam.
        """
        sans = advise(Spot(hero="Ah 5h", big_blind=1, **BULLE))
        avec = advise(Spot(hero="Ah 5h", big_blind=1, ante=0.2, **BULLE))
        self.assertEqual(sans.action, "FOLD")
        self.assertTrue(avec.action.startswith("JAM"))
        self.assertGreater(avec.ev_icm, sans.ev_icm)

    def test_places_permutables(self) -> None:
        """L'$EV de Harville ne dépend pas de l'ordre des sièges."""
        a = advise(Spot(hero="Kh 9d", big_blind=1, **BULLE))
        b = advise(Spot(hero="Kh 9d", big_blind=1, players=3,
                        stacks="1, 10, 40", hero_seat=1, villain_seat=2,
                        payouts="65, 35"))
        self.assertEqual(a.action, b.action)
        self.assertAlmostEqual(a.ev_icm, b.ev_icm, places=12)
        self.assertAlmostEqual(a.bubble, b.bubble, places=12)

    def test_tapis_acceptes_en_sequence(self) -> None:
        texte = advise(Spot(hero="Kh 9d", big_blind=1, **BULLE))
        liste = advise(Spot(hero="Kh 9d", big_blind=1, players=3,
                            stacks=[10.0, 40.0, 1.0], payouts=(65.0, 35.0)))
        self.assertAlmostEqual(texte.ev_icm, liste.ev_icm, places=12)

    def test_profond_retombe_sur_la_chart(self) -> None:
        # 60 bb effectifs : le jeu ne se réduit plus à pousser ou passer.
        a = advise(Spot(hero="Ah Ad", stacks="60, 60, 40", payouts="65, 35",
                        players=3, big_blind=1, position="BTN"))
        self.assertIn("chart", a.regime)

    def test_places_incoherentes_rejetees(self) -> None:
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah Ad", stacks="10, 40", payouts="65, 35",
                        hero_seat=0, villain_seat=0, big_blind=1))
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah Ad", stacks="10, 0, 5", payouts="65, 35",
                        big_blind=1))
        with self.assertRaises(ValueError):
            advise(Spot(hero="Ah Ad", stacks="10, 40, 1", payouts="0",
                        big_blind=1))


class TestAmputationDuBubbleFactor(unittest.TestCase):
    """Le contournement de ``_bubble``, et le cas exact qui l'exige.

    La branche perdante met le héros à zéro. La récurrence de Harville divise
    par la somme des tapis encore en course : elle n'est indéfinie que si elle
    doit descendre jusqu'au joueur busté, donc quand la structure classe
    autant de places qu'il y a de joueurs. Le diagnostic « ≥ 3 joueurs » est
    faux — le golden 3 tapis / 2 gains du conseiller passe très bien.
    """

    def test_la_division_par_zero_suit_len_payouts_pas_len_stacks(self) -> None:
        passe = ([10.0, 0.0], [100.0]), ([10.0, 40.0, 0.0], [65.0, 35.0]), \
                ([10.0, 40.0, 5.0, 0.0], [50.0, 30.0, 20.0])
        for stacks, payouts in passe:
            icm_equities(stacks, payouts)       # ne doit rien lever
        leve = (([10.0, 0.0], [65.0, 35.0]),
                ([10.0, 40.0, 0.0], [50.0, 30.0, 20.0]),
                ([10.0, 40.0, 5.0, 0.0], [50.0, 30.0, 20.0, 10.0]),
                ([10.0, 40.0, 5.0, 3.0, 0.0], [50.0, 30.0, 10.0, 6.0, 4.0]))
        for stacks, payouts in leve:
            with self.assertRaises(ZeroDivisionError):
                icm_equities(stacks, payouts)

    def test_le_golden_du_chantier_na_pas_besoin_de_lamputation(self) -> None:
        """3 tapis / 2 gains : la valeur exacte existe, l'écart est mesuré."""
        exact = bubble_factor([10.0, 40.0, 1.0], [65.0, 35.0], 0, 1,
                              amount=10.0)
        ampute = bubble_factor([10.0, 40.0, 1.0], [65.0, 35.0], 0, 1,
                               amount=10.0 * (1.0 - 1e-9))
        self.assertAlmostEqual(exact, 5.021764845, places=9)
        self.assertAlmostEqual(ampute, 5.021764802, places=9)
        self.assertLess(abs(ampute - exact) / exact, 1e-8)     # mesuré 8,5e-9

    def test_trois_gains_sur_trois_tapis_ne_marchent_que_par_lamputation(self):
        """Le cas que le conseiller expose et qui n'était épinglé nulle part."""
        with self.assertRaises(ZeroDivisionError):
            bubble_factor([10.0, 40.0, 1.0], [50.0, 30.0, 20.0], 0, 1,
                          amount=10.0)
        a = advise(Spot(hero="Kh 9d", big_blind=1, stacks="10, 40, 1",
                        payouts="50, 30, 20", players=3))
        self.assertAlmostEqual(a.bubble, 2.970558, places=6)
        # amputation mille fois plus fine : la limite est stable à 1e-8 près
        fin = bubble_factor([10.0, 40.0, 1.0], [50.0, 30.0, 20.0], 0, 1,
                            amount=10.0 * (1.0 - 1e-12))
        self.assertLess(abs(a.bubble - fin) / fin, 1e-8)       # mesuré 7,0e-9

    def test_heads_up_icm_est_lineaire_donc_bf_vaut_un(self) -> None:
        """2 tapis / 2 gains : $EV affine en jetons, et l'amputation obligatoire.

        Avec deux joueurs, P(1er) = s/T et P(2e) = 1 − s/T : le $EV vaut
        π₂ + (π₁ − π₂)·s/T, affine en jetons. Le bubble factor vaut donc 1
        exactement et l'équilibre ICM EST l'équilibre chipEV. Ce chemin est
        aussi celui qui exige l'amputation (2 gains pour 2 joueurs).
        """
        with self.assertRaises(ZeroDivisionError):
            bubble_factor([10.0, 40.0], [65.0, 35.0], 0, 1, amount=10.0)
        icm = advise(Spot(hero="Kh 9d", big_blind=1, stacks="10, 40",
                          payouts="65, 35", players=2))
        chip = advise(Spot(hero="Kh 9d", stack=10, big_blind=1))
        self.assertEqual(icm.bubble, 1.0)
        self.assertEqual(icm.ev_bb, chip.ev_bb)
        self.assertEqual(icm.action, chip.action)
        # pente exacte : (65 − 35)/(10 + 40) = 0,6 ; l'écart vient du plancher
        # relatif 1e-9 que ``pushfold`` pose sur ses tapis nuls
        self.assertAlmostEqual(icm.ev_icm / icm.ev_bb, 0.6, places=7)

    def test_heads_up_icm_ne_suppose_rien_donc_reste_certain(self) -> None:
        """À deux, il n'y a personne à supposer couché : le label tient."""
        a = advise(Spot(hero="Kh 9d", big_blind=1, stacks="10, 40",
                        payouts="65, 35", players=2))
        self.assertEqual(a.confidence, "certain")
        self.assertIn("2 joueurs", a.regime)
        self.assertFalse(any("suppose les autres déjà couchés" in r
                             for r in a.reasons))
        self.assertFalse(any("supposés déjà couchés" in x
                             for x in a.assumptions))

    def test_cagnotte_plate_rend_un_bubble_factor_infini(self) -> None:
        """Sans gain à départager, un jeton gagné ne rapporte rien.

        ``bubble_factor`` ne lève pas dans ce cas : le gain au dénominateur
        est nul et il rend l'infini. C'est le conseiller qui refuse le spot
        en amont (voir ``test_places_incoherentes_rejetees``).
        """
        self.assertEqual(
            bubble_factor([10.0, 40.0, 1.0], [0.0], 0, 1, amount=5.0),
            math.inf)


class TestIcmPostflop(unittest.TestCase):
    """Les cotes du pot corrigées par le bubble factor.

    Le bubble factor ne dépend que des RAPPORTS de tapis : les tapis peuvent
    donc rester en bb pendant que le pot et la mise sont en jetons.
    """

    # A3 troisième paire : 39,1 % d'équité contre la range « moyenne ».
    RIVER = dict(hero="Ah 3d", board="Qs 7d 2c 9h 3s", pot=100, bet=75,
                 stack=300, big_blind=10)

    def test_call_correct_en_cash_devient_fold_en_tournoi(self) -> None:
        # payer 75 dans 100 exige 30 % en cash ; à BF = 5,02 il en faut 68,3 %,
        # et 39,1 % tombe entre les deux : le call devient un fold.
        cash = advise(Spot(**self.RIVER))
        icm = advise(Spot(**self.RIVER, **BULLE))
        self.assertAlmostEqual(cash.required, 0.30, places=6)
        self.assertAlmostEqual(icm.required, 0.6827, places=3)
        self.assertGreater(icm.equity, cash.required)
        self.assertAlmostEqual(icm.mdf, cash.mdf, places=9)   # la MDF ne bouge pas
        self.assertTrue(cash.action.startswith("CALL"))
        self.assertEqual(icm.action, "FOLD")

    def test_alpha_icm_suit_la_formule(self) -> None:
        from pfs.core.icm import icm_required_equity

        a = advise(Spot(**self.RIVER, **BULLE))
        self.assertAlmostEqual(a.required,
                               icm_required_equity(100.0, 75.0, a.bubble),
                               places=9)


class TestSolveurRiver(unittest.TestCase):
    """Le spot de polarisation, résolu à travers le conseiller.

    AA (nuts) + 33 (air) contre QQ sur 2s2d7h8hKc, pot 100, mise pot :
    part de bluff dans la range de mise = b/(P+2b) = 1/3, MDF du vilain
    = 1 − b/(P+b) = 1/2, et 33 mise donc la moitié du temps.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.air = advise(Spot(
            hero="3d 3h", board="2s 2d 7h 8h Kc", pot=100, bet=0, stack=100,
            big_blind=1, position="BTN", hero_range="AA,33", villain="QQ",
            bet_sizes="1.0", max_bets=1))
        cls.nuts = advise(Spot(
            hero="As Ad", board="2s 2d 7h 8h Kc", pot=100, bet=0, stack=100,
            big_blind=1, position="BTN", hero_range="AA,33", villain="QQ",
            bet_sizes="1.0", max_bets=1))

    def test_air_bluffe_la_moitie_du_temps(self) -> None:
        """La grandeur à tester est la fréquence de MISE, pas ``frequency``.

        ``Advice.frequency`` est la fréquence de l'action de tête ; ici c'est
        le CHECK (0,5006) qui mène d'un cheveu, et le partage 50/50 rendait
        les deux indiscernables. La part de bluff se lit dans ``mix``.
        """
        self.assertIn("MIXTE", self.air.action)
        mise = dict(self.air.mix)["TAPIS (100, 100 % du pot)"]
        self.assertAlmostEqual(mise, 0.5, delta=0.05)
        self.assertEqual(self.air.mix[0][0], "CHECK")   # l'action de tête
        self.assertAlmostEqual(self.air.frequency, self.air.mix[0][1],
                               places=12)
        self.assertAlmostEqual(sum(f for _, f in self.air.mix), 1.0, places=9)

    def test_nuts_mise_toujours(self) -> None:
        self.assertAlmostEqual(self.nuts.frequency, 1.0, delta=0.03)
        self.assertAlmostEqual(self.nuts.size, 100.0, places=6)

    def test_ev_du_combo_est_lisible_dans_lobjet(self) -> None:
        """L'EV ne doit pas exister seulement dans une chaîne formatée.

        Mesuré : 0,0286 pour 33 (air) et 149,9553 pour AA (nuts) sur un pot
        de 100, big blind 1 — donc les mêmes valeurs en bb.
        """
        self.assertAlmostEqual(self.air.ev_bb, 0.0286, places=4)
        self.assertAlmostEqual(self.nuts.ev_bb, 149.9553, places=4)
        self.assertTrue(any("EV de ta main" in r for r in self.nuts.reasons))

    def test_mix_garde_ce_que_laffichage_filtre(self) -> None:
        """AA mise 99,97 % du temps : le check à 0,03 % sort de l'affichage.

        C'est exactement ce que ``_MIXTE_MIN`` cache, et ce que ``mix`` doit
        rendre.
        """
        rare = dict(self.nuts.mix)["CHECK"]
        self.assertLess(rare, 0.01)
        self.assertGreater(rare, 0.0)
        self.assertNotIn("CHECK", self.nuts.action)

    def test_vilain_defend_a_la_mdf(self) -> None:
        # 1 − b/(P+b) = 1 − 100/200 = 0,5, retrouvé par le solveur
        self.assertAlmostEqual(self.nuts.mdf, 0.5, places=9)
        self.assertAlmostEqual(self.nuts.villain_defence, 0.5, delta=0.05)

    def test_verdict_indicatif_car_ranges_supposees(self) -> None:
        self.assertEqual(self.nuts.confidence, "indicatif")
        self.assertIn("CFR", self.nuts.regime)
        self.assertTrue(any("hypothèse" in a for a in self.nuts.assumptions))

    def test_budget_nul_retombe_sur_lequite(self) -> None:
        a = advise(Spot(hero="As Ad", board="2s 2d 7h 8h Kc", pot=100, bet=0,
                        stack=100, big_blind=1, position="BTN",
                        hero_range="AA,33", villain="QQ", solver_budget_s=0.0))
        self.assertIn("équité", a.regime)
        self.assertIsNone(a.frequency)
        self.assertIsNotNone(a.equity)

    def test_budget_respecte(self) -> None:
        """Le budget borne le temps passé, pas seulement les itérations."""
        depart = time.perf_counter()
        advise(Spot(hero="Ah Kh", board="Kd 7h 2c 9s 3d", pot=100, bet=0,
                    stack=200, big_blind=10, position="BTN", villain="large",
                    solver_budget_s=0.3))
        ecoule = time.perf_counter() - depart
        # le budget est vérifié entre deux paquets de 50 itérations : un
        # dépassement d'un paquet est attendu, un facteur 4 ne l'est pas
        self.assertLess(ecoule, 1.2)

    def test_hors_de_position_le_check_nest_pas_labattage(self) -> None:
        """Hors de position, checker laisse le vilain miser — l'arbre diffère."""
        ip = advise(Spot(hero="3d 3h", board="2s 2d 7h 8h Kc", pot=100, bet=0,
                         stack=100, big_blind=1, position="BTN",
                         hero_range="AA,33", villain="QQ", bet_sizes="1.0",
                         max_bets=1))
        oop = advise(Spot(hero="3d 3h", board="2s 2d 7h 8h Kc", pot=100, bet=0,
                          stack=100, big_blind=1, position="BB",
                          hero_range="AA,33", villain="QQ", bet_sizes="1.0",
                          max_bets=1))
        self.assertIn("hors de position", " ".join(oop.assumptions))
        self.assertIn("en position", " ".join(ip.assumptions))

    def test_main_hors_range_est_declaree(self) -> None:
        a = advise(Spot(hero="7c 2c", board="2s 2d 7h 8h Kc", pot=100, bet=0,
                        stack=100, big_blind=1, position="BTN",
                        villain="serree", bet_sizes="0.75", max_bets=1))
        self.assertTrue(any("pas dans la range" in x for x in a.assumptions))

    def test_flop_reste_sur_lequite(self) -> None:
        # le solveur ne couvre que turn et river : le flop garde l'ancien régime
        a = advise(Spot(hero="Ah Kh", board="Kd 7h 2c", pot=50, bet=0,
                        stack=300, big_blind=10))
        self.assertIn("équité", a.regime)


class TestCoutDuSolveRiver(unittest.TestCase):
    """Ce que coûtent ``RIVER_SOLVE_ITERATIONS`` et ``RIVER_SOLVE_BUDGET_S``.

    Configuration de référence des deux docstrings : board Kd 7h 2c 9s 3d,
    pot 100, tapis 200, tailles 0,33 / 0,75 / 1,25 × pot, 2 mises par street.
    Le solve est déterministe — ces valeurs se rejouent au chiffre près.
    """

    BOARD = "Kd 7h 2c 9s 3d"
    HERO = "Ah Kh"
    TAILLES = (0.33, 0.75, 1.25)

    @classmethod
    def solveur(cls, nom: str, en_position: bool) -> PostflopSolver:
        board = parse_cards(cls.BOARD)
        hero = parse_cards(cls.HERO)
        vil = parse_range(_DEFAULT_VILLAIN[nom])
        poids = vil.weights.copy()
        poids[combo_index(hero[0], hero[1])] = 1.0
        mienne = Range(poids)
        if en_position:
            return PostflopSolver(
                board, vil, mienne, pot=100.0, stack=200.0,
                oop_bet_fracs=(), ip_bet_fracs=cls.TAILLES,
                raise_fracs=cls.TAILLES, max_bets=2)
        return PostflopSolver(
            board, mienne, vil, pot=100.0, stack=200.0,
            oop_bet_fracs=cls.TAILLES, ip_bet_fracs=cls.TAILLES,
            raise_fracs=cls.TAILLES, max_bets=2)

    @classmethod
    def setUpClass(cls) -> None:
        cls.ip = cls.solveur("large", True)
        cls.ip.solve(RIVER_SOLVE_ITERATIONS)
        cls.oop = cls.solveur("large", False)
        cls.oop.solve(RIVER_SOLVE_ITERATIONS)

    def test_larbre_hors_de_position_est_le_plus_lourd(self) -> None:
        """20 nœuds contre 11 : le compte ne dépend pas de la machine.

        Le docstring de ``RIVER_SOLVE_BUDGET_S`` désignait l'arbre à 11 nœuds
        comme le pire cas ; c'est celui du chemin en position, où le vilain
        a déjà checké et ne peut plus miser.
        """
        ip = advise(Spot(hero=self.HERO, board=self.BOARD, pot=100, bet=0,
                         stack=200, big_blind=10, position="BTN",
                         villain="large", solver_budget_s=0.05))
        oop = advise(Spot(hero=self.HERO, board=self.BOARD, pot=100, bet=0,
                          stack=200, big_blind=10, position="BB",
                          villain="large", solver_budget_s=0.05))
        self.assertIn("arbre de 11 nœuds", " ".join(ip.assumptions))
        self.assertIn("arbre de 20 nœuds", " ".join(oop.assumptions))

    def test_combos_apres_retrait_du_board(self) -> None:
        """Les ranges nommées, écrites puis réduites par le board.

        Le docstring annonçait « 328 combos » sans dire sur quel board :
        390 combos écrits, 328 une fois Kd 7h 2c 9s 3d retiré.
        """
        self.assertEqual(
            int(np.count_nonzero(parse_range(_DEFAULT_VILLAIN["large"]).weights)),
            390)
        self.assertEqual(
            int(np.count_nonzero(parse_range(_DEFAULT_VILLAIN["moyenne"]).weights)),
            230)
        self.assertEqual(self.ip.players[0].cards.shape[0], 328)
        self.assertEqual(
            self.solveur("moyenne", True).players[0].cards.shape[0], 196)

    def test_exploitabilite_a_400_iterations(self) -> None:
        """Les deux chiffres de ``RIVER_SOLVE_ITERATIONS``, sur les deux arbres.

        Mesuré : 0,052 % du pot en position, 0,139 % hors de position, pour
        deux ranges « large ». C'est le second chiffre qui compte, puisque
        c'est l'arbre que le budget par défaut n'atteint pas.
        """
        self.assertAlmostEqual(self.ip.exploitability() * 100.0, 0.052,
                               places=3)
        self.assertAlmostEqual(self.oop.exploitability() * 100.0, 0.139,
                               places=3)

    def test_ce_que_le_filtre_mixte_cache_est_borne(self) -> None:
        """Les chiffres de ``_MIXTE_MIN``, sur le même solve golden.

        La moyenne DCFR n'atteint jamais zéro : aucune des 1312 entrées de la
        racine n'est nulle, la plus petite vaut 1,6e-4. Ce que le filtre à 1 %
        retire d'un combo est borné — 2,2 points au pire, 0,2 en médiane.
        """
        sigma = self.ip.average_strategy(self.ip.node_at(("check",)))
        self.assertEqual(sigma.size, 1312)
        self.assertEqual(int(np.count_nonzero(sigma == 0.0)), 0)
        self.assertAlmostEqual(float(sigma.min()), 1.6e-4, places=5)
        cachee = (sigma * (sigma < _MIXTE_MIN)).sum(axis=0)
        self.assertLess(float(cachee.max()), 0.022)
        self.assertLess(float(np.median(cachee)), 0.0021)

    def test_le_budget_par_defaut_tronque_hors_de_position(self) -> None:
        """La troncature est déclarée, jamais tue.

        Le budget par défaut (2,0 s) ne suffit pas aux 400 itérations sur
        l'arbre à 20 nœuds de cette machine. Quel que soit le verdict de la
        machine qui exécute le test, la règle testable est la même : dès que
        le conseiller coupe, il le dit et il chiffre l'exploitabilité.
        """
        a = advise(Spot(hero=self.HERO, board=self.BOARD, pot=100, bet=0,
                        stack=200, big_blind=10, position="BB",
                        villain="large", solver_budget_s=0.4))
        self.assertTrue(any("Budget de 0.4 s atteint avant" in r
                            for r in a.reasons))
        self.assertTrue(any("exploitabilité" in r for r in a.reasons))


if __name__ == "__main__":
    unittest.main()
