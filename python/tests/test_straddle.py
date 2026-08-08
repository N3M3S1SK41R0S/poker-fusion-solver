"""Tests du support des straddles (simple, double, Mississippi) dans le parseur HH.

Règle centrale vérifiée ici : le straddle est de l'argent mort posté AVANT la
distribution des cartes. Il est donc enregistré comme ``ActionType.POST`` —
exactement comme les blindes — et ne compte **jamais** comme VPIP. Seule une
action volontaire ultérieure (call, bet, raise, all-in) du straddler le rend
VPIP.
"""

from __future__ import annotations

import unittest

from pfs.data.hand_history import (
    ActionType,
    Street,
    parse_pokerstars,
    parse_winamax,
    player_key,
)

SALT = "s"


def _k(nickname: str) -> str:
    """Clé hachée d'un pseudo, avec le sel commun aux fixtures."""
    return player_key(nickname, SALT)


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES — modelées sur la constante WNMX de test_app_and_data.py
# ═══════════════════════════════════════════════════════════════════════

# Main de référence SANS straddle (non-régression).
WNMX_PLAIN = """Winamax Poker - CashGame - HandId: #12345-678-1234567890 - Holdem no limit (0.50€)/(1€) - 2026/08/06 20:14:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Bob raises 2€ to 3€
Carol folds
Dave calls 2.50€
Alice folds
*** FLOP *** [Ks 7d 2c]
Dave checks
Bob bets 4€
Dave folds
Bob collected 7€ from pot
*** SUMMARY ***
Total pot 7€"""

# Straddle simple : Bob (UTG) straddle 2€ puis FOLD sur la relance de Carol.
WNMX_STRADDLE_FOLD = """Winamax Poker - CashGame - HandId: #12345-678-2000000001 - Holdem no limit (0.50€)/(1€) - 2026/08/07 20:14:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Bob posts straddle 2€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Carol raises 4€ to 6€
Dave folds
Alice folds
Bob folds
Carol collected 5.50€ from pot
*** SUMMARY ***
Total pot 5.50€"""

# Straddle simple, verbe alternatif « straddles », puis check de l'option.
WNMX_STRADDLE_CHECK = """Winamax Poker - CashGame - HandId: #12345-678-2000000002 - Holdem no limit (0.50€)/(1€) - 2026/08/07 20:20:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Bob straddles 2€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Carol folds
Dave calls 1.50€
Alice calls 1€
Bob checks
*** FLOP *** [Ks 7d 2c]
Dave checks
Alice checks
Bob checks
*** TURN *** [Ks 7d 2c][5h]
Dave checks
Alice checks
Bob checks
*** RIVER *** [Ks 7d 2c][5h][9s]
Dave checks
Alice checks
Bob checks
Dave collected 6€ from pot
*** SUMMARY ***
Total pot 6€"""

# Straddle simple : Bob straddle 2€ puis COMPLÈTE volontairement (call).
WNMX_STRADDLE_CALL = """Winamax Poker - CashGame - HandId: #12345-678-2000000003 - Holdem no limit (0.50€)/(1€) - 2026/08/07 20:30:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Bob posts straddle 2€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Carol raises 4€ to 6€
Dave folds
Alice folds
Bob calls 4€
*** FLOP *** [Ks 7d 2c]
Bob checks
Carol bets 6€
Bob folds
Carol collected 13.50€ from pot
*** SUMMARY ***
Total pot 13.50€"""

# Double straddle : Bob 2€ puis Carol re-straddle 4€ — l'ordre fait foi.
WNMX_DOUBLE_STRADDLE = """Winamax Poker - CashGame - HandId: #12345-678-2000000004 - Holdem no limit (0.50€)/(1€) - 2026/08/07 20:40:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Bob posts straddle 2€
Carol posts straddle 4€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Dave folds
Alice folds
Bob folds
Carol collected 3.50€ from pot
*** SUMMARY ***
Total pot 3.50€"""

# Straddle Mississippi : Carol est au bouton (Seat #3 is the button) et
# straddle. Rien de spécial à parser : l'ordre des lignes du HH suffit.
WNMX_MISSISSIPPI = """Winamax Poker - CashGame - HandId: #12345-678-2000000005 - Holdem no limit (0.50€)/(1€) - 2026/08/07 20:50:11 UTC
Table: 'Bordeaux 05' 6-max (real money) Seat #3 is the button
Seat 1: Alice (100€)
Seat 2: Bob (85.50€)
Seat 3: Carol (120€)
Seat 4: Dave (97€)
*** ANTE/BLINDS ***
Dave posts small blind 0.50€
Alice posts big blind 1€
Carol posts straddle 2€
Dealt to Dave [Ah Kd]
*** PRE-FLOP ***
Bob folds
Dave folds
Alice folds
Carol collected 2.50€ from pot
*** SUMMARY ***
Total pot 2.50€"""

# PokerStars, straddle avec montant explicite « posts straddle €1 ».
PS_STRADDLE = """PokerStars Hand #987654322:  Hold'em No Limit (€0.25/€0.50 EUR) - 2026/08/07 21:02:11 CET
Table 'Andromeda' 6-max Seat #2 is the button
Seat 1: Erin (50 in chips)
Seat 2: Frank (62.50 in chips)
Seat 3: Gina (48 in chips)
Erin: posts small blind 0.25
Frank: posts big blind 0.50
Gina: posts straddle €1
*** HOLE CARDS ***
Dealt to Gina [Qs Qh]
Erin: folds
Frank: folds
Gina collected 0.75 from pot
*** SUMMARY ***
Total pot 0.75"""

# PokerStars, forme nue « X: straddles » sans montant → convention 2× BB.
PS_BARE_STRADDLE = """PokerStars Hand #987654323:  Hold'em No Limit (€0.25/€0.50 EUR) - 2026/08/07 21:10:11 CET
Table 'Andromeda' 6-max Seat #2 is the button
Seat 1: Erin (50 in chips)
Seat 2: Frank (62.50 in chips)
Seat 3: Gina (48 in chips)
Erin: posts small blind 0.25
Frank: posts big blind 0.50
Gina: straddles
*** HOLE CARDS ***
Dealt to Gina [Qs Qh]
Erin: folds
Frank: folds
Gina collected 0.75 from pot
*** SUMMARY ***
Total pot 0.75"""


# ═══════════════════════════════════════════════════════════════════════
# STRADDLE SIMPLE — WINAMAX
# ═══════════════════════════════════════════════════════════════════════


class TestWinamaxSimpleStraddle(unittest.TestCase):
    """Parsing d'un straddle simple, montant et big blind effective."""

    def test_straddle_is_parsed_with_player_and_amount(self) -> None:
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Bob"), 2.0)])

    def test_effective_bb_equals_the_straddle(self) -> None:
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        self.assertEqual(h.effective_bb, 2.0)
        # La big blind affichée par la table, elle, ne bouge pas.
        self.assertEqual(h.big_blind, 1.0)

    def test_straddle_action_is_a_preflop_post(self) -> None:
        """Le straddle apparaît dans les actions comme POST préflop (argent mort)."""
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        posts = [a for a in h.street_actions(Street.PREFLOP)
                 if a.player == _k("Bob") and a.action is ActionType.POST]
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].amount, 2.0)

    def test_straddles_verb_form_is_recognised(self) -> None:
        """Variante « X straddles 2€ » : même résultat que « posts straddle »."""
        h = parse_winamax(WNMX_STRADDLE_CHECK, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Bob"), 2.0)])
        self.assertEqual(h.effective_bb, 2.0)

    def test_other_players_stats_are_unaffected(self) -> None:
        """Le straddle ne corrompt pas les stats des autres joueurs."""
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        self.assertTrue(h.voluntarily_put_in_pot(_k("Carol")))
        self.assertTrue(h.preflop_raise(_k("Carol")))
        self.assertFalse(h.voluntarily_put_in_pot(_k("Dave")))  # SB fold


# ═══════════════════════════════════════════════════════════════════════
# DOUBLE STRADDLE
# ═══════════════════════════════════════════════════════════════════════


class TestDoubleStraddle(unittest.TestCase):
    """Deux straddles : entrées ordonnées, le dernier fixe la BB effective."""

    def test_two_entries_in_posting_order(self) -> None:
        h = parse_winamax(WNMX_DOUBLE_STRADDLE, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Bob"), 2.0), (_k("Carol"), 4.0)])

    def test_effective_bb_is_the_last_straddle(self) -> None:
        h = parse_winamax(WNMX_DOUBLE_STRADDLE, salt=SALT)
        self.assertEqual(h.effective_bb, 4.0)

    def test_neither_straddler_is_vpip_after_folding_or_winning_blind(self) -> None:
        h = parse_winamax(WNMX_DOUBLE_STRADDLE, salt=SALT)
        self.assertFalse(h.voluntarily_put_in_pot(_k("Bob")))
        self.assertFalse(h.voluntarily_put_in_pot(_k("Carol")))


# ═══════════════════════════════════════════════════════════════════════
# NON-RÉGRESSION — MAIN SANS STRADDLE
# ═══════════════════════════════════════════════════════════════════════


class TestNoStraddleRegression(unittest.TestCase):
    """Une main ordinaire doit rester strictement inchangée."""

    def test_straddles_list_is_empty(self) -> None:
        h = parse_winamax(WNMX_PLAIN, salt=SALT)
        self.assertEqual(h.straddles, [])

    def test_effective_bb_falls_back_to_big_blind(self) -> None:
        h = parse_winamax(WNMX_PLAIN, salt=SALT)
        self.assertEqual(h.effective_bb, h.big_blind)
        self.assertEqual(h.effective_bb, 1.0)

    def test_blind_posts_still_do_not_count_as_vpip(self) -> None:
        """La règle historique sur les blindes reste intacte."""
        h = parse_winamax(WNMX_PLAIN, salt=SALT)
        self.assertFalse(h.voluntarily_put_in_pot(_k("Alice")))  # BB fold


# ═══════════════════════════════════════════════════════════════════════
# VPIP — LE STRADDLE EST DE L'ARGENT MORT
# ═══════════════════════════════════════════════════════════════════════


class TestStraddleAndVpip(unittest.TestCase):
    """Un straddle seul n'est jamais VPIP ; une action volontaire ensuite, si."""

    def test_straddler_who_folds_is_not_vpip(self) -> None:
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        self.assertFalse(h.voluntarily_put_in_pot(_k("Bob")))

    def test_stat_observations_reports_vpip_false_for_folding_straddler(self) -> None:
        """`stat_observations` (inchangé) : le straddler-foldeur compte VPIP=False."""
        h = parse_winamax(WNMX_STRADDLE_FOLD, salt=SALT)
        obs = h.stat_observations(_k("Bob"))
        self.assertIn("vpip", obs)          # il était assis : l'occasion existe
        self.assertFalse(obs["vpip"])
        self.assertFalse(obs["pfr"])

    def test_straddler_who_checks_his_option_is_not_vpip(self) -> None:
        """Checker l'option du straddle = gratuit, pas volontaire (comme la BB)."""
        h = parse_winamax(WNMX_STRADDLE_CHECK, salt=SALT)
        self.assertFalse(h.voluntarily_put_in_pot(_k("Bob")))

    def test_straddler_who_voluntarily_calls_is_vpip(self) -> None:
        h = parse_winamax(WNMX_STRADDLE_CALL, salt=SALT)
        self.assertTrue(h.voluntarily_put_in_pot(_k("Bob")))
        self.assertTrue(h.stat_observations(_k("Bob"))["vpip"])


# ═══════════════════════════════════════════════════════════════════════
# STRADDLE MISSISSIPPI (BOUTON)
# ═══════════════════════════════════════════════════════════════════════


class TestMississippiStraddle(unittest.TestCase):
    """Straddle posté du bouton : l'ordre des lignes du HH suffit au parseur."""

    def test_button_straddle_is_parsed_like_any_other(self) -> None:
        h = parse_winamax(WNMX_MISSISSIPPI, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Carol"), 2.0)])
        self.assertEqual(h.effective_bb, 2.0)
        # Carol est bien le bouton — c'est ce qui fait de ce straddle un
        # Mississippi, sans que le parseur ait besoin de le savoir.
        carol_seat = next(s for s in h.seats if s.player == _k("Carol"))
        self.assertEqual(carol_seat.seat, h.button_seat)

    def test_mississippi_straddler_who_never_acts_is_not_vpip(self) -> None:
        h = parse_winamax(WNMX_MISSISSIPPI, salt=SALT)
        self.assertFalse(h.voluntarily_put_in_pot(_k("Carol")))


# ═══════════════════════════════════════════════════════════════════════
# POKERSTARS
# ═══════════════════════════════════════════════════════════════════════


class TestPokerStarsStraddle(unittest.TestCase):
    """Formats PokerStars : « X: posts straddle €2 » et « X: straddles » nu."""

    def test_posts_straddle_with_amount(self) -> None:
        h = parse_pokerstars(PS_STRADDLE, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Gina"), 1.0)])
        self.assertEqual(h.effective_bb, 1.0)
        self.assertEqual(h.big_blind, 0.50)

    def test_bare_straddles_falls_back_to_twice_the_big_blind(self) -> None:
        """Sans montant sur la ligne, convention universelle : 2× la BB."""
        h = parse_pokerstars(PS_BARE_STRADDLE, salt=SALT)
        self.assertEqual(h.straddles, [(_k("Gina"), 1.0)])
        self.assertEqual(h.effective_bb, 1.0)

    def test_ps_straddle_is_a_post_and_not_vpip(self) -> None:
        h = parse_pokerstars(PS_STRADDLE, salt=SALT)
        gina = _k("Gina")
        posts = [a for a in h.street_actions(Street.PREFLOP)
                 if a.player == gina and a.action is ActionType.POST]
        self.assertEqual(len(posts), 1)
        self.assertFalse(h.voluntarily_put_in_pot(gina))
        self.assertFalse(h.stat_observations(gina)["vpip"])


if __name__ == "__main__":
    unittest.main()
