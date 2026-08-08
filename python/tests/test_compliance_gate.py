"""
Tests du ComplianceGate — dont les faux positifs que le critère naïf
« pas de symbole € donc c'est fictif » aurait laissé passer.
"""

from __future__ import annotations

import pytest

from pfs.compliance.gate import (
    ComplianceGate,
    GateLockedError,
    Mode,
    StakeKind,
    TableObservation,
)

HWND = 0xA1B2


def obs(**kw) -> TableObservation:
    base = dict(
        hwnd=HWND,
        window_title="",
        stack_texts=(),
        pot_text="",
        play_money_badge_detected=None,
        tournament_ui_detected=None,
    )
    base.update(kw)
    return TableObservation(**base)


# ═══════════════════════════════════════════════════════════════════════
# LE CAS QUE PIERRE VEUT : play money, armé, corroboré
# ═══════════════════════════════════════════════════════════════════════


def test_play_money_armed_and_corroborated_unlocks_live() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(
            window_title="Table Argent Fictif 42 - NL Hold'em",
            stack_texts=("100 BB", "87 BB", "142 BB"),
            pot_text="12 BB",
            play_money_badge_detected=True,
        )
    )
    assert d.mode is Mode.LIVE
    assert d.stake_kind is StakeKind.PLAY_MONEY
    assert d.live_allowed is True


def test_play_money_badge_alone_corroborates() -> None:
    """Titre neutre mais badge client détecté : corroboration suffisante."""
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(window_title="Table 7", stack_texts=("2 500", "1 800"),
            play_money_badge_detected=True)
    )
    assert d.mode is Mode.LIVE


# ═══════════════════════════════════════════════════════════════════════
# ⚠️ LES FAUX POSITIFS DU CRITÈRE NAÏF — le cœur de ce module
# ═══════════════════════════════════════════════════════════════════════


def test_real_money_tournament_shows_chips_without_currency_stays_review() -> None:
    """PIÈGE N°1 — un MTT à 50 € affiche « 12 450 », sans aucun symbole.

    Le critère naïf « pas de € ⇒ fictif » aurait déverrouillé LIVE ici.
    """
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(
            window_title="Tournoi 50€ Deepstack - Table 12",
            stack_texts=("12 450", "8 900", "31 200"),   # aucun symbole monétaire
            pot_text="4 800",
            tournament_ui_detected=True,
        )
    )
    assert d.mode is Mode.REVIEW
    assert d.stake_kind is StakeKind.TOURNAMENT_REAL


def test_tournament_ui_alone_blocks_even_with_neutral_title() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(window_title="Table 3", stack_texts=("12 450",), tournament_ui_detected=True)
    )
    assert d.mode is Mode.REVIEW
    assert d.stake_kind is StakeKind.TOURNAMENT_REAL


def test_cash_table_displayed_in_big_blinds_stays_review() -> None:
    """PIÈGE N°2 — option « afficher les stacks en BB » sur une table réelle.

    Les stacks lisent « 100 BB » : identique à une table fictive.
    Seul le titre sauve la mise.
    """
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(window_title="NL50 - Table Bordeaux", stack_texts=("100 BB", "64 BB"))
    )
    assert d.mode is Mode.REVIEW


def test_freeroll_stays_review() -> None:
    """PIÈGE N°3 — pas de buy-in, mais dotation réelle : adversaires lésés."""
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(window_title="Freeroll du Dimanche - Table 9", stack_texts=("3 000",))
    )
    assert d.mode is Mode.REVIEW


@pytest.mark.parametrize(
    "title",
    [
        "Spin & Go 2€ - Table 1",
        "Sit & Go 10€",
        "Satellite WSOP - Table 4",
        "Qualif Main Event",
        "PLO100 - Table Lyon",
        "$5/$10 NL Hold'em",
        "Cash Game £2/£5",
    ],
)
def test_real_money_formats_all_blocked(title: str) -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    assert g.evaluate(obs(window_title=title)).mode is Mode.REVIEW


@pytest.mark.parametrize("stack", ["1 250 €", "$340", "£88.50", "220 EUR", "75 USD"])
def test_currency_glyph_anywhere_blocks(stack: str) -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(
        obs(window_title="Argent fictif", stack_texts=(stack,),
            play_money_badge_detected=True)
    )
    assert d.mode is Mode.REVIEW
    assert d.stake_kind is StakeKind.CASH_REAL


# ═══════════════════════════════════════════════════════════════════════
# FAIL-CLOSED : l'absence de preuve n'est jamais une preuve
# ═══════════════════════════════════════════════════════════════════════


def test_absence_of_currency_alone_never_unlocks() -> None:
    """LE test central. Aucun symbole, aucun signe de tournoi, fenêtre armée —
    et pourtant REVIEW, faute de corroboration positive."""
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(obs(window_title="Table 5", stack_texts=("100", "85", "132")))
    assert d.mode is Mode.REVIEW
    assert "corroboration" in d.reason


def test_corroboration_without_manual_arming_is_not_enough() -> None:
    """Même une table manifestement fictive reste en REVIEW sans armement."""
    g = ComplianceGate()
    d = g.evaluate(
        obs(window_title="Play Money Table 3", play_money_badge_detected=True)
    )
    assert d.mode is Mode.REVIEW
    assert "non armée" in d.reason


def test_arming_is_bound_to_one_hwnd() -> None:
    """Armer une fenêtre n'arme pas les autres tables ouvertes."""
    g = ComplianceGate()
    g.arm_window(HWND)
    other = obs(hwnd=0xDEAD, window_title="Play Money Table 4",
                play_money_badge_detected=True)
    assert g.evaluate(other).mode is Mode.REVIEW


def test_disarm_revokes_immediately() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    good = obs(window_title="Argent fictif", play_money_badge_detected=True)
    assert g.evaluate(good).mode is Mode.LIVE
    g.disarm_window(HWND)
    assert g.evaluate(good).mode is Mode.REVIEW


def test_live_is_revoked_mid_session_when_a_signal_flips() -> None:
    """Le gate est réévalué à chaque frame : si un € apparaît, LIVE tombe."""
    g = ComplianceGate()
    g.arm_window(HWND)
    good = obs(window_title="Argent fictif", stack_texts=("100 BB",),
               play_money_badge_detected=True)
    assert g.evaluate(good).mode is Mode.LIVE

    flipped = obs(window_title="Argent fictif", stack_texts=("100 BB", "50 €"),
                  play_money_badge_detected=True)
    assert g.evaluate(flipped).mode is Mode.REVIEW


def test_uncalibrated_badge_is_inconclusive_not_supportive() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(obs(window_title="Table 5", play_money_badge_detected=None))
    assert d.mode is Mode.REVIEW


def test_missing_badge_is_inconclusive_not_supportive() -> None:
    """Absence de badge ⇒ non concluant. Les tables réelles n'ont pas de badge inverse."""
    g = ComplianceGate()
    g.arm_window(HWND)
    d = g.evaluate(obs(window_title="Table 5", play_money_badge_detected=False))
    assert d.mode is Mode.REVIEW


# ═══════════════════════════════════════════════════════════════════════
# AUDIT POST-MAIN — le filet de sécurité sur le hand-history
# ═══════════════════════════════════════════════════════════════════════


def test_audit_locks_gate_when_assisted_hand_was_real_money() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    good = obs(window_title="Argent fictif", play_money_badge_detected=True)
    assert g.evaluate(good).mode is Mode.LIVE

    g.audit_hand(assisted_live=True, hh_is_real_money=True)

    assert g.is_locked is True
    assert g.evaluate(good).mode is Mode.REVIEW
    with pytest.raises(GateLockedError):
        g.arm_window(HWND)


def test_audit_is_silent_when_everything_is_consistent() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    g.audit_hand(assisted_live=True, hh_is_real_money=False)
    g.audit_hand(assisted_live=False, hh_is_real_money=True)
    assert g.is_locked is False


def test_unlock_requires_explicit_call_then_rearm() -> None:
    g = ComplianceGate()
    g.arm_window(HWND)
    g.audit_hand(assisted_live=True, hh_is_real_money=True)
    g.unlock()
    good = obs(window_title="Argent fictif", play_money_badge_detected=True)
    # L'audit a désarmé toutes les fenêtres : il faut ré-armer explicitement.
    assert g.evaluate(good).mode is Mode.REVIEW
    g.arm_window(HWND)
    assert g.evaluate(good).mode is Mode.LIVE


# ═══════════════════════════════════════════════════════════════════════
# PROPRIÉTÉ GLOBALE
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("armed", [True, False])
@pytest.mark.parametrize("badge", [None, True, False])
@pytest.mark.parametrize(
    "title", ["", "Table 5", "Play Money", "Tournoi 20€", "NL100", "Freeroll"]
)
@pytest.mark.parametrize("stacks", [(), ("100",), ("100 BB",), ("50 €",)])
def test_live_never_unlocks_without_armed_and_corroboration(
    armed: bool, badge: bool | None, title: str, stacks: tuple[str, ...]
) -> None:
    """Propriété : LIVE ⟹ (armé ∧ corroboration forte ∧ zéro contradiction)."""
    g = ComplianceGate()
    if armed:
        g.arm_window(HWND)
    d = g.evaluate(obs(window_title=title, stack_texts=stacks,
                       play_money_badge_detected=badge))
    if d.mode is Mode.LIVE:
        assert armed, "LIVE sans armement manuel"
        corroborated = badge is True or "play money" in title.lower()
        assert corroborated, "LIVE sans corroboration forte"
        blob = " ".join(stacks)
        assert "€" not in blob and "$" not in blob, "LIVE avec symbole monétaire"
