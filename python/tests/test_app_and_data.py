"""Tests de l'application : parseurs HH, entraînement, TDA, champ moyen, serveur."""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from pfs.app.server import create_server
from pfs.data.hand_history import (
    ActionType,
    HandHistoryError,
    Room,
    Street,
    detect_room,
    iter_hands,
    parse_pokerstars,
    parse_winamax,
    player_key,
)
from pfs.fusion.meanfield import (
    MeanFieldConfig,
    aggregate_baseline,
    multiway_equity_penalty,
    solve_mean_field,
)
from pfs.fusion.topology import detect_patterns, h0_persistence
from pfs.train.drill import (
    DrillSession,
    Grade,
    LeakFinder,
    SpacedRepetition,
    session_monitor,
)

WNMX = """Winamax Poker - CashGame - HandId: #12345-678-1234567890 - Holdem no limit (0.50€)/(1€) - 2026/08/06 20:14:11 UTC
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

PS = """PokerStars Hand #987654321:  Hold'em No Limit (€0.25/€0.50 EUR) - 2026/08/06 21:02:11 CET
Table 'Andromeda' 6-max Seat #2 is the button
Seat 1: Erin (50 in chips)
Seat 2: Frank (62.50 in chips)
Seat 3: Gina (48 in chips)
Erin: posts small blind 0.25
Frank: posts big blind 0.50
*** HOLE CARDS ***
Dealt to Gina [Qs Qh]
Gina: raises 1 to 1.50
Erin: folds
Frank: calls 1
*** FLOP *** [2h 8c Jd]
Frank: checks
Gina: bets 2
Frank: folds
Gina collected 3.50 from pot
*** SUMMARY ***
Total pot 3.50"""


# ═══════════════════════════════════════════════════════════════════════
# HAND HISTORY
# ═══════════════════════════════════════════════════════════════════════


def test_player_key_is_stable_and_not_reversible() -> None:
    a = player_key("Alice", "sel")
    assert a == player_key("alice", "sel") == player_key("  ALICE ", "sel")
    assert a != player_key("Alice", "autre-sel")
    assert "alice" not in a.lower()
    assert len(a) == 16


def test_detect_room() -> None:
    assert detect_room(WNMX) is Room.WINAMAX
    assert detect_room(PS) is Room.POKERSTARS
    assert detect_room("blah") is Room.UNKNOWN


def test_winamax_header_and_seats() -> None:
    h = parse_winamax(WNMX, salt="s")
    assert h.room is Room.WINAMAX
    assert h.hand_id == "12345-678-1234567890"
    assert h.big_blind == 1.0
    assert h.is_real_money is True
    assert h.is_tournament is False
    assert h.button_seat == 3
    assert len(h.seats) == 4
    assert [s.stack for s in h.seats] == [100.0, 85.5, 120.0, 97.0]


def test_winamax_hero_board_and_actions() -> None:
    h = parse_winamax(WNMX, salt="s")
    assert h.hero_cards == ("Ah", "Kd")
    assert h.board == ("Ks", "7d", "2c")
    assert [a.action for a in h.street_actions(Street.FLOP)] == [
        ActionType.CHECK, ActionType.BET, ActionType.FOLD
    ]
    assert h.pot == 7.0


def test_winamax_derived_stats() -> None:
    h = parse_winamax(WNMX, salt="s")
    bob, dave = h.seats[1].player, h.seats[3].player
    assert h.voluntarily_put_in_pot(bob) and h.preflop_raise(bob)
    assert h.voluntarily_put_in_pot(dave) and not h.preflop_raise(dave)
    assert h.faced_cbet(dave) and h.folded_to_cbet(dave)
    assert h.stat_observations(dave)["fold_to_cbet"] is True


def test_no_stat_is_emitted_without_opportunity() -> None:
    """Injecter un « fold to cbet = False » sans occasion biaiserait l'estimation."""
    h = parse_winamax(WNMX, salt="s")
    carol = h.seats[2].player       # a foldé préflop
    assert "fold_to_cbet" not in h.stat_observations(carol)


def test_pokerstars_parses() -> None:
    h = parse_pokerstars(PS, salt="s")
    assert h.room is Room.POKERSTARS
    assert h.hand_id == "987654321"
    assert h.big_blind == 0.50
    assert h.hero_cards == ("Qs", "Qh")
    assert h.board == ("2h", "8c", "Jd")
    assert len(h.seats) == 3


def test_iter_hands_multi_and_tolerates_garbage() -> None:
    text = WNMX + "\n\n" + PS + "\n\nWinamax Poker - corrompu\n\n" + WNMX
    hands = list(iter_hands(text, "s"))
    assert len(hands) == 3
    assert {h.room for h in hands} == {Room.WINAMAX, Room.POKERSTARS}


def test_unknown_room_raises() -> None:
    from pfs.data.hand_history import parse_text
    with pytest.raises(HandHistoryError):
        parse_text("ceci n'est pas un hand history")


# ═══════════════════════════════════════════════════════════════════════
# SM-2 / DRILLS
# ═══════════════════════════════════════════════════════════════════════


def test_sm2_intervals_follow_the_algorithm() -> None:
    s = SpacedRepetition()
    c = s.review("AA", Grade.PERFECT)
    assert c.interval_days == 1.0
    c = s.review("AA", Grade.PERFECT)
    assert c.interval_days == 6.0
    c = s.review("AA", Grade.PERFECT)
    assert c.interval_days == pytest.approx(6.0 * c.ease)


def test_sm2_failure_resets_interval_but_keeps_ease_floor() -> None:
    s = SpacedRepetition()
    for _ in range(3):
        s.review("KK", Grade.PERFECT)
    c = s.review("KK", Grade.BLACKOUT)
    assert c.interval_days == 0.0
    assert c.repetitions == 0
    assert c.ease >= 1.3


def test_sm2_ease_never_below_floor() -> None:
    s = SpacedRepetition()
    for _ in range(40):
        s.review("72o", Grade.BLACKOUT)
    assert s.card("72o").ease == pytest.approx(1.3)


def test_due_items_come_back_first() -> None:
    s = SpacedRepetition()
    s.review("A", Grade.PERFECT)      # dû dans 1 jour
    s.review("B", Grade.BLACKOUT)     # dû tout de suite
    due = [c.key for c in s.due()]
    assert due[0] == "B"
    s.advance(2.0)
    assert set(c.key for c in s.due()) == {"A", "B"}


def test_drill_grades_track_error() -> None:
    d = DrillSession(positions=("UTG",), difficulty="medium", seed=0)
    item = next(i for i in d.items if i.correct_frequency == 1.0)
    assert d.answer(item, 1.0).grade is Grade.PERFECT
    assert d.answer(item, 0.0).grade is Grade.BLACKOUT


def test_slow_correct_answer_is_downgraded() -> None:
    """Une bonne réponse lente signale une récupération laborieuse."""
    d = DrillSession(positions=("UTG",), seed=0)
    item = next(i for i in d.items if i.correct_frequency == 1.0)
    fast = d.answer(item, 1.0, seconds=1.0)
    slow = d.answer(item, 1.0, seconds=20.0)
    assert slow.grade < fast.grade


def test_leak_finder_detects_a_planted_bias() -> None:
    d = DrillSession(positions=("UTG", "BTN"), seed=1)
    rng = np.random.default_rng(0)
    for _ in range(180):
        it = d.next_item()
        bias = 0.25 if it.group.endswith("o") else 0.0
        d.answer(it, float(np.clip(it.correct_frequency + bias, 0, 1)), seconds=3.0)
    leaks = LeakFinder.analyse(d.answers)
    assert leaks
    assert any("offsuit" in l.label for l in leaks)
    assert all(l.direction == "trop large" for l in leaks if "offsuit" in l.label)
    # Trié par coût en EV, pas par fréquence d'erreur.
    assert leaks == sorted(leaks, key=lambda x: -x.ev_loss_bb)


def test_cognitive_state_refuses_to_guess_on_thin_sample() -> None:
    d = DrillSession(positions=("UTG",), seed=0)
    for _ in range(5):
        d.answer(d.next_item(), 0.5, seconds=3.0)
    st = session_monitor(d.answers)
    assert st.decision_time_drift is None
    assert "insuffisant" in st.advice


def test_cognitive_state_detects_slowdown() -> None:
    d = DrillSession(positions=("UTG",), seed=0)
    for i in range(60):
        d.answer(d.next_item(), 0.5, seconds=2.0 + i * 0.2)   # ralentissement net
    st = session_monitor(d.answers)
    assert st.decision_time_drift is not None and st.decision_time_drift > 0.05


def test_only_close_keeps_mixed_spots_only() -> None:
    d = DrillSession(positions=("BTN",), only_close=True, seed=0)
    assert d.items and all(i.is_mixed for i in d.items)


# ═══════════════════════════════════════════════════════════════════════
# TDA — le test qui empêche les faux exploits
# ═══════════════════════════════════════════════════════════════════════


def test_h0_persistence_has_n_classes() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(40, 3))
    diag = h0_persistence(pts)
    assert len(diag) == 40
    assert sum(1 for p in diag if math.isinf(p.persistence)) == 1


def test_pure_noise_is_never_declared_significant() -> None:
    """LE test central : sans lui, la TDA fabrique des exploits inexistants."""
    for seed in range(4):
        rng = np.random.default_rng(seed)
        res = detect_patterns(rng.normal(size=(120, 5)), n_permutations=60,
                              n_tests=1, seed=seed)
        assert res.significant is False
        assert res.p_value_h0 > 0.01


def test_bonferroni_tightens_the_threshold() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 4))
    one = detect_patterns(x, n_permutations=40, n_tests=1, seed=0)
    many = detect_patterns(x, n_permutations=40, n_tests=20, seed=0)
    assert many.alpha == pytest.approx(one.alpha / 20)


def test_p_value_is_bounded_away_from_zero() -> None:
    """Estimateur (1+k)/(1+B) : jamais p = 0, ce qui serait une sur-affirmation."""
    rng = np.random.default_rng(2)
    res = detect_patterns(rng.normal(size=(60, 3)), n_permutations=30, seed=0)
    assert res.p_value_h0 >= 1.0 / 31.0


def test_rejects_too_few_hands() -> None:
    with pytest.raises(ValueError):
        detect_patterns(np.zeros((5, 3)))


# ═══════════════════════════════════════════════════════════════════════
# CHAMP MOYEN
# ═══════════════════════════════════════════════════════════════════════


def test_multiway_equity_penalty_golden() -> None:
    assert multiway_equity_penalty(0.70, 3) == pytest.approx(0.343, abs=1e-6)
    assert multiway_equity_penalty(0.70, 1) == pytest.approx(0.70)
    assert multiway_equity_penalty(1.0, 5) == pytest.approx(1.0)


def test_mean_field_finds_the_rps_equilibrium() -> None:
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    res = solve_mean_field(lambda s, f: A @ f, 3, MeanFieldConfig(n_opponents=3))
    assert np.allclose(res.strategy, 1 / 3, atol=0.05)
    assert res.strategy.sum() == pytest.approx(1.0)


def test_mean_field_reports_its_own_approximation_error() -> None:
    """Le module doit dire quand il ne sert à rien — c'est le point."""
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    res = solve_mean_field(lambda s, f: A @ f, 3, MeanFieldConfig(n_opponents=3))
    assert res.approximation_error_bound == pytest.approx(0.5)
    assert res.worth_the_complexity is False


def test_mean_field_strategy_is_always_a_distribution() -> None:
    rng = np.random.default_rng(4)
    for _ in range(5):
        M = rng.normal(size=(4, 4))
        res = solve_mean_field(lambda s, f: M @ f, 4)
        assert res.strategy.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(res.strategy >= 0)


def test_aggregate_baseline_is_a_distribution() -> None:
    A = np.eye(3)
    s = aggregate_baseline(lambda x, f: A @ f, 3, 2)
    assert s.sum() == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════
# SERVEUR
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def server():
    srv, token = create_server(0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", token
    srv.shutdown()
    srv.server_close()


def _post(base: str, token: str, route: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base}/api/{route}?t={token}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def test_server_binds_loopback_only(server) -> None:
    base, _ = server
    assert base.startswith("http://127.0.0.1:")


def test_server_serves_ui_with_token_injected(server) -> None:
    base, token = server
    html = urllib.request.urlopen(f"{base}/", timeout=10).read().decode()
    assert "__TOKEN__" not in html
    assert token in html
    assert "Poker Fusion Solver" in html


def test_server_rejects_missing_token(server) -> None:
    base, _ = server
    req = urllib.request.Request(f"{base}/api/range", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 403


def test_server_rejects_wrong_token(server) -> None:
    base, _ = server
    req = urllib.request.Request(f"{base}/api/range?t=nope", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 403


def test_api_range_and_rules(server) -> None:
    base, token = server
    r = _post(base, token, "range", {"position": "BTN"})
    assert len(r["groups"]) == 169
    assert 500 < r["combos"] < 560
    rules = _post(base, token, "range/rules", {"position": "UTG"})
    assert rules["fidelity"] > 0.9
    assert len(rules["reconstructed"]) == 169


def test_api_fusion_matches_the_core(server) -> None:
    base, token = server
    r = _post(base, token, "fusion", {"observed": 0.75, "n": 400})
    assert r["significant"] is True
    assert r["bet"] > r["gto_bet"]
    thin = _post(base, token, "fusion", {"observed": 0.75, "n": 10})
    assert thin["significant"] is False


def test_api_bankroll_golden(server) -> None:
    base, token = server
    r = _post(base, token, "bankroll",
              {"winrate": 5, "stddev": 100, "bankroll": 3000, "hands": 10000})
    assert r["ror"] == pytest.approx(0.0497870684, abs=1e-9)
    assert r["ci_contains_zero"] is True


def test_api_solve_converges(server) -> None:
    base, token = server
    r = _post(base, token, "solve", {"iterations": 300})
    assert r["game_value"] == pytest.approx(-1 / 18, abs=5e-3)
    assert r["exploitability"] < 5e-2


def test_api_drill_round_trip(server) -> None:
    base, token = server
    q = _post(base, token, "drill/start", {"difficulty": "medium"})
    assert "group" in q
    a = _post(base, token, "drill/answer", {"given": 0.5, "seconds": 2.0})
    assert "grade_name" in a and "next" in a
    rep = _post(base, token, "drill/report", {})
    assert rep["score"]["n"] >= 1


def test_api_analyse_hand_history(server) -> None:
    base, token = server
    r = _post(base, token, "analyse", {"text": WNMX})
    assert r["n_hands"] == 1
    assert r["real_money"] == 1
    assert len(r["players"]) == 4


def test_api_unknown_route_is_404(server) -> None:
    base, token = server
    req = urllib.request.Request(f"{base}/api/nope?t={token}", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 404


def test_api_errors_are_surfaced_not_swallowed(server) -> None:
    base, token = server
    req = urllib.request.Request(
        f"{base}/api/fusion?t={token}", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=10)
    assert exc.value.code == 400
    assert "error" in json.loads(exc.value.read())


# ═══════════════════════════════════════════════════════════════════════
# ROUTE /api/postflop : nodelock (P2), rake (L8), feuille P3/EQR (L7)
# ═══════════════════════════════════════════════════════════════════════
#
# Ces tests parlent au VRAI serveur en HTTP : ce sont les jointures
# route ↔ solveur qui sont éprouvées ici, pas le solveur (déjà couvert par
# tests/test_postflop_advanced.py et tests/test_rake.py en direct).


def _post_longue(base: str, token: str, route: str,
                 payload: dict) -> tuple[int, dict]:
    """POST avec délai long (solves, entraînement EQR) ; les erreurs HTTP ne
    lèvent pas — leur corps JSON est la réponse que l'interface reçoit."""
    req = urllib.request.Request(
        f"{base}/api/{route}?t={token}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as rep:
            return rep.status, json.loads(rep.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


#: Spot de polarisation, le même que tests/test_rake.py : AA+33 (polarisé)
#: contre QQ (bluff-catcheur), river sèche, pot 100, mise pot unique.
_SPOT_POLAR = {"board": "2s 2d 7h 8h Kc", "pot": 100,
               "bet_fracs": [1.0], "max_bets": 1}


def test_api_postflop_nodelock_change_le_noeud_et_resolve_autour(server) -> None:
    """Nodelock 2.0 par la route : le nœud verrouillé rend la stratégie
    forcée, et le joueur NON verrouillé re-solve librement autour.

    QQ (OOP) verrouillé à 90 % de fold face au bet : l'IP re-solvé bluffe
    tout son air (fréquence de mise au nœud IP en forte hausse) et son EV
    dépasse nettement le Nash — le comportement déjà prouvé du solveur
    (test_overfolding_villain_gets_exploited), vérifié ICI à travers HTTP.
    """
    base, token = server
    spot = {**_SPOT_POLAR, "oop_range": "QQ", "ip_range": "AA,33",
            "stack": 300, "iterations": 600, "nodes": [["check"]]}
    code, nash = _post_longue(base, token, "postflop", spot)
    assert code == 200, nash
    code, locke = _post_longue(base, token, "postflop", {
        **spot,
        "locks": [{"path": ["check", "bet 1p"],
                   "strategy": {"fold": 0.9, "call": 0.1}}],
    })
    assert code == 200, locke

    # 1) le nœud verrouillé rend EXACTEMENT la stratégie forcée
    (verrou,) = locke["locks"]
    assert verrou["path"] == ["check", "bet 1p"] and verrou["player"] == "OOP"
    assert verrou["frequencies"]["fold"] == pytest.approx(0.9, abs=1e-9)
    assert verrou["frequencies"]["call"] == pytest.approx(0.1, abs=1e-9)

    # 2) le non-verrouillé re-solve autour : l'IP exploite l'overfold
    assert locke["ev_ip"] > nash["ev_ip"] + 5.0, (
        f"EV IP verrouillée {locke['ev_ip']:.1f} ≤ Nash {nash['ev_ip']:.1f}")

    def freq_bet(rep: dict) -> float:
        """Fréquence de mise au nœud IP après check (le nœud demandé)."""
        (noeud,) = rep["nodes"]
        assert noeud["player"] == "IP"
        return next(v for k, v in noeud["frequencies"].items() if k != "check")

    assert freq_bet(locke) > freq_bet(nash) + 0.1, (
        "l'IP n'a pas ajusté ses bluffs autour du lock")

    # 3) sans locks, la réponse n'en prétend pas
    assert "locks" not in nash


def test_api_postflop_lock_chemin_inconnu_est_nomme(server) -> None:
    """Un chemin de lock inexistant → 400, l'action absente est NOMMÉE."""
    base, token = server
    code, rep = _post_longue(base, token, "postflop", {
        **_SPOT_POLAR, "oop_range": "QQ", "ip_range": "AA,33", "stack": 300,
        "iterations": 50,
        "locks": [{"path": ["bet 7p"], "strategy": {"fold": 1.0}}],
    })
    assert code == 400
    assert "bet 7p" in rep.get("error", ""), rep


def test_api_postflop_rake_baisse_les_ev_et_deplace_l_equilibre(server) -> None:
    """Rake par la route : EV en baisse, bluffs en HAUSSE, calls EFFONDRÉS.

    Le piège n°4 de PASSATION.md, vérifié en forme close dans
    tests/test_rake.py : β* = b/net(P+2b) > b/(P+2b) (bluffs ↑) et
    c* = 1 − b/net(P+b) < 1/2 (calls ↓). Ici le polarisé est mis OOP pour
    que ses bluffs se lisent à la racine de la réponse, et le nœud
    ``["bet 1p"]`` est demandé pour lire la fréquence de call de QQ.
    Stack 300 : le bet pot (100) garde son label « bet 1p » au lieu d'être
    requalifié « all-in » (convention du solveur quand la mise = le tapis).
    """
    base, token = server
    spot = {**_SPOT_POLAR, "oop_range": "AA,33", "ip_range": "QQ",
            "stack": 300, "iterations": 800, "nodes": [["bet 1p"]]}
    code, sans = _post_longue(base, token, "postflop", spot)
    assert code == 200, sans
    code, avec = _post_longue(base, token, "postflop", {
        **spot, "rake": {"pct": 0.2, "cap": 1000.0}})
    assert code == 200, avec

    # 1) l'EV rendue baisse, pour les deux joueurs, et la comptabilité est
    #    exacte : pot − (EV_OOP + EV_IP) = E[rake] (identité, pas convergence)
    assert avec["ev_oop"] < sans["ev_oop"] - 1.0
    assert avec["ev_ip"] < sans["ev_ip"] - 1.0
    assert avec["rake"]["expected_rake"] > 0.0
    assert avec["ev_oop"] + avec["ev_ip"] + avec["rake"]["expected_rake"] == (
        pytest.approx(100.0, abs=1e-6))
    assert "rake" not in sans, "sans rake demandé, la réponse n'en publie pas"

    # 2) les bluffs MONTENT à l'équilibre (33 à la racine : ~0.50 → ~0.71)
    def bluff_moyen(rep: dict) -> float:
        (bet,) = [a for a in rep["root_actions"] if a["label"] != "check"]
        vals = [v for k, v in bet["per_combo"].items()
                if k[0] == "3" and k[2] == "3"]
        assert vals, f"aucun combo 33 dans {sorted(bet['per_combo'])}"
        return sum(vals) / len(vals)
    assert bluff_moyen(avec) > bluff_moyen(sans) + 0.1

    # 3) les calls S'EFFONDRENT (QQ face au bet : ~0.50 → ~0.375)
    def call_qq(rep: dict) -> float:
        (noeud,) = rep["nodes"]
        assert noeud["player"] == "IP"
        return noeud["frequencies"]["call"]
    assert call_qq(avec) < call_qq(sans) - 0.05


def test_api_postflop_rake_invalide_est_refuse(server) -> None:
    base, token = server
    charge = {**_SPOT_POLAR, "oop_range": "QQ", "ip_range": "AA,33",
              "stack": 100, "iterations": 50}
    code, rep = _post_longue(base, token, "postflop",
                             {**charge, "rake": {"cap": 1.0}})
    assert code == 400 and "pct" in rep.get("error", "")
    code, rep = _post_longue(base, token, "postflop",
                             {**charge, "rake": {"pct": 5, "cap": 1.0}})
    assert code == 400, "pct=5 (au lieu de 0.05) doit être refusé"


def test_api_postflop_leaf_model_eqr_est_atteignable_et_honnete(server) -> None:
    """La feuille EQR (L7) par la route : entraînée au premier usage,
    mémoïsée, et sa limite MESURÉE republiée dans la réponse.

    Contrat DIRECTIONNEL seulement : l'EQR relève l'EV du joueur en position
    par rapport au rollout nu (coefficient IP > 0), avec une dérive de la
    somme des EV bornée. Le « plus proche du solve complet que le rollout »
    du test direct (test_eqr_leaf_corrects_toward_full) ne vaut que pour SON
    petit modèle (8 spots) : mesuré ici, le modèle canonique de la route
    (train_eqr() par défaut, 24 spots) SUR-corrige sur ce spot artificiel
    (|EV−full| 19,0 contre 14,2 pour le rollout) — c'est exactement la
    limite que la réponse doit publier, et ce test vérifie qu'elle l'est.
    """
    base, token = server
    spot = {"board": "2s 2d 7h 8h", "oop_range": "QQ, 99",
            "ip_range": "KK, 55", "pot": 60, "stack": 180,
            "bet_fracs": [0.75], "max_bets": 1, "iterations": 150}
    code, full = _post_longue(base, token, "postflop", spot)
    assert code == 200, full
    code, roll = _post_longue(base, token, "postflop",
                              {**spot, "leaf_model": "rollout"})
    assert code == 200, roll
    code, eqr = _post_longue(base, token, "postflop",
                             {**spot, "leaf_model": "eqr"})
    assert code == 200, eqr

    # rollout : somme-exacte, arbre effondré, limite annoncée
    assert roll["ev_oop"] + roll["ev_ip"] == pytest.approx(60.0, abs=1e-6)
    assert roll["n_nodes"] < full["n_nodes"] / 20
    assert roll["leaf"]["model"] == "rollout" and "limite" in roll["leaf"]

    # eqr : directionnel (position ↑ vs rollout), dérive de la somme bornée
    assert eqr["ev_ip"] > roll["ev_ip"]
    assert abs(eqr["ev_oop"] + eqr["ev_ip"] - 60.0) < 0.5 * 60.0

    # la limite est PUBLIÉE avec le R² et le n mesurés du modèle entraîné,
    # y compris l'absence de garantie d'écart au solve complet
    assert eqr["leaf"]["model"] == "eqr"
    assert 0.0 < eqr["leaf"]["r2"] < 1.0 and eqr["leaf"]["n"] >= 8
    assert "DIRECTIONNELLE" in eqr["leaf"]["limite"]
    assert "sur-corriger" in eqr["leaf"]["limite"]
    assert f"{eqr['leaf']['r2']:.2f}" in eqr["leaf"]["limite"]

    # le mode complet, lui, n'annonce aucune feuille approchée
    assert "leaf" not in full


def test_api_postflop_leaf_model_refuse_la_river(server) -> None:
    """Profondeur limitée = board turn : sur une river, refus nommé."""
    base, token = server
    code, rep = _post_longue(base, token, "postflop", {
        **_SPOT_POLAR, "oop_range": "QQ", "ip_range": "AA,33",
        "stack": 100, "iterations": 50, "leaf_model": "rollout"})
    assert code == 400
    assert "turn" in rep.get("error", ""), rep


# ═══════════════════════════════════════════════════════════════════════
# PRIORS EXTERNES (SharkScope / OPR)
# ═══════════════════════════════════════════════════════════════════════

from pfs.fusion.particle import Archetype as _Arch
from pfs.fusion.skill_prior import (
    ExternalRating,
    GameFormat,
    RatingSource,
    adaptation_propensity_from_skill,
    archetype_prior_from_skill,
    estimate_skill,
    tournaments_needed,
)


def test_tournaments_needed_golden() -> None:
    """Le chiffre qui relativise tous les ROI affichés."""
    assert tournaments_needed(0.10) == 1766
    assert tournaments_needed(0.05) == 7064
    assert tournaments_needed(0.20) < tournaments_needed(0.10)


def test_small_sample_roi_is_heavily_shrunk() -> None:
    """+40 % sur 200 MTT n'est pas un ROI de 40 % : 64 % du signal est du bruit."""
    e = estimate_skill(ExternalRating(
        RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE, 200, 0.40))
    assert e.shrunk_roi < 0.20
    assert e.shrinkage > 0.5
    assert "asymétrique" in e.verdict          # caveat sur petit échantillon


def test_large_sample_is_barely_shrunk() -> None:
    e = estimate_skill(ExternalRating(
        RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE, 12000, 0.08))
    assert e.shrinkage < 0.10
    assert e.shrunk_roi == pytest.approx(0.08, abs=0.01)
    assert e.significant is True


def test_zero_tournaments_gives_neutral_prior() -> None:
    e = estimate_skill(ExternalRating(RatingSource.OPR, GameFormat.SNG, 0, 0.0))
    assert e.skill == 0.5
    assert e.shrinkage == 1.0
    assert e.significant is False


def test_shrinkage_is_monotone_in_sample_size() -> None:
    prev = 1.1
    for n in (50, 200, 1000, 5000, 20000):
        e = estimate_skill(ExternalRating(
            RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE, n, 0.15))
        assert e.shrinkage < prev
        prev = e.shrinkage


def test_archetype_prior_shifts_with_skill_and_stays_a_distribution() -> None:
    weak = archetype_prior_from_skill(0.05)
    strong = archetype_prior_from_skill(0.95)
    for p in (weak, strong):
        assert sum(p.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(v > 0 for v in p.values())
    assert strong[_Arch.GTO] > weak[_Arch.GTO]
    assert weak[_Arch.STATION] > strong[_Arch.STATION]


def test_neutral_skill_leaves_prior_untouched() -> None:
    from pfs.fusion.particle import ARCHETYPE_PRIORS
    p = archetype_prior_from_skill(0.5)
    for k, v in ARCHETYPE_PRIORS.items():
        assert p[k] == pytest.approx(v, abs=1e-9)


def test_strength_zero_disables_the_external_prior() -> None:
    from pfs.fusion.particle import ARCHETYPE_PRIORS
    p = archetype_prior_from_skill(0.99, strength=0.0)
    for k, v in ARCHETYPE_PRIORS.items():
        assert p[k] == pytest.approx(v, abs=1e-9)


def test_adaptation_propensity_is_bounded() -> None:
    assert adaptation_propensity_from_skill(0.0) == pytest.approx(0.05)
    assert adaptation_propensity_from_skill(1.0) == pytest.approx(0.85)
    assert adaptation_propensity_from_skill(0.0) < adaptation_propensity_from_skill(1.0)


def test_module_makes_no_network_call() -> None:
    """Garde-fou architectural : aucun import réseau dans ce module."""
    import inspect
    from pfs.fusion import skill_prior
    src = inspect.getsource(skill_prior)
    for banned in ("requests", "urllib.request", "httpx", "socket", "aiohttp"):
        assert banned not in src


# ═══════════════════════════════════════════════════════════════════════
# BASE LOCALE DE PROFILS — enrichissement hors ligne, lookup en direct
# ═══════════════════════════════════════════════════════════════════════

from pfs.data.player_notes import STALE_AFTER_DAYS, PlayerNotes


def _notes() -> PlayerNotes:
    db = PlayerNotes(":memory:", salt="test")
    db.upsert("Fish", ExternalRating(RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE,
                                     5200, -0.19), note="fish")
    db.upsert("Reg", ExternalRating(RatingSource.SHARKSCOPE, GameFormat.MTT_LARGE,
                                    8400, 0.11), note="reg")
    return db


def test_notes_roundtrip_and_hashing() -> None:
    db = _notes()
    p = db.lookup("Fish")
    assert p is not None and p.note == "fish"
    assert "fish" not in p.key.lower()          # pseudo jamais en clair
    assert db.lookup("fish") is not None        # insensible à la casse
    assert db.lookup("Inconnu") is None
    assert len(db) == 2


def test_notes_lookup_many_is_the_table_call() -> None:
    db = _notes()
    found = db.lookup_many(["Fish", "Reg", "X", "Y", "Z", "W"])
    assert set(found) == {"Fish", "Reg"}


def test_losing_player_gets_low_adaptation_propensity() -> None:
    """Un perdant établi ne te contre-exploitera pas — c'est le ρ de F13."""
    db = _notes()
    assert db.lookup("Fish").rho < 0.15
    assert db.lookup("Reg").rho > 0.60


def test_losing_player_prior_favours_station() -> None:
    db = _notes()
    prior = db.lookup("Fish").archetype_prior
    assert max(prior, key=prior.get) is _Arch.STATION


def test_prior_is_washed_out_by_observation() -> None:
    """Propriété essentielle : ce que tu observes doit reprendre la main."""
    db = _notes()
    p = db.lookup("Fish")
    assert p.prior_weight(0) == pytest.approx(1.0)
    assert p.prior_weight(50) == pytest.approx(0.5)
    assert p.prior_weight(200) < 0.07
    weights = [p.prior_weight(n) for n in (0, 25, 50, 100, 200)]
    assert weights == sorted(weights, reverse=True)


def test_blended_prior_converges_to_neutral() -> None:
    from pfs.fusion.particle import ARCHETYPE_PRIORS
    db = _notes()
    far = db.blended_prior("Fish", 1000)
    for k, v in ARCHETYPE_PRIORS.items():
        assert far[k] == pytest.approx(v, abs=0.02)
    assert sum(far.values()) == pytest.approx(1.0)


def test_unknown_player_gets_neutral_prior() -> None:
    from pfs.fusion.particle import ARCHETYPE_PRIORS
    db = _notes()
    p = db.blended_prior("Jamais_vu", 0)
    for k, v in ARCHETYPE_PRIORS.items():
        assert p[k] == pytest.approx(v)


def test_stale_profiles_are_flagged_and_downweighted() -> None:
    import time as _t
    db = _notes()
    old = _t.time() - (STALE_AFTER_DAYS + 10) * 86400
    db._db.execute("UPDATE players SET updated_at = ?", (old,))
    db._db.commit()
    p = db.lookup("Fish")
    assert p.is_stale is True
    assert p.prior_weight(0) == pytest.approx(0.5)   # pénalisé de moitié


def test_csv_import_export_round_trip(tmp_path) -> None:
    src = tmp_path / "in.csv"
    src.write_text(
        "nickname,source,format,tournaments,roi,note\n"
        "Alpha,sharkscope,mtt_large,4000,12,bon reg\n"      # ROI en %
        "Beta,opr,sng,900,-0.08,perdant\n"                   # ROI en fraction
        "Gamma,manual,spin,50,,\n",
        encoding="utf-8")
    db = PlayerNotes(":memory:", salt="t")
    n, errors = db.import_csv(src)
    assert n == 3 and not errors
    assert db.lookup("Alpha").observed_roi == pytest.approx(0.12)
    assert db.lookup("Beta").observed_roi == pytest.approx(-0.08)

    out = tmp_path / "out.csv"
    assert db.export_csv(out) == 3
    text = out.read_text(encoding="utf-8")
    assert "Alpha" not in text and "Beta" not in text   # export anonymisé


def test_csv_import_reports_bad_rows_without_aborting() -> None:
    import tempfile, pathlib as _pl
    with tempfile.TemporaryDirectory() as d:
        f = _pl.Path(d) / "x.csv"
        f.write_text("nickname,source,format,tournaments,roi\n"
                     "Ok,sharkscope,mtt_large,100,10\n"
                     ",sharkscope,mtt_large,100,10\n"
                     "Bad,inexistant,mtt_large,100,10\n", encoding="utf-8")
        db = PlayerNotes(":memory:", salt="t")
        n, errors = db.import_csv(f)
        assert n == 1 and len(errors) == 2


def test_notes_module_makes_no_network_call() -> None:
    """Garde-fou architectural : le lookup en direct ne sort jamais."""
    import inspect
    from pfs.data import player_notes
    src = inspect.getsource(player_notes)
    for banned in ("requests", "urllib.request", "httpx", "aiohttp", "socket.socket"):
        assert banned not in src


def test_engine_applies_local_prior_from_nickname() -> None:
    from pfs.engine import FusionEngine
    db = _notes()
    eng = FusionEngine(notes=db)
    prof = eng.bind_nickname("seat3", "Fish")
    assert prof is not None
    eng.start_hand("seat3")
    arch = eng.belief("seat3").archetypes
    assert arch["station"] > arch["gto"]


def test_engine_without_notes_still_works() -> None:
    from pfs.engine import FusionEngine
    eng = FusionEngine()
    assert eng.bind_nickname("s1", "Quiconque") is None
    eng.start_hand("s1")
    assert sum(eng.belief("s1").archetypes.values()) == pytest.approx(1.0, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# FILE D'ENRICHISSEMENT — les deux verrous
# ═══════════════════════════════════════════════════════════════════════

from pfs.data.player_notes import EnrichmentQueue, ManualProvider


def _queue():
    db = PlayerNotes(":memory:", salt="t")
    q = EnrichmentQueue(db, min_interval_s=0.0)
    prov = ManualProvider({
        "adversaire": ExternalRating(RatingSource.SHARKSCOPE,
                                     GameFormat.MTT_LARGE, 5200, -0.19),
        "spectateur": ExternalRating(RatingSource.SHARKSCOPE,
                                     GameFormat.MTT_LARGE, 900, 0.05),
    })
    return db, q, prov


def test_lock_b_participation_required() -> None:
    """Verrou (b) : un joueur jamais croisé dans un pot est refusé."""
    db, q, prov = _queue()
    assert q.enqueue("Adversaire") is False
    q.hand_ended("T1", ["Adversaire"])
    assert q.enqueue("Adversaire") is True
    assert q.enqueue("Spectateur") is False        # jamais dans un pot
    assert q.pending == ("Adversaire",)


def test_lock_a_no_flush_while_a_hand_is_live() -> None:
    """Verrou (a) : rien ne part tant qu'une main est vivante."""
    db, q, prov = _queue()
    q.hand_ended("T1", ["Adversaire"])
    q.enqueue("Adversaire")
    q.hand_started("T2")
    n, msg = q.flush(prov)
    assert n == 0 and "Main en cours" in msg
    assert len(db) == 0
    q.hand_ended("T2", [])
    n, _ = q.flush(prov)
    assert n == 1 and len(db) == 1


def test_multi_table_all_hands_must_be_over() -> None:
    db, q, prov = _queue()
    q.hand_ended("T1", ["Adversaire"])
    q.enqueue("Adversaire")
    q.hand_started("T1"); q.hand_started("T2")
    q.hand_ended("T1", [])
    assert q.has_live_hand is True                 # T2 encore vivante
    assert q.flush(prov)[0] == 0
    q.hand_ended("T2", [])
    assert q.flush(prov)[0] == 1


def test_already_known_player_is_not_requeued() -> None:
    db, q, prov = _queue()
    q.hand_ended("T1", ["Adversaire"])
    q.enqueue("Adversaire"); q.flush(prov)
    assert q.enqueue("Adversaire") is False
    assert q.pending == ()


def test_rate_limit_between_flushes() -> None:
    db = PlayerNotes(":memory:", salt="t")
    q = EnrichmentQueue(db, min_interval_s=60.0)
    prov = ManualProvider({"a": ExternalRating(
        RatingSource.SHARKSCOPE, GameFormat.SNG, 900, 0.05)})
    q.hand_ended("T", ["A", "B"])
    q.enqueue("A")
    assert q.flush(prov)[0] == 1
    q.enqueue("B")
    n, msg = q.flush(prov)
    assert n == 0 and "Cadence" in msg


def test_audit_log_records_refusals_and_successes() -> None:
    db, q, prov = _queue()
    q.enqueue("Adversaire")                        # refusé : pas de participation
    q.hand_ended("T1", ["Adversaire"])
    q.enqueue("Adversaire")
    q.flush(prov)
    summary = q.audit_summary()
    assert summary.get("refusé", 0) >= 1
    assert summary.get("enrichi", 0) == 1
    assert all("Adversaire" not in e.nickname_key for e in q.audit_log)


def test_missing_rating_is_logged_not_crashed() -> None:
    db, q, _ = _queue()
    empty = ManualProvider({})
    q.hand_ended("T", ["Fantome"])
    q.enqueue("Fantome")
    n, _ = q.flush(empty)
    assert n == 0
    assert q.audit_summary().get("introuvable") == 1


def test_queue_itself_makes_no_network_call() -> None:
    import inspect
    from pfs.data import player_notes
    src = inspect.getsource(player_notes.EnrichmentQueue)
    for banned in ("requests", "urlopen", "httpx", "socket"):
        assert banned not in src
