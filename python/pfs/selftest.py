"""Auto-vérification : rejoue les valeurs golden du Plan Directeur.

    python -m pfs --selftest

Sert de contrôle d'intégrité après installation, sans avoir besoin de pytest.
"""

from __future__ import annotations

import math
import sys

CHECKS: list[tuple[str, callable, float, float]] = []


def run_selftest() -> bool:
    import numpy as np

    from pfs.core.bankroll import bankroll_for_ror, risk_of_ruin
    from pfs.core.bluffcatch import analyse_bluffcatch, required_equity
    from pfs.core.equity import equity_vs_range
    from pfs.core.icm import (PkoSpot, analyse_pko_spot, icm_equities,
                              icm_required_equity, risk_premium)
    from pfs.core.range_model import (GTO_PRESETS, N_COMBOS, RANKS, SUITS,
                                      Range, parse_range)
    from pfs.solver.postflop import PostflopSolver
    from pfs.fusion.bottleneck import compress_range
    from pfs.fusion.dynamic_beta import DynamicBetaTracker
    from pfs.fusion.geometry import fisher_rao_distance
    from pfs.fusion.hmm import MentalState, Observation, OnlineHMM
    from pfs.solver.dcfr import DCFRSolver, KuhnPoker

    t = DynamicBetaTracker("vpip", discount=1.0)
    t.update_batch(successes=14, trials=40)
    b = t.belief
    h = OnlineHMM().update(Observation.WILD)
    a = analyse_bluffcatch(100.0, 75.0, 0.34, 0.09)
    r = DCFRSolver(KuhnPoker()).solve(400)
    rs = compress_range(parse_range(GTO_PRESETS["UTG"]), n_rules=8)

    def card(txt: str) -> int:
        return RANKS.index(txt[0]) * 4 + SUITS.index(txt[1])

    # L5 — golden PKO winner-take-all vérifié à la main
    pko = analyse_pko_spot(PkoSpot(
        stacks=(100.0, 0.0, 100.0), payouts=(100.0,),
        bounties=(50.0, 50.0, 50.0), hero=0, villain=1, pot=100.0, bet=100.0))
    # équité river exacte : tirage couleur 7/44
    eq = equity_vs_range([card("Ah"), card("Kh")], parse_range("99"),
                         [card("Qh"), card("7h"), card("2s"), card("9d")])
    # L1 — spot de polarisation : solution analytique EV_IP = 3P/4
    pf = PostflopSolver([card(x) for x in ("2s", "2d", "7h", "8h", "Kc")],
                        parse_range("QQ"), parse_range("AA,33"),
                        pot=100, stack=100, bet_fracs=(1.0,), max_bets=1)
    pf.solve(500)
    _, ev_ip = pf.values()

    checks = [
        ("F9  RoR(5,100,3000)", risk_of_ruin(5, 100, 3000), 0.0497870684, 1e-9),
        ("F9  bankroll RoR 1 %", bankroll_for_ror(5, 100, 0.01), 4605.17, 0.01),
        ("F10 pot odds 100/75", required_equity(100, 75), 0.30, 1e-12),
        ("F10 P(call +EV)", a.prob_call_is_plus_ev, 0.671639, 1e-6),
        ("F1  theta (14/40)", b.mean, 0.3536585, 1e-7),
        ("F1  sigma", b.std, 0.0737733, 1e-6),
        ("F2  HMM TILT", h[MentalState.TILT], 0.2409, 5e-4),
        ("F5  Fisher-Rao", fisher_rao_distance(
            [0.40, 0.25, 0.20, 0.10, 0.05], [0.55, 0.20, 0.15, 0.07, 0.03]),
         0.306904480, 1e-6),
        ("F8  valeur de Kuhn", r.game_value, -1 / 18, 3e-3),
        ("F8  exploitabilité", r.exploitability, 0.0, 2e-2),
        ("F6  fidélité UTG (8 règles)", rs.fidelity_ratio, 0.91, 0.03),
        ("F14 ICM $EV_A (50/30/20)", float(icm_equities(
            [50, 30, 20], [50, 30, 20])[0]), 38.392857, 1e-5),
        ("F14 alpha_ICM (BF=2)", icm_required_equity(100, 75, 2.0),
         150 / 325, 1e-12),
        ("F14 prime de risque (1.5)", risk_premium(1.5), 0.1, 1e-12),
        ("L5  PKO équité requise", pko.required_with_bounty, 4 / 13, 1e-9),
        ("L3  équité tirage (7/44)", eq.equity, 7 / 44, 1e-9),
        ("L1  polarisation EV_IP", ev_ip, 75.0, 0.8),
        ("L1  exploitabilité river", pf.exploitability(), 0.0, 6e-3),
        ("core 1326 combos", float(Range.full().n_combos), 1326.0, 0.0),
    ]

    ok = True
    # Une console Windows en cp1252 ne sait imprimer ni « ─ » ni « ✓ » : le
    # selftest plantait AVANT de vérifier quoi que ce soit (UnicodeEncodeError),
    # ce qui ressemblait à un échec des calculs alors que rien n'avait tourné.
    # On force un flux UTF-8 avec remplacement — le verdict passe avant l'ornement.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    print("─" * 62)
    print("  AUTO-VÉRIFICATION — valeurs golden du Plan Directeur v2.0")
    print("─" * 62)
    for name, got, want, tol in checks:
        good = abs(got - want) <= tol
        ok &= good
        mark = "✓" if good else "✗"
        print(f"  {mark} {name:<26} {got:>14.9g}  (attendu {want:.9g})")
    print("─" * 62)
    print(f"  {'TOUT EST CONFORME' if ok else 'ÉCHEC — voir ci-dessus'}")
    print("─" * 62)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if run_selftest() else 1)
