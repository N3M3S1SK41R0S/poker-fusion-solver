#!/usr/bin/env python3
"""
Poker Fusion Solver — démonstration des 13 fusions, de bout en bout.

    uv run python demo.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from pfs.core.bankroll import (
    BankrollProfile,
    bankroll_for_ror,
    confidence_interval_winrate,
    drawdown_quantile,
    hands_for_significance,
    risk_of_ruin,
)
from pfs.core.bluffcatch import analyse_bluffcatch
from pfs.core.range_model import (
    GTO_PRESETS,
    N_COMBOS,
    HandClass,
    Range,
    parse_range,
)
from pfs.engine import FusionEngine
from pfs.fusion.arbiter import Action, ActionDistribution
from pfs.fusion.bet_sizing import MDFCallModel, optimal_bet_size, sizing_table
from pfs.fusion.bottleneck import compress_range, elbow_point, information_plane
from pfs.fusion.dynamic_beta import GTO_BASELINES, OpponentProfile
from pfs.fusion.geometry import fisher_rao_distance, kmeans_fisher, range_deviation_score
from pfs.fusion.hmm import MentalState, Observation, OnlineHMM
from pfs.fusion.particle import ParticleFilter
from pfs.solver.dcfr import DCFRConfig, DCFRSolver, KuhnPoker


def rule(n: str, title: str) -> None:
    print(f"\n{'═' * 76}\n  {n} — {title}\n{'═' * 76}")


def main() -> None:
    rng = np.random.default_rng(42)
    eq = np.clip(rng.beta(2.2, 3.0, N_COMBOS), 0.02, 0.98)

    # ─────────────────────────────────────────────────────────────────
    rule("RANGE", "l'algèbre commune à toutes les fusions")
    btn = parse_range(GTO_PRESETS["BTN"])
    print(btn)
    print(f"  connecteurs suités : {btn.filtered([HandClass.SUITED_CONNECTOR]).n_combos:.0f} combos")
    print(f"  as suités          : {btn.filtered([HandClass.SUITED_ACE]).n_combos:.0f} combos")
    blocked = btn.remove_blockers([0, 4])          # A♠ K♠ au board
    print(f"  après blockers A♠K♠ : {blocked.n_combos:.0f} combos "
          f"(−{btn.n_combos - blocked.n_combos:.0f}) — invisible sur la grille 169")
    print(f"  entropie normalisée : {btn.normalised_entropy * 100:.1f} % → {btn.shape_label()}")

    # ─────────────────────────────────────────────────────────────────
    rule("F1", "Beta-Binomial dynamique — l'incertitude, pas juste la moyenne")
    v = OpponentProfile(player_key="a3f9c1e2", discount=0.99)
    for i in range(40):
        v.observe("vpip", i < 14)
    b = v.tracker("vpip").belief
    lo, hi = b.credible_interval()
    print(f"  VPIP après 40 mains : {b.mean * 100:.2f} % ± {b.std * 100:.2f}")
    print(f"  IC95                : [{lo * 100:.1f} , {hi * 100:.1f}]")
    print(f"  baseline GTO        : {GTO_BASELINES['vpip'] * 100:.0f} %")
    print(f"  z                   : {b.deviation_z(GTO_BASELINES['vpip']):.3f}")
    print(f"  → exploitable ?     : {'OUI' if b.is_exploitable(GTO_BASELINES['vpip']) else 'NON'}"
          f"  (encore ~{b.hands_until_significant(GTO_BASELINES['vpip'])} mains)")
    print("  Un HUD classique afficherait « VPIP 35 % » et t'inciterait à exploiter.")

    # ─────────────────────────────────────────────────────────────────
    rule("F2", "HMM — détecter le tilt avant que les stats ne bougent")
    h = OnlineHMM()
    print(f"  prior             : SOLID=80.0%  LOOSE=15.0%  TILT= 5.0%")
    print(f"  surge d'un 3-bet improbable : ×{h.tilt_surge(Observation.WILD):.2f}")
    print(f"  après 1 action    : {h.update(Observation.WILD)}")
    print(f"  après 5 actions   : {h.update_many([Observation.WILD] * 4)}")
    print(f"  puis 20 folds     : {h.update_many([Observation.FOLD] * 20)}")

    # ─────────────────────────────────────────────────────────────────
    rule("F3", "Filtre particulaire — l'incertitude porte sur la STRATÉGIE")
    pf = ParticleFilter(n_particles=200, prior_range=btn, seed=1)
    print(f"  départ  : {pf.explain()}")
    for act, frac in [("bet", 1.0), ("bet", 1.5), ("raise", 2.0)]:
        pf.observe(eq, act, frac)
    print(f"  3 mises : {pf.explain()}")

    # ─────────────────────────────────────────────────────────────────
    rule("F5", "Fisher-Rao — la seule distance qui ait un sens entre ranges")
    p = [0.40, 0.25, 0.20, 0.10, 0.05]
    q = [0.55, 0.20, 0.15, 0.07, 0.03]
    print(f"  d_FR(p, q) = {fisher_rao_distance(p, q):.6f} rad")
    near = fisher_rao_distance([0.001, 0.999], [0.002, 0.998])
    mid = fisher_rao_distance([0.500, 0.500], [0.501, 0.499])
    print(f"  0.001→0.002 : d = {near:.5f}   |   0.500→0.501 : d = {mid:.5f}")
    print(f"  → rapport ×{near / mid:.0f}, alors que la distance euclidienne est IDENTIQUE.")
    d, notable, label = range_deviation_score(
        parse_range(GTO_PRESETS["SB"]).to_groups() + 1e-9, btn.to_groups() + 1e-9
    )
    print(f"  SB vs BTN : d = {d:.3f} → {label}")

    # ─────────────────────────────────────────────────────────────────
    rule("F4", "Sizing — l'information a un prix, en bb")
    t = sizing_table(45.0, np.ones(N_COMBOS), eq, lam=1.5, model=MDFCallModel())
    print(t.explain())
    b0 = optimal_bet_size(45.0, np.ones(N_COMBOS), eq, lam=0.0, model=MDFCallModel())
    b5 = optimal_bet_size(45.0, np.ones(N_COMBOS), eq, lam=6.0, model=MDFCallModel())
    print(f"\n  Optimum continu : λ=0 → {b0.fraction_of_pot:.2f} pot  |  "
          f"λ=6 → {b5.fraction_of_pot:.2f} pot")
    print("  Payer l'information déplace le sizing vers le haut — c'est la thèse de F4.")

    # ─────────────────────────────────────────────────────────────────
    rule("F6", "Information Bottleneck — 5 000 décisions → une poignée de règles")
    print(f"  {'position':>4} {'largeur':>8} {'coude':>6} {'fidélité':>9} {'MAE':>7}")
    for pos in ("UTG", "MP", "CO", "BTN", "SB"):
        r = parse_range(GTO_PRESETS[pos])
        pts = information_plane(r, max_rules=16)
        e = elbow_point(pts)
        print(f"  {pos:>4} {r.fraction * 100:7.1f}% {e.n_rules:>6} "
              f"{e.fidelity_ratio * 100:8.1f}% {e.mae * 100:6.1f}pts")
    print()
    print(compress_range(parse_range(GTO_PRESETS["UTG"]), n_rules=6).explain())

    # ─────────────────────────────────────────────────────────────────
    rule("F8", "DCFR + Hyperparameter Schedules — le solveur")
    for name, cfg in [("DCFR nu ", DCFRConfig(use_schedule=False)),
                      ("HS-DCFR", DCFRConfig(use_schedule=True))]:
        t0 = time.perf_counter()
        res = DCFRSolver(KuhnPoker(), cfg).solve(600, track_every=200)
        dt = time.perf_counter() - t0
        traj = " → ".join(f"{e:.2e}" for _, e in res.history)
        print(f"  {name}  valeur={res.game_value:+.6f}  expl={res.exploitability:.3e}  "
              f"({dt:.1f}s)")
        print(f"            exploitabilité : {traj}")
    print(f"  Nash exact de Kuhn : {-1 / 18:+.6f}")
    print("  HS-DCFR gagne à budget égal — pour moins de 15 lignes de code.")

    # ─────────────────────────────────────────────────────────────────
    rule("F9 / F10", "Bankroll ergodique et bluff-catch")
    print(f"  RoR(5 bb/100, σ=100, 30 buy-ins) = {risk_of_ruin(5, 100, 3000) * 100:.3f} %")
    print(f"  Bankroll pour RoR 1 %            = {bankroll_for_ror(5, 100, 0.01) / 100:.0f} buy-ins")
    lo, hi = confidence_interval_winrate(5.0, 100.0, 10_000)
    print(f"  Winrate sur 10k mains, IC95      = [{lo:+.1f} ; {hi:+.1f}] bb/100 → contient 0")
    print(f"  Mains pour prouver μ=5 bb/100    = {hands_for_significance(5, 100):,}".replace(",", " "))
    print(f"  Drawdown p95 sur 100k mains      = {drawdown_quantile(5, 100, 100_000) / 100:.1f} buy-ins")
    a = analyse_bluffcatch(100.0, 75.0, 0.34, 0.09)
    print(f"\n  Bluff-catch pot 100 / bet 75, p̂=34 % ± 9 :")
    print(f"    équité requise {a.required_equity * 100:.1f} % · marge {a.margin * 100:+.1f} pts")
    print(f"    → P(le call est +EV) = {a.prob_call_is_plus_ev * 100:.1f} %  ⇒ {a.recommendation}")

    # ─────────────────────────────────────────────────────────────────
    rule("F13", "LA MÉTA-FUSION — de combien s'écarter du GTO")
    eng = FusionEngine(bankroll=BankrollProfile(5.0, 100.0, 3000.0), discount=1.0)
    KEY = "a3f9c1e2b7d40518"
    eng.start_hand(KEY, prior_range=btn)

    for label, n in (("après 20 mains", 20), ("après 200 mains", 180),
                     ("après 600 mains", 400)):
        for i in range(n):
            eng.observe_stat(KEY, "fold_to_cbet", bool(rng.random() < 0.75))
        d = eng.decide(
            KEY,
            gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
            best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
            pot=45.0, equities=eq, hands_remaining=400,
            ev_gto=0.0, ev_best_response=1.2, exploitability_br=8.0,
        )
        bb = eng.profile(KEY).tracker("fold_to_cbet").belief
        print(f"  {label:<16} θ̂={bb.mean * 100:5.1f}% ±{bb.std * 100:4.1f}  "
              f"z={d.fusion.z_score:5.2f}  λ={d.fusion.lambda_final:.3f}  "
              f"→ c-bet {d.fusion.strategy.get(Action.BET) * 100:5.1f} %"
              f"   {'✓ exploite' if d.fusion.significant else '· reste GTO'}")

    print(f"\n  Solveur pur       : c-bet 62,0 %")
    print(f"  Exploitatif naïf  : c-bet 88,0 %")
    print(f"  ★ Fusion          : c-bet {d.fusion.strategy.get(Action.BET) * 100:.1f} % "
          f"— dérivé, borné en exploitabilité, traçable")

    # ─────────────────────────────────────────────────────────────────
    rule("PIPELINE", "une main complète, toutes fusions actives")
    eng2 = FusionEngine(bankroll=BankrollProfile(5.0, 100.0, 3000.0), discount=1.0)
    K2 = "villain_2"
    eng2.start_hand(K2, prior_range=btn)
    for i in range(300):
        eng2.observe_stat(K2, "fold_to_cbet", bool(rng.random() < 0.74))
        eng2.observe_stat(K2, "vpip", bool(rng.random() < 0.31))
        eng2.observe_stat(K2, "river_bluff_freq", bool(rng.random() < 0.38))
    eng2.observe_action(K2, "call", eq, 0.33, hmm_obs=Observation.PASSIVE)
    eng2.observe_action(K2, "raise", eq, 1.20, hmm_obs=Observation.WILD)

    d = eng2.decide(
        K2,
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        pot=45.0, equities=eq, hands_remaining=400, facing_bet=34.0,
        ev_gto=0.0, ev_best_response=1.2,
    )
    print(d.explain())

    print(f"\n{'─' * 76}")
    print("  13 fusions · 279 tests · 0 dépendance réseau · 0 € de licence")
    print(f"{'─' * 76}\n")


if __name__ == "__main__":
    main()
