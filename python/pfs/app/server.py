"""
Serveur applicatif local — le logiciel lui-même.

Contraintes de conception, dérivées de l'addendum v2.1 :
  * **zéro dépendance réseau** : uniquement la bibliothèque standard
    (``http.server``). Aucun framework web, donc aucune socket sortante ;
  * **écoute sur 127.0.0.1 uniquement** — jamais 0.0.0.0 ;
  * jeton d'authentification aléatoire par démarrage, pour qu'aucun autre
    processus local ne puisse interroger l'API ;
  * aucune donnée ne quitte la machine.

Lancement :

    python -m pfs            # ouvre le navigateur sur l'interface
    python -m pfs --port 8731 --no-browser
"""

from __future__ import annotations

import json
import math
import secrets
import threading
import webbrowser
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

from pfs.core.bankroll import (
    BankrollProfile,
    bankroll_for_ror,
    confidence_interval_winrate,
    drawdown_quantile,
    hands_for_significance,
    kelly_fraction,
    risk_of_ruin,
    should_take_shot,
)
from pfs.core.bluffcatch import analyse_bluffcatch
from pfs.core.range_model import (
    GTO_PRESETS,
    N_COMBOS,
    N_GROUPS,
    Range,
    group_name,
    parse_range,
)
from pfs.data.hand_history import iter_hands, player_key
from pfs.fusion.arbiter import Action, ActionDistribution, FusionInput, arbitrate
from pfs.fusion.bet_sizing import MDFCallModel, knowledge_price, sizing_table
from pfs.fusion.bottleneck import compress_range, elbow_point, information_plane
from pfs.fusion.dynamic_beta import GTO_BASELINES, DynamicBetaTracker
from pfs.fusion.geometry import fisher_rao_distance
from pfs.fusion.hmm import MentalState, Observation, OnlineHMM
from pfs.fusion.skill_prior import (
    ExternalRating,
    GameFormat,
    RatingSource,
    adaptation_propensity_from_skill,
    archetype_prior_from_skill,
    estimate_skill,
    tournaments_needed,
)
from pfs.solver.dcfr import DCFRConfig, DCFRSolver, KuhnPoker
from pfs.train.drill import DrillSession, Grade, LeakFinder, session_monitor

__all__ = ["create_server", "run", "API"]

UI_PATH = Path(__file__).with_name("ui.html")

# Session d'entraînement en mémoire, une par processus.
_STATE: dict[str, Any] = {"drill": None, "answers": []}


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(_key(k)): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if hasattr(obj, "value"):        # Enum
        return obj.value
    return obj


def _key(k: Any) -> Any:
    return k.value if hasattr(k, "value") else k


@lru_cache(maxsize=256)
def _drawdown_cached(mu: float, sd: float, n: int) -> float:
    """Drawdown Monte-Carlo, mémorisé et échantillonné pour rester interactif.

    6 000 tirages suffisent pour un quantile à 95 % (erreur-type ≈ 0,3 %) et
    ramènent l'appel de 1 255 ms à moins de 200 ms — au-dessous du seuil de
    perception d'une latence dans une interface.
    """
    return drawdown_quantile(mu, sd, n, quantile=0.95, n_sims=6000)


@lru_cache(maxsize=64)
def _rules_cached(spec: str, n_rules: int, max_rules: int) -> tuple:
    plane = information_plane(parse_range(spec), max_rules=max_rules)
    elbow = elbow_point(plane)
    rs = compress_range(parse_range(spec), n_rules=n_rules or elbow.n_rules)
    return rs, plane, elbow


# ═══════════════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════════════


class API:
    """Toutes les routes. Chaque méthode reçoit un dict et renvoie un dict."""

    # ── ranges ───────────────────────────────────────────────────────────
    @staticmethod
    def range_get(p: dict) -> dict:
        spec = p.get("spec") or GTO_PRESETS.get(p.get("position", "BTN"), "")
        r = parse_range(spec)
        return {
            "groups": [float(x) for x in r.to_groups()],
            "names": [group_name(g) for g in range(N_GROUPS)],
            "combos": r.n_combos,
            "fraction": r.fraction,
            "entropy": r.entropy_bits,
            "normalised_entropy": r.normalised_entropy,
            "shape": r.shape_label(),
            "spec": spec,
        }

    @staticmethod
    def range_compare(p: dict) -> dict:
        a = Range.from_groups(np.array(p["a"], dtype=float))
        b = parse_range(p.get("spec") or GTO_PRESETS[p.get("position", "BTN")])
        ga, gb = a.to_groups(), b.to_groups()
        err = np.abs(ga - gb)
        weights = np.array([4 if "s" in group_name(g) else (6 if len(group_name(g)) == 2 else 12)
                            for g in range(N_GROUPS)], dtype=float)
        return {
            "accuracy": float(1.0 - np.sum(err * weights) / weights.sum()),
            "fisher": fisher_rao_distance(ga + 1e-9, gb + 1e-9),
            "gto": [float(x) for x in gb],
            "errors": [float(x) for x in err],
            "entropy_you": a.entropy_bits,
            "entropy_gto": b.entropy_bits,
        }

    @staticmethod
    def range_rules(p: dict) -> dict:
        spec = p.get("spec") or GTO_PRESETS[p.get("position", "BTN")]
        n = int(p.get("n_rules", 0))
        rs, plane, elbow = _rules_cached(spec, n, int(p.get("max_rules", 16)))
        return {
            "rules": [
                {"predicate": x.predicate_name, "frequency": x.frequency,
                 "coverage": x.coverage, "n_groups": x.n_groups}
                for x in rs.rules
            ],
            "default": rs.default_frequency,
            "fidelity": rs.fidelity_ratio,
            "complexity": rs.complexity_bits,
            "mae": rs.mae,
            "elbow": elbow.n_rules,
            "plane": [
                {"n": x.n_rules, "complexity": x.complexity_bits,
                 "fidelity": x.fidelity_ratio, "mae": x.mae}
                for x in plane
            ],
            "reconstructed": [float(x) for x in rs.apply()],
        }

    # ── fusion F13 ───────────────────────────────────────────────────────
    @staticmethod
    def fusion_arbitrate(p: dict) -> dict:
        stat = p.get("stat", "fold_to_cbet")
        observed = float(p["observed"])
        n = int(p["n"])
        baseline = float(p.get("baseline", GTO_BASELINES.get(stat, 0.5)))

        t = DynamicBetaTracker(stat, discount=float(p.get("discount", 1.0)))
        t.update_batch(successes=int(round(observed * n)), trials=n)
        b = t.belief
        lo, hi = b.credible_interval()

        mental = {
            MentalState.SOLID: float(p.get("p_solid", 0.5)),
            MentalState.LOOSE: float(p.get("p_loose", 0.3)),
            MentalState.TILT: float(p.get("p_tilt", 0.2)),
        }
        gto_bet = float(p.get("gto_bet", 0.62))
        br_bet = float(p.get("br_bet", 0.88))

        res = arbitrate(FusionInput(
            gto=ActionDistribution({Action.BET: gto_bet, Action.CHECK: 1 - gto_bet}),
            best_response=ActionDistribution({Action.BET: br_bet, Action.CHECK: 1 - br_bet}),
            deviation=abs(b.mean - baseline),
            deviation_std=b.std,
            mental_state_probs=mental,
            ev_gto=0.0,
            ev_best_response=float(p.get("ev_br", 1.2)),
            exploitability_gto=0.0,
            exploitability_br=float(p.get("expl_br", 8.0)),
            realized_gift=float(p["gift"]) if p.get("gift") not in (None, "") else math.inf,
        ))
        return {
            "theta": b.mean, "std": b.std, "ci": [lo, hi], "n": b.n_observations,
            "baseline": baseline,
            "z": res.z_score, "significant": res.significant,
            "lambda_raw": res.lambda_raw, "lambda": res.lambda_final,
            "rho": res.adaptation_risk,
            "bet": res.strategy.get(Action.BET),
            "gto_bet": gto_bet, "br_bet": br_bet,
            "ev_gain": res.ev_gain_vs_gto,
            "exploitability": res.exploitability,
            "capped": res.exploitability_capped,
            "rationale": res.rationale,
            "hands_needed": b.hands_until_significant(baseline),
        }

    # ── sizing F4 ────────────────────────────────────────────────────────
    @staticmethod
    def sizing(p: dict) -> dict:
        pot = float(p.get("pot", 45.0))
        n = 400
        rng = np.random.default_rng(int(p.get("seed", 7)))
        a = float(p.get("range_strength", 2.0))
        bnd = float(p.get("range_weakness", 3.0))
        eq = np.clip(rng.beta(a, bnd, n), 0.02, 0.98)
        lam = float(p.get("lam", knowledge_price(int(p.get("hands_left", 300)),
                                                 float(p.get("sigma", 0.06)))))
        t = sizing_table(pot, np.ones(n), eq, lam=lam, model=MDFCallModel(
            slack=float(p.get("slack", 0.0))))
        return {
            "lam": lam,
            "entropy": t.entropy_before,
            "candidates": [
                {"bet": c.bet, "frac": c.fraction_of_pot, "p_fold": c.p_fold,
                 "ev": c.ev, "ig": c.info_gain, "obj": c.objective}
                for c in t.candidates
            ],
            "best_ev": t.best_ev.fraction_of_pot,
            "best_info": t.best_info.fraction_of_pot,
            "best_fused": t.best_fused.fraction_of_pot,
        }

    # ── bluff-catch F10 ──────────────────────────────────────────────────
    @staticmethod
    def bluffcatch(p: dict) -> dict:
        a = analyse_bluffcatch(float(p["pot"]), float(p["bet"]),
                               float(p["bluff_freq"]), float(p["bluff_std"]))
        return {
            "alpha": a.required_equity, "mdf": a.mdf, "margin": a.margin,
            "z": a.z_margin if math.isfinite(a.z_margin) else None,
            "p_plus_ev": a.prob_call_is_plus_ev,
            "ev_call": a.ev_call, "reco": a.recommendation,
        }

    # ── bankroll F9 ──────────────────────────────────────────────────────
    @staticmethod
    def bankroll(p: dict) -> dict:
        mu = float(p.get("winrate", 5.0))
        sd = float(p.get("stddev", 100.0))
        bb = float(p.get("bankroll", 3000.0))
        n = int(p.get("hands", 50_000))
        out = {
            "ror": risk_of_ruin(mu, sd, bb),
            "buyins": bb / 100.0,
            "bankroll_1pct": bankroll_for_ror(mu, sd, 0.01) if mu > 0 else None,
            "bankroll_5pct": bankroll_for_ror(mu, sd, 0.05) if mu > 0 else None,
            "kelly_half": kelly_fraction(mu, sd, 0.5),
            "hands_significant": hands_for_significance(mu, sd) if mu > 0 else None,
            "drawdown_p95": _drawdown_cached(mu, sd, n),
        }
        lo, hi = confidence_interval_winrate(mu, sd, n)
        out["ci_low"], out["ci_high"] = lo, hi
        out["ci_contains_zero"] = lo < 0.0 < hi
        if p.get("shot_buyin"):
            ok, m = should_take_shot(
                BankrollProfile(mu, sd, bb), float(p["shot_buyin"]),
                float(p.get("shot_winrate", mu * 0.5)),
                float(p.get("shot_stddev", sd * 1.1)),
            )
            out["shot"] = {"ok": ok, **{k: float(v) for k, v in m.items()}}
        return out

    # ── HMM F2 ───────────────────────────────────────────────────────────
    @staticmethod
    def hmm(p: dict) -> dict:
        h = OnlineHMM()
        traj = []
        for o in p.get("observations", []):
            b = h.update(int(o))
            traj.append([b[s] for s in MentalState])
        b = h.belief
        return {
            "probs": [b[s] for s in MentalState],
            "labels": [s.name for s in MentalState],
            "trajectory": traj,
            "most_likely": b.most_likely.name,
            "entropy": b.entropy_bits,
            "confident": b.is_confident,
            "surges": {o.name: OnlineHMM().tilt_surge(o) for o in Observation},
        }

    # ── solveur F8 ───────────────────────────────────────────────────────
    @staticmethod
    def solve(p: dict) -> dict:
        it = max(50, min(int(p.get("iterations", 600)), 4000))
        cfg = DCFRConfig(use_schedule=bool(p.get("schedule", True)))
        r = DCFRSolver(KuhnPoker(), cfg).solve(it, track_every=max(1, it // 8))
        return {
            "iterations": r.iterations,
            "game_value": r.game_value,
            "nash_value": -1 / 18,
            "exploitability": r.exploitability,
            "history": [{"t": t, "expl": e} for t, e in r.history],
            "strategy": {k: {a: float(x) for a, x in r.freq(k).items()}
                         for k in sorted(r.strategy)},
        }

    # ── entraînement ─────────────────────────────────────────────────────
    @staticmethod
    def drill_start(p: dict) -> dict:
        s = DrillSession(
            positions=tuple(p.get("positions", ["UTG", "MP", "CO", "BTN", "SB"])),
            difficulty=p.get("difficulty", "medium"),
            only_close=bool(p.get("only_close", False)),
            seed=int(p.get("seed", 0)),
        )
        _STATE["drill"] = s
        _STATE["answers"] = []
        return API.drill_next({})

    @staticmethod
    def drill_next(p: dict) -> dict:
        s: DrillSession | None = _STATE.get("drill")
        if s is None:
            return {"error": "aucune session — démarrer d'abord"}
        it = s.next_item()
        _STATE["current"] = it
        card = s.srs.cards.get(it.key)
        return {
            "position": it.position, "group": it.group, "combos": it.combos,
            "is_mixed": it.is_mixed, "difficulty": s.difficulty,
            "seen": 0 if card is None else len(card.history),
            "due": len(s.srs.due()),
            "n_answers": len(s.answers),
        }

    @staticmethod
    def drill_answer(p: dict) -> dict:
        s: DrillSession | None = _STATE.get("drill")
        it = _STATE.get("current")
        if s is None or it is None:
            return {"error": "aucune question en cours"}
        ans = s.answer(it, float(p["given"]), float(p.get("seconds", 0.0)))
        s.srs.advance(0.02)     # temps logique : ~50 réponses = 1 jour
        # ⚠ le champ s'appelle « deviation », PAS « error » : la clé « error »
        # est réservée aux échecs d'API, et la collision faisait avorter
        # silencieusement l'affichage du feedback côté interface.
        return {
            "correct": it.correct_frequency, "given": ans.given,
            "deviation": ans.error, "grade": int(ans.grade), "grade_name": ans.grade.name,
            "ev_loss": ans.ev_loss_bb,
            "score": _jsonable(s.score()),
            "next": API.drill_next({}),
        }

    @staticmethod
    def drill_report(p: dict) -> dict:
        s: DrillSession | None = _STATE.get("drill")
        if s is None or not s.answers:
            return {"leaks": [], "score": {}, "cognitive": None}
        leaks = LeakFinder.analyse(s.answers)
        cog = session_monitor(s.answers)
        return {
            "score": _jsonable(s.score()),
            "leaks": [
                {"label": x.label, "magnitude": x.magnitude, "ev_loss": x.ev_loss_bb,
                 "n": x.n, "direction": x.direction, "reco": x.recommendation}
                for x in leaks
            ],
            "cognitive": {
                "time_drift": cog.decision_time_drift,
                "error_drift": cog.error_rate_drift,
                "dispersion": cog.sizing_dispersion,
                "lambda": cog.loss_aversion_lambda,
                "n": cog.n_observations,
                "advice": cog.advice,
            },
            "workload": s.srs.workload(21),
            "mastery": s.srs.mastery(),
        }

    # ── analyse de hand-history ──────────────────────────────────────────
    @staticmethod
    def analyse_hh(p: dict) -> dict:
        text = p.get("text", "")
        salt = p.get("salt", "pfs-local")
        hands = list(iter_hands(text, salt))
        if not hands:
            return {"error": "aucune main reconnue (Winamax ou PokerStars attendus)"}

        trackers: dict[str, dict[str, DynamicBetaTracker]] = {}
        for h in hands:
            for seat in h.seats:
                obs = h.stat_observations(seat.player)
                for stat, val in obs.items():
                    d = trackers.setdefault(seat.player, {})
                    if stat not in d:
                        d[stat] = DynamicBetaTracker(
                            stat, discount=1.0,
                            prior_mean=GTO_BASELINES.get(stat), prior_strength=2.0)
                    d[stat].update(val)

        players = []
        for key, stats in trackers.items():
            row = {"player": key[:8], "stats": {}}
            for stat, t in sorted(stats.items()):
                b = t.belief
                lo, hi = b.credible_interval()
                base = GTO_BASELINES.get(stat)
                row["stats"][stat] = {
                    "mean": b.mean, "std": b.std, "ci": [lo, hi], "n": b.n_observations,
                    "baseline": base,
                    "exploitable": bool(base is not None and b.is_exploitable(base)),
                    "z": b.deviation_z(base) if base is not None else None,
                }
            players.append(row)
        players.sort(key=lambda r: -max((s["n"] for s in r["stats"].values()), default=0))

        return {
            "n_hands": len(hands),
            "rooms": sorted({h.room.value for h in hands}),
            "real_money": sum(1 for h in hands if h.is_real_money),
            "tournaments": sum(1 for h in hands if h.is_tournament),
            "players": players[:24],
            "sample": repr(hands[0]),
        }

    # ── prior de compétence externe (SharkScope / OPR) ───────────────────
    @staticmethod
    def skill(p: dict) -> dict:
        """Ingère un rating **saisi manuellement**. Aucun appel réseau."""
        fmt = GameFormat(p.get("format", "mtt_large"))
        r = ExternalRating(
            source=RatingSource(p.get("source", "sharkscope")),
            fmt=fmt,
            n_tournaments=int(p.get("n", 0)),
            observed_roi=float(p.get("roi", 0.0)),
        )
        e = estimate_skill(r)
        prior = archetype_prior_from_skill(e.skill, float(p.get("strength", 1.0)))
        needed = (tournaments_needed(abs(r.observed_roi), fmt)
                  if abs(r.observed_roi) > 1e-6 else None)
        return {
            "shrunk_roi": e.shrunk_roi, "std": e.std, "ci": list(e.ci),
            "shrinkage": e.shrinkage, "skill": e.skill,
            "significant": e.significant, "verdict": e.verdict,
            "rho": adaptation_propensity_from_skill(e.skill),
            "needed": needed,
            "archetypes": {k.value: v for k, v in
                           sorted(prior.items(), key=lambda kv: -kv[1])},
        }

    @staticmethod
    def presets(p: dict) -> dict:
        return {"presets": GTO_PRESETS, "baselines": GTO_BASELINES}

    # ── Équité (moteur exact/MC, multiway L3) ────────────────────────────
    @staticmethod
    def equity(p: dict) -> dict:
        from pfs.core.equity import equity_multiway, equity_vs_range
        from pfs.core.range_model import RANKS, SUITS, parse_range

        def card(t: str) -> int:
            t = t.strip()
            if len(t) != 2 or t[0].upper() not in RANKS or t[1].lower() not in SUITS:
                raise ValueError(f"carte illisible : {t!r} (attendu ex. 'Ah')")
            return RANKS.index(t[0].upper()) * 4 + SUITS.index(t[1].lower())

        hero = [card(t) for t in str(p["hero"]).replace(",", " ").split()]
        board = [card(t) for t in str(p.get("board", "")).replace(",", " ").split()
                 if t.strip()]
        specs = [str(p["range"])]
        extra = str(p.get("range2", "") or "").strip()
        if extra:
            specs.append(extra)
        ranges = [parse_range(s) for s in specs]
        n_sims = int(p.get("n_sims", 100_000))
        seed = int(p.get("seed", 0))
        if len(ranges) == 1:
            r = equity_vs_range(hero, ranges[0], board, n_sims=n_sims, seed=seed)
        else:
            r = equity_multiway(hero, ranges, board, n_sims=n_sims, seed=seed)
        return {
            "equity": r.equity, "win": r.win, "tie": r.tie, "lose": r.lose,
            "exact": r.exact, "n_scenarios": r.n_scenarios,
            "std_error": r.std_error, "summary": str(r),
            "n_villains": len(ranges),
        }

    # ── ICM F14 (+ PKO L5, FGS L4) ───────────────────────────────────────
    @staticmethod
    def icm(p: dict) -> dict:
        from pfs.core.icm import (
            IcmSpot, PkoSpot, analyse_icm_spot, analyse_pko_spot,
            fgs_equities, icm_equities,
        )

        stacks = [float(s) for s in p["stacks"]]
        payouts = [float(x) for x in p["payouts"]]
        out: dict = {
            "equities": [float(e) for e in icm_equities(stacks, payouts)],
            "chip_share": [s / sum(stacks) * sum(payouts) for s in stacks],
        }
        eq = p.get("hero_equity")
        eqf = float(eq) if eq is not None else None
        bounties = [float(b) for b in (p.get("bounties") or [])]

        if "hero" in p and "villain" in p:
            hero, villain = int(p["hero"]), int(p["villain"])
            # convention UI : pot = argent mort AVANT la mise adverse
            pot = float(p.get("pot", 0.0) or 0.0)
            bet = float(p.get("bet", 0.0) or min(stacks[hero],
                                                 max(stacks[villain], 1.0)))
            if stacks[villain] > 0:
                spot = IcmSpot(stacks=tuple(stacks), payouts=tuple(payouts),
                               hero=hero, villain=villain, pot=pot, bet=bet)
                a = analyse_icm_spot(spot, eqf)
                out["spot"] = {
                    "bubble_factor": a.bubble if math.isfinite(a.bubble) else None,
                    "risk_premium": a.premium,
                    "alpha_cash": a.alpha_cash,
                    "alpha_icm": a.alpha_icm,
                    "verdict": a.verdict,
                    "explain": a.explain(),
                }
            else:
                # vilain déjà à tapis (stack 0, jetons au pot) : le BF par
                # tapis effectif est indéfini — équités requises par
                # différences de $EV (le même moteur que le PKO, primes à 0)
                pk0 = analyse_pko_spot(PkoSpot(
                    stacks=tuple(stacks), payouts=tuple(payouts),
                    bounties=tuple([0.0] * len(stacks)), hero=hero,
                    villain=villain, pot=pot + bet, bet=bet), eqf)
                gain = pk0.ev_win - pk0.ev_fold
                loss = pk0.ev_fold - pk0.ev_lose
                bf = (loss / gain) if gain > 0 else None
                out["spot"] = {
                    "bubble_factor": bf,
                    "risk_premium": (bf / (1 + bf) - 0.5) if bf else None,
                    "alpha_cash": bet / (pot + 2 * bet) if bet > 0 else None,
                    "alpha_icm": pk0.required_no_bounty,
                    "verdict": pk0.verdict,
                    "explain": pk0.explain(),
                }
            if bounties and any(b > 0 for b in bounties):
                pk = analyse_pko_spot(PkoSpot(
                    stacks=tuple(stacks), payouts=tuple(payouts),
                    bounties=tuple(bounties), hero=hero, villain=villain,
                    pot=pot + bet, bet=bet), eqf)
                out["pko"] = {
                    "bounty_value": pk.bounty_value,
                    "villain_eliminated": pk.villain_eliminated,
                    "required_no_bounty": pk.required_no_bounty,
                    "required_with_bounty": pk.required_with_bounty,
                    "discount_pts": pk.discount_pts,
                    "verdict": pk.verdict,
                }

        n_fgs = int(p.get("fgs_hands", 0) or 0)
        if n_fgs > 0:
            r = fgs_equities(stacks, payouts,
                             button=int(p.get("button", 0) or 0),
                             sb=float(p.get("sb", 0.0) or 0.0),
                             bb=float(p.get("bb", 1.0) or 1.0),
                             ante=float(p.get("ante", 0.0) or 0.0),
                             n_hands=n_fgs)
            out["fgs"] = {
                "static": [float(x) for x in r.static],
                "mean": [float(x) for x in r.fgs_mean],
                "deltas_last": [float(x) for x in r.deltas[-1]],
                "stacks_last": [float(x) for x in r.stacks_path[-1]],
            }
        return out


    # ── Solveur postflop réel L1 ─────────────────────────────────────────
    @staticmethod
    def postflop(p: dict) -> dict:
        from pfs.core.range_model import RANKS, SUITS, parse_range
        from pfs.solver.postflop import PostflopSolver

        def card(t: str) -> int:
            t = t.strip()
            if len(t) != 2 or t[0].upper() not in RANKS or t[1].lower() not in SUITS:
                raise ValueError(f"carte illisible : {t!r}")
            return RANKS.index(t[0].upper()) * 4 + SUITS.index(t[1].lower())

        board = [card(t) for t in str(p["board"]).replace(",", " ").split()]
        fracs = [float(x) for x in (p.get("bet_fracs") or [0.75])]
        iters = min(int(p.get("iterations", 400)), 2000)
        s = PostflopSolver(
            board,
            parse_range(str(p["oop_range"])),
            parse_range(str(p["ip_range"])),
            pot=float(p["pot"]),
            stack=float(p["stack"]),
            bet_fracs=tuple(fracs),
            max_bets=int(p.get("max_bets", 2)),
        )
        s.solve(iters)
        r = s.result()
        return {
            "ev_oop": r.ev_oop, "ev_ip": r.ev_ip, "pot": r.pot,
            "exploitability": r.exploitability,
            "iterations": r.iterations, "n_nodes": r.n_nodes,
            "street": "river" if len(board) == 5 else "turn",
            "root_player": "OOP",
            "root_actions": [
                {"label": a.label, "frequency": a.frequency,
                 "per_combo": a.per_combo}
                for a in r.root_actions
            ],
        }


    # ── P4 : re-solve depuis une range inférée/observée ──────────────────
    @staticmethod
    def resolve(p: dict) -> dict:
        from pfs.core.range_model import RANKS, SUITS, parse_range
        from pfs.engine import FusionEngine

        def card(t: str) -> int:
            t = t.strip()
            if len(t) != 2 or t[0].upper() not in RANKS or t[1].lower() not in SUITS:
                raise ValueError(f"carte illisible : {t!r}")
            return RANKS.index(t[0].upper()) * 4 + SUITS.index(t[1].lower())

        board = [card(t) for t in str(p["board"]).replace(",", " ").split()]
        eng = FusionEngine()
        key = "villain#resolve"
        if p.get("villain_range"):        # range observée/inférée fournie
            eng.start_hand(key, prior_range=parse_range(str(p["villain_range"])))
        kw = {}
        if p.get("reference_villain"):
            kw["reference_villain"] = parse_range(str(p["reference_villain"]))
        rep = eng.resolve_spot(
            key, board=board, pot=float(p["pot"]),
            hero_range=parse_range(str(p["hero_range"])),
            hero_position=str(p.get("hero_position", "ip")),
            stack=float(p["stack"]) if p.get("stack") else None,
            hero_stack=float(p["hero_stack"]) if p.get("hero_stack") else None,
            villain_stack=(float(p["villain_stack"])
                           if p.get("villain_stack") else None),
            bet_fracs=tuple(float(x) for x in (p.get("bet_fracs") or [0.75])),
            iterations=min(int(p.get("iterations", 250)), 1000),
            game_format=str(p.get("game_format", "cash")),
            **kw,
        )
        return {
            "villain_source": rep.villain_source,
            "ev_exploit": rep.ev_hero_exploit,
            "ev_gto_locked": rep.ev_hero_gto_locked,
            "exploit_gain": rep.exploit_gain,
            "lam": rep.lam,
            "exploitability": rep.exploitability,
            "confidence": rep.confidence,
            "explain": rep.explain(),
            "root_actions": [
                {"label": a.label, "frequency": a.frequency,
                 "per_combo": dict(list(a.per_combo.items())[:10])}
                for a in rep.root_actions
            ],
        }


    # ── Revue de session : analyse a posteriori de mains DÉJÀ jouées ─────
    @staticmethod
    def review(p: dict) -> dict:
        """Revue post-partie d'un dossier d'historiques (jamais en direct).

        Payload : {"path": "<dossier d'historiques iPoker/XML>"}. Renvoie le
        profil statistique du héros et l'analyse d'équité des tapis. Ne lit
        que des mains terminées ; ne regarde aucune partie en cours.
        """
        from pfs.analysis import review_folder

        path = str(p.get("path", "")).strip()
        if not path:
            raise ValueError("champ 'path' requis (dossier d'historiques).")
        rep = review_folder(path)
        pr = rep.profile
        return {
            "profile": {
                "n_hands": pr.n_hands,
                "net_bb": round(pr.net_bb, 1),
                "vpip": round(pr.vpip, 1), "vpip_opp": pr.vpip_opp,
                "pfr": round(pr.pfr, 1), "pfr_opp": pr.pfr_opp,
                "three_bet": round(pr.three_bet, 1), "three_bet_opp": pr.three_bet_opp,
                "fold_to_cbet": round(pr.fold_to_cbet, 1), "fold_cbet_opp": pr.fold_cbet_opp,
                "wtsd": round(pr.wtsd, 1), "wtsd_opp": pr.wtsd_opp,
            },
            "allin": {
                "measured": rep.n_allin_measured,
                "total": rep.n_allin_total,
                "skipped": rep.n_allin_skipped,
                "avg_equity": round(rep.avg_allin_equity, 1),
                "realized": round(rep.total_realized, 0),
                "expected": round(rep.total_expected, 0),
                "luck_bb": round(rep.luck_bb, 1),
            },
            "explain": rep.explain(),
        }


    # ── Revue shove/fold : décisions préflop face au Nash jam/fold ───────
    @staticmethod
    def review_pushfold(p: dict) -> dict:
        """Confronte les tapis préflop déjà joués à l'équilibre push/fold.

        Payload : {"path": "<dossier d'historiques>"}. Chaque décision SB
        heads-up est jugée par le solveur Nash, l'écart chiffré en bb.
        """
        from pfs.analysis import review_pushfold_folder

        path = str(p.get("path", "")).strip()
        if not path:
            raise ValueError("champ 'path' requis (dossier d'historiques).")
        rev = review_pushfold_folder(path)
        return {
            "judged": len(rev.spots),
            "mistakes": rev.n_mistakes,
            "loose_jams": len(rev.loose_jams),
            "tight_folds": len(rev.tight_folds),
            "skipped_multiway": rev.n_skipped_multiway,
            "skipped_deep": rev.n_skipped_deep,
            "total_cost_bb": round(rev.total_cost_bb, 2),
            "worst": [
                {"hand": s.hand, "eff_bb": s.eff_bb, "decision": s.decision,
                 "verdict": s.verdict, "cost_bb": round(s.cost_bb, 2)}
                for s in rev.worst(10) if s.verdict != "ok"
            ],
            "explain": rev.explain(),
        }


    # ── Reconnaissance de cartes depuis une image ───────────────────────
    @staticmethod
    def recognize(p: dict) -> dict:
        """Reconnaît des cartes dans une image (chemin local).

        Payload : {"path": "capture.png", "rois": [[x,y,w,h], ...]}. Sans
        ``rois``, l'image entière est traitée comme une seule carte.
        """
        from pfs.vision import identify_card, recognize_cards

        path = str(p.get("path", "")).strip()
        if not path:
            raise ValueError("champ 'path' requis (image).")
        rois = p.get("rois")
        if rois:
            matches = recognize_cards(path, [tuple(int(v) for v in r) for r in rois])
        else:
            matches = [identify_card(path)]
        return {
            "cards": [m.card for m in matches],
            "detail": [
                {"card": m.card, "distance": m.distance, "margin": m.margin,
                 "confidence": m.confidence, "runner_up": m.runner_up}
                for m in matches
            ],
        }

    # ── Conseil sur un spot déjà joué (« qu'aurais-je dû faire ? ») ──────
    @staticmethod
    def advise(p: dict) -> dict:
        """Verdict sur une main terminée, décrite comme sur une capture.

        Payload : {"hero": "Ah Kd", "board": "Qs 7d 2c", "pot": 100,
                   "bet": 75, "stack": 300, "big_blind": 10,
                   "position": "BTN", "villain": "moyenne", "players": 2}
        """
        from pfs.analysis import Spot, advise as _advise

        a = _advise(Spot(
            hero=str(p.get("hero", "")),
            board=str(p.get("board", "") or ""),
            pot=float(p.get("pot", 0) or 0),
            bet=float(p.get("bet", 0) or 0),
            stack=float(p.get("stack", 0) or 0),
            big_blind=float(p.get("big_blind", 1) or 1),
            position=str(p.get("position", "BTN")),
            villain=str(p.get("villain", "moyenne")),
            players=int(p.get("players", 2) or 2),
        ))
        return {
            "hand": a.hand, "action": a.action, "confidence": a.confidence,
            "regime": a.regime, "equity": a.equity, "required": a.required,
            "mdf": a.mdf, "ev_bb": a.ev_bb, "reasons": a.reasons,
            "assumptions": a.assumptions, "explain": a.explain(),
        }


ROUTES: dict[str, Callable[[dict], dict]] = {
    "range": API.range_get,
    "range/compare": API.range_compare,
    "range/rules": API.range_rules,
    "fusion": API.fusion_arbitrate,
    "sizing": API.sizing,
    "bluffcatch": API.bluffcatch,
    "bankroll": API.bankroll,
    "hmm": API.hmm,
    "solve": API.solve,
    "drill/start": API.drill_start,
    "drill/next": API.drill_next,
    "drill/answer": API.drill_answer,
    "drill/report": API.drill_report,
    "analyse": API.analyse_hh,
    "skill": API.skill,
    "presets": API.presets,
    "icm": API.icm,
    "equity": API.equity,
    "postflop": API.postflop,
    "resolve": API.resolve,
    "review": API.review,
    "review/pushfold": API.review_pushfold,
    "advise": API.advise,
    "recognize": API.recognize,
}


# ═══════════════════════════════════════════════════════════════════════════
# SERVEUR
# ═══════════════════════════════════════════════════════════════════════════


def _make_handler(token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PokerFusionSolver/2.3"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # silence
            pass

        # -- helpers ------------------------------------------------------
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: Any) -> None:
            self._send(code, json.dumps(_jsonable(payload)).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _authorised(self, query: dict) -> bool:
            return (query.get("t", [""])[0] == token
                    or self.headers.get("X-PFS-Token") == token)

        # -- routes -------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                html = UI_PATH.read_text(encoding="utf-8").replace("__TOKEN__", token)
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if u.path == "/favicon.ico":
                # Pique noir en SVG — évite un 404 dans la console à chaque chargement.
                svg = (b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                       b'<rect width="16" height="16" rx="3" fill="#11111b"/>'
                       b'<text x="8" y="12.5" font-size="12" text-anchor="middle" '
                       b'fill="#a855f7">&#9824;</text></svg>')
                self._send(200, svg, "image/svg+xml")
                return
            if u.path == "/api/health":
                self._json(200, {"ok": True, "version": "2.3"})
                return
            self._json(404, {"error": "route inconnue"})

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            # Consommer le corps AVANT toute réponse. En keep-alive HTTP/1.1,
            # répondre une erreur (403/404) en laissant le corps non lu dans
            # le tampon fait envoyer un RST par la pile Windows : le client
            # reçoit alors une ConnectionAborted au lieu du code d'erreur.
            # Drainer d'abord rend la réponse déterministe.
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                n = 0
            raw = self.rfile.read(n) if n > 0 else b""

            if not self._authorised(q):
                self._json(403, {"error": "jeton invalide"})
                return
            if not u.path.startswith("/api/"):
                self._json(404, {"error": "route inconnue"})
                return
            route = u.path[len("/api/"):]
            fn = ROUTES.get(route)
            if fn is None:
                self._json(404, {"error": f"route inconnue : {route}"})
                return
            try:
                payload = json.loads(raw or b"{}")
                self._json(200, fn(payload))
            except Exception as exc:  # remonter l'erreur, ne jamais l'avaler
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def create_server(port: int = 8731) -> tuple[ThreadingHTTPServer, str]:
    """Crée le serveur, lié à **127.0.0.1 uniquement**, avec jeton aléatoire."""
    token = secrets.token_urlsafe(24)
    srv = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(token))
    return srv, token


def run(port: int = 8731, open_browser: bool = True) -> None:
    srv, token = create_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/?t={token}"
    print("╔" + "═" * 66 + "╗")
    print("║  ♠  POKER FUSION SOLVER — interface locale" + " " * 24 + "║")
    print("╠" + "═" * 66 + "╣")
    print(f"║  {url[:62]:<62}  ║")
    print("║" + " " * 66 + "║")
    print("║  Écoute sur 127.0.0.1 uniquement · aucune sortie réseau" + " " * 10 + "║")
    print("║  Ctrl+C pour arrêter" + " " * 45 + "║")
    print("╚" + "═" * 66 + "╝")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt.")
    finally:
        srv.server_close()
