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
import os
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

# Sessions d'entraînement en mémoire, une par processus. « drill » est la
# session de ranges (fréquences GTO) ; « fuites » celle des drills ciblés
# générés depuis les erreurs réelles d'un dossier d'historiques — deux objets
# distincts, pour qu'ouvrir l'une ne détruise pas l'autre.
_STATE: dict[str, Any] = {"drill": None, "answers": [],
                          "fuites": None, "fuites_courant": None}


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


@lru_cache(maxsize=1)
def _eqr_entraine():
    """Modèle EQR (L7, ``fusion.eqr``), entraîné UNE fois puis mémoïsé.

    ``train_eqr()`` coûte ~24 solves river (mesuré : ~4 s sur cette machine) :
    le premier appel de ``/api/postflop`` avec ``leaf_model="eqr"`` paie ce
    coût, les suivants relisent le cache du processus. Limite ASSUMÉE,
    republiée dans chaque réponse avec le R² et le n MESURÉS du modèle
    entraîné (~0,6 sur ~48 échantillons river par défaut) : une valeur
    DIRECTIONNELLE — elle relève l'EV du joueur en position par rapport au
    rollout nu, sans garantie d'écart au solve complet (mesuré : elle peut
    sur-corriger) — et la somme des EV dérive.
    """
    from pfs.fusion.eqr import train_eqr

    return train_eqr()


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
            # fréquence de bluff estimée : c'est l'« équité » du bluff-catch,
            # ce qu'on compare au seuil alpha — l'interface la trace en jauge.
            "bluff_freq": a.estimated_bluff_freq,
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
        """Prochaine question de la session de ranges en cours.

        STATUT (audit du 14 août 2026) : la ROUTE ``drill/next`` n'est
        appelée par aucune UI — ``drill/start`` et ``drill/answer``
        embarquent déjà ``next`` dans leur réponse (et appellent cette
        fonction en direct). Conservée : c'est le seul moyen, pour
        l'outillage/CLI, de relire la question courante SANS répondre ni
        redémarrer — la retirer laisserait l'état de session accessible
        uniquement par des appels qui le modifient.
        """
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

    # ── drills ciblés : les fuites d'une revue, remises à l'entraînement ─
    @staticmethod
    def drill_fuites(p: dict) -> dict:
        """Génère les drills des fuites RÉELLES d'un dossier et ouvre la session.

        Payload : ``{"dossier": "<dossier d'historiques>"}`` — le même chemin
        que l'onglet Mes sessions — plus ``max_drills`` optionnel. La
        génération relit les mains, confronte chaque décision préflop au Nash
        (``pfs.train.leak_drills``) et ne fabrique un exercice QUE là où la
        bonne réponse est certaine ; le reste est compté, jamais approximé.

        Renvoie le résumé du corpus (drills, coût ciblé avec sa part non
        mesurée, fuites par famille, rapport texte) et, s'il y a au moins un
        drill, ``session: true`` avec la première question dans ``next``.
        ``session: false`` n'est pas une erreur : un corpus sans fuite
        chiffrable est une bonne nouvelle, et le rapport le dit.
        """
        from pfs.train.leak_drills import (
            SessionDrillsCibles,
            generer_drills_dossier,
        )

        path = str(p.get("dossier", "") or p.get("path", "")).strip()
        if not path:
            raise ValueError("champ 'dossier' requis (dossier d'historiques).")
        if not Path(path).is_dir():
            raise ValueError(f"dossier introuvable : {path}")
        max_drills = int(p["max_drills"]) if p.get("max_drills") else None

        rapport = generer_drills_dossier(path, max_drills=max_drills)
        resume = {
            "drills": len(rapport),
            "n_mains": rapport.n_mains,
            "n_erreurs": rapport.n_erreurs,
            "n_corrects": rapport.n_spots_corrects,
            "n_ecartes": rapport.n_ecartes,
            "bb_ciblees": round(rapport.cout_cible_bb, 2),
            "bb_mesurees": round(rapport.cout_mesure_bb, 2),
            "bb_non_mesurees": round(rapport.cout_non_mesure_bb, 2),
            # (famille, nombre de drills, coût bb, coût mesuré ?) — la part
            # « limp » est une convention, l'interface doit pouvoir le dire.
            "fuites": [
                {"fuite": f, "n": n, "cout_bb": round(c, 2),
                 "mesure": all(d.cout_mesure for d in rapport
                               if d.fuite == f)}
                for f, n, c in rapport.fuites
            ],
            "rapport": rapport.explain(),
        }
        if not len(rapport):
            _STATE["fuites"] = None
            _STATE["fuites_courant"] = None
            return {**resume, "session": False}
        _STATE["fuites"] = SessionDrillsCibles(list(rapport))
        _STATE["fuites_courant"] = None
        return {**resume, "session": True, "next": API.drill_fuites_next({})}

    @staticmethod
    def drill_fuites_next(p: dict) -> dict:
        """La prochaine question : spot dû d'abord, sinon la fuite la plus chère.

        L'énoncé ne contient JAMAIS la bonne réponse ni l'EV — c'est le
        contrat de ``SpotDrill.question()``, et un test le verrouille côté
        route. ``{"fini": true}`` quand tous les spots sont vus et qu'aucun
        n'est dû.
        """
        s = _STATE.get("fuites")
        if s is None:
            return {"error": "aucune session de fuites — lancer drill/fuites"}
        d = s.prochain()
        if d is None:
            _STATE["fuites_courant"] = None
            return {"fini": True, "score": _jsonable(s.score())}
        _STATE["fuites_courant"] = d
        carte = s.srs.cards.get(d.cle)
        return {
            "question": d.question(),
            "options": list(d.options),
            "main": d.main,
            "cartes": d.cartes,
            "tapis_bb": d.tapis_bb,
            "pot_bb": d.pot_bb,
            "mise_bb": d.mise_bb,
            "fuite": d.fuite,
            "cout_total_bb": round(d.cout_total_bb, 2),
            "cout_mesure": d.cout_mesure,
            "occurrences": d.occurrences,
            "vu": 0 if carte is None else len(carte.history),
            "dus": len(s.srs.due()),
            "n_reponses": len(s.reponses),
            "n_drills": len(s.drills),
        }

    @staticmethod
    def drill_fuites_answer(p: dict) -> dict:
        """Note une réponse, replanifie le spot (SM-2) et rend le corrigé.

        Payload : ``{"reponse": "JAM"|"FOLD", "seconds": 3.2}``. Une réponse
        fausse remet l'intervalle à zéro : le spot revient immédiatement.
        Le corrigé est ``SpotDrill.explication()`` — la réponse, le solve qui
        l'a tranchée, et le coût avec sa part non mesurée s'il y en a une.
        """
        s = _STATE.get("fuites")
        d = _STATE.get("fuites_courant")
        if s is None or d is None:
            return {"error": "aucune question de fuites en cours"}
        note = s.repondre(d, str(p.get("reponse", "")),
                          float(p.get("seconds", 0.0)))
        # Temps logique : même cadence que le drill de ranges (~50 réponses
        # par jour logique), pour que les intervalles SM-2 soient comparables.
        s.srs.advance(0.02)
        return {
            "correcte": note.correcte,
            "reponse": note.reponse,
            "bonne_reponse": d.bonne_reponse,
            "grade": int(note.grade),
            "grade_name": note.grade.name,
            "cout_bb": round(note.cout_bb, 3),
            "cout_mesure": d.cout_mesure,
            "explication": d.explication(),
            "score": _jsonable(s.score()),
            "next": API.drill_fuites_next({}),
        }

    # ── emplacements d'historiques détectés sur la machine ───────────────
    @staticmethod
    def emplacements(p: dict) -> dict:
        """Dossiers d'historiques PMU détectés. Payload : ``{}``.

        ``dossiers`` : ceux qui existent réellement (l'interface préremplit
        avec le premier) ; ``connus`` : tous les endroits où l'on a cherché —
        pour qu'une détection vide reste diagnosticable (« on a regardé là et
        là » plutôt qu'un champ muet).
        """
        from pfs.data.emplacements import dossiers_connus, dossiers_detectes

        return {"dossiers": list(dossiers_detectes()),
                "connus": list(dossiers_connus())}

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
        """Préréglages GTO et bases de référence, tels que l'app les utilise.

        STATUT (audit du 14 août 2026) : servie pour l'outillage/CLI,
        AUCUNE UI ne l'appelle (``ui.html`` embarque ses propres listes).
        Conservée : introspection à coût nul (les deux tables sont déjà
        importées ici), et la retirer casserait silencieusement tout script
        qui lit les présets EXACTS du logiciel plutôt que d'en recopier.
        """
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


    # ── Solveur postflop réel L1 (+ nodelock P2, rake L8, feuille P3) ────
    @staticmethod
    def _noeud_frequences(s, chemin: list[str]) -> dict:
        """Fréquences moyennes d'un nœud, pondérées par la range de l'acteur.

        Même convention que ``root_report`` : moyenne inconditionnelle sur la
        range entière de l'acteur (pas pondérée par le reach du chemin). Lève
        ``PostflopError`` si le chemin est inconnu — le message nomme l'action
        absente et les actions disponibles.
        """
        idx = s.node_at(tuple(chemin))
        node = s._nodes[idx]        # lecture seule, comme root_report
        sigma = s.average_strategy(idx)
        w = s.players[node.player].weights
        tot = float(w.sum()) or 1.0
        return {
            "path": list(chemin),
            "player": "OOP" if node.player == 0 else "IP",
            "frequencies": {lab: float((sigma[a] * w).sum() / tot)
                            for a, lab in enumerate(node.labels)},
        }

    @staticmethod
    def postflop(p: dict) -> dict:
        """Solve postflop range-contre-range. Ajouts au payload (optionnels) :

        ``rake``: ``{"pct": 0.05, "cap": 1.0}`` — modèle « % + cap »
        (``pfs.core.rake``, convention won-pot) ; ``cap`` absent = sans
        plafond. Défaut : AUCUN rake (identité bit à bit avec l'appel nu).

        ``locks``: ``[{"path": ["check","bet 1p"], "strategy":
        {"fold": 0.9, "call": 0.1}, "combos": "QQ"?}]`` — nodelock appliqué
        AVANT le solve (P2, signature Pio) ; le non-verrouillé re-solve
        librement autour (nodelock 2.0). Chemin inconnu → 400 avec le nom de
        l'action absente.

        ``leaf_model``: ``"rollout"`` ou ``"eqr"`` (board turn uniquement) —
        P3, profondeur limitée. ``"eqr"`` entraîne le modèle L7 UNE fois par
        processus puis le mémoïse ; sa limite mesurée est republiée dans la
        réponse (``leaf``), pour que l'interface ne l'affiche jamais comme
        exact.

        ``nodes``: ``[["check"], ["bet 1p"]]`` — fréquences moyennes rendues
        pour ces nœuds (``nodes`` en réponse), en plus de la racine.
        """
        from pfs.core.rake import NO_RAKE, RakeModel
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

        # Rake optionnel — défaut NO_RAKE, le comportement historique.
        rake = NO_RAKE
        rk = p.get("rake")
        if rk:
            if not isinstance(rk, dict) or "pct" not in rk:
                raise ValueError(
                    'rake : objet {"pct": 0.05, "cap": 1.0} requis '
                    "(cap absent = pas de plafond).")
            rake = RakeModel(
                pct=float(rk["pct"]),
                cap=float(rk["cap"]) if rk.get("cap") is not None else math.inf)

        # Feuille P3 optionnelle — "eqr" branche fusion.eqr (L7), entraîné
        # une fois puis mémoïsé au niveau du processus.
        leaf = str(p.get("leaf_model", "") or "full")
        if leaf not in ("full", "rollout", "eqr"):
            raise ValueError("leaf_model ∈ {'full', 'rollout', 'eqr'}.")
        kw: dict[str, Any] = {}
        if leaf == "rollout":
            kw = {"leaf_model": "rollout"}
        elif leaf == "eqr":
            kw = {"leaf_model": "eqr", "eqr_model": _eqr_entraine()}

        s = PostflopSolver(
            board,
            parse_range(str(p["oop_range"])),
            parse_range(str(p["ip_range"])),
            pot=float(p["pot"]),
            stack=float(p["stack"]),
            bet_fracs=tuple(fracs),
            max_bets=int(p.get("max_bets", 2)),
            rake=rake,
            **kw,
        )

        # Nodelock P2 : appliqué AVANT le solve, validé lock par lock.
        locks = p.get("locks") or []
        if not isinstance(locks, list):
            raise ValueError("locks : liste de {path, strategy[, combos]}.")
        for i, lk in enumerate(locks):
            if not isinstance(lk, dict) or "path" not in lk or "strategy" not in lk:
                raise ValueError(
                    f"locks[{i}] : objet {{path, strategy[, combos]}} requis.")
            strat = lk["strategy"]
            if not isinstance(strat, dict) or not strat:
                raise ValueError(
                    f"locks[{i}].strategy : {{action: fréquence}} non vide.")
            s.lock_node(
                tuple(str(x) for x in lk["path"]),
                {str(k): float(v) for k, v in strat.items()},
                combos=str(lk["combos"]) if lk.get("combos") else None)

        s.solve(iters)
        r = s.result()
        out = {
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
        if locks:
            # La stratégie RENDUE aux nœuds verrouillés — la preuve, côté
            # réponse, que le lock a bien pris (average_strategy rapporte la
            # stratégie verrouillée sur les combos du masque).
            out["locks"] = [
                API._noeud_frequences(s, [str(x) for x in lk["path"]])
                for lk in locks
            ]
        if p.get("nodes"):
            out["nodes"] = [
                API._noeud_frequences(s, [str(x) for x in chemin])
                for chemin in p["nodes"]
            ]
        if rake is not NO_RAKE:
            out["rake"] = {
                "pct": rake.pct, "cap": rake.cap,
                # E[rake] = pot − (EV_OOP + EV_IP), identité comptable exacte.
                "expected_rake": s.expected_rake(),
            }
        if leaf == "rollout":
            out["leaf"] = {
                "model": "rollout",
                "limite": ("la feuille ignore la mise river : somme des EV "
                           "exacte, l'écart au solve complet est le levier "
                           "de mise river."),
            }
        elif leaf == "eqr":
            m = _eqr_entraine()
            out["leaf"] = {
                "model": "eqr", "r2": m.r2, "n": m.n,
                "limite": (f"valeur DIRECTIONNELLE : ridge R² {m.r2:.2f} sur "
                           f"{m.n} échantillons river. Elle relève l'EV du "
                           "joueur en position par rapport au rollout, SANS "
                           "garantie d'écart au solve complet (elle peut "
                           "sur-corriger), et la somme des EV dérive — le "
                           "mode rollout reste somme-exacte."),
            }
        return out


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
                "af_postflop": round(pr.af_postflop, 2),
                "af_opp": pr.calls_postflop,
                "sieges_median": pr.sieges_median,
            },
            # Les bornes affichées par l'interface : MESURÉES sur corpus
            # public, plus jamais des fourchettes de manuel.
            "reperes": API._reperes_json(rep.jeu_de_reperes(), rep.ecarts()),
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


    # ── Repères mesurés sur corpus public ────────────────────────────────
    @staticmethod
    def _reperes_json(jeu, ecarts=()) -> dict:
        """Sérialise un jeu de repères et, s'il y en a, les écarts du héros.

        Chaque repère part avec son dénominateur, son intervalle de Wilson et
        la dispersion inter-joueurs : l'interface n'a pas le droit d'afficher
        une borne sans dire sur quoi elle a été mesurée.
        """
        return {
            "cle": jeu.cle,
            "source": jeu.source,
            "n_mains": jeu.n_mains,
            "n_mains_joueurs": jeu.n_mains_joueurs,
            "positions": list(jeu.positions),
            "concluant": jeu.concluant,
            "limites": list(jeu.limites),
            "stats": {
                stat: {
                    "valeur": round(r.valeur, 2),
                    "occasions": r.occasions,
                    "succes": r.succes,
                    "ic_bas": round(r.ic_bas, 2),
                    "ic_haut": round(r.ic_haut, 2),
                    "q1": round(r.q1, 2),
                    "mediane": round(r.mediane, 2),
                    "q3": round(r.q3, 2),
                    "n_joueurs": r.n_joueurs,
                    "unite": r.unite,
                }
                for stat, r in jeu.reperes.items()
            },
            "ecarts": [
                {
                    "stat": e.stat,
                    "valeur": round(e.valeur, 2),
                    "repere": round(e.repere, 2),
                    "ecart": round(e.ecart, 2),
                    "unite": e.unite,
                    "occasions": e.occasions_utilisateur,
                    "comparable": e.comparable,
                    "pourquoi": e.pourquoi,
                }
                for e in ecarts
            ],
        }

    @staticmethod
    def reperes(p: dict) -> dict:
        """Les repères mesurés sur corpus, sans avoir besoin d'historiques.

        Payload : ``{}`` pour tous les jeux, ou ``{"jeu": "pluribus_6max"}``
        pour un seul, ou ``{"sieges": 3}`` pour celui qui convient à une table
        de cette taille. Aucune lecture de disque : la table est gelée dans
        ``pfs/analysis/reperes.py`` et vérifiée par ``banc_reperes_corpus.py
        --verifier``.
        """
        from pfs.analysis.reperes import REPERES, jeu_par_defaut

        cle = str(p.get("jeu", "")).strip()
        if not cle and p.get("sieges"):
            cle = jeu_par_defaut(int(p["sieges"]))
        if cle:
            if cle not in REPERES:
                raise ValueError(
                    f"jeu de repères inconnu : « {cle} ». "
                    f"Disponibles : {sorted(REPERES)}.")
            return {"jeux": [API._reperes_json(REPERES[cle])],
                    "defaut": cle}
        return {"jeux": [API._reperes_json(j) for j in REPERES.values()],
                "defaut": jeu_par_defaut(3)}


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
        """Reconnaît des cartes dans une image (chemin local OU collée).

        Payload : soit ``{"path": "capture.png"}`` (fichier local), soit
        ``{"image_b64": "data:image/png;base64,..."}`` (image collée depuis
        le navigateur). Avec ``rois`` : [[x,y,w,h], ...] une carte par région ;
        sans ``rois``, l'image entière est traitée comme une seule carte.
        """
        import base64
        import io

        from pfs.vision import identify_card, recognize_cards

        b64 = str(p.get("image_b64", "")).strip()
        path = str(p.get("path", "")).strip()
        if b64:
            from PIL import Image
            if "," in b64:               # préfixe « data:image/png;base64, »
                b64 = b64.split(",", 1)[1]
            source = Image.open(io.BytesIO(base64.b64decode(b64)))
        elif path:
            source = path
        else:
            raise ValueError("fournir 'path' (fichier) ou 'image_b64' (image collée).")

        rois = p.get("rois")
        if rois:
            matches = recognize_cards(source, [tuple(int(v) for v in r) for r in rois])
        else:
            matches = [identify_card(source)]

        detail = [
            {"card": m.card, "distance": m.distance, "margin": m.margin,
             "confidence": m.confidence, "runner_up": m.runner_up,
             # exposé même en cas de refus : c'est ce qui rend l'échec
             # diagnostiquable côté interface
             "best_guess": m.best_guess, "statut": m.statut}
            for m in matches
        ]

        # Archiver ce qui n'a PAS été lu avec certitude. Sans cette collecte,
        # chaque échec réel disparaissait avec l'onglet, et il ne restait
        # que des tables synthétiques pour mesurer — c'est ce qui a laissé
        # passer les défauts les plus coûteux.
        if b64 and len(matches) == 1 and matches[0].statut != "sure":
            from pfs.vision.archive import enregistrer_echec
            try:
                enregistrer_echec(b64, {
                    **detail[0],
                    "cible": str(p.get("cible", "")),
                    "taille_px": list(p.get("taille_px", [])),
                })
            except Exception:
                pass       # l'archivage ne doit jamais casser la lecture

        return {"cards": [m.card for m in matches], "detail": detail}

    # ── Archive des captures et des échecs ──────────────────────────────
    @staticmethod
    def capture(p: dict) -> dict:
        """Archive une capture entière. Payload : {"image_b64": "..."}."""
        from pfs.vision.archive import dossier_archive, enregistrer_capture

        b64 = str(p.get("image_b64", "")).strip()
        if not b64:
            raise ValueError("champ 'image_b64' requis.")
        chemin = enregistrer_capture(b64, str(p.get("note", "")))
        return {"chemin": str(chemin), "dossier": str(dossier_archive())}

    @staticmethod
    def archive(p: dict) -> dict:
        """Inventaire de l'archive : combien de captures, combien d'échecs."""
        from pfs.vision.archive import dossier_archive, lister_echecs

        d = dossier_archive()
        echecs = lister_echecs()
        return {
            "dossier": str(d),
            "captures": len(list(d.glob("*.png"))),
            "echecs": len(echecs),
            "refus": sum(1 for e in echecs if e.statut == "refus"),
            "proposes": sum(1 for e in echecs if e.statut == "propose"),
            "annotes": sum(1 for e in echecs if e.verite),
            "derniers": [
                {"fichier": e.image.name, "statut": e.statut,
                 "candidat": e.diagnostic.get("best_guess"),
                 "distance": e.diagnostic.get("distance"),
                 "marge": e.diagnostic.get("margin"),
                 "verite": e.verite}
                for e in echecs[:20]
            ],
        }

    @staticmethod
    def lire_capture(p: dict) -> dict:
        """Capture entière → cartes du héros et du board, SANS cadrage.

        Le cadrage à la souris n'a plus de raison d'être : le détecteur
        localise les cartes et leur attribue un rôle. Il restait exigé parce
        que l'onglet avait été écrit avant que la détection ne fonctionne.

        Payload : ``{"image_b64": "..."}``. Renvoie les cartes lues avec leur
        confiance, les montants lus (pot, mise, tapis, blinde — chacun avec
        son motif de refus le cas échéant), et ce qui a échoué — pour que
        l'échec reste rattrapable à la main plutôt que silencieux.

        Les images **JPEG sont refusées** (octets magiques ``FF D8 FF``,
        détectés avant tout décodage) : la compression destructrice fait
        affirmer de fausses lectures — recoller en PNG (Win+Shift+S en
        produit). Les formats sans perte (PNG, BMP…) suivent le chemin normal.
        """
        import base64
        import io

        from PIL import Image

        from pfs.vision.archive import enregistrer_echec
        from pfs.vision.card_recognizer import identify_card_autour
        from pfs.vision.table_detector import read_table
        from pfs.vision.zones_montants import lire_montants

        b64 = str(p.get("image_b64", "")).strip()
        if not b64:
            raise ValueError("champ 'image_b64' requis.")
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        octets = base64.b64decode(b64)
        # Refus AVANT tout décodage-lecture, sur les octets magiques (FF D8
        # FF), pas sur le préfixe data: — mesuré sur les 57 captures réelles :
        # dès qualité 75, la compression JPEG produit des lectures FAUSSES
        # affirmées (un 3h lu « 8h » sûr ; un tapis 3,79 lu 11,50 BB à
        # confiance 0,79). Le contrat du projet est zéro valeur inventée,
        # donc zéro JPEG. Le PNG et les autres formats sans perte (BMP…)
        # passent tels quels.
        if octets[:3] == b"\xff\xd8\xff":
            raise ValueError(
                "capture JPEG refusée : la compression JPEG est destructrice "
                "et fait affirmer de fausses lectures (cartes et montants). "
                "Recolle la capture en PNG — Win+Shift+S produit du PNG.")
        image = Image.open(io.BytesIO(octets)).convert("RGB")

        table = read_table(image)
        sortie: dict[str, list] = {"hero": [], "board": [], "autres": []}
        cle = {"hero": "hero", "board": "board", "others": "autres"}
        for role in ("hero", "board", "others"):
            for b in getattr(table, role):
                m = identify_card_autour(image, (b.x, b.y, b.w, b.h))
                sortie[cle[role]].append({
                    "carte": m.card, "candidat": m.best_guess,
                    "statut": m.statut, "ecart": m.distance,
                    "marge": m.margin, "boite": [b.x, b.y, b.w, b.h]})
                # Seuls les rôles identifiés instruisent : un dos
                # d'adversaire est illisible par nature, l'archiver noierait
                # les vrais échecs.
                if m.statut != "sure" and role in ("hero", "board"):
                    decoupe = image.crop((b.x, b.y, b.x + b.w, b.y + b.h))
                    tampon = io.BytesIO()
                    decoupe.save(tampon, format="PNG")
                    enregistrer_echec(
                        base64.b64encode(tampon.getvalue()).decode("ascii"),
                        {"statut": m.statut, "distance": m.distance,
                         "margin": m.margin, "best_guess": m.best_guess,
                         "role": role, "origine": "collage",
                         "boite": [b.x, b.y, b.w, b.h]})

        sures = [c for r in ("hero", "board") for c in sortie[r]
                 if c["statut"] == "sure"]
        total = len(sortie["hero"]) + len(sortie["board"])
        return {
            **sortie,
            "largeur": image.width, "hauteur": image.height,
            "sures": len(sures), "total": total,
            "main": [c["carte"] for c in sortie["hero"] if c["carte"]],
            "tableau": [c["carte"] for c in sortie["board"] if c["carte"]],
            # Les montants viennent APRÈS les cartes : leurs cadres de visée
            # dérivent des boîtes détectées. Chaque zone refusée porte son
            # motif — c'est lui que l'UI montre à côté du champ à saisir.
            "montants": lire_montants(image, table),
        }

    # ── Calibration en direct : LIRE l'écran, jamais conseiller ──────────
    #
    # Ces deux routes capturent la fenêtre d'un client et rendent ce que le
    # recogniseur y a lu, avec la confiance de chaque carte. Elles ne
    # renvoient aucun verdict et n'appellent aucun calculateur de décision :
    # c'est un banc d'essai de la perception, pas une assistance de jeu.
    # `tests/test_live_sans_conseil.py` verrouille cette frontière.
    @staticmethod
    def live_fenetres(_p: dict) -> dict:
        """Fenêtres capturables, les tables probables en tête."""
        from pfs.vision.live import SondeIntrouvable, fenetres_disponibles

        try:
            return {"fenetres": fenetres_disponibles()}
        except SondeIntrouvable as e:
            return {"fenetres": [], "erreur": str(e)}

    @staticmethod
    def live_lire(p: dict) -> dict:
        """Capture une fenêtre et rend UNIQUEMENT ce qui y a été lu.

        Payload : ``{"fenetre": "PMU"}`` — absent ou vide pour la détection
        automatique d'un client poker connu.
        """
        from dataclasses import asdict

        from pfs.vision.live import (
            CaptureImpossible,
            SondeIntrouvable,
            lire_ecran,
        )

        titre = str(p.get("fenetre", "")).strip() or None
        try:
            lecture = lire_ecran(titre)
        except (CaptureImpossible, SondeIntrouvable) as e:
            return {"erreur": str(e)}
        return {
            "fenetre": lecture.fenetre,
            "largeur": lecture.largeur,
            "hauteur": lecture.hauteur,
            "sures": lecture.sures,
            "total": len(lecture.cartes),
            "taux": lecture.taux,
            "resume": lecture.resume(),
            "cartes": [asdict(c) for c in lecture.cartes],
            "image_b64": lecture.image_b64,
        }

    # ── Mode Live : conseil en direct, argent fictif prouvé uniquement ───
    #
    # Trois routes NOUVELLES, séparées des deux ci-dessus qui restent sans
    # conseil. La frontière : `pfs.app.mode_live` (perception + gate, zéro
    # import de conseil) rend un verdict, et le conseiller n'est convoqué
    # qu'APRÈS `gate.mode == "live"` — jamais avant, jamais en cas de doute,
    # jamais si la lecture est insuffisante. `tests/test_live_conseil_gate.py`
    # verrouille chacun de ces « jamais ».
    @staticmethod
    def live_armer(p: dict) -> dict:
        """Arme une fenêtre pour le mode Live — confirmation explicite exigée.

        Payload : ``{"fenetre": "PMU PLAY", "confirme": true}``. La
        confirmation porte l'hypothèse que ni badge ni titre ne peuvent
        prouver : « cette table est en argent fictif ». Sans elle : refus.
        """
        titre = str(p.get("fenetre", "")).strip()
        if not titre:
            raise ValueError("champ 'fenetre' requis.")
        return _mode_live().armer(titre, confirme=p.get("confirme") is True)

    @staticmethod
    def live_desarmer(p: dict) -> dict:
        """Désarme une fenêtre (ou toutes si 'fenetre' est absent)."""
        titre = str(p.get("fenetre", "")).strip() or None
        return _mode_live().desarmer(titre)

    @staticmethod
    def live_table(p: dict) -> dict:
        """Une itération du mode Live : preuve, lecture, et conseil si prouvé.

        Payload : ``{"fenetre": "PMU PLAY"}``. Réponse : ``gate`` (verdict et
        signaux), ``lecture`` (cartes + montants), ``conseil`` (le verdict du
        conseiller) ou ``conseil_refus`` (pourquoi il n'y en a pas). Le
        conseil n'existe QUE si le gate dit LIVE **et** que la lecture
        suffit — un refus est explicite, jamais un conseil sur des chiffres
        devinés.
        """
        from pfs.vision.live import CaptureImpossible, SondeIntrouvable

        titre = str(p.get("fenetre", "")).strip()
        if not titre:
            raise ValueError("champ 'fenetre' requis.")
        try:
            resultat = _mode_live().table(titre)
        except (CaptureImpossible, SondeIntrouvable) as e:
            return {"erreur": str(e)}

        resultat["conseil"] = None
        if resultat["gate"]["mode"] != "live":
            resultat["conseil_refus"] = ("table non prouvée fictive — "
                                         + resultat["gate"]["raison"])
            return resultat

        lecture = resultat["lecture"]
        m = lecture["montants"]
        manquants = [nom for nom in ("pot", "mise", "tapis", "blinde")
                     if m[nom]["valeur"] is None]
        if len(lecture["main"]) != 2 or manquants:
            causes = []
            if len(lecture["main"]) != 2:
                causes.append(f"main du héros : {len(lecture['main'])}/2 "
                              "carte(s) sûre(s)")
            causes += [f"{nom} : {m[nom]['refus']}" for nom in manquants]
            resultat["conseil_refus"] = " ; ".join(causes)
            return resultat

        # Le gate a dit LIVE et la lecture est complète : le conseiller est
        # convoqué MAINTENANT, sur les seuls chiffres lus — les mêmes que
        # ceux journalisés dans `lecture`, rien d'autre.
        resultat["conseil"] = API.advise({
            "hero": " ".join(lecture["main"]),
            "board": " ".join(lecture["tableau"]),
            "pot": m["pot"]["valeur"], "bet": m["mise"]["valeur"],
            "stack": m["tapis"]["valeur"], "big_blind": m["blinde"]["valeur"],
        })
        return resultat

    # ── Conseil sur un spot déjà joué (« qu'aurais-je dû faire ? ») ──────
    @staticmethod
    def advise(p: dict) -> dict:
        """Verdict sur une main terminée, décrite comme sur une capture.

        Payload : {"hero": "Ah Kd", "board": "Qs 7d 2c", "pot": 100,
                   "bet": 75, "stack": 300, "big_blind": 10,
                   "position": "BTN", "villain": "moyenne", "players": 2,
                   "stacks": "35, 125, 27", "payouts": "50, 30, 20",
                   "ante": 0.1}

        ``stacks`` et ``payouts`` font basculer le conseil en régime ICM.
        Sans eux, le verdict est celui du chipEV : sur une bulle de tournoi,
        les deux peuvent être OPPOSÉS. Ces deux champs étaient construits
        puis **jetés** — le régime ICM, pourtant implémenté, documenté et
        testé, était donc inatteignable depuis l'interface, qui rendait
        silencieusement un verdict de cash game sur des spots de tournoi.
        """
        from pfs.analysis import Spot, advise as _advise

        def _liste(v) -> str:
            """Accepte « 35, 125, 27 » comme [35, 125, 27]."""
            if v is None:
                return ""
            if isinstance(v, (list, tuple)):
                return ", ".join(str(x) for x in v)
            return str(v)

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
            stacks=_liste(p.get("stacks")),
            payouts=_liste(p.get("payouts")),
            ante=float(p.get("ante", 0) or 0),
        ))
        return {
            "hand": a.hand, "action": a.action, "confidence": a.confidence,
            "regime": a.regime, "equity": a.equity, "required": a.required,
            "mdf": a.mdf, "ev_bb": a.ev_bb, "ev_icm": a.ev_icm,
            "bubble": a.bubble, "size": a.size, "frequency": a.frequency,
            "reasons": a.reasons,
            "assumptions": a.assumptions, "explain": a.explain(),
        }


    # ── Simulateur de main ───────────────────────────────────────────────
    @staticmethod
    def simuler(p: dict) -> dict:
        """Tire une main au hasard et dit quoi faire selon la composition.

        Payload : {"joueurs": 2, "cartes_board": 0, "cartes_visibles": false,
                   "villain": "moyenne", "positions": ["BTN"],
                   "tapis": [5,8,10,12,15,20,25], "pot": 1.5, "bet": 0}
        """
        from pfs.train.simulateur import TAPIS_PAR_DEFAUT, simuler as _sim

        r = _sim(
            joueurs=int(p.get("joueurs", 2)),
            cartes_board=int(p.get("cartes_board", 0)),
            cartes_visibles=bool(p.get("cartes_visibles", False)),
            tapis=[float(x) for x in (p.get("tapis") or TAPIS_PAR_DEFAUT)],
            positions=[str(x) for x in (p.get("positions") or ["BTN"])],
            villain=str(p.get("villain", "moyenne")),
            pot=float(p.get("pot", 1.5) or 1.5),
            bet=float(p.get("bet", 0) or 0),
        )
        return {
            "hero": list(r.main.hero), "groupe": r.main.groupe,
            "board": list(r.main.board),
            "villains": [list(v) for v in r.main.villains],
            "cartes_visibles": r.cartes_visibles,
            "equite_reelle": r.equite_reelle,
            "equite_supposee": r.equite_supposee,
            "bascule_bb": r.bascule_bb,
            "verdicts": [
                {"tapis_bb": v.tapis_bb, "position": v.position,
                 "joueurs": v.joueurs, "action": v.action,
                 "certain": v.certain, "ev_bb": v.ev_bb,
                 "equite": v.equite, "requise": v.requise}
                for v in r.verdicts
            ],
            "explain": r.explain(),
        }

    # ── Lexique ──────────────────────────────────────────────────────────
    @staticmethod
    def lexique(p: dict) -> dict:
        """Vocabulaire du poker et du solveur. Payload : {"q": "motif"}."""
        from pfs.lexique import chercher

        return {"termes": [
            {"nom": t.nom, "categorie": t.categorie,
             "definition": t.definition, "pourquoi": t.pourquoi,
             "aussi": list(t.aussi)}
            for t in chercher(str(p.get("q", "")))
        ]}


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
    # drill/next : aucune UI ne l'appelle (start/answer embarquent « next »
    # dans leur réponse) — servie pour l'outillage/CLI, voir sa docstring.
    "drill/next": API.drill_next,
    "drill/answer": API.drill_answer,
    "drill/report": API.drill_report,
    "drill/fuites": API.drill_fuites,
    "drill/fuites/next": API.drill_fuites_next,
    "drill/fuites/answer": API.drill_fuites_answer,
    "emplacements": API.emplacements,
    "analyse": API.analyse_hh,
    "skill": API.skill,
    # presets : aucune UI ne l'appelle — introspection pour l'outillage/CLI
    # (les présets EXACTS du logiciel), voir sa docstring.
    "presets": API.presets,
    "icm": API.icm,
    "equity": API.equity,
    "postflop": API.postflop,
    "resolve": API.resolve,
    "review": API.review,
    "review/pushfold": API.review_pushfold,
    "reperes": API.reperes,
    "advise": API.advise,
    "recognize": API.recognize,
    "simuler": API.simuler,
    "lexique": API.lexique,
    "lire_capture": API.lire_capture,
    "live/fenetres": API.live_fenetres,
    "live/lire": API.live_lire,
    "live/armer": API.live_armer,
    "live/desarmer": API.live_desarmer,
    "live/table": API.live_table,
    "capture": API.capture,
    "archive": API.archive,
}


# Le mode Live est un état de processus (armements, mémoire du badge) : une
# seule instance, créée au premier usage — le serveur reste importable sans
# que le gate ni la vision du badge ne se construisent.
_MODE_LIVE_INSTANCE = None


def _mode_live():
    global _MODE_LIVE_INSTANCE
    if _MODE_LIVE_INSTANCE is None:
        from pfs.app.mode_live import ModeLive
        _MODE_LIVE_INSTANCE = ModeLive()
    return _MODE_LIVE_INSTANCE


# ═══════════════════════════════════════════════════════════════════════════
# SERVEUR
# ═══════════════════════════════════════════════════════════════════════════


def _make_handler(token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "PokerFusionSolver/4.5"
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
                # Sonde MANUELLE : le seul GET d'API, sans jeton — pour
                # vérifier depuis un navigateur ou `curl 127.0.0.1:8731/api/health`
                # que le serveur qui répond est bien le processus attendu
                # (voir _ServeurExclusif : un vieux listener peut capter tout
                # le port). Ne rend que {ok, version} : rien de sensible.
                self._json(200, {"ok": True, "version": "4.5"})
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


def _jeton_persistant() -> str:
    """Jeton aléatoire, conservé d'un démarrage à l'autre.

    Un jeton régénéré à chaque lancement invalide toutes les pages déjà
    ouvertes : chaque redémarrage renvoyait un 403 silencieux, et l'onglet
    semblait « ne plus rien reconnaître » alors que rien n'était calculé.
    Le jeton est donc tiré une fois puis relu, dans le dossier de données
    local de l'utilisateur.

    Le modèle de menace ne change pas : le serveur n'écoute que sur la boucle
    locale, et le fichier n'est lisible que par le compte qui l'a créé — les
    mêmes conditions que l'URL déjà présente dans l'historique du navigateur.
    """
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    chemin = Path(base) / "PokerFusionSolver" / "jeton"
    try:
        jeton = chemin.read_text(encoding="utf-8").strip()
        if len(jeton) >= 16:
            return jeton
    except OSError:
        pass
    jeton = secrets.token_urlsafe(24)
    try:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        chemin.write_text(jeton, encoding="utf-8")
        os.chmod(chemin, 0o600)
    except OSError:
        pass          # jeton de session : le logiciel reste utilisable
    return jeton


class _ServeurExclusif(ThreadingHTTPServer):
    """Serveur qui REFUSE de démarrer si le port est déjà pris.

    Ce détail a coûté des heures de diagnostic. `HTTPServer` active
    `allow_reuse_address` (SO_REUSEADDR) par défaut. Sur Unix cela sert
    seulement à réutiliser un port en TIME_WAIT ; **sur Windows, la même
    option autorise un second processus à se lier à un port déjà en
    écoute**. Mesuré sur cette machine : trois processus ont pu écouter
    simultanément sur le même port, `netstat` affichant les trois.

    Ce n'est pas un partage équitable, et c'est ce qui rend le symptôme si
    trompeur : **le plus ancien listener capte la totalité des connexions**
    (12/12 puis 6/6 dans la mesure), le suivant ne prenant le relais qu'à
    la mort du précédent.

    Symptôme observé : après modification du code, chaque redémarrage
    lançait un nouveau serveur pendant que l'ancien continuait de répondre —
    et de répondre à *tout*. Les routes fraîchement ajoutées renvoyaient
    donc « route inconnue » systématiquement, jamais par intermittence, et
    le défaut semblait venir du routeur.

    À ne pas confondre avec un autre motif, inoffensif celui-là : `python
    -m pfs` fait toujours apparaître **deux** PID, parce que le `python.exe`
    du venv n'est qu'un lanceur qui démarre le vrai `python3.13.exe` du
    paquet Store. C'est une filiation parent/enfant normale, et un seul
    socket est ouvert.

    `SO_EXCLUSIVEADDRUSE` rétablit le comportement attendu : le second
    démarrage échoue franchement, avec un message qui dit quoi faire.
    """

    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET,
                                       socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass    # option refusée : le garde-fou saute, pas le serveur
        super().server_bind()


def create_server(port: int = 8731,
                  token: str | None = None) -> tuple[ThreadingHTTPServer, str]:  # noqa: E501
    """Crée le serveur, lié à **127.0.0.1 uniquement**.

    ``token`` explicite (les tests en fournissent un) ou jeton persistant.
    """
    token = token if token is not None else _jeton_persistant()
    srv = _ServeurExclusif(("127.0.0.1", port), _make_handler(token))
    return srv, token


def run(port: int = 8731, open_browser: bool = True) -> None:
    try:
        srv, token = create_server(port)
    except OSError as e:
        # Sans ce message, un second démarrage échouait en silence pendant
        # que l'ancien serveur — donc l'ancien CODE — continuait de servir.
        print(f"\n  ✗ Impossible d'écouter sur le port {port} : {e}\n")
        print("  Un Poker Fusion Solver tourne déjà. Deux possibilités :")
        print(f"    • rejoins-le            : http://127.0.0.1:{port}/")
        print("    • ou arrête-le d'abord  : "
              "Get-Process python | Stop-Process\n")
        print(f"  (ou démarre ailleurs : python -m pfs --port {port + 1})\n")
        raise SystemExit(1) from e
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
