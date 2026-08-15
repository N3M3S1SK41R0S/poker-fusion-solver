"""
L'orchestrateur — là où les 13 fusions deviennent une seule décision.

Chaîne complète, dans l'ordre du pipeline temps réel :

    GameState (perception)
        │
        ├─ F1  Beta-Binomial dynamique  → θ̂ ± IC95 par statistique
        ├─ F2  HMM 3 états              → P(SOLID / LOOSE / TILT)
        ├─ F3  Filtre particulaire      → range adverse marginale sur les modèles
        │
        ├─ F5  Fisher-Rao               → écart de range au GTO, invariant
        ├─ F4  Lagrangien EV + λ·IG     → sizing optimal et prix de l'information
        ├─ F10 Détection du signal      → P(le call est +EV)
        ├─ F8  DCFR/blueprint           → σ_GTO
        │
        └─ F13 Arbitrage                → σ* = (1−λ)σ_GTO + λ σ_BR, borné en
                                            exploitabilité (Ganzfried-Sandholm)
        │
        └─ F9  Bankroll ergodique       → contexte de risque de la décision

F6 (Information Bottleneck) et F7 (topologie) tournent hors ligne : ils
alimentent respectivement le mode TRAIN et le mode ANALYZE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.bankroll import BankrollProfile, risk_of_ruin
from pfs.core.bluffcatch import BluffCatchAnalysis, analyse_bluffcatch
from pfs.core.range_model import N_COMBOS, Range, parse_range, GTO_PRESETS
from pfs.fusion.arbiter import (
    Action,
    ActionDistribution,
    FusionInput,
    FusionResult,
    arbitrate,
)
from pfs.fusion.bet_sizing import (
    MDFCallModel,
    SizingAnalysis,
    knowledge_price,
    sizing_table,
)
from pfs.fusion.dynamic_beta import GTO_BASELINES, OpponentProfile
from pfs.fusion.geometry import fisher_rao_distance, range_deviation_score
from pfs.fusion.hmm import MentalState as HMMState
from pfs.fusion.hmm import Observation, OnlineHMM
from pfs.data.player_notes import PlayerNotes, PlayerProfile
from pfs.fusion.particle import ParticleFilter

__all__ = ["Decision", "OpponentBelief", "FusionEngine"]

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class OpponentBelief:
    """Ce que le système croit savoir d'un adversaire, à un instant donné."""

    player_key: str
    stats: dict[str, tuple[float, float, tuple[float, float], int]]
    """stat → (θ̂, σ, IC95, n)"""
    mental: dict[HMMState, float]
    archetypes: dict[str, float]
    range_estimate: Range
    exploitable: tuple[str, ...]

    def explain(self) -> str:
        lines = [f"Adversaire {self.player_key[:8]}"]
        for name, (mean, std, (lo, hi), n) in sorted(self.stats.items()):
            flag = "  ⚠ exploitable" if name in self.exploitable else ""
            lines.append(
                f"  {name:<16} {mean * 100:5.1f}% ± {std * 100:4.1f}  "
                f"IC95 [{lo * 100:4.1f}, {hi * 100:4.1f}]  n={n}{flag}"
            )
        lines.append(
            "  état mental      "
            + "  ".join(f"{s.name}={self.mental[s] * 100:.0f}%" for s in HMMState)
        )
        top = sorted(self.archetypes.items(), key=lambda kv: -kv[1])[:3]
        lines.append("  archétypes       " + "  ".join(f"{k}={v * 100:.0f}%" for k, v in top))
        lines.append(f"  range estimée    {self.range_estimate}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Decision:
    """Recommandation complète, traçable jusqu'à chaque fusion qui l'a produite."""

    fusion: FusionResult
    sizing: SizingAnalysis | None
    bluffcatch: BluffCatchAnalysis | None
    opponent: OpponentBelief
    range_deviation: tuple[float, bool, str]
    knowledge_price_bb: float
    risk_of_ruin: float | None
    confidence: float

    @property
    def action(self) -> Action:
        return self.fusion.strategy.top()[0]

    def explain(self) -> str:
        parts = [
            "═══ RECOMMANDATION ═══",
            self.fusion.explain(),
            "",
            self.opponent.explain(),
            "",
            f"Écart de range au GTO (Fisher-Rao) : {self.range_deviation[0]:.3f} rad"
            f" — {self.range_deviation[2]}",
            f"Prix de l'information λ : {self.knowledge_price_bb:.2f} bb/bit",
        ]
        if self.sizing is not None:
            parts += ["", "── Sizing ──", self.sizing.explain()]
        if self.bluffcatch is not None:
            parts += ["", "── Bluff-catch ──", self.bluffcatch.explain()]
        if self.risk_of_ruin is not None:
            parts.append(f"\nRisque de ruine au niveau courant : {self.risk_of_ruin * 100:.2f} %")
        parts.append(f"\nConfiance globale : {self.confidence * 100:.0f} %")
        return "\n".join(parts)


class FusionEngine:
    """Assemble les 13 fusions en une décision unique et traçable.

    Un moteur par table. Les croyances adverses persistent d'une main à l'autre
    (c'est tout leur intérêt) ; l'inférence de range est réinitialisée à chaque
    nouvelle main.
    """

    __slots__ = ("_profiles", "_hmms", "_filters", "_bankroll", "_discount",
                 "_seed", "_notes", "_nicknames")

    def __init__(
        self,
        bankroll: BankrollProfile | None = None,
        discount: float = 0.99,
        seed: int | None = 0,
        notes: PlayerNotes | None = None,
    ) -> None:
        self._profiles: dict[str, OpponentProfile] = {}
        self._hmms: dict[str, OnlineHMM] = {}
        self._filters: dict[str, ParticleFilter] = {}
        self._bankroll = bankroll
        self._discount = discount
        self._seed = seed
        self._notes = notes
        self._nicknames: dict[str, str] = {}

    # ── cycle de vie ─────────────────────────────────────────────────────
    def profile(self, key: str) -> OpponentProfile:
        if key not in self._profiles:
            self._profiles[key] = OpponentProfile(player_key=key, discount=self._discount)
        return self._profiles[key]

    def hmm(self, key: str) -> OnlineHMM:
        if key not in self._hmms:
            self._hmms[key] = OnlineHMM()
        return self._hmms[key]

    def bind_nickname(self, key: str, nickname: str) -> PlayerProfile | None:
        """Associe un pseudo lu à l'écran à une clé de joueur.

        C'est **le seul** point de contact entre la vision et la base de
        profils : le pseudo sert de clé de lookup local, jamais de requête
        sortante. Retourne le profil trouvé, ou ``None``.
        """
        self._nicknames[key] = nickname
        return self._notes.lookup(nickname) if self._notes is not None else None

    def start_hand(self, key: str, prior_range: Range | None = None) -> None:
        """Réinitialise l'inférence de range — les stats, elles, persistent.

        Si un pseudo a été associé (:meth:`bind_nickname`) et qu'un profil
        externe existe en base locale, le prior d'archétype du filtre
        particulaire en est déformé — pondéré par le nombre de mains déjà
        observées, de sorte que l'observation directe reprenne la main.
        """
        priors = None
        if self._notes is not None and key in self._nicknames:
            n_obs = max((t.belief.n_observations for t in
                         self.profile(key).trackers.values()), default=0)
            priors = self._notes.blended_prior(self._nicknames[key], n_obs)
        self._filters[key] = ParticleFilter(
            n_particles=100, prior_range=prior_range, priors=priors, seed=self._seed
        )

    # ── ingestion ────────────────────────────────────────────────────────
    def observe_stat(self, key: str, stat: str, occurred: bool) -> None:
        """Une occurrence binaire (F1)."""
        self.profile(key).observe(stat, occurred)

    def observe_action(
        self,
        key: str,
        action: str,
        equities: F64,
        bet_frac: float = 0.0,
        hmm_obs: Observation | None = None,
    ) -> Range:
        """Une action observée : met à jour le HMM (F2) et le filtre (F3)."""
        if hmm_obs is not None:
            self.hmm(key).update(hmm_obs)
        if key not in self._filters:
            self.start_hand(key)
        return self._filters[key].observe(equities, action, bet_frac)

    # ── croyances ────────────────────────────────────────────────────────
    def belief(self, key: str) -> OpponentBelief:
        prof = self.profile(key)
        stats: dict[str, tuple[float, float, tuple[float, float], int]] = {}
        for name, tracker in prof.trackers.items():
            b = tracker.belief
            stats[name] = (b.mean, b.std, b.credible_interval(), b.n_observations)

        pf = self._filters.get(key)
        return OpponentBelief(
            player_key=key,
            stats=stats,
            mental=self.hmm(key).belief.as_mapping(),
            archetypes={k.value: v for k, v in (pf.archetype_posterior().items() if pf else [])},
            range_estimate=pf.marginal_range() if pf else Range.full(),
            exploitable=tuple(prof.exploitable_stats()),
        )

    # ── décision ─────────────────────────────────────────────────────────
    def decide(
        self,
        key: str,
        *,
        gto: ActionDistribution,
        best_response: ActionDistribution,
        pivot_stat: str = "fold_to_cbet",
        bluff_stat: str = "river_bluff_freq",
        pot: float | None = None,
        equities: F64 | None = None,
        hands_remaining: int = 200,
        facing_bet: float | None = None,
        ev_gto: float = 0.0,
        ev_best_response: float = 0.0,
        exploitability_br: float = 8.0,
        realized_gift: float = math.inf,
        reference_range: Range | None = None,
    ) -> Decision:
        """Produit la recommandation fusionnée.

        Parameters
        ----------
        pivot_stat
            Statistique sur laquelle porte l'écart exploité. C'est elle qui
            détermine λ via son incertitude postérieure (F1).
        facing_bet
            Si renseigné, déclenche l'analyse de bluff-catch (F10) — à condition
            que ``bluff_stat`` ait été observée. Utiliser la stat pivot comme
            fréquence de bluff serait un contresens sémantique.
        """
        prof = self.profile(key)
        tracker = prof.tracker(pivot_stat)
        b = tracker.belief
        baseline = GTO_BASELINES.get(pivot_stat, 0.5)

        mental = self.hmm(key).belief.as_mapping()

        fusion = arbitrate(
            FusionInput(
                gto=gto,
                best_response=best_response,
                deviation=abs(b.mean - baseline),
                deviation_std=b.std,
                mental_state_probs=mental,
                ev_gto=ev_gto,
                ev_best_response=ev_best_response,
                exploitability_gto=0.0,
                exploitability_br=exploitability_br,
                realized_gift=realized_gift,
            )
        )

        lam = knowledge_price(hands_remaining, b.std)

        sizing = None
        if pot is not None and equities is not None:
            sizing = sizing_table(
                pot, self.belief(key).range_estimate.weights, equities,
                lam=lam, model=MDFCallModel(),
            )

        bluff = None
        if facing_bet is not None and pot is not None and bluff_stat in prof.trackers:
            bb = prof.trackers[bluff_stat].belief
            bluff = analyse_bluffcatch(pot, facing_bet, bb.mean, bb.std)

        pf = self._filters.get(key)
        ref = reference_range or parse_range(GTO_PRESETS["BTN"])
        est = pf.marginal_range() if pf else Range.full()
        deviation = range_deviation_score(est.to_groups() + 1e-9, ref.to_groups() + 1e-9)

        ror = None
        if self._bankroll is not None:
            ror = risk_of_ruin(
                self._bankroll.winrate_bb100,
                self._bankroll.stddev_bb100,
                self._bankroll.bankroll_bb,
            )

        # Confiance : combine la taille d'échantillon, la netteté de la
        # croyance mentale et la santé du filtre particulaire.
        n_factor = min(1.0, b.n_observations / 150.0)
        mental_factor = 1.0 - self.hmm(key).belief.entropy_bits / math.log2(3)
        pf_factor = (
            pf.effective_sample_size / len(pf.particles) if pf else 0.5
        )
        confidence = float(np.clip(0.5 * n_factor + 0.25 * mental_factor + 0.25 * pf_factor, 0, 1))

        return Decision(
            fusion=fusion,
            sizing=sizing,
            bluffcatch=bluff,
            opponent=self.belief(key),
            range_deviation=deviation,
            knowledge_price_bb=lam,
            risk_of_ruin=ror,
            confidence=confidence,
        )


# ═══════════════════════════════════════════════════════════════════════════
# P4 — LE DIFFÉRENCIATEUR FINAL : re-solve depuis la range inférée en direct
# ═══════════════════════════════════════════════════════════════════════════
#
# Personne ne fait ça : PioSOLVER re-solve depuis des ranges SAISIES à la
# main ; GTO Wizard depuis des ranges de préset ; les IA de recherche jouent
# l'équilibre sans regarder l'adversaire. Ici : la range adverse est la
# MARGINALE A POSTERIORI du filtre particulaire (F3), nourrie par les
# observations de la session — et le solveur (L1) re-solve le spot exact
# contre CETTE range, avec l'incertitude qui tempère (F13).


@dataclass(frozen=True, slots=True)
class ResolveReport:
    """Résultat d'un re-solve en direct depuis la range inférée."""
    villain_source: str            # « inférée (ESS=…) » ou « préset (repli) »
    hero_position: str             # "oop" | "ip"
    ev_hero_exploit: float         # EV héros contre la range inférée
    ev_hero_gto_locked: float      # EV de la stratégie GTO-préset FIGÉE,
    #                                contre la même range inférée re-solvée
    exploit_gain: float            # la valeur de l'inférence, en jetons
    lam: float                     # tempérance F13 appliquée à la lecture
    exploitability: float          # du solve exploit (fraction du pot)
    root_actions: tuple            # stratégie racine héros (RootAction…)
    confidence: float

    def explain(self) -> str:
        return (
            f"vilain : {self.villain_source} · héros {self.hero_position.upper()}\n"
            f"EV re-solve {self.ev_hero_exploit:.2f}"
            f"  vs GTO figé {self.ev_hero_gto_locked:.2f}"
            f"  → gain d'inférence {self.exploit_gain:+.2f} jetons"
            f" (λ={self.lam:.2f}, confiance {self.confidence:.2f})"
        )


def _lock_player_everywhere(dst, src, player: int) -> None:
    """Fige, dans le solveur ``dst``, la stratégie moyenne de ``src`` pour
    ``player`` à chaque nœud (arbres identiques exigés)."""
    for idx, node in enumerate(dst._nodes):
        if node.player != player:
            continue
        sigma = src.average_strategy(idx)
        n = dst.players[player].n
        node.locked_mask = np.ones(n, dtype=bool)
        node.locked_sigma = np.asarray(sigma, dtype=np.float64).copy()


def resolve_spot(
    engine: "FusionEngine",
    key: str,
    *,
    board: Sequence[int],
    pot: float,
    hero_range: Range,
    hero_position: str = "ip",
    hero_stack: float | None = None,
    villain_stack: float | None = None,
    stack: float | None = None,
    bet_fracs: Sequence[float] = (0.75,),
    max_bets: int = 2,
    iterations: int = 300,
    reference_villain: Range | None = None,
    game_format: str = "cash",
    rake=None,
    min_ess: float = 0.25,
) -> ResolveReport:
    """Re-solve le spot contre la range adverse inférée par F3.

    Le rapport chiffre la valeur de l'inférence : EV du re-solve MOINS l'EV
    de la stratégie GTO-préset figée face à la même range (le vilain inféré
    répondant au mieux dans les deux cas). λ (F13) tempère la lecture :
    à faible confiance, la range inférée est mélangée au préset
    (λ·inférée + (1−λ)·préset) — on n'exploite que ce qu'on sait.

    Stacks asymétriques : ``hero_stack``/``villain_stack`` → tapis effectif
    min des deux (la seule quantité qui compte en heads-up).
    Format : "cash" (rake appliqué) ou "mtt" (rake à zéro par défaut).
    """
    from pfs.core.rake import NO_RAKE, RakeModel
    from pfs.solver.postflop import IP, OOP, PostflopSolver

    if hero_position not in ("oop", "ip"):
        raise ValueError("hero_position ∈ {'oop','ip'}.")
    if game_format not in ("cash", "mtt"):
        raise ValueError("game_format ∈ {'cash','mtt'}.")
    if len(board) == 3:
        # Depuis Phase 2, PostflopSolver ACCEPTE un flop — mais un re-solve
        # live en construirait DEUX en profondeur complète (~13 s/itération
        # à ranges réalistes, banc_flop.py) : hors budget du temps réel.
        # Le live re-solve turn/river ; le flop attend le blueprint.
        raise ValueError(
            "resolve_spot : board de 3 cartes — le re-solve flop en "
            "profondeur complète est hors budget live (mesures "
            "banc_flop.py) ; turn/river uniquement jusqu'au blueprint "
            "Phase 2. Le flop reste accessible via PostflopSolver (API "
            "Python).")
    if stack is None:
        if hero_stack is None or villain_stack is None:
            raise ValueError("stack OU (hero_stack ET villain_stack) requis.")
        stack = min(float(hero_stack), float(villain_stack))
    rake = rake if rake is not None else NO_RAKE
    if game_format == "mtt" and rake is not NO_RAKE:
        raise ValueError("pas de rake par main en MTT.")

    ref = reference_villain or parse_range(GTO_PRESETS["BTN"])
    pf = engine._filters.get(key)
    lam = 0.0
    if pf is not None:
        ess = pf.effective_sample_size / len(pf.particles)
        b = engine.belief(key)
        lam = float(np.clip(b.confidence if hasattr(b, "confidence") else ess, 0, 1))
        lam = max(lam, 0.0)
        inferred = pf.marginal_range()
        if ess < min_ess:
            villain, source = ref, f"préset (repli : ESS {ess:.2f} < {min_ess})"
            lam = 0.0
        else:
            blended = Range(np.clip(
                lam * inferred.weights + (1.0 - lam) * ref.weights, 0.0, 1.0))
            villain = blended
            source = f"inférée ⊕ préset (λ={lam:.2f}, ESS={ess:.2f})"
    else:
        villain, source = ref, "préset (aucune observation)"

    def _make(v: Range) -> PostflopSolver:
        oop_r = hero_range if hero_position == "oop" else v
        ip_r = v if hero_position == "oop" else hero_range
        return PostflopSolver(board, oop_r, ip_r, pot=pot, stack=stack,
                              bet_fracs=bet_fracs, max_bets=max_bets,
                              rake=rake)

    hero_side = OOP if hero_position == "oop" else IP

    # 1) le re-solve exploitant (contre la range inférée/mélangée)
    s_exploit = _make(villain).solve(iterations)
    ev_exploit = s_exploit.values()[hero_side]

    # 2) la stratégie « GTO » : même arbre, vilain = préset
    s_gto = _make(ref).solve(iterations)

    # 3) sa valeur RÉELLE contre la range inférée : héros FIGÉ sur σ_GTO,
    #    le vilain inféré re-solve librement autour (pire cas honnête)
    s_locked = _make(villain)
    _lock_player_everywhere(s_locked, s_gto, hero_side)
    s_locked.solve(iterations)
    ev_locked = s_locked.values()[hero_side]

    conf = 0.0
    if pf is not None:
        conf = pf.effective_sample_size / len(pf.particles)

    return ResolveReport(
        villain_source=source,
        hero_position=hero_position,
        ev_hero_exploit=float(ev_exploit),
        ev_hero_gto_locked=float(ev_locked),
        exploit_gain=float(ev_exploit - ev_locked),
        lam=lam,
        exploitability=float(s_exploit.exploitability()),
        root_actions=s_exploit.result().root_actions,
        confidence=float(conf),
    )


FusionEngine.resolve_spot = resolve_spot
