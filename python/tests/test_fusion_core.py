"""
Tests du noyau mathématique : F2, F3, F4, F5, F6, F8, range algebra, orchestrateur.

Chaque valeur golden du Plan Directeur §4 est verrouillée ici. Les relations
métamorphiques vérifient ce qui doit tenir pour *toute* entrée — c'est ce qui
attrape les bugs qu'aucun cas de test ne couvre.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfs.core.range_model import (
    GROUP_COMBO_COUNT,
    N_COMBOS,
    N_GROUPS,
    HandClass,
    Range,
    RangeError,
    combo_cards,
    combo_index,
    group_name,
    parse_range,
    GTO_PRESETS,
)
from pfs.fusion.bet_sizing import (
    LogisticCallModel,
    MDFCallModel,
    entropy_bits,
    expected_value,
    information_gain,
    knowledge_price,
    optimal_bet_size,
    sizing_table,
)
from pfs.fusion.bottleneck import (
    POKER_PREDICATES,
    compress_range,
    elbow_point,
    information_plane,
)
from pfs.fusion.geometry import (
    bhattacharyya_coefficient,
    fisher_rao_distance,
    hellinger_distance,
    jensen_shannon_distance,
    kmeans_fisher,
    natural_gradient_step,
    normalise,
)
from pfs.fusion.hmm import (
    DEFAULT_EMISSION,
    DEFAULT_TRANSITION,
    MentalState,
    Observation,
    OnlineHMM,
    fit_baum_welch,
)
from pfs.fusion.particle import Archetype, ParticleFilter
from pfs.solver.dcfr import (
    DCFRConfig,
    DCFRSolver,
    KuhnPoker,
    hyperparameter_schedule,
    regret_matching,
)


# ═══════════════════════════════════════════════════════════════════════
# RANGE — indexation, algèbre, métriques
# ═══════════════════════════════════════════════════════════════════════


def test_combo_indexing_is_a_bijection() -> None:
    seen = set()
    for i in range(N_COMBOS):
        a, b = combo_cards(i)
        assert a < b
        assert combo_index(a, b) == i
        assert combo_index(b, a) == i          # symétrique
        seen.add((a, b))
    assert len(seen) == N_COMBOS == math.comb(52, 2)


def test_group_combo_counts_are_6_4_12() -> None:
    """6 combos par paire, 4 par suited, 12 par offsuit — et 1326 au total."""
    assert GROUP_COMBO_COUNT.sum() == N_COMBOS
    pairs = [GROUP_COMBO_COUNT[g] for g in range(N_GROUPS) if g // 13 == g % 13]
    suited = [GROUP_COMBO_COUNT[g] for g in range(N_GROUPS) if g // 13 < g % 13]
    offsuit = [GROUP_COMBO_COUNT[g] for g in range(N_GROUPS) if g // 13 > g % 13]
    assert set(pairs) == {6} and len(pairs) == 13
    assert set(suited) == {4} and len(suited) == 78
    assert set(offsuit) == {12} and len(offsuit) == 78


def test_group_names_round_trip() -> None:
    assert group_name(0) == "AA"
    assert group_name(1) == "AKs"
    assert group_name(13) == "AKo"
    assert group_name(168) == "22"


def test_parse_range_basic() -> None:
    assert parse_range("AA").n_combos == 6
    assert parse_range("AKs").n_combos == 4
    assert parse_range("AKo").n_combos == 12
    assert parse_range("77+").n_combos == 8 * 6          # 77..AA
    assert parse_range("A2s+").n_combos == 12 * 4        # A2s..AKs


def test_parse_range_weights() -> None:
    r = parse_range("KQo:0.7")
    assert r.n_combos == pytest.approx(12 * 0.7)


def test_full_range_is_1326_combos() -> None:
    assert Range.full().n_combos == N_COMBOS
    assert Range.full().fraction == pytest.approx(1.0)


def test_blockers_only_exist_at_1326_resolution() -> None:
    """Retirer A♠ élimine 51 combos — invisible sur la grille 169."""
    r = Range.full()
    blocked = r.remove_blockers([0])          # A♠ = rank 0, suit 0
    assert r.n_combos - blocked.n_combos == 51


def test_range_algebra() -> None:
    a = parse_range("77+")
    b = parse_range("99+")
    assert (a & b).n_combos == b.n_combos
    assert (a | b).n_combos == a.n_combos
    assert (a - b).n_combos == a.n_combos - b.n_combos


def test_filtered_by_class() -> None:
    r = Range.full().filtered([HandClass.PAIR])
    assert r.n_combos == 13 * 6


def test_entropy_bounds() -> None:
    full = Range.full()
    assert full.entropy_bits == pytest.approx(math.log2(N_COMBOS), abs=1e-9)
    assert full.normalised_entropy == pytest.approx(1.0)
    single = Range.from_groups({"AA": 1.0})
    assert single.entropy_bits == pytest.approx(math.log2(6), abs=1e-9)


def test_polarised_range_has_lower_normalised_entropy() -> None:
    uniform = parse_range("22+")
    mixed = parse_range("AA:1.0, KK:0.1, QQ:0.02")
    assert mixed.normalised_entropy < uniform.normalised_entropy


def test_bayes_update_moves_mass_toward_likelihood() -> None:
    r = Range.full()
    eq = np.linspace(0.0, 1.0, N_COMBOS)
    post = r.bayes_update(eq)
    assert post.weights[-1] > post.weights[0]
    assert 0.0 <= post.weights.min() and post.weights.max() <= 1.0


def test_information_gain_is_bounded_by_one_bit() -> None:
    """Une observation binaire ne peut pas révéler plus d'un bit."""
    rng = np.random.default_rng(0)
    r = Range.full()
    for _ in range(20):
        lik = rng.random(N_COMBOS)
        assert 0.0 <= r.information_gain(lik) <= 1.0 + 1e-9


def test_gto_presets_have_plausible_widths() -> None:
    widths = {p: parse_range(s).fraction for p, s in GTO_PRESETS.items()}
    assert widths["UTG"] < widths["MP"] < widths["CO"] < widths["BTN"] < widths["SB"]
    assert 0.05 < widths["UTG"] < 0.15
    assert 0.35 < widths["BTN"] < 0.55


def test_gto_presets_contain_mixed_frequencies() -> None:
    """Une vraie solution de solveur n'est jamais binaire — les presets non plus."""
    for pos, spec in GTO_PRESETS.items():
        g = parse_range(spec).to_groups()
        assert sum(1 for x in g if 0.05 < x < 0.95) >= 5, pos


# ═══════════════════════════════════════════════════════════════════════
# F2 — HMM
# ═══════════════════════════════════════════════════════════════════════


def test_hmm_golden_single_wild_action() -> None:
    """Plan §4 F2 : α = [0.4728, 0.2864, 0.2409] après un 3-bet improbable."""
    h = OnlineHMM()
    b = h.update(Observation.WILD)
    assert b[MentalState.SOLID] == pytest.approx(0.4728, abs=5e-4)
    assert b[MentalState.LOOSE] == pytest.approx(0.2864, abs=5e-4)
    assert b[MentalState.TILT] == pytest.approx(0.2409, abs=5e-4)


def test_hmm_tilt_surges_by_factor_4_to_5() -> None:
    """De 5,4 % à 24,1 % : ×4,5 sur UNE action. Aucune stat brute ne bouge autant."""
    h = OnlineHMM()
    surge = h.tilt_surge(Observation.WILD)
    assert 4.0 < surge < 5.0


def test_hmm_belief_is_always_a_distribution() -> None:
    rng = np.random.default_rng(1)
    h = OnlineHMM()
    for _ in range(500):
        b = h.update(int(rng.integers(0, 5)))
        assert b.probs.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(b.probs > 0.0)


def test_hmm_repeated_folds_push_toward_solid() -> None:
    h = OnlineHMM()
    b = h.update_many([Observation.FOLD] * 30)
    assert b.most_likely is MentalState.SOLID
    assert b[MentalState.SOLID] > 0.75


def test_hmm_repeated_wild_pushes_toward_tilt() -> None:
    h = OnlineHMM()
    b = h.update_many([Observation.WILD] * 15)
    assert b.most_likely is MentalState.TILT
    assert b[MentalState.TILT] > 0.75


def test_hmm_recovers_from_tilt() -> None:
    """Un joueur doit pouvoir sortir du tilt — le plancher garantit l'atteignabilité."""
    h = OnlineHMM()
    h.update_many([Observation.WILD] * 20)
    b = h.update_many([Observation.FOLD] * 40)
    assert b.most_likely is MentalState.SOLID


def test_hmm_stationary_distribution_is_a_fixed_point() -> None:
    h = OnlineHMM()
    pi = h.stationary_distribution()
    assert pi.sum() == pytest.approx(1.0)
    assert np.allclose(pi @ DEFAULT_TRANSITION, pi, atol=1e-9)


def test_hmm_rejects_bad_matrices() -> None:
    with pytest.raises(ValueError):
        OnlineHMM(transition=np.ones((3, 3)))
    with pytest.raises(ValueError):
        OnlineHMM(prior=np.array([0.5, 0.4, 0.4]))


def test_baum_welch_recovers_structure_on_synthetic_data() -> None:
    """Sur des séquences engendrées par le modèle, l'EM doit augmenter la
    log-vraisemblance et retrouver des matrices stochastiques valides."""
    rng = np.random.default_rng(3)
    seqs = []
    for _ in range(40):
        state = 0
        seq = []
        for _ in range(60):
            state = int(rng.choice(3, p=DEFAULT_TRANSITION[state]))
            seq.append(int(rng.choice(5, p=DEFAULT_EMISSION[state])))
        seqs.append(seq)
    A, B, pi, ll = fit_baum_welch(seqs, n_iter=40)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(B.sum(axis=1), 1.0, atol=1e-6)
    assert math.isfinite(ll)


# ═══════════════════════════════════════════════════════════════════════
# F5 — GÉOMÉTRIE DE L'INFORMATION
# ═══════════════════════════════════════════════════════════════════════


def test_fisher_rao_golden() -> None:
    """Plan §4 F5, recalculé exactement : BC = 0.988249291, d = 0.306904480."""
    p = [0.40, 0.25, 0.20, 0.10, 0.05]
    q = [0.55, 0.20, 0.15, 0.07, 0.03]
    assert bhattacharyya_coefficient(p, q) == pytest.approx(0.988249291, abs=1e-7)
    assert fisher_rao_distance(p, q) == pytest.approx(0.306904480, abs=1e-6)


def test_fisher_rao_is_a_metric() -> None:
    rng = np.random.default_rng(2)
    pts = [normalise(rng.random(8)) for _ in range(12)]
    for p in pts:
        assert fisher_rao_distance(p, p) == pytest.approx(0.0, abs=1e-7)
    for p in pts:
        for q in pts:
            assert fisher_rao_distance(p, q) == pytest.approx(
                fisher_rao_distance(q, p), abs=1e-12
            )
    for p in pts[:5]:
        for q in pts[:5]:
            for r in pts[:5]:
                assert (
                    fisher_rao_distance(p, r)
                    <= fisher_rao_distance(p, q) + fisher_rao_distance(q, r) + 1e-7
                )


def test_fisher_rao_beats_euclidean_near_zero() -> None:
    """0.001→0.002 est un plus grand changement d'information que 0.500→0.501.

    C'est exactement ce que la distance euclidienne rate.
    """
    near_zero = (np.array([0.001, 0.999]), np.array([0.002, 0.998]))
    mid_range = (np.array([0.500, 0.500]), np.array([0.501, 0.499]))

    euclid_a = float(np.linalg.norm(near_zero[0] - near_zero[1]))
    euclid_b = float(np.linalg.norm(mid_range[0] - mid_range[1]))
    assert euclid_a == pytest.approx(euclid_b, rel=1e-9)   # euclidien : identiques

    d_a = fisher_rao_distance(*near_zero)
    d_b = fisher_rao_distance(*mid_range)
    assert d_a > 10.0 * d_b   # Fisher : un ordre de grandeur d'écart


def test_hellinger_and_js_are_bounded() -> None:
    rng = np.random.default_rng(4)
    for _ in range(30):
        p, q = normalise(rng.random(10)), normalise(rng.random(10))
        assert 0.0 <= hellinger_distance(p, q) <= 1.0
        assert 0.0 <= jensen_shannon_distance(p, q) <= 1.0 + 1e-9


def test_natural_gradient_stays_in_simplex_and_follows_gradient() -> None:
    p = normalise(np.full(5, 0.2))
    g = np.array([1.0, 0.0, 0.0, 0.0, -1.0])
    q = natural_gradient_step(p, g, step=0.5)
    assert q.sum() == pytest.approx(1.0)
    assert np.all(q > 0)
    assert q[0] > p[0] and q[4] < p[4]


def test_kmeans_fisher_separates_two_clear_clusters() -> None:
    rng = np.random.default_rng(6)
    a = [normalise(np.array([10.0, 1, 1, 1]) + rng.random(4) * 0.2) for _ in range(15)]
    b = [normalise(np.array([1.0, 1, 1, 10]) + rng.random(4) * 0.2) for _ in range(15)]
    _, labels = kmeans_fisher(a + b, k=2, seed=1)
    assert len(set(labels[:15])) == 1
    assert len(set(labels[15:])) == 1
    assert labels[0] != labels[15]


# ═══════════════════════════════════════════════════════════════════════
# F4 — BET SIZING
# ═══════════════════════════════════════════════════════════════════════


def test_information_gain_never_exceeds_one_bit() -> None:
    rng = np.random.default_rng(8)
    for _ in range(50):
        w = rng.random(200)
        c = rng.random(200)
        assert 0.0 <= information_gain(w, c) <= 1.0 + 1e-9


def test_information_gain_is_zero_when_action_is_uninformative() -> None:
    w = np.ones(100)
    assert information_gain(w, np.full(100, 0.5)) == pytest.approx(0.0, abs=1e-12)
    assert information_gain(w, np.ones(100)) == pytest.approx(0.0, abs=1e-12)


def test_information_gain_is_maximal_on_a_clean_split() -> None:
    """Un split 50/50 parfaitement séparant vaut exactement 1 bit."""
    w = np.ones(100)
    c = np.concatenate([np.ones(50), np.zeros(50)])
    assert information_gain(w, c) == pytest.approx(1.0, abs=1e-9)


def test_mdf_model_defends_the_theoretical_fraction() -> None:
    eq = np.linspace(0.0, 1.0, 1000)
    pot, bet = 45.0, 45.0
    probs = MDFCallModel(softness=0.0).call_probs(bet, pot, eq)
    assert probs.mean() == pytest.approx(pot / (pot + bet), abs=0.01)


def test_lambda_shifts_optimal_size_upward() -> None:
    """Payer l'information ⇒ sizing plus grand. C'est la thèse de F4."""
    rng = np.random.default_rng(9)
    eq = np.clip(rng.beta(2.0, 3.0, 400), 0.02, 0.98)
    w = np.ones(400)
    b0 = optimal_bet_size(45.0, w, eq, lam=0.0).bet
    b5 = optimal_bet_size(45.0, w, eq, lam=5.0).bet
    assert b5 > b0


def test_pure_information_maximisation_would_always_shove() -> None:
    """La raison d'être de la contrainte d'EV : sans elle, on part all-in."""
    rng = np.random.default_rng(10)
    eq = np.clip(rng.beta(2.0, 3.0, 300), 0.02, 0.98)
    t = sizing_table(45.0, np.ones(300), eq, lam=0.0,
                     fractions=(0.33, 0.55, 0.75, 1.0, 1.5, 2.0))
    assert t.best_info.fraction_of_pot == max(c.fraction_of_pot for c in t.candidates)


def test_knowledge_price_scales_with_horizon_and_uncertainty() -> None:
    assert knowledge_price(0, 0.10) == 0.0
    assert knowledge_price(400, 0.20) > knowledge_price(50, 0.20)
    assert knowledge_price(400, 0.20) > knowledge_price(400, 0.02)


def test_expected_value_scale_invariance() -> None:
    """Doubler pot ET bet double l'EV — homogénéité de degré 1."""
    rng = np.random.default_rng(12)
    eq = rng.random(120)
    w = np.ones(120)
    m = LogisticCallModel()
    ev1 = expected_value(w, eq, m.call_probs(30.0, 45.0, eq), 45.0, 30.0)
    ev2 = expected_value(w, eq, m.call_probs(60.0, 90.0, eq), 90.0, 60.0)
    assert ev2 == pytest.approx(2.0 * ev1, rel=1e-9)


def test_entropy_bits_matches_range_entropy() -> None:
    r = parse_range(GTO_PRESETS["CO"])
    assert entropy_bits(r.weights) == pytest.approx(r.entropy_bits, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# F6 — INFORMATION BOTTLENECK
# ═══════════════════════════════════════════════════════════════════════


def test_predicates_are_all_interpretable_and_nonempty() -> None:
    assert len(POKER_PREDICATES) > 100
    for p in POKER_PREDICATES:
        assert isinstance(p.name, str) and p.name
        m = p.mask()
        assert m.shape == (N_GROUPS,)


@pytest.mark.parametrize(
    "position,max_rules,min_fidelity",
    [("UTG", 8, 0.88), ("MP", 12, 0.85), ("CO", 16, 0.85),
     ("BTN", 16, 0.85), ("SB", 16, 0.83)],
)
def test_a_handful_of_rules_captures_most_of_a_range(
    position: str, max_rules: int, min_fidelity: float
) -> None:
    """La thèse de F6, mesurée sur des ranges à **fréquences mixtes**.

    Mesuré : UTG 8 règles → 91 % · MP 12 → 88 % · CO/BTN 16 → 87 % · SB 16 → 85 %.

    Ces chiffres sont plus bas que sur des ranges binaires (où UTG atteint
    95,5 % en 4 règles), et c'est le résultat intéressant : **ce sont les
    fréquences mixtes qui résistent à la compression**. C'est exactement ce que
    la communauté observe empiriquement — les spots mixtes sont impossibles à
    mémoriser — et l'Information Bottleneck le quantifie.
    """
    rs = compress_range(parse_range(GTO_PRESETS[position]), n_rules=max_rules)
    assert rs.fidelity_ratio >= min_fidelity
    assert rs.mae < 0.09
    assert rs.n_rules <= max_rules


def test_pure_spots_compress_far_better_than_mixed_ones() -> None:
    """Corollaire vérifiable : une range binarisée se compresse nettement mieux.

    C'est la mesure de ce que coûtent, en complexité, les stratégies mixtes.
    """
    mixed = parse_range(GTO_PRESETS["UTG"])
    binary = Range.from_groups((mixed.to_groups() > 0.5).astype(float))
    fm = compress_range(mixed, n_rules=6).fidelity_ratio
    fb = compress_range(binary, n_rules=6).fidelity_ratio
    assert fb > fm + 0.05



def test_fidelity_increases_monotonically_with_rules() -> None:
    r = parse_range(GTO_PRESETS["BTN"])
    pts = information_plane(r, max_rules=12)
    fid = [p.fidelity_ratio for p in pts]
    assert all(b >= a - 1e-9 for a, b in zip(fid, fid[1:]))


def test_elbow_is_where_gains_stop() -> None:
    pts = information_plane(parse_range(GTO_PRESETS["UTG"]), max_rules=16)
    e = elbow_point(pts, min_gain=0.01)
    assert 2 <= e.n_rules <= 12
    assert e.fidelity_ratio > 0.80


def test_ruleset_reconstruction_matches_reported_error() -> None:
    r = parse_range(GTO_PRESETS["CO"])
    rs = compress_range(r, n_rules=10)
    rec = rs.apply()
    mass = GROUP_COMBO_COUNT.astype(float)
    mae = float(np.sum(np.abs(rec - r.to_groups()) * mass) / mass.sum())
    assert mae == pytest.approx(rs.mae, abs=1e-9)


def test_rules_are_named_and_frequencies_are_rounded() -> None:
    rs = compress_range(parse_range(GTO_PRESETS["BTN"]), n_rules=8, round_to=0.05)
    for rule in rs.rules:
        assert rule.predicate_name
        assert abs(rule.frequency / 0.05 - round(rule.frequency / 0.05)) < 1e-9
        assert 0.0 <= rule.frequency <= 1.0


def test_specificity_penalty_matters() -> None:
    """s = 0 (glouton SSE pur) est nettement moins bon sur une range large."""
    r = parse_range(GTO_PRESETS["BTN"])
    naive = compress_range(r, n_rules=12, specificity=0.0)
    tuned = compress_range(r, n_rules=12, specificity=0.75)
    assert tuned.fidelity_ratio > naive.fidelity_ratio + 0.10


# ═══════════════════════════════════════════════════════════════════════
# F3 — FILTRE PARTICULAIRE
# ═══════════════════════════════════════════════════════════════════════


def test_particle_filter_starts_uniform_over_archetypes() -> None:
    pf = ParticleFilter(n_particles=200, seed=0)
    post = pf.archetype_posterior()
    assert sum(post.values()) == pytest.approx(1.0)
    assert pf.effective_sample_size == pytest.approx(200.0, rel=1e-6)


def test_aggressive_actions_shift_posterior_toward_aggressive_archetypes() -> None:
    rng = np.random.default_rng(13)
    eq = np.clip(rng.beta(2.0, 3.0, N_COMBOS), 0.02, 0.98)
    pf = ParticleFilter(n_particles=200, seed=0)
    before = pf.archetype_posterior()
    for _ in range(6):
        pf.observe(eq, "bet", 1.0)
    after = pf.archetype_posterior()
    aggro_before = before[Archetype.LAG] + before[Archetype.MANIAC]
    aggro_after = after[Archetype.LAG] + after[Archetype.MANIAC]
    assert aggro_after > aggro_before


def test_particle_filter_resamples_when_ess_collapses() -> None:
    rng = np.random.default_rng(14)
    eq = np.clip(rng.beta(2.0, 3.0, N_COMBOS), 0.02, 0.98)
    pf = ParticleFilter(n_particles=100, seed=0)
    for _ in range(12):
        pf.observe(eq, "bet", 1.5)
    assert pf.n_resamples >= 1
    assert pf.effective_sample_size > 1.0


def test_marginal_range_stays_valid() -> None:
    rng = np.random.default_rng(15)
    eq = np.clip(rng.beta(2.0, 3.0, N_COMBOS), 0.02, 0.98)
    pf = ParticleFilter(n_particles=60, seed=0)
    for act in ("call", "bet", "check", "raise", "fold"):
        r = pf.observe(eq, act, 0.5)
        assert r.weights.min() >= 0.0 and r.weights.max() <= 1.0


def test_betting_narrows_the_range() -> None:
    """Une mise doit réduire l'entropie de la range estimée."""
    rng = np.random.default_rng(16)
    eq = np.clip(rng.beta(2.0, 3.0, N_COMBOS), 0.02, 0.98)
    pf = ParticleFilter(n_particles=100, seed=0)
    h0 = pf.marginal_range().entropy_bits
    for _ in range(4):
        pf.observe(eq, "bet", 1.0)
    assert pf.marginal_range().entropy_bits < h0


# ═══════════════════════════════════════════════════════════════════════
# F8 — DCFR
# ═══════════════════════════════════════════════════════════════════════


def test_regret_matching_is_a_distribution() -> None:
    assert regret_matching(np.array([3.0, 1.0])).sum() == pytest.approx(1.0)
    uniform = regret_matching(np.array([-1.0, -2.0, -3.0]))
    assert np.allclose(uniform, 1 / 3)


def test_hyperparameter_schedule_converges_to_dcfr_defaults() -> None:
    a0, b0, g0 = hyperparameter_schedule(1, 1000)
    a1, b1, g1 = hyperparameter_schedule(1000, 1000)
    assert a0 > a1 and g0 > g1 and b0 < b1
    assert a1 == pytest.approx(1.5, abs=0.2)
    assert g1 == pytest.approx(2.0, abs=0.2)


def test_kuhn_converges_to_the_known_nash_value() -> None:
    """Valeur du jeu de Kuhn à l'équilibre : exactement −1/18."""
    r = DCFRSolver(KuhnPoker()).solve(iterations=800)
    assert r.game_value == pytest.approx(-1 / 18, abs=2e-3)


def test_kuhn_exploitability_decreases_monotonically() -> None:
    r = DCFRSolver(KuhnPoker()).solve(iterations=800, track_every=200)
    expl = [e for _, e in r.history]
    assert all(b <= a + 1e-9 for a, b in zip(expl, expl[1:]))
    assert expl[-1] < 1e-2


def test_hs_dcfr_beats_plain_dcfr() -> None:
    """Zhang, McAleer & Sandholm (2024) : à budget égal, le schedule gagne."""
    plain = DCFRSolver(KuhnPoker(), DCFRConfig(use_schedule=False)).solve(600)
    hs = DCFRSolver(KuhnPoker(), DCFRConfig(use_schedule=True)).solve(600)
    assert hs.exploitability < plain.exploitability


def test_kuhn_equilibrium_structure() -> None:
    """Dans tout équilibre de Kuhn : le joueur 1 mise le Roi 3× plus souvent
    que le Valet, et ne mise jamais la Dame en premier."""
    r = DCFRSolver(KuhnPoker()).solve(3000)
    jack = r.freq("0|")["b"]
    queen = r.freq("1|")["b"]
    king = r.freq("2|")["b"]
    assert queen < 0.05
    assert king == pytest.approx(3.0 * jack, abs=0.12)


def test_strategies_are_valid_distributions() -> None:
    r = DCFRSolver(KuhnPoker()).solve(300)
    for key, probs in r.strategy.items():
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(probs >= -1e-12)


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATEUR
# ═══════════════════════════════════════════════════════════════════════


def _engine_with_history(fold_rate: float, n: int, seed: int = 0):
    from pfs.core.bankroll import BankrollProfile
    from pfs.engine import FusionEngine

    rng = np.random.default_rng(seed)
    eng = FusionEngine(bankroll=BankrollProfile(5.0, 100.0, 3000.0), discount=1.0)
    key = "test_villain"
    eng.start_hand(key, prior_range=parse_range(GTO_PRESETS["BTN"]))
    for _ in range(n):
        eng.observe_stat(key, "fold_to_cbet", bool(rng.random() < fold_rate))
    eq = np.clip(rng.beta(2.2, 3.0, N_COMBOS), 0.02, 0.98)
    return eng, key, eq


def test_engine_recommends_gto_on_thin_sample() -> None:
    from pfs.fusion.arbiter import Action, ActionDistribution

    eng, key, eq = _engine_with_history(0.71, 12)
    d = eng.decide(
        key,
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        pot=45.0, equities=eq,
    )
    assert not d.fusion.significant
    assert d.fusion.strategy.get(Action.BET) < 0.72


def test_engine_exploits_on_thick_sample() -> None:
    from pfs.fusion.arbiter import Action, ActionDistribution

    eng, key, eq = _engine_with_history(0.75, 600)
    d = eng.decide(
        key,
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        pot=45.0, equities=eq,
    )
    assert d.fusion.significant
    assert d.fusion.strategy.get(Action.BET) > 0.72
    assert "fold_to_cbet" in d.opponent.exploitable


def test_engine_decision_is_fully_traceable() -> None:
    from pfs.fusion.arbiter import Action, ActionDistribution
    from pfs.fusion.hmm import Observation

    eng, key, eq = _engine_with_history(0.70, 200)
    eng.observe_action(key, "raise", eq, 1.2, hmm_obs=Observation.WILD)
    d = eng.decide(
        key,
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        pot=45.0, equities=eq, facing_bet=34.0,
    )
    txt = d.explain()
    for expected in ("RECOMMANDATION", "Adversaire", "état mental",
                     "archétypes", "Fisher-Rao", "Sizing", "Confiance"):
        assert expected in txt
    assert 0.0 <= d.confidence <= 1.0
    assert d.risk_of_ruin == pytest.approx(0.0497870684, abs=1e-9)


def test_engine_hmm_and_particle_are_wired() -> None:
    from pfs.fusion.hmm import MentalState, Observation

    eng, key, eq = _engine_with_history(0.5, 20)
    for _ in range(10):
        eng.observe_action(key, "bet", eq, 1.5, hmm_obs=Observation.WILD)
    b = eng.belief(key)
    assert b.mental[MentalState.TILT] > 0.4
    assert sum(b.archetypes.values()) == pytest.approx(1.0, abs=1e-6)
    assert b.range_estimate.entropy_bits > 0.0
