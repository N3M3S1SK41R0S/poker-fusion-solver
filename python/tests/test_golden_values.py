"""
Golden tests — chaque valeur chiffrée du Plan Directeur v2.0 §4 est vérifiée ici.

Si un de ces tests casse, c'est soit le code, soit le plan qui est faux.
Les deux doivent rester d'accord.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pfs.core.bankroll import (
    BankrollProfile,
    bankroll_for_ror,
    confidence_interval_winrate,
    ergodic_penalty,
    growth_rate,
    hands_for_significance,
    kelly_fraction,
    risk_of_ruin,
    should_take_shot,
)
from pfs.core.bluffcatch import (
    analyse_bluffcatch,
    minimum_defence_frequency,
    required_equity,
)
from pfs.fusion.arbiter import (
    Action,
    ActionDistribution,
    FusionInput,
    MentalState,
    arbitrate,
)
from pfs.fusion.dynamic_beta import GTO_BASELINES, DynamicBetaTracker


# ═══════════════════════════════════════════════════════════════════════
# F9 — BANKROLL  (Plan §4 F9)
# ═══════════════════════════════════════════════════════════════════════


def test_ror_golden() -> None:
    """RoR(mu=5, sigma=100, B=3000) = exp(-3) = 0.049787."""
    assert risk_of_ruin(5.0, 100.0, 3000.0) == pytest.approx(0.0497870684, abs=1e-9)


def test_bankroll_for_1pct_ror() -> None:
    """B pour RoR=1 % : 10000 × ln(100) / 10 = 4605.17 bb ≈ 46 buy-ins."""
    b = bankroll_for_ror(5.0, 100.0, 0.01)
    assert b == pytest.approx(4605.17, abs=0.01)
    assert b / 100.0 == pytest.approx(46.05, abs=0.01)


def test_ror_and_bankroll_are_inverse() -> None:
    for mu in (1.0, 3.0, 5.0, 10.0):
        for sd in (60.0, 100.0, 140.0):
            for target in (0.001, 0.01, 0.05):
                b = bankroll_for_ror(mu, sd, target)
                assert risk_of_ruin(mu, sd, b) == pytest.approx(target, rel=1e-9)


def test_half_kelly_keeps_75pct_of_growth() -> None:
    """Résultat analytique : g(f*/2) = 0.75 · g(f*), exactement."""
    mu, sd = 5.0, 100.0
    f_full = kelly_fraction(mu, sd, 1.0)
    f_half = kelly_fraction(mu, sd, 0.5)
    assert growth_rate(f_half, mu, sd) == pytest.approx(
        0.75 * growth_rate(f_full, mu, sd), rel=1e-12
    )


def test_full_kelly_maximises_growth() -> None:
    mu, sd = 5.0, 100.0
    f_star = kelly_fraction(mu, sd, 1.0)
    g_star = growth_rate(f_star, mu, sd)
    for f in np.linspace(1e-6, 2.0 * f_star, 500):
        assert growth_rate(float(f), mu, sd) <= g_star + 1e-12


def test_ror_is_one_for_losing_player() -> None:
    assert risk_of_ruin(-1.0, 100.0, 10_000.0) == 1.0
    assert risk_of_ruin(0.0, 100.0, 10_000.0) == 1.0


def test_ror_monotonicity() -> None:
    """Plus de bankroll ⇒ moins de risque. Plus de variance ⇒ plus de risque."""
    prev = 1.0
    for b in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
        cur = risk_of_ruin(5.0, 100.0, b)
        assert cur < prev
        prev = cur
    assert risk_of_ruin(5.0, 140.0, 3000.0) > risk_of_ruin(5.0, 100.0, 3000.0)


def test_hands_for_significance_order_of_magnitude() -> None:
    """mu=5, sigma=100 ⇒ ~314 000 mains. Le chiffre qui tue les 'winrates' à 30k."""
    n = hands_for_significance(5.0, 100.0)
    assert 300_000 <= n <= 320_000


def test_ci_width_at_10k_hands() -> None:
    lo, hi = confidence_interval_winrate(5.0, 100.0, 10_000)
    assert hi - lo == pytest.approx(39.2, abs=0.1)
    assert lo < 0.0 < hi  # l'IC contient 0 : aucune preuve d'être gagnant


def test_ergodic_penalty_decreases_with_bankroll() -> None:
    p_small = ergodic_penalty(5.0, 100.0, 1000.0)
    p_large = ergodic_penalty(5.0, 100.0, 10_000.0)
    assert p_small > p_large
    assert p_small / p_large == pytest.approx(100.0, rel=1e-9)


def test_shot_refused_when_ror_too_high() -> None:
    profile = BankrollProfile(winrate_bb100=5.0, stddev_bb100=100.0, bankroll_bb=2000.0)
    ok, m = should_take_shot(profile, 400.0, 2.0, 110.0, max_acceptable_ror=0.05)
    assert ok is False
    assert m["buyins_at_shot"] == pytest.approx(5.0)


# ═══════════════════════════════════════════════════════════════════════
# F10 — BLUFF-CATCH  (Plan §4 F10)
# ═══════════════════════════════════════════════════════════════════════


def test_pot_odds_golden() -> None:
    """Pot 100, bet 75 ⇒ alpha = 75/250 = 0.30."""
    assert required_equity(100.0, 75.0) == pytest.approx(0.30, abs=1e-12)


def test_mdf_golden() -> None:
    assert minimum_defence_frequency(100.0, 75.0) == pytest.approx(100.0 / 175.0)


def test_bluffcatch_confidence_golden() -> None:
    """Exemple du plan : p=0.34, sigma=0.09 ⇒ z=0.444, P(+EV)=0.6715."""
    a = analyse_bluffcatch(100.0, 75.0, 0.34, 0.09)
    assert a.margin == pytest.approx(0.04, abs=1e-12)
    assert a.z_margin == pytest.approx(0.4444444, abs=1e-6)
    assert a.prob_call_is_plus_ev == pytest.approx(0.671639, abs=1e-6)
    assert a.recommendation.startswith("CALL")


def test_bluffcatch_indifference_at_alpha() -> None:
    """À p = alpha exactement, EV(call) = EV(fold) = 0 et P(+EV) = 0.5."""
    alpha = required_equity(100.0, 75.0)
    a = analyse_bluffcatch(100.0, 75.0, alpha, 0.09)
    assert a.ev_call == pytest.approx(0.0, abs=1e-9)
    assert a.prob_call_is_plus_ev == pytest.approx(0.5, abs=1e-9)


def test_bluffcatch_high_uncertainty_is_marginal() -> None:
    """Même marge, mais incertitude énorme ⇒ la reco devient MARGINAL."""
    a = analyse_bluffcatch(100.0, 75.0, 0.34, 0.40)
    assert a.recommendation.startswith("MARGINAL")


@pytest.mark.parametrize("bet", [10.0, 33.0, 50.0, 75.0, 100.0, 150.0, 300.0])
def test_alpha_plus_mdf_identity(bet: float) -> None:
    """Identité : alpha = (1-MDF)/(1+... ) — on vérifie la cohérence bornes."""
    pot = 100.0
    alpha = required_equity(pot, bet)
    mdf = minimum_defence_frequency(pot, bet)
    assert 0.0 < alpha < 0.5
    assert 0.0 < mdf < 1.0
    # Un bet plus gros exige plus d'équité et permet de défendre moins.
    assert alpha == pytest.approx(bet / (pot + 2 * bet))
    assert mdf == pytest.approx(pot / (pot + bet))


# ═══════════════════════════════════════════════════════════════════════
# F1 — BETA-BINOMIAL DYNAMIQUE  (Plan §4 F1)
# ═══════════════════════════════════════════════════════════════════════


def test_beta_binomial_golden_no_discount() -> None:
    """Sans oubli : 14 VPIP / 40 mains, prior Jeffreys ⇒ theta = 14.5/41."""
    t = DynamicBetaTracker("vpip", discount=1.0)
    t.update_batch(successes=14, trials=40)
    b = t.belief
    assert b.alpha == pytest.approx(14.5)
    assert b.beta == pytest.approx(26.5)
    assert b.mean == pytest.approx(0.3536585, abs=1e-7)
    assert b.variance == pytest.approx(0.0054425, abs=1e-7)
    assert b.std == pytest.approx(0.0737733, abs=1e-6)


def test_beta_binomial_credible_interval_golden() -> None:
    """L'IC95 exact [0.212, 0.505] — c'est lui qui dit 'n'exploite pas'."""
    t = DynamicBetaTracker("vpip", discount=1.0)
    t.update_batch(successes=14, trials=40)
    lo, hi = t.belief.credible_interval(0.95)
    assert lo == pytest.approx(0.212, abs=0.005)
    assert hi == pytest.approx(0.505, abs=0.005)


def test_beta_binomial_not_exploitable_at_40_hands() -> None:
    """|0.3537 - 0.24| = 0.1137  <  1.96 × 0.0738 = 0.1446 ⇒ ne pas exploiter."""
    t = DynamicBetaTracker("vpip", discount=1.0)
    t.update_batch(successes=14, trials=40)
    b = t.belief
    assert b.is_exploitable(GTO_BASELINES["vpip"]) is False
    assert abs(b.deviation_z(GTO_BASELINES["vpip"])) < 1.96


def test_beta_binomial_becomes_exploitable_with_more_hands() -> None:
    """Même fréquence, plus d'échantillon ⇒ l'écart devient significatif."""
    t = DynamicBetaTracker("vpip", discount=1.0)
    t.update_batch(successes=140, trials=400)
    assert t.belief.is_exploitable(GTO_BASELINES["vpip"]) is True


def test_state_stays_in_unit_interval_always() -> None:
    """Property : contrairement au Kalman gaussien, theta ne sort JAMAIS de (0,1)."""
    rng = np.random.default_rng(0)
    for delta in (0.90, 0.95, 0.99, 1.0):
        t = DynamicBetaTracker("x", discount=delta)
        for _ in range(2000):
            b = t.update(bool(rng.random() < 0.03))
            assert 0.0 < b.mean < 1.0
            assert b.variance > 0.0


def test_variance_decreases_with_sample() -> None:
    """Sans oubli, la variance postérieure décroît en O(1/n).

    Elle n'est pas monotone *pas à pas* (chaque observation déplace la
    moyenne, donc p(1-p)), mais elle décroît sur toute échelle utile.
    """
    t = DynamicBetaTracker("x", discount=1.0)
    marks: dict[int, float] = {}
    for i in range(1, 4001):
        b = t.update(i % 3 == 0)
        if i in (50, 200, 800, 4000):
            marks[i] = b.variance
    assert marks[50] > marks[200] > marks[800] > marks[4000]
    # Décroissance en 1/n : var(4000)/var(50) doit être proche de 50/4000.
    assert marks[4000] / marks[50] == pytest.approx(50 / 4000, rel=0.15)


def test_discount_bounds_effective_sample() -> None:
    """Avec oubli, la variance ne tend PAS vers 0 — c'est le point du modèle."""
    t = DynamicBetaTracker("x", discount=0.99)
    assert t.effective_sample_size == pytest.approx(100.0)
    for i in range(5000):
        t.update(i % 4 == 0)
    b = t.belief
    # n_eff ~ 100 ⇒ alpha+beta plafonne autour de ~101, pas 5000.
    assert 90.0 < (b.alpha + b.beta) < 115.0
    assert b.std > 0.03  # incertitude irréductible : l'adversaire peut changer


def test_tracker_adapts_to_regime_change() -> None:
    """Un adversaire qui passe de 20 % à 60 % doit être suivi, pas moyenné."""
    t = DynamicBetaTracker("vpip", discount=0.97)
    for i in range(300):
        t.update(i % 5 == 0)          # 20 %
    before = t.belief.mean
    for i in range(300):
        t.update(i % 5 != 0)          # 80 %
    after = t.belief.mean
    assert before == pytest.approx(0.20, abs=0.06)
    assert after > 0.65
    # Un estimateur sans oubli aurait donné 0.50 — inutilisable.


# ═══════════════════════════════════════════════════════════════════════
# F13 — MÉTA-FUSION  (Plan §4 F13)
# ═══════════════════════════════════════════════════════════════════════


def _cbet_case(gift: float = math.inf) -> FusionInput:
    sigma = math.sqrt(0.71 * 0.29 / 84)
    return FusionInput(
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        deviation=0.16,
        deviation_std=sigma,
        mental_state_probs={
            MentalState.SOLID: 0.4728,
            MentalState.LOOSE: 0.2864,
            MentalState.TILT: 0.2408,
        },
        ev_gto=0.0,
        ev_best_response=1.20,
        exploitability_gto=0.0,
        exploitability_br=8.0,
        realized_gift=gift,
    )


def test_fusion_golden_lambda_and_frequency() -> None:
    """Exemple du plan §F13 : lambda = 0.5812, c-bet fusionné = 77.11 %."""
    res = arbitrate(_cbet_case())
    assert res.z_score == pytest.approx(3.231703, abs=1e-5)
    assert res.significant is True
    assert res.lambda_raw == pytest.approx(0.898261, abs=1e-5)
    assert res.adaptation_risk == pytest.approx(0.353000, abs=1e-6)
    assert res.lambda_final == pytest.approx(0.581175, abs=1e-5)
    assert res.strategy.get(Action.BET) == pytest.approx(0.771105, abs=1e-5)


def test_fusion_falls_back_to_gto_when_sample_is_thin() -> None:
    """Même déviation, 12 mains seulement ⇒ lambda ~ 0, on joue GTO."""
    sigma = math.sqrt(0.71 * 0.29 / 12)
    inp = FusionInput(
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        deviation=0.16,
        deviation_std=sigma,
        mental_state_probs={MentalState.SOLID: 1.0},
    )
    res = arbitrate(inp)
    assert res.significant is False
    assert res.lambda_final < 0.25
    assert res.strategy.get(Action.BET) < 0.70


def test_fusion_safe_exploitation_cap() -> None:
    """La borne Ganzfried-Sandholm plafonne lambda quand le cadeau est faible."""
    res = arbitrate(_cbet_case(gift=2.0))
    assert res.exploitability_capped is True
    assert res.lambda_final == pytest.approx(2.0 / 8.0, abs=1e-9)
    assert res.exploitability <= 2.0 + 1e-9


def test_fusion_tilt_allows_more_exploitation_than_solid() -> None:
    """Un joueur en tilt ne se défend pas : lambda doit être plus élevé."""
    base = _cbet_case()
    solid = arbitrate(
        replace(base, mental_state_probs={MentalState.SOLID: 1.0})
    )
    tilt = arbitrate(
        replace(base, mental_state_probs={MentalState.TILT: 1.0})
    )
    assert tilt.lambda_final > solid.lambda_final
    assert tilt.strategy.get(Action.BET) > solid.strategy.get(Action.BET)


def test_blend_is_always_a_valid_distribution() -> None:
    """Property : le mélange reste dans le simplexe pour tout poids."""
    a = ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38})
    b = ActionDistribution({Action.BET: 0.88, Action.RAISE: 0.12})
    for w in np.linspace(0.0, 1.0, 101):
        m = a.blend(b, float(w))
        assert sum(m.probs.values()) == pytest.approx(1.0, abs=1e-9)
        assert all(p >= -1e-12 for p in m.probs.values())


def test_blend_endpoints() -> None:
    a = ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38})
    b = ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12})
    assert a.blend(b, 0.0).get(Action.BET) == pytest.approx(0.62)
    assert a.blend(b, 1.0).get(Action.BET) == pytest.approx(0.88)


def test_fusion_zero_sigma_fails_safe_to_gto() -> None:
    """sigma = 0 ⇒ on ne divise pas par zéro, on retombe sur le GTO."""
    inp = FusionInput(
        gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
        best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
        deviation=0.16,
        deviation_std=0.0,
        mental_state_probs={MentalState.SOLID: 1.0},
    )
    res = arbitrate(inp)
    assert res.lambda_final == 0.0
    assert res.strategy.get(Action.BET) == pytest.approx(0.62)


def test_invalid_distribution_rejected() -> None:
    with pytest.raises(ValueError):
        ActionDistribution({Action.BET: 0.6, Action.CHECK: 0.3})


# ═══════════════════════════════════════════════════════════════════════
# RELATIONS MÉTAMORPHIQUES  (Plan §9 niveau 3)
# ═══════════════════════════════════════════════════════════════════════


def test_metamorphic_scale_invariance_of_pot_odds() -> None:
    """Doubler pot et bet ne change ni alpha ni MDF."""
    for k in (0.5, 2.0, 10.0, 137.0):
        assert required_equity(100.0 * k, 75.0 * k) == pytest.approx(
            required_equity(100.0, 75.0)
        )
        assert minimum_defence_frequency(100.0 * k, 75.0 * k) == pytest.approx(
            minimum_defence_frequency(100.0, 75.0)
        )


def test_metamorphic_bankroll_unit_scaling() -> None:
    """RoR est invariant si (mu, sigma, B) sont mis à la même échelle en 'stakes'."""
    base = risk_of_ruin(5.0, 100.0, 3000.0)
    # mu et sigma sont en bb ; doubler la stake divise mu, sigma, B en bb ? Non :
    # la propriété exacte est l'invariance par mise à l'échelle de (mu, sigma^2/B).
    assert risk_of_ruin(10.0, 100.0, 1500.0) == pytest.approx(base, rel=1e-12)
    assert risk_of_ruin(5.0, 200.0, 12000.0) == pytest.approx(base, rel=1e-12)


def test_metamorphic_more_information_never_hurts_lambda_monotonicity() -> None:
    """À déviation fixe, réduire sigma (plus de mains) augmente lambda."""
    prev = -1.0
    for n in (20, 50, 100, 300, 1000):
        sigma = math.sqrt(0.71 * 0.29 / n)
        res = arbitrate(
            FusionInput(
                gto=ActionDistribution({Action.BET: 0.62, Action.CHECK: 0.38}),
                best_response=ActionDistribution({Action.BET: 0.88, Action.CHECK: 0.12}),
                deviation=0.16,
                deviation_std=sigma,
                mental_state_probs={MentalState.SOLID: 1.0},
            )
        )
        assert res.lambda_final >= prev - 1e-12
        prev = res.lambda_final
