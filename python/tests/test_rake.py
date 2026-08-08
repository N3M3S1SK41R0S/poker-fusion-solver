"""Tests L8 — modèle de rake (% + cap) et branchement au solveur postflop.

Quatre étages de preuve :

1. **Modèle** : ``take``/``net`` dans les deux régimes (pourcentage qui
   domine, cap qui mord), préréglages, validation stricte (``RakeError``).
2. **Non-régression bit à bit** : ``rake=NO_RAKE`` ⇒ le solveur rend
   EXACTEMENT (égalité flottante stricte, pas de tolérance) les mêmes
   regrets, stratégies moyennes, EV et exploitabilité que l'appel sans le
   paramètre — ``net(pot) = pot − 0.0`` est l'identité IEEE 754, et le CFR
   du solveur est entièrement déterministe (aucun tirage aléatoire), donc
   « mêmes seeds » = mêmes itérations suffit.
3. **Comptabilité** : à chaque terminal u_OOP + u_IP = pot − rake(pot),
   donc pot − (EV_OOP + EV_IP) = E[rake] ∈ (0, min(cap, pct·pot_max)] ; et
   quand le cap mord à TOUS les terminaux, E[rake] = cap à la précision
   machine, quel que soit le profil (identité comptable, pas un résultat de
   convergence).
4. **Équilibre analytique du jeu raké** — le point subtil. Sur le spot de
   polarisation (AA+33 contre QQ, mise pot), les indifférences donnent en
   forme close, avec net(·) = pot − min(cap, pct·pot) :

       ratio de bluff dans la range de mise   β* = b / net(P+2b)
       fréquence de call du bluff-catcheur    c* = 1 − b / net(P+b)

   (sans rake : β* = b/(P+2b) = 1/3 et c* = 1/2 pour P = b.) Le rake
   DIMINUE net(·), donc à l'équilibre les CALLS BAISSENT (attraper un bluff
   paie moins : la MDF s'effondre) et, par la contrainte d'indifférence du
   bluff-catcheur, les BLUFFS MONTENT légèrement (fréquence de bluff des 33
   x* = β*/(1−β*) : 100/199 ≈ 0.5025 au cap Winamax, 5/7 ≈ 0.714 à 20 %
   sans cap — contre 1/2 sans rake). L'intuition « le rake réduit
   l'incitation à bluffer » est vraie au sens EXPLOITATIF — face à une
   défense FIGÉE à 1/2, bluffer devient strictement −EV (testé sur
   l'arithmétique du modèle) — mais elle S'INVERSE à l'équilibre : le
   solveur doit retrouver les formes closes ci-dessus, pas l'intuition.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from pfs.core.rake import NO_RAKE, WINAMAX_MICRO, RakeError, RakeModel
from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.solver.postflop import IP, PostflopError, PostflopSolver


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def cs(*ts: str) -> list[int]:
    return [c(t) for t in ts]


RIVER_DRY = cs("2s", "2d", "7h", "8h", "Kc")

HEAVY = RakeModel(0.2, 1000.0)      # 20 % sans cap effectif : effets visibles

# ── solves partagés (déterministes) sur le spot de polarisation ───────────
# AA+33 contre QQ, pot 100, mise pot unique — mêmes réglages que
# tests/test_postflop.py::TestPolarizationNash (800 itérations).

_POLAR_CACHE: dict[str, PostflopSolver] = {}


def polar_solver(key: str, rake: RakeModel, iterations: int = 800) -> PostflopSolver:
    """Résout (et mémoïse) le spot de polarisation pour un modèle de rake."""
    if key not in _POLAR_CACHE:
        s = PostflopSolver(RIVER_DRY, parse_range("QQ"), parse_range("AA,33"),
                           pot=100, stack=100, bet_fracs=(1.0,), max_bets=1,
                           rake=rake)
        _POLAR_CACHE[key] = s.solve(iterations)
    return _POLAR_CACHE[key]


def bluff_value_call_freqs(solver: PostflopSolver) -> tuple[float, float, float]:
    """(fréq. de bluff des 33, fréq. de mise des AA, fréq. de call de QQ)."""
    root = solver._nodes[solver._root]
    ip_idx = root.children[root.labels.index("check")]
    node = solver._nodes[ip_idx]
    sigma = solver.average_strategy(ip_idx)
    bet_a = next(a for a, lab in enumerate(node.labels) if lab != "check")
    cards = solver.players[IP].cards
    is_air = ((cards >> 2) == RANKS.index("3")).all(axis=1)
    is_nuts = ((cards >> 2) == RANKS.index("A")).all(axis=1)
    oop_idx = node.children[bet_a]
    oop_node = solver._nodes[oop_idx]
    sig_oop = solver.average_strategy(oop_idx)
    call_a = oop_node.labels.index("call")
    return (float(sigma[bet_a][is_air].mean()),
            float(sigma[bet_a][is_nuts].mean()),
            float(sig_oop[call_a].mean()))


# ═══════════════════════════════════════════════════════════════════════════
# 1. LE MODÈLE
# ═══════════════════════════════════════════════════════════════════════════


class TestRakeModel(unittest.TestCase):
    def test_take_and_net_percentage_regime(self) -> None:
        """Sous le point de bascule cap/pct, le pourcentage domine."""
        r = RakeModel(0.05, 10.0)
        self.assertAlmostEqual(r.take(100.0), 5.0, places=12)
        self.assertAlmostEqual(r.net(100.0), 95.0, places=12)
        self.assertAlmostEqual(r.take(0.0), 0.0, places=12)

    def test_cap_bites_on_big_pots(self) -> None:
        """pct × pot > cap ⇒ le prélèvement est plafonné au cap."""
        r = RakeModel(0.05, 1.0)
        self.assertGreater(0.05 * 100.0, r.cap)         # le cap mord bien
        self.assertEqual(r.take(100.0), 1.0)
        self.assertEqual(r.net(100.0), 99.0)
        self.assertEqual(r.take(20.0), 1.0)             # point de bascule exact

    def test_no_rake_identity_bitwise(self) -> None:
        """NO_RAKE : net = identité BIT À BIT (x − 0.0 est exact en IEEE)."""
        for pot in (0.0, 1.0, 37.25, 100.0, 1e-9, 12345.6789):
            self.assertEqual(NO_RAKE.take(pot), 0.0)
            self.assertEqual(NO_RAKE.net(pot), pot)

    def test_presets(self) -> None:
        self.assertEqual((NO_RAKE.pct, NO_RAKE.cap), (0.0, 0.0))
        self.assertEqual((WINAMAX_MICRO.pct, WINAMAX_MICRO.cap), (0.05, 1.0))

    def test_uncapped_and_zero_cap_edges(self) -> None:
        """cap = inf désactive le plafond ; cap = 0 annule tout rake."""
        self.assertAlmostEqual(RakeModel(0.05, math.inf).take(1e6), 5e4)
        self.assertEqual(RakeModel(0.2, 0.0).take(1e6), 0.0)

    def test_validation_rejects_bad_parameters(self) -> None:
        for pct, cap in [(-0.01, 1.0), (0.2000001, 1.0), (5.0, 1.0),
                         (0.05, -1.0), (math.nan, 1.0), (0.05, math.nan)]:
            with self.assertRaises(RakeError):
                RakeModel(pct, cap)

    def test_take_rejects_negative_pot(self) -> None:
        with self.assertRaises(RakeError):
            WINAMAX_MICRO.take(-1.0)

    def test_rake_error_is_value_error(self) -> None:
        self.assertTrue(issubclass(RakeError, ValueError))


# ═══════════════════════════════════════════════════════════════════════════
# 2. NON-RÉGRESSION BIT À BIT (rake = NO_RAKE)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoRakeNonRegression(unittest.TestCase):
    """Deux solveurs, mêmes itérations : sans le paramètre vs rake=NO_RAKE.

    Le CFR du solveur est déterministe ⇒ toute divergence, même d'un ulp,
    signalerait que le branchement du rake a altéré le chemin de calcul.
    """

    @classmethod
    def setUpClass(cls) -> None:
        args = (RIVER_DRY, parse_range("QQ"), parse_range("AA,33"))
        kw = dict(pot=100, stack=100, bet_fracs=(1.0,), max_bets=1)
        cls.before = PostflopSolver(*args, **kw).solve(150)
        cls.after = PostflopSolver(*args, **kw, rake=NO_RAKE).solve(150)

    def test_values_identical_bitwise(self) -> None:
        self.assertEqual(self.before.values(), self.after.values())

    def test_exploitability_identical_bitwise(self) -> None:
        self.assertEqual(self.before.exploitability(),
                         self.after.exploitability())

    def test_internal_state_identical_bitwise(self) -> None:
        """Regrets et stratégies cumulées : égalité stricte nœud par nœud."""
        self.assertEqual(len(self.before._nodes), len(self.after._nodes))
        for nb, na in zip(self.before._nodes, self.after._nodes):
            np.testing.assert_array_equal(nb.regrets, na.regrets)
            np.testing.assert_array_equal(nb.strat_sum, na.strat_sum)

    def test_root_report_identical(self) -> None:
        self.assertEqual(self.before.root_report(), self.after.root_report())

    def test_expected_rake_machine_zero(self) -> None:
        self.assertLess(abs(self.after.expected_rake()), 1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# 3. COMPTABILITÉ DU JEU RAKÉ
# ═══════════════════════════════════════════════════════════════════════════


class TestRakedAccounting(unittest.TestCase):
    """Sur le spot de polarisation, tout pot terminal ∈ {100, 200, 300} et
    5 % × 100 = 5 > cap = 1 : le cap Winamax mord à CHAQUE terminal, donc
    E[rake] = cap = 1.0 exactement, indépendamment des stratégies."""

    def test_sum_of_evs_below_pot(self) -> None:
        s = polar_solver("winamax", WINAMAX_MICRO)
        ev_oop, ev_ip = s.values()
        self.assertLess(ev_oop + ev_ip, s.pot0)

    def test_expected_rake_bounds_and_golden_value(self) -> None:
        s = polar_solver("winamax", WINAMAX_MICRO)
        er = s.expected_rake()
        pot_max = s.pot0 + 2.0 * s.stack            # 300 : les deux à tapis
        self.assertGreater(er, 0.0)
        self.assertLessEqual(er, min(WINAMAX_MICRO.cap,
                                     WINAMAX_MICRO.pct * pot_max) + 1e-9)
        self.assertAlmostEqual(er, 1.0, places=6)   # cap mordu partout

    def test_more_rake_means_less_total_ev(self) -> None:
        """Monotonie : rake pointwise plus lourd ⇒ somme des EV plus faible.

        Modèles à cap mordant partout ⇒ sommes EXACTES (100, 99, 97) à
        n'importe quel nombre d'itérations — identité comptable.
        """
        sums = []
        for rake in (NO_RAKE, RakeModel(0.05, 1.0), RakeModel(0.10, 3.0)):
            s = PostflopSolver(RIVER_DRY, parse_range("QQ"),
                               parse_range("AA,33"), pot=100, stack=100,
                               bet_fracs=(1.0,), max_bets=1, rake=rake)
            s.solve(120)
            sums.append(sum(s.values()))
        self.assertAlmostEqual(sums[0], 100.0, places=8)
        self.assertAlmostEqual(sums[1], 99.0, places=8)
        self.assertAlmostEqual(sums[2], 97.0, places=8)
        self.assertLess(sums[1], sums[0])
        self.assertLess(sums[2], sums[1])

    def test_full_river_spot_rake_between_zero_and_cap(self) -> None:
        """Solve complet (ranges larges, deux tailles, relance permise) :
        pot − (EV_OOP + EV_IP) ∈ (0, cap]."""
        s = PostflopSolver(RIVER_DRY, parse_range("22+, ATs+, KQs"),
                           parse_range("77+, AJs+, AQo+"),
                           pot=80, stack=300, bet_fracs=(0.5, 1.25),
                           max_bets=2, rake=WINAMAX_MICRO)
        s.solve(120)
        er = s.expected_rake()
        self.assertGreater(er, 0.0)
        self.assertLessEqual(er, WINAMAX_MICRO.cap + 1e-9)
        self.assertAlmostEqual(er, 1.0, places=6)   # 5 % de 80 = 4 > cap

    def test_turn_chance_node_accounting(self) -> None:
        """Le rake traverse le nœud de chance : prélevé UNE fois, au terminal
        de la rivière tirée — jamais au nœud de chance lui-même."""
        s = PostflopSolver(cs("2s", "2d", "7h", "8h"),
                           parse_range("QQ, 99"), parse_range("KK, 55"),
                           pot=60, stack=180, bet_fracs=(0.75,), max_bets=1,
                           rake=WINAMAX_MICRO)
        s.solve(60)
        ev_oop, ev_ip = s.values()
        self.assertAlmostEqual(s.expected_rake(), 1.0, places=6)
        self.assertAlmostEqual(ev_oop + ev_ip, 59.0, places=7)

    def test_corrected_exploitability_target(self) -> None:
        """Cible décalée : expl → −E[rake]/(2·pot) à l'équilibre, donc la
        mesure corrigée expl + E[rake]/(2·pot) est ≥ 0 et → 0."""
        for key, rake in (("winamax", WINAMAX_MICRO), ("heavy", HEAVY)):
            s = polar_solver(key, rake)
            corrected = s.exploitability() + s.expected_rake() / (2.0 * s.pot0)
            self.assertGreaterEqual(corrected, -1e-9)
            self.assertLess(corrected, 0.01)

    def test_rake_argument_validated(self) -> None:
        with self.assertRaises(PostflopError):
            PostflopSolver(RIVER_DRY, parse_range("AA"), parse_range("KK"),
                           pot=10, stack=10, rake=0.05)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 4. ÉQUILIBRE DU JEU RAKÉ (formes closes retrouvées par le solveur)
# ═══════════════════════════════════════════════════════════════════════════


class TestRakedPolarizationEquilibrium(unittest.TestCase):
    """Voir la docstring de module : β* = b/net(P+2b), c* = 1 − b/net(P+b).

    Avec P = b = 100 :
    - Winamax (net ≡ pot − 1) : x* = 100/199 ≈ 0.5025, c* = 99/199 ≈ 0.4975 ;
    - 20 % sans cap (net ≡ 0.8·pot) : x* = 5/7 ≈ 0.7143, c* = 0.375,
      EV_IP = 45, EV_OOP = 80/7 ≈ 11.43, E[rake] = 1220/28 ≈ 43.57.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.f_none = bluff_value_call_freqs(polar_solver("none", NO_RAKE))
        cls.f_wina = bluff_value_call_freqs(polar_solver("winamax", WINAMAX_MICRO))
        cls.f_heavy = bluff_value_call_freqs(polar_solver("heavy", HEAVY))

    def test_exploitative_bluff_incentive_drops(self) -> None:
        """Le sens VRAI de « le rake réduit l'incitation à bluffer » : face à
        une défense figée au call 1/2 (l'équilibre non raké), l'EV d'un bluff
        pot-size passe de 0 (indifférence classique) à strictement < 0."""
        def ev_bluff_vs_half(rake: RakeModel) -> float:
            # gagne net(P+b) − b si fold (1/2), perd b si call (1/2)
            return 0.5 * (rake.net(200.0) - 100.0) - 0.5 * 100.0
        self.assertAlmostEqual(ev_bluff_vs_half(NO_RAKE), 0.0, places=12)
        self.assertLess(ev_bluff_vs_half(WINAMAX_MICRO), 0.0)   # −0.5
        self.assertLess(ev_bluff_vs_half(HEAVY), 0.0)           # −20.0

    def test_equilibrium_bluff_ratio_closed_form(self) -> None:
        """À l'ÉQUILIBRE, les bluffs MONTENT : x* = β*/(1−β*) avec
        β* = b/net(P+2b) > b/(P+2b). Assertion directionnelle large sur le
        rake lourd (écart ≈ +0.21, très au-dessus du bruit CFR) ; au cap
        Winamax l'écart (+0.0025) est confirmé contre la forme close."""
        bluff_none, _, _ = self.f_none
        bluff_wina, _, _ = self.f_wina
        bluff_heavy, _, _ = self.f_heavy
        self.assertAlmostEqual(bluff_none, 0.5, delta=0.05)
        self.assertAlmostEqual(bluff_wina, 100.0 / 199.0, delta=0.03)
        self.assertAlmostEqual(bluff_heavy, 5.0 / 7.0, delta=0.05)
        self.assertGreater(bluff_heavy, bluff_none + 0.1)

    def test_equilibrium_calls_drop(self) -> None:
        """La MDF, elle, S'EFFONDRE avec le rake : c* = 1 − b/net(P+b) —
        c'est le défenseur qui paie le rake à l'équilibre."""
        _, _, call_none = self.f_none
        _, _, call_wina = self.f_wina
        _, _, call_heavy = self.f_heavy
        self.assertAlmostEqual(call_none, 0.5, delta=0.05)
        self.assertAlmostEqual(call_wina, 99.0 / 199.0, delta=0.03)
        self.assertAlmostEqual(call_heavy, 0.375, delta=0.05)
        self.assertLess(call_heavy, call_none - 0.05)

    def test_value_bets_stay_pure(self) -> None:
        """Les nuts misent toujours, raké ou non (dominance stricte)."""
        for _, value_freq, _ in (self.f_none, self.f_wina, self.f_heavy):
            self.assertGreater(value_freq, 0.97)

    def test_heavy_rake_evs_match_analytic(self) -> None:
        s = polar_solver("heavy", HEAVY)
        ev_oop, ev_ip = s.values()
        self.assertAlmostEqual(ev_ip, 45.0, delta=0.6)
        self.assertAlmostEqual(ev_oop, 80.0 / 7.0, delta=0.6)
        self.assertAlmostEqual(s.expected_rake(), 1220.0 / 28.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
