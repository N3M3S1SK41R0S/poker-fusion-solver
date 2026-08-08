"""Non-régression du chemin exact d'équité — correction ET coût.

L'optimisation clé : la main du héros ne dépend que du runout (ses deux
cartes et le board sont fixes), elle est donc évaluée une fois par runout
— 990 fois au flop — au lieu d'une fois par couple (runout, combo), soit
169 290 fois. Ce test verrouille les deux propriétés :

1. **le résultat ne change pas** — vérifié contre un oracle indépendant,
   recalculé ici combo par combo sans aucune des astuces d'indexation ;
2. **le travail redondant ne revient pas** — on compte les mains réellement
   passées à ``evaluate7`` : au-delà de ~1,3 × le minimum théorique, c'est
   que l'évaluation du héros a été re-tiliée.
"""

from __future__ import annotations

import unittest

import numpy as np

from pfs.core import equity as eqmod
from pfs.core.equity import equity_vs_range, evaluate7
from pfs.core.range_model import RANKS, SUITS, parse_range


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


class TestExactPathCorrectness(unittest.TestCase):
    """Le chemin optimisé doit égaler un oracle naïf, au bit près."""

    @staticmethod
    def _oracle(hero: list[int], spec: str, board: list[int]) -> float:
        """Équité recalculée sans indexation astucieuse : boucle explicite."""
        rng = parse_range(spec)
        dead = set(hero) | set(board)
        remaining = [x for x in range(52) if x not in dead]
        need = 5 - len(board)
        if need == 1:
            runouts = [[x] for x in remaining]
        elif need == 2:
            runouts = [[a, b] for i, a in enumerate(remaining)
                       for b in remaining[i + 1:]]
        else:
            runouts = [[]]

        from pfs.core.range_model import combo_cards
        combos = [(combo_cards(int(i)), float(rng.weights[i]))
                  for i in np.nonzero(rng.weights > 0)[0]]
        combos = [(cc, w) for cc, w in combos
                  if cc[0] not in dead and cc[1] not in dead]

        num = den = 0.0
        for run in runouts:
            for (a, b), w in combos:
                if a in run or b in run:
                    continue
                h = evaluate7(np.array([hero + board + run]))[0]
                v = evaluate7(np.array([[a, b] + board + run]))[0]
                num += w * (1.0 if h > v else 0.5 if h == v else 0.0)
                den += w
        return num / den

    def test_river_matches_oracle(self) -> None:
        hero = [c("Ah"), c("Qd")]
        board = [c("Qs"), c("7d"), c("2c"), c("9h"), c("3s")]
        spec = "TT+, AJs+, KQs, AKo, AQo"
        got = equity_vs_range(hero, parse_range(spec), board).equity
        self.assertAlmostEqual(got, self._oracle(hero, spec, board), places=12)

    def test_turn_matches_oracle(self) -> None:
        hero = [c("Kh"), c("Kc")]
        board = [c("2s"), c("7d"), c("9c"), c("Jh")]
        spec = "AA, QQ, AKs"
        got = equity_vs_range(hero, parse_range(spec), board).equity
        self.assertAlmostEqual(got, self._oracle(hero, spec, board), places=12)

    def test_flop_matches_oracle(self) -> None:
        # deux rues à venir : c'est le chemin le plus lourd, donc le plus
        # susceptible d'une erreur d'indexation entre runouts et combos
        hero = [c("Ah"), c("Ad")]
        board = [c("2s"), c("7d"), c("9c")]
        spec = "KK, QQ"
        got = equity_vs_range(hero, parse_range(spec), board).equity
        self.assertAlmostEqual(got, self._oracle(hero, spec, board), places=12)


class TestNoRedundantEvaluation(unittest.TestCase):
    """Le héros ne doit pas être réévalué une fois par combo adverse."""

    def test_hero_evaluated_once_per_runout(self) -> None:
        seen: list[int] = []
        original = eqmod.evaluate7

        def counting(cards):
            arr = np.asarray(cards)
            seen.append(arr.shape[0] if arr.ndim == 2 else 0)
            return original(cards)

        hero = [c("Ah"), c("Qd")]
        board = [c("Qs"), c("7d"), c("2c")]          # flop : 990 runouts
        rng = parse_range("22+, A8s+, KQs, AKo")
        eqmod.evaluate7 = counting
        try:
            res = equity_vs_range(hero, rng, board)
        finally:
            eqmod.evaluate7 = original

        n_pairs = res.n_scenarios
        total = sum(seen)
        # minimum théorique : 990 (héros, un par runout) + n_pairs (vilain).
        # Sans l'optimisation on paierait 2 × n_pairs, très au-delà du seuil.
        minimum = 990 + n_pairs
        self.assertLessEqual(
            total, int(1.3 * minimum),
            f"{total} mains évaluées pour {n_pairs} scénarios — le héros est "
            "probablement réévalué par combo (régression de l'optimisation).",
        )


if __name__ == "__main__":
    unittest.main()
