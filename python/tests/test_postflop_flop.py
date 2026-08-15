"""Tests Phase 2, étage 1 — le solveur postflop s'étend au FLOP.

La comptabilité d'abord (piège n°1 de PASSATION.md, qui a déjà mordu au
turn) : depuis un flop, le nœud de chance du turn divise par **45**
(52 − 3 board − 2×2 mains), puis chaque nœud de chance de la river par
**44** (52 − 4 − 2×2) — soit 45×44 runouts ordonnés et 45×44/2 = **990
confrontations turn+river** par feuille flop. Tout autre dénominateur casse
« somme des EV = pot », et ce théorème vaut pour TOUT profil de stratégies
(chaque terminal porte u_OOP + u_IP = pot ; un nœud de chance en prend la
moyenne exacte par confrontation compatible) : on le teste donc AU PROFIL
UNIFORME — zéro itération, aucun équilibre requis — puis après DCFR.

Trois goldens à la main (mêmes conventions que les tests river existants) :
1. **Dénombrement 990** : AA contre KK sur 2s2d7h, à tapis derrière — 83
   des 990 runouts font gagner KK (calcul complet dans la docstring du
   test), vérifié à la précision machine : c'est le test DIRECT du
   dénombrement « 45×44/2, cartes mortes 3+2+2 ».
2. **Carré contre paire** : AhAc (carré d'as au flop AsAd8h) est
   invincible sur tout runout → l'EV d'équilibre est exactement le pot.
3. **Miroir** : QQ contre QQ sur board sans couleur ni quinte possibles —
   tout abattage est une égalité, payer domine folder, EV = pot/2 chacun.
"""

from __future__ import annotations

import unittest

import numpy as np

from pfs.core.range_model import RANKS, SUITS, parse_range
from pfs.solver.postflop import (
    IP,
    OOP,
    PostflopError,
    PostflopSolver,
    _Chance,
)


def c(t: str) -> int:
    return RANKS.index(t[0]) * 4 + SUITS.index(t[1])


def cs(*ts: str) -> list[int]:
    return [c(t) for t in ts]


FLOP_DRY = cs("2s", "2d", "7h")        # préfixe de TURN_DRY / RIVER_DRY


class TestComptabiliteFlopPleineProfondeur(unittest.TestCase):
    """Somme des EV = pot sur l'arbre flop COMPLET (flop→turn→river).

    Le spot : QQ,99 contre KK,55 (12 combos chacun), l'IP peut miser 0.75p
    une fois par street — un arbre de ~28 500 nœuds portant 49 tirages turn
    × 48 tirages river. La somme est vérifiée au profil UNIFORME (aucune
    itération : c'est un théorème de comptabilité, pas de convergence),
    puis après 16 itérations DCFR ; l'exploitabilité (par ensemble
    d'information, piège n°3) doit décroître entre 4 et 16 itérations.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = PostflopSolver(FLOP_DRY, parse_range("QQ, 99"),
                                    parse_range("KK, 55"),
                                    pot=60, stack=180,
                                    oop_bet_fracs=(), ip_bet_fracs=(0.75,),
                                    max_bets=1)
        cls.ev_uniforme = cls.solver.values()      # profil uniforme exact
        cls.solver.solve(4)
        cls.e1 = cls.solver.exploitability()
        cls.solver.solve(12)
        cls.e2 = cls.solver.exploitability()
        cls.ev = cls.solver.values()

    def test_somme_uniforme_egale_pot(self) -> None:
        """Le théorème vaut pour TOUT profil — donc déjà à zéro itération.

        C'est le test qui mord si un diviseur de chance est faux (49 ou 44
        au tirage du turn au lieu de 45, 48 ou 45 à la river au lieu de 44).
        """
        self.assertAlmostEqual(sum(self.ev_uniforme), 60.0, places=6)

    def test_somme_apres_dcfr_egale_pot(self) -> None:
        self.assertAlmostEqual(sum(self.ev), 60.0, places=6)

    def test_chances_emboitees_45_puis_44(self) -> None:
        """Structure : 49 turns divisés par 45, puis 48 rivières par 44.

        Le diviseur est le nombre de tirages VALIDES par confrontation
        (cartes mortes : board effectif + 2×2 mains), pas le nombre de
        cartes énumérées — exactement le piège n°1, en version emboîtée.
        """
        formes = {(len(t.deals), t.n_unknown)
                  for nd in self.solver._nodes for t in nd.terminal
                  if isinstance(t, _Chance)}
        self.assertEqual(formes, {(49, 45), (48, 44)})

    def test_exploitabilite_decroit(self) -> None:
        self.assertLess(self.e2, self.e1)
        self.assertLess(self.e2, 0.12)             # mesuré : 0.074 à 16 it.

    def test_exploitabilite_non_negative(self) -> None:
        self.assertGreaterEqual(self.e2, -1e-9)


class TestGoldensFlopALaMain(unittest.TestCase):
    """Les trois goldens calculables à la main depuis un flop."""

    def test_denombrement_990_equite_checked_down(self) -> None:
        """AA contre KK sur 2s2d7h, à tapis : KK gagne 83 runouts sur 990.

        Calcul à la main — cartes mortes 3 (board) + 2 (AA) + 2 (KK) = 7,
        donc 45 cartes inconnues et C(45,2) = 990 runouts turn+river par
        confrontation (le « 45×44/2 » de la passation) :

        * KK ne dépasse AA que par un brelan/carré de rois : il faut AU
          MOINS un des 2 rois restants, sans as (un as redonne à AA le
          brelan supérieur). Paires contenant ≥ 1 roi :
          C(45,2) − C(43,2) = 990 − 903 = 87 ; dont avec un as : 2×2 = 4 ;
          restent 87 − 4 = **83**.
        * Rien d'autre ne sauve KK sur ce board : aucune couleur possible
          (3 cartes du board sont de couleurs distinctes, 2 runouts → au
          plus 4 d'une couleur avec la main) ; aucune quinte différentielle
          (il manque toujours ≥ 3 cartes à toute quinte de A ou de K) ; les
          doubles paires, fulls et carrés DU BOARD gardent le kicker/la
          paire d'AA au-dessus. Aucune égalité possible.

        Donc équité checked-down de CHAQUE combo KK = 83/990, et de chaque
        AA = 907/990 — vérifié à la précision machine via la feuille
        rollout (part d'abattage / masse compatible), qui emprunte
        exactement la normalisation des nœuds de chance.
        """
        s = PostflopSolver(FLOP_DRY, parse_range("AA"), parse_range("KK"),
                           pot=100, stack=100, oop_bet_fracs=(1.0,),
                           ip_bet_fracs=(), max_bets=1,
                           leaf_model="rollout")
        for place, attendu in ((IP, 83.0 / 990.0), (OOP, 907.0 / 990.0)):
            q = s.players[1 - place].weights
            share = s._leaf_share(place, q, ())
            compat = s._compat_mass(place, q, ())
            np.testing.assert_allclose(share / compat, attendu, rtol=1e-9)

    def test_quads_prennent_le_pot_pleine_profondeur(self) -> None:
        """AhAc (carré d'as sur AsAd8h) contre KK : EV d'équilibre = pot.

        La range OOP « AA » perd AsXX et AdXX au blocage du board : reste
        AhAc, carré d'as dès le flop. Aucun runout ne le bat : la seule
        main supérieure serait une quinte flush, or KK ne détient que des
        rois et le board ne peut aligner ni 5 cartes assorties ni 4 cartes
        consécutives utiles. À l'équilibre, KK (information commune : la
        range adverse est 100 % nuts) ne met plus un jeton : EV_OOP = pot
        = 100 exactement. La moyenne DCFR converge PAR AU-DESSUS (les
        calls résiduels de KK paient OOP) : mesuré 108,5 à 10 itérations,
        103,7 à 20, 102,3 à 30 — on borne au-dessus de 99,5 (jamais moins
        que le pot) et en dessous de 112 à 10 itérations.
        """
        s = PostflopSolver(cs("As", "Ad", "8h"), parse_range("AA"),
                           parse_range("KK"), pot=100, stack=200,
                           oop_bet_fracs=(1.0,), ip_bet_fracs=(),
                           max_bets=1)
        s.solve(10)
        ev0, ev1 = s.values()
        self.assertGreater(ev0, 99.5)
        self.assertLess(ev0, 112.0)
        self.assertAlmostEqual(ev0 + ev1, 100.0, places=6)

    def test_quads_rollout_serre(self) -> None:
        """Même spot, tronqué au turn : la feuille rollout « check-down »
        donne aussi tout à un carré — la limite EV = pot se vérifie serrée
        (300 itérations, mesuré 100,11)."""
        s = PostflopSolver(cs("As", "Ad", "8h"), parse_range("AA"),
                           parse_range("KK"), pot=100, stack=200,
                           oop_bet_fracs=(1.0,), ip_bet_fracs=(),
                           max_bets=1, leaf_model="rollout")
        s.solve(300)
        ev0, ev1 = s.values()
        self.assertAlmostEqual(ev0, 100.0, delta=1.0)
        self.assertAlmostEqual(ev0 + ev1, 100.0, places=6)

    def test_miroir_partage_le_pot(self) -> None:
        """QQ contre QQ sur 2s2d7h : EV = pot/2 chacun, exactement.

        Sur ce board aucune couleur (arc-en-ciel + 2 runouts → au plus 4
        d'une couleur) ni quinte (toute quinte de Q exige 3 cartes utiles
        de runout) ne distingue deux combos QQ : TOUT abattage est une
        égalité. Payer rapporte alors toujours la moitié du pot final —
        strictement mieux que folder (0) : personne ne folde à l'équilibre,
        et chaque joueur encaisse pot/2 = 40. Le spot force aussi le
        card-removal du combo STRICTEMENT identique à travers le nœud de
        chance (ranges identiques — le cas qui casse les implémentations
        naïves, cf. test river éponyme).
        """
        s = PostflopSolver(FLOP_DRY, parse_range("QQ"), parse_range("QQ"),
                           pot=80, stack=200, bet_fracs=(1.0,), max_bets=1,
                           leaf_model="rollout")
        s.solve(200)
        ev0, ev1 = s.values()
        self.assertAlmostEqual(ev0, 40.0, delta=1.0)
        self.assertAlmostEqual(ev1, 40.0, delta=1.0)
        self.assertAlmostEqual(ev0 + ev1, 80.0, places=6)


class TestProfondeurLimiteeFlop(unittest.TestCase):
    """P3 au flop : arbre flop→turn joué, feuille à la place du round river."""

    OOP_R, IP_R = "QQ, 99", "KK, 55"

    def _base(self, **kw) -> PostflopSolver:
        return PostflopSolver(FLOP_DRY, parse_range(self.OOP_R),
                              parse_range(self.IP_R), pot=60, stack=180,
                              bet_fracs=(0.75,), max_bets=1, **kw)

    def test_effondrement_des_noeuds(self) -> None:
        full = self._base()
        roll = self._base(leaf_model="rollout")
        self.assertLess(len(roll._nodes), len(full._nodes) / 20)

    def test_rollout_somme_exacte(self) -> None:
        s = self._base(leaf_model="rollout")
        s.solve(150)
        ev0, ev1 = s.values()
        self.assertAlmostEqual(ev0 + ev1, 60.0, places=6)
        self.assertLess(s.exploitability(), 0.03)  # mesuré : 0.0084

    def test_eqr_directionnel(self) -> None:
        """Même contrat honnête qu'au turn : l'EQR relève l'EV du joueur en
        position par rapport au rollout nu, la somme des EV peut dériver
        (prix de la valeur apprise) mais reste bornée."""
        from pfs.fusion.eqr import train_eqr
        model = train_eqr(n_spots=8, iterations=80, seed=0)
        roll = self._base(leaf_model="rollout")
        roll.solve(150)
        eqr = self._base(leaf_model="eqr", eqr_model=model)
        eqr.solve(150)
        r0, r1 = roll.values()
        e0, e1 = eqr.values()
        self.assertGreater(e1, r1)                 # direction : position ↑
        self.assertLess(abs(e0 + e1 - 60.0), 0.5 * 60.0)


class TestValidationFlop(unittest.TestCase):
    def test_bornes_du_board(self) -> None:
        r = parse_range("AA")
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("2s", "2d"), r, r, pot=10, stack=10)
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("2s", "2d", "7h", "8h", "Kc", "3c"), r, r,
                           pot=10, stack=10)

    def test_rollout_reste_refuse_sur_river(self) -> None:
        r = parse_range("AA")
        with self.assertRaises(PostflopError):
            PostflopSolver(cs("2s", "2d", "7h", "8h", "Kc"), r, r,
                           pot=10, stack=10, leaf_model="rollout")

    def test_route_api_flop_en_profondeur_limitee(self) -> None:
        """La route /api/postflop accepte un board de 3 cartes en profondeur
        LIMITÉE (rollout/eqr) et refuse — en le chiffrant — la profondeur
        complète, hors budget d'une route synchrone jusqu'au blueprint."""
        from pfs.app.server import API
        rep = API.postflop({
            "board": "2s 2d 7h", "oop_range": "QQ, 99", "ip_range": "KK, 55",
            "pot": 60, "stack": 180, "bet_fracs": [0.75], "max_bets": 1,
            "iterations": 40, "leaf_model": "rollout",
        })
        self.assertEqual(rep["street"], "flop")
        self.assertAlmostEqual(rep["ev_oop"] + rep["ev_ip"], 60.0, places=6)
        with self.assertRaises(ValueError):
            API.postflop({
                "board": "2s 2d 7h", "oop_range": "QQ", "ip_range": "KK",
                "pot": 60, "stack": 180, "iterations": 10,
            })


if __name__ == "__main__":
    unittest.main()
