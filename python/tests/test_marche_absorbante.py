r"""Tests du banc de la marche absorbante — le juge externe du biais de Harville.

Discipline des goldens, à lire avant de toucher une valeur
----------------------------------------------------------
Les valeurs de la marche figées ici viennent d'un run semé
(``numpy.random.default_rng``, flux PCG64 stable) : mêmes paramètres, mêmes
chiffres au bit près. La tolérance de chaque golden vaut **3 erreurs-types
CALCULÉES sur le run courant** (:math:`3\sqrt{\hat p(1-\hat p)/n}`), jamais un
nombre choisi à la main :

* tant que le simulateur n'est pas modifié, l'écart est exactement nul ;
* si un refactor réordonne les tirages, le nouveau run est un tirage
  indépendant : l'écart des deux estimations a pour écart-type
  :math:`\sqrt{2}\,\sigma`, et 3σ le couvre à 2,1 écarts combinés ;
* si le test casse au-delà, c'est que la LOI de la marche a changé — et
  c'est précisément ce qu'il doit signaler. On ne l'élargit pas : on
  cherche ce qui a changé la loi.

Les références Harville, elles, sont EXACTES et refaites à la main dans les
docstrings — pas relues depuis le code jugé.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import banc_marche_absorbante as banc  # noqa: E402


class TestRaccourciDeuxJoueurs(unittest.TestCase):
    """À deux joueurs la marche est la ruine du joueur : loi exacte, zéro pas."""

    def test_ruine_du_joueur_exacte(self) -> None:
        r"""``[2, 1]`` : martingale bornée absorbée en 0 ou 3, donc
        :math:`3p = 2 \Rightarrow p = 2/3` — calcul à la main, aucune
        simulation nécessaire. Le banc doit le savoir : ``pas == 0``.

        Tolérance : :math:`3\sqrt{(2/9)/30000} = 0{,}00817`.
        """
        res = banc.simuler_rangs((2, 1), n_sims=30_000, seed=0)
        self.assertEqual(res.pas, 0)
        self.assertLess(abs(float(res.probs[0, 0]) - 2.0 / 3.0), 0.00817)

    def test_entrees_refusees(self) -> None:
        """Tapis non entiers, non positifs, joueur seul, politique inconnue."""
        with self.assertRaises(banc.MarcheError):
            banc.simuler_rangs((10.5, 5), n_sims=10)
        with self.assertRaises(banc.MarcheError):
            banc.simuler_rangs((10, 0), n_sims=10)
        with self.assertRaises(banc.MarcheError):
            banc.simuler_rangs((10,), n_sims=10)
        with self.assertRaises(banc.MarcheError):
            banc.simuler_rangs((10, 5), n_sims=10, politique="martingale")


class TestGoldenUnitaire(unittest.TestCase):
    r"""Le run de référence : ``[50, 30, 20]``, politique unitaire.

    Références Harville refaites à la main (récurrence
    :math:`P(i\,2^e \mid j\,1^{er}) = s_i/(S - s_j)`) :

    * P(leader 2ᵉ) = 0,3·(50/70) + 0,2·(50/80) = 3/14 + 1/8 = **19/56**
      = 0,339286 ;
    * P(petit 2ᵉ)  = 0,5·(20/50) + 0,3·(20/70) = 1/5 + 3/35 = 10/35 = **2/7**
      = 0,285714.

    Goldens de la marche (graine 7, 40 000 tournois, figés le 14 août 2026) :
    P(leader 2ᵉ) = 0,363475 · P(petit 2ᵉ) = 0,263425. Le biais de Harville —
    −2,42 pt sur la première case, +2,23 pt sur la seconde, z ≈ 10 chacune —
    n'est donc pas un bruit : à 3 erreurs-types, les deux modèles sont
    incompatibles, dans des directions opposées selon le tapis.
    """

    GOLDEN_LEADER_2E = 0.363475
    GOLDEN_PETIT_2E = 0.263425
    HARVILLE_LEADER_2E = 19.0 / 56.0
    HARVILLE_PETIT_2E = 2.0 / 7.0

    @classmethod
    def setUpClass(cls) -> None:
        cls.b = banc.biais_harville((50, 30, 20), n_sims=40_000, seed=7)

    def test_les_goldens_de_la_marche(self) -> None:
        """Tolérance : 3 erreurs-types du run courant, pas un chiffre choisi."""
        p, se = self.b.marche.probs, self.b.marche.se
        self.assertLess(abs(float(p[0, 1]) - self.GOLDEN_LEADER_2E),
                        3.0 * float(se[0, 1]))
        self.assertLess(abs(float(p[2, 1]) - self.GOLDEN_PETIT_2E),
                        3.0 * float(se[2, 1]))

    def test_harville_recalcule_a_la_main(self) -> None:
        """La référence du banc doit être celle de la récurrence, exactement."""
        self.assertAlmostEqual(float(self.b.harville[0, 1]),
                               self.HARVILLE_LEADER_2E, places=12)
        self.assertAlmostEqual(float(self.b.harville[2, 1]),
                               self.HARVILLE_PETIT_2E, places=12)

    def test_le_biais_est_significatif_et_dans_le_bon_sens(self) -> None:
        """Harville SOUS-estime « leader 2ᵉ » et SUR-estime « petit 2ᵉ »,
        chacun au-delà de 3 erreurs-types. Si l'un des signes s'inversait, la
        conclusion du banc serait fausse et ce test doit crier."""
        self.assertLess(float(self.b.biais[0, 1]),
                        -3.0 * float(self.b.marche.se[0, 1]))
        self.assertGreater(float(self.b.biais[2, 1]),
                           3.0 * float(self.b.marche.se[2, 1]))

    def test_ancre_martingale(self) -> None:
        """P(rang 1) = part de jetons, pour la marche ET pour Harville :
        le biais du rang 1 est structurellement nul (arrêt optionnel d'une
        martingale bornée). C'est le témoin de bon fonctionnement du
        simulateur, à 3 erreurs-types près."""
        parts = np.array([0.5, 0.3, 0.2])
        ecarts = np.abs(self.b.marche.probs[:, 0] - parts)
        self.assertTrue(bool((ecarts <= 3.0 * self.b.marche.se[:, 0]).all()),
                        f"écarts {ecarts}, se {self.b.marche.se[:, 0]}")
        np.testing.assert_allclose(self.b.harville[:, 0], parts, atol=1e-12)

    def test_ancre_bistochastique(self) -> None:
        """Chaque joueur a un rang, chaque rang a un joueur — par construction,
        donc à l'exactitude flottante et pas à 3 SE."""
        np.testing.assert_allclose(self.b.marche.probs.sum(axis=1), 1.0,
                                   atol=1e-9)
        np.testing.assert_allclose(self.b.marche.probs.sum(axis=0), 1.0,
                                   atol=1e-9)


class TestGoldenAllin(unittest.TestCase):
    r"""L'autre politique : le biais dépend de la dynamique supposée.

    Référence Harville refaite à la main — P(le tapis de 10 finit dernier)
    sur ``[40, 30, 20, 10]`` : somme sur les six ordres d'arrivée des trois
    autres, :math:`P(a,b,c) = \frac{s_a}{100}\cdot\frac{s_b}{100-s_a}\cdot
    \frac{s_c}{100-s_a-s_b}` :

    ========  ==========================  =========
    ordre     produit                     valeur
    ========  ==========================  =========
    40,30,20  0,4 · 30/60 · 20/30         0,133333
    40,20,30  0,4 · 20/60 · 30/40         0,100000
    30,40,20  0,3 · 40/70 · 20/30         0,114286
    30,20,40  0,3 · 20/70 · 40/50         0,068571
    20,40,30  0,2 · 40/80 · 30/40         0,075000
    20,30,40  0,2 · 30/80 · 40/50         0,060000
    ========  ==========================  =========

    Total : **0,551190**. Golden de la marche allin (graine 11, 200 000
    tournois, figé le 14 août 2026) : 0,426445 — Harville SURESTIME de
    12,5 points la probabilité que le micro-tapis finisse dernier quand la
    dynamique est une loterie de tapis (z = 113), alors qu'il la SOUS-estime
    de 5,4 points sous la dynamique diffusive. Le « biais de Harville » est
    une fonction de la dynamique, pas un scalaire — c'est ce que cette classe
    épingle face à :class:`TestGoldenUnitaire`.
    """

    GOLDEN_PETIT_DERNIER = 0.426445
    HARVILLE_PETIT_DERNIER = 0.5511904761904762

    @classmethod
    def setUpClass(cls) -> None:
        cls.b = banc.biais_harville((40, 30, 20, 10), n_sims=200_000, seed=11,
                                    politique="allin")

    def test_le_golden_de_la_marche(self) -> None:
        p, se = self.b.marche.probs, self.b.marche.se
        self.assertLess(abs(float(p[3, 3]) - self.GOLDEN_PETIT_DERNIER),
                        3.0 * float(se[3, 3]))

    def test_harville_recalcule_a_la_main(self) -> None:
        self.assertAlmostEqual(float(self.b.harville[3, 3]),
                               self.HARVILLE_PETIT_DERNIER, places=12)

    def test_le_signe_du_biais_s_inverse_avec_la_dynamique(self) -> None:
        """Sous allin, Harville SURESTIME la dernière place du micro-tapis
        (sous unitaire il la sous-estime : voir le banc). Les deux sens sont
        significatifs — c'est la preuve mesurée que le biais n'est défini que
        relativement à une politique de pas."""
        self.assertGreater(float(self.b.biais[3, 3]),
                           3.0 * float(self.b.marche.se[3, 3]))


if __name__ == "__main__":
    unittest.main()
