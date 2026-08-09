"""Tests de la validation prédictive de l'inférence adverse.

Le corpus est SYNTHÉTIQUE et construit pour que la bonne réponse soit connue
d'avance — c'est la seule façon de savoir si la métrique mesure ce qu'elle
prétend :

* un joueur volontairement très large contre un très serré → l'écart DOIT
  être détecté, et significativement ;
* une population homogène → il ne DOIT rien être détecté (contrôle négatif) ;
* un écart purement de TABLE → la permutation globale le voit, la permutation
  stratifiée ne le voit pas : c'est exactement ce qui distingue « lire un
  adversaire » de « lire une structure de tournoi », et le taux de rejet est
  compté sur 20 tirages plutôt qu'affirmé sur un seul.

Absence de fuite temporelle : quatre preuves indépendantes, dont l'écart
mesuré à un oracle qui, lui, triche.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

from pfs.analysis import inference_check
from pfs.analysis.inference_check import (
    InferenceCheckError,
    PointBalayage,
    PointCourbe,
    ResultatStat,
    _baseline_prequentielle,
    _cle_chronologique,
    _collecter,
    _log_vraisemblance,
    _n_obs_requis,
    _predire_sequentiel,
    _taux_population,
    _wilcoxon,
    evaluer_dossiers,
    evaluer_inference,
    force_prior_empirique,
)
from pfs.data.hand_history import (
    ActionType,
    HandAction,
    ParsedHand,
    PlayerSeat,
    Room,
    Street,
    parse_file,
)

# Force du prior utilisée dans les tests : volontairement faible (2
# pseudo-mains) pour que la détection soit lisible sur quelques dizaines de
# mains. Ce n'est PAS la calibration du projet (empirical Bayes, jusqu'à 30) ;
# les tests mesurent le pouvoir de la métrique, pas le réglage de production.
FORCE_TEST = 2.0


def _main(
    hand_id: str,
    paires: list[tuple[str, ActionType]],
    table: str = "TABLE",
    horodatage: str | None = None,
) -> ParsedHand:
    """Main préflop minimale : chaque joueur suit (VPIP) ou se couche.

    Aucune relance, aucun board : ``stat_observations`` n'émet donc que
    ``vpip`` et ``pfr``, et l'évaluation d'une seule statistique est isolée de
    tout le reste.
    """
    raw = ""
    if horodatage is not None:
        raw = f"<game><general><startdate>{horodatage}</startdate></general></game>"
    return ParsedHand(
        room=Room.IPOKER,
        hand_id=hand_id,
        is_real_money=True,
        is_tournament=True,
        big_blind=20.0,
        ante=0.0,
        table=table,
        button_seat=1,
        seats=[PlayerSeat(i + 1, j, 1000.0) for i, (j, _) in enumerate(paires)],
        actions=[HandAction(j, Street.PREFLOP, a) for j, a in paires],
        raw=raw,
    )


def _corpus(
    taux: dict[str, float], n_mains: int, seed: int,
    table: str = "TABLE", decalage_id: int = 0,
) -> list[ParsedHand]:
    """Tire ``n_mains`` mains où chaque joueur suit avec sa probabilité propre."""
    rng = np.random.default_rng(seed)
    mains = []
    for i in range(n_mains):
        paires = [
            (j, ActionType.CALL if rng.random() < t else ActionType.FOLD)
            for j, t in taux.items()
        ]
        mains.append(_main(str(decalage_id + i + 1), paires, table))
    return mains


def _evaluer(mains, **kwargs):
    """Évaluation courte (peu de tirages) de la seule statistique vpip."""
    params = dict(
        stats=("vpip",), force_prior=FORCE_TEST,
        n_permutations=199, n_bootstrap=400, seed=1234,
    )
    params.update(kwargs)
    return evaluer_inference(mains, **params).resultat("vpip")


LARGES = {f"L{i}": 0.85 for i in range(4)}
SERRES = {f"S{i}": 0.15 for i in range(4)}

_NOMS_XML = ("Alpha", "Bravo", "Charlie")


def _session_ipoker(code: str, suivis: Sequence[Sequence[bool]]) -> str:
    """Session iPoker minimale : trois joueurs, une décision préflop chacun.

    ``evaluer_dossiers`` part de fichiers sur disque et non de ``ParsedHand``.
    Sans vraie session XML, tout ce qui distingue cette fonction d'un simple
    appel à ``evaluer_inference`` — parcours récursif, parsing, comptage des
    fichiers rejetés — resterait non exercé, alors que c'est par elle que
    passent toutes les mesures sur corpus réel.
    """
    jeux = []
    for g, suivi in enumerate(suivis):
        preflop = "".join(
            f'<action no="{3 + i}" player="{nom}" sum="20" '
            f'type="{"3" if suivi[i] else "0"}"/>'
            for i, nom in enumerate(_NOMS_XML))
        joueurs = "".join(
            f'<player bet="20" chips="1000" dealer="{1 if i == 2 else 0}" '
            f'name="{nom}" seat="{i + 1}" win="0"/>'
            for i, nom in enumerate(_NOMS_XML))
        jeux.append(
            f'<game gamecode="{code}{g:03d}">'
            f'<general><startdate>2026-04-{1 + g // 20:02d} '
            f'{g % 20:02d}:00:00</startdate>'
            f'<smallblind>10</smallblind><bigblind>20</bigblind>'
            f'<players>{joueurs}</players></general>'
            f'<round no="0">'
            f'<action no="1" player="{_NOMS_XML[0]}" sum="10" type="1"/>'
            f'<action no="2" player="{_NOMS_XML[1]}" sum="20" type="2"/>'
            f'</round>'
            f'<round no="1">{preflop}</round>'
            f'</game>')
    return ('<?xml version="1.0" encoding="utf-8"?>'
            f'<session sessioncode="{code}"><general><mode>real</mode>'
            '<gametype>Holdem NL</gametype><nickname>Zulu</nickname>'
            f'<tournamentcode>{code}</tournamentcode></general>'
            f'{"".join(jeux)}</session>')


def _resultat(
    gain: float = 0.01,
    p_permutation: float = 0.001,
    p_permutation_table: float = 0.001,
    courbe: tuple[PointCourbe, ...] = (),
    n_obs_requis: float = float("inf"),
) -> ResultatStat:
    """``ResultatStat`` fabriqué de toutes pièces pour tester ses verdicts.

    Les quatre branches de ``conclusion`` dépendent d'un triplet (gain, deux
    p-values) qu'aucun corpus ne permet de placer où l'on veut : deux d'entre
    elles ne sortiraient jamais d'un tirage synthétique, et ce sont justement
    celles qui produisent les verdicts « non concluant » du corpus réel.
    """
    return ResultatStat(
        stat="vpip", n_obs=100, n_joueurs=10, taux_population=0.5,
        force_prior=2.0, logv_baseline=-0.7, logv_modele=-0.7 + gain,
        brier_baseline=0.25, brier_modele=0.25, p_wilcoxon=0.5,
        p_permutation=p_permutation, p_permutation_table=p_permutation_table,
        ic_bootstrap=(-0.01, 0.03), erreur_type=0.01,
        n_obs_requis=n_obs_requis, courbe=courbe, balayage=(),
    )


def _rapport(**kwargs) -> inference_check.RapportInference:
    """Rapport à une statistique, pour tester ce que ``explain()`` imprime."""
    return inference_check.RapportInference(
        n_mains=50, n_joueurs=10, n_tables=2, n_joueurs_multi_tables=1,
        discount=1.0, exclure_heros=True, baseline_prequentielle=False,
        n_permutations=499, resultats=(_resultat(**kwargs),),
    )


class TestDetectionHeterogeneite(unittest.TestCase):
    """4 joueurs très larges (85 %) contre 4 très serrés (15 %), 80 mains."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.res = _evaluer(_corpus({**LARGES, **SERRES}, 80, seed=1))

    def test_corpus_conforme(self) -> None:
        self.assertEqual(self.res.n_obs, 640)      # 8 joueurs × 80 mains
        self.assertEqual(self.res.n_joueurs, 8)
        # taux de population à mi-chemin : le prior seul ne peut rien dire
        self.assertAlmostEqual(self.res.taux_population, 0.5, delta=0.05)

    def test_gain_de_log_vraisemblance(self) -> None:
        self.assertGreater(self.res.gain_logv, 0.15)
        self.assertGreater(self.res.logv_modele, self.res.logv_baseline)

    def test_gain_de_brier(self) -> None:
        self.assertGreater(self.res.gain_brier, 0.05)
        self.assertLess(self.res.brier_modele, self.res.brier_baseline)

    def test_significativite(self) -> None:
        self.assertLess(self.res.p_permutation, 0.05)
        self.assertLess(self.res.p_permutation_table, 0.05)
        self.assertLess(self.res.p_wilcoxon, 0.05)
        self.assertTrue(self.res.significatif)
        self.assertIn("BAT", self.res.conclusion)

    def test_ic_bootstrap_exclut_zero(self) -> None:
        bas, haut = self.res.ic_bootstrap
        self.assertGreater(bas, 0.0)
        self.assertGreater(haut, bas)

    def test_courbe_monte_puis_se_stabilise(self) -> None:
        par_tranche = {p.k_min: p for p in self.res.courbe if p.n_obs}
        self.assertEqual(par_tranche[0].gain_logv, 0.0)   # rien vu, rien gagné
        self.assertGreater(par_tranche[1].gain_logv, 0.0)
        self.assertEqual(self.res.k_seuil, 1)             # gain dès la 2e main


class TestControleNegatif(unittest.TestCase):
    """Population homogène : il n'y a rien à apprendre, et rien ne doit l'être."""

    def test_aucun_gain_significatif(self) -> None:
        res = _evaluer(_corpus({f"H{i}": 0.5 for i in range(8)}, 80, seed=2))
        self.assertFalse(res.significatif)
        self.assertGreater(res.p_permutation, 0.05)
        # Apprendre un paramètre qui n'existe pas COÛTE : le gain est négatif.
        self.assertLess(res.gain_logv, 0.0)

    def test_stable_sur_plusieurs_tirages(self) -> None:
        """Le coût d'apprendre un paramètre inexistant, chiffré et encadré.

        Les bornes ne sont pas rondes : ce sont les valeurs mesurées sur ces
        cinq graines exactement (−0,023060 à −0,015362), élargies d'un
        cinquième. Une fourchette arrondie « à peu près » masquerait
        justement la graine 6, la plus proche de zéro.
        """
        for seed in (2, 3, 4, 5, 6):
            with self.subTest(seed=seed):
                res = _evaluer(
                    _corpus({f"H{i}": 0.5 for i in range(8)}, 80, seed=seed),
                    n_permutations=99, n_bootstrap=200)
                self.assertFalse(res.significatif)
                self.assertGreater(res.gain_logv, -0.0277)
                self.assertLess(res.gain_logv, -0.0123)

    def test_verdict_annonce_le_retrait(self) -> None:
        res = _evaluer(_corpus({f"H{i}": 0.5 for i in range(8)}, 80, seed=2),
                       n_permutations=99, n_bootstrap=200)
        self.assertIn("retrait", res.conclusion)


class TestConfondantTable(unittest.TestCase):
    """Écart purement de TABLE : 4 joueurs à 85 % d'un côté, 4 à 15 % de l'autre.

    Aucun joueur ne se distingue de ses voisins de table. Le gain du modèle
    est réel — il apprend le niveau de la table — mais il n'est pas de la
    lecture d'adversaire, et le protocole doit le dire.
    """

    @classmethod
    def setUpClass(cls) -> None:
        a = _corpus({f"A{i}": 0.85 for i in range(4)}, 80, seed=100,
                    table="TABLE_A")
        b = _corpus({f"B{i}": 0.15 for i in range(4)}, 80, seed=500,
                    table="TABLE_B", decalage_id=1000)
        cls.res = _evaluer(a + b)

    def test_le_gain_existe(self) -> None:
        self.assertGreater(self.res.gain_logv, 0.1)

    def test_permutation_globale_le_prend_pour_du_signal(self) -> None:
        self.assertLess(self.res.p_permutation, 0.05)

    def test_permutation_stratifiee_le_rejette(self) -> None:
        self.assertGreater(self.res.p_permutation_table, 0.05)
        self.assertFalse(self.res.significatif)
        self.assertIn("table constante", self.res.conclusion)


class TestCalibrationStratifiee(unittest.TestCase):
    """Le taux de rejet des deux permutations, compté sur 20 tirages.

    Une seule paire de graines ne dit rien d'un taux de rejet : elle montre
    au mieux qu'un cas existe. Ici l'effet est PUREMENT de table sur chacun
    des 20 corpus, donc l'hypothèse nulle de la permutation stratifiée est
    vraie à chaque fois — le nombre de rejets qu'elle produit EST son niveau
    empirique.

    Mesuré (25 mains × 8 joueurs, B = 99, α = 5 %) : globale 20/20,
    stratifiée 0/20, la plus petite p stratifiée valant 0,070. Un rejet sur
    20 resterait compatible avec un niveau nominal de 5 % ; ce qui compte est
    l'écart entre les deux colonnes, pas la valeur exacte de la seconde.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.p_globale: list[float] = []
        cls.p_stratifiee: list[float] = []
        for t in range(20):
            a = _corpus({f"A{i}": 0.85 for i in range(4)}, 25,
                        seed=1000 + t, table="TABLE_A")
            b = _corpus({f"B{i}": 0.15 for i in range(4)}, 25,
                        seed=5000 + t, table="TABLE_B", decalage_id=10_000)
            res = _evaluer(a + b, n_permutations=99, n_bootstrap=100)
            cls.p_globale.append(res.p_permutation)
            cls.p_stratifiee.append(res.p_permutation_table)

    def test_la_globale_rejette_a_chaque_tirage(self) -> None:
        self.assertEqual(sum(p < 0.05 for p in self.p_globale), 20)

    def test_la_stratifiee_ne_rejette_jamais(self) -> None:
        self.assertEqual(sum(p < 0.05 for p in self.p_stratifiee), 0)
        self.assertGreater(min(self.p_stratifiee), 0.05)


class TestFuiteTemporelle(unittest.TestCase):
    """Quatre preuves indépendantes que la prédiction n'utilise que le passé."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mains = _corpus({**LARGES, **SERRES}, 80, seed=1)
        cls.occ = _collecter(cls.mains, ("vpip",), True)["vpip"]
        cls.p, cls.k = _predire_sequentiel(
            "vpip", cls.occ.joueurs, cls.occ.resultats, 0.5, FORCE_TEST, 1.0)

    def test_premiere_occasion_egale_le_prior(self) -> None:
        """À k = 0 le modèle n'a rien vu : il ne peut que répéter la baseline."""
        premieres = self.k == 0
        self.assertEqual(int(premieres.sum()), self.occ.n_joueurs)
        np.testing.assert_allclose(self.p[premieres], 0.5, atol=1e-12)

    def test_posterieur_analytique_sans_lookahead(self) -> None:
        """p_i vaut exactement (a0 + succès des i−1 précédentes)/(a0+b0+i).

        Toute fuite — même d'une seule observation — déplacerait ce rapport.
        """
        a0 = b0 = 0.5 * FORCE_TEST
        succes: dict[int, int] = {}
        vus: dict[int, int] = {}
        for i in range(self.occ.joueurs.size):
            j = int(self.occ.joueurs[i])
            s, n = succes.get(j, 0), vus.get(j, 0)
            self.assertAlmostEqual(self.p[i], (a0 + s) / (a0 + b0 + n), places=12)
            succes[j] = s + int(self.occ.resultats[i])
            vus[j] = n + 1

    def test_moins_bon_qu_un_oracle_qui_triche(self) -> None:
        """L'oracle utilise le taux plein échantillon, futur compris.

        Il est structurellement imbattable : si le modèle l'égalait, c'est
        qu'il tricherait aussi.
        """
        s = np.bincount(self.occ.joueurs,
                        weights=self.occ.resultats.astype(np.float64))
        n = np.bincount(self.occ.joueurs).astype(np.float64)
        p_oracle = ((s + 0.5) / (n + 1.0))[self.occ.joueurs]
        lv_oracle = _log_vraisemblance(p_oracle, self.occ.resultats).mean()
        lv_modele = _log_vraisemblance(self.p, self.occ.resultats).mean()
        self.assertLess(lv_modele, lv_oracle)

    def test_permuter_les_etiquettes_degrade(self) -> None:
        """Casser le lien joueur ↔ comportement doit faire chuter le score."""
        rng = np.random.default_rng(99)
        reference = float(_log_vraisemblance(self.p, self.occ.resultats).mean())
        for _ in range(5):
            p_perm, _ = _predire_sequentiel(
                "vpip", rng.permutation(self.occ.joueurs), self.occ.resultats,
                0.5, FORCE_TEST, 1.0)
            degrade = float(
                _log_vraisemblance(p_perm, self.occ.resultats).mean())
            self.assertLess(degrade, reference)


class TestEchangeabilite(unittest.TestCase):
    """Pourquoi « permuter l'ordre » ne teste PAS la fuite temporelle.

    À δ = 1 la log-vraisemblance préquentielle est la vraisemblance marginale
    de la séquence : elle ne dépend que des comptages, donc pas de l'ordre.
    Le fait est mesuré ici pour qu'aucun test futur ne s'appuie dessus.
    """

    @staticmethod
    def _total(x: np.ndarray, discount: float) -> float:
        joueur = np.zeros(x.size, dtype=np.int64)
        p, _ = _predire_sequentiel("v", joueur, x, 0.4, FORCE_TEST, discount)
        return float(_log_vraisemblance(p, x).sum())

    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.x = rng.random(200) < 0.7
        self.melange = rng.permutation(self.x)

    def test_invariant_a_delta_un(self) -> None:
        """Écart mesuré sur cette séquence : 1,4e-14 — la borne est à 1e-12."""
        self.assertLessEqual(
            abs(self._total(self.x, 1.0) - self._total(self.melange, 1.0)),
            1e-12)

    def test_sensible_des_que_delta_est_inferieur(self) -> None:
        """Avec oubli, l'ordre compte de nouveau — sans direction privilégiée."""
        self.assertNotAlmostEqual(
            self._total(self.x, 0.9), self._total(self.melange, 0.9), places=3)

    def test_invariant_a_l_echelle_et_a_la_calibration_du_projet(self) -> None:
        """La même propriété sur 920 occasions, 8 joueurs, prior de force 30.

        La séquence à un joueur ci-dessus est trop douce pour dire quelque
        chose de l'erreur d'arrondi réelle : elle donne 1,4e-14 là où le
        corpus de production monte à 1,14e-13 (vpip, 921 occasions, 5
        permutations intra-joueur). La borne 1e-12 est fixée une décade
        au-dessus de ce pire cas mesuré, pas au jugé.
        """
        occ = _collecter(_corpus({**LARGES, **SERRES}, 115, seed=11),
                         ("vpip",), True)["vpip"]
        taux, force = 0.63, 30.0
        p, _ = _predire_sequentiel(
            "vpip", occ.joueurs, occ.resultats, taux, force, 1.0)
        reference = float(_log_vraisemblance(p, occ.resultats).sum())

        rng = np.random.default_rng(77)
        for _ in range(5):
            ordre = np.arange(occ.joueurs.size)
            for j in range(occ.n_joueurs):
                bloc = np.flatnonzero(occ.joueurs == j)
                ordre[bloc] = bloc[rng.permutation(bloc.size)]
            p_melange, _ = _predire_sequentiel(
                "vpip", occ.joueurs[ordre], occ.resultats[ordre],
                taux, force, 1.0)
            total = float(
                _log_vraisemblance(p_melange, occ.resultats[ordre]).sum())
            self.assertLessEqual(abs(total - reference), 1e-12)


class TestBaselinePrequentielle(unittest.TestCase):
    """Le mode symétrique : ni la baseline ni le modèle ne voient le futur.

    Handicaper la seule baseline transformerait la variante « plus dure » en
    variante plus facile, le modèle gardant un prior estimé sur le corpus
    entier. Le symptôme est chiffré ci-dessous : sur ce corpus, ce prior
    conservé rapporte +0,035 de log-vraisemblance par occasion AVANT que le
    modèle ait vu la moindre main du joueur.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.mains = _corpus({**LARGES, **SERRES}, 80, seed=1)
        cls.occ = _collecter(cls.mains, ("vpip",), True)["vpip"]
        cls.p_base = _baseline_prequentielle(cls.occ.resultats)
        cls.logv_base = _log_vraisemblance(cls.p_base, cls.occ.resultats)

    def _gain_a_k_zero(self, prior: float | np.ndarray) -> tuple[float, float]:
        p, k = _predire_sequentiel(
            "vpip", self.occ.joueurs, self.occ.resultats, prior,
            FORCE_TEST, 1.0)
        d = _log_vraisemblance(p, self.occ.resultats) - self.logv_base
        return float(d[k == 0].mean()), float(np.abs(d[k == 0]).max())

    def test_la_baseline_ne_lit_que_le_passe(self) -> None:
        """Elle vaut le taux Jeffreys des occasions strictement antérieures."""
        x = np.array([True, False, True, True], dtype=np.bool_)
        np.testing.assert_allclose(
            _baseline_prequentielle(x),
            [0.5 / 1.0, 1.5 / 2.0, 1.5 / 3.0, 2.5 / 4.0], atol=1e-15)

    def test_une_issue_future_ne_deplace_pas_le_passe(self) -> None:
        x = self.occ.resultats.copy()
        y = x.copy()
        y[-1] = not y[-1]
        np.testing.assert_array_equal(
            _baseline_prequentielle(x)[:-1], _baseline_prequentielle(y)[:-1])

    def test_le_prior_plein_corpus_offre_un_gain_gratuit(self) -> None:
        moyen, _ = self._gain_a_k_zero(_taux_population(self.occ.resultats))
        self.assertGreater(moyen, 0.03)

    def test_le_prior_suivi_annule_ce_gain(self) -> None:
        moyen, pire = self._gain_a_k_zero(self.p_base)
        self.assertLessEqual(abs(moyen), 1e-15)
        self.assertLessEqual(pire, 1e-15)

    def test_delta_nul_a_k_zero_dans_le_rapport(self) -> None:
        res = _evaluer(self.mains, baseline_prequentielle=True,
                       n_permutations=0, n_bootstrap=100)
        self.assertEqual(res.courbe[0].n_obs, 8)
        self.assertLessEqual(abs(res.courbe[0].gain_logv), 1e-15)

    def test_le_mode_plein_corpus_est_nul_a_l_arrondi_pres(self) -> None:
        """Δ à k = 0 reste nul quand le prior est un scalaire.

        « Nul » veut dire nul à l'arrondi de α/(α+β) près, pas nécessairement
        au bit : c'est exactement 0 sur les cinq statistiques du corpus réel
        et sur ce corpus-ci, mais 2,5e-17 dès qu'on inclut le héros. La borne
        retenue est donc numérique, pas rhétorique.
        """
        res = _evaluer(self.mains, n_permutations=0, n_bootstrap=100)
        self.assertLessEqual(abs(res.courbe[0].gain_logv), 1e-15)


class TestOrdreChronologique(unittest.TestCase):
    def test_estampille_relue_dans_raw(self) -> None:
        tard = _main("2", [("A", ActionType.CALL)],
                     horodatage="2026-01-02 10:00:00")
        tot = _main("1", [("A", ActionType.FOLD)],
                    horodatage="2026-01-01 09:00:00")
        self.assertEqual(_cle_chronologique(tard), ("2026-01-02 10:00:00", 2))
        self.assertEqual(
            [h.hand_id for h in sorted([tard, tot], key=_cle_chronologique)],
            ["1", "2"])

    def test_sans_estampille_l_ordre_d_entree_est_conserve(self) -> None:
        sans = _main("abc", [("A", ActionType.CALL)])
        self.assertEqual(_cle_chronologique(sans), ("", 0))

    def test_le_tri_rend_le_resultat_independant_de_l_ordre_d_entree(self) -> None:
        """Mélanger les mains à l'entrée ne change rien : le tri les remet.

        Le témoin est le score de Brier, PAS la log-vraisemblance : celle-ci
        est échangeable (cf. :class:`TestEchangeabilite`) et resterait égale
        même sans tri, donc ne prouverait rien. Le Brier et la courbe
        d'apprentissage, eux, dépendent de l'ordre — mesuré ci-dessous en
        neutralisant le tri.
        """
        mains = _corpus({**LARGES, **SERRES}, 40, seed=8)
        horodates = [
            _main(m.hand_id,
                  [(a.player, a.action) for a in m.actions],
                  horodatage=f"2026-03-{1 + i // 24:02d} {i % 24:02d}:00:00")
            for i, m in enumerate(mains)
        ]
        melange = list(np.random.default_rng(5).permutation(  # type: ignore[arg-type]
            np.array(horodates, dtype=object)))
        params = dict(n_permutations=0, n_bootstrap=100)

        sans_tri = inference_check._cle_chronologique
        try:
            inference_check._cle_chronologique = lambda h: ("", 0)
            a, b = _evaluer(horodates, **params), _evaluer(melange, **params)
            self.assertNotAlmostEqual(a.brier_modele, b.brier_modele, places=6)
        finally:
            inference_check._cle_chronologique = sans_tri

        droit, tordu = _evaluer(horodates, **params), _evaluer(melange, **params)
        self.assertAlmostEqual(droit.brier_modele, tordu.brier_modele, places=12)
        self.assertAlmostEqual(droit.courbe[1].gain_logv,
                               tordu.courbe[1].gain_logv, places=12)


class TestVerdicts(unittest.TestCase):
    """Les quatre verdicts possibles, et les libellés qui les accompagnent."""

    def test_bat_le_prior(self) -> None:
        res = _resultat(gain=0.01, p_permutation=0.002,
                        p_permutation_table=0.01)
        self.assertTrue(res.significatif)
        self.assertIn("BAT", res.conclusion)

    def test_gain_non_attribuable_au_joueur(self) -> None:
        res = _resultat(gain=0.01, p_permutation=0.002,
                        p_permutation_table=0.30)
        self.assertFalse(res.significatif)
        self.assertIn("table constante", res.conclusion)

    def test_modele_en_retrait(self) -> None:
        res = _resultat(gain=-0.01, p_permutation=0.90,
                        p_permutation_table=0.90)
        self.assertIn("retrait", res.conclusion)

    def test_gain_positif_non_significatif(self) -> None:
        """Le verdict des deux statistiques non concluantes du corpus réel."""
        res = _resultat(gain=0.008, p_permutation=0.09,
                        p_permutation_table=0.12)
        self.assertFalse(res.significatif)
        self.assertEqual(res.conclusion, "gain positif mais NON significatif")

    def test_k_seuil_absent_quand_aucune_tranche_ne_gagne(self) -> None:
        courbe = (PointCourbe(0, 0, 10, 0.0, 0.0),
                  PointCourbe(1, 2, 10, -0.02, -0.01),
                  PointCourbe(3, 5, 0, 0.9, 0.9))   # tranche vide : ignorée
        self.assertIsNone(_resultat(courbe=courbe).k_seuil)

    def test_libelles_des_tranches(self) -> None:
        self.assertEqual(PointCourbe(0, 0, 1, 0.0, 0.0).libelle, "k=0")
        self.assertEqual(PointCourbe(1, 2, 1, 0.0, 0.0).libelle, "k=1–2")
        self.assertEqual(
            PointCourbe(51, 1_000_000, 1, 0.0, 0.0).libelle, "k≥51")


class TestWilcoxon(unittest.TestCase):
    def test_sans_difference_reelle_pas_de_p_value(self) -> None:
        self.assertTrue(np.isnan(_wilcoxon(np.zeros(20))))

    def test_le_bruit_d_arrondi_ne_compte_pas_pour_une_difference(self) -> None:
        """1e-16 est l'écart du prior recopié, pas une prédiction meilleure."""
        self.assertTrue(np.isnan(_wilcoxon(np.full(20, 2e-16))))

    def test_differences_reelles_produisent_une_p_value(self) -> None:
        p = _wilcoxon(np.concatenate([np.zeros(10), np.full(10, 0.01)]))
        self.assertLess(p, 0.05)


class TestExtrapolationPuissance(unittest.TestCase):
    """``n_obs_requis`` est imprimé par ``explain()`` : il doit être vérifié."""

    def test_effet_nul_ou_variance_nulle_sont_hors_de_portee(self) -> None:
        self.assertEqual(_n_obs_requis(0.0, 0.05, 900), float("inf"))
        self.assertEqual(_n_obs_requis(0.01, 0.0, 900), float("inf"))

    def test_l_effectif_rendu_amene_pile_au_seuil_de_puissance(self) -> None:
        """À n′ occasions, |gain| / erreur-type doit valoir z_α + z_β.

        C'est la définition même de « 80 % de puissance à 5 % bilatéral » ; la
        vérifier ainsi teste la formule sans la réécrire.
        """
        gain, se, n = 0.011, 0.0047, 921
        n_prime = _n_obs_requis(gain, se, n)
        se_prime = se * np.sqrt(n / n_prime)
        self.assertAlmostEqual(
            abs(gain) / se_prime,
            inference_check._Z_ALPHA + inference_check._Z_BETA, places=9)

    def test_un_effet_deux_fois_plus_gros_coute_quatre_fois_moins(self) -> None:
        self.assertAlmostEqual(
            _n_obs_requis(0.02, 0.05, 900) * 4.0,
            _n_obs_requis(0.01, 0.05, 900), places=6)

    def test_un_effet_tenu_demande_bien_plus_que_le_corpus(self) -> None:
        """Huit joueurs séparés de 2 points de VPIP : mesuré ×9 le corpus."""
        res = _evaluer(
            _corpus({f"T{i}": 0.45 + 0.02 * i for i in range(8)}, 80, seed=1),
            n_permutations=0, n_bootstrap=200)
        self.assertTrue(np.isfinite(res.n_obs_requis))
        self.assertGreater(res.n_obs_requis, 5.0 * res.n_obs)

    def test_la_ligne_n_apparait_que_si_le_corpus_ne_suffit_pas(self) -> None:
        court, large = _rapport(n_obs_requis=9_700.0), _rapport(n_obs_requis=42.0)
        self.assertIn("80 % de puissance", court.explain())
        self.assertIn("9 700 occasions", court.explain())
        self.assertNotIn("80 % de puissance", large.explain())


class TestBalayage(unittest.TestCase):
    """La sensibilité à la force du prior sert de conclusion : elle est testée."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.res = _evaluer(_corpus({**LARGES, **SERRES}, 80, seed=1),
                           n_permutations=0, n_bootstrap=200)

    def test_forces_balayees_dans_l_ordre_demande(self) -> None:
        self.assertEqual([b.force for b in self.res.balayage],
                         [1.0, 2.0, 5.0, 10.0, 30.0])
        for b in self.res.balayage:
            self.assertIsInstance(b, PointBalayage)
            self.assertEqual(b.n_obs, self.res.n_obs)

    def test_le_point_du_balayage_a_la_force_utilisee_redonne_le_gain(self) -> None:
        """Sans cette égalité, le balayage mesurerait autre chose que le titre."""
        point = next(b for b in self.res.balayage if b.force == FORCE_TEST)
        self.assertAlmostEqual(point.gain_logv, self.res.gain_logv, places=12)

    def test_le_sens_du_balayage_depend_de_l_heterogeneite_reelle(self) -> None:
        """Le balayage n'a pas de bon sens a priori — c'est ce qui le rend utile.

        Population franchement hétérogène : le prior est un poids mort, la
        force 1 domine la force 30 (+0,2692 contre +0,1981). Population
        homogène : apprendre du bruit coûte, et c'est le prior fort qui
        protège (−0,0230 contre −0,0029). Un balayage qui pencherait toujours
        du même côté ne diagnostiquerait rien.
        """
        homogene = _evaluer(_corpus({f"H{i}": 0.5 for i in range(8)}, 80,
                                    seed=1),
                            n_permutations=0, n_bootstrap=200)
        for res, attendu in ((self.res, "faible"), (homogene, "fort")):
            faible = next(b for b in res.balayage if b.force == 1.0)
            fort = next(b for b in res.balayage if b.force == 30.0)
            gagnant = "faible" if faible.gain_logv > fort.gain_logv else "fort"
            self.assertEqual(gagnant, attendu)

    def test_balayage_desactivable(self) -> None:
        rapport = evaluer_inference(
            _corpus({"A": 0.8, "B": 0.2}, 10, seed=4), stats=("vpip",),
            force_prior=FORCE_TEST, forces_balayage=(),
            n_permutations=0, n_bootstrap=50)
        self.assertEqual(rapport.resultat("vpip").balayage, ())
        self.assertNotIn("sensibilité à la force du prior :", rapport.explain())


class TestEvaluerDossiers(unittest.TestCase):
    """L'API par laquelle passent TOUTES les mesures sur corpus réel."""

    SUIVIS = [[True, False, True]] * 12 + [[False, True, True]] * 12

    def test_lit_recursivement_et_agrege(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            racine = Path(d)
            (racine / "sous").mkdir()
            (racine / "a.xml").write_text(
                _session_ipoker("11", self.SUIVIS), encoding="utf-8")
            (racine / "sous" / "b.xml").write_text(
                _session_ipoker("22", self.SUIVIS), encoding="utf-8")
            rapport = evaluer_dossiers(
                [racine], stats=("vpip",), force_prior=FORCE_TEST,
                n_permutations=0, n_bootstrap=100)
        self.assertEqual(rapport.n_mains, 48)
        self.assertEqual(rapport.n_joueurs, 3)
        self.assertEqual(rapport.resultat("vpip").n_obs, 144)
        self.assertEqual(rapport.n_fichiers_ignores, 0)
        self.assertNotIn("ATTENTION", rapport.explain())

    def test_un_fichier_illisible_est_compte_et_signale(self) -> None:
        """Un corpus à moitié perdu ne doit pas produire un rapport normal."""
        with tempfile.TemporaryDirectory() as d:
            racine = Path(d)
            (racine / "bon.xml").write_text(
                _session_ipoker("33", self.SUIVIS), encoding="utf-8")
            (racine / "vide.xml").write_text("<session/>", encoding="utf-8")
            (racine / "brut.xml").write_text("ceci n'est pas du XML",
                                             encoding="utf-8")
            # Un répertoire portant l'extension surveillée : le seul cas qui
            # fasse VRAIMENT lever la lecture, les fichiers malformés
            # ressortant vides sans exception.
            (racine / "piege.xml").mkdir()
            rapport = evaluer_dossiers(
                [racine], stats=("vpip",), force_prior=FORCE_TEST,
                n_permutations=0, n_bootstrap=100)
        self.assertEqual(rapport.n_mains, 24)
        self.assertEqual(rapport.n_fichiers_ignores, 3)
        self.assertIn("3 fichier(s) sans main exploitable", rapport.explain())

    def test_le_sel_repseudonymise_sans_deplacer_un_chiffre(self) -> None:
        """Le sel est transmis au parseur, et n'est QUE de la pseudonymisation.

        S'il changeait un résultat, deux dossiers salés différemment
        donneraient deux vérités — or c'est exactement ce que la docstring
        interdit à l'appelant de faire.
        """
        with tempfile.TemporaryDirectory() as d:
            racine = Path(d)
            fichier = racine / "a.xml"
            fichier.write_text(_session_ipoker("44", self.SUIVIS),
                               encoding="utf-8")
            noms = {s: {seat.player
                        for h in parse_file(fichier, s) for seat in h.seats}
                    for s in ("", "poivre")}
            params = dict(stats=("vpip",), force_prior=FORCE_TEST,
                          n_permutations=0, n_bootstrap=50)
            sans = evaluer_dossiers([racine], **params)
            avec = evaluer_dossiers([racine], salt="poivre", **params)
        self.assertEqual(noms[""] & noms["poivre"], set())
        self.assertEqual(sans.n_joueurs, avec.n_joueurs)
        self.assertAlmostEqual(sans.resultat("vpip").gain_logv,
                               avec.resultat("vpip").gain_logv, places=12)

    def test_dossier_sans_main_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "rien.xml").write_text("<session/>", encoding="utf-8")
            with self.assertRaises(InferenceCheckError):
                evaluer_dossiers([Path(d)], stats=("vpip",))


class TestApi(unittest.TestCase):
    def test_force_prior_empirique_plafonnee(self) -> None:
        self.assertEqual(force_prior_empirique(10_000), 30.0)   # plafond n0
        self.assertEqual(force_prior_empirique(150), 15.0)      # n_pop / 10
        self.assertEqual(force_prior_empirique(0), 0.0)

    def test_effectif_negatif_refuse(self) -> None:
        with self.assertRaises(InferenceCheckError):
            force_prior_empirique(-1)

    def test_corpus_vide_refuse(self) -> None:
        with self.assertRaises(InferenceCheckError):
            evaluer_inference([], stats=("vpip",))

    def test_parametres_invalides_refuses(self) -> None:
        mains = _corpus({"A": 0.5, "B": 0.5}, 5, seed=9)
        with self.assertRaises(InferenceCheckError):
            evaluer_inference(mains, stats=("vpip",), discount=1.5)
        with self.assertRaises(InferenceCheckError):
            evaluer_inference(mains, stats=("vpip",), force_prior=0.0)

    def test_statistique_sans_occasion_ne_fabrique_pas_de_ligne(self) -> None:
        """Aucune occasion ≠ zéro occasion réussie : la stat sort du rapport."""
        rapport = evaluer_inference(
            _corpus({"A": 0.5, "B": 0.5}, 10, seed=9),
            stats=("vpip", "wtsd"), n_permutations=0, n_bootstrap=50)
        self.assertEqual([r.stat for r in rapport.resultats], ["vpip"])

    def test_statistique_non_evaluee(self) -> None:
        rapport = evaluer_inference(
            _corpus({"A": 0.5, "B": 0.5}, 10, seed=9),
            stats=("vpip",), n_permutations=0, n_bootstrap=50)
        with self.assertRaises(InferenceCheckError):
            rapport.resultat("wtsd")

    def test_explain_en_francais_avec_les_chiffres(self) -> None:
        rapport = evaluer_inference(
            _corpus({**LARGES, **SERRES}, 30, seed=1),
            stats=("vpip",), force_prior=FORCE_TEST,
            n_permutations=99, n_bootstrap=200)
        texte = rapport.explain()
        for attendu in ("INFÉRENCE ADVERSE", "Mains analysées", "vpip",
                        "log-vraisemblance/obs", "Brier/obs",
                        "permutation globale", "permutation à table constante",
                        "VERDICT"):
            self.assertIn(attendu, texte)
        self.assertEqual(rapport.n_mains, 30)
        self.assertEqual(rapport.n_joueurs, 8)
        self.assertEqual(rapport.n_tables, 1)
        self.assertEqual(rapport.n_joueurs_multi_tables, 0)

    def test_heros_exclu_par_defaut(self) -> None:
        """Un siège marqué héros ne doit produire aucune occasion."""
        mains = []
        for i in range(20):
            m = _main(str(i + 1),
                      [("HERO", ActionType.CALL), ("VIL", ActionType.FOLD)])
            m.seats = [PlayerSeat(1, "HERO", 1000.0, is_hero=True),
                       PlayerSeat(2, "VIL", 1000.0)]
            mains.append(m)
        res = _evaluer(mains, n_permutations=0, n_bootstrap=50)
        self.assertEqual(res.n_joueurs, 1)
        self.assertEqual(res.n_obs, 20)


if __name__ == "__main__":
    unittest.main()
