r"""L'incertitude du Monte-Carlo ICM : mesurée, et rendue à l'appelant.

Ce que ce fichier protège
-------------------------
`_finish_probs_mc` calculait ``se = sqrt(0.25/n)`` puis le rendait dans un
tuple dont tous les appelants écrivaient ``probs, _ = ...``. Deux défauts
superposés :

1. **ce n'était pas l'erreur — c'était sa BORNE.** :math:`\sqrt{p(1-p)/n}`
   atteint :math:`\sqrt{0{,}25/n}` en p = 0,5 seulement. Sur une table à neuf
   joueurs, les probabilités de place tournent autour de 1/9, où l'erreur
   réelle vaut le tiers de la borne. Mesuré ici : la borne surestime l'erreur
   par place d'un facteur **1,0 à 7,4** ; sur le $EV — le chiffre que
   l'utilisateur lit — d'un facteur **6,3** (0,49 $ réels contre 3,12 $
   annoncés, à 10 000 tirages sur la table PMU à neuf joueurs).

2. **elle n'arrivait à personne.** Une largeur d'intervalle NATIVE, déjà
   calculée, était jetée à chaque appel — alors que le projet veut justement
   construire la propagation par intervalles.

Ce que ce fichier NE prétend pas
--------------------------------
``erreur_type`` mesure le bruit d'ÉCHANTILLONNAGE. Sur le chemin exact il
vaut zéro, et c'est vrai : la récurrence de Harville ne tire rien au sort.
Cela ne dit rien de l'erreur de MODÈLE — Harville reste une approximation de
la marche aléatoire absorbante, et cet écart-là n'est pas mesuré ici.

Les chiffres sont reproductibles :

    python -m pytest tests/test_icm_erreur_type.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pfs.core.icm import (
    EXACT_LIMIT,
    IcmError,
    _finish_probs_exact,
    _finish_probs_mc,
    erreur_type_exacte,
    icm_equities,
    icm_equities_mesurees,
)

TABLE_9 = (35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
PAYOUTS_9 = (176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34)

CAS = [
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="bulle-4-3"),
    pytest.param(TABLE_9, PAYOUTS_9, id="table-PMU-9"),
    pytest.param((9000.0, 500.0, 300.0, 200.0), (70.0, 30.0),
                 id="chipleader-ecrasant"),
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. L'ERREUR PAR PLACE EST MESURÉE, PLUS BORNÉE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", CAS)
@pytest.mark.parametrize("n_sims", [10_000, 40_000])
def test_l_erreur_par_place_suit_la_loi_binomiale(stacks, payouts, n_sims) -> None:
    r"""L'erreur rendue est bien :math:`\sqrt{\hat p(1-\hat p)/n}`, place par place.

    C'est une identité algébrique sur l'estimateur, donc vérifiable
    exactement : la recalculer depuis les fréquences rendues doit retomber
    dessus au bit près.
    """
    probs, erreur = _finish_probs_mc(tuple(stacks), len(payouts), n_sims, 0)
    attendu = np.sqrt(probs * (1.0 - probs) / n_sims)
    np.testing.assert_allclose(erreur, attendu, rtol=0, atol=1e-18)


@pytest.mark.parametrize("stacks,payouts", CAS)
def test_l_erreur_mesuree_est_strictement_sous_l_ancienne_borne(
    stacks, payouts
) -> None:
    """La borne √(0,25/n) majore, et de loin — c'est tout le défaut.

    Elle n'est jamais FAUSSE (c'est un maximum), elle est inutilisable :
    annoncer trois à sept fois trop d'incertitude sur chaque place fait
    déclarer « fragile » des verdicts qui ne le sont pas.
    """
    n_sims = 20_000
    borne = float(np.sqrt(0.25 / n_sims))
    _, erreur = _finish_probs_mc(tuple(stacks), len(payouts), n_sims, 0)
    assert float(erreur.max()) <= borne + 1e-15
    positives = erreur[erreur > 0.0]
    assert float(borne / positives.min()) > 2.0, (
        "la borne ne surestimerait plus rien : le défaut corrigé n'en "
        "serait pas un")


@pytest.mark.parametrize("n_sims", [10_000, 100_000])
def test_sur_le_dollar_la_borne_surestime_d_un_facteur_six(n_sims: int) -> None:
    """Le chiffre que l'utilisateur lit, c'est le $EV — pas une probabilité.

    Transposer l'ancienne borne au $EV demandait de la multiplier par la
    dotation (chaque place pouvant valoir jusqu'à son gain). Mesuré sur la
    table PMU à neuf joueurs : 3,12 $ annoncés contre 0,49 $ réels à 10 000
    tirages, 0,99 $ contre 0,16 $ à 100 000 — un facteur 6,3 dans les deux
    cas, puisque les deux décroissent en 1/√n.
    """
    pay = np.asarray(PAYOUTS_9)
    probs, _ = _finish_probs_mc(TABLE_9, len(PAYOUTS_9), n_sims, 0)
    reelle = float(erreur_type_exacte(probs, pay, n_sims).max())
    ancienne = float(np.sqrt(0.25 / n_sims)) * float(pay.sum())
    assert ancienne / reelle == pytest.approx(6.3, abs=0.4)


# ═══════════════════════════════════════════════════════════════════════════
# 2. L'ERREUR SUR LE $EV — ET LE FAIT QU'ELLE SOIT JUSTE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", CAS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_l_ecart_au_calcul_exact_tient_dans_l_erreur_annoncee(
    stacks, payouts, seed
) -> None:
    r"""Le seul test qui prouve que l'erreur annoncée est la BONNE.

    Une erreur-type n'est vérifiable que contre la vérité. Ici la vérité
    existe : la récurrence de Harville donne les probabilités exactes. On
    mesure donc l'écart RÉEL entre les deux chemins, en unités de l'erreur
    annoncée, et on exige qu'il tienne dans 5 σ.

    Un σ surestimé rendrait ce test trivialement vert — c'est pourquoi il est
    doublé du test de non-trivialité ci-dessous, qui exige que le |z| observé
    ne soit pas ridiculement petit sur l'ensemble des graines.
    """
    n_sims = 40_000
    pay = np.asarray(payouts)
    exact = _finish_probs_exact(tuple(stacks), len(payouts))
    sigma = erreur_type_exacte(exact, pay, n_sims)
    probs, _ = _finish_probs_mc(tuple(stacks), len(payouts), n_sims, seed)

    z = np.abs(probs @ pay - exact @ pay) / np.where(sigma > 0, sigma, np.inf)
    assert float(z.max()) <= 5.0, f"écart de {float(z.max()):.2f} σ"


@pytest.mark.parametrize("stacks,payouts", CAS)
def test_l_erreur_annoncee_n_est_pas_gonflee(stacks, payouts) -> None:
    """Le revers du test précédent : σ ne doit pas être une borne déguisée.

    Sur 12 graines, le |z| MAXIMAL observé doit dépasser 1 : si l'erreur
    annoncée était surdimensionnée, les écarts réels tiendraient tous dans une
    fraction de σ et le test à 5 σ ne prouverait rien. C'est exactement le
    piège que l'ancienne borne tendait.
    """
    n_sims = 40_000
    pay = np.asarray(payouts)
    exact = _finish_probs_exact(tuple(stacks), len(payouts))
    sigma = erreur_type_exacte(exact, pay, n_sims)
    pires = []
    for seed in range(12):
        probs, _ = _finish_probs_mc(tuple(stacks), len(payouts), n_sims, seed)
        z = np.abs(probs @ pay - exact @ pay) / np.where(sigma > 0, sigma, np.inf)
        pires.append(float(z.max()))
    assert max(pires) > 1.0, (
        f"|z| max = {max(pires):.2f} sur 12 graines : σ est surdimensionnée")


def test_l_erreur_du_ev_n_est_pas_la_somme_des_erreurs_par_place() -> None:
    """Les places sont mutuellement exclusives, donc corrélées NÉGATIVEMENT.

    Sommer les erreurs par place (ou même les sommer en quadrature) surestime
    la largeur : le calcul juste passe par la variance de la variable « gain
    encaissé », pas par une agrégation des variances marginales. Ce test
    épingle l'écart pour qu'une « simplification » future ne le rétablisse
    pas en silence.
    """
    n_sims = 40_000
    pay = np.asarray(PAYOUTS_9)
    probs, err_places = _finish_probs_mc(TABLE_9, len(PAYOUTS_9), n_sims, 0)
    juste = erreur_type_exacte(probs, pay, n_sims)
    quadrature = np.sqrt(((err_places * pay[None, :]) ** 2).sum(axis=1))
    assert np.all(quadrature > juste)
    assert float((quadrature / juste).max()) > 1.5


def test_la_variance_negative_d_arrondi_ne_produit_pas_de_nan() -> None:
    """Cas dégénéré : une seule place possible, variance nulle en théorie.

    ``probs @ pay**2 - (probs @ pay)**2`` peut sortir à −1e-18 ; sans le
    plancher à zéro, la racine rendrait `nan` et l'intervalle deviendrait
    inexploitable au lieu de devenir nul.
    """
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    pay = np.array([1e8, 1e8 - 1.0])
    err = erreur_type_exacte(probs, pay, 1000)
    assert np.all(np.isfinite(err))
    assert np.all(err >= 0.0)


def test_erreur_type_exacte_refuse_zero_tirage() -> None:
    with pytest.raises(IcmError):
        erreur_type_exacte(np.array([[1.0]]), np.array([100.0]), 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. L'API PUBLIQUE — l'incertitude remonte enfin
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", CAS)
def test_les_ev_mesures_sont_ceux_de_l_api_historique(stacks, payouts) -> None:
    """Ajouter l'incertitude ne doit RIEN changer aux chiffres."""
    m = icm_equities_mesurees(stacks, payouts)
    np.testing.assert_allclose(m.equities, icm_equities(stacks, payouts),
                               rtol=0, atol=0)


@pytest.mark.parametrize("stacks,payouts", CAS)
def test_le_chemin_exact_annonce_une_erreur_exactement_nulle(
    stacks, payouts
) -> None:
    """Pas « petite » : nulle. La récurrence de Harville ne tire rien au sort."""
    m = icm_equities_mesurees(stacks, payouts)
    assert m.exact is True
    assert m.n_sims == 0
    assert float(m.erreur_type.max()) == 0.0
    assert float(m.erreur_places.max()) == 0.0
    assert float(m.demi_largeur_95.max()) == 0.0


def test_au_dela_de_la_limite_exacte_l_erreur_devient_positive() -> None:
    """13 joueurs : Monte-Carlo, donc incertitude non nulle et rapportée.

    C'est le test qui tombe si quelqu'un rétablit le ``probs, _ = ...``.
    """
    n = EXACT_LIMIT + 1
    stacks = tuple(1000.0 + 200.0 * i for i in range(n))
    payouts = (100.0, 60.0, 40.0, 25.0, 15.0)
    m = icm_equities_mesurees(stacks, payouts, n_sims=20_000, seed=0)

    assert m.exact is False
    assert m.n_sims == 20_000
    assert m.erreur_type.shape == (n,)
    assert m.erreur_places.shape == (n, len(payouts))
    assert float(m.erreur_type.min()) > 0.0
    np.testing.assert_allclose(m.demi_largeur_95, 1.959963984540054 * m.erreur_type)
    # Conservation : la dotation est distribuée, incertitude ou pas.
    assert float(m.equities.sum()) == pytest.approx(sum(payouts), abs=1e-9)


def test_l_erreur_decroit_en_racine_de_n() -> None:
    r"""σ ∝ 1/√n : quadrupler les tirages doit HALVER l'erreur.

    C'est la signature d'une erreur d'échantillonnage authentique. Une borne
    constante, ou une erreur bricolée, ne la respecterait pas.
    """
    n = EXACT_LIMIT + 1
    stacks = tuple(1000.0 + 200.0 * i for i in range(n))
    payouts = (100.0, 60.0, 40.0, 25.0, 15.0)
    petit = icm_equities_mesurees(stacks, payouts, n_sims=5_000, seed=0)
    grand = icm_equities_mesurees(stacks, payouts, n_sims=20_000, seed=0)
    rapport = petit.erreur_type / grand.erreur_type
    assert float(rapport.min()) > 1.8
    assert float(rapport.max()) < 2.2


def test_un_joueur_a_tapis_nul_ne_porte_aucune_incertitude() -> None:
    """Sa place vient d'une convention, pas d'un tirage : erreur-type nulle.

    Et c'est une affirmation vérifiable, pas une commodité : `icm_equities`
    sert les morts par la branche de partage, qui n'appelle aucun générateur.
    """
    n = EXACT_LIMIT + 2
    stacks = [1000.0 + 100.0 * i for i in range(n)]
    stacks[3] = 0.0
    payouts = (100.0, 60.0, 40.0, 25.0, 15.0)
    m = icm_equities_mesurees(stacks, payouts, n_sims=8_000, seed=0)
    assert float(m.erreur_type[3]) == 0.0
    assert float(np.delete(m.erreur_type, 3).min()) > 0.0


def test_sans_gain_a_distribuer_tout_est_nul_et_exact() -> None:
    m = icm_equities_mesurees([100.0, 50.0], [])
    assert m.exact is True
    assert float(np.abs(m.equities).max()) == 0.0
    assert m.erreur_places.shape == (2, 0)
