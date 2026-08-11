"""Les invariants de l'ICM — ce que le calcul doit préserver, quoi qu'on lui donne.

Pourquoi ce fichier existe
--------------------------
Les cinq défauts graves trouvés récemment étaient tous AUX JOINTURES, chaque
composant étant juste isolément. Un invariant est le seul test qui n'a pas
besoin de connaître la bonne réponse : il énonce une propriété que le
résultat doit vérifier QUELLE QUE SOIT l'entrée. C'est ce qui attrape les
défauts qu'aucune relecture ne voit.

Deux invariants manquaient, et l'un des deux aurait signalé directement le
contournement dit « de l'amputation » (un tapis rogné pour éviter une
division par zéro) :

* **échelle** : multiplier tous les tapis par une constante ne change RIEN
  aux équités. Tout calcul accroché à la VALEUR ABSOLUE des jetons, plutôt
  qu'à leurs proportions, le viole immédiatement.
* **permutation** : permuter deux joueurs permute leurs équités, et ne
  touche à rien d'autre.

Ce que ces tests refusent d'être
--------------------------------
Tautologiques. Deux gabarits de cartes intervertis ont survécu à 1057 tests
parce que chacun se comparait à lui-même. Ici, chaque invariant est
accompagné d'un test « à dents » qui vérifie qu'une implémentation FAUSSE le
ferait bien échouer — sans quoi un invariant vert ne prouve rien.

Les chiffres cités viennent tous de ``banc_invariants_icm.py``, rejouable :

    python banc_invariants_icm.py --large --seuils
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pfs.core.icm import (
    IcmError,
    PkoSpot,
    _finish_probs_exact,
    _finish_probs_mc,
    analyse_pko_spot,
    bounty_capture_value,
    bubble_factor,
    icm_equities,
    spot_pko_face_a_tapis,
)

# ── Tables de référence ────────────────────────────────────────────────────

PAYOUTS_9 = (176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34)
TABLE_9 = (35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
PRIMES_9 = (8.13, 8.97, 7.43, 4.17, 6.0, 6.0, 6.0, 6.0, 6.0)

FACTEURS = (1e-3, 1.0, 1e3, 1e6)
"""1e-3 ≈ compter en milliers de jetons, 1e6 ≈ en millionièmes.
L'ICM ne doit pas s'en apercevoir."""

TABLES = [
    pytest.param(TABLE_9, PAYOUTS_9, id="table-PMU-9-joueurs"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="bulle-4-joueurs-3-gains"),
    pytest.param((1000.0, 1000.0, 1000.0), (50.0, 30.0, 20.0), id="ex-aequo"),
    pytest.param((0.0, 10.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0),
                 id="un-tapis-nul"),
    pytest.param((0.0, 0.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0),
                 id="deux-tapis-nuls"),
    pytest.param((6000.0, 3000.0, 1000.0), (100.0,), id="gain-unique"),
    pytest.param((900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0),
                 id="satellite-3-places-egales-sur-5"),
    pytest.param((7000.0, 3000.0), (60.0, 40.0), id="heads-up"),
    pytest.param((100.0, 50.0), (50.0, 30.0, 20.0),
                 id="plus-de-gains-que-de-joueurs"),
    pytest.param((100.0,), (50.0, 30.0, 20.0), id="joueur-unique"),
]


def _dotation_attribuable(stacks, payouts) -> float:
    """La dotation réellement distribuable : les gains sont TRONQUÉS.

    C'est la définition qui décide du résultat du test de conservation. Une
    3ᵉ place n'existe pas à deux joueurs : `_validate` coupe les gains à
    ``len(stacks)``, et la somme des équités vaut donc
    ``sum(payouts[:n])`` — pas ``sum(payouts)``.
    """
    return float(sum(payouts[: len(stacks)]))


# ═══════════════════════════════════════════════════════════════════════════
# 1 — INVARIANCE D'ÉCHELLE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", TABLES)
@pytest.mark.parametrize("k", FACTEURS)
def test_l_echelle_ne_change_rien_aux_equites(stacks, payouts, k) -> None:
    """Les jetons n'ont pas d'unité : seules leurs PROPORTIONS comptent.

    C'est l'invariant qui aurait signalé l'amputation du tapis : rogner un
    tapis d'une quantité absolue, c'est faire dépendre le résultat de
    l'échelle. Mesuré : écart maximal 2,1e-14 sur les tables de référence.
    """
    ref = icm_equities(stacks, payouts)
    got = icm_equities([s * k for s in stacks], payouts)
    np.testing.assert_allclose(
        got, ref, atol=1e-11 * max(_dotation_attribuable(stacks, payouts), 1.0),
        err_msg=f"les équités dépendent de l'unité des jetons (k={k})")


@pytest.mark.parametrize("k", FACTEURS)
def test_l_echelle_ne_change_rien_au_bubble_factor(k) -> None:
    """Le facteur de bulle est un RAPPORT de différences de $EV : sans unité."""
    ref = bubble_factor(TABLE_9, PAYOUTS_9, hero=0, villain=1)
    got = bubble_factor([s * k for s in TABLE_9], PAYOUTS_9, hero=0, villain=1)
    assert got == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("k", FACTEURS)
def test_le_chemin_monte_carlo_est_aussi_invariant_d_echelle(k) -> None:
    """Au-delà de 12 joueurs le calcul bascule en Monte-Carlo.

    Les tirages y sont proportionnels aux tapis restants, donc à graine fixée
    l'invariance doit être EXACTE, au bit près — pas seulement à la tolérance
    flottante. Un écart non nul signalerait un seuil absolu caché dans le
    tirage.
    """
    stacks = [float(x) for x in range(1000, 14000, 1000)]   # 13 joueurs
    payouts = [50.0, 30.0, 20.0]
    ref = icm_equities(stacks, payouts, n_sims=5_000, seed=3)
    got = icm_equities([s * k for s in stacks], payouts, n_sims=5_000, seed=3)
    np.testing.assert_array_equal(got, ref)


def test_l_invariant_d_echelle_a_des_dents() -> None:
    """Un invariant qu'aucune implémentation fausse ne fait échouer ne prouve rien.

    On rejoue ici la forme naturelle du contournement historique : un
    PLANCHER ABSOLU sur les tapis, pour éviter la division par zéro. C'est
    exactement le genre de correctif qui paraît inoffensif et qui accroche le
    calcul à la valeur absolue des jetons — à petite échelle, le plancher
    cesse d'être négligeable et devient un tapis à part entière.

    Ce test échouerait si l'invariant d'échelle était vérifié trop mollement.
    """
    stacks = (0.0, 10.0, 20.0, 30.0)
    payouts = (100.0, 50.0, 25.0, 10.0)

    def icm_avec_plancher(s, k: float):
        # « pour ne pas diviser par zéro, on met un epsilon » — le réflexe.
        return icm_equities([max(x * k, 1e-9) for x in s], payouts)

    ref = icm_avec_plancher(stacks, 1.0)
    degrade = icm_avec_plancher(stacks, 1e-9)
    ecart = float(np.max(np.abs(degrade - ref)))
    assert ecart > 1.0, (
        "le plancher absolu devrait rendre le résultat dépendant de l'échelle ; "
        f"écart observé {ecart:.3e}. Si ce test passe alors que le suivant "
        "échoue, c'est la tolérance de l'invariant qu'il faut resserrer.")

    # …et l'implémentation réelle, elle, ne bouge pas.
    vrai = icm_equities([x * 1e-9 for x in stacks], payouts)
    np.testing.assert_allclose(vrai, icm_equities(stacks, payouts), atol=1e-11)


# ═══════════════════════════════════════════════════════════════════════════
# 2 — INVARIANCE PAR PERMUTATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", TABLES)
def test_permuter_deux_joueurs_permute_leurs_equites(stacks, payouts) -> None:
    """Aucun siège n'est privilégié : l'indice d'un joueur ne porte rien.

    Un calcul qui trierait implicitement les tapis, ou qui indexerait les
    gains sur la position à la table plutôt que sur le classement, violerait
    cet invariant.
    """
    n = len(stacks)
    if n < 2:
        pytest.skip("permutation sans objet à un seul joueur")
    ref = icm_equities(stacks, payouts)
    rng = random.Random(20260811)
    for _ in range(20):
        perm = list(range(n))
        rng.shuffle(perm)
        got = icm_equities([stacks[i] for i in perm], payouts)
        np.testing.assert_allclose(got, ref[perm], atol=1e-11)


def test_le_bubble_factor_suit_ses_deux_joueurs() -> None:
    """Renuméroter la table déplace hero et villain, sans changer le nombre."""
    rng = random.Random(55)
    for _ in range(40):
        n = rng.randint(3, 7)
        stacks = [rng.uniform(1, 500) for _ in range(n)]
        m = rng.randint(2, n)
        payouts = sorted((rng.uniform(1, 100) for _ in range(m)), reverse=True)
        h, v = rng.sample(range(n), 2)
        ref = bubble_factor(stacks, payouts, h, v)
        perm = list(range(n))
        rng.shuffle(perm)
        got = bubble_factor([stacks[i] for i in perm], payouts,
                            perm.index(h), perm.index(v))
        assert got == pytest.approx(ref, rel=1e-9)


def test_la_permutation_ne_change_rien_a_l_analyse_pko() -> None:
    """Les primes doivent suivre leurs joueurs, comme les tapis."""
    spot = spot_pko_face_a_tapis(TABLE_9, PAYOUTS_9, PRIMES_9,
                                 hero=0, villain=3,
                                 blindes_mortes=1.4, deja_engage_hero=1.0)
    base = analyse_pko_spot(spot)
    rng = random.Random(7)
    for _ in range(10):
        perm = list(range(len(spot.stacks)))
        rng.shuffle(perm)
        a = analyse_pko_spot(PkoSpot(
            stacks=tuple(spot.stacks[i] for i in perm),
            payouts=spot.payouts,
            bounties=tuple(spot.bounties[i] for i in perm),
            hero=perm.index(spot.hero), villain=perm.index(spot.villain),
            pot=spot.pot, bet=spot.bet))
        assert a.villain_eliminated == base.villain_eliminated
        assert a.bounty_value == pytest.approx(base.bounty_value, abs=1e-9)
        assert a.required_no_bounty == pytest.approx(base.required_no_bounty,
                                                     abs=1e-9)
        assert a.required_with_bounty == pytest.approx(base.required_with_bounty,
                                                       abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# 3 — CONSERVATION DE LA DOTATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", TABLES)
def test_la_dotation_est_conservee(stacks, payouts) -> None:
    """Rien ne se crée, rien ne se perd — dans TOUTES les configurations.

    Y compris celles qui sortent du cadre nominal : tapis nuls, ex æquo,
    joueur unique, plus de gains que de joueurs, plus de joueurs que de
    gains. C'est là que les branches spéciales vivent, et c'est là qu'elles
    se trompent.
    """
    eq = icm_equities(stacks, payouts)
    attendu = _dotation_attribuable(stacks, payouts)
    assert float(eq.sum()) == pytest.approx(attendu, rel=1e-9)
    assert all(float(x) >= -1e-12 for x in eq), "aucune équité négative"


def test_les_gains_sont_tronques_au_nombre_de_joueurs() -> None:
    """La dotation de référence est la dotation ATTRIBUABLE, et c'est explicite.

    À deux joueurs, la 3ᵉ place n'existe pas : son gain n'est versé à
    personne. Écrire le test contre ``sum(payouts)`` le ferait échouer pour
    une raison qui n'est pas un défaut — d'où cette convention, épinglée ici
    plutôt que supposée.
    """
    eq = icm_equities([100.0, 50.0], [50.0, 30.0, 20.0])
    assert float(eq.sum()) == pytest.approx(80.0)     # 50 + 30, pas 100
    eq1 = icm_equities([100.0], [50.0, 30.0, 20.0])
    assert float(eq1.sum()) == pytest.approx(50.0)


def test_conservation_sur_un_balayage_avec_beaucoup_de_tapis_nuls() -> None:
    """Le tapis nul est le cas qui a motivé le contournement : on l'inonde."""
    rng = random.Random(31337)
    pire = 0.0
    for _ in range(300):
        n = rng.randint(1, 9)
        stacks = [0.0 if rng.random() < 0.35 else rng.uniform(1, 1000)
                  for _ in range(n)]
        if sum(stacks) <= 0:
            continue
        m = rng.randint(1, n + 3)
        payouts = sorted((rng.uniform(1, 200) for _ in range(m)), reverse=True)
        eq = icm_equities(stacks, payouts)
        attendu = sum(payouts[:n])
        pire = max(pire, abs(float(eq.sum()) - attendu) / attendu)
    assert pire < 1e-9, f"écart relatif maximal {pire:.3e}"


def test_une_table_sans_aucun_jeton_est_refusee() -> None:
    """Le comportement RÉEL, épinglé — parce qu'il ne l'était pas.

    `icm_equities` portait une branche « personne n'a de jeton : la dotation
    entière se partage » qui était INATTEIGNABLE : les tapis étant validés
    positifs ou nuls, « aucun vivant » implique une somme nulle, que
    `_validate` refuse d'abord. Un lecteur pouvait croire le cas traité. La
    branche a été retirée ; voici ce que l'appel fait vraiment.
    """
    with pytest.raises(IcmError, match="somme des stacks nulle"):
        icm_equities([0.0, 0.0], [100.0, 50.0])


def test_les_tapis_nuls_se_partagent_les_dernieres_places_a_egalite() -> None:
    """Corollaire de la conservation ET de la permutation.

    Deux joueurs déjà éliminés ne sont pas départageables : leur donner des
    équités différentes violerait la symétrie, leur donner zéro violerait la
    conservation.
    """
    eq = icm_equities([0.0, 0.0, 20.0, 30.0], [100.0, 50.0, 25.0, 10.0])
    assert float(eq[0]) == pytest.approx(float(eq[1]))
    assert float(eq[0] + eq[1]) == pytest.approx(25.0 + 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# 4 — MONOTONIE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", TABLES)
def test_gagner_des_jetons_ne_peut_pas_faire_perdre_de_l_equite(
        stacks, payouts) -> None:
    """À structure fixée, l'équité est croissante en son propre tapis.

    Deux façons de grossir, toutes deux testées : prendre les jetons d'un
    autre joueur (le total est conservé) ou en recevoir de nulle part (le
    total augmente). La première est la seule qui arrive à une table ; la
    seconde attrape les calculs qui normaliseraient mal.
    """
    n = len(stacks)
    if n < 2:
        pytest.skip("monotonie sans objet à un seul joueur")
    for hero in range(n):
        for autre in range(n):
            if autre == hero or stacks[autre] <= 0:
                continue
            prec = None
            for frac in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
                s = list(stacks)
                d = stacks[autre] * frac
                s[hero] += d
                s[autre] -= d
                v = float(icm_equities(s, payouts)[hero])
                if prec is not None:
                    assert v >= prec - 1e-9, (
                        f"joueur {hero} recule en prenant {d:.4f} jetons "
                        f"au joueur {autre} : {prec:.6f} → {v:.6f}")
                prec = v
        prec = None
        for d in (0.0, 1.0, 10.0, 100.0, 1e4):
            s = list(stacks)
            s[hero] += d
            v = float(icm_equities(s, payouts)[hero])
            if prec is not None:
                assert v >= prec - 1e-9, (
                    f"joueur {hero} recule en recevant {d} jetons : "
                    f"{prec:.6f} → {v:.6f}")
            prec = v


def test_la_monotonie_a_des_dents() -> None:
    """L'équité doit VRAIMENT bouger : un calcul constant serait monotone aussi.

    Sans cette vérification, `lambda *_: 0.0` passerait le test précédent.
    """
    stacks = [5000.0, 3000.0, 1500.0, 500.0]
    payouts = [50.0, 30.0, 20.0]
    petit = float(icm_equities(stacks, payouts)[3])
    gros = list(stacks)
    gros[3] += 4000.0
    gros[0] -= 4000.0
    assert float(icm_equities(gros, payouts)[3]) > petit + 5.0


# ═══════════════════════════════════════════════════════════════════════════
# 5 — NON-LINÉARITÉ (le fait fondateur de l'ICM)
# ═══════════════════════════════════════════════════════════════════════════


def _part_de_jetons(stacks, payouts) -> np.ndarray:
    """Ce que vaudrait chaque tapis si les jetons étaient linéaires (chipEV)."""
    pool = _dotation_attribuable(stacks, payouts)
    total = float(sum(stacks))
    return np.array([s / total * pool for s in stacks], dtype=np.float64)


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param(TABLE_9, PAYOUTS_9, id="table-PMU-9-joueurs"),
    pytest.param((6000.0, 3000.0, 1000.0), (50.0, 30.0, 20.0), id="3-joueurs"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="bulle-4-3"),
    pytest.param((7000.0, 3000.0), (60.0, 40.0), id="heads-up-2-gains"),
    pytest.param((900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0),
                 id="satellite-3-places-egales-sur-5"),
])
def test_le_gros_tapis_est_decote_et_le_petit_surcote(stacks, payouts) -> None:
    """Le fait fondateur : doubler son tapis ne double pas son espérance.

    L'affirmation porte sur les EXTRÊMES seulement — le plus gros et le plus
    petit tapis. Voir `test_les_tapis_intermediaires_ne_sont_pas_contraints`
    pour le contre-exemple qui interdit de la généraliser.

    Le satellite est dans la liste exprès : k places de valeur ÉGALE reste
    fortement non linéaire, ce sont les places NON payées qui font la
    courbure. Le classifier d'un banc s'y est trompé.
    """
    ecarts = icm_equities(stacks, payouts) - _part_de_jetons(stacks, payouts)
    gros, petit = int(np.argmax(stacks)), int(np.argmin(stacks))
    assert float(ecarts[gros]) < 0.0, (
        f"le plus gros tapis devrait valoir MOINS que sa part de jetons "
        f"(écart {float(ecarts[gros]):+.4f})")
    assert float(ecarts[petit]) > 0.0, (
        f"le plus petit tapis devrait valoir PLUS que sa part de jetons "
        f"(écart {float(ecarts[petit]):+.4f})")


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param((6000.0, 3000.0, 1000.0), (100.0,), id="winner-take-all-3"),
    pytest.param((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (50.0,), id="winner-take-all-6"),
    pytest.param((6000.0, 3000.0, 1000.0), (100.0, 0.0, 0.0),
                 id="une-seule-place-payee"),
])
def test_une_seule_place_payee_rend_l_icm_exactement_lineaire(
        stacks, payouts) -> None:
    """La frontière du phénomène, et elle est nette.

    Avec une seule place payée, P(1ᵉʳ) = s/S sous Harville, donc l'équité
    vaut EXACTEMENT la part de jetons. Sans cette borne, « le gros tapis est
    décoté » serait affirmé là où c'est faux.
    """
    np.testing.assert_allclose(icm_equities(stacks, payouts),
                               _part_de_jetons(stacks, payouts), atol=1e-11)


def test_des_tapis_egaux_valent_exactement_la_moyenne() -> None:
    """L'autre frontière : sans inégalité de tapis, pas de courbure visible."""
    eq = icm_equities([1000.0] * 5, [50.0, 30.0, 20.0])
    np.testing.assert_allclose(eq, [100.0 / 5] * 5, atol=1e-11)


def test_les_tapis_intermediaires_ne_sont_pas_contraints() -> None:
    """Contre-exemple explicite, pour interdire la généralisation.

    Sur six tapis régulièrement espacés et quatre gains, les écarts changent
    de signe au milieu du classement : trois décotés, trois surcotés. Écrire
    « tout tapis au-dessus de la moyenne est décoté » serait donc faux, et
    un test qui l'affirmerait épinglerait un comportement inexistant.
    """
    stacks = (100.0, 90.0, 80.0, 70.0, 60.0, 50.0)
    payouts = (40.0, 30.0, 20.0, 10.0)
    ecarts = icm_equities(stacks, payouts) - _part_de_jetons(stacks, payouts)
    signes = np.sign(np.round(ecarts, 9))
    assert set(signes.tolist()) == {-1.0, 1.0}
    assert signes.tolist() == [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# 6 — CONCORDANCE EXACT ↔ MONTE-CARLO
# ═══════════════════════════════════════════════════════════════════════════

N_SIMS = 40_000
"""Assez de tirages pour que la bande à 5 σ reste étroite (< 1 % de l'équité,
vérifié plus bas), assez peu pour que la suite reste rapide."""


def _sigma_exacte(probs: np.ndarray, payouts: np.ndarray,
                  n_sims: int) -> np.ndarray:
    r"""Erreur-type EXACTE de l'estimateur Monte-Carlo, joueur par joueur.

    Le tirage répète ``n_sims`` fois la variable aléatoire « gain encaissé
    par le joueur *i* » (zéro hors des places payées). Sa loi est donnée par
    les probabilités exactes de Harville, donc sa variance aussi :

    .. math:: \operatorname{Var}(X_i) = \sum_k p_{ik}\pi_k^2
              - \Bigl(\sum_k p_{ik}\pi_k\Bigr)^2

    d'où σ = √(Var/n_sims). C'est la borne à laquelle comparer l'écart —
    CALCULÉE, pas supposée. La valeur que renvoie `_finish_probs_mc`
    (√(0,25/n)) borne une proportion, pas un gain : elle ne convient pas ici.
    """
    esp = probs @ payouts
    var = probs @ (payouts ** 2) - esp ** 2
    return np.sqrt(np.maximum(var, 0.0) / n_sims)


CAS_MC = [
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="bulle-4-3"),
    pytest.param((100.0, 100.0, 100.0), (50.0, 30.0, 20.0), id="ex-aequo-3"),
    pytest.param((176.0, 124.0, 92.0, 70.0, 55.0, 44.0),
                 (100.0, 60.0, 40.0, 25.0), id="6-joueurs-4-gains"),
    pytest.param((9000.0, 500.0, 300.0, 200.0), (70.0, 30.0),
                 id="chipleader-ecrasant"),
]


@pytest.mark.parametrize("stacks,payouts", CAS_MC)
@pytest.mark.parametrize("seed", [0, 1])
def test_le_monte_carlo_concorde_avec_l_exact(stacks, payouts, seed) -> None:
    """Deux chemins de calcul, une seule vérité — à l'échantillonnage près.

    L'écart est mesuré EN UNITÉS DE σ, et σ est calculée depuis les
    probabilités exactes. Sur le banc (5 tables × 6 graines × 2 tailles), le
    |z| maximal observé vaut 2,85 σ ; le seuil est posé à 5 σ, soit plus de
    deux σ de marge. Un seuil « à vue » aurait été soit inutile, soit
    fragile.
    """
    pay = np.asarray(payouts, dtype=np.float64)
    exact = _finish_probs_exact(tuple(stacks), len(payouts))
    sigma = _sigma_exacte(exact, pay, N_SIMS)
    mc, _ = _finish_probs_mc(tuple(stacks), len(payouts), N_SIMS, seed)

    eq_exact = exact @ pay
    eq_mc = mc @ pay
    z = np.abs(eq_mc - eq_exact) / np.where(sigma > 0, sigma, np.inf)
    assert float(np.max(z)) <= 5.0, (
        f"écart {float(np.max(np.abs(eq_mc - eq_exact))):.4f} $ = "
        f"{float(np.max(z)):.2f} σ — au-delà de l'échantillonnage")


@pytest.mark.parametrize("stacks,payouts", CAS_MC)
def test_la_concordance_certifie_une_precision_connue(stacks, payouts) -> None:
    """Ce que le test précédent prouve — et ce qu'il ne prouve PAS.

    « L'écart tient dans l'erreur d'échantillonnage » est une phrase vide si
    cette erreur est énorme. On chiffre donc la bande, et on énonce sa
    portée exacte plutôt que de la supposer étroite.

    Mesuré à ``N_SIMS`` = 40 000 tirages, sur les quatre tables de `CAS_MC` :

    * en absolu, la bande à 5 σ vaut au plus **0,48 % de la dotation**. Un
      ICM faux de plus de 1 % du prize pool serait donc attrapé.
    * par joueur, en relatif, elle monte à **5,35 %** — sur le tapis le plus
      court, dont l'équité est petite et la variance comparativement grande.

    Autrement dit : ce test ne certifie PAS les équités des micro-tapis à
    mieux que ~5 % relatifs. Les affiner demanderait 25 fois plus de
    tirages (σ ∝ 1/√n) pour un gain sans rapport avec ce qu'on cherche ici,
    qui est la concordance des deux chemins de calcul. La borne est écrite
    pour être relue, pas pour flatter.
    """
    pay = np.asarray(payouts, dtype=np.float64)
    exact = _finish_probs_exact(tuple(stacks), len(payouts))
    eq = exact @ pay
    bande = 5.0 * _sigma_exacte(exact, pay, N_SIMS)
    pool = _dotation_attribuable(stacks, payouts)

    part_dotation = float(np.max(bande)) / pool
    assert part_dotation < 0.005, (
        f"bande à 5 σ = {part_dotation:.3%} de la dotation — la concordance "
        "ne certifierait plus rien d'utile")

    # La limite, épinglée : si elle se dégradait, ce test le dirait.
    pire_relatif = float(np.max(bande / eq))
    assert pire_relatif < 0.06, (
        f"bande à 5 σ = {pire_relatif:.2%} de l'équité du joueur le plus "
        "court — au-delà, augmenter N_SIMS plutôt que relâcher la borne")


def test_le_monte_carlo_prend_le_relais_au_dela_de_douze_joueurs() -> None:
    """La bascule exacte → Monte-Carlo ne doit rien casser de la conservation."""
    stacks = [float(x) for x in range(1000, 14000, 1000)]   # 13 joueurs
    eq = icm_equities(stacks, [50.0, 30.0, 20.0], n_sims=20_000, seed=7)
    assert float(eq.sum()) == pytest.approx(100.0, rel=1e-9)
    assert all(eq[i] <= eq[i + 1] + 1e-9 for i in range(12)), (
        "les $EV doivent rester ordonnés comme les tapis")


# ═══════════════════════════════════════════════════════════════════════════
# 7 — BUBBLE FACTOR
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param((6000.0, 4000.0), (100.0,), id="hu-1-gain"),
    pytest.param((6000.0, 4000.0), (60.0, 40.0), id="hu-2-gains"),
    pytest.param((100.0, 100.0), (70.0, 30.0), id="hu-tapis-egaux"),
    pytest.param((1000.0, 1.0), (60.0, 40.0), id="hu-tres-desequilibre"),
])
def test_en_heads_up_le_bubble_factor_vaut_exactement_un(stacks, payouts) -> None:
    """À deux, l'ICM est AFFINE en jetons : aucune pression de bulle.

    Chaque joueur touche ``π₂ + (π₁−π₂)·s/S`` — une droite. Ce que l'un gagne
    en $ quand il gagne des jetons, il le perd exactement quand il en perd.

    L'égalité tient à 1e-9 près jusqu'à des ratios de tapis de 1e6 ; au-delà
    c'est la SOUSTRACTION de $EV proches qui perd ses chiffres, pas l'ICM
    (banc : |BF−1| = 3,6e-7 au ratio 1:1e9). La borne est posée en
    conséquence.
    """
    for h, v in ((0, 1), (1, 0)):
        assert bubble_factor(stacks, payouts, h, v) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param((100.0, 200.0, 300.0), (50.0,), id="3-joueurs"),
    pytest.param((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (50.0,), id="6-joueurs"),
])
def test_un_gain_unique_annule_aussi_la_pression_de_bulle(stacks, payouts) -> None:
    """L'autre cas affine : sans deuxième place, les jetons redeviennent linéaires.

    Il compte, parce que « plus d'un joueur ⇒ BF > 1 » est faux, et qu'un
    test qui l'affirmerait épinglerait un comportement inexistant.
    """
    assert bubble_factor(stacks, payouts, 0, 1) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param(TABLE_9, PAYOUTS_9, id="table-PMU-9-joueurs"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="bulle-4-3"),
    pytest.param((900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0),
                 id="satellite"),
])
def test_le_bubble_factor_depasse_un_des_qu_il_y_a_plusieurs_gains(
        stacks, payouts) -> None:
    """Dès qu'une place suivante existe, risquer coûte plus que gagner ne rapporte."""
    assert bubble_factor(stacks, payouts, hero=0, villain=1) > 1.0


def test_le_bubble_factor_depasse_un_sur_un_balayage() -> None:
    """La propriété doit tenir partout, pas sur trois tables choisies.

    Banc (2000 tirages) : minimum observé 1,0012.
    """
    rng = random.Random(1234)
    mini = math.inf
    for _ in range(300):
        n = rng.randint(3, 8)
        stacks = [rng.uniform(1, 1000) for _ in range(n)]
        m = rng.randint(2, n)
        payouts = sorted({round(rng.uniform(1, 100), 3) for _ in range(m)},
                         reverse=True)
        if len(payouts) < 2:
            continue
        h, v = rng.sample(range(n), 2)
        mini = min(mini, bubble_factor(stacks, payouts, h, v))
    assert mini > 1.0, f"facteur de bulle minimal observé : {mini!r}"


def test_le_bubble_factor_croit_avec_le_tapis_du_vilain() -> None:
    """Plus l'adversaire nous couvre, plus le call devient cher.

    Le tapis du vilain est gonflé aux dépens de deux joueurs neutres, à
    total CONSTANT : sinon on mesurerait l'effet du total, pas celui du
    vilain.
    """
    prec = None
    for v_s in (500.0, 1000.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0):
        neutre = 12000.0 - 3000.0 - v_s
        s = [3000.0, v_s, neutre / 2, neutre / 2]
        bf = bubble_factor(s, [50.0, 30.0, 20.0], hero=0, villain=1)
        if prec is not None:
            assert bf > prec, (
                f"la pression recule quand le vilain grossit à {v_s:.0f} : "
                f"{prec:.6f} → {bf:.6f}")
        prec = bf


@pytest.mark.parametrize("stacks,payouts", [
    pytest.param((1.0, 2.0, 3.0), (1.0, 1.0, 1.0), id="3-joueurs-3-gains-egaux"),
    pytest.param((100.0, 200.0, 300.0), (50.0, 50.0, 50.0), id="variante-echelle"),
    pytest.param((10.0, 20.0, 30.0, 40.0), (7.0, 7.0, 7.0, 7.0), id="4-joueurs"),
    pytest.param((100.0, 100.0), (50.0, 50.0), id="heads-up"),
])
def test_une_structure_degeneree_donne_l_infini_jamais_un_nombre_negatif(
        stacks, payouts) -> None:
    """Régression : `bubble_factor` renvoyait **−1,0**.

    Quand tous les joueurs encore en lice touchent le même gain (fin de
    satellite : les places restantes sont équivalentes), déplacer des jetons
    ne change aucune équité. Le rapport est un 0/0 : `gain` et `loss` valent
    zéro EN THÉORIE et ±1 ulp en pratique.

    Le garde-fou comparait `gain` à zéro exactement, donc laissait passer le
    résidu — et le MÊME cas dégénéré rendait deux réponses opposées au gré de
    l'arrondi : ``bubble_factor([1, 2, 3], [1, 1, 1], 0, 1)`` valait −1,0
    (facteur de bulle NÉGATIF, qui faisait ensuite échouer `analyse_icm_spot`
    sur le message trompeur « bubble factor doit être > 0 »), tandis que
    ``([10, 20, 30, 40], [7, 7, 7, 7])`` valait déjà +inf.

    La bonne réponse est +inf : gagner des jetons ne rapporte rien, donc
    aucune équité ne justifie de risquer.
    """
    bf = bubble_factor(stacks, payouts, hero=0, villain=1)
    assert bf > 0.0, f"un facteur de bulle négatif n'a pas de sens : {bf!r}"
    assert math.isinf(bf)


def test_le_bubble_factor_n_est_jamais_negatif() -> None:
    """La propriété générale derrière la régression ci-dessus."""
    rng = random.Random(4242)
    for _ in range(400):
        n = rng.randint(2, 7)
        stacks = [rng.uniform(1, 1000) for _ in range(n)]
        k = rng.randint(1, n)
        # Les égalités de gains sont AUTORISÉES : c'est ce qui déclenche le 0/0.
        payouts = sorted((round(rng.uniform(1, 100), 2) for _ in range(k)),
                         reverse=True)
        h, v = rng.sample(range(n), 2)
        bf = bubble_factor(stacks, payouts, h, v)
        assert bf > 0.0, f"BF={bf!r} sur stacks={stacks} payouts={payouts}"


# ═══════════════════════════════════════════════════════════════════════════
# 8 — PKO : les mêmes invariants sur les primes
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("k", FACTEURS)
def test_la_valeur_d_une_prime_ne_depend_pas_de_l_unite_des_jetons(k) -> None:
    """Les tapis se comptent en jetons, les primes en euros : deux mondes.

    Changer l'unité des JETONS ne doit toucher ni la valeur capturable, ni
    l'équité exigée. Les tapis sont ici ceux qu'on lit à l'écran, passés par
    `spot_pko_face_a_tapis` — le chemin réel.
    """
    def analyser(facteur: float):
        spot = spot_pko_face_a_tapis(
            [t * facteur for t in TABLE_9], PAYOUTS_9, PRIMES_9,
            hero=0, villain=3,
            blindes_mortes=1.4 * facteur, deja_engage_hero=1.0 * facteur)
        return analyse_pko_spot(spot)

    ref, got = analyser(1.0), analyser(k)
    assert got.villain_eliminated is ref.villain_eliminated
    assert got.bounty_value == pytest.approx(ref.bounty_value, abs=1e-9)
    assert got.required_no_bounty == pytest.approx(ref.required_no_bounty,
                                                   abs=1e-9)
    assert got.required_with_bounty == pytest.approx(ref.required_with_bounty,
                                                     abs=1e-9)


@pytest.mark.parametrize("k", FACTEURS)
def test_bounty_capture_value_est_invariante_d_echelle(k) -> None:
    """Elle ne dépend que de P(finir 1ᵉʳ) = s/S — une proportion."""
    ref = bounty_capture_value([200.0, 0.0, 100.0], 0, 50.0)
    got = bounty_capture_value([200.0 * k, 0.0, 100.0 * k], 0, 50.0)
    assert got == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("k", [1.0, 1e-6, 1e-11, 1e-12, 1e-13, 1e-15])
def test_un_vilain_non_couvert_ne_devient_pas_eliminable_en_changeant_d_unite(
        k) -> None:
    """Régression : le test d'élimination utilisait un seuil ABSOLU.

    ``eliminated = s[villain] <= 1e-12`` compare des jetons à une constante,
    alors que l'ICM ne connaît que des proportions. Conséquence mesurée sur
    ce spot exact — héros 100, vilain 1 jeton restant, donc NON couvert :

    ======================  =========  ==========  ===============
    unité                   éliminé ?  prime       équité exigée
    ======================  =========  ==========  ===============
    1 (jetons)              non        0,00        60,0 %
    1e-11                   non        0,00        60,0 %
    1e-12                   **oui**    **41,61**   **40,0 %**
    ======================  =========  ==========  ===============

    Même table, même décision, vingt points d'équité d'écart — au seul motif
    du choix de l'unité. Le seuil est désormais relatif au total des tapis.
    """
    a = analyse_pko_spot(PkoSpot(
        stacks=(100.0 * k, 1.0 * k, 100.0 * k), payouts=(100.0, 50.0),
        bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
        pot=100.0 * k, bet=100.0 * k))
    assert not a.villain_eliminated, (
        "un vilain qui garde des jetons n'est pas éliminable, quelle que "
        "soit l'unité dans laquelle on les compte")
    assert a.bounty_value == pytest.approx(0.0)
    assert a.required_with_bounty == pytest.approx(a.required_no_bounty)


def test_un_vilain_reellement_a_tapis_reste_eliminable_a_toute_echelle() -> None:
    """Le pendant du test précédent : le seuil relatif ne doit rien perdre."""
    for k in (1e-3, 1.0, 1e3, 1e6):
        a = analyse_pko_spot(PkoSpot(
            stacks=(100.0 * k, 0.0, 100.0 * k), payouts=(100.0,),
            bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
            pot=100.0 * k, bet=100.0 * k))
        assert a.villain_eliminated
        assert a.bounty_value == pytest.approx(50.0 * (0.5 + 0.5 * 2 / 3))
        assert a.required_with_bounty == pytest.approx(4 / 13)


def test_l_invariant_d_echelle_pko_a_des_dents() -> None:
    """Vérifie que ce seuil-là est bien mesurable par l'invariant.

    On rejoue le seuil ABSOLU d'origine sur les mêmes entrées : il fait
    basculer le verdict entre deux unités. Si cette assertion cessait de
    tenir, c'est que le cas limite choisi ne mord plus, et les deux tests
    précédents deviendraient décoratifs.
    """
    def eliminated_seuil_absolu(stack_vilain: float) -> bool:
        return stack_vilain <= 1e-12          # l'écriture d'origine

    assert not eliminated_seuil_absolu(1.0 * 1.0)
    assert eliminated_seuil_absolu(1.0 * 1e-13), (
        "le seuil absolu doit bien confondre « 1 jeton » et « zéro » à petite "
        "échelle — c'est le défaut que l'invariant relatif corrige")
    # Le seuil relatif, lui, ne confond jamais : 1 jeton sur 201 reste 0,5 %.
    assert not (1.0 * 1e-13 <= 1e-12 * (201.0 * 1e-13))
