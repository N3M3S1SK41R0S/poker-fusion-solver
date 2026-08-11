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

Un troisième manquait encore, et c'est le seul qui regarde le facteur de bulle
DU JOUEUR QUI VA MOURIR (§ 9) :

* **continuité en zéro** : la branche « tapis nul » doit être la LIMITE de la
  récursion, pas une valeur voisine. Le facteur de bulle du joueur COUVERT
  met en rapport ``$EV_perte``, évaluée à tapis zéro donc dans la branche
  spéciale, et ``$EV_gain``, évaluée par la récursion ordinaire : deux
  RÉGIMES DE CODE différents. Un écart entre les deux est absorbé dans les
  équités — conservation, échelle, permutation, monotonie restent toutes
  vertes — mais AMPLIFIÉ par le quotient : mesuré ×11,9 à ×76,5 (§ 9).

Et quatre propriétés des GAINS n'étaient pas mesurées du tout (§ 10) :
translation, homogénéité, tapis tous égaux, dénominateurs nuls.

Ce que ces tests refusent d'être
--------------------------------
Tautologiques. Deux gabarits de cartes intervertis ont survécu à 1057 tests
parce que chacun se comparait à lui-même. Ici, chaque invariant est
accompagné d'un test « à dents » qui vérifie qu'une implémentation FAUSSE le
ferait bien échouer — sans quoi un invariant vert ne prouve rien.

Les chiffres cités viennent tous de ``banc_invariants_icm.py``, rejouable :

    python banc_invariants_icm.py --large --seuils
    python banc_invariants_icm.py --continuite
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pfs.core.icm import (
    IcmError,
    IcmSpot,
    PkoSpot,
    _finish_probs_exact,
    _finish_probs_mc,
    _validate,
    analyse_icm_spot,
    analyse_pko_spot,
    bounty_capture_value,
    bubble_factor,
    icm_dollar_ev,
    icm_equities,
    icm_required_equity,
    risk_premium,
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
    probabilités exactes. Sur le banc en mode ``--large`` (5 tables ×
    6 graines × 2 tailles = 60 estimations), le |z| maximal observé vaut
    **3,02 σ** ; le seuil est posé à 5 σ, soit près de deux σ de marge. Un
    seuil « à vue » aurait été soit inutile, soit fragile.
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


def test_la_limite_du_garde_fou_bubble_factor_est_connue_et_bornee() -> None:
    """La limite que le correctif introduit, épinglée plutôt que découverte.

    Le garde-fou anti-0/0 de `bubble_factor` compare le gain à 1e-12 fois
    l'échelle des $EV. Il mord donc un cas LÉGITIME : le heads-up dont les
    tapis sont dans un rapport ≳ 1e12, où le gain vrai (3,3e-13 en relatif)
    passe sous le seuil et où la fonction renvoie +inf au lieu de 1.

    Ce n'est pas une perte de précision cachée sous le tapis : à ce rapport,
    la soustraction de deux $EV voisins a déjà perdu tous ses chiffres. Le
    banc le montre — dès un rapport de 1e11 la valeur rendue vaut 0,99996 au
    lieu de 1. Répondre +inf (« je ne tranche pas, suppose la pression
    maximale ») est prudent ; rendre un nombre plausible et faux serait pire.

    Un tournoi réel ne dépasse pas un rapport de ~1e6, où l'égalité tient
    encore à 3,6e-10 près. La borne est donc six ordres de grandeur au-delà
    de tout usage — mais elle existe, et la voici.
    """
    for ratio in (1e6, 1e9, 1e11):
        assert bubble_factor([ratio, 1.0], [60.0, 40.0], 0, 1) < math.inf, (
            f"le garde-fou ne doit pas mordre au rapport 1:{ratio:.0e}")
    for ratio in (1e12, 1e15):
        assert math.isinf(bubble_factor([ratio, 1.0], [60.0, 40.0], 0, 1)), (
            f"au rapport 1:{ratio:.0e} le calcul n'a plus de chiffres "
            "significatifs : +inf est la réponse prudente")


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

    Banc en mode ``--large`` (2000 tirages) : minimum observé **1,000093**,
    sur une table à 3 joueurs. La marge au-dessus de 1 peut donc être
    très mince — d'où l'intérêt de balayer plutôt que de choisir.
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


# ═══════════════════════════════════════════════════════════════════════════
# 9 — CONTINUITÉ EN ZÉRO : la branche « tapis nul » est-elle la LIMITE ?
# ═══════════════════════════════════════════════════════════════════════════
#
# L'angle mort que les huit sections précédentes laissaient ouvert.
#
# `bubble_factor` calcule ($EV − $EV_perte)/($EV_gain − $EV). Pour le joueur
# COUVERT — le tapis court, celui pour qui ce nombre décide de tout —
# $EV_perte est évaluée À TAPIS ZÉRO, donc par la branche spéciale de
# `icm_equities`, tandis que $EV et $EV_gain passent par la récursion
# ordinaire. Le quotient met en rapport DEUX RÉGIMES DE CODE. (Pour le joueur
# COUVRANT, c'est $EV_gain qui traverse la branche : gagner, c'est éliminer.)
#
# Un écart entre la branche et la limite serait ABSORBÉ dans les équités : la
# somme reste conservée, l'échelle, la permutation et la monotonie restent
# vertes. Mais il est AMPLIFIÉ par le quotient — voir
# `test_une_erreur_d_equite_est_amplifiee_par_le_quotient`.
#
# RÉSULTAT DE LA MESURE : la branche et la limite COÏNCIDENT. Extrapolé en
# zéro, l'écart vaut au plus 2,0e-15 en relatif sur huit spots. Il n'y avait
# donc RIEN À CORRIGER dans `pfs/core/icm.py` — mais rien ne le prouvait.
#
# ── Ce que ces tests attrapent, et que le socle précédent laissait passer ──
#
# On injecte dans la branche « tapis nul » un écart RELATIF ρ sur ce que
# touche le joueur éliminé, et on le REDISTRIBUE sur les vivants pour que la
# dotation reste exacte au centime. La mutation est donc invisible pour la
# conservation ; elle est sans dimension, donc invisible pour l'échelle ; elle
# est symétrique, donc invisible pour la permutation ; elle est constante,
# donc invisible pour la monotonie. C'est exactement le mécanisme décrit par
# la revue : « invariant vert, valeur fausse ».
#
# Dans sa version la plus furtive (l'écart n'est appliqué que s'il reste au
# moins DEUX survivants, ce qui épargne le heads-up et les cas affines — les
# seuls endroits où l'ancien socle voyait quelque chose) :
#
# Comptage sur les quatre fichiers qui touchent l'ICM — ce fichier,
# `test_icm.py`, `test_icm_tapis_nul.py` et `test_golden_values.py` :
#
#     ρ        socle précédent      ces invariants
#     1e-6     2 échecs             35 échecs
#     1e-9     **0 échec**          21 échecs
#     1e-11    **0 échec**          12 échecs
#
# À ρ = 1e-9, tout ce qui existait avant est VERT sur un ICM faux.
# (Le banc rejoue la mesure de continuité : `--continuite`, section 10.)

EPSILONS = (10.0, 1.0, 0.1, 1e-6, 1e-9, 1e-12)
"""Tapis résiduels balayés. Six décades : assez pour distinguer une
CONVERGENCE (l'écart suit ε) d'un PALIER (l'écart reste)."""


def _bf_par_continuite(stacks, payouts, hero: int, villain: int,
                       eps: float) -> float:
    """Le facteur de bulle où le tapis qui tombe à zéro garde ``eps`` jeton.

    Ce n'est pas le même terme du quotient qui traverse la branche selon qui
    couvre qui : ``$EV_perte`` pour le héros couvert, ``$EV_gain`` pour le
    héros couvrant. On ne perturbe QUE le terme concerné — perturber les deux
    mélangerait deux effets d'ordre epsilon et brouillerait la pente. Le
    transfert est réduit d'autant, donc le total de jetons reste celui du
    spot.

    ``eps = 0`` doit redonner exactement `bubble_factor` : c'est ce que
    vérifie `test_le_harnais_a_epsilon_nul_redonne_la_fonction_de_production`,
    sans quoi tout ce qui suit comparerait deux choses différentes.
    """
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = icm_dollar_ev(s, p, hero)
    d_win = amt - eps if s[villain] <= s[hero] else amt
    d_lose = amt - eps if s[hero] <= s[villain] else amt
    win = list(s)
    win[hero] += d_win
    win[villain] -= d_win
    lose = list(s)
    lose[hero] -= d_lose
    lose[villain] += d_lose
    return ((base - icm_dollar_ev(lose, p, hero))
            / (icm_dollar_ev(win, p, hero) - base))


def _extrapole_en_zero(f, e1: float, e2: float) -> float:
    """``f(0)`` par extrapolation de Richardson au premier ordre.

    Si ``f(ε) = f(0) + Cε``, elle rend ``f(0)`` EXACTEMENT quel que soit C —
    bien mieux que « f(1e-12) est proche de f(0) », qui ne distinguerait pas
    une convergence lente d'un petit palier. Une branche fausse fait un
    palier : l'extrapolée rate alors la branche de tout le saut.
    """
    return (e1 * f(e2) - e2 * f(e1)) / (e1 - e2)


# Huit spots où un tapis tombe à zéro, dans les DEUX sens de couverture.
CONTINUITE_BF = [
    pytest.param(TABLE_9, PAYOUTS_9, 3, 1, "couvert", id="PMU9-court-vs-chipleader"),
    pytest.param(TABLE_9, PAYOUTS_9, 3, 0, "couvert", id="PMU9-court-vs-moyen"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 3, 0,
                 "couvert", id="bulle-4-3-court-vs-gros"),
    pytest.param((900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0),
                 4, 0, "couvert", id="satellite-court-vs-gros"),
    pytest.param((600.0, 300.0, 100.0), (50.0, 30.0, 20.0), 2, 0,
                 "couvert", id="3-joueurs-court-vs-gros"),
    pytest.param((7000.0, 3000.0), (60.0, 40.0), 1, 0, "couvert",
                 id="heads-up-affine"),
    pytest.param(TABLE_9, PAYOUTS_9, 1, 3, "couvrant", id="PMU9-chipleader-paie"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 0, 3,
                 "couvrant", id="bulle-4-3-gros-paie"),
]

# Un SEUL tapis nul : la continuité y est exigible (voir § « plusieurs zéros »).
CONTINUITE_VECTEUR = [
    pytest.param((0.0, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0),
                 PAYOUTS_9, id="9-joueurs-9-gains"),
    pytest.param((0.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0),
                 id="4-joueurs-3-gains-le-mort-ne-touche-rien"),
    pytest.param((0.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0, 10.0),
                 id="4-joueurs-4-gains-le-mort-touche-la-4e-place"),
    pytest.param((0.0, 100.0), (50.0, 30.0, 20.0), id="2-joueurs-gains-tronques"),
    pytest.param((0.0, 100.0, 200.0), (100.0,), id="winner-take-all"),
    pytest.param((0.0, 700.0, 500.0, 300.0, 100.0), (100.0, 100.0, 100.0),
                 id="satellite-3-places-egales"),
    pytest.param((300.0, 0.0, 200.0, 100.0), (50.0, 30.0, 20.0, 10.0),
                 id="zero-au-milieu-de-la-table"),
]


def _icm_mort_a_zero(stacks, payouts) -> np.ndarray:
    """TÉMOIN : la branche FAUSSE — un joueur à zéro touche 0.

    C'est l'écriture naïve (« il n'a plus de jetons, il ne vaut plus rien »),
    et c'est celle que le correctif du tapis nul a remplacée. Elle sert à
    prouver que les invariants de cette section MORDENT : un invariant qu'une
    implémentation fausse ne fait pas tomber ne mesure rien.
    """
    s, p = _validate(stacks, payouts)
    viv = [i for i, v in enumerate(s) if v > 0.0]
    if len(viv) == len(s):
        return icm_equities(s, p)
    out = np.zeros(len(s), dtype=np.float64)
    haut = list(p[: len(viv)])
    if haut:
        out[viv] = icm_equities([s[i] for i in viv], haut)
    return out


def _bf_mort_a_zero(stacks, payouts, hero: int, villain: int) -> float:
    """Le facteur de bulle qu'on obtiendrait avec la branche fausse."""
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = float(_icm_mort_a_zero(s, p)[hero])
    win = list(s)
    win[hero] += amt
    win[villain] -= amt
    lose = list(s)
    lose[hero] -= amt
    lose[villain] += amt
    return ((base - float(_icm_mort_a_zero(lose, p)[hero]))
            / (float(_icm_mort_a_zero(win, p)[hero]) - base))


@pytest.mark.parametrize("stacks,payouts,hero,villain,sens", CONTINUITE_BF)
def test_le_harnais_a_epsilon_nul_redonne_la_fonction_de_production(
        stacks, payouts, hero, villain, sens) -> None:
    """Sans cela, toute la section comparerait deux choses différentes.

    `_bf_par_continuite(..., eps=0)` doit reproduire `bubble_factor` AU BIT
    PRÈS : mêmes états, même arithmétique, seule la valeur d'epsilon change
    ensuite. Un harnais qui divergerait déjà en zéro mesurerait sa propre
    dérive, pas celle de la branche.
    """
    assert _bf_par_continuite(stacks, payouts, hero, villain, 0.0) == (
        bubble_factor(stacks, payouts, hero, villain))


@pytest.mark.parametrize("stacks,payouts,hero,villain,sens", CONTINUITE_BF)
@pytest.mark.parametrize("eps", EPSILONS)
def test_le_bubble_factor_converge_vers_la_branche_tapis_nul(
        stacks, payouts, hero, villain, sens, eps) -> None:
    """La branche « tapis nul » est-elle la LIMITE de la récursion ?

    On remplace le tapis à zéro par un tapis résiduel ``ε`` décroissant, en
    prélevant les jetons sur le même adversaire : la table garde son total,
    donc l'invariance d'échelle n'est pas sollicitée en douce.

    La borne est ``3·ε/tapis_court + 1e-13·BF``. Le premier terme est SANS
    DIMENSION — ε rapporté au tapis court — donc valable à toute échelle ; la
    pente réellement mesurée vaut 1,0 à 2,2 (banc, ε ≥ 1e-9). Le second est le
    plancher de bruit : à ε = 1e-12 sur le heads-up, le terme linéaire tombe à
    1e-15, c'est-à-dire sous le dernier bit des $EV soustraites.

    Écarts mesurés (banc ``--continuite``), spot PMU court vs chipleader :
    9,5e-1 · 1,1e-1 · 1,1e-2 · 1,1e-7 · 1,1e-10 · 1,1e-13 pour
    ε = 10 · 1 · 0,1 · 1e-6 · 1e-9 · 1e-12. Le rapport écart/ε est CONSTANT à
    0,1136 sur six décades : c'est une convergence linéaire, pas un palier.
    """
    court = min(stacks[hero], stacks[villain])
    if eps >= court:
        pytest.skip("ε doit rester petit devant le tapis court")
    b0 = bubble_factor(stacks, payouts, hero, villain)
    ecart = abs(_bf_par_continuite(stacks, payouts, hero, villain, eps) - b0)
    borne = 3.0 * eps / court + 1e-13 * b0
    assert ecart <= borne, (
        f"la branche « tapis nul » s'écarte de la limite : à ε={eps:g} "
        f"l'écart vaut {ecart:.3e} pour une borne de {borne:.3e}. "
        f"BF(branche)={b0:.12f}")


@pytest.mark.parametrize("stacks,payouts,hero,villain,sens", CONTINUITE_BF)
def test_la_limite_extrapolee_en_zero_est_exactement_la_branche(
        stacks, payouts, hero, villain, sens) -> None:
    """Le cœur du chantier, et sa réponse : elles COÏNCIDENT.

    « L'écart est petit » ne suffirait pas — un palier de 1e-13 serait petit
    aussi. On extrapole donc en zéro depuis ε = 1e-6 et 1e-7 : si l'écart est
    linéaire en ε, l'extrapolation rend la limite EXACTEMENT ; s'il y a un
    palier, elle le rate de tout le saut.

    Écart relatif mesuré sur les huit spots : **au plus 2,0e-15**, soit le
    bruit de la double précision. Le seuil est posé à 1e-11 — 5 000 fois plus
    haut — et le témoin ci-dessous (`..._a_des_dents`) rate de 23 %.

    Conclusion, chiffrée : il n'y a PAS d'écart à corriger entre la branche
    et la limite.
    """
    b0 = bubble_factor(stacks, payouts, hero, villain)
    f0 = _extrapole_en_zero(
        lambda e: _bf_par_continuite(stacks, payouts, hero, villain, e),
        1e-6, 1e-7)
    assert f0 == pytest.approx(b0, rel=1e-11), (
        f"la limite extrapolée ({f0:.15f}) n'est pas la valeur de la branche "
        f"({b0:.15f}) : la branche « tapis nul » n'est pas la limite de la "
        f"récursion")


def test_la_continuite_du_bubble_factor_a_des_dents() -> None:
    """Un invariant qu'aucune implémentation fausse ne fait tomber ne prouve rien.

    Le témoin est la branche naïve : « un joueur à zéro touche 0 ». Sur la
    table PMU à neuf joueurs, vue du tapis court qui paie le chipleader :

    * le facteur de bulle passe de **1,6069 à 2,0855** (+29,8 %), soit
      **+6,5 points** d'équité exigée sur un call pot 100 / mise 100 ;
    * et surtout l'écart à la limite ne DÉCROÎT PAS : il fait un palier à
      9,34 — exactement le dernier gain, celui que la branche fausse oublie —
      identique à ε = 1e-6, 1e-9 et 1e-12.

    C'est ce palier que l'extrapolation de Richardson attrape : elle rend la
    vraie limite (1,6069) là où la branche fausse annonce 2,0855.
    """
    b_faux = _bf_mort_a_zero(TABLE_9, PAYOUTS_9, 3, 1)
    b_vrai = bubble_factor(TABLE_9, PAYOUTS_9, 3, 1)
    assert b_faux == pytest.approx(2.0855, abs=1e-3)
    assert b_vrai == pytest.approx(1.6069, abs=1e-3)
    assert abs(b_faux - b_vrai) / b_vrai > 0.25, (
        "le témoin doit s'écarter franchement, sinon l'invariant est décoratif")

    # …en points d'équité exigée, sur un call de la taille du pot.
    ecart_pts = (icm_required_equity(100.0, 100.0, b_faux)
                 - icm_required_equity(100.0, 100.0, b_vrai)) * 100.0
    assert ecart_pts > 5.0, f"écart mesuré {ecart_pts:.2f} points"

    # Le test de continuité, appliqué au témoin, DOIT tomber : palier, pas
    # convergence.
    f0 = _extrapole_en_zero(
        lambda e: _bf_par_continuite(TABLE_9, PAYOUTS_9, 3, 1, e), 1e-6, 1e-7)
    assert f0 != pytest.approx(b_faux, rel=1e-11), (
        "l'extrapolation devrait rater la valeur de la branche fausse")
    assert f0 == pytest.approx(b_vrai, rel=1e-11), (
        "…et retomber sur la vraie limite, qui est la branche de production")


def test_la_branche_fausse_est_invisible_du_cote_du_couvrant() -> None:
    """Le deuxième angle mort, épinglé pour qu'il ne soit pas redécouvert.

    La branche « tapis nul » ne change QUE l'équité du joueur à zéro : les
    survivants reçoivent ``icm(vivants, π[:n−1])`` dans les deux versions.
    Donc le défaut ne se voit que du côté du joueur QUI VA MOURIR.

    Conséquence directe : tester le facteur de bulle du COUVRANT — le
    chipleader qui paie le tapis court, le spot qu'on regarde d'instinct —
    n'aurait rien attrapé. C'est le côté couvert qu'il fallait regarder, et
    c'est celui que personne ne regardait.
    """
    # couvrant : identique entre la vraie branche et le témoin
    assert _bf_mort_a_zero(TABLE_9, PAYOUTS_9, 1, 3) == pytest.approx(
        bubble_factor(TABLE_9, PAYOUTS_9, 1, 3), rel=1e-12)
    # couvert : 30 % d'écart
    assert _bf_mort_a_zero(TABLE_9, PAYOUTS_9, 3, 1) != pytest.approx(
        bubble_factor(TABLE_9, PAYOUTS_9, 3, 1), rel=0.25)

    # Le second angle mort : moins de gains que de joueurs. Le mort ne touche
    # rien de toute façon, les deux branches se rejoignent.
    quatre = (5000.0, 3000.0, 1500.0, 500.0)
    assert _bf_mort_a_zero(quatre, (50.0, 30.0, 20.0), 3, 0) == pytest.approx(
        bubble_factor(quatre, (50.0, 30.0, 20.0), 3, 0), rel=1e-12)
    # …alors qu'avec un 4ᵉ gain, l'écart réapparaît.
    assert _bf_mort_a_zero(quatre, (50.0, 30.0, 20.0, 10.0), 3, 0) != (
        pytest.approx(bubble_factor(quatre, (50.0, 30.0, 20.0, 10.0), 3, 0),
                      rel=0.05))


@pytest.mark.parametrize("stacks,payouts,hero,villain,sens", CONTINUITE_BF)
def test_une_erreur_d_equite_est_amplifiee_par_le_quotient(
        stacks, payouts, hero, villain, sens) -> None:
    """Pourquoi les huit sections précédentes ne pouvaient PAS voir ce défaut.

    Une erreur sur ``$EV_perte`` est absorbée dans la somme des équités —
    la conservation reste verte, l'échelle et la permutation aussi. Mais le
    facteur de bulle la lit TELLE QUELLE au numérateur, et la rapporte à un
    dénominateur (``$EV_gain − $EV``) qui est petit.

    On injecte donc une erreur de 1 % de la dotation et on mesure ce qu'elle
    devient. Amplification mesurée sur les huit spots : **×11,9 à ×76,5**.
    Autrement dit : un ICM juste à 1 % près peut rendre un facteur de bulle
    faux de 20 %, et une équité exigée fausse de plusieurs points.
    """
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = icm_dollar_ev(s, p, hero)
    win = list(s)
    win[hero] += amt
    win[villain] -= amt
    lose = list(s)
    lose[hero] -= amt
    lose[villain] += amt
    gain = icm_dollar_ev(win, p, hero) - base
    perte = base - icm_dollar_ev(lose, p, hero)
    b0 = perte / gain

    delta = 0.01 * sum(p[: len(s)])          # 1 % de la dotation
    amplification = abs((perte + delta) / gain - b0) / b0 / 0.01
    assert amplification >= 10.0, (
        f"amplification mesurée ×{amplification:.2f} — si elle tombait sous "
        "10, l'argument « la conservation ne voit pas cet écart » perdrait "
        "sa force et cette section deviendrait secondaire")


@pytest.mark.parametrize("stacks,payouts", CONTINUITE_VECTEUR)
@pytest.mark.parametrize("eps", EPSILONS)
def test_le_vecteur_d_equites_est_continu_a_un_seul_tapis_nul(
        stacks, payouts, eps) -> None:
    """L'invariant de continuité, sur TOUTES les équités et non le seul héros.

    Il est exigé à UN SEUL tapis nul : le groupe qui s'évanouit est alors un
    singleton, il n'y a plus de rapport entre tapis mourants, et la limite est
    unique. (À plusieurs zéros, voir la section suivante : la discontinuité
    est authentique et ATTENDUE.)

    La borne est ``4·ε·dotation/total`` — sans dimension elle aussi, donc
    valable à toute échelle. Pente mesurée : au plus 2,696 (banc).
    """
    i0 = [i for i, v in enumerate(stacks) if v == 0.0][0]
    ref = icm_equities(stacks, payouts)
    perturbe = list(stacks)
    perturbe[i0] = eps
    ecart = float(np.max(np.abs(icm_equities(perturbe, payouts) - ref)))
    echelle = _dotation_attribuable(stacks, payouts) / sum(stacks)
    assert ecart <= 4.0 * eps * echelle, (
        f"la branche « tapis nul » n'est pas la limite : à ε={eps:g} l'écart "
        f"vaut {ecart:.3e} pour une borne de {4.0 * eps * echelle:.3e}")


def test_la_continuite_du_vecteur_a_des_dents() -> None:
    """Le témoin fait un PALIER là où la vraie branche converge.

    Sur la table PMU à neuf joueurs avec un tapis nul, la branche fausse
    laisse un écart CONSTANT de 9,34 $ — le dernier gain — que ε vaille
    1e-6, 1e-9 ou 1e-12. La vraie branche, elle, descend à 2,5e-12.

    Sans cette vérification, l'invariant précédent pourrait passer sur une
    implémentation qui ne converge pas du tout.
    """
    s = (0.0, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
    ref_faux = _icm_mort_a_zero(s, PAYOUTS_9)
    ref_vrai = icm_equities(s, PAYOUTS_9)
    for eps in (1e-6, 1e-9, 1e-12):
        perturbe = list(s)
        perturbe[0] = eps
        palier = float(np.max(np.abs(_icm_mort_a_zero(perturbe, PAYOUTS_9)
                                     - ref_faux)))
        assert palier == pytest.approx(9.34, abs=1e-3), (
            f"le témoin devrait plafonner au dernier gain ; vu {palier:.6f}")
        vrai = float(np.max(np.abs(icm_equities(perturbe, PAYOUTS_9)
                                   - ref_vrai)))
        assert vrai < 1e-5 * palier, (
            "la vraie branche, elle, doit converger — sinon les deux "
            "comportements seraient indiscernables")


# ── plusieurs tapis nuls : la discontinuité est AUTHENTIQUE ────────────────


def test_l_icm_est_authentiquement_discontinu_a_plusieurs_tapis_nuls() -> None:
    """La nuance capitale : ici, exiger la continuité serait une ERREUR.

    Avec DEUX tapis qui s'évanouissent, la limite dépend de leur RAPPORT.
    Pour ``[100, a·ε, b·ε]`` et des gains ``(π₀, π₁, π₂)``, le survivant prend
    la première place presque sûrement, puis les deux mourants se disputent la
    seconde à la Harville, d'où :

        limite = [π₀, (a·π₁ + b·π₂)/(a+b), (b·π₁ + a·π₂)/(a+b)]

    Quatre chemins, quatre limites différentes — donc il n'existe PAS de
    valeur en zéro, et un invariant de continuité qui l'exigerait fabriquerait
    un faux échec, puis ferait « corriger » du code juste.

    Mesuré (banc ``--continuite``), gains 50/30/20 :

    ================  =========================
    chemin            limite
    ================  =========================
    [100, ε, ε]       (50 ; 25 ; 25)
    [100, ε, ε/2]     (50 ; 26,667 ; 23,333)
    [100, ε, ε/100]   (50 ; 29,901 ; 20,099)
    [100, 3ε, 7ε]     (50 ; 23 ; 27)
    ================  =========================
    """
    p = (50.0, 30.0, 20.0)
    limites = set()
    for a, b in ((1.0, 1.0), (1.0, 0.5), (1.0, 0.01), (3.0, 7.0)):
        attendu = [p[0],
                   (a * p[1] + b * p[2]) / (a + b),
                   (b * p[1] + a * p[2]) / (a + b)]
        got = icm_equities([100.0, a * 1e-9, b * 1e-9], p)
        np.testing.assert_allclose(got, attendu, atol=1e-6, err_msg=(
            f"la limite du chemin a={a}, b={b} ne suit pas la forme fermée"))
        limites.add(tuple(np.round(got, 6).tolist()))

    assert len(limites) == 4, (
        f"quatre chemins doivent donner quatre limites distinctes ; "
        f"vu {len(limites)} — sans quoi la discontinuité serait une histoire")

    # Et la conservation tient sur CHAQUE chemin : la discontinuité porte sur
    # la répartition entre les mourants, pas sur le total.
    for a, b in ((1.0, 0.5), (3.0, 7.0)):
        assert float(icm_equities([100.0, a * 1e-9, b * 1e-9], p).sum()) == (
            pytest.approx(100.0, rel=1e-9))


def test_a_plusieurs_zeros_la_branche_rend_la_limite_symetrique() -> None:
    """Ce que la branche choisit, et pourquoi c'est le bon choix.

    À tapis EXACTEMENT nuls, les rapports du test précédent n'existent plus :
    l'ordre d'élimination n'est pas dans les données. La branche partage les
    gains restants à parts égales — c'est-à-dire la limite du chemin
    SYMÉTRIQUE ``a = b``, et la seule réponse compatible avec l'invariance par
    permutation (deux joueurs à zéro ne sont pas départageables).

    C'est un choix documenté, pas une convergence. Le test l'épingle comme
    tel : `[100, 0, 0]` vaut (50 ; 25 ; 25) et NON (50 ; 26,667 ; 23,333),
    qui est pourtant la limite d'un chemin parfaitement légitime.
    """
    p = (50.0, 30.0, 20.0)
    branche = icm_equities([100.0, 0.0, 0.0], p)
    np.testing.assert_allclose(branche, [50.0, 25.0, 25.0], atol=1e-11)
    np.testing.assert_allclose(
        branche, icm_equities([100.0, 1e-9, 1e-9], p), atol=1e-8)

    # …et elle ne vaut PAS la limite d'un chemin dissymétrique.
    assert float(branche[1]) != pytest.approx(26.666667, abs=1e-3)

    # Trois zéros : même règle, gains 50/30/20/10.
    p4 = (50.0, 30.0, 20.0, 10.0)
    np.testing.assert_allclose(icm_equities([100.0, 0.0, 0.0, 0.0], p4),
                               [50.0, 20.0, 20.0, 20.0], atol=1e-11)
    np.testing.assert_allclose(
        icm_equities([100.0, 1e-4, 5e-5, 2.5e-5], p4),
        [50.0, 24.666667, 20.0, 15.333333], atol=1e-3)


@pytest.mark.parametrize("survivants,mourants,payouts", [
    pytest.param([300.0, 100.0], [1.0, 0.5], (50.0, 30.0, 20.0, 10.0),
                 id="2-survivants-2-mourants"),
    pytest.param([300.0, 200.0, 100.0], [3.0, 1.0],
                 (50.0, 30.0, 20.0, 10.0, 5.0), id="3-survivants-2-mourants"),
    pytest.param([500.0], [1.0, 0.5, 0.25], (50.0, 30.0, 20.0, 10.0),
                 id="1-survivant-3-mourants"),
    pytest.param([400.0, 100.0], [1.0, 2.0, 4.0],
                 (60.0, 40.0, 30.0, 20.0, 10.0), id="2-survivants-3-mourants"),
])
def test_la_factorisation_asymptotique_generale(
        survivants, mourants, payouts) -> None:
    """La forme générale des deux tests précédents, et leur justification.

    Quand ``k`` tapis s'évanouissent proportionnellement à ε devant ``r``
    survivants, le problème se FACTORISE exactement :

        lim  icm(survivants ⊕ ε·mourants, π)
        ε→0      =  icm(survivants, π[:r])  ⊕  icm(mourants, π[r:])

    Deux conséquences, et ce sont les deux moitiés du chantier :

    * ``k = 1`` : le second facteur est un singleton, ``icm([a], π[r:])`` ne
      dépend plus de ``a`` — la limite est unique, la continuité est exigible.
    * ``k ≥ 2`` : elle dépend des rapports ``a_i``, la limite n'existe pas.

    Et la branche de production rend exactement le second facteur évalué à
    tapis ÉGAUX, c'est-à-dire la moyenne des gains restants. Ce test le
    vérifie des deux côtés.
    """
    r = len(survivants)
    attendu = np.concatenate([icm_equities(survivants, payouts[:r]),
                              icm_equities(mourants, payouts[r:])])
    got = icm_equities(survivants + [a * 1e-9 for a in mourants], payouts)
    np.testing.assert_allclose(got, attendu, atol=1e-6, err_msg=(
        "la limite ne se factorise pas en survivants × mourants"))

    # La branche, à tapis exactement nuls, == le cas symétrique du facteur.
    symetrique = np.concatenate([icm_equities(survivants, payouts[:r]),
                                 icm_equities([1.0] * len(mourants),
                                              payouts[r:])])
    np.testing.assert_allclose(
        icm_equities(survivants + [0.0] * len(mourants), payouts),
        symetrique, atol=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# 10 — GAINS : TRANSLATION, HOMOGÉNÉITÉ, TAPIS ÉGAUX, DÉNOMINATEURS NULS
# ═══════════════════════════════════════════════════════════════════════════


# Les tables de `TABLES` qui ont AU MOINS autant de gains que de joueurs.
# Écrites en clair plutôt que filtrées : la condition m ≥ n est le sujet même
# du test, elle doit se lire, pas se déduire d'une compréhension de liste.
TABLES_TOUTES_PLACES_PAYEES = [
    pytest.param(TABLE_9, PAYOUTS_9, id="table-PMU-9-joueurs-9-gains"),
    pytest.param((1000.0, 1000.0, 1000.0), (50.0, 30.0, 20.0), id="ex-aequo-3-3"),
    pytest.param((0.0, 10.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0),
                 id="un-tapis-nul-4-4"),
    pytest.param((0.0, 0.0, 20.0, 30.0), (100.0, 50.0, 25.0, 10.0),
                 id="deux-tapis-nuls-4-4"),
    pytest.param((7000.0, 3000.0), (60.0, 40.0), id="heads-up-2-2"),
    pytest.param((100.0, 50.0), (50.0, 30.0, 20.0),
                 id="plus-de-gains-que-de-joueurs-2-3"),
    pytest.param((100.0,), (50.0, 30.0, 20.0), id="joueur-unique-1-3"),
]


@pytest.mark.parametrize("stacks,payouts", TABLES_TOUTES_PLACES_PAYEES)
@pytest.mark.parametrize("c", [0.5, 7.0, 1000.0])
def test_ajouter_une_constante_a_tous_les_gains_translate_les_equites(
        stacks, payouts, c) -> None:
    """Invariant de translation — valable EXACTEMENT quand m ≥ n.

    Si toutes les places sont payées, chaque joueur en occupe une :
    ``Σₖ pᵢₖ = 1``, donc ajouter ``c`` à chaque gain ajoute ``c`` à chaque
    équité. C'est une contrainte forte et non triviale : elle épingle le fait
    que les probabilités de place somment bien à 1 joueur par joueur, tapis
    nuls compris.

    La borne du cas contraire (m < n) est dans le test suivant : là, la
    translation change la structure, et ce n'est PAS un défaut.
    """
    ref = icm_equities(stacks, payouts)
    got = icm_equities(stacks, [x + c for x in payouts])
    np.testing.assert_allclose(got, ref + c, atol=1e-9 * (
        _dotation_attribuable(stacks, payouts) + c * len(stacks) + 1.0),
        err_msg="les probabilités de place ne somment pas à 1 par joueur")


@pytest.mark.parametrize("stacks,payouts,hero,villain", [
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 3, 0,
                 id="bulle-4-joueurs-3-gains"),
    pytest.param((900.0, 700.0, 500.0, 300.0, 100.0), (100.0, 60.0, 30.0),
                 4, 0, id="5-joueurs-3-gains"),
    pytest.param((176.0, 124.0, 92.0, 70.0, 55.0, 44.0),
                 (100.0, 60.0, 40.0, 25.0), 5, 0, id="6-joueurs-4-gains"),
])
def test_la_translation_n_est_PAS_un_invariant_quand_m_inferieur_a_n(
        stacks, payouts, hero, villain) -> None:
    """Le contre-exemple qui interdit de généraliser — avec sa limite prédite.

    Avec moins de gains que de joueurs, ``Σₖ pᵢₖ = P(finir dans les places)``
    est **< 1 et dépend du joueur** : le tapis court y arrive moins souvent.
    Ajouter ``c`` à tous les gains, c'est donc ajouter une prime « pour avoir
    atteint l'argent », qui profite davantage aux gros tapis. La structure
    change ; le facteur de bulle aussi. Ce n'est pas un défaut, c'est la
    définition.

    Et cette dépendance a une LIMITE CALCULABLE AILLEURS, ce qui donne au test
    une référence externe plutôt qu'une valeur recopiée du code : quand
    ``c → ∞``, les gains deviennent tous égaux à ``c`` en proportion, la
    structure tend vers un SATELLITE PUR (k places de valeur identique), et le
    facteur de bulle doit tendre vers celui qu'on obtient en passant
    directement des gains tous égaux.

    Mesuré sur la bulle 4 joueurs / 3 gains, tapis court contre le gros :

    ==========  =============
    c           BF
    ==========  =============
    0           1,433617
    1 000       1,625918
    1e9         1,631957
    satellite   **1,631957**
    ==========  =============

    Un calcul qui ignorerait la troncature des gains, ou qui normaliserait les
    équités au lieu de sommer des places, raterait cette convergence.
    """
    ref = icm_equities(stacks, payouts)
    got = icm_equities(stacks, [x + 1000.0 for x in payouts])
    assert float(np.max(np.abs(got - (ref + 1000.0)))) > 1.0, (
        "avec m < n la translation DOIT déplacer les équités autrement qu'en "
        "bloc — si ce n'était plus le cas, le test précédent serait vide")

    bf0 = bubble_factor(stacks, payouts, hero, villain)
    limite = bubble_factor(stacks, (1.0,) * len(payouts), hero, villain)
    assert bf0 != pytest.approx(limite, rel=1e-3), (
        "le point de départ doit être franchement distinct de la limite, "
        "sinon la convergence ci-dessous ne prouve rien")

    precedent = abs(bf0 - limite)
    for c in (10.0, 1e3, 1e5, 1e9):
        bf = bubble_factor(stacks, [x + c for x in payouts], hero, villain)
        ecart = abs(bf - limite)
        assert ecart < precedent, (
            f"à c = {c:g}, BF = {bf:.9f} ne se rapproche pas du satellite pur "
            f"({limite:.9f})")
        precedent = ecart
    assert precedent < 1e-6 * limite, (
        f"à c = 1e9 il reste {precedent:.3e} d'écart au satellite pur")


@pytest.mark.parametrize("stacks,payouts", TABLES)
@pytest.mark.parametrize("lam", [1e-6, 3.0, 1e6])
def test_multiplier_les_gains_multiplie_les_equites(stacks, payouts, lam) -> None:
    """Homogénéité de degré 1 : le dollar n'a pas plus d'unité que le jeton.

    Compter la dotation en centimes, en euros ou en millions doit donner le
    même partage. C'est le pendant, côté GAINS, de l'invariance d'échelle
    côté tapis — et il manquait.

    Écart relatif mesuré : au plus 7,5e-17 (banc).
    """
    ref = icm_equities(stacks, payouts)
    got = icm_equities(stacks, [x * lam for x in payouts])
    np.testing.assert_allclose(got, lam * ref, rtol=0.0, atol=1e-12 * lam * max(
        _dotation_attribuable(stacks, payouts), 1.0))


@pytest.mark.parametrize("stacks,payouts,hero,villain", [
    pytest.param(TABLE_9, PAYOUTS_9, 3, 1, id="PMU-9-joueurs"),
    pytest.param((5000.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0), 3, 0,
                 id="bulle-4-3"),
    pytest.param((0.0, 3000.0, 1500.0, 500.0), (50.0, 30.0, 20.0, 10.0), 3, 1,
                 id="un-tapis-nul"),
])
@pytest.mark.parametrize("lam", [1e-6, 3.0, 1e6])
def test_le_bubble_factor_est_homogene_de_degre_zero_en_les_gains(
        stacks, payouts, hero, villain, lam) -> None:
    """Corollaire : un RAPPORT de différences de $EV ne voit pas l'échelle des $.

    Le cas « un tapis nul » est dans la liste exprès : c'est celui qui fait
    traverser la branche spéciale, et donc celui où une constante absolue
    cachée se verrait.
    """
    assert bubble_factor(stacks, [x * lam for x in payouts], hero, villain) == (
        pytest.approx(bubble_factor(stacks, payouts, hero, villain), abs=1e-9))


@pytest.mark.parametrize("n,payouts", [
    pytest.param(2, (50.0, 30.0, 20.0), id="2-joueurs-3-gains"),
    pytest.param(3, (50.0, 30.0, 20.0), id="3-joueurs-3-gains"),
    pytest.param(5, (50.0, 30.0, 20.0), id="5-joueurs-3-gains"),
    pytest.param(5, PAYOUTS_9, id="5-joueurs-9-gains"),
    pytest.param(9, PAYOUTS_9, id="9-joueurs-9-gains"),
    pytest.param(4, (100.0, 60.0, 40.0, 25.0), id="4-joueurs-4-gains"),
    pytest.param(6, (100.0,), id="6-joueurs-1-gain"),
])
def test_des_tapis_tous_egaux_valent_la_moyenne_des_gains_attribuables(
        n, payouts) -> None:
    """Le seul cas où la valeur exacte se calcule de tête — donc le seul où
    l'invariant peut être comparé à une référence INDÉPENDANTE du code.

    Par symétrie, chaque joueur a la même loi de place, donc la même équité ;
    par conservation, cette équité vaut ``Σ π[:n] / n``. La formule ne
    dépend ni de la récursion ni de la branche spéciale : c'est une vraie
    référence externe, pas une comparaison du code à lui-même.

    Le cas n = 9, m = 9 (moyenne 69,26) et le cas n = 5, m = 3 (moyenne 20)
    couvrent les deux régimes m ≥ n et m < n.
    """
    eq = icm_equities([1000.0] * n, payouts)
    moyenne = sum(payouts[:n]) / n
    np.testing.assert_allclose(eq, [moyenne] * n,
                               atol=1e-11 * max(sum(payouts[:n]), 1.0))


@pytest.mark.parametrize("n,payouts,affine", [
    pytest.param(3, (50.0, 30.0, 20.0), False, id="3-joueurs-3-gains"),
    pytest.param(5, (50.0, 30.0, 20.0), False, id="5-joueurs-3-gains"),
    pytest.param(9, PAYOUTS_9, False, id="9-joueurs-9-gains"),
    pytest.param(6, (100.0,), True, id="6-joueurs-1-gain-affine"),
    pytest.param(2, (50.0, 30.0), True, id="heads-up-affine"),
])
def test_a_tapis_egaux_le_bubble_factor_est_le_meme_pour_toutes_les_paires(
        n, payouts, affine) -> None:
    """Corollaire de la symétrie, et il n'est pas gratuit.

    Aucune paire n'est distinguable quand tous les tapis sont égaux : le
    facteur de bulle doit donc être une CONSTANTE de la table. Un calcul qui
    indexerait les gains sur la position, ou qui traiterait le premier siège à
    part, le violerait.

    Et la valeur, elle, discrimine : elle vaut exactement 1 quand l'équité est
    affine en jetons (une seule place payée, ou heads-up), et strictement plus
    sinon — 1,3333 à 3 joueurs, 1,9215 à 9. Sans cette seconde assertion,
    « constante » serait vrai d'un calcul qui rendrait 1 partout.
    """
    s = [1000.0] * n
    vals = [bubble_factor(s, payouts, i, j)
            for i in range(n) for j in range(n) if i != j]
    assert max(vals) - min(vals) <= 1e-9, (
        f"le facteur de bulle dépend du siège : {min(vals)} … {max(vals)}")
    if affine:
        assert vals[0] == pytest.approx(1.0, abs=1e-9)
    else:
        assert vals[0] > 1.0 + 1e-6, (
            f"BF = {vals[0]} : à tapis égaux et plusieurs gains, risquer doit "
            "coûter plus que gagner ne rapporte")


def test_le_denominateur_nul_du_bubble_factor_traverse_toute_la_chaine() -> None:
    """Le garde-fou rend +inf : encore faut-il que la suite le supporte.

    Historiquement, `bubble_factor` rendait −1,0 sur une structure dégénérée
    et `analyse_icm_spot` échouait ensuite sur le message trompeur
    « bubble factor doit être > 0 ». Le garde-fou est testé plus haut ; ce
    test-ci vérifie le CHEMIN COMPLET, qui ne l'était pas : +inf doit traverser
    `risk_premium`, `icm_required_equity` et `analyse_icm_spot` sans lever.

    Sémantique attendue : rien n'est en jeu, donc aucune équité ne justifie de
    risquer — prime de risque maximale (0,5) et équité exigée de 100 %.
    """
    spot = IcmSpot(stacks=(1.0, 2.0, 3.0), payouts=(1.0, 1.0, 1.0),
                   hero=0, villain=1, pot=100.0, bet=50.0)
    a = analyse_icm_spot(spot, hero_equity=0.99)
    assert math.isinf(a.bubble)
    assert a.premium == 0.5
    assert a.alpha_icm == 1.0
    assert "FOLD" in a.verdict
    assert a.explain()                        # ne doit pas lever sur un inf

    assert risk_premium(math.inf) == 0.5
    assert icm_required_equity(100.0, 50.0, math.inf) == 1.0


def test_le_bubble_factor_ne_rend_jamais_zero_meme_avec_tapis_nuls() -> None:
    """L'autre dénominateur nul possible, celui qu'on n'a pas mesuré.

    `icm_required_equity` REFUSE ``bf <= 0``. Un facteur de bulle nul —
    ``max(loss, 0)/gain`` avec une perte négative de bruit — casserait donc la
    chaîne avec le même message trompeur qu'à l'époque du −1,0.

    Le balayage inonde le cas qui pourrait le produire : 30 % de tapis nuls et
    des gains délibérément EX ÆQUO (c'est l'égalité des gains qui annule les
    différences de $EV). Mesuré sur 1 005 spots retenus : **zéro** facteur nul,
    142 structures dégénérées (→ +inf, la réponse prudente), et le plus petit
    facteur fini vaut 1,000000000000.

    Ce dernier chiffre est utile en soi : il dit que le minorant n'est pas 1
    au sens strict — l'arithmétique flottante descend à 0,9999999999996 sur
    les cas affines — et que la borne à écrire est donc « > 0 », pas « ≥ 1 ».
    """
    rng = random.Random(20260811)
    zeros = infs = retenus = 0
    mini = math.inf
    for _ in range(2000):
        n = rng.randint(2, 7)
        stacks = [0.0 if rng.random() < 0.30 else rng.uniform(1, 1000)
                  for _ in range(n)]
        if sum(stacks) <= 0:
            continue
        k = rng.randint(1, n + 1)
        base_p = [round(rng.uniform(0, 100), 1) for _ in range(k)]
        if rng.random() < 0.30:
            base_p = [base_p[0]] * k          # force des ex æquo
        payouts = sorted(base_p, reverse=True)
        h, v = rng.sample(range(n), 2)
        if min(stacks[h], stacks[v]) <= 0:
            continue
        retenus += 1
        bf = bubble_factor(stacks, payouts, h, v)
        assert bf > 0.0, (
            f"BF = {bf!r} casserait icm_required_equity ; "
            f"stacks={stacks} payouts={payouts} hero={h} villain={v}")
        if bf == 0.0:
            zeros += 1
        elif math.isinf(bf):
            infs += 1
        else:
            mini = min(mini, bf)
    assert retenus > 800, f"balayage trop maigre : {retenus} spots retenus"
    assert infs > 0, (
        "aucune structure dégénérée tirée : le balayage n'explore pas le cas "
        "qui produit le 0/0, et ne prouve donc rien sur lui")
    assert zeros == 0
    assert mini < 1.0 + 1e-9, (
        f"le plus petit facteur fini vaut {mini!r} — s'il montait franchement "
        "au-dessus de 1, le balayage aurait cessé de couvrir les cas affines")
