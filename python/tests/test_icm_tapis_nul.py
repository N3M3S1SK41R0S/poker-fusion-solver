"""L'ICM doit supporter un tapis nul, et le spot PKO se construire sans piège.

Deux défauts rencontrés en analysant une vraie table de tournoi.

1. `bubble_factor` **plantait** (`ZeroDivisionError`) dès qu'un tapis tombait
   à zéro. Or c'est exactement ce qu'il évalue : « combien vaut mon tapis si
   je perds tout ? ». Le nombre qui dit de combien resserrer par rapport au
   cash était donc inutilisable — silencieusement, puisque l'exception
   remontait comme une panne et non comme un résultat.

2. `PkoSpot` attend des tapis **après** engagement : le vilain à tapis y
   figure à zéro, ses jetons comptés dans le pot. La convention est correcte
   mais se retourne : en lui passant les tapis lus à l'écran, le vilain
   n'était pas vu comme éliminé, la prime valait zéro, et l'équité exigée
   sortait à 90 % au lieu de 53 %. Un chiffre faux et plausible.
   `spot_pko_face_a_tapis` fait la conversion.
"""

from __future__ import annotations

import pytest

from pfs.core.icm import (
    IcmError,
    PkoSpot,
    analyse_pko_spot,
    bubble_factor,
    icm_equities,
    spot_pko_face_a_tapis,
)

PAYOUTS = [176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34]


def test_un_tapis_nul_touche_le_dernier_gain() -> None:
    """Un joueur sans jeton a fini dernier — il touche le dernier gain.

    Lui donner zéro paraît naturel et c'est faux : il a bien terminé le
    tournoi, en dernière position. L'erreur n'est pas cosmétique, elle
    surestime ce qu'on perd en perdant tout, donc la pression de bulle.
    """
    eq = icm_equities([0.0, 10.0, 20.0, 30.0], [100.0, 50.0, 25.0, 10.0])
    assert eq[0] == pytest.approx(10.0), "il doit recevoir le gain de la 4e place"
    assert eq[1] < eq[2] < eq[3], "les autres restent classés par tapis"


def test_plusieurs_tapis_nuls_se_partagent_les_dernieres_places() -> None:
    """Rien ne permet de départager deux joueurs déjà éliminés."""
    eq = icm_equities([0.0, 0.0, 20.0, 30.0], [100.0, 50.0, 25.0, 10.0])
    assert eq[0] == pytest.approx(eq[1])
    assert eq[0] + eq[1] == pytest.approx(25.0 + 10.0)
    assert sum(eq) == pytest.approx(185.0)


def test_les_cas_qui_departagent_les_conventions_du_tapis_nul() -> None:
    """Quatre cas où trois conventions défendables donnent des réponses différentes.

    Un joueur à zéro peut être traité de trois façons : exclu du calcul,
    crédité du dernier gain de la grille complète, ou crédité du dernier gain
    **parmi les places encore en jeu**. Les trois passent le test de
    conservation, et deux d'entre elles rendent des `bubble_factor` faux sur
    une bulle. La convention retenue ici est la troisième.

    Ces cas ont été proposés par une revue externe comme départageurs. Ils
    sont conservés tels quels : c'est leur pouvoir de discrimination qui fait
    leur valeur, pas leur réalisme.
    """
    # Le dernier gain vaut zéro : le mort ne doit RIEN toucher, et les deux
    # vivants à égalité se partagent 50 et 30.
    eq = icm_equities([100.0, 100.0, 0.0], [50.0, 30.0, 0.0])
    assert eq == pytest.approx([40.0, 40.0, 0.0])

    # Deux morts, grille complète : ils se partagent 30 + 20 à parts égales.
    # Les départager par index romprait l'invariance par permutation.
    eq = icm_equities([100.0, 0.0, 0.0], [50.0, 30.0, 20.0])
    assert eq == pytest.approx([50.0, 25.0, 25.0])
    assert eq[1] == pytest.approx(eq[2]), "les morts doivent être symétriques"

    # Winner-takes-all : il n'existe pas de deuxième prix à aller chercher.
    assert icm_equities([100.0, 0.0], [100.0]) == pytest.approx([100.0, 0.0])
    assert icm_equities([100.0, 0.0, 0.0], [100.0]) == pytest.approx(
        [100.0, 0.0, 0.0])


def test_l_equite_ne_depend_pas_de_l_unite_des_jetons() -> None:
    """Invariance d'échelle — l'invariant qui attrape l'ancienne « amputation ».

    Un calcul qui dépendrait de la valeur ABSOLUE des jetons plutôt que de
    leurs proportions se trahit ici. C'est exactement ce que faisait le
    contournement précédent, qui retirait 1e-9 du tapis : sur des jetons
    comptés en millions, retirer 1e-9 ne change rien ; sur des tapis comptés
    en blindes, cela déplace le résultat.
    """
    gains = [50.0, 30.0, 20.0]
    base = icm_equities([30.0, 100.0, 20.0], gains)
    for facteur in (1e-3, 1e3, 1e6):
        mis = icm_equities([30.0 * facteur, 100.0 * facteur, 20.0 * facteur],
                           gains)
        assert mis == pytest.approx(base, abs=1e-9), (
            f"l'équité dépend de l'unité des jetons (facteur {facteur:g})")


def test_les_gains_sont_conserves_avec_un_tapis_nul() -> None:
    """La somme des équités reste la dotation : rien ne se crée ni ne se perd."""
    payouts = [100.0, 50.0, 25.0, 10.0]
    eq = icm_equities([0.0, 10.0, 20.0, 30.0], payouts)
    assert sum(eq) == pytest.approx(sum(payouts), rel=1e-6)


def test_le_facteur_de_bulle_se_calcule() -> None:
    """C'est le calcul que le défaut rendait impossible.

    Un facteur > 1 signifie qu'il faut plus d'équité qu'en cash. Face à un
    adversaire qui nous couvre largement, il doit être nettement plus élevé
    que face à un joueur plus court : c'est tout le sens de la pression de
    bulle.
    """
    stacks = [35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0]
    contre_gros = bubble_factor(stacks, PAYOUTS, hero=0, villain=1)
    contre_court = bubble_factor(stacks, PAYOUTS, hero=0, villain=3)
    assert contre_gros > 1.0 and contre_court > 1.0
    assert contre_gros > contre_court, (
        f"la pression devrait être plus forte face au gros tapis "
        f"({contre_gros:.2f}) que face au court ({contre_court:.2f})")


def test_le_spot_pko_place_le_vilain_a_zero() -> None:
    """Les tapis lus à l'écran deviennent des tapis après engagement."""
    s = spot_pko_face_a_tapis(
        [35.64, 19.25], [100.0, 40.0], [8.13, 4.17], hero=0, villain=1,
        blindes_mortes=1.4, deja_engage_hero=1.0)
    assert s.stacks[1] == pytest.approx(0.0), "le vilain à tapis doit être à 0"
    assert s.pot == pytest.approx(19.25 + 1.4)
    assert s.bet == pytest.approx(19.25 - 1.0)


def test_la_prime_est_bien_capturee_quand_on_couvre() -> None:
    """Couvrir le vilain doit valoir sa prime, et alléger l'équité exigée.

    C'est le résultat que la mauvaise construction du spot faisait
    disparaître : elle rendait une prime nulle et un seuil de 90 %.
    """
    tapis = [35.64, 125.74, 27.41, 19.25] + [52.0] * 5
    primes = [8.13, 8.97, 7.43, 4.17] + [6.0] * 5

    spot = spot_pko_face_a_tapis(tapis, PAYOUTS, primes, hero=0, villain=3,
                                 blindes_mortes=1.4, deja_engage_hero=1.0)
    a = analyse_pko_spot(spot)
    assert a.villain_eliminated, "un vilain couvert doit être éliminable"
    assert a.bounty_value > 0.0, "sa prime doit avoir une valeur"
    assert a.required_with_bounty < a.required_no_bounty, (
        "la prime doit abaisser l'équité exigée")
    assert a.discount_pts == pytest.approx(
        a.required_no_bounty - a.required_with_bounty)
    assert a.required_with_bounty < 0.70, (
        f"seuil implausible ({a.required_with_bounty:.1%}) — signe que le "
        "vilain n'est pas compté comme éliminé")


def test_un_vilain_qui_nous_couvre_ne_rapporte_aucune_prime() -> None:
    """On ne peut pas éliminer plus gros que soi : la prime reste à zéro."""
    tapis = [35.64, 125.74, 27.41, 19.25] + [52.0] * 5
    primes = [8.13, 8.97, 7.43, 4.17] + [6.0] * 5
    a = analyse_pko_spot(spot_pko_face_a_tapis(
        tapis, PAYOUTS, primes, hero=0, villain=1,
        blindes_mortes=1.4, deja_engage_hero=1.0))
    assert not a.villain_eliminated
    assert a.bounty_value == pytest.approx(0.0)
    assert a.required_with_bounty == pytest.approx(a.required_no_bounty)


def test_a_trois_le_facteur_depend_vraiment_de_la_structure() -> None:
    """Le vrai test de non-dégénérescence : n = 3, où 1,0 n'est plus la réponse.

    À deux joueurs, l'ICM est affine en jetons et le bubble factor vaut 1,0
    pour *toute* implémentation — y compris une qui renverrait la constante.
    Le contrôle « heads-up = 1,0 » ne prouve donc rien. À trois joueurs il
    doit s'écarter de 1, croître avec le tapis de l'adversaire, et réagir à
    la platitude de la structure de gains.
    """
    tapis = [30.0, 100.0, 20.0]
    plate = [40.0, 35.0, 25.0]      # gains resserrés
    raide = [90.0, 8.0, 2.0]        # presque winner-take-all

    for gains in (plate, raide):
        bf = bubble_factor(tapis, gains, hero=0, villain=1)
        assert bf > 1.0, (
            f"à trois joueurs le facteur doit dépasser 1 ({bf:.4f} sur "
            f"{gains}) : une implémentation dégénérée rendrait 1,0")

    # Le sens de cette inégalité est contre-intuitif, et il a été établi par
    # la mesure après qu'une première version de ce test l'ait posé à
    # l'envers (mesuré : 2,60 sur la structure plate contre 1,15 sur la
    # raide). Une structure PLATE met le plus de pression : monter d'une
    # place rapporte presque autant que gagner, donc sauter coûte cher.
    # Une structure raide s'approche du winner-take-all, où seule la
    # première place compte — la pression ICM s'évanouit et le facteur tend
    # vers 1, c'est-à-dire vers le chipEV.
    assert (bubble_factor(tapis, plate, hero=0, villain=1)
            > bubble_factor(tapis, raide, hero=0, villain=1)), (
        "une structure plate doit peser PLUS qu'une structure raide")

    # Et il doit croître avec le tapis d'en face : c'est tout le sens de la
    # pression de bulle.
    contre_gros = bubble_factor(tapis, plate, hero=0, villain=1)
    contre_court = bubble_factor(tapis, plate, hero=0, villain=2)
    assert contre_gros > contre_court, (
        f"pression identique face au gros ({contre_gros:.3f}) et au court "
        f"({contre_court:.3f})")


def test_un_engagement_nul_est_refuse() -> None:
    """Mieux vaut une erreur explicite qu'un spot silencieusement absurde."""
    with pytest.raises(IcmError):
        spot_pko_face_a_tapis([30.0, 1.0], [100.0, 40.0], [5.0, 5.0],
                              hero=0, villain=1, deja_engage_hero=1.0)
