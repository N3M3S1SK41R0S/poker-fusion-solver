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


def test_un_engagement_nul_est_refuse() -> None:
    """Mieux vaut une erreur explicite qu'un spot silencieusement absurde."""
    with pytest.raises(IcmError):
        spot_pko_face_a_tapis([30.0, 1.0], [100.0, 40.0], [5.0, 5.0],
                              hero=0, villain=1, deja_engage_hero=1.0)
