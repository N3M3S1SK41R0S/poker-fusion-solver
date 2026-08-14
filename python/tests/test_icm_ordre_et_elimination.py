"""Quatre défauts de `pfs.core.icm`, chacun avec son cas minimal.

Ce que ce fichier protège, et pourquoi chaque test MORD
-------------------------------------------------------
A. **Le vilain est-il éliminé ?** La question était tranchée par un seuil
   ``s[v] <= 1e-12 · Σ tapis``, c'est-à-dire « zéro exact aux arrondis près ».
   Sur une table finale de MTT PKO réel (3 337 entrants × 5 000 jetons,
   6 685 000 jetons en jeu) ce seuil vaut 6,7e-6 jeton : le moindre résidu —
   un millième de jeton d'arrondi d'ante, ou le jeton entier de l'exemple du
   tour 4 — fait tomber la prime à zéro et passer l'équité exigée du mauvais
   côté du fold (49,95 % → 53,59 % ici ; 36,6 % → 42,2 % sur l'exemple du
   tour 4). La réponse est maintenant DÉCLARÉE (``villain_all_in``), à défaut
   lue en unités de jeton entier (``unite_jeton``), et en dernier repli jugée
   à l'échelle de la TRANSACTION qui a produit le résidu
   (``SEUIL_RESIDU_TRANSACTION · max(pot, bet)``) — plus jamais aux tapis des
   autres joueurs, qui n'entrent pas dans le ``stack − bet`` d'où sort le
   nombre jugé.

B. **Qui est sorti en dernier ?** Les joueurs à tapis nul se partageaient les
   places restantes à parts égales. C'est la bonne convention d'IGNORANCE,
   mais `bubble_factor` n'ignore rien : dans l'état « le héros perd », le
   héros vient de mourir, donc APRÈS tous ceux qui étaient déjà à zéro. Le
   partage égal lui donnait la moyenne des places restantes au lieu de la
   meilleure — 15 au lieu de 20 sur ``[100, 100, 0, 0]`` avec 50/30/20/10,
   soit 25 % d'erreur sur le NUMÉRATEUR du facteur de bulle. Mesuré sur une
   vraie table PMU à 9 joueurs dont un déjà éliminé : facteur 1,900 au lieu de
   1,612, équité exigée 48,7 % au lieu de 44,6 %.

   Ce n'est pas un arbitrage de conventions : c'est la LIMITE de la récursion.
   Un héros à ε jeton (l'autre mort restant à zéro exact) vaut 20,0000002 à
   ε = 1e-6 et 20,000000000 à ε = 1e-9. Le partage égal est discontinu en
   zéro ; l'ordre déclaré est continu.

C. **Combien de places existent ?** ``_validate`` tronquait les gains au
   nombre de joueurs sans rien dire. La troncature est correcte (une place
   déjà payée n'est versée à personne) mais indiscernable d'une saisie
   incomplète des tapis. Elle est désormais annoncée par `PayoutsTronques`.

D. **Que valent les équités requises hors bande ?** ``min(max(r, 0), 1)``
   effaçait deux informations. Elles sortent maintenant en clair.

Chaque test a été vérifié PAR MUTATION : le protocole et les résultats sont
dans `test_les_mutations_font_tomber_ces_tests`.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from pfs.core.icm import (
    IcmError,
    PayoutsTronques,
    PkoSpot,
    _equite_requise,
    _validate,
    _verdict_pko,
    analyse_pko_spot,
    bubble_factor,
    icm_dollar_ev,
    icm_equities,
    icm_required_equity,
    spot_pko_face_a_tapis,
)

# ── Une table finale de MTT PKO réel ──────────────────────────────────────
# 3 337 entrants, 5 000 jetons de départ. Les tapis de la table finale font
# 6 685 000 jetons : c'est CETTE échelle qui rend le seuil relatif inopérant.
MTT_TAPIS = (742_000.0, 1_310_000.0, 455_000.0, 288_000.0,
             910_000.0, 620_000.0, 1_100_000.0, 830_000.0, 430_000.0)
MTT_GAINS = (176.0, 124.0, 92.0, 70.0, 55.0, 44.0, 33.0, 20.0, 9.34)
MTT_PRIMES = (8.13, 8.97, 7.43, 4.17, 6.0, 6.0, 6.0, 6.0, 6.0)

# La même table PMU à 9 joueurs que le reste du dépôt, mais avec un joueur
# DÉJÀ éliminé : c'est la configuration où l'ordre d'élimination cesse d'être
# inconnu, donc celle où le partage égal devient faux.
PMU_UN_MORT = (35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 0.0)


def _spot_mtt(residu: float, **kw) -> PkoSpot:
    """Le vilain (siège 3) part à tapis, le héros (siège 0) le couvre.

    Parameters
    ----------
    residu : float
        Jetons que l'appelant a laissés au vilain au lieu de zéro : arrondi
        d'ante, tapis lu à l'écran, division par la grosse blinde.
    """
    tapis = list(MTT_TAPIS)
    engage = MTT_TAPIS[3] - residu
    tapis[3] = residu
    return PkoSpot(stacks=tuple(tapis), payouts=MTT_GAINS,
                   bounties=MTT_PRIMES, hero=0, villain=3,
                   pot=engage + 20_000.0, bet=engage, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# A — LE VILAIN EST-IL ÉLIMINÉ ?
# ═══════════════════════════════════════════════════════════════════════════


def test_le_repli_juge_le_residu_a_l_echelle_de_la_transaction() -> None:
    """Un artefact de saisie ne fait plus tomber la prime — jeton du tour 4 inclus.

    L'ancien repli ``s[v] <= 1e-12 · Σ tapis`` comparait le résidu aux tapis
    des AUTRES joueurs, qui n'entrent jamais dans le ``stack − bet`` qui l'a
    produit : sur cette table (6 685 000 jetons) il valait 6,7e-6 jeton, et
    un millième de jeton d'arrondi d'ante — puis le jeton entier de l'exemple
    du tour 4 — faisait tomber la prime en silence, l'équité exigée passant
    de 49,95 % à 53,59 % (mesuré ici même, sur l'ancien code).

    Le repli juge désormais le résidu devant ``max(pot, bet)`` — ici
    1e-5 × 307 999 ≈ 3,08 jetons. Le bruit flottant (~1e-12 de la
    transaction), le millième de jeton et le jeton de saisie passent
    dessous ; un vrai tapis de 1 000 jetons — le pas des tapis de cette
    table — reste un joueur vivant.
    """
    plein = analyse_pko_spot(_spot_mtt(0.0))
    assert plein.villain_eliminated
    assert plein.source_elimination == "échelle de la transaction (repli)"
    assert plein.required_with_bounty == pytest.approx(0.4995, abs=5e-4)

    for residu in (1e-3, 1.0):
        a = analyse_pko_spot(_spot_mtt(residu))
        assert a.villain_eliminated, (
            f"un résidu de {residu:g} jeton devant un all-in de 288 000 est "
            f"un artefact de saisie, pas un joueur")
        assert a.bounty_value > 0.0
        assert a.required_with_bounty == pytest.approx(
            plein.required_with_bounty, abs=1e-5)

    vrai = analyse_pko_spot(_spot_mtt(1000.0))
    assert not vrai.villain_eliminated, (
        "1 000 jetons — le pas des tapis de cette table — est un joueur vivant")
    assert vrai.bounty_value == 0.0
    ecart = vrai.required_with_bounty - plein.required_with_bounty
    assert ecart > 0.03, (
        f"perdre la prime doit coûter plus de 3 points d'équité, mesuré "
        f"{ecart * 100:.2f}")


def test_l_artefact_et_le_vrai_jeton_sur_le_spot_calculable_a_la_main() -> None:
    """Golden à la main : winner-take-all 3-way, tapis égaux, prime 50.

    États (pot 100 déjà au milieu, bet 100, héros 0 contre vilain 1) :

    * fold  → le pot revient au vilain : (100, 100, 100), trois tapis
      égaux, d'où $EV_fold = 100/3 ;
    * gagne → (200, ~0, 100) : $EV_gagne = 200/300 · 100 = 200/3 ;
    * perd  → (0, 200, 100) : dernier d'un winner-take-all, $EV_perd = 0.

    Prime capturée : BV = 50 · (1/2 + 1/2 · P(1er)) avec P(1er) = 200/300,
    donc BV = 50 · 5/6 = 125/3. Équité requise avec prime :

        r* = (100/3 − 0) / (200/3 + 125/3 − 0) = 100/325 = 4/13 ≈ 0,3077.

    Un résidu de 1e-4 — un MILLIONIÈME de la transaction, donc un artefact —
    doit rendre ce même 4/13. Un résidu de 1 jeton — un CENTIÈME de la
    transaction, donc un joueur — doit rendre le spot sans prime :
    r* = (100/301) / (200/301) = 1/2 exactement, quelle que soit l'échelle.
    """
    def spot(residu: float) -> PkoSpot:
        return PkoSpot(stacks=(100.0, residu, 100.0), payouts=(100.0,),
                       bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
                       pot=100.0, bet=100.0)

    artefact = analyse_pko_spot(spot(1e-4))
    assert artefact.villain_eliminated
    assert artefact.bounty_value == pytest.approx(125 / 3, abs=1e-4)
    assert artefact.required_with_bounty == pytest.approx(4 / 13, abs=1e-6)

    joueur = analyse_pko_spot(spot(1.0))
    assert not joueur.villain_eliminated
    assert joueur.bounty_value == 0.0
    assert joueur.required_with_bounty == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("k", [1e-3, 1.0, 1e3, 1e6])
def test_le_repli_transaction_est_invariant_d_echelle(k) -> None:
    """Le rapport résidu/transaction est sans dimension.

    Compter en jetons, en blindes ou en millions ne doit changer aucun
    verdict du repli — c'est le reproche qui a tué le seuil absolu, et que
    le tour 4 a retourné contre ``1e-12 · Σ tapis`` : la somme des tapis des
    autres joueurs est, pour le résidu du vilain, une unité aussi arbitraire
    qu'une constante.
    """
    def analyser(facteur: float):
        tapis = [t * facteur for t in MTT_TAPIS]
        tapis[3] = 1.0 * facteur
        engage = (MTT_TAPIS[3] - 1.0) * facteur
        return analyse_pko_spot(PkoSpot(
            stacks=tuple(tapis), payouts=MTT_GAINS, bounties=MTT_PRIMES,
            hero=0, villain=3, pot=engage + 20_000.0 * facteur, bet=engage))

    ref, got = analyser(1.0), analyser(k)
    assert ref.villain_eliminated and got.villain_eliminated
    assert got.bounty_value == pytest.approx(ref.bounty_value, abs=1e-9)
    assert got.required_with_bounty == pytest.approx(ref.required_with_bounty,
                                                     abs=1e-9)


def test_l_unite_de_jeton_tranche_ce_que_le_seuil_relatif_devinait() -> None:
    """Un millième de jeton n'existe pas : c'est zéro, et ça se décide en jetons.

    Le critère devient « moins d'un demi-jeton », seul seuil qui ait un sens
    physique — entre zéro et un jeton, il n'y a rien à une table de poker.
    """
    a = analyse_pko_spot(_spot_mtt(1e-3, unite_jeton=1.0))
    assert a.villain_eliminated
    assert a.source_elimination.startswith("unité de jeton")
    assert a.bounty_value > 0.0
    assert a.required_with_bounty == pytest.approx(
        analyse_pko_spot(_spot_mtt(0.0)).required_with_bounty, abs=1e-6)


def test_un_vilain_qui_garde_vraiment_un_jeton_n_est_pas_eliminable() -> None:
    """Le pendant : l'unité de jeton ne doit pas éliminer un joueur vivant.

    Un jeton, c'est un joueur encore assis. La prime n'est pas capturable, et
    le critère doit le dire — sinon on aurait remplacé un faux négatif par un
    faux positif.
    """
    a = analyse_pko_spot(_spot_mtt(1.0, unite_jeton=1.0))
    assert not a.villain_eliminated
    assert a.bounty_value == 0.0


@pytest.mark.parametrize("k", [1e-3, 1.0, 1e3, 1e6])
def test_le_couple_tapis_unite_reste_invariant_d_echelle(k) -> None:
    """L'arbitrage assumé, vérifié : l'unité voyage AVEC les tapis.

    Comparer en « jetons entiers » n'est pas invariant d'échelle en soi — un
    jeton est une grandeur dimensionnée. Ça le redevient dès que l'unité
    accompagne les nombres : multiplier les tapis ET ``unite_jeton`` par le
    même facteur ne doit changer aucun verdict.
    """
    def analyser(facteur: float):
        tapis = [t * facteur for t in MTT_TAPIS]
        tapis[3] = 1e-3 * facteur
        engage = (MTT_TAPIS[3] - 1e-3) * facteur
        return analyse_pko_spot(PkoSpot(
            stacks=tuple(tapis), payouts=MTT_GAINS, bounties=MTT_PRIMES,
            hero=0, villain=3, pot=engage + 20_000.0 * facteur, bet=engage,
            unite_jeton=1.0 * facteur))

    ref, got = analyser(1.0), analyser(k)
    assert got.villain_eliminated is ref.villain_eliminated
    assert got.bounty_value == pytest.approx(ref.bounty_value, abs=1e-9)
    assert got.required_with_bounty == pytest.approx(ref.required_with_bounty,
                                                     abs=1e-9)


def test_une_declaration_qui_contredit_les_tapis_est_refusee() -> None:
    """Le piège de la convention `PkoSpot`, refusé au lieu d'être subi.

    Déclarer « il est à tapis » tout en laissant ses jetons dans ``stacks``,
    c'est exactement l'erreur qui sortait 90 % au lieu de 53 % (docstring de
    `spot_pko_face_a_tapis`). Refuser vaut mieux que calculer.
    """
    with pytest.raises(IcmError, match="villain_all_in=True"):
        analyse_pko_spot(_spot_mtt(1.0, villain_all_in=True))
    # …et la déclaration cohérente passe.
    assert analyse_pko_spot(_spot_mtt(0.0, villain_all_in=True)).villain_eliminated


def test_une_declaration_negative_prime_sur_un_tapis_a_zero() -> None:
    """``villain_all_in=False`` doit primer, même sur un tapis à zéro.

    Un tapis à zéro dans ``stacks`` ne prouve pas que le héros couvre : il
    peut aussi être à tapis pour plus que le héros dans un pot secondaire.
    L'appelant a le dernier mot, sinon la déclaration ne servirait à rien.
    """
    a = analyse_pko_spot(_spot_mtt(0.0, villain_all_in=False))
    assert not a.villain_eliminated
    assert a.bounty_value == 0.0
    assert a.source_elimination == "déclaré par l'appelant"


def test_spot_pko_face_a_tapis_declare_la_couverture_qu_il_a_calculee() -> None:
    """Le constructeur SAIT qui couvre qui : il ne doit plus le faire deviner."""
    tapis = [35.64, 125.74, 27.41, 19.25] + [52.0] * 5
    primes = [8.13, 8.97, 7.43, 4.17] + [6.0] * 5

    couvre = spot_pko_face_a_tapis(tapis, MTT_GAINS, primes, hero=0, villain=3,
                                   blindes_mortes=1.4, deja_engage_hero=1.0)
    assert couvre.villain_all_in is True
    assert analyse_pko_spot(couvre).source_elimination == "déclaré par l'appelant"

    couvert = spot_pko_face_a_tapis(tapis, MTT_GAINS, primes, hero=0, villain=1,
                                    blindes_mortes=1.4, deja_engage_hero=1.0)
    assert couvert.villain_all_in is False
    assert not analyse_pko_spot(couvert).villain_eliminated


def test_une_unite_de_jeton_absurde_est_refusee() -> None:
    with pytest.raises(IcmError, match="unite_jeton"):
        analyse_pko_spot(_spot_mtt(0.0, unite_jeton=0.0))


# ═══════════════════════════════════════════════════════════════════════════
# B — QUI EST SORTI EN DERNIER ?
# ═══════════════════════════════════════════════════════════════════════════


def test_le_dernier_elimine_prend_la_meilleure_place_restante() -> None:
    """Le cas minimal du défaut B, dans les deux sens.

    ``[100, 100, 0, 0]`` avec 50/30/20/10 : les deux morts valent 15 chacun
    tant qu'on ignore l'ordre. Déclarer que le joueur 3 est sorti AVANT le
    joueur 2 leur rend 10 et 20 — la 4ᵉ et la 3ᵉ place.
    """
    ignorance = icm_equities((100.0, 100.0, 0.0, 0.0), (50.0, 30.0, 20.0, 10.0))
    assert ignorance == pytest.approx([40.0, 40.0, 15.0, 15.0])

    connu = icm_equities((100.0, 100.0, 0.0, 0.0), (50.0, 30.0, 20.0, 10.0),
                         ordre_elimination=[3, 2])
    assert connu == pytest.approx([40.0, 40.0, 20.0, 10.0])

    # …et l'ordre inverse échange les deux, sinon le paramètre ne serait pas lu.
    inverse = icm_equities((100.0, 100.0, 0.0, 0.0), (50.0, 30.0, 20.0, 10.0),
                           ordre_elimination=[2, 3])
    assert inverse == pytest.approx([40.0, 40.0, 10.0, 20.0])


def test_l_ordre_declare_conserve_la_dotation() -> None:
    """Redistribuer n'est pas créer : la somme reste la dotation attribuable."""
    rng = random.Random(20260811)
    pire = 0.0
    for _ in range(400):
        n = rng.randint(2, 8)
        stacks = [0.0 if rng.random() < 0.4 else rng.uniform(1.0, 1000.0)
                  for _ in range(n)]
        if sum(stacks) <= 0:
            continue
        m = rng.randint(1, n)
        payouts = sorted((rng.uniform(1.0, 200.0) for _ in range(m)),
                         reverse=True)
        morts = [i for i, x in enumerate(stacks) if x <= 0.0]
        rng.shuffle(morts)
        ordre = morts[:rng.randint(0, len(morts))]
        eq = icm_equities(stacks, payouts, ordre_elimination=ordre)
        attendu = sum(payouts[:n])
        pire = max(pire, abs(float(eq.sum()) - attendu) / attendu)
    assert pire < 1e-12, f"écart relatif maximal {pire:.3e}"


def test_l_ordre_par_defaut_reste_le_partage_egal() -> None:
    """Non-régression de la convention d'IGNORANCE, qui reste la bonne.

    Sans déclaration, deux morts ne sont pas départageables : leur donner des
    équités différentes violerait l'invariance par permutation. Le paramètre
    ajoute une possibilité, il ne change pas le défaut.
    """
    assert icm_equities((100.0, 0.0, 0.0), (50.0, 30.0, 20.0)) == pytest.approx(
        [50.0, 25.0, 25.0])
    assert icm_equities((100.0, 0.0, 0.0), (50.0, 30.0, 20.0),
                        ordre_elimination=None) == pytest.approx(
        [50.0, 25.0, 25.0])
    assert icm_equities((100.0, 0.0, 0.0), (50.0, 30.0, 20.0),
                        ordre_elimination=[]) == pytest.approx(
        [50.0, 25.0, 25.0])


def test_les_morts_non_declares_sortent_avant_les_declares() -> None:
    """La déclaration partielle a un sens précis, et il est vérifié.

    Déclarer un seul mort, c'est dire « celui-là est sorti en dernier ». Les
    autres se partagent ce qui reste — la convention d'ignorance appliquée au
    sous-ensemble encore indistinct.
    """
    eq = icm_equities((100.0, 0.0, 0.0, 0.0), (50.0, 30.0, 20.0, 10.0),
                      ordre_elimination=[1])
    assert eq[1] == pytest.approx(30.0), "le déclaré prend la 2ᵉ place"
    assert eq[2] == pytest.approx(15.0)   # (20 + 10) / 2
    assert eq[3] == pytest.approx(15.0)
    assert float(eq.sum()) == pytest.approx(110.0)


def test_moins_de_places_que_de_morts_ne_cree_pas_de_gain() -> None:
    """Grille plus courte que la table : les places manquantes valent zéro.

    4 joueurs, 3 gains, 2 morts : il ne reste qu'une place payée. Le dernier
    sorti la prend, l'autre ne touche rien — et la somme reste la dotation.
    """
    eq = icm_equities((100.0, 100.0, 0.0, 0.0), (50.0, 30.0, 20.0),
                      ordre_elimination=[3, 2])
    assert eq[2] == pytest.approx(20.0)
    assert eq[3] == pytest.approx(0.0)
    assert float(eq.sum()) == pytest.approx(100.0)


@pytest.mark.parametrize("ordre,motif", [
    ([0], "encore"),            # joueur vivant
    ([9], "hors table"),        # indice hors table
    ([2, 2], "double"),         # doublon
])
def test_un_ordre_incoherent_est_refuse(ordre, motif) -> None:
    """Mieux vaut une erreur explicite qu'un classement silencieusement faux."""
    with pytest.raises(IcmError, match=motif):
        icm_equities((100.0, 100.0, 0.0, 0.0), (50.0, 30.0, 20.0, 10.0),
                     ordre_elimination=ordre)


def _bf_partage_egal(stacks, payouts, hero: int, villain: int) -> float:
    """TÉMOIN : le facteur de bulle AVANT correction (partage égal partout).

    Reproduit l'arithmétique exacte de `bubble_factor`, mais sans déclarer
    l'ordre d'élimination. Un test que ce témoin ne fait pas tomber ne mesure
    rien.
    """
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = icm_dollar_ev(s, p, hero)
    win = list(s); win[hero] += amt; win[villain] -= amt
    lose = list(s); lose[hero] -= amt; lose[villain] += amt
    return ((base - icm_dollar_ev(lose, p, hero))
            / (icm_dollar_ev(win, p, hero) - base))


def test_le_facteur_de_bulle_calcule_a_la_main() -> None:
    """Malmuth-Harville à la main sur ``[100, 100, 0, 0]``, gains 50/30/20/10.

    Base : deux vivants à tapis égaux, P(1er) = P(2e) = 1/2, donc
    $EV_héros = (50 + 30)/2 = 40. All-in du héros 0 contre le vilain 1
    (amt = 100) :

    * gagne → (200, 0, 0, 0) : héros seul vivant, $EV_gagne = 50 ;
    * perd  → (0, 200, 0, 0) : le héros sort en DERNIER (l'autre mort
      l'était déjà avant la main), il prend la meilleure place restante,
      la 2e : $EV_perd = 30. Le témoin du partage égal lui donne la
      moyenne des places restantes : (30 + 20 + 10)/3 = 20.

    D'où, exactement, sans machine :

        BF_prod   = (40 − 30) / (50 − 40) = 1
        BF_témoin = (40 − 20) / (50 − 40) = 2

    Le cas minimal où la convention change le facteur du simple au double —
    c'est lui qui ancre les goldens à 9 joueurs du test suivant.
    """
    stacks = (100.0, 100.0, 0.0, 0.0)
    gains = (50.0, 30.0, 20.0, 10.0)
    assert bubble_factor(stacks, gains, 0, 1) == pytest.approx(1.0, abs=1e-12)
    assert _bf_partage_egal(stacks, gains, 0, 1) == pytest.approx(2.0,
                                                                  abs=1e-12)


@pytest.mark.parametrize("hero,villain,avant,apres", [
    (3, 1, 1.9002683, 1.6118303),
    (0, 1, 2.2613563, 2.0334389),
    (2, 4, 1.9392982, 1.7084969),
    (3, 0, 1.7140596, 1.4538859),
])
def test_le_facteur_de_bulle_change_sur_une_vraie_table(
        hero, villain, avant, apres) -> None:
    """L'effet mesuré sur une table PMU à 9 joueurs dont un déjà éliminé.

    Les valeurs « avant » viennent du témoin `_bf_partage_egal`, les
    « après » de la fonction de production. L'écart va de 10 à 15 %, soit
    2,7 à 4,1 points d'équité requise — largement de quoi retourner une
    décision push/fold.

    PROVENANCE DES GOLDENS — l'arbitrage du 14 août 2026 : les valeurs
    enregistrées à mi-chantier (avant : 1,900342 ; 2,261386 ; 1,939333 ;
    1,714123) s'écartaient du témoin actuel de 1,3e-5 à 3,9e-5 en relatif.
    Conformément à l'avertissement du chantier — ne JAMAIS élargir une
    tolérance pour faire passer un test — les deux candidats ont été
    départagés par un recalcul INDÉPENDANT : Malmuth-Harville par
    énumération complète des 8! = 40 320 ordres d'arrivée des vivants
    (P(ordre) = Π sᵢ/reste), sans aucun import de `pfs.core.icm`.
    L'énumération reproduit le témoin actuel à mieux que 1e-9 en relatif
    sur les quatre cas (p. ex. héros 3 contre vilain 0 : 1,7140596405
    contre 1,7140596405) ; les goldens de mi-chantier étaient donc les
    valeurs fausses, enregistrées contre une implémentation qui a bougé
    ensuite. Ce sont les goldens qui ont été corrigés, pas le code. La
    convention elle-même est ancrée par le calcul à la main du test
    précédent.
    """
    temoin = _bf_partage_egal(PMU_UN_MORT, MTT_GAINS, hero, villain)
    prod = bubble_factor(PMU_UN_MORT, MTT_GAINS, hero, villain)
    assert temoin == pytest.approx(avant, rel=1e-5)
    assert prod == pytest.approx(apres, rel=1e-5)
    assert prod < temoin, "le partage égal SURESTIMAIT la pression de bulle"

    ecart_pts = (icm_required_equity(100.0, 100.0, temoin)
                 - icm_required_equity(100.0, 100.0, prod))
    assert ecart_pts > 0.025, (
        f"l'effet doit dépasser 2,5 points d'équité, mesuré "
        f"{ecart_pts * 100:.2f}")


def _bf_hero_a_epsilon(stacks, payouts, hero: int, villain: int,
                       eps: float) -> float:
    """Facteur de bulle où le héros perdant garde ``eps`` jeton.

    Les autres morts restent à ZÉRO exact : c'est ce qui rend la limite
    unique. Le héros à ε est un vivant comme un autre, donc aucune convention
    de partage n'intervient dans son $EV — c'est la référence externe.
    """
    s, p = _validate(stacks, payouts)
    amt = min(s[hero], s[villain])
    base = icm_dollar_ev(s, p, hero)
    win = list(s); win[hero] += amt; win[villain] -= amt
    lose = list(s); lose[hero] -= amt - eps; lose[villain] += amt - eps
    return ((base - icm_dollar_ev(lose, p, hero))
            / (icm_dollar_ev(win, p, hero) - base))


@pytest.mark.parametrize("hero,villain", [(3, 1), (0, 1), (2, 4), (3, 0)])
def test_l_ordre_declare_est_la_limite_de_la_recursion(hero, villain) -> None:
    """Ce n'est pas une convention : c'est la valeur en zéro.

    On fait tendre le tapis résiduel du héros vers zéro en laissant l'autre
    mort à zéro exact. La suite converge — et elle converge vers la valeur que
    rend l'ordre déclaré, pas vers le partage égal. Le témoin s'en écarte de
    plus de 10 % et n'y converge jamais : il fait un PALIER.
    """
    prod = bubble_factor(PMU_UN_MORT, MTT_GAINS, hero, villain)
    temoin = _bf_partage_egal(PMU_UN_MORT, MTT_GAINS, hero, villain)

    court = min(PMU_UN_MORT[hero], PMU_UN_MORT[villain])
    precedent = math.inf
    for eps in (1e-2, 1e-4, 1e-6, 1e-9):
        ecart = abs(_bf_hero_a_epsilon(PMU_UN_MORT, MTT_GAINS, hero, villain,
                                       eps) - prod)
        assert ecart <= 5.0 * eps / court + 1e-12 * prod, (
            f"ε={eps:g} : écart {ecart:.3e} au-dessus de la borne linéaire")
        assert ecart <= precedent + 1e-15, "la suite doit décroître"
        precedent = ecart

    assert abs(temoin - prod) / prod > 0.10, (
        "le témoin doit rester loin de la limite — sinon ce test ne prouve "
        "rien")


def test_bubble_factor_refuse_qu_on_lui_dicte_l_ordre() -> None:
    """Il le connaît par construction : le lui passer ne peut que le contredire."""
    with pytest.raises(IcmError, match="détermine lui-même"):
        bubble_factor(PMU_UN_MORT, MTT_GAINS, 3, 1, ordre_elimination=[8])


def test_sans_mort_prealable_rien_ne_change() -> None:
    """Non-régression : la correction ne doit toucher QUE le cas à ≥ 2 morts.

    Quand le héros est le seul à tomber à zéro, il n'y a rien à départager et
    l'ancien partage égal donnait déjà la bonne réponse. Les goldens du dépôt
    en dépendent.
    """
    table = (35.64, 125.74, 27.41, 19.25, 52.0, 52.0, 52.0, 52.0, 52.0)
    for hero, villain in ((3, 1), (0, 1), (1, 3)):
        assert bubble_factor(table, MTT_GAINS, hero, villain) == (
            _bf_partage_egal(table, MTT_GAINS, hero, villain))


# ═══════════════════════════════════════════════════════════════════════════
# C — LA TRONCATURE DES GAINS
# ═══════════════════════════════════════════════════════════════════════════


def test_la_troncature_des_gains_est_annoncee() -> None:
    """Une table finale mal saisie ne doit plus perdre des places en silence."""
    with pytest.warns(PayoutsTronques, match="2 place"):
        eq = icm_equities((100.0, 50.0, 25.0), (50.0, 30.0, 20.0, 10.0, 5.0))
    assert float(eq.sum()) == pytest.approx(100.0), (
        "la troncature elle-même reste correcte : c'est le silence qui était "
        "le défaut")


def test_l_avertissement_nomme_les_places_perdues_et_la_dotation() -> None:
    """Un avertissement qu'on ne peut pas recouper ne sert à rien."""
    with pytest.warns(PayoutsTronques) as capture:
        icm_equities((100.0, 50.0), (50.0, 30.0, 20.0))
    message = str(capture[0].message)
    assert "20" in message and "80" in message and "100" in message


def test_une_grille_completee_de_zeros_n_avertit_pas() -> None:
    """Rien de perdu, rien à dire : sinon l'avertissement deviendrait du bruit.

    Un avertissement qui se déclenche pour rien est un avertissement qu'on
    apprend à ignorer.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", PayoutsTronques)
        eq = icm_equities((100.0, 50.0), (50.0, 30.0, 0.0, 0.0))
    assert float(eq.sum()) == pytest.approx(80.0)


def test_pas_d_avertissement_quand_la_grille_tient_dans_la_table() -> None:
    """Le cas nominal doit rester muet."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", PayoutsTronques)
        icm_equities((100.0, 50.0, 25.0), (50.0, 30.0))


def test_l_avertissement_traverse_les_points_d_entree_publics() -> None:
    """Il doit se voir depuis là où l'utilisateur appelle, pas seulement dans
    `_validate`."""
    with pytest.warns(PayoutsTronques):
        bubble_factor((100.0, 50.0), (50.0, 30.0, 20.0), 0, 1)
    with pytest.warns(PayoutsTronques):
        analyse_pko_spot(PkoSpot(
            stacks=(100.0, 0.0), payouts=(50.0, 30.0, 20.0),
            bounties=(5.0, 5.0), hero=0, villain=1, pot=50.0, bet=50.0,
            villain_all_in=True))


# ═══════════════════════════════════════════════════════════════════════════
# D — LES ÉQUITÉS REQUISES HORS BANDE
# ═══════════════════════════════════════════════════════════════════════════


def test_une_equite_requise_negative_sort_en_clair() -> None:
    """``max(r, 0)`` transformait « payer est gratuit » en « il faut 0 % »."""
    signaux: list[str] = []
    assert _equite_requise(-0.2, "PKO", signaux) == pytest.approx(-0.2)
    assert len(signaux) == 1
    assert "NÉGATIVE" in signaux[0]


def test_une_equite_requise_superieure_a_cent_pour_cent_sort_en_clair() -> None:
    """``min(r, 1)`` transformait une incohérence de saisie en « il faut 100 % »."""
    signaux: list[str] = []
    assert _equite_requise(1.4, "ICM pur", signaux) == pytest.approx(1.4)
    assert len(signaux) == 1
    assert "100 %" in signaux[0]


@pytest.mark.parametrize("r,attendu", [(-1e-15, 0.0), (1.0 + 1e-15, 1.0),
                                       (0.0, 0.0), (1.0, 1.0), (0.37, 0.37)])
def test_la_bande_de_bruit_colle_aux_bornes_sans_rien_signaler(r, attendu) -> None:
    """Un ulp de dépassement est de l'arrondi, pas une information.

    Sans cette bande, le moindre résidu de soustraction ferait crier le
    module à chaque appel — et un signal permanent n'est plus un signal.
    """
    signaux: list[str] = []
    assert _equite_requise(r, "PKO", signaux) == pytest.approx(attendu)
    assert signaux == []


def test_le_verdict_dit_ce_que_le_clamp_taisait() -> None:
    """Les deux phrases que la troncature rendait impossibles à écrire."""
    assert "CALL AUTOMATIQUE" in _verdict_pko(-0.2, 0.4, None)
    assert "FOLD FORCÉ" in _verdict_pko(1.4, 1.4, None)
    # …et le cas nominal n'est pas touché.
    assert _verdict_pko(0.4, 0.5, None).startswith("il faut")
    assert _verdict_pko(0.4, 0.5, 0.6).startswith("CALL")
    assert _verdict_pko(0.4, 0.5, 0.3).startswith("FOLD :")


def test_zero_et_un_restent_atteignables_et_muets() -> None:
    """La borne haute est ATTEINTE par un spot légitime : pot nul.

    Sans rien à gagner, il faut 100 % d'équité. C'est la preuve que la bande
    de bruit ne mord pas sur le domaine réel — et que 1,0 n'est pas un
    artefact du clamp.
    """
    a = analyse_pko_spot(PkoSpot(
        stacks=(1000.0, 500.0, 800.0), payouts=(100.0, 60.0, 40.0),
        bounties=(0.0, 0.0, 0.0), hero=0, villain=1, pot=0.0, bet=400.0))
    assert a.required_no_bounty == pytest.approx(1.0)
    assert a.signaux == ()


def test_le_domaine_zero_un_est_un_theoreme_pas_un_clamp() -> None:
    """Sur le chemin EXACT, aucun spot valide ne sort de [0, 1].

    C'est ce qui justifie de SIGNALER au lieu de tronquer : la monotonie ICM
    donne $EV_perte ≤ $EV_fold ≤ $EV_gain, donc un dépassement ne peut venir
    que d'une entrée incohérente ou du bruit d'échantillonnage Monte-Carlo
    (> 12 joueurs). Le tronquer revenait à effacer la seule trace de l'un ou
    de l'autre.

    Mesuré ici sur 2 500 spots aléatoires — tapis nuls, gains ex æquo, pot
    nul inclus.
    """
    rng = random.Random(31337)
    valides = 0
    for _ in range(2500):
        n = rng.randint(2, 7)
        stacks = [0.0 if rng.random() < 0.35 else rng.uniform(1.0, 1000.0)
                  for _ in range(n)]
        if sum(stacks) <= 0:
            continue
        m = rng.randint(1, n)
        payouts = sorted((rng.uniform(1.0, 200.0) for _ in range(m)),
                         reverse=True)
        bounties = [rng.uniform(0.0, 100.0) for _ in range(n)]
        h, v = rng.sample(range(n), 2)
        if stacks[h] <= 0:
            continue
        bet = rng.uniform(1e-6, stacks[h])
        pot = rng.choice([0.0, bet, rng.uniform(0.0, 5000.0)])
        try:
            a = analyse_pko_spot(PkoSpot(tuple(stacks), tuple(payouts),
                                         tuple(bounties), h, v, pot, bet))
        except IcmError:
            continue
        valides += 1
        assert a.signaux == (), f"signal inattendu : {a.signaux}"
        assert -1e-12 <= a.required_with_bounty <= 1.0 + 1e-12
        assert -1e-12 <= a.required_no_bounty <= 1.0 + 1e-12
    assert valides > 1500, f"seulement {valides} spots valides — balayage trop maigre"


# ═══════════════════════════════════════════════════════════════════════════
# LE PROTOCOLE DE MUTATION
# ═══════════════════════════════════════════════════════════════════════════


def test_les_mutations_font_tomber_ces_tests() -> None:
    """Ce que valent les tests ci-dessus, mesuré et non affirmé.

    Chaque correction a été remise dans son état d'AVANT, une à la fois, et
    le fichier relancé. Comptes re-mesurés le 14 août 2026, après l'ajout du
    repli par transaction et des goldens à la main (pytest, ce fichier seul,
    51 tests) :

    ======================================  ==========================
    mutation appliquée à `pfs/core/icm.py`  tests en échec
    ======================================  ==========================
    A : ignorer `villain_all_in` et         11 échecs
        `unite_jeton`, revenir au seuil
        relatif `1e-12 · Σ tapis` seul
    B : `bubble_factor` cesse de déclarer   9 échecs
        l'ordre (`_avec_ordre` rend `kw`)
    B' : `_places_des_morts` ignore         12 échecs
        `ordre` (partage égal toujours)
    C : `_validate` retronque en silence    3 échecs
    D : rétablir `min(max(r, 0), 1)` et     3 échecs
        les verdicts d'origine
    ======================================  ==========================

    Ce test-ci ne vérifie que la cohérence interne du protocole : il rappelle
    les cinq mutations et échoue si l'une des fonctions qu'elles visent a
    disparu du module. Les comptes ci-dessus sont reproductibles à la main.
    """
    from pfs.core import icm

    for nom in ("_places_des_morts", "_avec_ordre", "_vilain_elimine",
                "_equite_requise", "_verdict_pko", "PayoutsTronques",
                "SEUIL_RESIDU_TRANSACTION"):
        assert hasattr(icm, nom), (
            f"{nom} a disparu : le protocole de mutation ci-dessus ne "
            f"s'applique plus tel quel")
    assert "ordre_elimination" in icm.icm_equities.__doc__
    assert np.isclose(icm.TOLERANCE_EQUITE, 1e-12)
    assert np.isclose(icm.SEUIL_RESIDU_TRANSACTION, 1e-5)
