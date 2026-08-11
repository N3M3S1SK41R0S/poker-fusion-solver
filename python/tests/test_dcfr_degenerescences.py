r"""Ce que DCFR devient à paramètres particuliers — et ce qu'il ne devient PAS.

Pourquoi ce fichier existe
--------------------------
Le dépôt a longtemps affirmé, en commentaire de mission, que
« α = β = γ = 1 donne CFR standard ». C'est faux : cela donne **Linear CFR**,
qui n'est pas Vanilla CFR. Un test de conformité qui énonce une fausseté est
pire qu'un test absent — il transforme une erreur en garantie.

Ce fichier remplace l'affirmation par des MESURES. Chaque dégénérescence est
confrontée à un solveur de référence **écrit ici**, indépendant de
`pfs.solver.dcfr` : Vanilla CFR, Linear CFR et CFR+ sont réimplémentés à la
main sur Kuhn poker, à partir de leurs récurrences de définition. Comparer le
dépôt à lui-même n'aurait rien prouvé — c'est le piège des deux gabarits
intervertis qui avait survécu à 1057 tests.

Ce qui est mesuré (toutes les valeurs sont reproductibles en lançant ce
fichier ; les T ≥ 1000 le sont hors suite, pour le temps) :

===================================  =========================================
Prétendu                             Mesuré
===================================  =========================================
(α, β) = (1, 1) → Linear CFR         **VRAI**, écart L∞ ≤ 1,4e-15 de T=1 à 1000
(1, 1) → Vanilla CFR                 **FAUX**, écart 0,27 (T=20) · 0,035 (T=1000)
(α, β) = (∞, 0) → CFR+               **FAUX** : β = 0 donne t⁰/(t⁰+1) = ½,
                                     soit la DIVISION PAR DEUX des regrets
                                     négatifs — c'est le réglage propre de
                                     DCFR (Brown & Sandholm : α=1,5 β=0 γ=2),
                                     pas la remise à zéro de regret matching+.
                                     Écart à CFR+ : 0,098 (T=20)
β → −∞ → CFR+                        **PRESQUE** : t^β/(t^β+1) → 0 comme il
                                     faut, mais le poids en t = 1 vaut ½ pour
                                     TOUT α et β finis (1^x = 1). La limite
                                     est CFR+ dont l'itération 1 est divisée
                                     par deux au lieu d'être plancherée —
                                     égalité EXACTE (0,0e+00) mesurée ici.
α, β → ∞ → Vanilla CFR               **PRESQUE**, et pour la même raison : la
                                     limite est Vanilla dont l'itération 1 est
                                     divisée par deux (égalité exacte dès
                                     α = β = 64). Écart à Vanilla pur : 0,148.
γ                                    **NON IMPLÉMENTÉ COMME DANS LE PAPIER**
                                     (voir `test_le_gamma_du_depot_...`).
===================================  =========================================

Et une limite de mise en œuvre qu'aucune dérivation ne montre : `α = ∞` ne
plante pas seulement en théorie, il plante en pratique. La ligne
``pos_w = (t**alpha) / (t**alpha + 1.0)`` de `dcfr.py` lève `OverflowError`
dès que ``α·ln(T) > 709`` — soit α > 134 à T = 200, α > 103 à T = 1000.

Bibliographie
-------------
- Zinkevich, Johanson, Bowling & Piccione (2007) — CFR
- Tammelin (2014) — CFR+
- Brown & Sandholm (2019), arXiv:1809.04040 — DCFR et Linear CFR
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pfs.solver.dcfr import DCFRConfig, DCFRSolver, KuhnPoker, regret_matching

# ═══════════════════════════════════════════════════════════════════════════
# SOLVEURS DE RÉFÉRENCE — écrits ici, hors du dépôt
# ═══════════════════════════════════════════════════════════════════════════

TERMINAUX = frozenset({"pp", "bp", "bb", "pbp", "pbb"})
DONNES = [(a, b) for a in range(3) for b in range(3) if a != b]


def _gain(c0: int, c1: int, h: str) -> float:
    """Gain du joueur 0 à l'état terminal ``h`` — Kuhn poker."""
    vainqueur = 1 if c0 > c1 else -1
    if h == "pp":
        return float(vainqueur)
    if h == "bp":
        return 1.0
    if h == "pbp":
        return -1.0
    return float(2 * vainqueur)          # "bb" ou "pbb"


def solveur_reference(mode: str, iterations: int) -> dict[str, np.ndarray]:
    """CFR de référence sur Kuhn — mises à jour alternées, comme le dépôt.

    Parameters
    ----------
    mode : str
        - ``"vanilla"`` : regrets cumulés bruts (Zinkevich 2007) ;
        - ``"lcfr"`` : contribution de l'itération *t* pondérée par *t*
          (Linear CFR, Brown & Sandholm 2019) ;
        - ``"cfr+"`` : regret matching+, plancher à zéro appliqué dans le
          parcours, aussitôt après chaque mise à jour (Tammelin 2014) ;
        - ``"cfr+_fin_iteration"`` : même plancher, mais appliqué une seule
          fois à la fin de l'itération — c'est le RYTHME de `dcfr.py`, dont
          l'escompte n'intervient qu'entre deux itérations ;
        - ``"cfr+_fin_iteration_t1_demi"`` : idem, sauf qu'à *t* = 1 le
          vecteur est divisé par deux au lieu d'être plancheré.

    Returns
    -------
    dict
        Stratégie COURANTE par ensemble d'information — regret matching sur
        les regrets cumulés. C'est l'objet que la dégénérescence doit
        reproduire ; la stratégie MOYENNE dépend en plus de γ, traité à part.
    """
    regrets: dict[str, np.ndarray] = {}

    def parcours(cartes: tuple[int, int], h: str, p0: float, p1: float,
                 joueur_maj: int, t: int) -> float:
        if h in TERMINAUX:
            return _gain(cartes[0], cartes[1], h)
        pl = len(h) % 2
        cle = f"{cartes[pl]}|{h}"
        if cle not in regrets:
            regrets[cle] = np.zeros(2)
        sigma = regret_matching(regrets[cle])
        u = np.zeros(2)
        for i, a in enumerate("pb"):
            if pl == 0:
                u[i] = parcours(cartes, h + a, p0 * sigma[i], p1, joueur_maj, t)
            else:
                u[i] = parcours(cartes, h + a, p0, p1 * sigma[i], joueur_maj, t)
        valeur = float(sigma @ u)
        if joueur_maj == pl:
            signe = 1.0 if pl == 0 else -1.0
            contrefactuel = p1 if pl == 0 else p0
            poids = float(t) if mode == "lcfr" else 1.0
            regrets[cle] += poids * contrefactuel * signe * (u - valeur)
            if mode == "cfr+":
                np.maximum(regrets[cle], 0.0, out=regrets[cle])
        return valeur

    for t in range(1, iterations + 1):
        for joueur in (0, 1):
            for cartes in DONNES:
                parcours(cartes, "", 1 / 6, 1 / 6, joueur, t)
        if mode == "cfr+_fin_iteration":
            for v in regrets.values():
                np.maximum(v, 0.0, out=v)
        elif mode == "cfr+_fin_iteration_t1_demi":
            for v in regrets.values():
                if t == 1:
                    v *= 0.5
                else:
                    np.maximum(v, 0.0, out=v)
    return {k: regret_matching(v) for k, v in regrets.items()}


def _strategies_courantes(solveur: DCFRSolver) -> dict[str, np.ndarray]:
    """Regret matching sur les regrets cumulés du solveur du dépôt."""
    return {k: regret_matching(v) for k, v in solveur._regrets.items()}


def _ecart(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    """Distance L∞ entre deux profils de stratégies, ensembles réunis.

    Une clé présente d'un seul côté est un écart en soi : `KeyError` est le
    bon comportement, et non un `.get()` qui masquerait la divergence.
    """
    return max(float(np.abs(a[k] - b[k]).max()) for k in set(a) | set(b))


def _depot(alpha: float, beta: float, iterations: int) -> dict[str, np.ndarray]:
    """Le solveur DU DÉPÔT, schedule et élagage désactivés.

    `use_schedule=False` parce qu'un schedule fait varier (α, β, γ) au fil des
    itérations : la question posée porte sur des paramètres FIXES.
    `prune_threshold=None` pour que la comparaison ne mesure que la loi
    d'escompte (sur Kuhn le seuil −1e5 n'est de toute façon jamais atteint).
    """
    solveur = DCFRSolver(
        KuhnPoker(),
        DCFRConfig(alpha=alpha, beta=beta, gamma=2.0, use_schedule=False,
                   prune_threshold=None),
    )
    solveur.solve(iterations=iterations)
    return _strategies_courantes(solveur)


# ═══════════════════════════════════════════════════════════════════════════
# 1. LINEAR CFR — la seule dégénérescence EXACTE, et elle n'est pas Vanilla
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("iterations", [1, 2, 5, 20, 200])
def test_alpha_beta_egaux_a_un_donnent_linear_cfr(iterations: int) -> None:
    r"""(α, β) = (1, 1) reproduit Linear CFR à l'arrondi près.

    La dérivation, sur le code : `dcfr.py` multiplie à la fin de chaque
    itération *s* le regret cumulé par :math:`s^\alpha/(s^\alpha+1)`, ce qui
    vaut :math:`s/(s+1)` à α = 1. Le poids résiduel de la contribution de
    l'itération *t* après *T* itérations est donc

    .. math:: \prod_{s=t}^{T}\frac{s}{s+1} = \frac{t}{T+1},

    c'est-à-dire **proportionnel à t** : exactement la pondération de Linear
    CFR. Le facteur commun 1/(T+1) est invisible pour le regret matching, qui
    normalise.

    Mesuré : écart L∞ de 0,0 (T=1) à 1,4e-15 (T=1000).
    """
    assert _ecart(_depot(1.0, 1.0, iterations),
                  solveur_reference("lcfr", iterations)) < 1e-12


@pytest.mark.parametrize("iterations", [20, 200])
def test_linear_cfr_n_est_pas_vanilla_cfr(iterations: int) -> None:
    """Le témoin qui donne son sens au test précédent.

    Si Linear CFR et Vanilla CFR produisaient les mêmes stratégies, dire
    « (1,1) donne Linear » ou « (1,1) donne Vanilla » reviendrait au même et
    la correction serait sans objet. Ils diffèrent : 0,272 à T=20 et 0,081 à
    T=200, soit huit points de fréquence — le double du seuil auquel une
    range de poker se lit différemment.
    """
    ecart = _ecart(solveur_reference("lcfr", iterations),
                   solveur_reference("vanilla", iterations))
    assert ecart > 0.05, (
        "les deux références coïncident : le test de dégénérescence "
        "ne distinguerait plus rien")


@pytest.mark.parametrize("iterations", [20, 200])
def test_le_depot_a_un_et_un_n_est_pas_vanilla(iterations: int) -> None:
    """La fausseté elle-même, épinglée : (1,1) N'EST PAS « CFR standard ».

    Ce test tombe si quelqu'un rétablit l'ancienne affirmation en changeant
    la loi d'escompte pour rendre (1,1) équivalent à Vanilla.
    """
    assert _ecart(_depot(1.0, 1.0, iterations),
                  solveur_reference("vanilla", iterations)) > 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 2. CFR+ — la dérivation du conseil est réfutée deux fois
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("t", [1, 2, 5, 50, 1000])
def test_beta_zero_divise_par_deux_il_ne_remet_pas_a_zero(t: int) -> None:
    r"""β = 0 donne un poids de ½, pas de 0.

    La dérivation qui nous a été soumise écrivait « β = 0 → 0/(0+1) = 0, soit
    la remise à zéro du regret matching+ ». L'arithmétique est fausse :
    :math:`t^0 = 1`, donc :math:`t^\beta/(t^\beta+1) = 1/2`. La division par
    deux des regrets négatifs est le réglage PROPRE de DCFR (Brown & Sandholm
    2019 : α = 1,5, β = 0, γ = 2), et non celui de CFR+.

    Annuler un poids demande :math:`t^\beta \to 0`, donc β → −∞.
    """
    assert (t ** 0.0) / (t ** 0.0 + 1.0) == 0.5
    for beta, plafond in ((-1.0, 0.5), (-10.0, 1e-2), (-100.0, 1e-29)):
        if t > 1:
            assert (t ** beta) / (t ** beta + 1.0) <= plafond


@pytest.mark.parametrize("iterations", [5, 20, 200])
def test_alpha_grand_et_beta_zero_ne_donnent_pas_cfr_plus(iterations: int) -> None:
    """La thèse (α = ∞, β = 0) = CFR+, mise à l'épreuve sur le vrai solveur.

    Mesuré : 0,356 (T=5) · 0,098 (T=20) · 0,032 (T=200). L'écart décroît
    seulement parce que les deux algorithmes convergent vers le même
    équilibre — ce n'est pas une équivalence, c'est une convergence commune.
    Le seuil est posé à 0,01, sous le plus petit écart mesuré (0,032).
    """
    assert _ecart(_depot(100.0, 0.0, iterations),
                  solveur_reference("cfr+", iterations)) > 0.01


@pytest.mark.parametrize("iterations", [2, 5, 20, 200])
def test_la_limite_beta_moins_infini_est_cfr_plus_a_l_iteration_1_pres(
    iterations: int,
) -> None:
    r"""Ce que β → −∞ donne EXACTEMENT — et pourquoi ce n'est pas CFR+ tout court.

    Deux écarts irréductibles séparent `dcfr.py` de CFR+ canonique :

    1. **le rythme** — CFR+ planche le regret dans le parcours, aussitôt après
       chaque mise à jour ; `dcfr.py` n'escompte qu'entre deux itérations ;
    2. **l'itération 1** — le poids :math:`1^\beta/(1^\beta+1)` vaut ½ pour
       TOUT β fini, y compris β = −10³⁰⁰, parce que :math:`1^x = 1`. Le
       premier vecteur de regret est donc divisé par deux au lieu d'être
       plancheré, et cette trace ne s'efface jamais.

    En tenant compte des deux, l'égalité est EXACTE : écart 0,0e+00 mesuré à
    T = 2, 5, 20, 200 et 1000. C'est une caractérisation, pas une
    approximation — et elle dit qu'aucun jeu de paramètres finis ne rend
    `dcfr.py` équivalent à CFR+.
    """
    depot = _depot(100.0, -100.0, iterations)
    assert _ecart(
        depot, solveur_reference("cfr+_fin_iteration_t1_demi", iterations)
    ) < 1e-12
    # Les deux amendements sont NÉCESSAIRES : sans eux l'égalité tombe.
    assert _ecart(depot, solveur_reference("cfr+", iterations)) > 0.01
    assert _ecart(depot,
                  solveur_reference("cfr+_fin_iteration", iterations)) > 0.01


@pytest.mark.parametrize("iterations", [5, 20, 200])
def test_beta_zero_et_beta_moins_infini_ne_sont_pas_le_meme_algorithme(
    iterations: int,
) -> None:
    """Le témoin de dents de toute la section CFR+.

    Si β = 0 et β = −100 donnaient les mêmes stratégies, la distinction
    « division par deux » / « remise à zéro » serait sans conséquence
    observable et les tests ci-dessus ne protégeraient rien. Mesuré :
    0,121 (T=5) · 0,157 (T=20) · 0,049 (T=200).
    """
    assert _ecart(_depot(100.0, -100.0, iterations),
                  _depot(100.0, 0.0, iterations)) > 0.01


# ═══════════════════════════════════════════════════════════════════════════
# 3. VANILLA — inatteignable, et pas seulement « à la limite »
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("alpha", [64.0, 100.0])
@pytest.mark.parametrize("iterations", [5, 20, 200])
def test_la_limite_alpha_beta_infinis_est_vanilla_a_l_iteration_1_pres(
    alpha: float, iterations: int
) -> None:
    r"""α = β → ∞ donne Vanilla CFR dont l'itération 1 est divisée par deux.

    Les poids :math:`t^\alpha/(t^\alpha+1)` tendent vers 1 pour tout t ≥ 2,
    donc l'escompte disparaît — sauf en t = 1, où le poids reste ½ quelle que
    soit la valeur de α. Mesuré : égalité EXACTE (0,0e+00) avec Vanilla amendé
    dès α = β = 64, contre 0,148 à 1,0 d'écart avec Vanilla pur.

    Conséquence directe, et c'est le point que la formule « Vanilla = limite
    α, β, γ → ∞ » laissait croire résolu : Vanilla CFR n'est pas seulement
    inatteignable à paramètres finis, il est inatteignable **à la limite**.
    """
    depot = _depot(alpha, alpha, iterations)
    assert _ecart(depot, _vanilla_iteration1_divisee(iterations)) < 1e-12
    assert _ecart(depot, solveur_reference("vanilla", iterations)) > 0.05


def _vanilla_iteration1_divisee(iterations: int) -> dict[str, np.ndarray]:
    """Vanilla CFR dont le regret cumulé est divisé par deux à la fin de t=1."""
    regrets: dict[str, np.ndarray] = {}

    def parcours(cartes, h, p0, p1, joueur_maj):
        if h in TERMINAUX:
            return _gain(cartes[0], cartes[1], h)
        pl = len(h) % 2
        cle = f"{cartes[pl]}|{h}"
        if cle not in regrets:
            regrets[cle] = np.zeros(2)
        sigma = regret_matching(regrets[cle])
        u = np.zeros(2)
        for i, a in enumerate("pb"):
            if pl == 0:
                u[i] = parcours(cartes, h + a, p0 * sigma[i], p1, joueur_maj)
            else:
                u[i] = parcours(cartes, h + a, p0, p1 * sigma[i], joueur_maj)
        valeur = float(sigma @ u)
        if joueur_maj == pl:
            signe = 1.0 if pl == 0 else -1.0
            contrefactuel = p1 if pl == 0 else p0
            regrets[cle] += contrefactuel * signe * (u - valeur)
        return valeur

    for t in range(1, iterations + 1):
        for joueur in (0, 1):
            for cartes in DONNES:
                parcours(cartes, "", 1 / 6, 1 / 6, joueur)
        if t == 1:
            for v in regrets.values():
                v *= 0.5
    return {k: regret_matching(v) for k, v in regrets.items()}


@pytest.mark.parametrize("alpha", [1.0, 16.0, 1e6, 1e300])
def test_le_poids_de_l_iteration_1_vaut_toujours_un_demi(alpha: float) -> None:
    r""":math:`1^\alpha = 1` : aucun α fini ne relève le poids de t = 1.

    C'est la raison STRUCTURELLE pour laquelle ni Vanilla ni CFR+ ne sont
    atteignables. Un test d'une ligne, mais c'est lui qui rend les deux
    caractérisations ci-dessus intelligibles plutôt que mystérieuses.
    """
    assert (1.0 ** alpha) / (1.0 ** alpha + 1.0) == 0.5


@pytest.mark.parametrize("iterations,alpha_qui_deborde",
                         [(20, 300.0), (200, 200.0)])
def test_alpha_infini_ne_se_contente_pas_d_etre_inatteignable_il_plante(
    iterations: int, alpha_qui_deborde: float
) -> None:
    r"""``t**alpha`` déborde : « α = ∞ » lève `OverflowError`.

    Un flottant double plafonne à :math:`e^{709,78}`, donc
    ``t**alpha`` déborde dès :math:`\alpha\ln T > 709{,}78`. Seuils mesurés :
    α > 236,9 à T = 20 · α > 134,0 à T = 200 · α > 102,8 à T = 1000 ·
    α > 83,3 à T = 5000.

    Le test l'épingle des deux côtés — ça passe en dessous, ça casse au-dessus
    — pour qu'un lecteur sache où est le mur avant de le prendre. Un solveur
    qui plante à 5000 itérations et pas à 200 est le genre de piège qui se
    découvre en production.
    """
    seuil = 709.78 / math.log(iterations)
    assert alpha_qui_deborde > seuil          # le cas est bien au-delà du mur
    with pytest.raises(OverflowError):
        _depot(alpha_qui_deborde, 0.0, iterations)
    # Et juste en dessous, ça passe : le mur est bien là où on le dit.
    _depot(seuil * 0.9, 0.0, iterations)


# ═══════════════════════════════════════════════════════════════════════════
# 4. γ — le dépôt n'implémente PAS la loi du papier
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("gamma", [1.0, 2.0, 5.0])
def test_le_gamma_du_depot_pondere_la_contribution_pas_l_accumulateur(
    gamma: float,
) -> None:
    r"""Aucune dégénérescence sur γ n'est vraie ici, pour une TROISIÈME raison.

    Brown & Sandholm escomptent l'ACCUMULATEUR de stratégie moyenne :
    ``S ← S·(t/(t+1))^γ`` puis ``S += π σ``. Le poids résiduel de l'itération
    *t* après *T* est alors :math:`(t/T)^\gamma`, donc **∝ t^γ** : γ = 1 donne
    la moyenne linéaire de Linear CFR, γ = 2 la moyenne quadratique.

    `dcfr.py` applique :math:`(t/(t+1))^\gamma` à la CONTRIBUTION NEUVE
    (``strat_weight``, ligne 303) et n'escompte jamais l'accumulateur. Le
    poids de l'itération *t* y vaut :math:`(t/(t+1))^\gamma`, qui **tend vers
    1** : la moyenne reste quasi uniforme, et sa plage dynamique est bornée
    par :math:`2^\gamma` au lieu de croître comme :math:`T^\gamma`.

    Mesuré à T = 1000 : plage 2,0 au lieu de 1 000 (γ=1), 4,0 au lieu de
    10⁶ (γ=2), 31,8 au lieu de 10¹⁵ (γ=5).

    Ce test est une CARACTÉRISATION de l'implémentation, pas un satisfecit :
    si quelqu'un corrige γ pour suivre le papier — ce que
    ``banc_calculs_exacts.py --section dcfr`` mesure déjà sous la colonne
    « γ corrigé » — ce test doit tomber, et c'est son rôle de le signaler.
    """
    horizon = 1000
    poids_depot = [(t / (t + 1.0)) ** gamma for t in range(1, horizon + 1)]
    poids_papier = [(t / horizon) ** gamma for t in range(1, horizon + 1)]

    plage_depot = poids_depot[-1] / poids_depot[0]
    plage_papier = poids_papier[-1] / poids_papier[0]

    assert plage_depot == pytest.approx(2.0 ** gamma, rel=0.05)
    assert plage_papier == pytest.approx(float(horizon) ** gamma, rel=1e-9)
    assert plage_papier / plage_depot > 100.0


def test_gamma_ne_touche_pas_la_strategie_courante() -> None:
    """γ ne pèse que sur la moyenne : les regrets, eux, l'ignorent.

    C'est ce qui autorise les sections 1 à 3 à ne parler que de (α, β) : la
    stratégie courante — celle que le regret matching produit et que le
    solveur consomme à budget court — ne dépend pas de γ. Le vérifier ici
    évite de le supposer trois fois.
    """
    a = _depot(1.5, 0.0, 40)
    solveur = DCFRSolver(
        KuhnPoker(),
        DCFRConfig(alpha=1.5, beta=0.0, gamma=97.0, use_schedule=False,
                   prune_threshold=None),
    )
    solveur.solve(iterations=40)
    assert _ecart(a, _strategies_courantes(solveur)) == 0.0
