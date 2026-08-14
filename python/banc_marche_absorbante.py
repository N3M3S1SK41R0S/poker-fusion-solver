#!/usr/bin/env python
r"""Banc de la marche absorbante — le juge EXTERNE du biais de Harville.

    python banc_marche_absorbante.py              la grille complète (≈ 2 min 30)
    python banc_marche_absorbante.py --rapide     grille réduite (≈ 10 s)
    python banc_marche_absorbante.py --sims N     autre volume Monte-Carlo
    python banc_marche_absorbante.py --politique allin   l'autre dynamique

Pourquoi ce banc existe
-----------------------
``banc_invariants_icm.py`` vérifie que ``icm_equities`` conserve la dotation,
respecte l'échelle, la permutation, la monotonie : autant d'invariants que
TOUT membre de la famille des modèles de rang « raisonnables » satisfait. Un
banc d'invariants ne peut donc vérifier que l'appartenance à la famille,
jamais QUEL membre est implémenté — Malmuth-Harville, Weitzman, Henery et un
modèle de diffusion passent tous les mêmes invariants. Pour juger le MEMBRE,
il faut un modèle de référence construit AUTREMENT, et confronter les
probabilités de rang elles-mêmes.

Ce banc construit ce juge : le tournoi comme **marche aléatoire absorbante de
jetons**. Des paires de joueurs s'échangent des jetons par pas équitables
(espérance nulle), un joueur absorbé à zéro est éliminé et prend le rang
courant, le dernier survivant gagne. Les probabilités de rang sont estimées
par Monte-Carlo massif, avec erreurs-types binomiales par case, et comparées
à la récurrence de Malmuth-Harville (``pfs.core.icm._finish_probs_exact``).
L'écart mesuré EST le biais de Harville — relatif à cette dynamique.

Ce que la marche suppose — l'honnêteté avant les chiffres
---------------------------------------------------------
La marche absorbante est ELLE-MÊME un modèle. Elle suppose :

* **des pas équitables** — chaque échange est d'espérance nulle pour les deux
  joueurs. Pas de skill, pas de position, pas de blindes qui rognent les
  petits tapis : tous les joueurs sont identiques hors leur tapis. C'est
  exactement l'hypothèse d'ICM (« le tapis est la seule information »), posée
  au niveau de la DYNAMIQUE et non des rangs — c'est ce qui en fait un juge
  comparable et pourtant indépendant ;
* **des tirages indépendants** — la pièce de chaque échange est i.i.d. ; pas
  de tilt, pas de dynamique de table ;
* **une politique de pas**, qui est un CHOIX et change le verdict (mesuré
  ci-dessous, section « allin ») :

  - ``unitaire`` — paire uniforme parmi les vivants, ±1 jeton à pile ou
    face. C'est la discrétisation de la diffusion neutre sur le simplexe,
    le membre « limite des petits pots » de la famille. Le biais mesuré est
    STABLE sous la granularité (tapis ×½ et ×2 : −2,31 / −2,33 / −2,29 pt
    sur la même case), donc ce n'est pas un artefact de discrétisation ;
  - ``allin`` — paire uniforme, pile ou face pour ``min`` des deux tapis.
    La caricature inverse : chaque affrontement est un tapis intégral.

Pourquoi ce juge, tout modèle qu'il est, vaut mieux que l'auto-comparaison :
ses hypothèses portent sur la dynamique des jetons, PAS sur la forme des
probabilités de rang. Harville postule directement
``P(i deuxième | j premier) = s_i/(S−s_j)`` — une forme fermée jamais dérivée
d'un processus. La marche, elle, DÉRIVE ses rangs d'un processus explicite
dont chaque hypothèse est nommée et attaquable. Comparer Harville à la marche
compare deux constructions étrangères l'une à l'autre ; comparer Harville à
ses propres invariants ne compare rien.

Deux ancres exactes verrouillent le simulateur (mesurées à chaque run) :

* **martingale** — le tapis de chaque joueur est une martingale bornée, donc
  par arrêt optionnel ``P(i gagne) = s_i/S`` EXACTEMENT, pour toute politique
  équitable. Harville donne la même valeur : le biais du rang 1 est
  structurellement nul, et l'écart mesuré (max z = 2,3 sur la grille des
  deux politiques) ne mesure que le bruit Monte-Carlo — c'est le témoin de
  bon fonctionnement ;
* **symétrie** — sur des tapis égaux ``[25, 25, 25, 25]``, toutes les
  probabilités de rang valent ¼ dans les deux modèles : biais nul attendu,
  case par case (mesuré : max |z| = 2,5 à 100 000 tirages, compatible bruit,
  pire case +0,35 ± 0,14 pt).

Le biais de Harville, QUANTIFIÉ (politique ``unitaire``, 100 000 tirages,
graine 0, biais = Harville − marche, en points de probabilité)
----------------------------------------------------------------------------
Lecture : « +2,4 » = Harville donne 2,4 points de PLUS que la marche.

=================================  =========================================
Configuration                      Cases les plus biaisées (± 1 erreur-type)
=================================  =========================================
``[50, 30, 20]``                   leader 2e : **−2,33 ± 0,15** (z = 15) ;
                                   leader 3e : +2,30 ± 0,11 ;
                                   petit 2e : **+2,49 ± 0,14** (z = 18) ;
                                   petit 3e : −2,43 ± 0,16. Le tapis moyen
                                   est quasi neutre (|z| ≤ 1,1).
``[40, 30, 20, 10]``               petit dernier : **−5,35 ± 0,15** (z = 35) ;
                                   petit 3e : +3,24 ± 0,12 ; leader 2e :
                                   −2,34 ± 0,15 ; leader dernier : +2,44.
``[30, 25, 20, 12, 8, 5]``         micro-tapis dernier : **−6,32 ± 0,16** ;
                                   même signe sur toute la diagonale basse.
=================================  =========================================

La STRUCTURE du biais, constante sur toute la grille (3 à 6 joueurs, tapis
asymétriques) :

1. **Harville sous-estime les places intermédiaires du gros tapis** (« le
   leader finit 2e » : −2,2 à −2,4 pt partout) et surestime ses places
   basses. La récurrence retire le gagnant puis fait courir le leader sur
   ses jetons BRUTS ; la marche, elle, sait qu'un leader qui n'a pas gagné a
   souvent PERDU des jetons en route — il est plus bas que son tapis initial
   le laisse croire, mais rarement au point de finir dernier.
2. **Harville surestime les places intermédiaires du petit tapis** (+2,5 pt
   sur « le petit finit 2e » à trois joueurs) et sous-estime NETTEMENT sa
   dernière place (−5,4 pt à quatre joueurs, −6,3 à six). Symétriquement :
   un petit tapis encore en vie tard dans la marche a souvent GAGNÉ des
   jetons ; Harville le fait courir sur son tapis de départ.
3. Converti en $EV (grille 50/30/20, erreur-type par ``erreur_type_exacte``,
   ± ci-dessous = 3 erreurs-types) : **Harville SURÉVALUE les petits tapis
   et SOUS-ÉVALUE les gros**, de 0,2 à 1,3 % de la dotation selon la
   configuration — p. ex. +1,31 ± 0,16 % pour le tapis de 10 sur
   ``[40, 30, 20, 10]``, −0,71 ± 0,14 % pour le tapis de 40 ; −1,02 % pour
   le leader de ``[35, 25, 20, 12, 8]``. C'est le sens connu du folklore
   « l'ICM surpaie les shorts » : ici il est MESURÉ, avec erreurs-types,
   contre un modèle explicite.
4. Sous la politique ``allin``, le biais CHANGE DE SIGNE sur plusieurs cases
   (leader 2e de ``[40, 30, 20, 10]`` : +2,2 au lieu de −2,3 ; micro-tapis
   dernier à six joueurs : +17,6 au lieu de −6,3 ; et jusqu'à −24,4 pt sur
   « le tapis de 10 finit 2e » de ``[45, 45, 10]``). Le « biais de
   Harville » n'est donc pas un scalaire absolu : c'est une fonction de la
   dynamique supposée. Harville est un compromis entre la diffusion douce et
   la loterie de tapis — nettement plus proche de la première en $EV, et
   c'est la seule conclusion que ce banc s'autorise.

Ce que ce banc NE conclut PAS : que ``icm_equities`` est « faux ». Malmuth-
Harville est le standard de l'industrie (Pio, HRC, ICMIZER, GTO Wizard), et
l'écart au juge diffusif (≤ 0,8 % de dotation) est un biais de MODÈLE,
documenté ici, pas un défaut d'implémentation — l'implémentation, elle, est
verrouillée par ``test_icm_ordre_et_elimination.py`` contre l'énumération
8! complète.

Rejouabilité
------------
Aucun aléa non semé : ``numpy.random.default_rng(seed)``, flux PCG64 stable.
Deux exécutions rendent les mêmes chiffres au bit près. Les valeurs golden
figées par ``tests/test_marche_absorbante.py`` portent une tolérance de
3 erreurs-types CALCULÉES (jamais élargies) : elles survivent à un
réordonnancement des tirages, pas à un vrai changement de loi.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pfs.core.icm import _finish_probs_exact, erreur_type_exacte  # noqa: E402

POLITIQUES = ("unitaire", "allin")
"""Les politiques de pas implémentées. Chacune est un modèle : voir l'en-tête."""

GRILLE: tuple[tuple[int, ...], ...] = (
    (50, 30, 20),
    (60, 30, 10),
    (45, 45, 10),
    (40, 30, 20, 10),
    (70, 10, 10, 10),
    (25, 25, 25, 25),          # témoin symétrique : biais nul attendu partout
    (35, 25, 20, 12, 8),
    (30, 25, 20, 12, 8, 5),
)
"""Configurations mesurées : 3 à 6 joueurs, tapis asymétriques, total 100.

Le total est tenu à 100 jetons pour que la politique ``unitaire`` reste
simulable en masse (le temps d'absorption croît comme le carré du total) ;
la section « échelle » du rapport vérifie que le biais n'en dépend pas.
"""

PAYOUTS_EV = (50.0, 30.0, 20.0)
"""Grille de gains du volet $EV — la structure classique à trois places.

Elle sert UNIQUEMENT à convertir le biais des rangs en biais de valeur ;
aucune conclusion du banc ne dépend de ce choix (une autre grille décroissante
change les montants, pas les signes)."""


class MarcheError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# LA MARCHE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MarcheResultat:
    """Probabilités de rang estimées par la marche, avec leur incertitude.

    Attributes
    ----------
    stacks : tuple of int
        Les tapis simulés (jetons entiers — la marche est discrète).
    politique : str
        La politique de pas (voir :data:`POLITIQUES`).
    n_sims, seed : int
        Volume Monte-Carlo et graine — le couple qui rend le run rejouable.
    probs : numpy.ndarray, shape (n, n)
        ``probs[i, k]`` = fréquence de « le joueur *i* finit au rang *k+1* ».
        Chaque ligne somme à 1 (un joueur a un rang), chaque colonne aussi
        (un rang a un joueur) : la matrice est bistochastique par
        construction, et le banc le vérifie plutôt que le supposer.
    se : numpy.ndarray, shape (n, n)
        Erreur-type binomiale par case, :math:`\\sqrt{\\hat p(1-\\hat p)/n}`.
    pas : int
        Pas de simulation exécutés (boucles vectorisées) — un chiffre de
        coût, pas de précision.
    """

    stacks: tuple[int, ...]
    politique: str
    n_sims: int
    seed: int
    probs: np.ndarray
    se: np.ndarray
    pas: int


def simuler_rangs(stacks, n_sims: int = 100_000, seed: int = 0,
                  politique: str = "unitaire") -> MarcheResultat:
    r"""Simule le tournoi comme marche absorbante et estime les rangs.

    Le processus, par tournoi simulé :

    1. tant qu'il reste plus de deux joueurs : tirer une paire uniforme parmi
       les vivants, jouer un pas équitable (±1 jeton en ``unitaire``,
       ± min des deux tapis en ``allin``) ; un joueur à zéro est éliminé et
       reçoit le rang « nombre de vivants avant sa chute » ;
    2. à DEUX joueurs, plus rien à simuler : le tapis est une martingale
       bornée absorbée en 0 ou S, donc :math:`P(\text{gagner}) = s/S`
       exactement (ruine du joueur pour ``unitaire`` ; arrêt optionnel pour
       toute politique équitable). Le banc tire le vainqueur directement —
       même loi, sans les ~2 500 pas de la phase en tête-à-tête.

    Le raccourci du point 2 n'est pas une approximation : c'est la loi exacte
    de la marche restreinte à deux joueurs, et c'est lui qui rend le
    Monte-Carlo « massif » abordable (× 3 à × 40 selon la configuration).

    Parameters
    ----------
    stacks : sequence of int
        Tapis strictement positifs, en jetons entiers.
    n_sims : int
        Tournois simulés. L'erreur-type par case vaut au pire
        :math:`\sqrt{0{,}25/n}` — 0,16 point à 100 000.
    seed : int
        Graine du générateur (PCG64). Même graine = mêmes chiffres au bit
        près.
    politique : str
        ``"unitaire"`` ou ``"allin"`` (voir l'en-tête du module).

    Returns
    -------
    MarcheResultat

    Examples
    --------
    À deux joueurs la marche EST la ruine du joueur : ``[2, 1]`` gagne avec
    probabilité 2/3 exactement (martingale + arrêt optionnel, calcul à la
    main : :math:`p \cdot 3 = 2`). L'estimation à 30 000 tirages, graine 0,
    tombe à moins de trois erreurs-types (:math:`3\sqrt{(2/9)/30000} = 0{,}0082`)
    de cette valeur :

    >>> res = simuler_rangs((2, 1), n_sims=30_000, seed=0)
    >>> abs(float(res.probs[0, 0]) - 2.0 / 3.0) < 0.0082
    True
    >>> res.pas        # tout est réglé par le raccourci exact : zéro pas
    0
    """
    s = tuple(int(x) for x in stacks)
    if len(s) < 2:
        raise MarcheError("au moins deux joueurs requis.")
    if any(x <= 0 for x in s):
        raise MarcheError("tapis entiers strictement positifs requis "
                          "(la marche est discrète, l'absorption est en 0).")
    if any(float(x) != float(y) for x, y in zip(s, stacks)):
        raise MarcheError("tapis non entiers : la politique unitaire échange "
                          "des jetons indivisibles.")
    if politique not in POLITIQUES:
        raise MarcheError(f"politique inconnue : {politique!r} "
                          f"(attendu : {', '.join(POLITIQUES)}).")
    if n_sims < 1:
        raise MarcheError("n_sims >= 1 requis.")

    rng = np.random.default_rng(seed)
    n = len(s)
    total = sum(s)
    S = np.tile(np.asarray(s, dtype=np.int64), (n_sims, 1))
    rangs = np.zeros((n_sims, n), dtype=np.int8)
    vivants = np.full(n_sims, n, dtype=np.int8)
    counts = np.zeros((n, n), dtype=np.int64)
    pas = 0

    def basculer_hu(S: np.ndarray, rangs: np.ndarray,
                    vivants: np.ndarray) -> None:
        """Règle les tournois réduits à deux joueurs par la loi exacte s/S."""
        hu = vivants == 2
        if not hu.any():
            return
        idx = np.nonzero(hu)[0]
        sub = S[idx]
        masque = sub > 0
        a = np.argmax(masque, axis=1)
        reste = masque.copy()
        reste[np.arange(len(idx)), a] = False
        b = np.argmax(reste, axis=1)
        p_a = sub[np.arange(len(idx)), a] / total
        gagne_a = rng.random(len(idx)) < p_a
        w = np.where(gagne_a, a, b)
        v = np.where(gagne_a, b, a)
        rangs[idx, w] = 1
        rangs[idx, v] = 2
        vivants[idx] = 1

    def encaisser(S, rangs, vivants):
        """Compacte les tournois finis dans `counts` et rend les actifs."""
        fini = vivants <= 1
        if fini.any():
            r = rangs[fini]
            for k in range(n):
                counts[k] += np.bincount(r[:, k] - 1, minlength=n)
            S, rangs, vivants = S[~fini], rangs[~fini], vivants[~fini]
        return S, rangs, vivants

    basculer_hu(S, rangs, vivants)
    S, rangs, vivants = encaisser(S, rangs, vivants)

    while S.shape[0] > 0:
        pas += 1
        m = S.shape[0]
        am = np.arange(m)
        # Paire uniforme parmi les vivants : deux plus petites clés aléatoires,
        # les morts poussés à +inf pour n'être jamais tirés.
        cles = rng.random((m, n))
        cles[S == 0] = np.inf
        paire = np.argpartition(cles, 1, axis=1)[:, :2]
        i0, i1 = paire[:, 0], paire[:, 1]
        s0, s1 = S[am, i0], S[am, i1]
        pile = rng.integers(0, 2, size=m).astype(bool)
        if politique == "unitaire":
            q = np.ones(m, dtype=np.int64)
        else:
            q = np.minimum(s0, s1)
        d0 = np.where(pile, q, -q)
        S[am, i0] = s0 + d0
        S[am, i1] = s1 - d0
        # Seul le perdant du pas peut être à zéro : une élimination par pas.
        perdant = np.where(pile, i1, i0)
        elim = S[am, perdant] == 0
        rangs[am[elim], perdant[elim]] = vivants[elim]
        vivants[elim] -= 1
        basculer_hu(S, rangs, vivants)
        S, rangs, vivants = encaisser(S, rangs, vivants)

    probs = counts / float(n_sims)
    se = np.sqrt(np.maximum(probs * (1.0 - probs), 0.0) / n_sims)
    return MarcheResultat(stacks=s, politique=politique, n_sims=n_sims,
                          seed=seed, probs=probs, se=se, pas=pas)


# ═══════════════════════════════════════════════════════════════════════════
# LA CONFRONTATION À HARVILLE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BiaisHarville:
    """Le biais de Harville sur une configuration, chiffré case par case.

    Attributes
    ----------
    marche : MarcheResultat
        Le juge — les rangs de la marche, avec leurs erreurs-types.
    harville : numpy.ndarray, shape (n, n)
        Les rangs de Malmuth-Harville (exacts, récurrence mémoïsée du dépôt).
    biais : numpy.ndarray, shape (n, n)
        ``harville − marche.probs``, en probabilité. Positif = Harville
        surestime.
    z : numpy.ndarray, shape (n, n)
        ``|biais| / se`` — la significativité case par case. Sous hypothèse
        nulle (aucun biais), |z| > 3 sur une case donnée a une probabilité
        < 0,3 % ; les z de 15 à 80 mesurés ici ne laissent aucune ambiguïté.
    """

    marche: MarcheResultat
    harville: np.ndarray
    biais: np.ndarray
    z: np.ndarray


def biais_harville(stacks, n_sims: int = 100_000, seed: int = 0,
                   politique: str = "unitaire") -> BiaisHarville:
    """Simule la marche puis la confronte à Malmuth-Harville.

    C'est LA fonction du banc : tout le reste est rapport. La référence
    Harville vient du dépôt (``_finish_probs_exact``) — c'est voulu : c'est
    précisément l'implémentation en production qu'on veut juger, pas une
    réécriture locale qui pourrait diverger d'elle.
    """
    res = simuler_rangs(stacks, n_sims=n_sims, seed=seed, politique=politique)
    mh = _finish_probs_exact(tuple(float(x) for x in res.stacks),
                             len(res.stacks))
    biais = mh - res.probs
    z = np.abs(biais) / np.maximum(res.se, 1e-15)
    return BiaisHarville(marche=res, harville=np.asarray(mh), biais=biais, z=z)


def ecart_ev(b: BiaisHarville, payouts=PAYOUTS_EV) -> tuple[np.ndarray, np.ndarray]:
    r"""Biais de Harville converti en $EV, joueur par joueur.

    Returns
    -------
    delta : numpy.ndarray, shape (n,)
        :math:`\Delta_i = \sum_k (H_{ik} - M_{ik})\,\pi_k` — ce que Harville
        donne EN PLUS de la marche au joueur *i*, dans l'unité des payouts.
        La somme des deltas vaut 0 aux arrondis près : les deux modèles
        distribuent la même dotation.
    se : numpy.ndarray, shape (n,)
        Erreur-type de ``delta`` : celle de l'estimation Monte-Carlo du $EV
        de la marche, par :func:`pfs.core.icm.erreur_type_exacte` — la
        variance multinomiale exacte, PAS la somme des erreurs par place
        (corrélées négativement, la somme surestimerait).
    """
    n = len(b.marche.stacks)
    pay = np.zeros(n)
    pay[:min(len(payouts), n)] = payouts[:n]
    delta = b.biais @ pay
    se = erreur_type_exacte(b.marche.probs, pay, b.marche.n_sims)
    return delta, se


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═══════════════════════════════════════════════════════════════════════════


def _fmt_matrice(m: np.ndarray, mult: float = 100.0) -> str:
    return "\n".join(
        "    " + "  ".join(f"{mult * v:+6.2f}" for v in ligne) for ligne in m)


def rapport_config(stacks, n_sims: int, seed: int, politique: str) -> bool:
    """Mesure et imprime une configuration. Rend Faux si une ANCRE casse.

    Les ancres — bistochasticité, martingale du rang 1 — sont des propriétés
    du SIMULATEUR, pas du modèle jugé : les casser signifierait que le juge
    est corrompu, et le banc doit alors se taire plutôt que publier un biais.
    """
    t0 = time.perf_counter()
    b = biais_harville(stacks, n_sims=n_sims, seed=seed, politique=politique)
    dt = time.perf_counter() - t0
    res = b.marche
    n = len(res.stacks)
    total = sum(res.stacks)

    ok = True
    lignes = np.abs(res.probs.sum(axis=1) - 1.0).max()
    colonnes = np.abs(res.probs.sum(axis=0) - 1.0).max()
    if max(lignes, colonnes) > 1e-9:
        ok = False
    parts = np.asarray(res.stacks, dtype=float) / total
    z_martingale = float(np.max(np.abs(res.probs[:, 0] - parts)
                                / np.maximum(res.se[:, 0], 1e-15)))
    # 4 est la borne de politesse pour n comparaisons simultanées à 100 000
    # tirages : P(max |z| > 4 sous H0) < n · 6e-5. Ce n'est pas une tolérance
    # de modèle — c'est un seuil d'alarme du simulateur.
    if z_martingale > 4.0:
        ok = False

    delta, se_ev = ecart_ev(b)
    pire = np.unravel_index(np.argmax(np.abs(b.biais)), b.biais.shape)

    print(f"\n── {list(res.stacks)} · {politique} · {n_sims:,} tirages · "
          f"{dt:.1f} s · {res.pas} pas ──".replace(",", " "))
    print(f"  ancre bistochastique : écart max {max(lignes, colonnes):.1e} "
          f"{'✓' if max(lignes, colonnes) <= 1e-9 else '✗ SIMULATEUR CORROMPU'}")
    print(f"  ancre martingale (rang 1 = part de jetons) : max |z| = "
          f"{z_martingale:.1f} {'✓' if z_martingale <= 4.0 else '✗ SIMULATEUR CORROMPU'}")
    print("  biais Harville − marche (points de probabilité, rang 1 → dernier) :")
    print(_fmt_matrice(b.biais))
    print("  significativité |z| max par joueur : "
          + "  ".join(f"{v:.0f}" for v in b.z.max(axis=1)))
    i, k = int(pire[0]), int(pire[1])
    print(f"  pire case : joueur {i} (tapis {res.stacks[i]}) au rang {k + 1} : "
          f"{b.biais[i, k] * 100:+.2f} ± {res.se[i, k] * 100:.2f} pt "
          f"(z = {b.z[i, k]:.0f})")
    print(f"  $EV (payouts {'/'.join(f'{p:g}' for p in PAYOUTS_EV)}, "
          f"± = 3 SE) : "
          + "  ".join(f"{d:+.2f}±{3 * s:.2f}" for d, s in zip(delta, se_ev)))
    return ok


def section_echelle(n_sims: int, seed: int) -> bool:
    """Le biais dépend-il de la granularité des jetons ? Mesure directe.

    Même ratio 5:3:2 à trois échelles. Si le biais était un artefact de
    discrétisation, il devrait fondre quand les tapis grossissent ; il ne
    bouge pas d'une erreur-type.
    """
    print("\n═══ échelle : le biais est-il un artefact de la granularité ? ═══")
    ok = True
    reference: float | None = None
    for stacks, sims in (((25, 15, 10), n_sims), ((50, 30, 20), n_sims),
                         ((100, 60, 40), max(20_000, n_sims // 2))):
        b = biais_harville(stacks, n_sims=sims, seed=seed)
        case = float(b.biais[0, 1])
        se = float(b.marche.se[0, 1])
        print(f"  {list(stacks)} : leader 2e = {case * 100:+.2f} ± "
              f"{se * 100:.2f} pt")
        if reference is None:
            reference = case
        elif abs(case - reference) > 3.0 * se * np.sqrt(2.0):
            ok = False
            print("    ✗ incompatible avec l'échelle de référence à 3 SE")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rapide", action="store_true",
                    help="grille réduite et 20 000 tirages (~10 s)")
    ap.add_argument("--politique", choices=[*POLITIQUES, "les-deux"],
                    default="les-deux")
    a = ap.parse_args()

    n_sims = 20_000 if a.rapide else a.sims
    grille = GRILLE[:1] + GRILLE[3:4] + GRILLE[5:6] if a.rapide else GRILLE
    politiques = POLITIQUES if a.politique == "les-deux" else (a.politique,)

    depart = time.perf_counter()
    ok = True
    for politique in politiques:
        sims = n_sims if politique == "unitaire" else max(n_sims, 200_000)
        print(f"\n════════ politique {politique} ════════")
        for stacks in grille:
            ok &= rapport_config(stacks, sims, a.seed, politique)
    if not a.rapide:
        ok &= section_echelle(n_sims, a.seed)

    print(f"\n{time.perf_counter() - depart:.0f} s au total — "
          + ("ancres du simulateur : TOUTES TIENNENT"
             if ok else "AU MOINS UNE ANCRE CASSE : ne pas citer les biais"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
