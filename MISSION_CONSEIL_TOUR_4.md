# Mission — conseil des modèles, tour 4

*(GPT-5.6 Sol Thinking · Claude Opus 5 Thinking · Gemini 3.1 Pro Thinking)*

---

## 0. Cette fois le code est DANS ce message

Trois tours, trois échecs d'accès. Dépôt privé au tour 1, archive mentionnée
mais non jointe au tour 2, archive illisible par votre outillage au tour 3.
Vous avez eu raison de refuser tout verdict à chaque fois, et l'axe A — la
justesse des calculs — n'a **jamais** été traité.

Vous avez donné la solution vous-mêmes : *« collez le contenu en texte brut,
inline, dans le corps du message »*. C'est ce que fait ce message.

Il ne contient qu'**un seul module**, `pfs/core/icm.py`, 681 lignes,
intégralement. C'est celui que vous avez désigné comme prioritaire, c'est
celui qui pilote les ranges push/fold en tournoi, et c'est celui où un
correctif récent a introduit une branche de code. Les autres suivront au même
format, un par tour, jusqu'à ce que l'axe A soit couvert.

---

## 1. Ce que vous nous avez fait corriger depuis le tour 3

Vos objections ont porté. Trois d'entre elles nous ont fait retirer des
affirmations publiées.

**Le rapprochement des 31,7 % était fallacieux.** Vous avez démontré que
31,7 % de lignes *traversées* implique 68,3 % *non* traversées — l'exact
complément de ce que nous annoncions, pas une confirmation. Et le banc trouve
20 modules jamais atteints là où le README en annonçait 12 : le recensement
n'était pas confirmé, il était révisé à la hausse de 67 %. Retiré. Nous
publions désormais trois nombres distincts, dont celui que vous avez nommé et
que ni la couverture ni l'analyse statique ne voient : les lignes **atteintes
mais causalement inertes**, dont le résultat est calculé puis jeté. C'est la
catégorie qui contenait `/api/advise` jetant les tapis de tournoi.

**Le « 95 % » de lecture des cartes n'est pas une exactitude — et le vrai
chiffre est 76,7 %.** Le dénominateur était conditionnel à la détection, il
n'y avait aucune vérité-terrain, et « non refusé » n'est pas « juste ». Votre
indice chiffré était exact, et il pointait le bon endroit : 108 boîtes héros
sur 57 frames font 1,89 par frame au lieu de 2.

Les 57 captures ont été annotées à la main, carte par carte
(`tests/donnees/verite_captures.json`, 263 cartes relevées ; le relevé visuel
et un masque de couleur exact sur les quatre aplats du jeu concordent). Le
banc `banc_verite_captures.py` rejoue la chaîne de production et rend :

```
cartes réellement présentes (pleinement visibles) : 258   (et non 209)
RAPPEL DE LOCALISATION :  76,7 %  (198/258)
RAPPEL DE LECTURE      :  76,7 %  (198/258)   ← le taux à annoncer
  dont bon rôle        :  65,1 %  (168/258)
PRÉCISION              : 100,0 %  (199/199 lectures affirmées)
cartes JAMAIS localisées : 60      LECTURES FAUSSES AFFIRMÉES : 0
```

Trois choses, dans l'ordre de gravité :

* **le rappel était l'information manquante, et il est mauvais** : 60 cartes
  réellement présentes ne sont jamais vues. 45 sont les cartes du HÉROS de la
  table 7-max — jamais une seule, sur 24 des 25 captures ;
* **30 cartes du board y sont présentées comme cartes du héros**, conséquence
  mécanique de la précédente : `read_table` prend la rangée la plus basse
  pour la main du héros, et quand le siège n'est pas détecté, c'est le board
  qui hérite du rôle. Une carte juste au mauvais rôle vaut une carte fausse ;
* **une lecture fausse et affirmée a bien été trouvée** — le 6♣ du flop de
  `300_7-max_KO/0003`, saisi en pleine animation de retournement, sortait
  « Kc », statut « sure ». Le contrôle de dispersion que nous vous avions
  présenté comme la parade existait déjà et refusait bien la découpe : son
  refus était **muet**, la chaîne passait la main au hachage, et les 40
  cadrages essayés finissaient par en trouver un sous le seuil. Le refus est
  désormais franc, et un test emprunte le chemin de la CHAÎNE, pas celui du
  lecteur isolé. Le compteur est à 0.

La cause des 45 cartes du héros perdues est mesurée, et ce n'est ni une
occultation ni un siège vide : les cartes sont parfaitement visibles, la
bonne paire d'arêtes est trouvée, le recalage donne le bon rapport (1,017,
dans la plage « carte coupée par le bas »). C'est la règle « **3 abords
calmes sur 4** » du détecteur qui les rejette — l'habillage « KO » de cette
table pose un rail lumineux immédiatement à gauche et à droite du siège, et
une pastille de prime « 2,25 € » à cheval sur la carte de gauche. Cette règle
avait été calibrée sur des tables **synthétiques**, où le feutre est uni.
Ablation mesurée sur les vraies captures
(`banc_verite_captures.py --quiet-sides 2`) : rappel **76,7 % → 96,9 %**,
rôles justes **65,1 % → 96,9 %**, et **zéro carte inventée**. Sur les 144
tables synthétiques, la même ablation coûtait 1,9 % de boîtes fantômes. Nous
ne l'avons pas appliquée : c'est un chantier de correction, avec ses propres
mesures sur les deux bancs.

**Le seuil d'uniformité du fond repose sur une statistique qui sature.** Vous
avez raison : min, médiane, p95 et maximum tous à exactement 0,0 sur
199 mesures, ce n'est pas une mesure, c'est une saturation. Ré-encodage JPEG
en cours pour trouver où elle casse, et test des contaminations homogènes
contre lesquelles elle est aveugle.

**Et le test que vous jugez le plus urgent tourne en ce moment** : le
`bubble_factor` du joueur couvert, dont l'espérance en cas de perte est
évaluée dans la branche « tapis nul » — donc un quotient entre deux régimes
de code, où un écart absorbé par la conservation serait amplifié par la
division. Avec la nuance que vous avez signalée et que nous n'aurions pas
trouvée seuls : la continuité ne doit être exigée que pour **un seul** tapis
nul, le cas multi-zéro étant authentiquement discontinu — `[100, ε, ε/2]`
converge vers (30, 20), pas (25, 25).

---

## 2. Ce que nous vous demandons — l'axe A sur ce module

Le code suit au §5. Nous attendons, pour chaque point, **CONFIRMÉ / RÉFUTÉ /
NON VÉRIFIÉ**, avec valeur de référence, source et écart mesuré.

1. **Le noyau Malmuth-Harville est-il correct ?** La récurrence, la
   mémoïsation sur bitmask, le passage exact → Monte-Carlo à 12 joueurs.
2. **La branche « tapis nul » est-elle juste ?** Elle a été écrite après un
   `ZeroDivisionError`, puis corrigée une seconde fois parce que la première
   version donnait zéro au joueur éliminé au lieu du dernier gain. Les quatre
   cas discriminants que vous aviez proposés passent. **Cherchez le cinquième.**
3. **`bubble_factor` et `risk_premium`** — la définition retenue, le
   traitement du dénominateur, et le comportement au voisinage de zéro.
4. **Le chemin PKO** — `bounty_capture_value`, `analyse_pko_spot`, et la
   convention « ½ cash + ½ sur sa propre prime ». Cette convention est-elle
   celle de l'industrie, et le calcul de la part qui reste sur sa propre tête
   est-il correct ?
5. **Le FGS léger** — l'érosion des blindes futures au premier ordre. Est-ce
   une approximation défendable, et où casse-t-elle ?
6. **Ce que le banc d'invariants ne teste pas.** Il déclare échelle,
   permutation, conservation, monotonie, non-linéarité et concordance
   exact ↔ Monte-Carlo. Vous avez déjà trouvé la continuité en zéro.
   Qu'est-ce qui manque encore ?

---

## 3. Là où nous avons besoin de votre aide, pas de votre verdict

Quatre problèmes ouverts. Nous n'avons pas de solution satisfaisante et votre
raisonnement vaut ici plus que votre lecture de code.

### 3.1 La propagation par intervalles — comment la construire sans tout casser

Claude Opus 5 a proposé au tour 3 le seul mécanisme qui subsume tous nos cas
de refus : propager chaque entrée comme un **intervalle** plutôt qu'un
scalaire, et déclarer le verdict « fragile » dès qu'il n'est pas constant sur
la boîte. Cela couvre d'un coup l'arrondi d'affichage des montants, le choix
Harville contre Malmuth-Weitzman, l'hypothèse de range postflop, et
l'incertitude statistique sur nos propres fréquences.

Nous adoptons l'idée. Nous ne savons pas la réaliser proprement.

- Faut-il une **arithmétique d'intervalles** partout, avec le risque
  d'explosion de largeur bien connu, ou un **échantillonnage des sommets** de
  la boîte, ou une propagation par **dérivées** autour du point nominal ?
- Comment éviter que « fragile » ne devienne l'étiquette de tous les verdicts,
  auquel cas l'information disparaît ?
- Où placer la frontière : quelles entrées méritent un intervalle, et
  lesquelles peuvent rester scalaires sans mentir ?
- Que devient un seuil de bascule — aujourd'hui un scalaire — dans ce cadre ?

### 3.2 L'ancrage des zones de montants

Vous avez écarté le détecteur de texte, et écarté aussi l'ancrage sur le
centre du board — instable puisque le board a de 0 à 5 cartes et n'existe pas
préflop, précisément là où les verdicts sont « certains ». L'un de vous a
suggéré l'ellipse du feutre plus les cartes du siège.

- Comment estimer l'**échelle** de façon robuste à partir d'une seule capture,
  sans dépendre d'un élément qui peut manquer ?
- La conservation des pixels doit porter sur des **positions cumulées** et non
  sur une largeur totale, ce que nous acceptons — mais comment la formuler
  quand la police a des chasses variables, le « 1 » étant nettement plus
  étroit que le « 8 » ?
- Un point acquis, et c'est votre cadeau : **PMU affiche tout en blindes**,
  donc un pot de 1 200 BB face à des tapis de 98 BB est impossible par
  construction. Cette borne remplace-t-elle vraiment le veto de granularité
  que l'affichage arrondi nous fait perdre, ou faut-il autre chose ?

### 3.3 La dégénérescence DCFR

L'un de vous signale que « α = β = γ = 1 donne CFR standard » est suspect,
parce que **Linear CFR n'est pas Vanilla CFR**. Nous avons ce test de
conformité dans le dépôt et nous ne savons pas s'il est juste.

Quels sont les paramètres exacts qui rendent DCFR équivalent à Vanilla CFR,
à CFR+, et à Linear CFR ? Dérivez-les depuis les équations plutôt que de
citer, et dites-nous ce que notre test devrait affirmer.

### 3.4 Comparer à Pluribus sans se mentir

Nous avons téléchargé le corpus PHH de l'université de Toronto : **10 000
mains de Pluribus** et 83 mains des WSOP, licence MIT. Un lecteur du format
est en cours d'écriture, et nous rejouons chaque décision à travers notre
conseiller pour la comparer à ce que le bot a réellement fait.

Le problème est que Pluribus joue du **cash game 6-max sans ICM**, alors que
l'utilisateur joue des **tournois**. Un taux d'accord sur du cash ne valide
pas le conseil en tournoi, et nous ne voulons pas présenter l'un pour l'autre.

- Quelle **partie** de notre conseiller cette comparaison valide-t-elle
  légitimement — l'équité, les cotes du pot, les ranges d'ouverture ?
- Comment distinguer un désaccord qui révèle **notre** défaut d'un désaccord
  où Pluribus dévie volontairement pour exploiter ?
- Un désaccord **systématique** sur une famille de spots est le livrable que
  nous visons. Comment le caractériser sans tomber dans la pêche aux
  corrélations sur 10 000 mains ?

---

## 4. Format, inchangé

**CONFIRMÉ / RÉFUTÉ / NON VÉRIFIÉ**, pas de quatrième catégorie. Les deux
sections restent obligatoires : **« ce que je n'ai pas pu vérifier »** et
**« mon désaccord principal »**.

Répondez séparément, sans vous concerter. Sur trois tours, la bonne réponse a
été quatre fois celle d'un seul d'entre vous — le biais de Harville, le piège
SAGE, l'erreur d'équité de Gemini, et la fausse coïncidence des 31,7 %. Nous
ne l'aurions eue d'aucun avis unique.

Deux cadrages qui ne changent pas : **le périmètre ne se rediscute pas**, et
**la priorité est de faire marcher** — la chaîne s'arrête encore après la
lecture des cartes, faute de lecture des montants.

---

## 5. Le code — `pfs/core/icm.py`, intégral

```python
# ═══ pfs/core/icm.py ═══ (681 lignes, intégral)

"""
F14 — ICM : Independent Chip Model, bubble factor, $EV.

Sources
-------
- Malmuth, M. & Harville, D. (1973/1987) — le modèle standard de l'industrie,
  utilisé par PioSOLVER, HRC, ICMIZER, MonkerSolver, GTO Wizard.
- Harville, D.A. (1973), *Assigning Probabilities to the Outcomes of
  Multi-Entry Competitions*, JASA 68(342) — la récurrence d'origine (courses).
- Ganzfried & Sandholm (2015) pour l'articulation avec la safe exploitation.

Le principe en une phrase : **en tournoi, les jetons n'ont pas une valeur
linéaire.** Doubler son stack ne double pas son espérance de gain, parce que
les paiements sont plafonnés par place. L'ICM convertit une distribution de
stacks en valeur monétaire ($EV), et le **bubble factor** quantifie combien un
call devient plus cher qu'en cash game.

Pourquoi c'était notre lacune n°1 du benchmark : 5 solveurs de référence
l'exposent (Pio, HRC, ICMIZER, Monker, GTO Wizard), et sans lui toute
recommandation en tournoi est fausse près des paliers. Ce module la comble et
alimente directement F13 : l'équité requise d'un call devient
``alpha_icm = 1 - 1/(1 + bubble_factor · (P+b)/b)`` au lieu du pot odds brut.

Complexité : la récurrence de Harville est en O(n!) naïf ; l'implémentation
mémoïse sur les sous-ensembles (O(2ⁿ·n²)), exacte jusqu'à ~12 joueurs — le cas
d'usage réel (table finale). Au-delà, estimation Monte-Carlo par tirages
proportionnels sans remise, avec erreur-type rapportée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

__all__ = [
    "icm_equities",
    "icm_dollar_ev",
    "bubble_factor",
    "risk_premium",
    "icm_required_equity",
    "IcmSpot",
    "analyse_icm_spot",
]

EXACT_LIMIT = 12
"""Au-delà de 12 joueurs, bascule automatique en Monte-Carlo."""


class IcmError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# NOYAU MALMUTH-HARVILLE
# ═══════════════════════════════════════════════════════════════════════════


def _validate(stacks: Sequence[float], payouts: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    s = tuple(float(x) for x in stacks)
    p = tuple(float(x) for x in payouts)
    if not s:
        raise IcmError("au moins un stack requis.")
    if any(x < 0 for x in s):
        raise IcmError("stack négatif.")
    if sum(s) <= 0:
        raise IcmError("somme des stacks nulle.")
    if any(x < 0 for x in p):
        raise IcmError("payout négatif.")
    if len(p) > len(s):
        p = p[: len(s)]
    # Payouts décroissants : convention universelle (1re place d'abord).
    if any(p[i] < p[i + 1] for i in range(len(p) - 1)):
        raise IcmError("payouts doivent être décroissants (1re place d'abord).")
    return s, p


def _finish_probs_exact(stacks: tuple[float, ...], n_places: int) -> np.ndarray:
    """P(joueur i termine à la place k), récurrence de Harville mémoïsée.

    La mémoïsation porte sur le **sous-ensemble de joueurs encore en course**
    (bitmask) : chaque état n'est calculé qu'une fois, d'où O(2ⁿ·n²) au lieu
    de O(n!).
    """
    n = len(stacks)
    probs = np.zeros((n, n_places), dtype=np.float64)

    @lru_cache(maxsize=None)
    def sub_total(mask: int) -> float:
        return sum(stacks[i] for i in range(n) if mask & (1 << i))

    @lru_cache(maxsize=None)
    def place_probs(mask: int, place: int) -> tuple[float, ...]:
        """P(i termine à `place` | ensemble `mask` encore en course)."""
        total = sub_total(mask)
        if total <= 0.0:
            # Plus aucun jeton dans l'ensemble considéré : la récursion a
            # épuisé les joueurs, ou l'un d'eux a un tapis nul.
            #
            # Ce cas n'est pas théorique : `bubble_factor` évalue précisément
            # ce que vaut le tapis du héros APRÈS avoir tout perdu, donc avec
            # un tapis à zéro. La division par le total plantait alors, et
            # c'est tout le calcul de pression de bulle — le nombre qui dit
            # de combien resserrer par rapport au cash — qui était
            # inutilisable. Un joueur sans jeton n'a aucune chance de prendre
            # une place devant : probabilité nulle, pas une exception.
            return tuple([0.0] * n)
        first = tuple(
            (stacks[i] / total if mask & (1 << i) else 0.0) for i in range(n)
        )
        if place == 0:
            return first
        out = [0.0] * n
        for w in range(n):                      # w prend la place au-dessus
            pw = first[w]
            if pw <= 0.0:
                continue
            rest = place_probs(mask & ~(1 << w), place - 1)
            for i in range(n):
                out[i] += pw * rest[i]
        return tuple(out)

    full = (1 << n) - 1
    for k in range(min(n_places, n)):
        probs[:, k] = place_probs(full, k)
    return probs


def _finish_probs_mc(
    stacks: tuple[float, ...], n_places: int, n_sims: int, seed: int | None
) -> tuple[np.ndarray, float]:
    """Estimation Monte-Carlo : tirages successifs proportionnels aux stacks."""
    rng = np.random.default_rng(seed)
    n = len(stacks)
    counts = np.zeros((n, n_places), dtype=np.float64)
    base = np.asarray(stacks, dtype=np.float64)
    idx = np.arange(n)
    for _ in range(n_sims):
        remaining = base.copy()
        alive = idx.copy()
        for k in range(min(n_places, n)):
            p = remaining / remaining.sum()
            j = rng.choice(alive.size, p=p)
            counts[alive[j], k] += 1.0
            remaining = np.delete(remaining, j)
            alive = np.delete(alive, j)
    probs = counts / n_sims
    se = float(np.sqrt(0.25 / n_sims))          # borne supérieure de l'erreur-type
    return probs, se


def icm_equities(
    stacks: Sequence[float],
    payouts: Sequence[float],
    n_sims: int = 200_000,
    seed: int | None = 0,
) -> np.ndarray:
    r"""$EV de chaque joueur : :math:`\$EV_i = \sum_k P(i \text{ finit } k)\,\pi_k`.

    Exact (Harville mémoïsé) jusqu'à 12 joueurs, Monte-Carlo au-delà.

    Examples
    --------
    Le cas d'école : 3 joueurs à égalité, payouts 50/30/20 — chacun vaut
    exactement la moyenne :

    >>> icm_equities([1000, 1000, 1000], [50, 30, 20]).round(6).tolist()
    [33.333333, 33.333333, 33.333333]

    Et la non-linéarité, en une ligne : le gros stack vaut MOINS que sa part
    de jetons, le petit vaut PLUS :

    >>> eq = icm_equities([6000, 3000, 1000], [50, 30, 20])
    >>> float(eq[0]) < 60.0 and float(eq[2]) > 10.0
    True
    """
    s, p = _validate(stacks, payouts)
    if not p:
        return np.zeros(len(s))

    # Un joueur sans jeton est DÉJÀ éliminé : il n'entre pas dans la course
    # aux places, il occupe les dernières. Le traiter comme les autres fait
    # diviser par un total nul ; lui donner zéro fait disparaître son gain,
    # et la somme des équités cesse d'égaler la dotation.
    #
    # Ce n'est pas un cas d'école : `bubble_factor` évalue précisément ce que
    # vaut le tapis du héros APRÈS avoir tout perdu. Lui attribuer zéro au
    # lieu du dernier gain surestime ce qu'il perd, donc la pression de bulle
    # elle-même — mesuré sur une vraie table à neuf joueurs : facteur 2,42
    # face au gros tapis avec le zéro, 2,04 avec le dernier gain, soit une
    # équité exigée de 70,8 % au lieu de 67,1 %.
    #
    # Les joueurs à zéro se partagent à parts égales les dernières places :
    # rien ne permet de les départager, et l'ordre de leur élimination n'est
    # pas dans les données.
    #
    # `vivants` ne peut pas être vide ici : les tapis sont validés positifs ou
    # nuls, donc « aucun vivant » implique une somme nulle, que `_validate` a
    # déjà refusée. Le code qui prétendait partager la dotation dans ce cas
    # était inatteignable — un lecteur pouvait croire le cas traité alors que
    # l'appel lève `IcmError`. Le test `test_icm_invariants.py` épingle ce
    # comportement réel.
    vivants = [i for i, v in enumerate(s) if v > 0.0]
    if len(vivants) < len(s):
        out = np.zeros(len(s), dtype=np.float64)
        restes = list(p[len(vivants):])
        if restes:
            morts = [i for i in range(len(s)) if i not in set(vivants)]
            out[morts] = sum(restes) / len(morts)
        haut = list(p[:len(vivants)])
        if haut:
            out[vivants] = icm_equities([s[i] for i in vivants], haut,
                                        n_sims=n_sims, seed=seed)
        return out

    if len(s) <= EXACT_LIMIT:
        probs = _finish_probs_exact(s, len(p))
    else:
        probs, _ = _finish_probs_mc(s, len(p), n_sims, seed)
    return probs @ np.asarray(p, dtype=np.float64)


def icm_dollar_ev(
    stacks: Sequence[float], payouts: Sequence[float], player: int, **kw
) -> float:
    """$EV d'un joueur donné."""
    eq = icm_equities(stacks, payouts, **kw)
    if not (0 <= player < len(eq)):
        raise IcmError("indice de joueur hors table.")
    return float(eq[player])


# ═══════════════════════════════════════════════════════════════════════════
# BUBBLE FACTOR & ÉQUITÉ REQUISE
# ═══════════════════════════════════════════════════════════════════════════


def bubble_factor(
    stacks: Sequence[float],
    payouts: Sequence[float],
    hero: int,
    villain: int,
    amount: float | None = None,
    **kw,
) -> float:
    r"""Bubble factor de hero contre villain (convention ICMIZER/HRC).

    .. math::
        BF = \frac{\$EV_{\text{hero}} - \$EV_{\text{hero}\,|\,\text{perd}}}
                  {\$EV_{\text{hero}\,|\,\text{gagne}} - \$EV_{\text{hero}}}

    C'est le rapport « ce que je perds en $ si je perds le pot » sur « ce que
    je gagne en $ si je le gagne », pour un même montant de jetons. En cash
    game, BF = 1 partout. En tournoi, BF > 1 dès qu'il y a des paliers — et il
    explose à la bulle contre les stacks qui te couvrent.

    Parameters
    ----------
    amount
        Jetons disputés. Par défaut : all-in pour le tapis effectif
        ``min(stack_hero, stack_villain)`` — la définition standard.
    """
    s, p = _validate(stacks, payouts)
    if hero == villain:
        raise IcmError("hero et villain doivent différer.")
    for i in (hero, villain):
        if not (0 <= i < len(s)):
            raise IcmError("indice hors table.")
    eff = min(s[hero], s[villain])
    amt = eff if amount is None else float(amount)
    if not (0.0 < amt <= eff):
        raise IcmError("amount doit être dans (0, tapis effectif].")

    base = icm_dollar_ev(s, p, hero, **kw)

    win = list(s)
    win[hero] += amt
    win[villain] -= amt
    ev_win = icm_dollar_ev(win, p, hero, **kw)

    lose = list(s)
    lose[hero] -= amt
    lose[villain] += amt
    ev_lose = icm_dollar_ev(lose, p, hero, **kw)

    gain = ev_win - base
    loss = base - ev_lose

    # Le rapport est un 0/0 dès que la structure ne met plus rien en jeu : si
    # tous les joueurs encore en lice touchent le même gain (satellite dont il
    # ne reste que des places équivalentes), déplacer des jetons ne change
    # aucune équité. `gain` et `loss` valent alors zéro EN THÉORIE, et ±1 ulp
    # en pratique.
    #
    # Comparer `gain` à zéro exactement laissait donc passer le résidu, et le
    # même cas dégénéré rendait deux réponses opposées au gré de l'arrondi :
    # −1,0 sur ([1, 2, 3], [1, 1, 1]) — un facteur de bulle NÉGATIF, qui n'a
    # aucun sens et faisait ensuite échouer `analyse_icm_spot` sur le message
    # trompeur « bubble factor doit être > 0 » — et +inf sur
    # ([10, 20, 30, 40], [7, 7, 7, 7]).
    #
    # Le garde-fou se compare donc à l'ÉCHELLE des $EV, pas à zéro. Le seuil
    # 1e-12 est encadré par la mesure (banc_invariants_icm.py --seuils) : le
    # résidu d'arrondi du cas dégénéré pèse ~1e-16 en relatif, tandis que le
    # gain légitime d'un heads-up à 1 jeton contre 1e9 pèse ~3,3e-10.
    #
    # LIMITE MESURÉE, à ne pas découvrir plus tard : le garde-fou renvoie
    # +inf dès que le rapport des tapis atteint ~1e12 en heads-up, où le gain
    # légitime (3,3e-13 en relatif) passe sous le seuil. Ce n'est pas une
    # perte : à ce rapport, la soustraction de deux $EV voisins a déjà perdu
    # tous ses chiffres significatifs — le calcul rendait 0,99996 au lieu
    # de 1 dès 1e11, et du bruit pur au-delà. Répondre +inf (« je ne peux pas
    # trancher, suppose la pression maximale ») est prudent et cohérent, là
    # où un nombre plausible et faux serait le pire des deux. Un tournoi réel
    # ne dépasse pas un rapport de ~1e6.
    if gain <= 0.0:
        return math.inf
    return float(loss / gain)


def risk_premium(bf: float) -> float:
    r"""Prime de risque : équité supplémentaire requise vs cash game.

    Pour un pot symétrique (all-in 50/50 du montant), l'équité de
    neutralité $EV passe de 50 % à :math:`BF/(1+BF)` ; la prime est l'écart.

    >>> risk_premium(1.0)
    0.0
    >>> round(risk_premium(1.5), 4)
    0.1
    """
    if bf < 0:
        raise IcmError("bubble factor négatif.")
    if math.isinf(bf):
        return 0.5
    return float(bf / (1.0 + bf) - 0.5)


def icm_required_equity(pot: float, bet: float, bf: float) -> float:
    r"""Équité requise pour un call en tournoi — le pot odds corrigé ICM.

    En cash : :math:`\alpha = b/(P+2b)` (F10). En tournoi, chaque jeton risqué
    pèse BF fois plus que chaque jeton gagnable :

    .. math::
        \alpha_{ICM} = \frac{BF \cdot b}{P + b + BF \cdot b}

    Se réduit au pot odds classique quand BF = 1.

    >>> round(icm_required_equity(100, 75, 1.0), 4)   # cash
    0.3
    >>> round(icm_required_equity(100, 75, 2.0), 4)   # bulle sévère
    0.4615
    """
    if pot < 0 or bet <= 0:
        raise IcmError("pot >= 0 et bet > 0 requis.")
    if bf <= 0:
        raise IcmError("bubble factor doit être > 0.")
    if math.isinf(bf):
        return 1.0
    return float(bf * bet / (pot + bet + bf * bet))


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSE DE SPOT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class IcmSpot:
    stacks: tuple[float, ...]
    payouts: tuple[float, ...]
    hero: int
    villain: int
    pot: float
    bet: float


@dataclass(frozen=True, slots=True)
class IcmAnalysis:
    dollar_ev: tuple[float, ...]
    hero_ev: float
    bubble: float
    premium: float
    alpha_cash: float
    alpha_icm: float
    verdict: str

    def explain(self) -> str:
        lines = [
            "$EV par joueur : " + "  ".join(f"{v:.2f}" for v in self.dollar_ev),
            f"bubble factor hero→villain : {self.bubble:.3f}"
            f"  (prime de risque {self.premium * 100:+.1f} pts)",
            f"équité requise : cash {self.alpha_cash * 100:.1f} %"
            f"  →  ICM {self.alpha_icm * 100:.1f} %",
            f"→ {self.verdict}",
        ]
        return "\n".join(lines)


def analyse_icm_spot(spot: IcmSpot, hero_equity: float | None = None) -> IcmAnalysis:
    """Analyse complète d'un spot de call en tournoi.

    C'est le point de jonction avec F10/F13 : ``alpha_icm`` remplace le pot
    odds brut dans l'analyse de bluff-catch dès que le spot est en tournoi.
    """
    ev = icm_equities(spot.stacks, spot.payouts)
    bf = bubble_factor(spot.stacks, spot.payouts, spot.hero, spot.villain)
    a_cash = spot.bet / (spot.pot + 2.0 * spot.bet)
    a_icm = icm_required_equity(spot.pot, spot.bet, bf)

    if hero_equity is None:
        verdict = (f"il faut {a_icm * 100:.1f} % d'équité pour payer "
                   f"(contre {a_cash * 100:.1f} % en cash).")
    elif hero_equity >= a_icm:
        verdict = (f"CALL : {hero_equity * 100:.1f} % ≥ {a_icm * 100:.1f} % requis "
                   f"malgré l'ICM.")
    elif hero_equity >= a_cash:
        verdict = (f"FOLD ICM : {hero_equity * 100:.1f} % suffirait en cash "
                   f"({a_cash * 100:.1f} %) mais pas en tournoi "
                   f"({a_icm * 100:.1f} %). C'est LE fold que les joueurs de "
                   f"cash ratent.")
    else:
        verdict = f"FOLD : {hero_equity * 100:.1f} % < {a_cash * 100:.1f} % même en cash."

    return IcmAnalysis(
        dollar_ev=tuple(float(x) for x in ev),
        hero_ev=float(ev[spot.hero]),
        bubble=bf,
        premium=risk_premium(bf),
        alpha_cash=float(a_cash),
        alpha_icm=float(a_icm),
        verdict=verdict,
    )


# ═══════════════════════════════════════════════════════════════════════════
# L5 — BOUNTIES PKO (progressive knockout)
# ═══════════════════════════════════════════════════════════════════════════
#
# Règle PKO standard : éliminer un joueur rapporte la MOITIÉ de sa prime en
# cash ; l'autre moitié s'ajoute à ta propre prime. Ta propre prime ne t'est
# versée que si tu GAGNES le tournoi (le vainqueur encaisse sa prime).
#
# Valeur exacte d'une capture, sous Harville :
#     BV = B_vilain · (½ + ½ · P(héros finit 1er | stacks après victoire))
# avec P(1er) = s/S — le premier étage de Harville, exact.
#
# L'équité requise du call se calcule par différences de $EV, sans
# approximation de conversion jetons↔$ :
#     r* = ($EV_fold − $EV_perd) / ($EV_gagne + BV·1{vilain éliminé} − $EV_perd)
#
# Convention d'états (P = pot TOTAL déjà engagé, y compris la mise adverse ;
# b = ce que héros doit payer) :
#     fold   : vilain += P
#     gagne  : héros += P  (sa mise revient avec le pot)
#     perd   : héros −= b ; vilain += P + b


@dataclass(frozen=True, slots=True)
class PkoSpot:
    stacks: tuple[float, ...]
    payouts: tuple[float, ...]
    bounties: tuple[float, ...]
    hero: int
    villain: int
    pot: float          # pot total, mise adverse incluse
    bet: float          # montant à payer


@dataclass(frozen=True, slots=True)
class PkoAnalysis:
    ev_fold: float
    ev_win: float
    ev_lose: float
    bounty_value: float          # BV si le call couvre le vilain, sinon 0
    villain_eliminated: bool
    required_no_bounty: float    # équité requise ICM pure
    required_with_bounty: float  # équité requise avec la prime
    discount_pts: float          # points d'équité offerts par la prime
    verdict: str

    def explain(self) -> str:
        lines = [
            f"$EV : fold {self.ev_fold:.2f} · gagne {self.ev_win:.2f}"
            f" · perd {self.ev_lose:.2f}",
            (f"prime capturable : {self.bounty_value:.2f} $"
             if self.villain_eliminated else
             "le call ne couvre pas le vilain : pas de prime en jeu"),
            f"équité requise : ICM {self.required_no_bounty * 100:.1f} %"
            f"  →  PKO {self.required_with_bounty * 100:.1f} %"
            f"  (−{self.discount_pts * 100:.1f} pts)",
            f"→ {self.verdict}",
        ]
        return "\n".join(lines)


def bounty_capture_value(
    stacks_after_win: Sequence[float],
    hero: int,
    villain_bounty: float,
    **kw,
) -> float:
    """Valeur $ exacte de la capture : ½ cash + ½ sur sa propre prime.

    La moitié ajoutée à sa propre prime ne vaut que P(finir 1er), qui sous
    Harville est la part de jetons — exact, pas une approximation.

    >>> # Héros à 200/300 jetons après la victoire, prime 50 :
    >>> round(bounty_capture_value([200, 0, 100], 0, 50.0), 4)
    41.6667
    """
    if villain_bounty < 0:
        raise IcmError("prime négative.")
    s = [float(x) for x in stacks_after_win]
    total = sum(s)
    if total <= 0 or not (0 <= hero < len(s)):
        raise IcmError("stacks ou indice héros invalides.")
    p_first = s[hero] / total
    return float(villain_bounty * (0.5 + 0.5 * p_first))


def analyse_pko_spot(spot: PkoSpot, hero_equity: float | None = None) -> PkoAnalysis:
    """Analyse exacte d'un call PKO par différences de $EV (+ prime).

    >>> # Winner-take-all 3-way, stacks égaux, prime 50 : le call 50/50
    >>> # devient un call à 30.8 % grâce à la prime.
    >>> s = PkoSpot(stacks=(100.0, 0.0, 100.0), payouts=(100.0,),
    ...             bounties=(50.0, 50.0, 50.0), hero=0, villain=1,
    ...             pot=100.0, bet=100.0)
    >>> a = analyse_pko_spot(s)
    >>> round(a.required_no_bounty, 6), round(a.required_with_bounty, 6)
    (0.5, 0.307692)
    """
    s, p = _validate(spot.stacks, spot.payouts)
    n = len(s)
    if len(spot.bounties) != n:
        raise IcmError("il faut une prime par joueur.")
    if any(b < 0 for b in spot.bounties):
        raise IcmError("prime négative.")
    h, v = spot.hero, spot.villain
    if h == v or not (0 <= h < n) or not (0 <= v < n):
        raise IcmError("indices héros/vilain invalides.")
    if spot.bet <= 0 or spot.pot < 0:
        raise IcmError("pot >= 0 et bet > 0 requis.")
    if spot.bet > s[h]:
        raise IcmError("héros ne peut pas payer plus que son stack.")

    fold_s = list(s); fold_s[v] += spot.pot
    win_s = list(s);  win_s[h] += spot.pot
    lose_s = list(s); lose_s[h] -= spot.bet; lose_s[v] += spot.pot + spot.bet

    ev_fold = icm_dollar_ev(fold_s, p, h)
    ev_win = icm_dollar_ev(win_s, p, h)
    ev_lose = icm_dollar_ev(lose_s, p, h)

    # Vilain à tapis : son stack est au pot, donc à zéro ici. Le seuil est
    # RELATIF au total en jeu, parce qu'un seuil absolu rendait le verdict
    # dépendant de l'UNITÉ dans laquelle on compte les jetons — alors que
    # l'ICM, lui, est invariant d'échelle. Mesuré sur le même spot (héros 100,
    # vilain 1 jeton restant) : exprimer les tapis en unités de 1e-12 jeton
    # faisait basculer le vilain de « survit » à « éliminé », la prime de 0 à
    # 41,61, et l'équité exigée de 60,0 % à 40,0 %. Même table, même décision,
    # deux réponses.
    eliminated = s[v] <= 1e-12 * sum(s)
    bv = (bounty_capture_value(win_s, h, spot.bounties[v])
          if eliminated else 0.0)

    denom_icm = ev_win - ev_lose
    denom_pko = ev_win + bv - ev_lose
    if denom_icm <= 0 or denom_pko <= 0:
        raise IcmError("spot dégénéré : gagner ne rapporte rien.")
    r_icm = (ev_fold - ev_lose) / denom_icm
    r_pko = (ev_fold - ev_lose) / denom_pko
    r_icm = min(max(r_icm, 0.0), 1.0)
    r_pko = min(max(r_pko, 0.0), 1.0)

    if hero_equity is None:
        verdict = (f"il faut {r_pko * 100:.1f} % d'équité pour payer "
                   f"(ICM pur : {r_icm * 100:.1f} %).")
    elif hero_equity >= r_pko:
        extra = " — la prime fait basculer le call" if hero_equity < r_icm else ""
        verdict = f"CALL : {hero_equity * 100:.1f} % ≥ {r_pko * 100:.1f} %{extra}."
    else:
        verdict = f"FOLD : {hero_equity * 100:.1f} % < {r_pko * 100:.1f} %."

    return PkoAnalysis(ev_fold, ev_win, ev_lose, bv, eliminated,
                       r_icm, r_pko, r_icm - r_pko, verdict)


def spot_pko_face_a_tapis(
    tapis: Sequence[float],
    payouts: Sequence[float],
    primes: Sequence[float],
    hero: int,
    villain: int,
    *,
    blindes_mortes: float = 0.0,
    deja_engage_hero: float = 0.0,
) -> PkoSpot:
    """Construit le spot « il part à tapis, je paie » depuis les tapis RÉELS.

    `PkoSpot` attend des tapis **après** engagement : le vilain à tapis y
    figure à zéro, ses jetons étant comptés dans ``pot``. Cette convention
    est correcte mais se retourne facilement — je m'y suis pris les pieds en
    analysant une vraie table, en passant le tapis entier du vilain : il
    n'était alors pas vu comme éliminé, la prime valait zéro, et l'équité
    exigée sortait à 90 % au lieu de 53 %. Un chiffre faux et plausible, le
    pire genre.

    Cette fonction prend les tapis tels qu'on les lit à l'écran et fait la
    conversion.

    Parameters
    ----------
    tapis : sequence of float
        Tapis observés, dans la même unité que ``payouts`` n'a pas à
        partager (les jetons suffisent, l'ICM est invariant d'échelle).
    primes : sequence of float
        Une prime par joueur, en euros.
    blindes_mortes : float
        Ce qui traîne déjà au pot sans appartenir aux deux joueurs : petite
        blinde couchée, antes de tout le monde.
    deja_engage_hero : float
        Ce que le héros a déjà posé (sa grosse blinde, typiquement) et qu'il
        n'a donc pas à payer une seconde fois.

    Returns
    -------
    PkoSpot
        Prêt pour `analyse_pko_spot`.

    Examples
    --------
    >>> # Vilain à 19,25 bb part à tapis ; héros à 35,64 bb en grosse blinde.
    >>> s = spot_pko_face_a_tapis(
    ...     [35.64, 19.25], [100.0, 40.0], [8.13, 4.17], hero=0, villain=1,
    ...     blindes_mortes=1.4, deja_engage_hero=1.0)
    >>> s.stacks[1], round(s.pot, 2), round(s.bet, 2)
    (0.0, 20.65, 18.25)
    """
    n = len(tapis)
    if not (0 <= hero < n) or not (0 <= villain < n) or hero == villain:
        raise IcmError("indices héros/vilain invalides.")

    # Le vilain ne peut engager que ce qu'il a, et le héros ne peut perdre
    # que jusqu'à concurrence de son propre tapis.
    engage = min(float(tapis[villain]), float(tapis[hero]))
    apres = list(float(t) for t in tapis)
    apres[villain] = float(tapis[villain]) - engage   # 0 s'il est couvert
    apres[hero] = float(tapis[hero]) - deja_engage_hero

    pot = engage + blindes_mortes
    bet = engage - deja_engage_hero
    if bet <= 0:
        raise IcmError("rien à payer : le vilain n'engage pas plus que "
                       "ce que le héros a déjà posé.")
    return PkoSpot(stacks=apres, payouts=list(payouts),
                   bounties=list(primes), hero=hero, villain=villain,
                   pot=pot, bet=bet)


# ═══════════════════════════════════════════════════════════════════════════
# L4 — FGS LÉGER : érosion des blindes futures
# ═══════════════════════════════════════════════════════════════════════════
#
# La critique standard de l'ICM statique : il ignore QUI paiera les blindes.
# Le FGS complet (HRC) simule les mains futures avec équilibres push-fold ;
# la version légère, au premier ordre, fait « passer » les blindes : à chaque
# main future, tout le monde se couche sauf la BB (modèle du walk), les jetons
# sont conservés, les stacks courts saignent. L'ICM recalculé sur ces stacks
# érodés donne des $EV et bubble factors FGS-ajustés.
#
# Hypothèse documentée : érosion fold-everything — pessimiste pour les stacks
# courts, neutre pour les gros. C'est la correction de premier ordre ; le FGS
# à équilibres viendra avec le solveur préflop (Phase 2).


@dataclass(frozen=True, slots=True)
class FgsResult:
    equities: np.ndarray        # (n_hands+1, n_players) — ligne 0 = statique
    stacks_path: np.ndarray     # (n_hands+1, n_players)
    deltas: np.ndarray          # equities[k] − equities[0]

    @property
    def static(self) -> np.ndarray:
        return self.equities[0]

    @property
    def fgs_mean(self) -> np.ndarray:
        """$EV moyen sur l'horizon — la valeur FGS-ajustée."""
        return self.equities[1:].mean(axis=0)


def fgs_equities(
    stacks: Sequence[float],
    payouts: Sequence[float],
    button: int,
    sb: float,
    bb: float,
    ante: float = 0.0,
    n_hands: int | None = None,
    **kw,
) -> FgsResult:
    """ICM sur stacks érodés par les blindes des ``n_hands`` prochaines mains.

    Modèle du walk : SB paie sb, chaque joueur l'ante, la BB ramasse tout —
    conservation des jetons garantie. Un joueur à 0 est éliminé de la
    rotation (son $EV reste défini : 0 jeton → plus bas rang Harville).
    Ce que le modèle mesure : l'ORDRE de passage des blindes — un stack court
    qui doit poster avant de encaisser son walk perd du $EV pendant tout
    l'intervalle, ce que l'ICM statique ne voit pas.

    >>> import numpy as np
    >>> # bouton en 1 : le short (indice 2) poste la SB dès la main suivante
    >>> r = fgs_equities([5000, 3000, 500, 1500], [50, 30, 20],
    ...                  button=1, sb=100, bb=200, n_hands=4)
    >>> bool(r.deltas[1:, 2].min() < -1.0)   # il saigne > 1 pt de $EV
    True
    >>> float(np.abs(r.equities.sum(axis=1) - 100.0).max()) < 1e-6
    True
    """
    s, p = _validate(stacks, payouts)
    n = len(s)
    if not (0 <= button < n):
        raise IcmError("indice de bouton invalide.")
    if sb < 0 or bb <= 0 or ante < 0:
        raise IcmError("blindes invalides (bb > 0, sb >= 0, ante >= 0).")
    if n_hands is None:
        n_hands = sum(1 for x in s if x > 0)
    if n_hands < 1:
        raise IcmError("n_hands >= 1 requis.")

    cur = list(s)
    path = [list(cur)]
    eqs = [list(icm_equities(cur, p, **kw))]
    btn = button
    for _ in range(n_hands):
        alive = [i for i in range(n) if cur[i] > 0]
        if len(alive) < 2:
            path.append(list(cur))
            eqs.append(list(icm_equities(cur, p, **kw)))
            continue
        # rotation : SB = suivant du bouton parmi les vivants, BB = suivant
        order = sorted(alive, key=lambda i: ((i - btn - 1) % n))
        sb_i, bb_i = order[0], order[1]
        pot = 0.0
        pay_sb = min(cur[sb_i], sb)
        cur[sb_i] -= pay_sb
        pot += pay_sb
        for i in alive:                   # antes (celle de la BB revient au pot
            if i != bb_i:                 # qu'elle ramasse : net nul, omise)
                a = min(cur[i], ante)
                cur[i] -= a
                pot += a
        cur[bb_i] += pot                  # walk : la BB ramasse tout
        btn = sb_i                        # le bouton avance d'un cran
        path.append(list(cur))
        eqs.append(list(icm_equities(cur, p, **kw)))

    equities = np.array(eqs, dtype=np.float64)
    stacks_path = np.array(path, dtype=np.float64)
    return FgsResult(equities, stacks_path, equities - equities[0])


def fgs_bubble_factor(
    stacks: Sequence[float],
    payouts: Sequence[float],
    hero: int,
    villain: int,
    button: int,
    sb: float,
    bb: float,
    ante: float = 0.0,
    n_hands: int | None = None,
    **kw,
) -> tuple[float, float]:
    """(BF statique, BF sur stacks érodés à l'horizon) — l'écart mesure la
    pression des blindes sur la décision présente."""
    static = bubble_factor(stacks, payouts, hero, villain, **kw)
    r = fgs_equities(stacks, payouts, button, sb, bb, ante, n_hands, **kw)
    eroded = r.stacks_path[-1]
    if eroded[hero] <= 0 or eroded[villain] <= 0:
        return static, math.inf
    fgs = bubble_factor(eroded.tolist(), payouts, hero, villain, **kw)
    return static, fgs

```

---

## 6. Et le banc qui prétend le valider

`banc_invariants_icm.py` déclare « TOUS LES INVARIANTS TIENNENT », avec des
écarts de l'ordre de 1e-16. Il couvre : invariance d'échelle, invariance par
permutation, conservation de la dotation, monotonie, non-linéarité,
concordance exact ↔ Monte-Carlo, et les mêmes invariants étendus au chemin
PKO.

Il fait 622 lignes et n'est pas reproduit ici pour ne pas noyer le module
lui-même. **Dites-nous ce qu'un tel banc doit contenir** et nous le
compléterons — vous avez déjà trouvé la continuité en zéro, que six
invariants globaux et algébriques ne pouvaient pas voir.

La question qui nous intéresse le plus : **un banc d'invariants peut-il
jamais attraper une erreur de modèle**, ou ne fait-il que vérifier la
cohérence interne d'un calcul qui pourrait être faux de bout en bout ? Si la
réponse est non, quelle est la seule chose qui puisse valider un ICM — et
concorder avec ICMIZER ou HRC n'en est pas une, puisque cela ne prouverait
que la reproduction fidèle d'un modèle que vous avez vous-mêmes qualifié de
structurellement biaisé.
