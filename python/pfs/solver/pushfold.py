r"""Solveur push/fold Nash heads-up — jam/fold à la HoldemResources (HRC).

À tapis courts, la stratégie préflop se réduit à un jeu fini 169 × 169 : la
small blind (héros) choisit *jam* (tout son tapis) ou *fold* pour chacun des
169 groupes de mains, la big blind choisit *call* ou *fold* face au jam.
C'est LE calcul signature de HoldemResources Calculator. Miltersen &
Sørensen (2007) ont montré qu'en heads-up l'équilibre jam/fold est quasi
optimal dans le jeu non contraint jusqu'à ≈ 10-15 bb — d'où son statut
d'étalon pour les fins de tournoi.

Deux monnaies d'évaluation, même algorithme :

- **chipEV** (``payouts=None``) : jeu à somme nulle en jetons — cadre du
  théorème minimax (von Neumann, 1928) ;
- **$EV ICM** (``payouts`` fourni) : chaque issue terminale — fold du héros,
  jam non payé, call gagné, call perdu — est un vecteur de stacks converti
  en dollars par Malmuth-Harville (Harville, 1973) via ``pfs.core.icm``.
  Les autres joueurs de la table sont supposés déjà couchés : ils ne
  contestent pas le pot mais leurs stacks (et antes) pèsent dans le $EV.
  C'est le **repli N-way** — l'équilibre simultané à plusieurs défenseurs
  n'est pas modélisé ici.

Équilibre par **fictitious play** (Brown, 1951) : chaque camp joue la
meilleure réponse à la stratégie *moyenne* de l'adversaire, et Robinson
(1951) garantit la convergence des moyennes vers un équilibre de Nash
(Nash, 1950) dans tout jeu fini à somme nulle — le moyennage amortit
l'oscillation caractéristique de l'itération de meilleures réponses pures.
Le pas de moyennage est :math:`\alpha_t = 1/\sqrt{t}` (fictitious play
affaibli généralisé, Leslie & Collins 2006 : la convergence tient pour tout
:math:`\alpha_t \to 0` de somme divergente) — mesuré ici 50 à 100 fois plus
rapide en exploitabilité que le pas classique :math:`1/t`.

Trois critères d'arrêt, du plus fort au plus faible :

1. **certificat pur** — un profil mutuellement meilleure réponse est un
   équilibre de Nash pur EXACT du jeu discrétisé (certificat, pas une
   heuristique) ;
2. **certificat ε-Nash** — l'exploitabilité du profil moyen (gain maximal
   d'une déviation unilatérale, pondéré par la loi des mains) passe sous
   ``tol`` fois l'enjeu du tapis effectif : indispensable aux équilibres
   *mixtes*, où la moyenne oscille à :math:`\pm\alpha_t` autour du mélange
   exact sans jamais stabiliser la distance L1 ;
3. **stabilité L1** — distance L1 entre stratégies moyennes successives
   sous ``tol``.

Matrice d'équité 169 × 169
--------------------------
``E[i, j]`` = équité préflop (égalités comptées ½) du groupe ``i`` contre le
groupe ``j``, estimée par Monte-Carlo semé (``seed = i·169 + j``) via
``equity_vs_range`` : héros = un combo représentatif du groupe ``i``,
vilain = la range des combos du groupe ``j`` (les combos bloqués sont
retirés par le moteur d'équité).

Deux propriétés EXACTES par symétrie des couleurs justifient les économies :

1. *Un combo représentatif suffit.* Permuter les couleurs envoie tout combo
   du groupe ``i`` sur tout autre combo du même groupe en laissant le groupe
   ``j`` globalement invariant : l'équité conditionnelle est identique pour
   chaque représentant — aucun biais, seulement le bruit Monte-Carlo.
2. *Antisymétrie.* La moyenne s'écrit uniformément sur les paires de combos
   compatibles (le nombre de combos de ``j`` non bloqués est constant sur le
   groupe ``i``), donc ``E[i, j] + E[j, i] = 1`` exactement. Seul le
   triangle supérieur (14 365 paires, diagonale incluse) est simulé ; le
   triangle inférieur est complété par complément — coût divisé par deux ET
   cohérence à somme nulle garantie de la matrice.

Précision et coût : à 3 000 simulations/paire, erreur-type ≈ ±0,9 pt
d'équité (:math:`\sqrt{0.25/3000} \approx 0.0091`) ; à 2 000, ≈ ±1,1 pt.
Construction mesurée : ≈ 7 min en séquentiel, ≈ 3-4 min avec 2 processus
(``n_jobs``) — une seule fois, puis cache disque ``.npy`` versionné dans le
dossier du module.

Approximation documentée (ICM) : l'équité agrège les égalités en demi-gains ;
en $EV, la valeur d'un partage n'est pas exactement la moyenne des états
gagné/perdu. Les partages préflop pèsent ~2 %, le biais résultant est
< 0,2 % de $EV — traitement identique à HRC/ICMIZER.

Références
----------
- Nash, J. (1950). *Equilibrium Points in N-Person Games.*
- von Neumann, J. (1928). *Zur Theorie der Gesellschaftsspiele.*
- Brown, G.W. (1951). *Iterative Solution of Games by Fictitious Play.*
- Robinson, J. (1951). *An Iterative Method of Solving a Game.*
- Leslie, D.S. & Collins, E.J. (2006). *Generalised Weakened Fictitious
  Play.* Games and Economic Behavior 56(2).
- Miltersen, P.B. & Sørensen, T.B. (2007). *A Near-Optimal Strategy for a
  Heads-Up No-Limit Texas Hold'em Poker Tournament.*
- Harville, D.A. (1973). *Assigning Probabilities to the Outcomes of
  Multi-Entry Competitions.*
- Metropolis, N. & Ulam, S. (1949). *The Monte Carlo Method.*
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pfs.core.equity import equity_vs_range
from pfs.core.icm import IcmError, icm_equities
from pfs.core.range_model import (
    COMBO_TO_GROUP,
    GROUP_COMBO_COUNT,
    N_COMBOS,
    N_GROUPS,
    RANKS,
    Range,
    combo_cards,
)

__all__ = [
    "PushFoldError",
    "PushFoldSolution",
    "chart",
    "equity_matrix_169",
    "solve_hu_pushfold",
]

F64 = npt.NDArray[np.float64]

DEFAULT_N_SIMS: int = 3000
"""Simulations Monte-Carlo par paire de groupes (erreur-type ≈ ±0,9 pt)."""

MATRIX_VERSION: int = 1
"""Version du format de cache disque — à incrémenter si la sémantique change."""


class PushFoldError(ValueError):
    """Erreur de paramétrage ou d'état du solveur push/fold."""


# ═══════════════════════════════════════════════════════════════════════════
# MATRICE D'ÉQUITÉ 169 × 169
# ═══════════════════════════════════════════════════════════════════════════


def _rep_combo(group: int) -> tuple[int, int]:
    """Combo représentatif canonique d'un groupe de la grille 13 × 13.

    Paires → ♠♥ ; suited → ♠♠ ; offsuit → ♠♥. Par symétrie des couleurs,
    l'équité contre une range fermée par couleur ne dépend pas du
    représentant choisi (voir docstring du module, propriété 1).

    Parameters
    ----------
    group : int
        Indice de groupe dans ``[0, 169)``.

    Returns
    -------
    tuple[int, int]
        Les deux cartes (encodage ``rang*4 + couleur``).

    Raises
    ------
    PushFoldError
        Si ``group`` est hors domaine.
    """
    if not (0 <= group < N_GROUPS):
        raise PushFoldError(f"groupe {group} hors domaine [0, {N_GROUPS}).")
    r, c = divmod(group, 13)
    if r == c:                      # paire : deux couleurs distinctes
        return r * 4 + 0, r * 4 + 1
    if r < c:                       # suited : même couleur
        return r * 4 + 0, c * 4 + 0
    return c * 4 + 0, r * 4 + 1     # offsuit : hi ♠, lo ♥


@lru_cache(maxsize=N_GROUPS)
def _group_range(group: int) -> Range:
    """Range réduite aux combos d'un seul groupe (poids 1, le reste 0)."""
    if not (0 <= group < N_GROUPS):
        raise PushFoldError(f"groupe {group} hors domaine [0, {N_GROUPS}).")
    return Range((group == COMBO_TO_GROUP).astype(np.float64))


def _pair_equity(i: int, j: int, n_sims: int) -> float:
    """Équité MC du représentant du groupe ``i`` contre la range du groupe ``j``.

    Graine déterministe ``i·169 + j`` : la matrice est reproductible quel que
    soit l'ordre (ou le processus) de calcul des paires.
    """
    hero = _rep_combo(i)
    result = equity_vs_range(
        list(hero), _group_range(j), (), n_sims=n_sims, seed=i * N_GROUPS + j
    )
    return float(result.equity)


def _equity_chunk(task: tuple[tuple[tuple[int, int], ...], int]) -> list[float]:
    """Calcule un paquet de paires — unité de travail du pool de processus."""
    pairs, n_sims = task
    return [_pair_equity(i, j, n_sims) for (i, j) in pairs]


def _resolve_jobs(n_jobs: int | None) -> int:
    """Nombre de processus : ``n_jobs`` explicite, sinon les cœurs visibles."""
    if n_jobs is not None:
        if n_jobs < 1:
            raise PushFoldError("n_jobs doit être >= 1.")
        return n_jobs
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):                    # pragma: no cover
        return max(1, os.cpu_count() or 1)


def _load_cache(path: Path) -> F64 | None:
    """Charge et valide le cache disque ; ``None`` si absent ou corrompu."""
    if not path.is_file():
        return None
    try:
        m = np.load(path)
    except (OSError, ValueError) as exc:
        warnings.warn(f"cache d'équité illisible ({exc}) : reconstruction.",
                      stacklevel=3)
        return None
    if (m.shape != (N_GROUPS, N_GROUPS) or not np.all(np.isfinite(m))
            or m.min() < -1e-9 or m.max() > 1.0 + 1e-9):
        warnings.warn("cache d'équité invalide : reconstruction.", stacklevel=3)
        return None
    return np.clip(m.astype(np.float64), 0.0, 1.0)


def _save_cache(path: Path, matrix: F64) -> None:
    """Écriture atomique (fichier temporaire puis ``os.replace``)."""
    try:
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(path.parent))
        os.close(fd)
        np.save(tmp, matrix)
        os.replace(tmp, path)
    except OSError as exc:                               # pragma: no cover
        warnings.warn(f"cache d'équité non sauvegardé ({exc}) — "
                      "la matrice sera recalculée à la prochaine session.",
                      stacklevel=3)


def equity_matrix_169(
    n_sims: int = DEFAULT_N_SIMS,
    force: bool = False,
    n_jobs: int | None = None,
    progress: bool = False,
    cache_dir: str | Path | None = None,
) -> F64:
    r"""Matrice d'équité préflop 169 × 169, avec cache disque versionné.

    ``E[i, j]`` = équité du groupe ``i`` contre le groupe ``j`` (égalités ½).
    Seul le triangle supérieur est simulé (Monte-Carlo semé ``i·169 + j``,
    combo représentatif contre range de groupe) ; le triangle inférieur vaut
    ``1 − E[j, i]``, égalité exacte par symétrie des couleurs (docstring du
    module). La diagonale est simulée directement et doit valoir ≈ 0,5.

    Parameters
    ----------
    n_sims : int, default 3000
        Simulations par paire. Erreur-type ≈ :math:`\sqrt{0.25/n}` :
        ±0,9 pt à 3 000, ±1,1 pt à 2 000.
    force : bool, default False
        Ignore le cache disque et recalcule.
    n_jobs : int or None
        Processus parallèles (défaut : cœurs visibles). Le résultat est
        identique quel que soit ``n_jobs`` (graines par paire).
    progress : bool, default False
        Affiche l'avancement sur stdout.
    cache_dir : str, Path or None
        Dossier du cache (défaut : dossier de ce module).

    Returns
    -------
    numpy.ndarray
        Matrice ``(169, 169)`` de ``float64`` dans ``[0, 1]``.

    Raises
    ------
    PushFoldError
        Si ``n_sims`` < 100 (bruit rédhibitoire) ou ``n_jobs`` < 1.
    """
    if n_sims < 100:
        raise PushFoldError("n_sims < 100 : bruit Monte-Carlo rédhibitoire.")
    directory = Path(cache_dir) if cache_dir is not None else Path(__file__).parent
    cache = directory / f"_pushfold_equity169_v{MATRIX_VERSION}_n{n_sims}.npy"
    if not force:
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    pairs = [(i, j) for i in range(N_GROUPS) for j in range(i, N_GROUPS)]
    jobs = _resolve_jobs(n_jobs)
    values: list[float]
    if jobs == 1:
        values = []
        for k, (i, j) in enumerate(pairs):
            values.append(_pair_equity(i, j, n_sims))
            if progress and (k + 1) % 2000 == 0:
                print(f"équité 169×169 : {k + 1}/{len(pairs)} paires")
    else:
        n_chunks = min(jobs * 16, len(pairs))
        chunks = [tuple(pairs[k::n_chunks]) for k in range(n_chunks)]
        tasks = [(chunk, n_sims) for chunk in chunks]
        values_by_chunk: list[list[float]] = []
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for res in pool.map(_equity_chunk, tasks):
                values_by_chunk.append(res)
                if progress:
                    done = sum(len(v) for v in values_by_chunk)
                    print(f"équité 169×169 : {done}/{len(pairs)} paires")
        # ré-entrelace : le chunk k contenait les paires k, k+n, k+2n, …
        values = [0.0] * len(pairs)
        for k, res in enumerate(values_by_chunk):
            values[k::n_chunks] = res

    matrix = np.empty((N_GROUPS, N_GROUPS), dtype=np.float64)
    for (i, j), eq in zip(pairs, values, strict=True):
        matrix[i, j] = eq
        if i != j:
            matrix[j, i] = 1.0 - eq          # antisymétrie exacte
    _save_cache(cache, matrix)
    return matrix


@lru_cache(maxsize=1)
def _unblocked_counts() -> F64:
    """``C[g, h]`` = combos du groupe ``h`` compatibles avec le représentant de ``g``.

    C'est le poids de card-removal exact au niveau des groupes : par symétrie
    des couleurs, ce compte est le même quel que soit le combo du groupe ``g``
    tenu. Sert de loi a priori ``P(vilain a h | héros a g)`` ∝ ``C[g, h]``.
    """
    cards = np.array([combo_cards(k) for k in range(N_COMBOS)], dtype=np.int64)
    counts = np.empty((N_GROUPS, N_GROUPS), dtype=np.float64)
    for g in range(N_GROUPS):
        a, b = _rep_combo(g)
        free = ((cards[:, 0] != a) & (cards[:, 0] != b)
                & (cards[:, 1] != a) & (cards[:, 1] != b))
        counts[g] = np.bincount(COMBO_TO_GROUP[free], minlength=N_GROUPS)
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# VALEURS TERMINALES (chipEV ou $EV ICM)
# ═══════════════════════════════════════════════════════════════════════════


def _terminal_stacks(
    stacks: F64, hero: int, villain: int, sb: float, bb: float, ante: float
) -> tuple[F64, F64, F64, F64]:
    """Stacks finaux des quatre issues terminales du jeu jam/fold.

    Tous les joueurs postent l'ante ; le héros poste la SB, le vilain la BB,
    les autres sont couchés (ils perdent leur ante dans chaque issue). Le
    montant disputé au showdown est le tapis effectif ``m = min(héros,
    vilain)`` — l'excédent du plus gros tapis est rendu. La conservation des
    jetons est vérifiée.

    Returns
    -------
    tuple of numpy.ndarray
        ``(fold_sb, jam_bbfold, sb_gagne, sb_perd)``.
    """
    n = stacks.size
    m = float(min(stacks[hero], stacks[villain]))

    def state(d_hero: float, d_villain: float) -> F64:
        out = stacks - ante                  # les couchés perdent leur ante
        out[hero] = stacks[hero] + d_hero
        out[villain] = stacks[villain] + d_villain
        return out

    fold_sb = state(-ante - sb, (n - 1) * ante + sb)
    jam_bbfold = state((n - 1) * ante + bb, -ante - bb)
    sb_win = state(m + (n - 2) * ante, -m)
    sb_lose = state(-m, m + (n - 2) * ante)

    total = float(stacks.sum())
    for st in (fold_sb, jam_bbfold, sb_win, sb_lose):
        if abs(float(st.sum()) - total) > 1e-6:          # pragma: no cover
            raise PushFoldError("conservation des jetons violée (bug interne).")
        if float(st.min()) < -1e-9:
            raise PushFoldError(
                "un stack devient négatif : blindes/antes trop grosses pour "
                "les tapis fournis."
            )
    return fold_sb, jam_bbfold, sb_win, sb_lose


def _terminal_values(
    states: tuple[F64, F64, F64, F64],
    payouts: Sequence[float] | None,
    hero: int,
    villain: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float]]:
    """Valeurs des issues pour chaque camp — jetons (chipEV) ou dollars (ICM).

    Returns
    -------
    tuple
        ``u_sb = (fold, jam_non_payé, gagne, perd)`` pour le héros et
        ``u_bb = (fold_sur_jam, gagne, perd)`` pour le défenseur. En chipEV
        ce sont les stacks finaux (les comparaisons d'EV sont invariantes
        par translation) ; en ICM, les $EV Malmuth-Harville.
    """
    if payouts is None:
        vals = [(float(st[hero]), float(st[villain])) for st in states]
    else:
        # Plancher relatif ε sur les stacks nuls : la récurrence de Harville
        # divise par la somme du sous-ensemble restant, indéfinie si un
        # joueur busté doit encore être classé. La limite est exacte — un
        # stack nul n'est jamais classé devant quiconque — et le plancher la
        # préserve à 1e-9 près.
        floor = 1e-9 * float(states[0].sum())
        try:
            eqs = [icm_equities(np.maximum(st, floor).tolist(), list(payouts))
                   for st in states]
        except IcmError as exc:
            raise PushFoldError(f"paramètres ICM invalides : {exc}") from exc
        vals = [(float(e[hero]), float(e[villain])) for e in eqs]
    u_sb = (vals[0][0], vals[1][0], vals[2][0], vals[3][0])
    u_bb = (vals[1][1], vals[3][1], vals[2][1])
    return u_sb, u_bb


# ═══════════════════════════════════════════════════════════════════════════
# MEILLEURES RÉPONSES (le cœur du fictitious play)
# ═══════════════════════════════════════════════════════════════════════════


def _jam_evs(
    call: F64, equity: F64, counts: F64,
    u_sb: tuple[float, float, float, float],
) -> F64:
    r"""EV du jam pour chacun des 169 groupes du héros, range de call donnée.

    .. math::
        EV_{jam}(i) = \sum_j \tilde w_{ij}\,[\,c_j\,(e_{ij}u_g + (1-e_{ij})u_p)
                      + (1-c_j)\,u_{nc}\,]

    avec :math:`\tilde w_{ij} \propto C[i,j]` (card-removal exact),
    :math:`c_j` la fréquence de call du groupe ``j``, :math:`u_g, u_p,
    u_{nc}` les valeurs gagné / perdu / jam non payé.
    """
    _, u_nocall, u_win, u_lose = u_sb
    w = counts / counts.sum(axis=1, keepdims=True)
    showdown = equity * u_win + (1.0 - equity) * u_lose
    branch = call[None, :] * showdown + (1.0 - call[None, :]) * u_nocall
    return (w * branch).sum(axis=1)


def _margin(*values: float) -> float:
    """Marge de dominance stricte, relative à l'échelle des valeurs.

    Les meilleures réponses exigent ``EV_action > EV_fold + marge`` : un EV
    égal au bruit d'arrondi près est traité comme un fold (l'action nulle).
    Sans cela, un jeu à enjeu plat (payouts identiques) produirait des
    ranges de poussière flottante au lieu d'être exactement trivial.
    """
    return 1e-9 * (1.0 + max(abs(v) for v in values))


def _br_sb(
    call: F64, equity: F64, counts: F64,
    u_sb: tuple[float, float, float, float],
) -> F64:
    """Meilleure réponse du héros : jam ssi ``EV_jam(i) > EV_fold``."""
    evs = _jam_evs(call, equity, counts, u_sb)
    return (evs > u_sb[0] + _margin(*u_sb)).astype(np.float64)


def _br_bb(
    jam: F64, equity: F64, counts: F64,
    u_bb: tuple[float, float, float],
) -> F64:
    r"""Meilleure réponse du défenseur : call ssi ``EV_call(j) > EV_fold``.

    La croyance sur la main du jammeur est bayésienne :
    :math:`P(i \mid j, jam) \propto jam_i \cdot C[j, i]` — fréquence de jam
    pondérée par le card-removal. Comme les valeurs terminales ne dépendent
    de la main adverse que via l'équité, l'EV du call est linéaire en
    l'équité agrégée :math:`\bar e_j`.
    """
    u_fold, u_win, u_lose = u_bb
    m = counts * jam[None, :]
    tot = m.sum(axis=1)
    call = np.zeros(N_GROUPS, dtype=np.float64)
    ok = tot > 1e-12                          # range de jam vide → pas de call
    if not np.any(ok):
        return call
    ebar = (m[ok] * equity[ok]).sum(axis=1) / tot[ok]
    ev_call = ebar * u_win + (1.0 - ebar) * u_lose
    call[ok] = (ev_call > u_fold + _margin(u_fold, u_win, u_lose)).astype(np.float64)
    return call


def _exploitability(
    jam: F64, call: F64, equity: F64, counts: F64,
    u_sb: tuple[float, float, float, float],
    u_bb: tuple[float, float, float],
) -> float:
    r"""Exploitabilité du profil : gain espéré maximal d'une déviation.

    :math:`\varepsilon = \mathbb E_i[\max(EV_{jam}, EV_{fold}) - EV_\sigma]
    + \mathbb E_j[P(jam \mid j)\,(\max(EV_{call}, EV_{fold}) - EV_\sigma)]`,
    espérances prises sous la loi des 169 groupes (poids en combos). Nulle
    exactement à l'équilibre de Nash ; ``ε ≤ tol × enjeu`` est le critère
    ε-Nash standard des solveurs (cf. HRC « Nash distance »).
    """
    prior = GROUP_COMBO_COUNT.astype(np.float64) / N_COMBOS
    ev_jam = _jam_evs(call, equity, counts, u_sb)
    cur_sb = jam * ev_jam + (1.0 - jam) * u_sb[0]
    gain_sb = float((prior * (np.maximum(ev_jam, u_sb[0]) - cur_sb)).sum())

    u_fold, u_win, u_lose = u_bb
    m = counts * jam[None, :]
    tot = m.sum(axis=1)
    p_jam = tot / counts.sum(axis=1)          # P(héros jamme | main du BB)
    safe = np.maximum(tot, 1e-300)
    ebar = np.where(tot > 1e-12, (m * equity).sum(axis=1) / safe, 0.0)
    ev_call = ebar * u_win + (1.0 - ebar) * u_lose
    cur_bb = call * ev_call + (1.0 - call) * u_fold
    gain_bb = float((prior * p_jam
                     * (np.maximum(ev_call, u_fold) - cur_bb)).sum())
    return gain_sb + gain_bb


# ═══════════════════════════════════════════════════════════════════════════
# LA SOLUTION ET LE SOLVEUR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PushFoldSolution:
    """Équilibre (ou meilleure approximation atteinte) du jeu jam/fold.

    Attributes
    ----------
    jam_range, call_range : numpy.ndarray
        Fréquences ``[0, 1]`` par groupe (grille 169). Un équilibre pur
        certifié donne des vecteurs binaires ; sinon ce sont les stratégies
        moyennes du fictitious play (mélanges au sens de Nash).
    jam_pct, call_pct : float
        Part des mains distribuées (pondérée par les 1326 combos).
    ev_jam_par_groupe : numpy.ndarray
        ``EV(jam) − EV(fold)`` par groupe du héros, contre la range de call
        finale — en bb (chipEV) ou en unités de payout (ICM). Positif ⇔ jam.
    n_iter : int
        Itérations de fictitious play effectuées.
    converged : bool
        Vrai si équilibre pur certifié, exploitabilité ≤ tol × enjeu
        (ε-Nash) ou distance L1 des moyennes < tol.
    """

    jam_range: F64
    call_range: F64
    jam_pct: float
    call_pct: float
    ev_jam_par_groupe: F64
    n_iter: int
    converged: bool

    def __repr__(self) -> str:
        tag = "équilibre" if self.converged else "non convergé"
        return (f"PushFoldSolution(jam {self.jam_pct * 100:.1f} %, "
                f"call {self.call_pct * 100:.1f} %, {self.n_iter} it., {tag})")


def _hand_pct(freq: F64) -> float:
    """Fréquences de groupes → part des mains (pondération par combos)."""
    return float((freq * GROUP_COMBO_COUNT).sum() / N_COMBOS)


def solve_hu_pushfold(
    hero_stack_bb: float,
    payouts: Sequence[float] | None = None,
    stacks: Sequence[float] | None = None,
    sb: float = 0.5,
    bb: float = 1.0,
    ante: float = 0.0,
    max_iter: int = 200,
    tol: float = 1e-4,
    *,
    hero: int = 0,
    villain: int = 1,
    equity: F64 | None = None,
    n_sims: int = DEFAULT_N_SIMS,
) -> PushFoldSolution:
    """Équilibre de Nash jam/fold SB contre BB, en chipEV ou en $EV ICM.

    Itération de meilleures réponses amortie par moyennage (fictitious play,
    Brown 1951 ; convergence : Robinson 1951) avec certification d'équilibre
    pur par meilleure réponse mutuelle. Toutes les grandeurs sont en big
    blinds (``sb``, ``bb``, ``ante``, stacks).

    Parameters
    ----------
    hero_stack_bb : float
        Tapis du héros (small blind) en bb, avant de poster.
    payouts : sequence of float, optional
        Paiements décroissants (1re place d'abord). ``None`` → chipEV ;
        fourni → $EV ICM (``stacks`` alors obligatoire).
    stacks : sequence of float, optional
        Tapis de TOUS les joueurs (en bb) pour l'ICM — les joueurs autres
        que ``hero`` et ``villain`` sont supposés couchés. Doit vérifier
        ``stacks[hero] == hero_stack_bb``.
    sb, bb, ante : float
        Structure de blindes en bb (``ante`` payée par chaque joueur).
    max_iter : int, default 200
        Plafond d'itérations de fictitious play.
    tol : float, default 1e-4
        Seuil de convergence, deux lectures (arrêt au premier atteint) :
        exploitabilité du profil moyen ≤ ``tol`` × enjeu du tapis effectif
        (certificat ε-Nash), ou distance L1 < ``tol`` entre stratégies
        moyennes successives. Un équilibre pur certifié arrête aussitôt.
    hero, villain : int, keyword-only
        Indices du héros (SB) et du défenseur (BB) dans ``stacks``.
    equity : numpy.ndarray, keyword-only, optional
        Matrice d'équité ``(169, 169)`` à utiliser (défaut :
        ``equity_matrix_169(n_sims=n_sims)``, cache disque).
    n_sims : int, keyword-only
        Simulations/paire si la matrice doit être construite.

    Returns
    -------
    PushFoldSolution

    Raises
    ------
    PushFoldError
        Paramètres incohérents (blindes, stacks, indices, matrice).
    """
    if not np.isfinite(hero_stack_bb) or hero_stack_bb <= 0:
        raise PushFoldError("hero_stack_bb doit être fini et > 0.")
    if bb <= 0 or sb < 0 or ante < 0:
        raise PushFoldError("il faut bb > 0, sb >= 0, ante >= 0.")
    if max_iter < 1:
        raise PushFoldError("max_iter doit être >= 1.")
    if tol <= 0:
        raise PushFoldError("tol doit être > 0.")

    if payouts is None:
        if stacks is not None:
            raise PushFoldError(
                "stacks sans payouts : ambigu. En chipEV les tapis sont "
                "symétriques (hero_stack_bb) ; fournir payouts pour l'ICM."
            )
        stack_vec = np.array([hero_stack_bb, hero_stack_bb], dtype=np.float64)
        h, v = 0, 1
    else:
        if stacks is None:
            raise PushFoldError("payouts fourni : stacks est obligatoire (ICM).")
        stack_vec = np.asarray(list(stacks), dtype=np.float64)
        if stack_vec.ndim != 1 or stack_vec.size < 2:
            raise PushFoldError("stacks doit contenir au moins 2 joueurs.")
        if np.any(~np.isfinite(stack_vec)) or np.any(stack_vec < 0):
            raise PushFoldError("stacks doivent être finis et >= 0.")
        h, v = hero, villain
        if h == v or not (0 <= h < stack_vec.size) or not (0 <= v < stack_vec.size):
            raise PushFoldError("indices hero/villain invalides.")
        if abs(float(stack_vec[h]) - hero_stack_bb) > 1e-6:
            raise PushFoldError(
                f"stacks[hero]={float(stack_vec[h])} ≠ hero_stack_bb="
                f"{hero_stack_bb} : incohérent."
            )
    eff = float(min(stack_vec[h], stack_vec[v]))
    if eff <= bb + ante:
        raise PushFoldError(
            "tapis effectif <= bb + ante : le jeu jam/fold est dégénéré "
            "(la BB est engagée d'office)."
        )
    if float(stack_vec[h]) <= sb + ante:
        raise PushFoldError("le héros ne couvre pas sb + ante.")

    if equity is None:
        eq_mat = equity_matrix_169(n_sims=n_sims)
    else:
        eq_mat = np.asarray(equity, dtype=np.float64)
        if eq_mat.shape != (N_GROUPS, N_GROUPS):
            raise PushFoldError(f"matrice d'équité attendue "
                                f"({N_GROUPS}, {N_GROUPS}), reçu {eq_mat.shape}.")
        if not np.all(np.isfinite(eq_mat)) or eq_mat.min() < 0 or eq_mat.max() > 1:
            raise PushFoldError("matrice d'équité hors [0, 1] ou non finie.")

    states = _terminal_stacks(stack_vec, h, v, sb, bb, ante)
    u_sb, u_bb = _terminal_values(states, payouts, h, v)
    counts = _unblocked_counts()

    # ── fictitious play affaibli : BR sur les stratégies moyennes ────────
    # Enjeu du tapis effectif : normalise le critère ε-Nash (en bb ou en $).
    swing = 0.5 * (abs(u_sb[2] - u_sb[3]) + abs(u_bb[1] - u_bb[2]))
    if swing < 1e-12:
        swing = 1.0                            # payouts plats : jeu trivial
    jam_avg = np.ones(N_GROUPS, dtype=np.float64)     # a priori : jam tout
    call_avg = np.zeros(N_GROUPS, dtype=np.float64)
    prev_jam, prev_call = jam_avg.copy(), call_avg.copy()
    jam_final, call_final = jam_avg, call_avg
    converged = False
    n_iter = 0

    for t in range(1, max_iter + 1):
        n_iter = t
        call_br = _br_bb(jam_avg, eq_mat, counts, u_bb)
        call_avg = call_avg + (call_br - call_avg) / np.sqrt(t)
        jam_br = _br_sb(call_avg, eq_mat, counts, u_sb)
        jam_avg = jam_avg + (jam_br - jam_avg) / np.sqrt(t + 1)

        # certificat n°1 : la dernière BR forme-t-elle un équilibre pur ?
        call_cand = _br_bb(jam_br, eq_mat, counts, u_bb)
        if np.array_equal(_br_sb(call_cand, eq_mat, counts, u_sb), jam_br):
            jam_final, call_final = jam_br, call_cand
            converged = True
            break
        # certificat n°2 : ε-Nash du profil moyen (équilibres mixtes)
        eps = _exploitability(jam_avg, call_avg, eq_mat, counts, u_sb, u_bb)
        if eps <= tol * swing:
            jam_final, call_final = jam_avg, call_avg
            converged = True
            break
        # repli : stabilité L1 des stratégies moyennes (Robinson)
        dist = float(np.abs(jam_avg - prev_jam).sum()
                     + np.abs(call_avg - prev_call).sum())
        if dist < tol:
            jam_final, call_final = jam_avg, call_avg
            converged = True
            break
        prev_jam, prev_call = jam_avg.copy(), call_avg.copy()
        jam_final, call_final = jam_avg, call_avg

    ev_jam = _jam_evs(call_final, eq_mat, counts, u_sb) - u_sb[0]
    return PushFoldSolution(
        jam_range=jam_final,
        call_range=call_final,
        jam_pct=_hand_pct(jam_final),
        call_pct=_hand_pct(call_final),
        ev_jam_par_groupe=ev_jam,
        n_iter=n_iter,
        converged=converged,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CHART TEXTE 13 × 13 (style HRC)
# ═══════════════════════════════════════════════════════════════════════════


def chart(solution: PushFoldSolution, threshold: float = 0.5) -> str:
    """Grille 13 × 13 de la range de jam — ``+`` jam, ``.`` fold.

    Convention solveur : diagonale = paires, au-dessus = suited, au-dessous
    = offsuit. Un groupe est marqué ``+`` si sa fréquence de jam atteint
    ``threshold``.

    Parameters
    ----------
    solution : PushFoldSolution
        Solution dont afficher ``jam_range``.
    threshold : float, default 0.5
        Fréquence minimale (dans ``(0, 1]``) pour marquer ``+``.

    Returns
    -------
    str
        14 lignes (en-tête + 13 rangées) puis un résumé.

    Raises
    ------
    PushFoldError
        Si ``threshold`` est hors ``(0, 1]`` ou la range mal dimensionnée.

    Examples
    --------
    >>> import numpy as np
    >>> sol = PushFoldSolution(np.ones(169), np.ones(169), 1.0, 1.0,
    ...                        np.zeros(169), 1, True)
    >>> print(chart(sol).splitlines()[1])
    A   +  +  +  +  +  +  +  +  +  +  +  +  +
    """
    if not (0.0 < threshold <= 1.0):
        raise PushFoldError("threshold doit être dans (0, 1].")
    jam = np.asarray(solution.jam_range, dtype=np.float64).ravel()
    if jam.size != N_GROUPS:
        raise PushFoldError(f"jam_range doit avoir {N_GROUPS} entrées.")
    lines = ["    " + "  ".join(RANKS)]
    for r in range(13):
        cells = "  ".join(
            "+" if jam[r * 13 + c] >= threshold else "." for c in range(13)
        )
        lines.append(f"{RANKS[r]}   {cells}")
    lines.append(f"jam : {solution.jam_pct * 100:.1f} % des mains "
                 f"(seuil {threshold:.2f})")
    return "\n".join(lines)
