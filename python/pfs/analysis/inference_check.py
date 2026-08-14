r"""
Validation prédictive de l'inférence adverse — le modèle bat-il le prior ?

Statut (audit du 14 août 2026) — banc de validation, NON branché
----------------------------------------------------------------
Importé par aucun code applicatif ni script livré : seul
``tests/test_inference_check.py`` l'exerce (et ce module est, hors tests,
l'unique consommateur de ``pfs.data.population``). Pourquoi : c'est un
instrument de mesure hors-ligne — il répond à « le modèle bat-il le
prior ? » sur un corpus de hand histories, pas à une question qu'un joueur
pose pendant ou après une session ; l'exécuter demande un corpus et du
temps de calcul, pas un clic. Le banc reste dans le dépôt (règle « mesurer
avant de publier ») même sans route. Ce qui existe déjà :
``evaluer_inference`` / ``evaluer_dossiers``, quatre lectures de
significativité, contrôles anti-fuite — testés. Accroche naturelle si on
voulait l'exposer : un script CLI à côté de ``banc_corpus_pluribus.py``, ou
un onglet « validation » de la route ``review``
(``pfs/app/server.py:API.review``). Règle du projet : aucun module fusionné
sans route + UI + test de bout en bout.

C'est la thèse centrale du projet, et elle n'était jusqu'ici pas mesurée : le
modèle adverse appris en ligne (trackers Beta-Binomiaux F1,
:mod:`pfs.fusion.dynamic_beta`, qui alimentent le filtre particulaire F3)
prédit-il l'action suivante d'un joueur MIEUX qu'un prior de population ?

Protocole préquentiel
---------------------
Les mains sont triées chronologiquement puis parcourues une seule fois. À
chaque occasion observée (joueur × statistique), deux prédicteurs annoncent la
probabilité de l'action avant de la voir :

* **baseline** — taux de population du corpus, lissé Jeffreys ;
* **modèle** — moyenne postérieure du tracker du joueur, lue AVANT d'ingérer
  l'observation courante.

À la première occasion d'un joueur, son tracker est créé sur le prior de
population lu à cet instant : les deux prédicteurs annoncent alors le même
nombre et la différence est nulle *par construction*, quel que soit le mode de
baseline — à l'arrondi de α/(α+β) près, soit 2,2e-16 au pire mesuré. C'est la
signature structurelle de l'absence de fuite, et elle est vérifiée par test
dans les deux modes.

Métriques : log-vraisemblance prédictive moyenne (plus haut = meilleur) et
score de Brier (plus bas = meilleur).

Significativité — quatre lectures, parce qu'une seule mentirait
---------------------------------------------------------------
1. **Wilcoxon** sur les différences par observation. Exigé par le protocole,
   mais ANTICONSERVATIF ici : les occasions d'un même joueur ne sont pas
   indépendantes. À lire comme une borne optimiste, jamais seule.
2. **Permutation des étiquettes de joueur** — le test qui a du sens. On
   redistribue au hasard les occasions entre les joueurs en conservant la
   taille des groupes : les fréquences marginales sont intactes, seul le lien
   joueur ↔ comportement est détruit. La p-value est la fraction de
   permutations qui égalent ou dépassent le gain observé.
3. **Permutation stratifiée par table** — la même chose, mais en ne
   redistribuant les étiquettes qu'entre joueurs d'une MÊME table. Confond
   nommé et écarté : un modèle « par joueur » capte aussi le niveau de la
   table (structure de blindes, profondeur des tapis, vitesse du tournoi), et
   la permutation globale, qui casse ce niveau-là aussi, le compterait comme
   du signal joueur. Le gain qui survit à la stratification est, lui,
   authentiquement individuel.
4. **Bootstrap par joueur** (cluster bootstrap) — intervalle de confiance du
   gain, en rééchantillonnant les JOUEURS et non les observations.

Piège mesuré : à ``discount = 1``, la log-vraisemblance préquentielle
Beta-Bernoulli est **échangeable** — permuter l'ordre des occasions d'un même
joueur ne change rien au total, puisque ce total est la vraisemblance
marginale de la séquence, fonction des seuls comptages. Une « permutation de
l'ordre » n'a donc aucun pouvoir pour détecter une fuite temporelle ; c'est la
permutation des étiquettes de joueur qui en a. Vérifié dans les tests.

Avantage laissé à la baseline : son taux de population est estimé sur le
corpus ENTIER, futur compris, alors que le modèle ne voit que le passé du
joueur. La comparaison est donc défavorable au modèle — un gain mesuré dans
ces conditions n'est pas un artefact de protocole.

``baseline_prequentielle`` retire cet avantage des DEUX côtés : le taux de
population n'utilise que le passé, et c'est ce même taux, lu à la première
occasion d'un joueur, qui sert de prior à son tracker. Ne handicaper que la
baseline rendrait la variante plus facile, pas plus dure — mesuré sur le
corpus réel, garder côté modèle le prior plein-corpus rapportait à lui seul
+0,011552 de log-vraisemblance sur la tranche k = 0 de vpip, c'est-à-dire
avant d'avoir vu la moindre main du joueur.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats as sps

from pfs.data.hand_history import ParsedHand, parse_file
from pfs.data.population import JEFFREYS, STATS
from pfs.fusion.dynamic_beta import DynamicBetaTracker

__all__ = [
    "STATS_EVALUEES",
    "InferenceCheckError",
    "PointCourbe",
    "PointBalayage",
    "ResultatStat",
    "RapportInference",
    "evaluer_inference",
    "evaluer_dossiers",
    "force_prior_empirique",
]

F64 = npt.NDArray[np.float64]

STATS_EVALUEES: tuple[str, ...] = STATS

# Bornes des tranches de la courbe d'apprentissage : « après k mains vues de
# ce joueur, le modèle bat-il le prior ? ». La tranche k = 0 est isolée parce
# que le modèle y recopie la baseline (Δ nul à l'arrondi près, cf. _NUL).
_TRANCHES: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 1_000_000),
)

# Les probabilités sortent de postérieurs Beta à paramètres strictement
# positifs, donc jamais 0 ni 1 ; le rognage ne sert qu'à rendre le log
# insensible à un prior dégénéré fourni par un appelant.
_EPS = 1e-12

# Seuil sous lequel une différence par observation est réputée nulle. À k = 0
# le modèle recopie la baseline, mais le rapport α/(α+β) ne restitue le prior
# à l'identique qu'à l'arrondi près : 2,2e-16 au pire sur toutes les
# configurations mesurées du corpus réel. Les différences réelles y commencent
# à 1,0e-5 — quatre décades plus haut, le seuil ne peut trancher au milieu de
# rien.
_NUL = 1e-9

_RE_STARTDATE = re.compile(r"<startdate>([^<]+)</startdate>")

# Puissance visée pour l'extrapolation du nombre d'observations nécessaires.
_Z_ALPHA = 1.959963984540054      # bilatéral 5 %
_Z_BETA = 0.8416212335729143      # puissance 80 %


class InferenceCheckError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# ORDRE CHRONOLOGIQUE
# ═══════════════════════════════════════════════════════════════════════════


def _cle_chronologique(hand: ParsedHand) -> tuple[str, int]:
    """Clé de tri temporel d'une main.

    Parameters
    ----------
    hand : ParsedHand
        Main parsée.

    Returns
    -------
    tuple[str, int]
        ``(startdate, gamecode)``. ``ParsedHand`` ne porte pas de champ
        horodatage ; l'estampille est relue dans ``raw``, où le parseur iPoker
        recopie le ``<game>`` intégral (``<startdate>YYYY-MM-DD HH:MM:SS</…>``,
        donc triable lexicographiquement). Sans estampille, la clé est
        constante et le tri — stable — préserve l'ordre d'entrée : c'est à
        l'appelant de fournir les mains dans l'ordre.
    """
    m = _RE_STARTDATE.search(hand.raw)
    date = m.group(1).strip() if m else ""
    hid = int(hand.hand_id) if hand.hand_id.isdigit() else 0
    return (date, hid)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTE DES OCCASIONS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class _Occasions:
    """Occasions d'une statistique, dans l'ordre chronologique.

    ``joueurs`` et ``tables`` indexent joueurs et tables par entier pour que
    permutation d'étiquettes et bootstrap restent vectorisables.
    """

    stat: str
    joueurs: npt.NDArray[np.int64]
    tables: npt.NDArray[np.int64]
    resultats: npt.NDArray[np.bool_]
    n_joueurs: int


def _collecter(
    hands: Sequence[ParsedHand], stats: Sequence[str], exclure_heros: bool
) -> dict[str, _Occasions]:
    """Extrait, par statistique, la suite chronologique des occasions.

    Une statistique absente de ``stat_observations`` signifie « pas
    d'occasion » : elle ne produit aucune ligne, sous peine de fabriquer des
    zéros qui n'existent pas (même règle que F1 et que le mining de
    population).
    """
    brut: dict[str, list[tuple[str, str, bool]]] = {s: [] for s in stats}
    for hand in sorted(hands, key=_cle_chronologique):
        for seat in hand.seats:
            if exclure_heros and seat.is_hero:
                continue
            obs = hand.stat_observations(seat.player)
            for stat in stats:
                if stat in obs:
                    brut[stat].append((seat.player, hand.table, bool(obs[stat])))

    out: dict[str, _Occasions] = {}
    for stat, lignes in brut.items():
        if not lignes:
            continue
        i_joueur: dict[str, int] = {}
        i_table: dict[str, int] = {}
        idx = np.empty(len(lignes), dtype=np.int64)
        tab = np.empty(len(lignes), dtype=np.int64)
        res = np.empty(len(lignes), dtype=np.bool_)
        for i, (joueur, table, valeur) in enumerate(lignes):
            idx[i] = i_joueur.setdefault(joueur, len(i_joueur))
            tab[i] = i_table.setdefault(table, len(i_table))
            res[i] = valeur
        out[stat] = _Occasions(stat, idx, tab, res, len(i_joueur))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# PRÉDICTEURS
# ═══════════════════════════════════════════════════════════════════════════


def force_prior_empirique(n_pop: int, n0: float = 30.0) -> float:
    """Force du prior de population, en pseudo-observations.

    Reprend la règle de :meth:`pfs.data.population.PopulationMiner.prior_for` :
    ``min(n0, n_pop / 10)`` — il faut 300 occasions de pool pour mériter un
    prior de force 30. Utilisée comme valeur par défaut pour que
    l'évaluation teste le modèle TEL QU'IL EST CALIBRÉ dans le projet, et non
    une variante ajustée après coup sur le résultat.
    """
    if n_pop < 0:
        raise InferenceCheckError("n_pop >= 0 requis.")
    return min(n0, n_pop / 10.0)


def _taux_population(resultats: npt.NDArray[np.bool_]) -> float:
    """Taux global du corpus, lissé Jeffreys — jamais 0 ni 1."""
    k = float(resultats.sum())
    n = float(resultats.size)
    return (k + JEFFREYS) / (n + 2.0 * JEFFREYS)


def _baseline_prequentielle(resultats: npt.NDArray[np.bool_]) -> F64:
    """Taux de population n'utilisant que les occasions strictement passées."""
    cumul = np.concatenate(([0.0], np.cumsum(resultats, dtype=np.float64)[:-1]))
    n = np.arange(resultats.size, dtype=np.float64)
    return (cumul + JEFFREYS) / (n + 2.0 * JEFFREYS)


def _predire_sequentiel(
    stat: str,
    joueurs: npt.NDArray[np.int64],
    resultats: npt.NDArray[np.bool_],
    taux_prior: float | F64,
    force: float,
    discount: float,
) -> tuple[F64, npt.NDArray[np.int64]]:
    """Prédictions préquentielles du modèle F1, joueur par joueur.

    Parameters
    ----------
    stat : str
        Nom de la statistique (passé au tracker).
    joueurs : ndarray of int
        Indice de joueur de chaque occasion, ordre chronologique.
    resultats : ndarray of bool
        Action observée.
    taux_prior : float or ndarray of float
        Moyenne du prior de population. Un tableau donne le taux connu à
        l'instant i, celui du tracker créé à cette occasion-là : c'est ce qui
        rend le modèle exactement aussi aveugle au futur que la baseline
        qu'on lui oppose, et donc Δ = 0 à k = 0 dans les deux modes.
    force : float
        Force du prior en pseudo-observations.
    discount : float
        Facteur d'actualisation de :class:`DynamicBetaTracker`.

    Returns
    -------
    p : ndarray of float
        Probabilité annoncée AVANT de voir l'occasion i.
    k : ndarray of int
        Nombre d'occasions du joueur déjà ingérées à l'instant i.

    Notes
    -----
    Le tracker réel du projet est utilisé, pas une réimplémentation : ce qui
    est mesuré ici est bien le code qui tourne en production.
    """
    prior = np.broadcast_to(
        np.asarray(taux_prior, dtype=np.float64), (joueurs.size,))
    trackers: dict[int, DynamicBetaTracker] = {}
    p = np.empty(joueurs.size, dtype=np.float64)
    k = np.empty(joueurs.size, dtype=np.int64)
    for i in range(joueurs.size):
        j = int(joueurs[i])
        tr = trackers.get(j)
        if tr is None:
            tr = DynamicBetaTracker(
                stat, discount=discount,
                prior_mean=float(prior[i]), prior_strength=force,
            )
            trackers[j] = tr
        b = tr.belief
        p[i] = b.mean
        k[i] = b.n_observations
        tr.update(bool(resultats[i]))
    return p, k


def _log_vraisemblance(p: F64, x: npt.NDArray[np.bool_]) -> F64:
    """Log-vraisemblance prédictive par observation (plus haut = meilleur)."""
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.where(x, np.log(q), np.log1p(-q))


def _brier(p: F64, x: npt.NDArray[np.bool_]) -> F64:
    """Score de Brier par observation (plus bas = meilleur)."""
    d = p - x.astype(np.float64)
    return d * d


# ═══════════════════════════════════════════════════════════════════════════
# TESTS DE SIGNIFICATIVITÉ
# ═══════════════════════════════════════════════════════════════════════════


def _wilcoxon(delta: F64) -> float:
    """Rangs signés sur les différences par observation, unilatéral (modèle >).

    Les différences numériquement nulles — première occasion de chaque
    joueur, où le modèle recopie la baseline — sont écartées : les laisser
    entrer donnerait un rang à du bruit d'arrondi. Renvoie ``nan`` si aucune
    différence réelle ne subsiste : sans variation, il n'y a rien à tester.
    """
    d = delta[np.abs(delta) > _NUL]
    if d.size < 1:
        return float("nan")
    return float(sps.wilcoxon(d, alternative="greater").pvalue)


def _bootstrap_joueurs(
    delta: F64, joueurs: npt.NDArray[np.int64], n_joueurs: int,
    rng: np.random.Generator, n_tirages: int,
) -> F64:
    """Distribution bootstrap du gain moyen, en rééchantillonnant les JOUEURS.

    Rééchantillonner les observations supposerait leur indépendance, qui est
    fausse : c'est la corrélation intra-joueur qui porte tout le signal.
    """
    sommes = np.bincount(joueurs, weights=delta, minlength=n_joueurs)
    tailles = np.bincount(joueurs, minlength=n_joueurs).astype(np.float64)
    tir = rng.integers(0, n_joueurs, size=(n_tirages, n_joueurs))
    return sommes[tir].sum(axis=1) / tailles[tir].sum(axis=1)


def _blocs(cles: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], ...]:
    """Indices des occasions regroupés par clé de bloc (une table)."""
    ordre = np.argsort(cles, kind="stable")
    coupes = np.flatnonzero(np.diff(cles[ordre])) + 1
    return tuple(b for b in np.split(ordre, coupes) if b.size > 1)


def _p_permutation(
    occ: _Occasions, base_logv: F64, taux_prior: F64, force: float,
    discount: float, gain_observe: float,
    rng: np.random.Generator, n_permutations: int,
    stratifiee: bool,
) -> float:
    """p-value par permutation des étiquettes de joueur.

    Parameters
    ----------
    stratifiee : bool
        Faux : permutation globale — hypothèse nulle « le comportement ne
        dépend d'aucune identité ». Vrai : permutation confinée à chaque
        table — hypothèse nulle « au sein d'une table, le comportement ne
        dépend pas du joueur », qui laisse intact tout effet de niveau de
        table et ne teste donc que l'hétérogénéité individuelle.

    Returns
    -------
    float
        ``(1 + #{gain_perm >= gain_obs}) / (B + 1)`` — estimateur qui ne
        renvoie jamais 0, une p-value nulle n'ayant pas de sens avec un
        nombre fini de tirages.
    """
    if n_permutations <= 0:
        return float("nan")
    groupes = _blocs(occ.tables) if stratifiee else None
    n_ge = 0
    for _ in range(n_permutations):
        if groupes is None:
            permutes = rng.permutation(occ.joueurs)
        else:
            permutes = occ.joueurs.copy()
            for g in groupes:
                permutes[g] = occ.joueurs[rng.permutation(g)]
        p_perm, _ = _predire_sequentiel(
            occ.stat, permutes, occ.resultats, taux_prior, force, discount)
        gain = float(
            (_log_vraisemblance(p_perm, occ.resultats) - base_logv).mean())
        if gain >= gain_observe:
            n_ge += 1
    return (1.0 + n_ge) / (n_permutations + 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# RAPPORT
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PointCourbe:
    """Gain du modèle sur une tranche de « mains déjà vues de ce joueur »."""

    k_min: int
    k_max: int
    n_obs: int
    gain_logv: float          # modèle − baseline, par observation
    gain_brier: float         # baseline − modèle, par observation

    @property
    def libelle(self) -> str:
        if self.k_min == self.k_max:
            return f"k={self.k_min}"
        if self.k_max >= 1_000_000:
            return f"k≥{self.k_min}"
        return f"k={self.k_min}–{self.k_max}"


@dataclass(frozen=True, slots=True)
class PointBalayage:
    """Sensibilité du gain à la force du prior de population."""

    force: float
    gain_logv: float
    n_obs: int


@dataclass(frozen=True, slots=True)
class ResultatStat:
    """Verdict pour une statistique."""

    stat: str
    n_obs: int
    n_joueurs: int
    taux_population: float
    force_prior: float
    logv_baseline: float
    logv_modele: float
    brier_baseline: float
    brier_modele: float
    p_wilcoxon: float
    p_permutation: float
    p_permutation_table: float
    ic_bootstrap: tuple[float, float]
    erreur_type: float
    n_obs_requis: float
    courbe: tuple[PointCourbe, ...]
    balayage: tuple[PointBalayage, ...]

    @property
    def gain_logv(self) -> float:
        """Log-vraisemblance gagnée par observation (positif = modèle meilleur)."""
        return self.logv_modele - self.logv_baseline

    @property
    def gain_brier(self) -> float:
        """Brier économisé par observation (positif = modèle meilleur)."""
        return self.brier_baseline - self.brier_modele

    @property
    def significatif(self) -> bool:
        """Gain positif ET les DEUX permutations sous 5 %.

        La permutation stratifiée est exigée en plus de la globale : sans
        elle, un simple effet de niveau de table passerait pour de la lecture
        d'adversaire.
        """
        return (self.gain_logv > 0.0
                and self.p_permutation < 0.05
                and self.p_permutation_table < 0.05)

    @property
    def k_seuil(self) -> int | None:
        """Première tranche (k ≥ 1) où le gain devient positif, si elle existe."""
        for pt in self.courbe:
            if pt.k_min >= 1 and pt.n_obs > 0 and pt.gain_logv > 0.0:
                return pt.k_min
        return None

    @property
    def conclusion(self) -> str:
        if self.significatif:
            return "le modèle BAT le prior de population"
        if (self.gain_logv > 0.0 and self.p_permutation < 0.05
                and self.p_permutation_table >= 0.05):
            return ("gain réel mais NON attribuable au joueur : il disparaît "
                    "quand on permute à table constante")
        if self.gain_logv < 0.0:
            return "aucune différence détectable (le modèle est même en retrait)"
        return "gain positif mais NON significatif"


@dataclass(frozen=True, slots=True)
class RapportInference:
    """Rapport de validation prédictive de l'inférence adverse."""

    n_mains: int
    n_joueurs: int
    n_tables: int
    n_joueurs_multi_tables: int
    discount: float
    exclure_heros: bool
    baseline_prequentielle: bool
    n_permutations: int
    resultats: tuple[ResultatStat, ...]
    n_fichiers_ignores: int = 0

    def resultat(self, stat: str) -> ResultatStat:
        for r in self.resultats:
            if r.stat == stat:
                return r
        raise InferenceCheckError(f"statistique « {stat} » non évaluée.")

    @property
    def stats_significatives(self) -> tuple[str, ...]:
        return tuple(r.stat for r in self.resultats if r.significatif)

    def explain(self) -> str:
        """Rapport lisible : chiffres, p-values, limites et conclusion."""
        base = ("préquentielle (sans fuite)" if self.baseline_prequentielle
                else "corpus entier (avantage laissé à la baseline)")
        lines = [
            "══════════════════════════════════════════════════════════════",
            "  INFÉRENCE ADVERSE — le modèle appris bat-il le prior ?",
            "══════════════════════════════════════════════════════════════",
            f"  Mains analysées   : {self.n_mains}",
            f"  Joueurs distincts : {self.n_joueurs}"
            f"   (héros {'exclu' if self.exclure_heros else 'inclus'})",
            f"  Tables distinctes : {self.n_tables}"
            f"   ·   joueurs revus sur ≥2 tables : "
            f"{self.n_joueurs_multi_tables}",
            f"  Baseline          : {base}",
            f"  Actualisation F1  : δ = {self.discount}",
            f"  Permutations      : {self.n_permutations}",
        ]
        if self.n_fichiers_ignores:
            lines.append(
                f"  ATTENTION : {self.n_fichiers_ignores} fichier(s) sans "
                "main exploitable — n_obs ci-dessous amputés d'autant.")
        lines.append("")
        for r in self.resultats:
            lines += [
                f"  ── {r.stat} ──  {r.n_obs} occasions, {r.n_joueurs} joueurs,"
                f" taux pool {r.taux_population * 100:.1f} %"
                f" (prior de force {r.force_prior:.1f})",
                f"     log-vraisemblance/obs : baseline {r.logv_baseline:+.5f}"
                f"   modèle {r.logv_modele:+.5f}   gain {r.gain_logv:+.5f}",
                f"     Brier/obs             : baseline {r.brier_baseline:.5f}"
                f"    modèle {r.brier_modele:.5f}    gain {r.gain_brier:+.5f}",
                f"     IC95 du gain (bootstrap joueurs) :"
                f" [{r.ic_bootstrap[0]:+.5f} , {r.ic_bootstrap[1]:+.5f}]",
                f"     p (permutation globale) = {r.p_permutation:.4f}"
                f"   ·   p (permutation à table constante) ="
                f" {r.p_permutation_table:.4f}",
                f"     p (Wilcoxon, anticonservatif — pour mémoire) ="
                f" {r.p_wilcoxon:.4g}",
                f"     → {r.conclusion}",
            ]
            seuil = r.k_seuil
            if seuil is not None:
                lines.append(
                    f"     gain positif à partir de la tranche k≥{seuil} "
                    "mains vues de ce joueur")
            lines.append("     courbe (gain log-vrais. par tranche de mains vues) :")
            for pt in r.courbe:
                if pt.n_obs == 0:
                    continue
                lines.append(
                    f"        {pt.libelle:<9} n={pt.n_obs:<5d}"
                    f" gain {pt.gain_logv:+.5f}")
            if r.balayage:
                detail = "  ".join(
                    f"{b.force:g}→{b.gain_logv:+.4f}" for b in r.balayage)
                lines.append(f"     sensibilité à la force du prior : {detail}")
            if math.isfinite(r.n_obs_requis) and r.n_obs_requis > r.n_obs:
                lines.append(
                    f"     pour conclure à 80 % de puissance : "
                    f"≈ {r.n_obs_requis:,.0f} occasions "
                    f"(×{r.n_obs_requis / r.n_obs:.1f} le corpus actuel)"
                    .replace(",", " "))
            lines.append("")

        gagnantes = self.stats_significatives
        lines += [
            "  ── VERDICT ──",
            (f"  Le modèle appris bat le prior sur : {', '.join(gagnantes)}."
             if gagnantes else
             "  Sur AUCUNE statistique le modèle appris ne bat le prior de "
             "population de façon significative."),
            "  Limites : seules les p-values de permutation sont valides ici "
            "(les occasions",
            "  d'un même joueur ne sont pas indépendantes) ; la sensibilité à "
            "la force du",
            "  prior est affichée pour que rien ne soit choisi après coup ; un "
            "gain de",
            "  log-vraisemblance n'est PAS une preuve de gain en bb/100 — cela "
            "reste à mesurer.",
            "══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════


def _courbe(
    k: npt.NDArray[np.int64], d_logv: F64, d_brier: F64
) -> tuple[PointCourbe, ...]:
    """Gain moyen par tranche de « nombre de mains déjà vues du joueur »."""
    pts: list[PointCourbe] = []
    for k_min, k_max in _TRANCHES:
        m = (k >= k_min) & (k <= k_max)
        n = int(m.sum())
        pts.append(PointCourbe(
            k_min=k_min, k_max=k_max, n_obs=n,
            gain_logv=float(d_logv[m].mean()) if n else 0.0,
            gain_brier=float(d_brier[m].mean()) if n else 0.0,
        ))
    return tuple(pts)


def _n_obs_requis(gain: float, erreur_type: float, n_obs: int) -> float:
    """Occasions nécessaires pour détecter le gain observé à 80 % de puissance.

    L'erreur-type décroît en :math:`1/\\sqrt{n}` : il faut donc
    :math:`n' = n\\,\\bigl(\\sigma\\,(z_\\alpha + z_\\beta) / |gain|\\bigr)^2`.
    L'extrapolation ne vaut que si l'effet observé est le vrai effet — c'est
    un ordre de grandeur, pas une promesse.
    """
    if gain == 0.0 or erreur_type <= 0.0:
        return float("inf")
    cible = abs(gain) / (_Z_ALPHA + _Z_BETA)
    return n_obs * (erreur_type / cible) ** 2


def evaluer_inference(
    hands: Sequence[ParsedHand],
    *,
    stats: Sequence[str] = STATS_EVALUEES,
    discount: float = 1.0,
    force_prior: float | None = None,
    forces_balayage: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 30.0),
    exclure_heros: bool = True,
    baseline_prequentielle: bool = False,
    n_permutations: int = 499,
    n_bootstrap: int = 2000,
    seed: int = 20260808,
) -> RapportInference:
    """Compare le modèle adverse appris au prior de population, sans fuite.

    Parameters
    ----------
    hands : Sequence[ParsedHand]
        Mains parsées, dans n'importe quel ordre : elles sont retriées par
        :func:`_cle_chronologique` (estampille ``<startdate>``, puis numéro de
        main). Sans estampille NI numéro numérique, l'ordre d'entrée fait foi.
    stats : Sequence[str]
        Statistiques binaires à évaluer, parmi celles que produit
        ``ParsedHand.stat_observations``.
    discount : float
        Facteur d'actualisation des trackers F1. ``1.0`` = adversaire supposé
        stationnaire ; c'est le défaut ici parce que sur des séquences de
        quelques dizaines de mains l'oubli n'a pas de quoi s'exprimer, et
        parce qu'à δ = 1 le score est exactement la vraisemblance marginale,
        donc indépendant de l'ordre intra-joueur — propriété qui rend le
        protocole vérifiable.
    force_prior : float | None
        Force du prior de population, en pseudo-observations. Par défaut la
        règle empirical-Bayes du projet (:func:`force_prior_empirique`).
    forces_balayage : Sequence[float]
        Forces supplémentaires évaluées pour montrer la sensibilité du
        résultat à cet hyperparamètre. Vide pour désactiver. Choisir a
        posteriori la force la plus favorable serait de la sélection sur le
        jeu de test : le balayage est un diagnostic, pas un réglage.
    exclure_heros : bool
        Exclut les sièges du héros. Le sujet est la modélisation du POOL ; le
        héros, sur-représenté, gonflerait mécaniquement l'effet.
    baseline_prequentielle : bool
        Si vrai, le taux de population n'utilise lui aussi que le passé — et
        le prior des trackers avec lui, sans quoi la variante avantagerait le
        modèle au lieu de le désavantager (cf. docstring du module).
    n_permutations, n_bootstrap : int
        Tailles des rééchantillonnages.
    seed : int
        Graine du générateur (résultat reproductible).

    Returns
    -------
    RapportInference
        Rapport avec ``.explain()`` en français.
    """
    if not (0.0 < discount <= 1.0):
        raise InferenceCheckError("discount doit être dans (0, 1].")
    if force_prior is not None and force_prior <= 0.0:
        raise InferenceCheckError("force_prior doit être strictement positif.")

    occasions = _collecter(hands, tuple(stats), exclure_heros)
    if not occasions:
        raise InferenceCheckError(
            "aucune occasion exploitable dans ce corpus.")

    rng = np.random.default_rng(seed)
    resultats: list[ResultatStat] = []

    for stat in stats:
        occ = occasions.get(stat)
        if occ is None:
            continue
        taux = _taux_population(occ.resultats)
        force = (force_prior if force_prior is not None
                 else force_prior_empirique(occ.resultats.size))
        force = max(force, 2.0 * JEFFREYS)   # un prior de force nulle n'existe pas

        p_base = (_baseline_prequentielle(occ.resultats)
                  if baseline_prequentielle
                  else np.full(occ.resultats.size, taux))
        p_mod, k = _predire_sequentiel(
            stat, occ.joueurs, occ.resultats, p_base, force, discount)

        logv_base = _log_vraisemblance(p_base, occ.resultats)
        logv_mod = _log_vraisemblance(p_mod, occ.resultats)
        brier_base = _brier(p_base, occ.resultats)
        brier_mod = _brier(p_mod, occ.resultats)
        d_logv = logv_mod - logv_base
        d_brier = brier_base - brier_mod
        gain = float(d_logv.mean())

        tirages = _bootstrap_joueurs(
            d_logv, occ.joueurs, occ.n_joueurs, rng, n_bootstrap)
        ic = (float(np.percentile(tirages, 2.5)),
              float(np.percentile(tirages, 97.5)))
        se = float(tirages.std(ddof=1))

        balayage = tuple(
            PointBalayage(
                force=f,
                gain_logv=float(
                    (_log_vraisemblance(
                        _predire_sequentiel(stat, occ.joueurs, occ.resultats,
                                            p_base, f, discount)[0],
                        occ.resultats) - logv_base).mean()),
                n_obs=occ.resultats.size,
            )
            for f in forces_balayage
        )

        resultats.append(ResultatStat(
            stat=stat,
            n_obs=int(occ.resultats.size),
            n_joueurs=occ.n_joueurs,
            taux_population=taux,
            force_prior=force,
            logv_baseline=float(logv_base.mean()),
            logv_modele=float(logv_mod.mean()),
            brier_baseline=float(brier_base.mean()),
            brier_modele=float(brier_mod.mean()),
            p_wilcoxon=_wilcoxon(d_logv),
            p_permutation=_p_permutation(
                occ, logv_base, p_base, force, discount, gain,
                rng, n_permutations, stratifiee=False),
            p_permutation_table=_p_permutation(
                occ, logv_base, p_base, force, discount, gain,
                rng, n_permutations, stratifiee=True),
            ic_bootstrap=ic,
            erreur_type=se,
            n_obs_requis=_n_obs_requis(gain, se, int(occ.resultats.size)),
            courbe=_courbe(k, d_logv, d_brier),
            balayage=balayage,
        ))

    tables_par_joueur: dict[str, set[str]] = {}
    for hand in hands:
        for seat in hand.seats:
            if exclure_heros and seat.is_hero:
                continue
            tables_par_joueur.setdefault(seat.player, set()).add(hand.table)

    return RapportInference(
        n_mains=len(hands),
        n_joueurs=len(tables_par_joueur),
        n_tables=len({h.table for h in hands}),
        n_joueurs_multi_tables=sum(
            1 for t in tables_par_joueur.values() if len(t) > 1),
        discount=discount,
        exclure_heros=exclure_heros,
        baseline_prequentielle=baseline_prequentielle,
        n_permutations=n_permutations,
        resultats=tuple(resultats),
    )


def evaluer_dossiers(
    chemins: Iterable[str | Path], salt: str = "", **kwargs
) -> RapportInference:
    """Charge tous les XML iPoker de plusieurs dossiers et les évalue ensemble.

    Parameters
    ----------
    chemins : Iterable[str | Path]
        Racines à explorer récursivement (``*.xml``).
    salt : str
        Sel de hachage des pseudos — doit être le même pour tous les dossiers,
        sinon un joueur croisé sur deux comptes serait compté deux fois.
    **kwargs
        Passés à :func:`evaluer_inference`.

    Returns
    -------
    RapportInference
        Avec ``n_fichiers_ignores`` renseigné, et signalé par ``explain()``
        s'il est non nul : un corpus à moitié illisible produirait sinon un
        rapport d'apparence normale, aux ``n_obs`` silencieusement amputés —
        or ces effectifs sont ce qui décide de la significativité.
    """
    hands: list[ParsedHand] = []
    ignores = 0
    for racine in chemins:
        for f in sorted(Path(racine).rglob("*.xml")):
            try:
                lot = parse_file(f, salt)
            except Exception:
                ignores += 1
                continue
            # Un fichier au XML valide mais sans main exploitable ressort vide
            # sans lever : le compter aussi, sinon la moitié des rejets
            # échappe au compteur.
            if not lot:
                ignores += 1
            hands.extend(lot)
    if not hands:
        raise InferenceCheckError("aucune main lisible dans ces dossiers.")
    return replace(evaluer_inference(hands, **kwargs),
                   n_fichiers_ignores=ignores)
