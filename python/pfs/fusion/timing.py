r"""
L2 — Tells temporels : le canal que personne n'exploite en direct.

La perception (Phase 1) horodate chaque action à la frame près. Ce module
transforme ces latences en trois informations exploitables :

1. **Surprise temporelle** — z-score du log-temps d'une décision contre la
   ligne de base du joueur pour cette classe d'action. Les temps de réaction
   humains sont log-normaux (Luce, 1986, *Response Times* ; Ulrich & Miller,
   1993) : on travaille donc sur log(t), où Welford (1962) donne moyenne et
   variance en ligne, en O(1) par observation.
2. **Dérive** — CUSUM (Page, 1954) sur les surprises : détecte un changement
   de régime temporel (fatigue, tilt, multi-tabling qui s'ajoute) bien avant
   qu'une moyenne glissante ne bouge. S'ajoute comme covariable au HMM (F2).
3. **Tells conditionnés aux showdowns** — quand une main va à l'abattage, on
   apprend l'association (classe d'action, tercile de temps) → force réelle
   montrée. Un tell n'est RAPPORTÉ que si son intervalle de Wilson (1927)
   exclut la fréquence de base : pas de tell fabriqué sur trois mains.

Aucune sémantique universelle codée en dur (« snap = faible » est un mythe de
population) : tout est appris PAR JOUEUR, l'appareil d'incertitude décide.

Anti-modes d'emploi : un z isolé n'est pas un tell ; seule la table de
showdowns, avec son IC, autorise une lecture directionnelle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

__all__ = [
    "ActionClass", "StrengthBin", "TimingError", "TimingObservation",
    "TimingProfile", "TimingTell", "wilson_interval",
]


class TimingError(ValueError):
    pass


class ActionClass(str, Enum):
    """Classes d'action — la granularité utile sans émietter l'échantillon."""
    CHECK = "check"
    CALL = "call"
    BET_SMALL = "bet_small"      # < 50 % pot
    BET_BIG = "bet_big"          # ≥ 50 % pot
    RAISE = "raise"
    FOLD = "fold"


class StrengthBin(str, Enum):
    """Force montrée à l'abattage, en terciles d'équité river."""
    AIR = "air"                  # < 33 % contre la range affrontée
    MEDIUM = "medium"
    STRONG = "strong"            # > 66 %


_TERCILES = ("rapide", "normal", "lent")


@dataclass(frozen=True, slots=True)
class TimingObservation:
    action: ActionClass
    seconds: float
    street: str = ""             # informatif, non requis


@dataclass(frozen=True, slots=True)
class TimingTell:
    """Un tell VALIDÉ : l'IC de Wilson exclut la fréquence de base."""
    action: ActionClass
    tercile: str                 # rapide / normal / lent
    strength: StrengthBin
    rate: float                  # P(force | action, tercile)
    baseline: float              # P(force | action) toutes vitesses
    ci_low: float
    ci_high: float
    n: int

    def __str__(self) -> str:
        arrow = "↑" if self.rate > self.baseline else "↓"
        return (f"{self.action.value} {self.tercile} → {self.strength.value} "
                f"{arrow} {self.rate * 100:.0f} % [{self.ci_low * 100:.0f}–"
                f"{self.ci_high * 100:.0f}] vs base {self.baseline * 100:.0f} % "
                f"(n={self.n})")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson (1927) pour une proportion — robuste aux petits n.

    >>> lo, hi = wilson_interval(8, 10)
    >>> 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    True
    """
    if n <= 0 or k < 0 or k > n:
        raise TimingError("comptage invalide.")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


class _Welford:
    """Moyenne/variance en ligne (Welford 1962) — O(1), stable numériquement."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n >= 2 else float("nan")

    @property
    def std(self) -> float:
        v = self.var
        return math.sqrt(v) if v == v and v > 0 else float("nan")


class TimingProfile:
    """Profil temporel d'UN joueur : lignes de base, surprise, dérive, tells.

    Paramètres
    ----------
    min_n
        Observations minimales par classe avant d'émettre un z-score
        (en dessous : la ligne de base globale sert de secours si elle-même
        est mûre, sinon z = nan).
    min_log_std
        Plancher de l'écart-type des log-temps (0.15 ≈ ±15 % de latence) :
        la variabilité humaine est irréductible ; sans plancher, une série
        localement constante ferait exploser les z.
    cusum_k, cusum_h
        Réglage CUSUM standard (Page 1954) : k = demi-décalage à détecter
        (en unités de σ), h = seuil d'alarme. (0.5, 5) → ARL₀ ≈ 470
        observations sous H₀ (deux côtés), détection d'un décalage d'1σ en
        moins de 10 observations.

    Examples
    --------
    >>> p = TimingProfile(seed_doc=True)
    >>> for _ in range(30):
    ...     _ = p.observe(TimingObservation(ActionClass.BET_BIG, 4.0))
    >>> abs(p.surprise(ActionClass.BET_BIG, 4.0)) < 0.7   # temps habituel
    True
    >>> p.surprise(ActionClass.BET_BIG, 0.4) < -2.0       # snap inhabituel
    True
    """

    def __init__(self, min_n: int = 8, min_log_std: float = 0.15,
                 cusum_k: float = 0.5, cusum_h: float = 5.0,
                 seed_doc: bool = False) -> None:
        if min_n < 2:
            raise TimingError("min_n >= 2 requis.")
        if min_log_std <= 0:
            raise TimingError("min_log_std > 0 requis.")
        self.min_n = min_n
        self.min_log_std = min_log_std
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self._per_action: dict[ActionClass, _Welford] = {
            a: _Welford() for a in ActionClass}
        self._global = _Welford()
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._drift_alarms = 0
        self._n_obs = 0
        # table showdown : (action, tercile) → comptes par force
        self._shows: dict[tuple[ActionClass, str], dict[StrengthBin, int]] = {}
        self._shows_base: dict[ActionClass, dict[StrengthBin, int]] = {}
        if seed_doc:                      # bruit léger pour les doctests
            for i in range(4):
                self.observe(TimingObservation(
                    ActionClass.BET_BIG, 3.5 + 0.3 * ((i % 3) - 1)))

    # ── observation & surprise ────────────────────────────────────────────

    def observe(self, obs: TimingObservation) -> float:
        """Ingère une latence, retourne la surprise z AVANT mise à jour.

        L'ordre (z d'abord, mise à jour ensuite) évite qu'une observation ne
        se juge elle-même — le classique data snooping en ligne.
        """
        if obs.seconds <= 0:
            raise TimingError("latence <= 0.")
        z = self.surprise(obs.action, obs.seconds)
        x = math.log(obs.seconds)
        self._per_action[obs.action].push(x)
        self._global.push(x)
        self._n_obs += 1
        if z == z:                        # CUSUM sur surprises valides
            zc = max(min(z, 6.0), -6.0)
            self._cusum_pos = max(0.0, self._cusum_pos + zc - self.cusum_k)
            self._cusum_neg = max(0.0, self._cusum_neg - zc - self.cusum_k)
            if self._cusum_pos > self.cusum_h or self._cusum_neg > self.cusum_h:
                self._drift_alarms += 1
                self._cusum_pos = self._cusum_neg = 0.0
        return z

    def surprise(self, action: ActionClass, seconds: float) -> float:
        """z-score du log-temps contre la ligne de base (classe, sinon globale)."""
        if seconds <= 0:
            raise TimingError("latence <= 0.")
        x = math.log(seconds)
        for w in (self._per_action[action], self._global):
            if w.n >= self.min_n:
                sd = w.std if w.std == w.std else 0.0
                return (x - w.mean) / max(sd, self.min_log_std)
        return float("nan")

    # ── dérive ────────────────────────────────────────────────────────────

    @property
    def drift_alarms(self) -> int:
        """Nombre d'alarmes CUSUM — covariable pour le HMM (F2)."""
        return self._drift_alarms

    @property
    def drifting(self) -> bool:
        return (self._cusum_pos > self.cusum_h * 0.5
                or self._cusum_neg > self.cusum_h * 0.5)

    # ── tells conditionnés aux showdowns ──────────────────────────────────

    def record_showdown(self, action: ActionClass, seconds: float,
                        strength: StrengthBin) -> None:
        """Associe la latence d'une action montrée à la force réelle."""
        if seconds <= 0:
            raise TimingError("latence <= 0.")
        t = self._tercile(action, seconds)
        key = (action, t)
        self._shows.setdefault(key, {s: 0 for s in StrengthBin})[strength] += 1
        self._shows_base.setdefault(
            action, {s: 0 for s in StrengthBin})[strength] += 1

    def _tercile(self, action: ActionClass, seconds: float) -> str:
        z = self.surprise(action, seconds)
        if z != z:
            return "normal"
        if z < -0.43:                     # terciles de la N(0,1)
            return "rapide"
        if z > 0.43:
            return "lent"
        return "normal"

    def tells(self, min_n: int = 12) -> list[TimingTell]:
        """Tells dont l'IC de Wilson exclut la base — les seuls rapportables.

        Sur du bruit pur, l'IC contient la base ~95 % du temps par cellule :
        le filtre est la défense contre les tells fantômes.
        """
        out: list[TimingTell] = []
        for (action, t), counts in self._shows.items():
            n_cell = sum(counts.values())
            if n_cell < min_n:
                continue
            base_counts = self._shows_base[action]
            n_base = sum(base_counts.values())
            for s in StrengthBin:
                k = counts[s]
                base = base_counts[s] / n_base
                lo, hi = wilson_interval(k, n_cell)
                if base < lo or base > hi:
                    out.append(TimingTell(action, t, s, k / n_cell,
                                          base, lo, hi, n_cell))
        out.sort(key=lambda x: -abs(x.rate - x.baseline))
        return out

    # ── état sérialisable (pour player_notes) ─────────────────────────────

    def summary(self) -> dict:
        return {
            "n_obs": self._n_obs,
            "drift_alarms": self._drift_alarms,
            "drifting": self.drifting,
            "baselines": {
                a.value: {"n": w.n,
                          "median_s": math.exp(w.mean) if w.n else None,
                          "log_std": w.std if w.n >= 2 else None}
                for a, w in self._per_action.items() if w.n
            },
            "n_tells": len(self.tells()),
        }


def strength_bin_from_equity(equity: float) -> StrengthBin:
    """Force montrée → tercile (branchement direct sur core.equity)."""
    if not (0.0 <= equity <= 1.0):
        raise TimingError("équité hors [0,1].")
    if equity < 1 / 3:
        return StrengthBin.AIR
    if equity <= 2 / 3:
        return StrengthBin.MEDIUM
    return StrengthBin.STRONG
