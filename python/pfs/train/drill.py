"""
F12 — Mode TRAIN : répétition espacée, drills et détection de leaks.

Sources
-------
- Ebbinghaus (1885), courbe de l'oubli
- Wozniak, P. (1990), algorithme **SM-2** (moteur d'Anki, 30 ans de validation)
- Kahneman & Tversky (1979), *Prospect Theory* — aversion aux pertes λ ≈ 2,25
- Sweller (1988), théorie de la charge cognitive

⚠️ Ce que ce module **ne fait pas** : inventer des jauges. Les indicateurs
« fatigue », « focus », « tilt » du mock-up v1 doivent être **dérivés de
mesures** (dérive du temps de décision, taux d'erreur sur les spots maîtrisés,
écart-type des sizings) — une jauge inventée qui dit « arrête-toi » sans base
est du théâtre. La littérature sur l'ego depletion est par ailleurs en crise
de réplication : on ne s'appuie que sur des observables.

C'est le mode à plus haute EV réelle du projet : le gain vient de ce que tu
internalises, pas de ce que l'écran t'affiche.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

from pfs.core.range_model import (
    GROUP_COMBO_COUNT,
    N_GROUPS,
    Range,
    group_name,
    parse_range,
    GTO_PRESETS,
)

__all__ = [
    "Grade",
    "CardState",
    "SpacedRepetition",
    "DrillItem",
    "DrillAnswer",
    "DrillSession",
    "Leak",
    "LeakFinder",
    "CognitiveState",
    "session_monitor",
]


class Grade(int, Enum):
    """Qualité de rappel SM-2, de 0 (raté total) à 5 (parfait)."""

    BLACKOUT = 0
    WRONG = 1
    WRONG_EASY = 2
    HARD = 3
    GOOD = 4
    PERFECT = 5


@dataclass(slots=True)
class CardState:
    """État de révision d'un item (un groupe de mains dans un spot donné)."""

    key: str
    interval_days: float = 0.0
    ease: float = 2.5
    repetitions: int = 0
    due_at: float = 0.0
    """Horodatage logique (jours écoulés depuis le début), pas une date système."""
    history: list[int] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.repetitions == 0

    @property
    def accuracy(self) -> float:
        if not self.history:
            return 0.0
        return sum(1 for g in self.history if g >= 3) / len(self.history)


class SpacedRepetition:
    """SM-2 (Wozniak, 1990), adapté au poker.

    Les combos mal joués reviennent immédiatement ; les combos maîtrisés sont
    espacés exponentiellement. C'est ce qui permet de couvrir 5 000 décisions
    sans les réviser toutes.

    Le temps est **logique** (jours écoulés depuis le début de l'entraînement),
    pas système — ce qui rend les sessions rejouables et testables.
    """

    __slots__ = ("cards", "_now")

    def __init__(self) -> None:
        self.cards: dict[str, CardState] = {}
        self._now: float = 0.0

    @property
    def now(self) -> float:
        return self._now

    def advance(self, days: float) -> None:
        if days < 0:
            raise ValueError("days doit être >= 0.")
        self._now += days

    def card(self, key: str) -> CardState:
        if key not in self.cards:
            self.cards[key] = CardState(key=key, due_at=self._now)
        return self.cards[key]

    def review(self, key: str, grade: Grade | int) -> CardState:
        r"""Met à jour un item selon SM-2.

        .. math::
            EF \leftarrow \max\!\big(1.3,\;
              EF + 0.1 - (5-q)(0.08 + 0.02(5-q))\big)

        Intervalles : 1 jour, puis 6 jours, puis :math:`I_{n} = I_{n-1}\cdot EF`.
        Un échec (q < 3) remet l'intervalle à zéro **sans** réinitialiser
        l'ease factor — c'est ce qui rend la courbe stable.
        """
        q = int(grade)
        if not (0 <= q <= 5):
            raise ValueError("grade doit être dans [0, 5].")
        c = self.card(key)
        c.history.append(q)

        c.ease = max(1.3, c.ease + 0.1 - (5 - q) * (0.08 + 0.02 * (5 - q)))

        if q < 3:
            c.repetitions = 0
            c.interval_days = 0.0
            c.due_at = self._now
        else:
            c.repetitions += 1
            if c.repetitions == 1:
                c.interval_days = 1.0
            elif c.repetitions == 2:
                c.interval_days = 6.0
            else:
                c.interval_days *= c.ease
            c.due_at = self._now + c.interval_days
        return c

    def due(self, limit: int | None = None) -> list[CardState]:
        """Items à réviser maintenant, les plus en retard d'abord."""
        d = [c for c in self.cards.values() if c.due_at <= self._now]
        d.sort(key=lambda c: (c.due_at, -len(c.history)))
        return d[:limit] if limit else d

    def workload(self, horizon_days: int = 30) -> list[int]:
        """Charge de révision prévue, jour par jour — évite les pics."""
        out = [0] * horizon_days
        for c in self.cards.values():
            offset = int(c.due_at - self._now)
            if 0 <= offset < horizon_days:
                out[offset] += 1
        return out

    def mastery(self) -> float:
        """Part des items dont l'intervalle dépasse 21 jours (seuil Anki)."""
        if not self.cards:
            return 0.0
        return sum(1 for c in self.cards.values() if c.interval_days >= 21.0) / len(self.cards)


# ═══════════════════════════════════════════════════════════════════════════
# DRILLS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class DrillItem:
    position: str
    group: str
    correct_frequency: float
    combos: int

    @property
    def key(self) -> str:
        return f"{self.position}|{self.group}"

    @property
    def is_mixed(self) -> bool:
        return 0.05 < self.correct_frequency < 0.95


@dataclass(frozen=True, slots=True)
class DrillAnswer:
    item: DrillItem
    given: float
    error: float
    grade: Grade
    ev_loss_bb: float
    seconds: float


class DrillSession:
    """Générateur de drills piloté par la répétition espacée.

    Trois modes de difficulté, calqués sur ce qui marche chez GTO Wizard :
      - ``easy``   : jouer / ne pas jouer (binaire)
      - ``medium`` : fréquence au quart près (0 / 25 / 50 / 75 / 100 %)
      - ``hard``   : fréquence exacte, tolérance 10 points

    Le mode ``only_close`` ne présente que les spots mixtes — ceux où l'erreur
    coûte le plus cher, et que les charts statiques ne transmettent pas.
    """

    __slots__ = ("srs", "items", "answers", "_rng", "difficulty", "only_close")

    TOLERANCE = {"easy": 0.50, "medium": 0.20, "hard": 0.10}

    def __init__(
        self,
        positions: Sequence[str] = ("UTG", "MP", "CO", "BTN", "SB"),
        difficulty: str = "medium",
        only_close: bool = False,
        seed: int | None = 0,
    ) -> None:
        if difficulty not in self.TOLERANCE:
            raise ValueError(f"difficulté inconnue : {difficulty!r}")
        self.srs = SpacedRepetition()
        self.difficulty = difficulty
        self.only_close = only_close
        self._rng = np.random.default_rng(seed)
        self.answers: list[DrillAnswer] = []

        self.items: list[DrillItem] = []
        for pos in positions:
            freqs = parse_range(GTO_PRESETS[pos]).to_groups()
            for g in range(N_GROUPS):
                item = DrillItem(pos, group_name(g), float(freqs[g]), int(GROUP_COMBO_COUNT[g]))
                if only_close and not item.is_mixed:
                    continue
                self.items.append(item)

        if not self.items:
            raise ValueError(
                "aucun spot ne correspond aux critères — les ranges choisies "
                "n'ont pas de fréquence mixte."
            )

    # ── sélection ────────────────────────────────────────────────────────
    def next_item(self) -> DrillItem:
        """Priorité aux items dus ; sinon un item neuf, pondéré par les combos."""
        due = self.srs.due(limit=50)
        if due:
            key = due[0].key
            for it in self.items:
                if it.key == key:
                    return it
        fresh = [it for it in self.items if it.key not in self.srs.cards]
        pool = fresh or self.items
        w = np.array([it.combos for it in pool], dtype=float)
        return pool[int(self._rng.choice(len(pool), p=w / w.sum()))]

    # ── notation ─────────────────────────────────────────────────────────
    def answer(self, item: DrillItem, given: float, seconds: float = 0.0) -> DrillAnswer:
        r"""Note une réponse et met à jour la répétition espacée.

        La note SM-2 est dérivée de l'erreur **et** du temps de réponse : une
        bonne réponse lente signale une récupération laborieuse, donc un
        intervalle plus court. C'est l'esprit du SM-2 d'origine, où la note
        encode la difficulté ressentie.

        La perte d'EV est approchée par
        :math:`\text{combos} \times \text{erreur}^2 \times 0{,}5` bb — un proxy
        quadratique, cohérent avec la courbure de l'EV autour de l'optimum.
        """
        if not (0.0 <= given <= 1.0):
            raise ValueError("given doit être dans [0, 1].")
        err = abs(given - item.correct_frequency)
        tol = self.TOLERANCE[self.difficulty]

        if err <= tol * 0.25:
            grade = Grade.PERFECT
        elif err <= tol * 0.5:
            grade = Grade.GOOD
        elif err <= tol:
            grade = Grade.HARD
        elif err <= tol * 2:
            grade = Grade.WRONG_EASY
        elif err <= tol * 3:
            grade = Grade.WRONG
        else:
            grade = Grade.BLACKOUT

        if grade >= Grade.HARD and seconds > 8.0:
            grade = Grade(max(int(Grade.HARD), int(grade) - 1))

        self.srs.review(item.key, grade)
        ans = DrillAnswer(
            item=item,
            given=given,
            error=err,
            grade=grade,
            ev_loss_bb=item.combos * err**2 * 0.5,
            seconds=seconds,
        )
        self.answers.append(ans)
        return ans

    # ── score ────────────────────────────────────────────────────────────
    def score(self) -> dict[str, float]:
        """Score de précision global et par découpage — l'équivalent GTOW Score."""
        if not self.answers:
            return {"accuracy": 0.0, "n": 0}
        acc = 1.0 - float(np.mean([a.error for a in self.answers]))
        mixed = [a for a in self.answers if a.item.is_mixed]
        pure = [a for a in self.answers if not a.item.is_mixed]
        out = {
            "accuracy": acc,
            "n": float(len(self.answers)),
            "ev_loss_bb": float(sum(a.ev_loss_bb for a in self.answers)),
            "mastery": self.srs.mastery(),
        }
        if mixed:
            out["accuracy_mixed"] = 1.0 - float(np.mean([a.error for a in mixed]))
        if pure:
            out["accuracy_pure"] = 1.0 - float(np.mean([a.error for a in pure]))
        for pos in sorted({a.item.position for a in self.answers}):
            sub = [a for a in self.answers if a.item.position == pos]
            out[f"accuracy_{pos}"] = 1.0 - float(np.mean([a.error for a in sub]))
        return out


# ═══════════════════════════════════════════════════════════════════════════
# DÉTECTION DE LEAKS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Leak:
    label: str
    magnitude: float
    """Écart moyen, en points de fréquence."""
    ev_loss_bb: float
    n: int
    direction: str
    """« trop large » ou « trop serré »."""
    recommendation: str

    def __str__(self) -> str:
        return (
            f"{self.label:<28} {self.direction:<12} "
            f"{self.magnitude * 100:+5.1f} pts · {self.ev_loss_bb:6.1f} bb perdus "
            f"(n={self.n})\n    → {self.recommendation}"
        )


class LeakFinder:
    """Agrège les erreurs de drill en leaks nommés, triés par coût.

    Le tri est fait par **perte d'EV**, pas par fréquence d'erreur : se tromper
    souvent sur un spot rare coûte moins qu'une erreur occasionnelle sur AA.
    """

    MIN_SAMPLE = 8

    @staticmethod
    def analyse(answers: Sequence[DrillAnswer]) -> list[Leak]:
        if not answers:
            return []
        buckets: dict[str, list[DrillAnswer]] = {}

        def add(label: str, a: DrillAnswer) -> None:
            buckets.setdefault(label, []).append(a)

        for a in answers:
            it = a.item
            add(f"{it.position} — global", a)
            add("spots mixtes" if it.is_mixed else "spots purs", a)
            g = it.group
            if len(g) == 2 and g[0] == g[1]:
                add("paires", a)
            elif g.endswith("s"):
                add("mains suited", a)
            else:
                add("mains offsuit", a)

        leaks: list[Leak] = []
        for label, group in buckets.items():
            if len(group) < LeakFinder.MIN_SAMPLE:
                continue
            signed = float(np.mean([a.given - a.item.correct_frequency for a in group]))
            mag = float(np.mean([a.error for a in group]))
            if mag < 0.08:
                continue
            ev = float(sum(a.ev_loss_bb for a in group))
            direction = "trop large" if signed > 0 else "trop serré"
            if "mixtes" in label:
                reco = "Travaille d'abord les décisions pures — les mixtes viennent après."
            elif "paires" in label:
                reco = "Revois les seuils de paire par position (règle « paire ≥ X »)."
            elif "suited" in label:
                reco = "Revois l'escalier suited : carte haute ET carte basse."
            elif "offsuit" in label:
                reco = "L'offsuit est le plus sur-joué : resserre par le bas."
            else:
                reco = f"Drill ciblé sur {label.split(' —')[0]}, mode « only close »."
            leaks.append(Leak(label, mag, ev, len(group), direction, reco))

        leaks.sort(key=lambda x: -x.ev_loss_bb)
        return leaks


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAT COGNITIF — MESURÉ, PAS INVENTÉ
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CognitiveState:
    """Indicateurs **dérivés d'observables**, avec leur incertitude.

    Chaque champ est ``None`` tant que l'échantillon est insuffisant : mieux
    vaut n'afficher aucune jauge qu'une jauge inventée.
    """

    decision_time_drift: float | None
    """Pente du temps de décision (s par décision). > 0 ⇒ ralentissement."""
    error_rate_drift: float | None
    """Pente du taux d'erreur sur les spots MAÎTRISÉS — le signal le plus propre."""
    sizing_dispersion: float | None
    """Écart-type des fréquences données. Une hausse signale l'incohérence."""
    loss_aversion_lambda: float | None
    """λ de Kahneman-Tversky mesuré sur tes propres décisions (référence ≈ 2,25)."""
    n_observations: int
    advice: str

    def explain(self) -> str:
        def fmt(v: float | None, unit: str) -> str:
            return "—" if v is None else f"{v:+.3f} {unit}"
        return (
            f"État cognitif (n={self.n_observations})\n"
            f"  dérive du temps de décision : {fmt(self.decision_time_drift, 's/décision')}\n"
            f"  dérive du taux d'erreur     : {fmt(self.error_rate_drift, 'pts/décision')}\n"
            f"  dispersion des réponses     : {fmt(self.sizing_dispersion, '')}\n"
            f"  aversion aux pertes λ       : {fmt(self.loss_aversion_lambda, '')}\n"
            f"  → {self.advice}"
        )


def _slope(y: Sequence[float]) -> float | None:
    n = len(y)
    if n < 12:
        return None
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, np.asarray(y, dtype=float), 1)[0])


def session_monitor(
    answers: Sequence[DrillAnswer],
    fold_calls: Sequence[tuple[float, bool]] = (),
    min_sample: int = 12,
) -> CognitiveState:
    """Construit l'état cognitif à partir des observables de la session.

    Parameters
    ----------
    fold_calls
        Décisions de bluff-catch : ``(marge_ev_en_bb, a_appelé)``. Sert à
        estimer λ, l'aversion aux pertes : un joueur avec λ élevé sous-appelle
        les spots marginalement +EV.

    Notes
    -----
    La règle d'arrêt correcte n'est pas un stop-loss — le théorème d'arrêt de
    Doob dit qu'aucune règle basée sur le résultat ne change l'espérance d'une
    martingale. La seule justification valide est que **ton winrate n'est pas
    constant** : il chute avec la fatigue. D'où un arrêt conditionné à l'état
    cognitif mesuré, jamais au résultat.
    """
    n = len(answers)
    if n < min_sample:
        return CognitiveState(None, None, None, None, n,
                              "Échantillon insuffisant — aucune jauge affichée.")

    times = [a.seconds for a in answers]
    t_drift = _slope(times) if any(times) else None

    mastered = [a for a in answers if not a.item.is_mixed]
    e_drift = _slope([a.error for a in mastered]) if len(mastered) >= min_sample else None

    dispersion = float(statistics.pstdev([a.given for a in answers])) if n >= min_sample else None

    lam = None
    marginal = [(m, c) for m, c in fold_calls if 0.0 < m < 2.0]
    if len(marginal) >= 10:
        call_rate = sum(1 for _, c in marginal if c) / len(marginal)
        # λ = 1 si le joueur appelle tous les spots +EV ; croît quand il sous-appelle.
        lam = float(1.0 / max(call_rate, 0.05))

    reasons = []
    if t_drift is not None and t_drift > 0.05:
        reasons.append("le temps de décision s'allonge")
    if e_drift is not None and e_drift > 0.002:
        reasons.append("le taux d'erreur monte sur des spots maîtrisés")
    if dispersion is not None and dispersion > 0.38:
        reasons.append("les réponses deviennent incohérentes")
    if lam is not None and lam > 2.5:
        reasons.append(f"aversion aux pertes élevée (λ={lam:.2f})")

    if not reasons:
        advice = "Aucun signal de dégradation — la session peut continuer."
    elif len(reasons) == 1:
        advice = f"Signal isolé : {reasons[0]}. Surveiller, pas encore arrêter."
    else:
        advice = ("Plusieurs signaux concordants (" + " ; ".join(reasons)
                  + "). Le winrate espéré est probablement dégradé : faire une pause.")

    return CognitiveState(t_drift, e_drift, dispersion, lam, n, advice)
