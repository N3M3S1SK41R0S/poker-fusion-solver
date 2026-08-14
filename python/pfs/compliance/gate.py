"""
Poker Fusion Solver — ComplianceGate : autorise le mode LIVE uniquement sur
des tables sans enjeu monétaire réel.

Statut (audit du 14 août 2026) — garde-fou NON branché
------------------------------------------------------
Importé par aucun code applicatif : seul ``tests/test_compliance_gate.py``
l'exerce. Pourquoi : il n'y a aujourd'hui RIEN à garder — la ligne éthique
du projet est tenue en amont, par l'absence totale de conseil pendant la
main (la calibration live v4.4 lit l'écran et n'affiche rien ; les routes
``live/*`` sont en lecture seule). Le gate devient OBLIGATOIRE avant toute
fonctionnalité qui afficherait quoi que ce soit en cours de main, même sur
argent fictif : c'est la condition de son branchement, pas une option.
Ce qui existe déjà : signaux fail-closed (fenêtre armée, titre, badge play
money, glyphe devise, indicateur tournoi), ``ComplianceGate.evaluate``,
verrouillage — testés. Accroche naturelle si on branche :
``pfs/app/server.py`` en tête des routes ``live/*`` (aucun rendu sans
``GateDecision.mode == Mode.LIVE``). Règle du projet : aucun module
fusionné sans route + UI + test de bout en bout.

═══════════════════════════════════════════════════════════════════════════
LE PIÈGE QUE CE MODULE EXISTE POUR ÉVITER
═══════════════════════════════════════════════════════════════════════════

Le critère intuitif — « mes jetons s'affichent en BB / sans symbole € donc
c'est de l'argent fictif » — est **faux et dangereux**. Trois contre-exemples,
tous fréquents :

1. **Tout tournoi en argent réel affiche des jetons, jamais une devise.**
   Un MTT à 50 € de buy-in montre « 12 450 », sans symbole. Le critère naïf
   déverrouillerait le mode LIVE sur *tous* tes MTT et SNG payants.

2. **Les clients cash proposent un affichage en big blinds.** Winamax,
   PokerStars et la plupart des clients ont une option « afficher les stacks
   en BB ». Une table NL50 réelle affiche alors « 100 BB » sans symbole.

3. **Les freerolls** n'ont pas de buy-in mais une dotation réelle : les
   adversaires jouent bien pour de l'argent.

D'où le principe de ce module :

    ⚠️  LE MODE LIVE EXIGE UNE PREUVE POSITIVE D'ARGENT FICTIF.
        L'ABSENCE DE SYMBOLE MONÉTAIRE N'EST PAS UNE PREUVE.

Le gate est **fail-closed** : tout doute, toute contradiction, toute panne de
signal ⇒ retour immédiat en mode REVIEW (aucun affichage pendant la main).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol, Sequence

__all__ = [
    "StakeKind",
    "Mode",
    "Verdict",
    "SignalReading",
    "Signal",
    "UserArmedWindowSignal",
    "WindowTitleSignal",
    "PlayMoneyBadgeSignal",
    "CurrencyGlyphSignal",
    "TournamentIndicatorSignal",
    "TableObservation",
    "GateDecision",
    "ComplianceGate",
    "GateLockedError",
]


class StakeKind(Enum):
    """Nature de l'enjeu à la table."""

    PLAY_MONEY = "play_money"
    """Jetons fictifs sans valeur de retrait. Aucun tiers lésé."""

    CASH_REAL = "cash_real"
    TOURNAMENT_REAL = "tournament_real"
    """⚠️ Affiche des JETONS, jamais une devise. Piège n°1."""

    FREEROLL = "freeroll"
    """Pas de buy-in, mais dotation réelle ⇒ traité comme argent réel."""

    UNKNOWN = "unknown"


class Mode(Enum):
    LIVE = "live"
    """Assistance complète pendant la main."""

    REVIEW = "review"
    """Perception active, mais aucun affichage avant la fin de la main."""


class Verdict(Enum):
    SUPPORTS_PLAY = "supports_play"
    CONTRADICTS_PLAY = "contradicts_play"
    INCONCLUSIVE = "inconclusive"


class Strength(Enum):
    STRONG = "strong"
    WEAK = "weak"


@dataclass(frozen=True, slots=True)
class SignalReading:
    """Résultat d'un signal, toujours traçable jusqu'à sa source."""

    source: str
    verdict: Verdict
    strength: Strength
    detail: str = ""


@dataclass(frozen=True, slots=True)
class TableObservation:
    """Tout ce que la perception sait de la table, à un instant donné."""

    hwnd: int
    window_title: str
    stack_texts: tuple[str, ...] = ()
    """Textes bruts lus dans les ROI de stack, AVANT normalisation."""

    pot_text: str = ""
    play_money_badge_detected: bool | None = None
    """Sprite « argent fictif » du client. None = ROI non calibrée."""

    tournament_ui_detected: bool | None = None
    """Minuteur de niveau, ID de tournoi, place restante, prize pool."""


class Signal(Protocol):
    def read(self, obs: TableObservation) -> SignalReading: ...


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAUX
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class UserArmedWindowSignal:
    """Armement manuel : Pierre déclare explicitement une fenêtre comme fictive.

    C'est le signal le plus fort, parce qu'il est le seul qui ne peut pas être
    trompé par un changement d'affichage du client. Il est **obligatoire** :
    aucune détection automatique seule ne déverrouille le mode LIVE.

    L'armement est lié à un HWND précis et expire à la fermeture de la fenêtre.
    """

    armed_hwnds: set[int] = field(default_factory=set)

    def arm(self, hwnd: int) -> None:
        self.armed_hwnds.add(hwnd)

    def disarm(self, hwnd: int) -> None:
        self.armed_hwnds.discard(hwnd)

    def disarm_all(self) -> None:
        self.armed_hwnds.clear()

    def read(self, obs: TableObservation) -> SignalReading:
        if obs.hwnd in self.armed_hwnds:
            return SignalReading(
                "user_armed", Verdict.SUPPORTS_PLAY, Strength.STRONG,
                f"fenêtre {obs.hwnd} armée manuellement",
            )
        return SignalReading(
            "user_armed", Verdict.INCONCLUSIVE, Strength.STRONG,
            "fenêtre non armée",
        )


# Marqueurs de table fictive dans le titre de fenêtre (multi-clients, multi-langues).
_PLAY_TITLE = re.compile(
    r"\b(play\s*money|argent\s*fictif|jetons?\s*fictifs?|fun\s*(money|chips)|"
    r"spielgeld|dinero\s*ficticio|denaro\s*virtuale|entra[îi]nement|practice)\b",
    re.IGNORECASE,
)

# Marqueurs d'argent réel dans le titre — devise, buy-in, format de tournoi.
_REAL_TITLE = re.compile(
    r"(€|\$|£|USD|EUR|GBP|\bcash\b|\bnl\d+\b|\bplo\d+\b|"
    r"\btournoi\b|\btournament\b|\bmtt\b|\bsit\s*&?\s*go\b|\bsng\b|\bspin\b|"
    r"\bfreeroll\b|\bsatellite\b|\bqualif)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class WindowTitleSignal:
    """Le titre de fenêtre — lisible de façon fiable via UI Automation.

    Contrairement aux pixels, le titre n'est pas affecté par le thème du
    client, le DPI ou l'option d'affichage en big blinds.
    """

    def read(self, obs: TableObservation) -> SignalReading:
        title = obs.window_title or ""
        real = _REAL_TITLE.search(title)
        if real:
            return SignalReading(
                "window_title", Verdict.CONTRADICTS_PLAY, Strength.STRONG,
                f"marqueur argent réel dans le titre : {real.group(0)!r}",
            )
        play = _PLAY_TITLE.search(title)
        if play:
            return SignalReading(
                "window_title", Verdict.SUPPORTS_PLAY, Strength.STRONG,
                f"marqueur argent fictif : {play.group(0)!r}",
            )
        return SignalReading(
            "window_title", Verdict.INCONCLUSIVE, Strength.STRONG,
            "titre sans marqueur exploitable",
        )


@dataclass(slots=True)
class PlayMoneyBadgeSignal:
    """Sprite « argent fictif » affiché par le client, dans une ROI calibrée."""

    def read(self, obs: TableObservation) -> SignalReading:
        if obs.play_money_badge_detected is None:
            return SignalReading(
                "play_money_badge", Verdict.INCONCLUSIVE, Strength.STRONG,
                "ROI du badge non calibrée",
            )
        if obs.play_money_badge_detected:
            return SignalReading(
                "play_money_badge", Verdict.SUPPORTS_PLAY, Strength.STRONG,
                "badge argent fictif détecté",
            )
        return SignalReading(
            "play_money_badge", Verdict.INCONCLUSIVE, Strength.STRONG,
            "badge absent — non concluant (les tables réelles n'ont pas de badge inverse)",
        )


_CURRENCY = re.compile(r"[€$£¥]|\b(EUR|USD|GBP|CHF)\b", re.IGNORECASE)


@dataclass(slots=True)
class CurrencyGlyphSignal:
    """Symbole monétaire dans les stacks ou le pot.

    ⚠️ Signal **asymétrique**, et c'est tout l'intérêt :
      - présence d'un symbole  ⇒ contradiction FORTE (c'est de l'argent réel)
      - absence d'un symbole   ⇒ NON CONCLUANT (tournois, affichage en BB…)

    C'est exactement l'erreur que le critère naïf commettait.
    """

    def read(self, obs: TableObservation) -> SignalReading:
        blob = " ".join((*obs.stack_texts, obs.pot_text))
        m = _CURRENCY.search(blob)
        if m:
            return SignalReading(
                "currency_glyph", Verdict.CONTRADICTS_PLAY, Strength.STRONG,
                f"symbole monétaire lu : {m.group(0)!r}",
            )
        return SignalReading(
            "currency_glyph", Verdict.INCONCLUSIVE, Strength.WEAK,
            "aucun symbole — NE PROUVE RIEN (tournoi réel ou affichage en BB)",
        )


@dataclass(slots=True)
class TournamentIndicatorSignal:
    """Minuteur de niveau, ID de tournoi, prize pool, place restante.

    Un tournoi affiche des jetons sans devise : c'est le faux positif n°1 du
    critère naïf. Ce signal l'attrape.
    """

    def read(self, obs: TableObservation) -> SignalReading:
        if obs.tournament_ui_detected:
            return SignalReading(
                "tournament_ui", Verdict.CONTRADICTS_PLAY, Strength.STRONG,
                "UI de tournoi détectée (jetons ≠ argent fictif)",
            )
        return SignalReading(
            "tournament_ui", Verdict.INCONCLUSIVE, Strength.WEAK,
            "pas d'UI de tournoi visible",
        )


# ═══════════════════════════════════════════════════════════════════════════
# LE GATE
# ═══════════════════════════════════════════════════════════════════════════


class GateLockedError(RuntimeError):
    """Le gate a été verrouillé par un audit post-main. Ré-armement manuel requis."""


@dataclass(frozen=True, slots=True)
class GateDecision:
    mode: Mode
    stake_kind: StakeKind
    readings: tuple[SignalReading, ...]
    reason: str

    @property
    def live_allowed(self) -> bool:
        return self.mode is Mode.LIVE

    def explain(self) -> str:
        lines = [f"Mode : {self.mode.value.upper()}  ({self.reason})"]
        for r in self.readings:
            mark = {"supports_play": "✓", "contradicts_play": "✗", "inconclusive": "·"}[
                r.verdict.value
            ]
            lines.append(f"  {mark} {r.source:<20} {r.detail}")
        return "\n".join(lines)


class ComplianceGate:
    """Décide, à chaque frame, si le mode LIVE est autorisé.

    Règle de décision — **preuve positive, fail-closed** :

    1. Si le gate est verrouillé par un audit ⇒ REVIEW.
    2. Si **un seul** signal contredit l'argent fictif ⇒ REVIEW.
    3. Sinon, LIVE exige **l'armement manuel** ET **au moins une corroboration
       automatique forte** (titre de fenêtre ou badge client).
    4. Tout le reste ⇒ REVIEW.

    Aucune combinaison de signaux faibles ne peut déverrouiller LIVE.
    """

    __slots__ = ("_signals", "_user_armed", "_locked", "_lock_reason", "_last")

    def __init__(self, signals: Sequence[Signal] | None = None) -> None:
        self._user_armed = UserArmedWindowSignal()
        self._signals: list[Signal] = list(
            signals
            if signals is not None
            else [
                self._user_armed,
                WindowTitleSignal(),
                PlayMoneyBadgeSignal(),
                CurrencyGlyphSignal(),
                TournamentIndicatorSignal(),
            ]
        )
        self._locked = False
        self._lock_reason = ""
        self._last: GateDecision | None = None

    # ── armement manuel ──────────────────────────────────────────────────
    def arm_window(self, hwnd: int) -> None:
        """Pierre déclare cette fenêtre comme table en argent fictif."""
        if self._locked:
            raise GateLockedError(self._lock_reason)
        self._user_armed.arm(hwnd)

    def disarm_window(self, hwnd: int) -> None:
        self._user_armed.disarm(hwnd)

    # ── verrouillage par audit ───────────────────────────────────────────
    @property
    def is_locked(self) -> bool:
        return self._locked

    def lock(self, reason: str) -> None:
        self._locked = True
        self._lock_reason = reason
        self._user_armed.disarm_all()

    def unlock(self) -> None:
        """Ré-armement manuel explicite après incident."""
        self._locked = False
        self._lock_reason = ""

    # ── décision ─────────────────────────────────────────────────────────
    def evaluate(self, obs: TableObservation) -> GateDecision:
        if self._locked:
            d = GateDecision(
                Mode.REVIEW, StakeKind.UNKNOWN, (),
                f"gate verrouillé : {self._lock_reason}",
            )
            self._last = d
            return d

        readings = tuple(s.read(obs) for s in self._signals)

        contradictions = [r for r in readings if r.verdict is Verdict.CONTRADICTS_PLAY]
        if contradictions:
            d = GateDecision(
                Mode.REVIEW,
                self._infer_real_kind(contradictions),
                readings,
                "contradiction détectée : " + " ; ".join(r.detail for r in contradictions),
            )
            self._last = d
            return d

        armed = any(
            r.source == "user_armed" and r.verdict is Verdict.SUPPORTS_PLAY
            for r in readings
        )
        corroboration = [
            r
            for r in readings
            if r.source != "user_armed"
            and r.verdict is Verdict.SUPPORTS_PLAY
            and r.strength is Strength.STRONG
        ]

        if armed and corroboration:
            d = GateDecision(
                Mode.LIVE, StakeKind.PLAY_MONEY, readings,
                "armement manuel + corroboration : "
                + ", ".join(r.source for r in corroboration),
            )
        elif armed:
            d = GateDecision(
                Mode.REVIEW, StakeKind.UNKNOWN, readings,
                "fenêtre armée mais AUCUNE corroboration automatique — "
                "preuve positive insuffisante",
            )
        else:
            d = GateDecision(
                Mode.REVIEW, StakeKind.UNKNOWN, readings,
                "fenêtre non armée manuellement",
            )

        self._last = d
        return d

    @staticmethod
    def _infer_real_kind(contradictions: Iterable[SignalReading]) -> StakeKind:
        srcs = {r.source for r in contradictions}
        if "tournament_ui" in srcs:
            return StakeKind.TOURNAMENT_REAL
        if "currency_glyph" in srcs:
            return StakeKind.CASH_REAL
        return StakeKind.UNKNOWN

    # ── audit post-main (filet de sécurité) ──────────────────────────────
    def audit_hand(self, *, assisted_live: bool, hh_is_real_money: bool) -> None:
        """Vérifie a posteriori, sur le hand-history, qu'aucune main assistée
        n'était en argent réel.

        Le HH est la vérité terrain : si une main jouée en mode LIVE s'avère
        avoir été en argent réel, le gate se verrouille **définitivement**
        jusqu'à ré-armement manuel explicite. On échoue bruyamment, jamais en
        silence.
        """
        if assisted_live and hh_is_real_money:
            self.lock(
                "INCIDENT : une main assistée en mode LIVE s'est avérée en "
                "argent réel au hand-history. Détection à revoir avant ré-armement."
            )
