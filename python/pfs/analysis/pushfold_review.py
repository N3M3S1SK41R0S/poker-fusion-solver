"""Revue shove/fold préflop — tes décisions de tapis face au Nash push/fold.

En hyperturbo short-stack (Twister, fin de SNG/MTT), la bonne stratégie
préflop se réduit à jam-ou-fold : c'est le domaine de HRC/ICMIZER. Ce
module confronte chaque décision **heads-up** du héros à l'équilibre de
Nash jam/fold du projet (``solver.pushfold``), et chiffre en big blinds le
coût de chaque écart — l'EV(jam) − EV(fold) sort directement du solveur.

Cible : la fuite mesurée par la revue de session (VPIP trop large). Un jam
trop lâche = jeter des jetons ; un fold trop serré = laisser de l'EV. Les
deux sont ici quantifiés à la main près.

Périmètre exact (honnêteté NEMESIS) :
  - **spot heads-up au moment de la décision** : soit la table est à 2, soit
    tous les autres se sont déjà couchés devant (fréquent en 3-max Twister,
    où le bouton parle en premier). Un spot encore multiway quand le héros
    parle sort du solveur HU — compté à part, jamais approximé ;
  - **héros premier à parler et dernier agresseur potentiel** (SB en HU,
    bouton en 3-max) : la décision jam-ou-fold du modèle ;
  - tapis effectif ``1 ≤ eff ≤ 25`` bb (au-delà, l'arbre limp/raise-fold
    compte : hors modèle) ;
  - **chipEV** (pas ICM) : référence standard des charts ; près de la bulle
    l'ICM resserre les jams — signalé, non appliqué ici.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from pfs.core.range_model import (
    COMBO_TO_GROUP,
    RANKS,
    SUITS,
    combo_index,
    group_name,
)
from pfs.data.hand_history import ActionType, ParsedHand, Street, parse_file
from pfs.solver.pushfold import equity_matrix_169, solve_hu_pushfold

__all__ = [
    "ShoveSpot",
    "PushFoldReview",
    "review_pushfold_hands",
    "review_pushfold_folder",
]

MAX_EFF_BB = 25.0    # au-delà : l'arbre limp/raise-fold compte, hors push/fold
_JAM_THRESHOLD = 0.5  # fréquence d'équilibre ≥ 0,5 ⇒ décision « jam »

_EQUITY = None


def _equity_matrix():
    """Matrice d'équité 169×169, chargée une fois (cache disque du solveur)."""
    global _EQUITY
    if _EQUITY is None:
        _EQUITY = equity_matrix_169()
    return _EQUITY


@lru_cache(maxsize=256)
def _solution(eff_bb_deci: int, ante_centi: int):
    """Équilibre jam/fold pour un tapis effectif (mémoïsé par 0,1 bb / 0,01 ante).

    Les clés entières (dixièmes de bb, centièmes d'ante) rendent le cache
    stable et bornent le nombre de solves distincts.
    """
    return solve_hu_pushfold(
        eff_bb_deci / 10.0,
        sb=0.5,
        bb=1.0,
        ante=ante_centi / 100.0,
        equity=_equity_matrix(),
    )


def _hand_group(cards: Sequence[str]) -> int:
    a = RANKS.index(cards[0][0]) * 4 + SUITS.index(cards[0][1])
    b = RANKS.index(cards[1][0]) * 4 + SUITS.index(cards[1][1])
    return int(COMBO_TO_GROUP[combo_index(a, b)])


@dataclass(slots=True)
class ShoveSpot:
    """Une décision jam/fold du héros au small blind, jugée par le Nash."""

    hand_id: str
    hand: str            # nom du groupe, ex. « A5s », « 72o »
    eff_bb: float
    decision: str        # "jam" | "fold" | "limp" | "raise"
    nash_jam_freq: float  # fréquence de jam à l'équilibre pour ce groupe
    ev_jam_minus_fold: float  # EV(jam) − EV(fold), bb (chipEV) ; >0 ⇒ jam
    verdict: str         # "ok" | "jam trop lâche" | "fold trop serré" | "limp"
    cost_bb: float       # bb perdues par l'écart (0 si ok)


@dataclass(slots=True)
class PushFoldReview:
    """Synthèse de la revue shove/fold."""

    spots: list[ShoveSpot] = field(default_factory=list)
    n_hu_sb: int = 0          # décisions SB heads-up examinées
    n_skipped_multiway: int = 0
    n_skipped_deep: int = 0   # tapis > MAX_EFF_BB

    @property
    def n_mistakes(self) -> int:
        return sum(1 for s in self.spots if s.verdict != "ok")

    @property
    def total_cost_bb(self) -> float:
        return sum(s.cost_bb for s in self.spots)

    @property
    def loose_jams(self) -> list[ShoveSpot]:
        return [s for s in self.spots if s.verdict == "jam trop lâche"]

    @property
    def tight_folds(self) -> list[ShoveSpot]:
        return [s for s in self.spots if s.verdict == "fold trop serré"]

    def worst(self, k: int = 10) -> list[ShoveSpot]:
        return sorted(self.spots, key=lambda s: s.cost_bb, reverse=True)[:k]

    def explain(self) -> str:
        n = len(self.spots)
        lines = [
            "══════════════════════════════════════════════════════════",
            "  REVUE SHOVE/FOLD — tes tapis préflop face au Nash (chipEV)",
            "══════════════════════════════════════════════════════════",
            f"  Décisions SB heads-up jugées : {n}"
            f"   (multiway ignorés : {self.n_skipped_multiway} ;"
            f" tapis >25bb : {self.n_skipped_deep})",
        ]
        if n:
            loose = len(self.loose_jams)
            tight = len(self.tight_folds)
            limps = sum(1 for s in self.spots if s.verdict == "limp")
            lines += [
                f"  Erreurs                     : {self.n_mistakes}/{n}",
                f"    · jams trop lâches        : {loose}   (fuite VPIP directe)",
                f"    · folds trop serrés       : {tight}",
                f"    · limps (hors push/fold)  : {limps}",
                f"  Coût total des écarts       : {self.total_cost_bb:.1f} bb",
                "",
                "  Pires écarts (à revoir en priorité) :",
            ]
            for s in self.worst(8):
                if s.verdict == "ok":
                    continue
                lines.append(
                    f"    {s.hand:>4} @ {s.eff_bb:4.1f}bb : tu {s.decision}s → "
                    f"{s.verdict} (−{s.cost_bb:.2f} bb)"
                )
        lines.append("══════════════════════════════════════════════════════════")
        return "\n".join(lines)


def _heads_up_opponent(hand: ParsedHand, hero: str) -> str | None:
    """L'unique adversaire encore en jeu quand le héros prend sa décision.

    Renvoie la clé de cet adversaire si le spot est heads-up au moment où
    le héros parle — soit la table est à deux, soit tous les joueurs avant
    lui se sont couchés (cas courant du 3-max : le bouton ouvre, les deux
    blindes restent… mais si une seule reste devant, ce n'est PAS encore
    résolu). On exige donc : exactement un adversaire n'ayant pas fold
    AVANT la première action volontaire du héros, et aucun autre joueur
    susceptible de parler après lui hormis celui-là.

    ``None`` si le spot est encore multiway (≥ 2 adversaires vivants).
    """
    pre = hand.street_actions(Street.PREFLOP)
    idx_hero = next(
        (
            i for i, a in enumerate(pre)
            if a.player == hero and a.action is not ActionType.POST
        ),
        None,
    )
    if idx_hero is None:
        return None
    folded_before = {
        a.player for a in pre[:idx_hero] if a.action is ActionType.FOLD
    }
    alive = [
        s.player for s in hand.seats
        if s.player != hero and s.player not in folded_before
    ]
    return alive[0] if len(alive) == 1 else None


def _sb_decision(hand: ParsedHand, hero: str, eff_chips: float) -> str | None:
    """Classe la décision préflop du héros SB : jam / fold / limp / raise.

    None si le héros n'a pas de décision volontaire préflop (ne devrait pas
    arriver au SB).
    """
    pre = [a for a in hand.street_actions(Street.PREFLOP) if a.player == hero]
    voluntary = [a for a in pre if a.action is not ActionType.POST]
    if not voluntary:
        return None
    committed = sum(a.amount for a in pre)
    if any(a.action is ActionType.ALLIN for a in voluntary) or committed >= 0.9 * eff_chips:
        return "jam"
    first = voluntary[0]
    if first.action is ActionType.FOLD:
        return "fold"
    if first.action is ActionType.CALL:
        return "limp"
    return "raise"


def _judge(decision: str, jam_freq: float, ev: float) -> tuple[str, float]:
    """Verdict + coût en bb d'une décision, vu l'équilibre.

    ev = EV(jam) − EV(fold) en bb. Un jam est correct si ev ≥ 0. Le coût
    d'un écart est l'EV sacrifiée : |ev| pour un jam↔fold inversé ; pour un
    limp, l'EV de la meilleure action pure abandonnée (max(ev, 0) si le jam
    était correct, sinon on ne pénalise pas — le fold reste disponible).
    """
    should_jam = ev >= 0.0
    if decision == "jam":
        return ("ok", 0.0) if should_jam else ("jam trop lâche", -ev)
    if decision == "fold":
        return ("ok", 0.0) if not should_jam else ("fold trop serré", ev)
    if decision == "limp":
        # limper est dominé en push/fold : coût = EV du jam s'il valait mieux
        return ("limp", max(ev, 0.0))
    return ("ok", 0.0)  # raise non-jam : hors périmètre, non pénalisé


def review_pushfold_hands(
    hands: Sequence[ParsedHand], hero: str | None = None
) -> PushFoldReview:
    """Revue shove/fold des décisions SB heads-up du héros."""
    hands = [h for h in hands if h.seats]
    if hero is None:
        counts: dict[str, int] = {}
        for h in hands:
            if h.hero:
                counts[h.hero] = counts.get(h.hero, 0) + 1
        hero = max(counts, key=counts.get) if counts else None
    if hero is None:
        return PushFoldReview()

    rev = PushFoldReview()
    for h in hands:
        if h.big_blind <= 0 or len(h.hero_cards) != 2:
            continue
        seats = h.seats
        if not any(s.player == hero for s in seats):
            continue
        hero_seat = next(s for s in seats if s.player == hero)
        # le héros doit être le premier à parler et le dernier agresseur
        # potentiel : bouton (= SB en HU, = ouvreur en 3-max)
        if hero_seat.seat != h.button_seat:
            continue
        opponent = _heads_up_opponent(h, hero)
        if opponent is None:
            rev.n_skipped_multiway += 1
            continue
        villain_seat = next(s for s in seats if s.player == opponent)
        eff_chips = min(hero_seat.stack, villain_seat.stack)
        eff_bb = eff_chips / h.big_blind
        if eff_bb > MAX_EFF_BB:
            rev.n_skipped_deep += 1
            continue
        if eff_bb < 1.0:
            continue

        rev.n_hu_sb += 1
        decision = _sb_decision(h, hero, eff_chips)
        if decision is None:
            continue

        g = _hand_group(h.hero_cards)
        ante_bb = h.ante / h.big_blind if h.ante else 0.0
        sol = _solution(round(eff_bb * 10), round(ante_bb * 100))
        jam_freq = float(sol.jam_range[g])
        ev = float(sol.ev_jam_par_groupe[g])
        verdict, cost = _judge(decision, jam_freq, ev)

        rev.spots.append(
            ShoveSpot(
                hand_id=h.hand_id,
                hand=group_name(g),
                eff_bb=round(eff_bb, 1),
                decision=decision,
                nash_jam_freq=jam_freq,
                ev_jam_minus_fold=ev,
                verdict=verdict,
                cost_bb=cost,
            )
        )
    return rev


def review_pushfold_folder(path: str | Path, salt: str = "") -> PushFoldReview:
    """Lit tous les XML iPoker d'un dossier et en fait la revue shove/fold."""
    root = Path(path)
    hands: list[ParsedHand] = []
    for f in sorted(root.rglob("*.xml")):
        try:
            hands.extend(parse_file(f, salt))
        except Exception:
            continue
    return review_pushfold_hands(hands)
