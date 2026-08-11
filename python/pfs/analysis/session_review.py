"""Revue de session — profil statistique du héros + équité des tapis.

Deux mesures, toutes deux **exactes** et **descriptives** (aucune ne juge
une décision avec la connaissance du futur autrement qu'en le disant) :

1. **Profil statistique** (VPIP, PFR, 3-bet, fold-to-cbet, WTSD, résultat
   net en bb) — agrégé sur les mains, à partir des observations exactes du
   hand-history. C'est le socle de la détection de fuites.

2. **Équité des tapis (« all-in adjusted »)** — pour chaque confrontation
   tapis où l'adversaire a été abattu (donc ses cartes sont connues), on
   calcule l'équité du héros AU MOMENT où les jetons sont entrés (préflop :
   0 carte ; flop : 3 ; turn : 4). La somme des gains réalisés comparée à
   la somme des espérances (équité × pot) donne la part de variance
   (« chance ») : au-dessus de zéro, le héros a gagné plus que son équité ;
   en dessous, moins. C'est exactement la métrique « all-in EV » des
   trackers, calculée ici sans réseau, sur ses propres mains.

Honnêteté (NEMESIS) : seules les confrontations tapis **heads-up avec
cartes adverses connues** sont mesurées exactement. Les tapis multiway et
ceux où l'adversaire s'est couché sans montrer sont comptés à part et
signalés, jamais noyés dans la moyenne.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from pfs.analysis.reperes import (
    REPERES,
    Ecart,
    JeuDeReperes,
    compter_agression,
    jeu_par_defaut,
    situer,
)
from pfs.core.equity import equity_vs_range
from pfs.core.range_model import RANKS, SUITS, Range, combo_index
from pfs.data.hand_history import (
    ActionType,
    ParsedHand,
    Street,
    parse_file,
)

__all__ = [
    "AllInSpot",
    "HeroProfile",
    "SessionReport",
    "review_hands",
    "review_folder",
]

_STREET_BOARD = {
    Street.PREFLOP: 0,
    Street.FLOP: 3,
    Street.TURN: 4,
    Street.RIVER: 5,
}

#: Noms lisibles des statistiques comparées aux repères.
_LIBELLES = {
    "vpip": "VPIP",
    "pfr": "PFR",
    "ecart_vpip_pfr": "écart VPIP−PFR",
    "three_bet": "3-bet",
    "fold_to_cbet": "fold to c-bet",
    "wtsd": "WTSD",
    "af_postflop": "AF postflop",
}


def _card_int(s: str) -> int:
    """« Ah », « Ts », « 5d » → indice carte 0-51 (RANKS·4 + SUITS)."""
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def _committed(hand: ParsedHand, player: str) -> float:
    """Total misé par un joueur dans la main (somme des incréments)."""
    return sum(a.amount for a in hand.actions if a.player == player)


def _single_combo_range(cards: Sequence[str]) -> Range:
    """Range dégénérée : poids 1 sur l'unique combo (les 2 cartes adverses)."""
    w = np.zeros(1326, dtype=np.float64)
    a, b = _card_int(cards[0]), _card_int(cards[1])
    w[combo_index(a, b)] = 1.0
    return Range(w)


@dataclass(slots=True)
class AllInSpot:
    """Une confrontation tapis heads-up, cartes adverses connues."""

    hand_id: str
    street: Street
    hero_cards: tuple[str, ...]
    villain_cards: tuple[str, ...]
    board: tuple[str, ...]
    equity: float          # équité héros au moment du tapis
    pot: float             # pot disputé (jetons)
    realized: float        # gain réel du héros (jetons)
    big_blind: float

    @property
    def expected(self) -> float:
        """Espérance de gain du héros dans ce pot : équité × pot."""
        return self.equity * self.pot

    @property
    def luck_bb(self) -> float:
        """Écart réalisé − espéré, normalisé en big blinds."""
        if self.big_blind <= 0:
            return 0.0
        return (self.realized - self.expected) / self.big_blind


@dataclass(slots=True)
class HeroProfile:
    """Statistiques agrégées du héros sur la session."""

    n_hands: int = 0
    vpip_opp: int = 0
    vpip_yes: int = 0
    pfr_opp: int = 0
    pfr_yes: int = 0
    three_bet_opp: int = 0
    three_bet_yes: int = 0
    fold_cbet_opp: int = 0
    fold_cbet_yes: int = 0
    wtsd_opp: int = 0
    wtsd_yes: int = 0
    net_bb: float = 0.0
    #: Actions agressives et suivis du héros APRÈS le préflop — de quoi
    #: calculer l'AF, la seule statistique de style que les repères de corpus
    #: donnent et que le profil ne mesurait pas.
    aggr_postflop: int = 0
    calls_postflop: int = 0
    #: Nombre de mains par taille de table. Sert à choisir le jeu de repères :
    #: un joueur de table à trois ne se compare pas à un agrégat 6-max.
    tables: dict[int, int] = field(default_factory=dict)

    @staticmethod
    def _pct(yes: int, opp: int) -> float:
        return 100.0 * yes / opp if opp else 0.0

    @property
    def vpip(self) -> float:
        return self._pct(self.vpip_yes, self.vpip_opp)

    @property
    def pfr(self) -> float:
        return self._pct(self.pfr_yes, self.pfr_opp)

    @property
    def three_bet(self) -> float:
        return self._pct(self.three_bet_yes, self.three_bet_opp)

    @property
    def fold_to_cbet(self) -> float:
        return self._pct(self.fold_cbet_yes, self.fold_cbet_opp)

    @property
    def wtsd(self) -> float:
        return self._pct(self.wtsd_yes, self.wtsd_opp)

    @property
    def af_postflop(self) -> float:
        """(mises + relances + tapis) / suivis, après le préflop.

        0.0 quand le héros n'a jamais suivi postflop : le ratio n'existe pas,
        et rendre l'infini ferait afficher un « très agressif » qui n'a été
        mesuré sur rien. ``calls_postflop`` dit toujours sur quoi il porte.
        """
        return self.aggr_postflop / self.calls_postflop if self.calls_postflop else 0.0

    @property
    def sieges_median(self) -> int:
        """Taille de table médiane des mains jouées (0 si aucune main).

        Médiane et non moyenne : un joueur de table à trois qui a joué
        quelques mains à six ne doit pas basculer sur le repère 6-max.
        """
        if not self.tables:
            return 0
        rangs = sorted(n for n, k in self.tables.items() for _ in range(k))
        return rangs[len(rangs) // 2]

    def mesures(self) -> dict[str, float]:
        """Le profil sous la forme qu'attend :func:`pfs.analysis.reperes.situer`."""
        return {
            "vpip": self.vpip,
            "pfr": self.pfr,
            "ecart_vpip_pfr": self.vpip - self.pfr,
            "three_bet": self.three_bet,
            "fold_to_cbet": self.fold_to_cbet,
            "wtsd": self.wtsd,
            "af_postflop": self.af_postflop,
        }

    def occasions(self) -> dict[str, int]:
        """Les dénominateurs correspondants, dans le même vocabulaire."""
        return {
            "vpip": self.vpip_opp,
            "pfr": self.pfr_opp,
            "ecart_vpip_pfr": self.vpip_opp,
            "three_bet": self.three_bet_opp,
            "fold_to_cbet": self.fold_cbet_opp,
            "wtsd": self.wtsd_opp,
            "af_postflop": self.calls_postflop,
        }


@dataclass(slots=True)
class SessionReport:
    """Rapport complet de revue de session."""

    profile: HeroProfile
    allin_spots: list[AllInSpot] = field(default_factory=list)
    n_allin_total: int = 0        # tapis héros détectés
    n_allin_measured: int = 0     # tapis HU mesurables (cartes connues)
    n_allin_skipped: int = 0      # multiway ou adversaire non montré

    @property
    def total_expected(self) -> float:
        return sum(s.expected for s in self.allin_spots)

    @property
    def total_realized(self) -> float:
        return sum(s.realized for s in self.allin_spots)

    @property
    def luck_bb(self) -> float:
        """Chance cumulée en bb : réalisé − espéré sur les tapis mesurés."""
        return sum(s.luck_bb for s in self.allin_spots)

    @property
    def avg_allin_equity(self) -> float:
        """Équité moyenne du héros quand tapis (>50 % = « get it in good »)."""
        if not self.allin_spots:
            return 0.0
        return 100.0 * sum(s.equity for s in self.allin_spots) / len(self.allin_spots)

    # ── confrontation aux repères mesurés sur corpus ─────────────────────
    def jeu_de_reperes(self) -> JeuDeReperes:
        """Le jeu de repères adapté à la taille de table réellement jouée.

        Une table à trois se compare au mélange de positions d'une table à
        trois, pas à un agrégat 6-max qui inclut l'UTG et le MP — voir
        :mod:`pfs.analysis.reperes`.
        """
        return REPERES[jeu_par_defaut(self.profile.sieges_median or 6)]

    def ecarts(self) -> list[Ecart]:
        """Écarts du héros aux repères, le plus grand comparable en tête."""
        return situer(self.profile.mesures(), self.jeu_de_reperes(),
                      self.profile.occasions())

    def explain(self) -> str:
        p = self.profile
        lines = [
            "══════════════════════════════════════════════════════════",
            "  REVUE DE SESSION — analyse a posteriori de tes mains",
            "══════════════════════════════════════════════════════════",
            f"  Mains analysées        : {p.n_hands}",
            f"  Résultat net           : {p.net_bb:+.1f} bb",
            "",
            "  Profil (fréquences, avec nombre d'occasions) :",
            f"    VPIP          {p.vpip:5.1f} %   ({p.vpip_yes}/{p.vpip_opp})",
            f"    PFR           {p.pfr:5.1f} %   ({p.pfr_yes}/{p.pfr_opp})",
            f"    3-bet         {p.three_bet:5.1f} %   ({p.three_bet_yes}/{p.three_bet_opp})",
            f"    Fold to c-bet {p.fold_to_cbet:5.1f} %   ({p.fold_cbet_yes}/{p.fold_cbet_opp})",
            f"    WTSD          {p.wtsd:5.1f} %   ({p.wtsd_yes}/{p.wtsd_opp})",
            "",
            "  Équité des tapis (all-in adjusted) :",
            f"    Confrontations mesurées : {self.n_allin_measured}"
            f"  (sur {self.n_allin_total} tapis ; {self.n_allin_skipped} non mesurables)",
        ]
        if self.n_allin_measured:
            lines += [
                f"    Équité moyenne quand tapis : {self.avg_allin_equity:.1f} %",
                f"    Gains réalisés  : {self.total_realized:.0f} jetons",
                f"    Gains espérés   : {self.total_expected:.0f} jetons",
                f"    Chance (réalisé − espéré) : {self.luck_bb:+.1f} bb",
            ]
            if self.luck_bb > 0:
                lines.append("    → tu as gagné PLUS que ton équité (variance favorable).")
            elif self.luck_bb < 0:
                lines.append("    → tu as gagné MOINS que ton équité (variance défavorable).")

        jeu = self.jeu_de_reperes()
        lines += [
            "",
            f"  Face aux repères MESURÉS « {jeu.cle} »",
            f"    ({jeu.n_mains} mains, {jeu.n_mains_joueurs} mains-joueurs) :",
        ]
        for e in self.ecarts():
            unite = e.unite or ""
            drapeau = "" if e.comparable else "  ⚠ non comparable"
            lines.append(
                f"    {_LIBELLES.get(e.stat, e.stat):<16}"
                f"{e.valeur:6.1f}{unite:<2} vs {e.repere:5.1f}{unite:<2}"
                f"  écart {e.ecart:+6.1f}{drapeau}")
            if not e.comparable:
                lines.append(f"       ({e.pourquoi})")
        for lim in jeu.limites:
            lines.append(f"    · {lim}")
        lines.append("══════════════════════════════════════════════════════════")
        return "\n".join(lines)


def _hero_all_in_street(hand: ParsedHand, hero: str) -> Street | None:
    """Rue où le héros a mis son dernier jeton, s'il est bien all-in.

    All-in = total misé ≥ tapis de départ du siège. La rue est celle de sa
    dernière action volontaire (après quoi il ne peut plus agir).
    """
    seat = next((s for s in hand.seats if s.player == hero), None)
    if seat is None or seat.stack <= 0:
        return None
    if _committed(hand, hero) < seat.stack - 1e-6:
        return None
    last = None
    for a in hand.actions:
        if a.player == hero and a.action is not ActionType.FOLD:
            last = a.street
    return last


def _lone_shown_opponent(
    hand: ParsedHand, hero: str
) -> tuple[str, tuple[str, ...]] | None:
    """L'unique adversaire abattu (cartes connues) ; None si 0 ou ≥2.

    Restreint la mesure exacte aux confrontations heads-up : au-delà, il
    faudrait l'équité multiway et une gestion des pots annexes — compté à
    part, jamais approximé en silence.
    """
    opponents = [
        s for s in hand.seats
        if s.player != hero and len(s.cards) == 2
    ]
    if len(opponents) != 1:
        return None
    return opponents[0].player, opponents[0].cards


def review_hands(hands: Sequence[ParsedHand], hero: str | None = None) -> SessionReport:
    """Construit le rapport de revue à partir de mains parsées.

    Parameters
    ----------
    hands
        Mains parsées (``ParsedHand``), typiquement d'un même compte.
    hero
        Clé (hachée) du héros. Par défaut, le héros majoritaire des mains.
    """
    hands = [h for h in hands if h.seats]
    if hero is None:
        counts: dict[str, int] = {}
        for h in hands:
            if h.hero:
                counts[h.hero] = counts.get(h.hero, 0) + 1
        hero = max(counts, key=counts.get) if counts else None

    profile = HeroProfile()
    spots: list[AllInSpot] = []
    n_allin_total = n_measured = n_skipped = 0

    for h in hands:
        if hero is None or not any(s.player == hero for s in h.seats):
            continue
        profile.n_hands += 1
        n_sieges = len(h.seats)
        profile.tables[n_sieges] = profile.tables.get(n_sieges, 0) + 1
        agr, sui = compter_agression(h, hero)
        profile.aggr_postflop += agr
        profile.calls_postflop += sui

        # résultat net en bb (gain − mises), normalisé par la bb de la main
        if h.big_blind > 0:
            net = h.winners.get(hero, 0.0) - _committed(h, hero)
            profile.net_bb += net / h.big_blind

        obs = h.stat_observations(hero)
        if "vpip" in obs:
            profile.vpip_opp += 1
            profile.vpip_yes += obs["vpip"]
        if "pfr" in obs:
            profile.pfr_opp += 1
            profile.pfr_yes += obs["pfr"]
        if "three_bet" in obs:
            profile.three_bet_opp += 1
            profile.three_bet_yes += obs["three_bet"]
        if "fold_to_cbet" in obs:
            profile.fold_cbet_opp += 1
            profile.fold_cbet_yes += obs["fold_to_cbet"]
        if "wtsd" in obs:
            profile.wtsd_opp += 1
            profile.wtsd_yes += obs["wtsd"]

        # équité des tapis
        street = _hero_all_in_street(h, hero)
        if street is None:
            continue
        n_allin_total += 1
        hero_seat = next(s for s in h.seats if s.player == hero)
        if len(hero_seat.cards) != 2 and len(h.hero_cards) != 2:
            n_skipped += 1
            continue
        opp = _lone_shown_opponent(h, hero)
        if opp is None:
            n_skipped += 1
            continue

        hero_cards = hero_seat.cards if len(hero_seat.cards) == 2 else h.hero_cards
        _opp_key, opp_cards = opp
        n_board = _STREET_BOARD[street]
        board = h.board[:n_board]
        try:
            hero_ints = [_card_int(c) for c in hero_cards]
            board_ints = [_card_int(c) for c in board]
            villain = _single_combo_range(opp_cards)
            eq = equity_vs_range(hero_ints, villain, board_ints, n_sims=20_000).equity
        except Exception:
            n_skipped += 1
            continue

        n_measured += 1
        spots.append(
            AllInSpot(
                hand_id=h.hand_id,
                street=street,
                hero_cards=tuple(hero_cards),
                villain_cards=tuple(opp_cards),
                board=board,
                equity=eq,
                pot=h.pot,
                realized=h.winners.get(hero, 0.0),
                big_blind=h.big_blind,
            )
        )

    return SessionReport(
        profile=profile,
        allin_spots=spots,
        n_allin_total=n_allin_total,
        n_allin_measured=n_measured,
        n_allin_skipped=n_skipped,
    )


def review_folder(path: str | Path, salt: str = "") -> SessionReport:
    """Lit tous les XML iPoker d'un dossier et en fait la revue.

    Emplacement PMU typique (Windows) :
    ``%LocalAppData%\\PMU PLAY 100%% Poker\\data\\<pseudo>\\History``.
    """
    root = Path(path)
    hands: list[ParsedHand] = []
    for f in sorted(root.rglob("*.xml")):
        try:
            hands.extend(parse_file(f, salt))
        except Exception:
            continue
    return review_hands(hands)
