"""
Parseurs de hand-history — Winamax, PokerStars, iPoker (PMU/partypoker).

Le tail de hand-history est le **second canal** du système, et le plus précieux
des deux pour trois raisons :

1. il alimente la modélisation d'adversaire avec des données exactes, pas OCR ;
2. il sert d'**oracle de validation** au scraper : comparer l'état reconstruit
   par la vision au HH de la même main donne une mesure de précision en
   production, gratuite et continue ;
3. il survit à tout — changement de skin, `WDA_EXCLUDEFROMCAPTURE`, mise à jour
   du client.

⚠️ Granularité : **fin de main**. Aucun opérateur n'écrit de fichier « main en
cours ». Le HH ne peut donc pas piloter une décision *dans* la main.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from typing import Iterator, Sequence

__all__ = [
    "Room",
    "Street",
    "ActionType",
    "HandAction",
    "PlayerSeat",
    "ParsedHand",
    "parse_winamax",
    "parse_pokerstars",
    "detect_room",
    "parse_text",
    "parse_file",
    "iter_hands",
    "player_key",
    "HH_PATHS",
]


class Room(str, Enum):
    WINAMAX = "winamax"
    POKERSTARS = "pokerstars"
    IPOKER = "ipoker"
    UNKNOWN = "unknown"


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class ActionType(str, Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALLIN = "allin"
    POST = "post"


class HandHistoryError(ValueError):
    pass


# Emplacements par défaut, à titre documentaire (Windows).
HH_PATHS: dict[Room, str] = {
    Room.WINAMAX: r"%AppData%\winamax\documents\accounts\<PSEUDO>\history",
    Room.POKERSTARS: r"C:\Program Files\PokerStars\HandHistory",
    Room.IPOKER: r"%LocalAppData%\<skin>\HandHistory",
}


def player_key(nickname: str, salt: str = "") -> str:
    """Clé stable et non réversible. Les pseudos ne sont **jamais** stockés en clair.

    C'est une donnée personnelle au sens du RGPD, et le data-sharing est
    explicitement interdit par les CGU Winamax.
    """
    h = blake2b(digest_size=8)
    h.update(salt.encode("utf-8"))
    h.update(nickname.strip().lower().encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class HandAction:
    player: str
    street: Street
    action: ActionType
    amount: float = 0.0
    total_bet: float = 0.0

    def __str__(self) -> str:
        amt = f" {self.amount:.2f}" if self.amount else ""
        return f"{self.street.value[:2]}:{self.player[:8]} {self.action.value}{amt}"


@dataclass(frozen=True, slots=True)
class PlayerSeat:
    seat: int
    player: str
    stack: float
    is_hero: bool = False
    cards: tuple[str, ...] = ()


@dataclass(slots=True)
class ParsedHand:
    room: Room
    hand_id: str
    is_real_money: bool
    is_tournament: bool
    big_blind: float
    ante: float
    table: str
    button_seat: int
    seats: list[PlayerSeat] = field(default_factory=list)
    hero: str | None = None
    hero_cards: tuple[str, ...] = ()
    board: tuple[str, ...] = ()
    actions: list[HandAction] = field(default_factory=list)
    # Straddles dans l'ordre de pose : (joueur, montant), joueur = clé hachée
    # (player_key), comme partout ailleurs. Un straddle Mississippi — posté du
    # bouton — ne demande aucun traitement particulier au parseur : l'ordre
    # d'apparition des lignes du HH suffit, et cette liste le préserve.
    straddles: list[tuple[str, float]] = field(default_factory=list)
    pot: float = 0.0
    winners: dict[str, float] = field(default_factory=dict)
    raw: str = ""

    @property
    def effective_bb(self) -> float:
        """Big blind effective de la main.

        Returns
        -------
        float
            Montant du **dernier** straddle posé s'il y en a un (en cas de
            double straddle, c'est le dernier qui fixe le niveau du jeu),
            sinon ``big_blind``. C'est la référence correcte pour normaliser
            les stacks en bb : sur une table 0.50/1 straddlée à 2€, un tapis
            de 100€ vaut 50 bb effectives, pas 100.
        """
        if self.straddles:
            return self.straddles[-1][1]
        return self.big_blind

    # ── statistiques dérivées (alimentent F1) ────────────────────────────
    def street_actions(self, street: Street) -> list[HandAction]:
        return [a for a in self.actions if a.street is street]

    def voluntarily_put_in_pot(self, player: str) -> bool:
        """VPIP : a mis de l'argent volontairement préflop (blindes exclues)."""
        return any(
            a.player == player
            and a.street is Street.PREFLOP
            and a.action in (ActionType.CALL, ActionType.BET, ActionType.RAISE, ActionType.ALLIN)
            for a in self.actions
        )

    def preflop_raise(self, player: str) -> bool:
        return any(
            a.player == player
            and a.street is Street.PREFLOP
            and a.action in (ActionType.RAISE, ActionType.ALLIN)
            for a in self.actions
        )

    def three_bet(self, player: str) -> bool:
        """A relancé alors qu'une relance préflop existait déjà."""
        raises = 0
        for a in self.street_actions(Street.PREFLOP):
            if a.action in (ActionType.RAISE, ActionType.ALLIN):
                raises += 1
                if raises >= 2 and a.player == player:
                    return True
        return False

    def faced_cbet(self, player: str) -> bool:
        flop = self.street_actions(Street.FLOP)
        for i, a in enumerate(flop):
            if a.action is ActionType.BET and a.player != player:
                return any(x.player == player for x in flop[i + 1:])
        return False

    def folded_to_cbet(self, player: str) -> bool:
        flop = self.street_actions(Street.FLOP)
        for i, a in enumerate(flop):
            if a.action is ActionType.BET and a.player != player:
                for x in flop[i + 1:]:
                    if x.player == player:
                        return x.action is ActionType.FOLD
        return False

    def went_to_showdown(self, player: str) -> bool:
        return any(s.player == player and s.cards for s in self.seats)

    def stat_observations(self, player: str) -> dict[str, bool]:
        """Toutes les statistiques observables sur cette main, pour ce joueur.

        Les clés absentes signifient « pas d'occasion » — il ne faut alors
        **rien** injecter dans le tracker, sous peine de biaiser l'estimation.
        """
        out: dict[str, bool] = {}
        if any(s.player == player for s in self.seats):
            out["vpip"] = self.voluntarily_put_in_pot(player)
            out["pfr"] = self.preflop_raise(player)
            pf = self.street_actions(Street.PREFLOP)
            if sum(1 for a in pf if a.action in (ActionType.RAISE, ActionType.ALLIN)) >= 1:
                out["three_bet"] = self.three_bet(player)
            if self.faced_cbet(player):
                out["fold_to_cbet"] = self.folded_to_cbet(player)
            if self.board:
                out["wtsd"] = self.went_to_showdown(player)
        return out

    def __repr__(self) -> str:
        return (
            f"ParsedHand({self.room.value} #{self.hand_id} "
            f"{'MTT' if self.is_tournament else 'cash'} "
            f"{'€' if self.is_real_money else 'fictif'} "
            f"bb={self.big_blind} {len(self.seats)}j "
            f"board={'-'.join(self.board) or '—'} {len(self.actions)} actions)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# WINAMAX
# ═══════════════════════════════════════════════════════════════════════════

_WNMX_HEADER = re.compile(
    r"Winamax Poker - (?P<kind>CashGame|Tournament)"
    r"(?:\s+\"(?P<tname>[^\"]*)\")?"
    r".*?HandId:\s*#(?P<hid>[\d\-]+)"
    r".*?\((?P<sb>[\d.,]+)\s*(?P<cur>[€$£]?)\)/\((?P<bb>[\d.,]+)\s*[€$£]?\)",
    re.IGNORECASE | re.DOTALL,
)
_WNMX_TABLE = re.compile(r"Table:\s*'(?P<table>[^']*)'.*?Seat\s*#(?P<btn>\d+)\s+is the button",
                         re.IGNORECASE | re.DOTALL)
_WNMX_SEAT = re.compile(r"Seat\s+(?P<seat>\d+):\s+(?P<name>.+?)\s+\((?P<stack>[\d.,]+)")
_WNMX_DEALT = re.compile(r"Dealt to\s+(?P<name>.+?)\s+\[(?P<cards>[^\]]+)\]")
_WNMX_ACTION = re.compile(
    r"^(?P<name>.+?)\s+(?P<verb>folds|checks|calls|bets|raises|posts|straddles)"
    r"(?:.*?(?P<amt>[\d.,]+))?",
    re.IGNORECASE,
)
_WNMX_BOARD = re.compile(r"\*\*\*\s+(?P<street>FLOP|TURN|RIVER)\s+\*\*\*\s+(?P<cards>[\[\]\w\s]+)")
_WNMX_COLLECT = re.compile(r"^(?P<name>.+?)\s+collected\s+(?P<amt>[\d.,]+)", re.IGNORECASE)
_WNMX_SHOWN = re.compile(r"^Seat\s+(?P<seat>\d+):\s+(?P<name>.+?)\s+showed\s+\[(?P<cards>[^\]]+)\]")


def _num(txt: str | None) -> float:
    if not txt:
        return 0.0
    return float(txt.replace(",", ".").replace(" ", ""))


def _cards(txt: str) -> tuple[str, ...]:
    return tuple(c for c in re.findall(r"[2-9TJQKA][shdc]", txt))


_STRADDLE_RE = re.compile(r"straddle", re.IGNORECASE)


def _is_straddle_line(line: str, verb: str, street: Street) -> bool:
    """Détecte une pose de straddle, quel que soit le libellé exact.

    Parameters
    ----------
    line : str
        Ligne brute du hand history.
    verb : str
        Verbe d'action déjà extrait par la regex de la room (minuscules).
    street : Street
        Rue courante au moment de la ligne.

    Returns
    -------
    bool
        True si la ligne est une pose de straddle. Tolérant par
        construction : tout verbe de type « post » dont la ligne contient le
        mot « straddle » (« posts straddle 2€ », « posts a live straddle »,
        « straddles €2 »…). Restreint au préflop : un straddle est de
        l'argent mort posté avant la distribution des cartes.
    """
    return (
        street is Street.PREFLOP
        and verb in ("posts", "straddles")
        and _STRADDLE_RE.search(line) is not None
    )


def _straddle_amount(hand: ParsedHand, parsed: float) -> float:
    """Montant d'un straddle, avec repli conventionnel si absent de la ligne.

    Parameters
    ----------
    hand : ParsedHand
        Main en cours de construction (fournit ``big_blind`` et les
        straddles déjà posés).
    parsed : float
        Montant lu sur la ligne, 0.0 si la ligne n'en portait pas
        (PokerStars écrit parfois « X: straddles » sans montant).

    Returns
    -------
    float
        Le montant lu s'il est positif, sinon 2× le niveau précédent —
        convention universelle : 2× la big blind pour le premier straddle,
        2× le dernier straddle pour un re-straddle.
    """
    if parsed > 0:
        return parsed
    base = hand.straddles[-1][1] if hand.straddles else hand.big_blind
    return 2.0 * base


def parse_winamax(text: str, salt: str = "") -> ParsedHand:
    """Parse une main Winamax (nouveau client et legacy AIR).

    Les poses de straddle (« X posts straddle 2€ », « X straddles 2€ ») sont
    enregistrées comme argent mort — ``ActionType.POST``, jamais VPIP, même
    règle que les blindes — et accumulées dans ``ParsedHand.straddles`` dans
    l'ordre de pose. Le straddle Mississippi (posté du bouton) est couvert
    par le même mécanisme : seul l'ordre des lignes compte.
    """
    m = _WNMX_HEADER.search(text)
    if not m:
        raise HandHistoryError("en-tête Winamax introuvable.")

    is_tourney = m.group("kind").lower() == "tournament"
    currency = m.group("cur") or ""
    # Un tournoi affiche des jetons ; « play money » apparaît dans le nom.
    tname = (m.group("tname") or "")
    is_play = "play money" in text.lower() or "argent fictif" in text.lower()

    tm = _WNMX_TABLE.search(text)
    hand = ParsedHand(
        room=Room.WINAMAX,
        hand_id=m.group("hid"),
        is_real_money=not is_play,
        is_tournament=is_tourney,
        big_blind=_num(m.group("bb")),
        ante=0.0,
        table=tm.group("table") if tm else tname,
        button_seat=int(tm.group("btn")) if tm else 0,
        raw=text,
    )

    street = Street.PREFLOP
    board: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Seat ") and "(" in line and "showed" not in line:
            sm = _WNMX_SEAT.match(line)
            if sm:
                hand.seats.append(
                    PlayerSeat(
                        seat=int(sm.group("seat")),
                        player=player_key(sm.group("name"), salt),
                        stack=_num(sm.group("stack")),
                    )
                )
            continue

        dm = _WNMX_DEALT.search(line)
        if dm:
            hand.hero = player_key(dm.group("name"), salt)
            hand.hero_cards = _cards(dm.group("cards"))
            continue

        bm = _WNMX_BOARD.search(line)
        if bm:
            street = {"FLOP": Street.FLOP, "TURN": Street.TURN,
                      "RIVER": Street.RIVER}[bm.group("street").upper()]
            for c in _cards(bm.group("cards")):
                if c not in board:
                    board.append(c)
            continue

        if "*** SHOW DOWN ***" in line or "*** SUMMARY ***" in line:
            street = Street.SHOWDOWN
            continue

        cm = _WNMX_COLLECT.match(line)
        if cm:
            k = player_key(cm.group("name"), salt)
            hand.winners[k] = hand.winners.get(k, 0.0) + _num(cm.group("amt"))
            continue

        shm = _WNMX_SHOWN.match(line)
        if shm:
            k = player_key(shm.group("name"), salt)
            for i, s in enumerate(hand.seats):
                if s.player == k:
                    hand.seats[i] = PlayerSeat(s.seat, s.player, s.stack, s.is_hero,
                                               _cards(shm.group("cards")))
            continue

        am = _WNMX_ACTION.match(line)
        if am and street is not Street.SHOWDOWN:
            verb = am.group("verb").lower()
            kind = {
                "folds": ActionType.FOLD, "checks": ActionType.CHECK,
                "calls": ActionType.CALL, "bets": ActionType.BET,
                "raises": ActionType.RAISE, "posts": ActionType.POST,
                "straddles": ActionType.POST,
            }[verb]
            key = player_key(am.group("name"), salt)
            amount = _num(am.group("amt"))
            if _is_straddle_line(line, verb, street):
                # Argent mort posté avant les cartes : reste un POST (donc
                # jamais VPIP), exactement comme les blindes.
                amount = _straddle_amount(hand, amount)
                hand.straddles.append((key, amount))
            elif "all-in" in line.lower():
                kind = ActionType.ALLIN
            hand.actions.append(
                HandAction(player=key, street=street, action=kind, amount=amount)
            )

    hand.board = tuple(board)
    hand.pot = sum(hand.winners.values())
    if hand.hero:
        hand.seats = [
            PlayerSeat(s.seat, s.player, s.stack, s.player == hand.hero, s.cards)
            for s in hand.seats
        ]
    return hand


# ═══════════════════════════════════════════════════════════════════════════
# POKERSTARS
# ═══════════════════════════════════════════════════════════════════════════

_PS_HEADER = re.compile(
    r"PokerStars\s+(?:Hand|Game)\s+#(?P<hid>\d+):\s+"
    r"(?P<kind>Tournament|.*?)\s*.*?"
    r"\((?P<cur>[€$£]?)(?P<sb>[\d.,]+)/[€$£]?(?P<bb>[\d.,]+)",
    re.IGNORECASE | re.DOTALL,
)
_PS_TABLE = re.compile(r"Table\s+'(?P<table>[^']*)'.*?Seat\s+#(?P<btn>\d+)\s+is the button",
                       re.IGNORECASE | re.DOTALL)
_PS_SEAT = re.compile(r"Seat\s+(?P<seat>\d+):\s+(?P<name>.+?)\s+\((?P<stack>[\d.,]+)")
_PS_DEALT = re.compile(r"Dealt to\s+(?P<name>.+?)\s+\[(?P<cards>[^\]]+)\]")
_PS_ACTION = re.compile(
    r"^(?P<name>.+?):\s+(?P<verb>folds|checks|calls|bets|raises|posts|straddles)"
    r"(?:.*?(?P<amt>[\d.,]+))?",
    re.IGNORECASE,
)
_PS_BOARD = re.compile(r"\*\*\*\s+(?P<street>FLOP|TURN|RIVER)\s+\*\*\*\s+(?P<cards>[\[\]\w\s]+)")
_PS_COLLECT = re.compile(r"^(?P<name>.+?)\s+collected\s+[€$£]?(?P<amt>[\d.,]+)", re.IGNORECASE)


def parse_pokerstars(text: str, salt: str = "") -> ParsedHand:
    """Parse une main PokerStars (cash et tournoi, argent réel ou fictif).

    Les poses de straddle (« X: posts straddle €2 », « X: straddles ») sont
    enregistrées comme argent mort — ``ActionType.POST``, jamais VPIP, même
    règle que les blindes — et accumulées dans ``ParsedHand.straddles`` dans
    l'ordre de pose. Montant absent de la ligne → convention 2× le niveau
    précédent. Le straddle Mississippi (posté du bouton) est couvert par le
    même mécanisme : seul l'ordre des lignes compte.
    """
    m = _PS_HEADER.search(text)
    if not m:
        raise HandHistoryError("en-tête PokerStars introuvable.")

    head = text.splitlines()[0].lower()
    is_tourney = "tournament" in head
    is_play = "play money" in head or (not m.group("cur") and not is_tourney)

    tm = _PS_TABLE.search(text)
    hand = ParsedHand(
        room=Room.POKERSTARS,
        hand_id=m.group("hid"),
        is_real_money=not is_play,
        is_tournament=is_tourney,
        big_blind=_num(m.group("bb")),
        ante=0.0,
        table=tm.group("table") if tm else "",
        button_seat=int(tm.group("btn")) if tm else 0,
        raw=text,
    )

    street = Street.PREFLOP
    board: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("Seat ") and "(" in line and ":" in line and "showed" not in line:
            sm = _PS_SEAT.match(line)
            if sm and "in chips" in line or (sm and "(" in line):
                hand.seats.append(
                    PlayerSeat(int(sm.group("seat")),
                               player_key(sm.group("name"), salt),
                               _num(sm.group("stack")))
                )
            continue

        dm = _PS_DEALT.search(line)
        if dm:
            hand.hero = player_key(dm.group("name"), salt)
            hand.hero_cards = _cards(dm.group("cards"))
            continue

        bm = _PS_BOARD.search(line)
        if bm:
            street = {"FLOP": Street.FLOP, "TURN": Street.TURN,
                      "RIVER": Street.RIVER}[bm.group("street").upper()]
            for c in _cards(bm.group("cards")):
                if c not in board:
                    board.append(c)
            continue

        if "*** SHOW DOWN ***" in line or "*** SUMMARY ***" in line:
            street = Street.SHOWDOWN
            continue

        cm = _PS_COLLECT.match(line)
        if cm:
            k = player_key(cm.group("name"), salt)
            hand.winners[k] = hand.winners.get(k, 0.0) + _num(cm.group("amt"))
            continue

        am = _PS_ACTION.match(line)
        if am and street is not Street.SHOWDOWN:
            verb = am.group("verb").lower()
            kind = {
                "folds": ActionType.FOLD, "checks": ActionType.CHECK,
                "calls": ActionType.CALL, "bets": ActionType.BET,
                "raises": ActionType.RAISE, "posts": ActionType.POST,
                "straddles": ActionType.POST,
            }[verb]
            key = player_key(am.group("name"), salt)
            amount = _num(am.group("amt"))
            if _is_straddle_line(line, verb, street):
                # Argent mort posté avant les cartes : reste un POST (donc
                # jamais VPIP), exactement comme les blindes.
                amount = _straddle_amount(hand, amount)
                hand.straddles.append((key, amount))
            elif "all-in" in line.lower():
                kind = ActionType.ALLIN
            hand.actions.append(HandAction(key, street, kind, amount))

    hand.board = tuple(board)
    hand.pot = sum(hand.winners.values())
    if hand.hero:
        hand.seats = [
            PlayerSeat(s.seat, s.player, s.stack, s.player == hand.hero, s.cards)
            for s in hand.seats
        ]
    return hand


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════


def detect_room(text: str) -> Room:
    head = text[:400].lower()
    if "winamax" in head:
        return Room.WINAMAX
    if "pokerstars" in head:
        return Room.POKERSTARS
    if "ipoker" in head or "game #" in head:
        return Room.IPOKER
    return Room.UNKNOWN


def parse_text(text: str, salt: str = "") -> ParsedHand:
    room = detect_room(text)
    if room is Room.WINAMAX:
        return parse_winamax(text, salt)
    if room in (Room.POKERSTARS, Room.IPOKER):
        return parse_pokerstars(text, salt)
    raise HandHistoryError("room non reconnue.")


def iter_hands(text: str, salt: str = "") -> Iterator[ParsedHand]:
    """Découpe un fichier multi-mains et parse chaque main indépendamment.

    Une main illisible ne fait pas échouer le fichier : elle est ignorée. Sur
    des millions de mains, un format aberrant finit toujours par apparaître.
    """
    blocks = re.split(r"\n\s*\n(?=(?:Winamax|PokerStars))", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            yield parse_text(block, salt)
        except (HandHistoryError, KeyError, ValueError):
            continue


def parse_file(path: str | Path, salt: str = "") -> list[ParsedHand]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return list(iter_hands(text, salt))
