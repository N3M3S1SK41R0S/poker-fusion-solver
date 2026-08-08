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
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
from pathlib import Path
from typing import Iterator, Sequence
from xml.etree import ElementTree as ET

__all__ = [
    "Room",
    "Street",
    "ActionType",
    "HandAction",
    "PlayerSeat",
    "ParsedHand",
    "parse_winamax",
    "parse_pokerstars",
    "parse_ipoker",
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
        """Le joueur a-t-il vu l'abattage : n'a pas fold et ≥2 survivants.

        Défini par les FOLDS, pas par la présence de cartes : en XML iPoker
        les cartes du héros sont toujours enregistrées (même sur un fold),
        donc « avoir des cartes » ≠ « être allé au showdown ». Cette
        définition est correcte pour toutes les rooms.
        """
        folded = {a.player for a in self.actions if a.action is ActionType.FOLD}
        if player in folded:
            return False
        survivors = {s.player for s in self.seats if s.player not in folded}
        return len(survivors) >= 2

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
# IPOKER (PMU, partypoker, Betclic, Unibet… — le réseau iPoker/Playtech)
# ═══════════════════════════════════════════════════════════════════════════
#
# Format XML, RADICALEMENT différent du texte Winamax/PokerStars : un fichier
# <session> contient N <game>, chaque <game> = UNE main. Codes décodés
# empiriquement sur le corpus HH réel (PMU PLAY / PMU) — voir le tableau :
#
#   action type :  0 fold · 1 SB · 2 BB · 15 ante · 3 call · 4 check
#                  5/7/23 agressif (bet si 1re mise de la rue, sinon raise)
#   round no    :  0 blindes+antes · 1 préflop · 2 flop · 3 turn · 4 river
#   carte       :  <couleur><rang> — « HA »→Ah, « S10 »→Ts, « D4 »→4d ;
#                  « X » = carte cachée (adversaire non abattu)
#   player@chips = TAPIS DE DÉPART de la main (invariant vérifié :
#                  start(N+1) = start(N) − mises(N)) → l'all-in se déduit
#                  structurellement (contribution cumulée == tapis), sans
#                  se fier au code d'action.

_IPOKER_SUITS = {"H": "h", "S": "s", "D": "d", "C": "c"}
_IPOKER_ROUND_STREET = {
    "0": Street.PREFLOP, "1": Street.PREFLOP, "2": Street.FLOP,
    "3": Street.TURN, "4": Street.RIVER,
}
# Sémantique de <action sum=...>, décodée empiriquement et vérifiée contre
# l'attribut de contrôle player@bet (total misé par joueur, exact) :
#   • post (blinde/ante) et call : sum = INCRÉMENT (jetons ajoutés) ;
#   • bet / raise               : sum = CUMUL « raise to » DE LA RUE
#     (préflop = round 0 blindes + round 1 relances réunis) — l'incrément
#     réel vaut donc sum − (déjà misé cette rue).
# L'ante (type 15) est de l'argent mort : compté au total, PAS dans la
# baseline « raise to » (« relancer à 40 » = blinde+relance, ante à part).
_IPOKER_FOLD = "0"
_IPOKER_CHECK = "4"
_IPOKER_ANTE = "15"
_IPOKER_BLINDS = {"1", "2"}
_IPOKER_CALL_CODES = {"3", "7"}
_IPOKER_RAISE_CODES = {"5", "23"}


def _ipoker_card(tok: str) -> str | None:
    """Convertit un jeton carte iPoker au format interne (rang + couleur).

    Parameters
    ----------
    tok : str
        Jeton brut, couleur en tête : « HA », « S10 », « D4 », « CJ ».

    Returns
    -------
    str | None
        Carte « Ah », « Ts », « 4d », « Jc »… ; ``None`` pour une carte
        cachée (« X », « XX ») ou un jeton illisible.
    """
    tok = tok.strip()
    if len(tok) < 2 or tok[0] not in _IPOKER_SUITS:
        return None
    suit = _IPOKER_SUITS[tok[0]]
    rank = tok[1:].upper()
    if rank == "10":
        rank = "T"
    if rank not in {"2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"}:
        return None
    return f"{rank}{suit}"


def _ipoker_amount(txt: str | None) -> float:
    """Montant iPoker : espaces = séparateurs de milliers (« 1 355 »)."""
    if not txt:
        return 0.0
    return float(txt.replace(" ", "").replace(" ", "").replace(",", "."))


def _parse_ipoker_game(
    session_general: ET.Element | None, game: ET.Element, salt: str
) -> ParsedHand:
    """Parse UN ``<game>`` iPoker (une main) en ``ParsedHand``.

    L'all-in n'est pas lu dans le code d'action (aucun code fiable ne l'isole
    des raises ordinaires) mais déduit : une action dont la contribution
    cumulée atteint le tapis de départ du joueur est un all-in.
    """
    sg = session_general
    nickname = sg.findtext("nickname") if sg is not None else None
    mode = (sg.findtext("mode") if sg is not None else "") or ""
    is_tourney = bool(
        (sg is not None and sg.findtext("tournamentcode"))
        or (sg is not None and sg.findtext("tournamentname"))
    )

    g_gen = game.find("general")
    sb = _ipoker_amount(g_gen.findtext("smallblind") if g_gen is not None else None)
    bb = _ipoker_amount(g_gen.findtext("bigblind") if g_gen is not None else None)

    hero_key = player_key(nickname, salt) if nickname else None
    hand = ParsedHand(
        room=Room.IPOKER,
        hand_id=game.get("gamecode", ""),
        is_real_money=(mode.lower() == "real"),
        is_tournament=is_tourney,
        big_blind=bb,
        ante=0.0,
        table=(sg.findtext("tablename") if sg is not None else "") or "",
        button_seat=0,
        hero=hero_key,
        raw=ET.tostring(game, encoding="unicode"),
    )

    # sièges + tapis de départ + total misé (@bet, vérité-terrain exacte)
    start_chips: dict[str, float] = {}
    bet_attr: dict[str, float] = {}
    name_to_key: dict[str, str] = {}
    if g_gen is not None:
        for p in g_gen.findall("players/player"):
            name = p.get("name", "")
            key = player_key(name, salt)
            name_to_key[name] = key
            stack = _ipoker_amount(p.get("chips"))
            start_chips[key] = stack
            bet_attr[key] = _ipoker_amount(p.get("bet"))
            hand.seats.append(
                PlayerSeat(
                    seat=int(p.get("seat", "0")),
                    player=key,
                    stack=stack,
                    is_hero=(key == hero_key),
                )
            )
            if p.get("dealer") == "1":
                hand.button_seat = int(p.get("seat", "0"))

    street_live: dict[str, float] = defaultdict(float)  # misé cette rue (hors ante)
    board: list[str] = []
    ante_total = 0.0
    shown: dict[str, tuple[str, ...]] = {}
    prev_street: Street | None = None
    street_opened = False

    for rnd in game.findall("round"):
        street = _IPOKER_ROUND_STREET.get(rnd.get("no", ""), Street.PREFLOP)
        if street is not prev_street:
            # nouvelle rue → la mise live repart de zéro. Préflop (round 0
            # blindes puis round 1 relances) est UNE seule rue : pas de reset
            # entre les deux, sinon les blindes sortiraient de la baseline.
            # La rue préflop est déjà « ouverte » par la BB (mise initiale) :
            # la 1re action agressive y est donc une RELANCE, pas un bet —
            # sans quoi la stat PFR (preflop_raise) manquerait les open-raises.
            street_live.clear()
            street_opened = street is Street.PREFLOP
            prev_street = street

        for cards in rnd.findall("cards"):
            ctype = (cards.get("type") or "").lower()
            toks = [c for c in (_ipoker_card(t) for t in (cards.text or "").split()) if c]
            if ctype == "pocket":
                key = name_to_key.get(cards.get("player", ""))
                if key and toks:
                    shown[key] = tuple(toks)
                    if key == hero_key:
                        hand.hero_cards = tuple(toks)
            else:  # Flop / Turn / River
                for c in toks:
                    if c not in board:
                        board.append(c)

        for act in rnd.findall("action"):
            name = act.get("player", "")
            key = name_to_key.get(name, player_key(name, salt))
            code = act.get("type", "")
            raw = _ipoker_amount(act.get("sum"))

            if code == _IPOKER_FOLD:
                hand.actions.append(HandAction(key, street, ActionType.FOLD, 0.0))
                continue
            if code == _IPOKER_CHECK:
                hand.actions.append(HandAction(key, street, ActionType.CHECK, 0.0))
                continue

            if code == _IPOKER_ANTE:               # argent mort, hors baseline
                inc = raw
                kind = ActionType.POST
                ante_total = max(ante_total, raw)
            elif code in _IPOKER_BLINDS:           # blinde live (incrément)
                inc = raw
                street_live[key] += raw
                kind = ActionType.POST
            elif code in _IPOKER_CALL_CODES:       # call (incrément)
                inc = raw
                street_live[key] += raw
                kind = ActionType.CALL
            elif code in _IPOKER_RAISE_CODES:      # bet/raise (« raise to » de la rue)
                inc = max(0.0, raw - street_live[key])
                street_live[key] = raw
                kind = ActionType.RAISE if street_opened else ActionType.BET
                street_opened = True
            else:                                   # code inconnu : traité en call prudent
                inc = raw
                street_live[key] += raw
                kind = ActionType.CALL

            hand.actions.append(
                HandAction(key, street, kind, amount=inc, total_bet=street_live[key])
            )

    hand.ante = ante_total
    hand.board = tuple(board)

    # All-in exact via la vérité-terrain @bet (le total misé y est donné sans
    # l'ambiguïté ante/relance du champ sum) : un joueur dont le total misé
    # atteint son tapis de départ est all-in. On ne requalifie en ALLIN que
    # sa dernière action AGRESSIVE (bet/raise) — cohérent avec le modèle où
    # ALLIN vaut relance (preflop_raise, three_bet le comptent ainsi). Un
    # suivi qui met all-in reste un CALL (VPIP oui, PFR non) ; son caractère
    # all-in se lit dans les tapis.
    for key, bet in bet_attr.items():
        stack = start_chips.get(key, 0.0)
        if stack <= 0 or bet < stack - 1e-9:
            continue
        for i in range(len(hand.actions) - 1, -1, -1):
            a = hand.actions[i]
            if a.player != key or a.action in (ActionType.FOLD, ActionType.CHECK):
                continue
            if a.action in (ActionType.BET, ActionType.RAISE):
                hand.actions[i] = HandAction(
                    a.player, a.street, ActionType.ALLIN, a.amount, a.total_bet
                )
            break  # dernière action volontaire traitée (call all-in laissé CALL)

    # cartes montrées (adversaires abattus) reportées dans les sièges
    if shown:
        hand.seats = [
            PlayerSeat(s.seat, s.player, s.stack, s.is_hero, shown.get(s.player, s.cards))
            for s in hand.seats
        ]

    # gains (jetons ; rake nul en tournoi)
    if g_gen is not None:
        for p in g_gen.findall("players/player"):
            win = _ipoker_amount(p.get("win"))
            if win > 0:
                hand.winners[player_key(p.get("name", ""), salt)] = win

    # Pot final = somme des gains : c'est le pot RÉELLEMENT attribué (tous
    # pots annexes compris), exact par définition. Reconstruire le pot depuis
    # les mises se heurte à une bizarrerie de comptabilité de l'ante côté
    # iPoker (l'ante du relanceur est tantôt fondue dans le « raise to »,
    # tantôt à part, selon la version du client) ; la somme des gains, elle,
    # ne dépend d'aucune de ces conventions. Le pot à un point de décision
    # précis se recalcule de toute façon depuis les incréments d'action.
    hand.pot = sum(hand.winners.values())
    return hand


def parse_ipoker(text: str, salt: str = "") -> list[ParsedHand]:
    """Parse un fichier session iPoker (XML) — potentiellement N mains.

    Contrairement à Winamax/PokerStars (une main par bloc texte), un
    ``<session>`` iPoker agrège plusieurs ``<game>``. Renvoie donc une liste.
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise HandHistoryError(f"XML iPoker illisible : {exc}") from exc
    if root.tag != "session":
        raise HandHistoryError("racine <session> iPoker absente.")
    sg = root.find("general")
    return [_parse_ipoker_game(sg, game, salt) for game in root.findall("game")]


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════


def detect_room(text: str) -> Room:
    head = text[:600].lower()
    if "winamax" in head:
        return Room.WINAMAX
    if "pokerstars" in head:
        return Room.POKERSTARS
    # iPoker : XML <session> (gametype « Holdem NL »), pas de bannière texte
    if "<session" in head or "ipoker" in head or "<gametype>" in head:
        return Room.IPOKER
    return Room.UNKNOWN


def parse_text(text: str, salt: str = "") -> ParsedHand:
    """Parse UNE main texte (Winamax/PokerStars).

    iPoker est multi-mains et XML : utiliser :func:`parse_ipoker` ou
    :func:`iter_hands`, qui aiguillent correctement.
    """
    room = detect_room(text)
    if room is Room.WINAMAX:
        return parse_winamax(text, salt)
    if room is Room.POKERSTARS:
        return parse_pokerstars(text, salt)
    if room is Room.IPOKER:
        hands = parse_ipoker(text, salt)
        if not hands:
            raise HandHistoryError("aucune main dans la session iPoker.")
        return hands[0]
    raise HandHistoryError("room non reconnue.")


def iter_hands(text: str, salt: str = "") -> Iterator[ParsedHand]:
    """Découpe un fichier multi-mains et parse chaque main indépendamment.

    Aiguille sur le format : XML iPoker (une session → N mains) ou texte
    Winamax/PokerStars (un bloc → une main). Une main illisible ne fait
    jamais échouer le fichier : elle est ignorée — sur des millions de
    mains, un format aberrant finit toujours par apparaître.
    """
    if detect_room(text) is Room.IPOKER:
        try:
            yield from parse_ipoker(text, salt)
        except HandHistoryError:
            return
        return
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
