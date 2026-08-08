"""Conseiller de spot — « qu'est-ce qu'il fallait faire ? » sur une main jouée.

Conçu pour l'étude d'une main **terminée** (capture d'écran, note, souvenir) :
on décrit le spot avec ce qu'une image donne — tes cartes, le board, le pot,
la mise subie, les tapis, la position — et le conseiller rend le verdict
adossé aux solveurs du projet.

Trois régimes, choisis selon le spot :

* **préflop tapis court** (≤ 25 bb, heads-up, héros premier de parole) :
  équilibre de Nash jam/fold exact (``solver.pushfold``) — la réponse est
  une vérité de théorie des jeux, pas une opinion ;
* **préflop tapis profond** : comparaison à la range d'ouverture de
  référence de la position (``GTO_PRESETS``, charts approximatives assumées) ;
* **postflop** : équité exacte de ta main contre une range adverse plausible
  (énumération complète dès le flop), confrontée aux cotes du pot.

Honnêteté (NEMESIS) : postflop, la range adverse est une **hypothèse**, pas
une observation. Le conseiller la déclare toujours, et donne le seuil qui
ferait basculer la décision — c'est cette sensibilité qui a une valeur
d'étude, pas un verdict péremptoire.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pfs.core.bluffcatch import minimum_defence_frequency, required_equity
from pfs.core.equity import equity_vs_range
from pfs.core.range_model import (
    COMBO_TO_GROUP,
    GTO_PRESETS,
    RANKS,
    SUITS,
    combo_index,
    group_name,
    parse_range,
)
from pfs.solver.pushfold import equity_matrix_169, solve_hu_pushfold

__all__ = ["Spot", "Advice", "advise", "parse_cards"]

MAX_PUSHFOLD_BB = 25.0

# Ranges adverses plausibles par défaut (postflop), en l'absence de lecture.
# Un vilain qui MISE sur une rue avancée est plus fort qu'un vilain qui checke.
_DEFAULT_VILLAIN = {
    "large": "22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 86s+, 75s+, 65s, "
             "A7o+, K9o+, QTo+, JTo",
    "moyenne": "22+, A8s+, A5s-A2s, K9s+, QTs+, JTs, T9s, 98s, "
               "ATo+, KJo+, QJo",
    "serree": "TT+, AJs+, KQs, AKo, AQo",
}

_EQUITY_MATRIX = None


def _equity_matrix():
    global _EQUITY_MATRIX
    if _EQUITY_MATRIX is None:
        _EQUITY_MATRIX = equity_matrix_169()
    return _EQUITY_MATRIX


_CARD_RE = re.compile(r"([2-9TJQKAtjqka]|10)\s*([shdcSHDC♠♥♦♣])")
_SUIT_ALIAS = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def parse_cards(text: str) -> list[int]:
    """Lit des cartes écrites librement → indices 0-51.

    Accepte « AhKd », « As Kd », « A♠ K♦ », « 10h Jc », majuscules ou non —
    tout ce qu'on recopie depuis une capture d'écran.

    >>> [c // 4 for c in parse_cards("Ah Kd")]
    [0, 1]
    """
    out: list[int] = []
    for rank, suit in _CARD_RE.findall(text or ""):
        r = "T" if rank == "10" else rank.upper()
        s = _SUIT_ALIAS.get(suit, suit.lower())
        out.append(RANKS.index(r) * 4 + SUITS.index(s))
    if len(set(out)) != len(out):
        raise ValueError(f"carte en double dans « {text} ».")
    return out


@dataclass(slots=True)
class Spot:
    """Un spot de poker tel qu'une capture d'écran le donne.

    Parameters
    ----------
    hero : str
        Tes deux cartes (« Ah Kd », « A♠K♦ »…).
    board : str
        Les cartes communes, vide en préflop.
    pot : float
        Pot AVANT la mise adverse, en jetons ou en bb (même unité partout).
    bet : float
        Mise à payer (0 si personne n'a misé).
    stack : float
        Ton tapis effectif (min des deux tapis en jeu).
    big_blind : float
        Valeur de la big blind, pour convertir en bb.
    position : str
        « BTN », « SB », « BB », « CO », « MP », « UTG », ou « ip »/« oop ».
    villain : str
        Range adverse supposée : « large », « moyenne », « serree », ou une
        range explicite (« TT+, AQs+ »).
    players : int
        Joueurs encore en jeu (2 = heads-up).
    """

    hero: str
    board: str = ""
    pot: float = 0.0
    bet: float = 0.0
    stack: float = 0.0
    big_blind: float = 1.0
    position: str = "BTN"
    villain: str = "moyenne"
    players: int = 2


@dataclass(slots=True)
class Advice:
    """Verdict du conseiller sur un spot."""

    action: str                # « JAM », « CALL », « FOLD », « MISER »…
    confidence: str            # « certain » (solveur) | « indicatif » (hypothèse)
    regime: str                # quel moteur a tranché
    equity: float | None = None
    required: float | None = None
    mdf: float | None = None
    ev_bb: float | None = None
    hand: str = ""
    reasons: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            "──────────────────────────────────────────────────────",
            f"  {self.hand}  →  {self.action}",
            f"  ({self.regime} · {self.confidence})",
            "──────────────────────────────────────────────────────",
        ]
        if self.equity is not None:
            lines.append(f"  Ton équité          : {self.equity * 100:5.1f} %")
        if self.required is not None:
            lines.append(f"  Équité requise      : {self.required * 100:5.1f} %"
                         "   (cotes du pot)")
        if self.mdf is not None:
            lines.append(f"  MDF                 : {self.mdf * 100:5.1f} %")
        if self.ev_bb is not None:
            lines.append(f"  EV                  : {self.ev_bb:+5.2f} bb")
        if self.reasons:
            lines.append("")
            lines += [f"  · {r}" for r in self.reasons]
        if self.assumptions:
            lines.append("")
            lines.append("  Hypothèses (à ajuster si tu avais une lecture) :")
            lines += [f"    – {a}" for a in self.assumptions]
        lines.append("──────────────────────────────────────────────────────")
        return "\n".join(lines)


def _hand_label(cards: list[int]) -> str:
    g = int(COMBO_TO_GROUP[combo_index(cards[0], cards[1])])
    return group_name(g)


def _villain_range(spec: str):
    return parse_range(_DEFAULT_VILLAIN.get(spec, spec))


def _advise_preflop_short(spot: Spot, hero: list[int], eff_bb: float) -> Advice:
    """Tapis court heads-up : l'équilibre de Nash jam/fold tranche."""
    g = int(COMBO_TO_GROUP[combo_index(hero[0], hero[1])])
    sol = solve_hu_pushfold(eff_bb, sb=0.5, bb=1.0, equity=_equity_matrix())
    ev = float(sol.ev_jam_par_groupe[g])
    jam_freq = float(sol.jam_range[g])
    action = "JAM (tapis)" if ev >= 0 else "FOLD"
    reasons = [
        f"À {eff_bb:.1f} bb effectifs, le spot se réduit à pousser ou passer.",
        f"EV(jam) − EV(fold) = {ev:+.2f} bb à l'équilibre.",
    ]
    if abs(ev) < 0.05:
        reasons.append("Spot quasi indifférent : les deux options se valent, "
                       "l'erreur y coûte presque rien.")
    if 0.0 < jam_freq < 1.0:
        reasons.append(f"Main de frontière (jam {jam_freq * 100:.0f} % du temps "
                       "à l'équilibre).")
    return Advice(
        action=action,
        confidence="certain",
        regime=f"Nash push/fold {eff_bb:.1f} bb",
        ev_bb=ev,
        hand=_hand_label(hero),
        reasons=reasons,
        assumptions=["Heads-up, chipEV — près d'une bulle l'ICM resserre les jams."],
    )


def _advise_preflop_deep(spot: Spot, hero: list[int]) -> Advice:
    """Tapis profond : comparaison à la range d'ouverture de la position."""
    pos = spot.position.upper()
    preset = GTO_PRESETS.get(pos)
    label = _hand_label(hero)
    if preset is None:
        return Advice(
            action="—",
            confidence="indicatif",
            regime="pas de chart pour cette position",
            hand=label,
            reasons=[f"Position « {spot.position} » inconnue ; "
                     f"positions couvertes : {', '.join(GTO_PRESETS)}."],
        )
    rng = parse_range(preset)
    weight = float(rng.weights[combo_index(hero[0], hero[1])])
    if weight >= 0.99:
        action, why = "OUVRIR (relance)", "Dans la range d'ouverture standard."
    elif weight <= 0.01:
        action, why = "FOLD", "Hors de la range d'ouverture standard."
    else:
        action = f"MIXTE — ouvrir {weight * 100:.0f} % du temps"
        why = "Main de frontière : le solveur la joue en fréquence mixte."
    return Advice(
        action=action,
        confidence="indicatif",
        regime=f"chart d'ouverture {pos}",
        hand=label,
        reasons=[why],
        assumptions=[
            "Chart 6-max 100 bb sans ante, approximative (à recalibrer sur "
            "tes propres solves) — et suppose que personne n'a ouvert devant.",
        ],
    )


def _advise_postflop(spot: Spot, hero: list[int], board: list[int]) -> Advice:
    """Postflop : équité exacte contre une range plausible vs cotes du pot."""
    villain = _villain_range(spot.villain)
    eq = equity_vs_range(hero, villain, board).equity
    label = _hand_label(hero)
    street = {3: "flop", 4: "turn", 5: "river"}[len(board)]
    assumptions = [
        f"Range adverse supposée « {spot.villain} » — c'est une hypothèse, "
        "pas une lecture.",
    ]

    if spot.bet <= 0:
        # personne n'a misé : miser pour la valeur au-dessus de la parité
        action = "MISER (valeur)" if eq >= 0.55 else (
            "CHECK" if eq >= 0.40 else "CHECK (ou bluff choisi)")
        reasons = [
            f"Au {street}, ta main réalise {eq * 100:.1f} % d'équité contre "
            "cette range.",
            "Personne n'a misé : la question est value ou contrôle du pot.",
        ]
        if eq >= 0.55:
            reasons.append("Au-dessus de 55 %, une mise de valeur est payée "
                           "par assez de mains plus faibles.")
        return Advice(action=action, confidence="indicatif",
                      regime=f"équité exacte au {street}", equity=eq,
                      hand=label, reasons=reasons, assumptions=assumptions)

    alpha = required_equity(spot.pot, spot.bet)
    mdf = minimum_defence_frequency(spot.pot, spot.bet)
    margin = eq - alpha
    ev_call = eq * (spot.pot + spot.bet) - (1 - eq) * spot.bet
    if spot.big_blind > 0:
        ev_call /= spot.big_blind

    if margin > 0.06:
        action = "CALL (confortable)"
    elif margin > 0.0:
        action = "CALL (juste)"
    elif margin > -0.04:
        action = "MARGINAL — proche de l'indifférence"
    else:
        action = "FOLD"

    reasons = [
        f"Au {street}, ta main vaut {eq * 100:.1f} % contre cette range ; "
        f"il en faut {alpha * 100:.1f} % pour payer {spot.bet:.0f} dans "
        f"{spot.pot:.0f}.",
        f"Marge : {margin * 100:+.1f} points d'équité.",
    ]
    # seuil de bascule : la décision tient-elle si la range change ?
    reasons.append(
        f"Bascule à {alpha * 100:.1f} % d'équité — si tu le crois plus "
        f"{'serré' if margin > 0 else 'large'} que supposé, la décision "
        "s'inverse."
    )
    if eq >= 0.75:
        reasons.append("Assez fort pour relancer plutôt que simplement payer.")
    return Advice(
        action=action, confidence="indicatif",
        regime=f"équité exacte au {street} vs cotes du pot",
        equity=eq, required=alpha, mdf=mdf, ev_bb=ev_call,
        hand=label, reasons=reasons, assumptions=assumptions,
    )


def advise(spot: Spot) -> Advice:
    """Rend le verdict sur un spot déjà joué : que fallait-il faire ?

    Examples
    --------
    >>> a = advise(Spot(hero="Ah Ad", stack=10, big_blind=1, position="BTN"))
    >>> a.action.startswith("JAM")
    True
    """
    hero = parse_cards(spot.hero)
    if len(hero) != 2:
        raise ValueError(f"il faut exactement 2 cartes pour le héros, "
                         f"reçu {len(hero)} depuis « {spot.hero} ».")
    board = parse_cards(spot.board) if spot.board else []
    if len(board) not in (0, 3, 4, 5):
        raise ValueError(f"board de 0, 3, 4 ou 5 cartes, reçu {len(board)}.")
    if set(hero) & set(board):
        raise ValueError("une carte du board est aussi dans ta main.")

    if board:
        return _advise_postflop(spot, hero, board)

    eff_bb = spot.stack / spot.big_blind if spot.big_blind > 0 else 0.0
    if spot.players == 2 and 1.0 <= eff_bb <= MAX_PUSHFOLD_BB:
        return _advise_preflop_short(spot, hero, eff_bb)
    return _advise_preflop_deep(spot, hero)
