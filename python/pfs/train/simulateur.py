"""Simulateur de main — tirage aléatoire réel, et ce qu'il fallait faire.

Trois usages, tous adossés aux solveurs du projet :

* **tirage réellement aléatoire** — la distribution vient de l'entropie du
  système (``secrets``), pas d'un générateur semé par l'horloge. Une graine
  explicite reste possible pour rejouer une main à l'identique, mais elle
  n'est jamais posée par défaut : un simulateur d'entraînement dont on
  devine la suite n'entraîne rien ;
* **mains adverses cachées ou visibles** — caché, c'est la condition de
  table : on décide contre une *range*. Visible, c'est la condition
  d'étude : on voit l'équité EXACTE contre les cartes réelles, et donc de
  combien la lecture change la décision. L'écart entre les deux est ce
  qu'on vient chercher ;
* **grille des compositions** — la même main jugée à plusieurs profondeurs
  de tapis et à plusieurs positions. En tapis court, chaque verdict est
  l'équilibre de Nash, donc *certain* ; le seuil où la décision bascule est
  ce qui se mémorise.

Honnêteté (NEMESIS) : ce simulateur distribue et juge, il ne joue pas la
main jusqu'au bout. Les verdicts postflop supposent une range adverse, et
le disent. Les verdicts préflop en tapis court, eux, ne supposent rien.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pfs.analysis.spot_advisor import Spot, advise
from pfs.core.equity import equity_vs_range
from pfs.core.range_model import (
    COMBO_TO_GROUP,
    RANKS,
    SUITS,
    Range,
    combo_index,
    group_name,
    parse_range,
)

__all__ = [
    "MainSimulee",
    "VerdictComposition",
    "RapportSimulation",
    "tirer_main",
    "simuler",
    "TAPIS_PAR_DEFAUT",
    "POSITIONS_PAR_DEFAUT",
]

# Profondeurs balayées par défaut : toute la zone où le push/fold tranche,
# c'est-à-dire là où le verdict est une vérité de théorie des jeux et non une
# hypothèse. Au-delà de 25 bb l'arbre de mise compte et le chart prend le
# relais — le simulateur le signale plutôt que de faire semblant.
TAPIS_PAR_DEFAUT: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)

POSITIONS_PAR_DEFAUT: tuple[str, ...] = ("BTN", "SB", "CO", "MP", "UTG")

_DECK: tuple[int, ...] = tuple(range(52))


def _nom(carte: int) -> str:
    return f"{RANKS[carte // 4]}{SUITS[carte % 4]}"


def _groupe(cartes: Sequence[str]) -> str:
    a = RANKS.index(cartes[0][0]) * 4 + SUITS.index(cartes[0][1])
    b = RANKS.index(cartes[1][0]) * 4 + SUITS.index(cartes[1][1])
    return group_name(int(COMBO_TO_GROUP[combo_index(a, b)]))


def _combo_range(cartes: Sequence[str]) -> Range:
    """Range dégénérée sur un seul combo — les cartes adverses connues."""
    w = np.zeros(1326, dtype=np.float64)
    a = RANKS.index(cartes[0][0]) * 4 + SUITS.index(cartes[0][1])
    b = RANKS.index(cartes[1][0]) * 4 + SUITS.index(cartes[1][1])
    w[combo_index(a, b)] = 1.0
    return Range(w)


@dataclass(slots=True)
class MainSimulee:
    """Une donne : ta main, celles des adversaires, le board éventuel."""

    hero: tuple[str, str]
    villains: tuple[tuple[str, str], ...] = ()
    board: tuple[str, ...] = ()

    @property
    def groupe(self) -> str:
        """Nom de la main dans la grille 169 : « AA », « A5s », « 72o »."""
        return _groupe(self.hero)

    def __str__(self) -> str:
        b = " ".join(self.board) if self.board else "—"
        return (f"{' '.join(self.hero)} ({self.groupe})   board {b}   "
                f"{len(self.villains)} adversaire(s)")


@dataclass(slots=True)
class VerdictComposition:
    """Ce qu'il fallait faire, pour une composition donnée."""

    tapis_bb: float
    joueurs: int
    position: str
    action: str
    certain: bool
    ev_bb: float | None = None
    equite: float | None = None
    requise: float | None = None


@dataclass(slots=True)
class RapportSimulation:
    """La donne, les verdicts par composition, et l'écart de lecture."""

    main: MainSimulee
    cartes_visibles: bool
    verdicts: list[VerdictComposition] = field(default_factory=list)
    equite_reelle: float | None = None
    equite_supposee: float | None = None
    range_supposee: str = ""

    @property
    def bascule_bb(self) -> float | None:
        """Profondeur où le verdict préflop change — le seuil à mémoriser."""
        pousse = [v for v in self.verdicts if v.position == "BTN"]
        pousse.sort(key=lambda v: v.tapis_bb)
        for prec, suiv in zip(pousse, pousse[1:]):
            if prec.action.split()[0] != suiv.action.split()[0]:
                return suiv.tapis_bb
        return None

    def explain(self) -> str:
        lignes = [
            "══════════════════════════════════════════════════════════",
            "  SIMULATION — tirage aléatoire, et ce qu'il fallait faire",
            "══════════════════════════════════════════════════════════",
            f"  Ta main   : {' '.join(self.main.hero)}  ({self.main.groupe})",
        ]
        if self.main.board:
            lignes.append(f"  Board     : {' '.join(self.main.board)}")
        if self.cartes_visibles and self.main.villains:
            adv = " · ".join(" ".join(v) for v in self.main.villains)
            lignes.append(f"  Adversaire: {adv}   (cartes VISIBLES)")
        elif self.main.villains:
            lignes.append(f"  Adversaire: {len(self.main.villains)} main(s) "
                          "CACHÉE(S) — comme à la table")

        if self.equite_reelle is not None:
            lignes.append("")
            lignes.append(f"  Équité EXACTE contre ses cartes : "
                          f"{self.equite_reelle * 100:.1f} %")
            if self.equite_supposee is not None:
                ecart = (self.equite_reelle - self.equite_supposee) * 100
                lignes.append(f"  Équité contre la range supposée « "
                              f"{self.range_supposee} » : "
                              f"{self.equite_supposee * 100:.1f} %")
                lignes.append(f"  Écart de lecture : {ecart:+.1f} points — "
                              "c'est ce que « voir ses cartes » t'apporte.")

        if self.verdicts:
            lignes += ["", "  Ce qu'il fallait faire, selon la composition :", ""]
            par_pos: dict[str, list[VerdictComposition]] = {}
            for v in self.verdicts:
                par_pos.setdefault(v.position, []).append(v)
            for pos, lst in par_pos.items():
                lst.sort(key=lambda v: v.tapis_bb)
                lignes.append(f"    {pos} ({lst[0].joueurs} joueurs)")
                for v in lst:
                    marque = "certain" if v.certain else "indicatif"
                    ev = f"  EV {v.ev_bb:+.2f} bb" if v.ev_bb is not None else ""
                    lignes.append(f"      {v.tapis_bb:5.1f} bb → "
                                  f"{v.action:<26} [{marque}]{ev}")
            b = self.bascule_bb
            if b is not None:
                lignes.append("")
                lignes.append(f"  ⚑ Bascule au bouton entre {b - 2:.0f} et "
                              f"{b:.0f} bb — c'est LE seuil à retenir.")
        lignes.append("══════════════════════════════════════════════════════════")
        return "\n".join(lignes)


def tirer_main(joueurs: int = 2, cartes_board: int = 0,
               graine: int | None = None) -> MainSimulee:
    """Distribue une donne aléatoire.

    Parameters
    ----------
    joueurs : int
        Nombre total de joueurs assis, héros compris (2 à 9).
    cartes_board : int
        0 (préflop), 3, 4 ou 5.
    graine : int, optionnel
        Pour rejouer une donne à l'identique. **Par défaut aucune** : le
        tirage vient de l'entropie du système, sinon l'entraînement se
        réduirait à mémoriser une suite.

    Returns
    -------
    MainSimulee
    """
    if not 2 <= joueurs <= 9:
        raise ValueError("joueurs doit être entre 2 et 9.")
    if cartes_board not in (0, 3, 4, 5):
        raise ValueError("cartes_board ∈ {0, 3, 4, 5}.")

    if graine is None:
        rng = np.random.default_rng(secrets.randbits(128))
    else:
        rng = np.random.default_rng(graine)
    paquet = rng.permutation(np.array(_DECK, dtype=np.int64))

    besoin = 2 * joueurs + cartes_board
    tirees = [int(x) for x in paquet[:besoin]]
    hero = (_nom(tirees[0]), _nom(tirees[1]))
    villains = tuple(
        (_nom(tirees[2 + 2 * i]), _nom(tirees[3 + 2 * i]))
        for i in range(joueurs - 1)
    )
    board = tuple(_nom(x) for x in tirees[2 * joueurs:besoin])
    return MainSimulee(hero=hero, villains=villains, board=board)


def simuler(
    main: MainSimulee | None = None,
    *,
    joueurs: int = 2,
    cartes_board: int = 0,
    cartes_visibles: bool = False,
    tapis: Sequence[float] = TAPIS_PAR_DEFAUT,
    positions: Sequence[str] = ("BTN",),
    villain: str = "moyenne",
    pot: float = 1.5,
    bet: float = 0.0,
    graine: int | None = None,
) -> RapportSimulation:
    """Tire une main (ou reprend celle fournie) et dit quoi faire.

    Parameters
    ----------
    main : MainSimulee, optionnel
        Donne imposée ; sinon elle est tirée au hasard.
    cartes_visibles : bool
        Révèle les cartes adverses et calcule l'équité EXACTE contre elles,
        en plus de l'équité contre la range supposée. L'écart entre les
        deux mesure ce qu'une lecture parfaite apporterait.
    tapis, positions
        Compositions balayées.
    villain : str
        Range adverse supposée : « large », « moyenne », « serree », ou une
        range explicite.

    Returns
    -------
    RapportSimulation
    """
    if main is None:
        main = tirer_main(joueurs, cartes_board, graine)

    rapport = RapportSimulation(main=main, cartes_visibles=cartes_visibles,
                                range_supposee=villain)

    # Équités : exacte contre les cartes réelles si on les montre, et contre
    # la range supposée dans tous les cas — c'est leur ÉCART qui instruit.
    if main.villains:
        from pfs.core.range_model import RANKS as _R, SUITS as _S

        def idx(t: str) -> int:
            return _R.index(t[0]) * 4 + _S.index(t[1])

        hero_i = [idx(c) for c in main.hero]
        board_i = [idx(c) for c in main.board]
        try:
            sup = equity_vs_range(hero_i, parse_range(
                {"large": "22+, A2s+, K5s+, Q8s+, J8s+, T8s+, 97s+, 86s+, "
                          "75s+, 65s, A7o+, K9o+, QTo+, JTo",
                 "moyenne": "22+, A8s+, A5s-A2s, K9s+, QTs+, JTs, T9s, 98s, "
                            "ATo+, KJo+, QJo",
                 "serree": "TT+, AJs+, KQs, AKo, AQo"}.get(villain, villain)),
                board_i, n_sims=20000)
            rapport.equite_supposee = sup.equity
        except Exception:
            rapport.equite_supposee = None
        if cartes_visibles:
            try:
                exact = equity_vs_range(hero_i, _combo_range(main.villains[0]),
                                        board_i, n_sims=20000)
                rapport.equite_reelle = exact.equity
            except Exception:
                rapport.equite_reelle = None

    board_txt = " ".join(main.board)
    for pos in positions:
        for t in tapis:
            a = advise(Spot(hero=" ".join(main.hero), board=board_txt,
                            pot=pot, bet=bet, stack=t, big_blind=1.0,
                            position=pos, villain=villain,
                            players=len(main.villains) + 1))
            rapport.verdicts.append(VerdictComposition(
                tapis_bb=float(t), joueurs=len(main.villains) + 1,
                position=pos, action=a.action,
                certain=(a.confidence == "certain"),
                ev_bb=a.ev_bb, equite=a.equity, requise=a.required,
            ))
    return rapport
