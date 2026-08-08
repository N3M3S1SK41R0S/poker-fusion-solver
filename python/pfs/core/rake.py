"""L8 — Modèle de rake : pourcentage plafonné, convention « dealt/won-pot ».

Sources
-------
- Barème public Winamax/PokerStars (cash game NLHE) : le prélèvement est un
  pourcentage du pot, plafonné par un *cap* dépendant de la limite et du
  nombre de joueurs — la forme ``rake = min(cap, pct × pot)`` est universelle
  dans l'industrie.
- Convention retenue : **« won-pot »** (rooms européennes) — le rake est
  prélevé sur le pot au moment où il est GAGNÉ, quel que soit le chemin
  (fold ou abattage). Les mises investies ne sont pas rakées en tant que
  telles : seul le pot encaissé l'est.
- Chen, B. & Ankenman, J. (2006), *The Mathematics of Poker* — le rake comme
  taxe sur le pot terminal : le jeu cesse d'être à somme constante,
  u_OOP + u_IP = pot − rake(pot terminal) à chaque feuille.

Le principe en une phrase : **le rake est une fonction concave affine par
morceaux du pot**, ``R(pot) = min(cap, pct·pot)`` — linéaire jusqu'au point
de bascule ``pot* = cap/pct``, constante au-delà. Toute la microéconomie du
rake (les caps qui « mordent » dans les gros pots, le pourcentage qui domine
dans les petits) tient dans ces deux paramètres.

Branchement solveur : ``pfs.solver.postflop.PostflopSolver(rake=...)`` —
le gagnant d'un terminal reçoit ``net(pot)`` au lieu de ``pot``, l'égalité
d'abattage partage ``net(pot)`` en deux. Voir la docstring du solveur pour
les conséquences sur la somme des utilités et l'exploitabilité.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "RakeError",
    "RakeModel",
    "NO_RAKE",
    "WINAMAX_MICRO",
]

MAX_PCT = 0.2
"""Borne supérieure de validation du pourcentage (20 %) — aucune room réelle
ne prélève davantage ; au-delà, c'est presque toujours une erreur d'unité
(5 passé au lieu de 0.05)."""


class RakeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RakeModel:
    r"""Rake « pourcentage plafonné » : :math:`R(pot) = \min(cap,\ pct \cdot pot)`.

    Le rake est prélevé sur le pot au moment où il est gagné (convention
    « dealt/won-pot » des rooms européennes) : le gagnant encaisse
    ``net(pot) = pot − take(pot)`` ; en cas d'égalité à l'abattage, c'est le
    pot NET qui est partagé. Les mises investies ne sont jamais rakées.

    Parameters
    ----------
    pct
        Pourcentage prélevé, dans ``[0, 0.2]`` (ex. ``0.05`` pour 5 %).
    cap
        Plafond du prélèvement, ``>= 0``, dans les MÊMES UNITÉS que le pot
        (bb si le pot est en bb, jetons sinon). ``cap = 0`` annule tout rake
        quel que soit ``pct`` ; ``cap = math.inf`` désactive le plafond.

    Examples
    --------
    Régime linéaire (le pourcentage domine), puis le cap qui mord :

    >>> r = RakeModel(pct=0.05, cap=1.0)
    >>> r.take(10.0)          # 5 % de 10 = 0.5 < cap
    0.5
    >>> r.take(30.0)          # 5 % de 30 = 1.5 → plafonné
    1.0
    >>> r.net(30.0)
    29.0

    Le point de bascule est à ``pot* = cap/pct = 20`` :

    >>> r.take(20.0)
    1.0
    """

    pct: float
    cap: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.pct <= MAX_PCT:
            raise RakeError(
                f"pct doit être dans [0, {MAX_PCT}] (reçu {self.pct!r}) — "
                f"5 % s'écrit 0.05, pas 5."
            )
        if not self.cap >= 0.0:
            raise RakeError(f"cap doit être >= 0 (reçu {self.cap!r}).")
        if math.isinf(self.pct):          # pragma: no cover — exclu par la borne
            raise RakeError("pct doit être fini.")

    def take(self, pot: float) -> float:
        r"""Rake prélevé sur un pot gagné : :math:`\min(cap,\ pct \cdot pot)`.

        Parameters
        ----------
        pot
            Pot terminal (mises adverses incluses), ``>= 0``.

        Returns
        -------
        float
            Montant prélevé, dans ``[0, min(cap, pct·pot)]``.

        Raises
        ------
        RakeError
            Si ``pot`` est négatif.
        """
        if pot < 0.0:
            raise RakeError(f"pot négatif ({pot!r}) : rien à raker.")
        return min(self.cap, self.pct * pot)

    def net(self, pot: float) -> float:
        """Pot net encaissé par le gagnant : ``pot − take(pot)``.

        À rake nul (``NO_RAKE``), retourne ``pot`` bit à bit : ``pot − 0.0``
        est exact en IEEE 754 — la non-régression du solveur en dépend.
        """
        return pot - self.take(pot)


NO_RAKE = RakeModel(0.0, 0.0)
"""Absence de rake : ``take ≡ 0``, ``net ≡ identité`` (cap 0 et pct 0)."""

WINAMAX_MICRO = RakeModel(0.05, 1.0)
"""Préréglage Winamax NL2–NL10 : ~5 % du pot, cap 0.5–1 bb PAR TRANCHE de
limite. Le cap est ici exprimé dans les unités du pot du spot (1.0 = 1 bb si
le pot est en bb) — à ajuster par l'utilisateur selon sa limite exacte et le
nombre de joueurs assis (le cap réel de la room varie avec les deux)."""
