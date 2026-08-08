"""
F6 — Compression d'une range en heuristiques mémorisables (Information Bottleneck).

Sources
-------
- Tishby, N., Pereira, F. & Bialek, W. (1999), *The Information Bottleneck Method*
- Strouse, D. & Schwab, D. (2017), *The Deterministic Information Bottleneck*,
  Neural Computation 29(6)

C'est la fusion la plus sous-estimée du projet, parce qu'elle attaque le vrai
goulot d'étranglement : **ta mémoire de travail, pas ton CPU.**

Problème : 6 positions × 5 situations × 169 combos = plus de 5 000 décisions,
dont beaucoup sont des stratégies mixtes non mémorisables.

Formulation :

.. math::
    \\min_{p(t|r)} \\; I(T;R) - \\beta\\,I(T;A)

- I(T;R) = complexité de la représentation (nombre de règles à retenir)
- I(T;A) = fidélité stratégique conservée
- β = curseur compression ↔ fidélité

On utilise la variante **déterministe** (DIB), qui produit des partitions
dures — donc des règles énonçables — là où l'IB classique produit des clusters
souples inexploitables par un humain.

La contrainte supplémentaire, essentielle et absente de la littérature
standard : les partitions sont restreintes à des **prédicats poker
interprétables** (paires, suited connectors, broadway, as suités…). Sans cette
contrainte, rien ne garantit que les clusters correspondent à des concepts
nommables — et une règle non nommable n'est pas une règle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import (
    GROUP_CLASSES,
    GROUP_COMBO_COUNT,
    N_GROUPS,
    HandClass,
    Range,
    group_name,
)

__all__ = [
    "Predicate",
    "POKER_PREDICATES",
    "Rule",
    "RuleSet",
    "InformationPlanePoint",
    "compress_range",
    "information_plane",
    "elbow_point",
]

F64 = npt.NDArray[np.float64]


class BottleneckError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# PRÉDICATS INTERPRÉTABLES — l'espace de partitions admissibles
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Predicate:
    """Un test énonçable en français sur un groupe de la grille 13×13."""

    name: str
    test: Callable[[int], bool]

    def mask(self) -> np.ndarray:
        return np.array([self.test(g) for g in range(N_GROUPS)], dtype=bool)


def _has(cls: HandClass) -> Callable[[int], bool]:
    return lambda g: cls in GROUP_CLASSES[g]


def _pair_at_least(rank: int) -> Callable[[int], bool]:
    def f(g: int) -> bool:
        r, c = divmod(g, 13)
        return r == c and r <= rank
    return f


def _suited_high_at_most(rank: int) -> Callable[[int], bool]:
    def f(g: int) -> bool:
        r, c = divmod(g, 13)
        return r < c and r <= rank
    return f


def _offsuit_high_at_most(rank: int) -> Callable[[int], bool]:
    def f(g: int) -> bool:
        r, c = divmod(g, 13)
        return r > c and c <= rank
    return f


def _gap_at_most(gap: int, suited: bool) -> Callable[[int], bool]:
    def f(g: int) -> bool:
        r, c = divmod(g, 13)
        if r == c:
            return False
        if (r < c) != suited:
            return False
        hi, lo = (r, c) if r < c else (c, r)
        return (lo - hi) <= gap
    return f


def _staircase(suited: bool, hi_max: int, lo_max: int) -> Callable[[int], bool]:
    """« suité/offsuit, carte haute ≥ H et carte basse ≥ L » — la forme d'escalier
    que dessinent réellement les charts préflop."""

    def f(g: int) -> bool:
        r, c = divmod(g, 13)
        if r == c:
            return False
        if (r < c) != suited:
            return False
        hi, lo = (r, c) if r < c else (c, r)
        return hi <= hi_max and lo <= lo_max

    return f


def _build_predicates() -> tuple[Predicate, ...]:
    """Espace de partitions admissibles : uniquement des prédicats énonçables.

    Trois familles, toutes lisibles à voix haute :
      1. seuils de paire        « paire ≥ 77 »
      2. escaliers suited/offsuit « suité, carte haute ≥ T et carte basse ≥ 7 »
      3. concepts nommés         « connecteur suité », « as suité », « broadway »

    L'escalier est la famille décisive : c'est littéralement la forme d'une
    range préflop sur la grille 13×13, donc celle qui compresse le mieux.
    """
    from pfs.core.range_model import RANKS

    preds: list[Predicate] = [Predicate("toute paire", _has(HandClass.PAIR))]
    for r in range(1, 13):
        preds.append(Predicate(f"paire ≥ {RANKS[r]}{RANKS[r]}", _pair_at_least(r)))

    for suited in (True, False):
        label = "suité" if suited else "offsuit"
        for hi in range(0, 12):
            for lo in range(hi + 1, 13):
                preds.append(
                    Predicate(
                        f"{label}, ≥{RANKS[hi]} et ≥{RANKS[lo]}",
                        _staircase(suited, hi, lo),
                    )
                )

    preds += [
        Predicate("as suité", _has(HandClass.SUITED_ACE)),
        Predicate("broadway", _has(HandClass.BROADWAY)),
        Predicate("connecteur suité", _has(HandClass.SUITED_CONNECTOR)),
        Predicate("gapper suité", _has(HandClass.SUITED_GAPPER)),
        Predicate("suité, gap ≤ 2", _gap_at_most(2, True)),
        Predicate("offsuit, gap ≤ 1", _gap_at_most(1, False)),
        Predicate("suité", _has(HandClass.SUITED)),
        Predicate("offsuit", _has(HandClass.OFFSUIT)),
    ]
    return tuple(preds)


POKER_PREDICATES: tuple[Predicate, ...] = _build_predicates()


# ═══════════════════════════════════════════════════════════════════════════
# RÈGLES ET JEUX DE RÈGLES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Rule:
    """« SI <prédicat> ALORS jouer à <fréquence> »."""

    predicate_name: str
    frequency: float
    coverage: float
    """Part du poids total de la range couverte par cette règle."""
    n_groups: int

    def __str__(self) -> str:
        return (
            f"SI {self.predicate_name:<28} → {self.frequency * 100:5.1f} %   "
            f"({self.n_groups:>3} groupes, {self.coverage * 100:4.1f} % de la range)"
        )


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Représentation compressée d'une range, plus ses métriques d'information."""

    rules: tuple[Rule, ...]
    default_frequency: float
    complexity_bits: float
    """I(T;R) — coût de la représentation."""
    fidelity_bits: float
    """I(T;A) — information conservée sur l'action optimale."""
    fidelity_ratio: float
    """I(T;A) / H(A) ∈ [0,1] — part de la valeur stratégique préservée."""
    mae: float
    """Erreur absolue moyenne sur les fréquences, pondérée par les combos."""
    ev_loss_proxy: float
    """Proxy de perte d'EV : erreur quadratique pondérée."""

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def apply(self) -> F64:
        """Reconstruit les 169 fréquences depuis les règles (première qui matche)."""
        out = np.full(N_GROUPS, self.default_frequency, dtype=np.float64)
        assigned = np.zeros(N_GROUPS, dtype=bool)
        by_name = {p.name: p for p in POKER_PREDICATES}
        for rule in self.rules:
            m = by_name[rule.predicate_name].mask() & ~assigned
            out[m] = rule.frequency
            assigned |= m
        return out

    def explain(self) -> str:
        head = (
            f"{self.n_rules} règles · fidélité {self.fidelity_ratio * 100:.1f} % "
            f"· complexité {self.complexity_bits:.2f} bits · MAE {self.mae * 100:.1f} pts"
        )
        body = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(self.rules))
        tail = f"  SINON → {self.default_frequency * 100:.1f} %"
        return f"{head}\n{body}\n{tail}"


# ═══════════════════════════════════════════════════════════════════════════
# MESURES D'INFORMATION
# ═══════════════════════════════════════════════════════════════════════════


def _entropy(p: F64) -> float:
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def _action_entropy(freqs: F64, mass: F64) -> float:
    """H(A) où A ∈ {jouer, ne pas jouer}, pondéré par le nombre de combos."""
    total = mass.sum()
    if total <= 0:
        return 0.0
    p_play = float(np.sum(freqs * mass) / total)
    p = np.array([p_play, 1.0 - p_play])
    return _entropy(p)


def _mutual_information_TA(labels: np.ndarray, freqs: F64, mass: F64) -> float:
    """I(T;A) : information que la partition conserve sur l'action optimale."""
    total = mass.sum()
    if total <= 0:
        return 0.0
    h_a = _action_entropy(freqs, mass)
    h_a_given_t = 0.0
    for t in np.unique(labels):
        m = labels == t
        w = mass[m].sum()
        if w <= 0:
            continue
        p_play = float(np.sum(freqs[m] * mass[m]) / w)
        h_a_given_t += (w / total) * _entropy(np.array([p_play, 1.0 - p_play]))
    return float(max(0.0, h_a - h_a_given_t))


def _complexity_bits(labels: np.ndarray, mass: F64) -> float:
    """I(T;R) : entropie de la partition, pondérée par la masse de combos."""
    total = mass.sum()
    if total <= 0:
        return 0.0
    probs = np.array(
        [mass[labels == t].sum() / total for t in np.unique(labels)], dtype=np.float64
    )
    return _entropy(probs)


# ═══════════════════════════════════════════════════════════════════════════
# COMPRESSION GLOUTONNE SOUS CONTRAINTE D'INTERPRÉTABILITÉ
# ═══════════════════════════════════════════════════════════════════════════


def compress_range(
    target: Range | F64,
    n_rules: int = 12,
    predicates: Sequence[Predicate] = POKER_PREDICATES,
    round_to: float | None = 0.05,
    specificity: float = 0.75,
) -> RuleSet:
    """Compresse une range en ``n_rules`` règles IF-THEN interprétables.

    Algorithme : sélection gloutonne dans l'espace des prédicats poker, en
    maximisant à chaque étape la réduction de l'erreur quadratique pondérée —
    ce qui, sur une partition dure, revient à maximiser I(T;A) (variante
    déterministe de Strouse & Schwab, 2017).

    Chaque règle est évaluée **sur les groupes non encore couverts**, donc
    l'ordre des règles compte : c'est un arbre de décision dégénéré, lisible
    de haut en bas comme une liste de priorités.

    Parameters
    ----------
    round_to
        Arrondi des fréquences (0.05 = par pas de 5 %). Les solveurs le font
        aussi : une fréquence à 3 décimales n'est ni mémorisable ni utile.
    specificity
        Exposant s de la pénalité de couverture. 0 = glouton SSE pur (mauvais
        sur les ranges larges) ; 0,75 = calibré sur UTG/CO/BTN/SB.

    Examples
    --------
    >>> from pfs.core.range_model import parse_range, GTO_PRESETS
    >>> rs = compress_range(parse_range(GTO_PRESETS["BTN"]), n_rules=8)
    >>> rs.n_rules
    8
    """
    freqs = target.to_groups() if isinstance(target, Range) else np.asarray(
        target, dtype=np.float64
    ).ravel()
    if freqs.size != N_GROUPS:
        raise BottleneckError(f"attendu {N_GROUPS} fréquences.")
    if n_rules < 1:
        raise BottleneckError("n_rules doit être >= 1.")

    mass = GROUP_COMBO_COUNT.astype(np.float64)
    total = mass.sum()

    masks = {p.name: p.mask() for p in predicates}
    remaining = np.ones(N_GROUPS, dtype=bool)
    labels = np.full(N_GROUPS, -1, dtype=np.int64)
    rules: list[Rule] = []

    def _wmean(m: np.ndarray) -> float:
        w = mass[m].sum()
        return float(np.sum(freqs[m] * mass[m]) / w) if w > 0 else 0.0

    def _sse(m: np.ndarray, value: float) -> float:
        return float(np.sum(mass[m] * (freqs[m] - value) ** 2))

    for step in range(n_rules):
        if not remaining.any():
            break
        base_value = _wmean(remaining)
        base_sse = _sse(remaining, base_value)

        # Critère : réduction de SSE, pénalisée par la couverture.
        #
        #     score = ΔSSE / (masse_couverte / masse_totale)^s
        #
        # La pénalité de spécificité est indispensable. Sans elle (s = 0), le
        # glouton sort d'abord un gros fourre-tout (« tout l'offsuit à 15 % »)
        # qui verrouille ensuite tous les groupes qu'il contient — défaut
        # classique des listes de décision à correspondance-première. Mesuré
        # sur les 4 ranges de référence : s = 0 donne 59 % de fidélité sur BTN,
        # s = 0,75 en donne 91 %.
        best_name, best_gain, best_mask, best_val = None, 0.0, None, 0.0
        for name, mk in masks.items():
            if any(r.predicate_name == name for r in rules):
                continue
            sel = mk & remaining
            n_sel = int(sel.sum())
            if n_sel == 0 or n_sel == int(remaining.sum()):
                continue
            v_in = _wmean(sel)
            rest = remaining & ~sel
            v_out = _wmean(rest)
            mass_in = float(mass[sel].sum())
            if mass_in <= 0.0:
                continue
            delta_sse = base_sse - (_sse(sel, v_in) + _sse(rest, v_out))
            if delta_sse <= 0.0:
                continue
            score = delta_sse / (mass_in / total) ** specificity
            if score > best_gain:
                best_name, best_gain, best_mask, best_val = name, score, sel, v_in

        if best_name is None or best_mask is None:
            break

        value = best_val
        if round_to:
            value = round(value / round_to) * round_to
        rules.append(
            Rule(
                predicate_name=best_name,
                frequency=float(np.clip(value, 0.0, 1.0)),
                coverage=float(mass[best_mask].sum() / total),
                n_groups=int(best_mask.sum()),
            )
        )
        labels[best_mask] = step
        remaining &= ~best_mask

    default_value = _wmean(remaining) if remaining.any() else 0.0
    if round_to:
        default_value = round(default_value / round_to) * round_to
    labels[remaining] = len(rules)

    reconstructed = np.full(N_GROUPS, float(np.clip(default_value, 0.0, 1.0)))
    assigned = np.zeros(N_GROUPS, dtype=bool)
    for i, rule in enumerate(rules):
        m = masks[rule.predicate_name] & ~assigned
        reconstructed[m] = rule.frequency
        assigned |= m

    err = np.abs(reconstructed - freqs)
    mae = float(np.sum(err * mass) / total)
    ev_proxy = float(np.sum(mass * (reconstructed - freqs) ** 2) / total)

    h_a = _action_entropy(freqs, mass)
    i_ta = _mutual_information_TA(labels, freqs, mass)

    return RuleSet(
        rules=tuple(rules),
        default_frequency=float(np.clip(default_value, 0.0, 1.0)),
        complexity_bits=_complexity_bits(labels, mass),
        fidelity_bits=i_ta,
        fidelity_ratio=float(i_ta / h_a) if h_a > 0 else 1.0,
        mae=mae,
        ev_loss_proxy=ev_proxy,
    )


@dataclass(frozen=True, slots=True)
class InformationPlanePoint:
    n_rules: int
    complexity_bits: float
    fidelity_ratio: float
    mae: float


def information_plane(
    target: Range | F64, max_rules: int = 20
) -> tuple[InformationPlanePoint, ...]:
    """Trace la courbe I(T;A) vs I(T;R) — le « plan d'information » de Tishby.

    C'est cette courbe qui répond à la vraie question : **à partir de combien
    de règles ajouter une règle cesse de payer ?**
    """
    pts = []
    for k in range(1, max_rules + 1):
        rs = compress_range(target, n_rules=k)
        pts.append(
            InformationPlanePoint(k, rs.complexity_bits, rs.fidelity_ratio, rs.mae)
        )
        if rs.n_rules < k:      # plus de prédicat utile disponible
            break
    return tuple(pts)


def elbow_point(
    points: Sequence[InformationPlanePoint], min_gain: float = 0.01
) -> InformationPlanePoint:
    """Coude de la courbe : dernière règle apportant ≥ ``min_gain`` de fidélité.

    Au-delà, les rendements sont décroissants et la charge mémoire augmente
    pour rien.
    """
    if not points:
        raise BottleneckError("aucun point.")
    best = points[0]
    for prev, cur in zip(points, points[1:]):
        if cur.fidelity_ratio - prev.fidelity_ratio >= min_gain:
            best = cur
        else:
            break
    return best
