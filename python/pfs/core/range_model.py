"""
Représentation canonique d'une range — le tissu conjonctif de toutes les fusions.

Une range vit à deux résolutions :
  - **1326 combos** — la vraie unité de calcul (C(52,2)), c'est là que vivent
    les blockers, l'équité et l'inférence bayésienne ;
  - **169 groupes** — la grille 13×13 d'affichage (paires, suited, offsuit),
    c'est là que vit la cognition humaine.

Toutes les métriques informationnelles (entropie, distance de Fisher, gain
d'information, Information Bottleneck) sont définies ici, de sorte que les
modules F1–F13 partagent une seule et même algèbre de range.

Conventions
-----------
Rangs : 0 = As … 12 = Deux  (ordre décroissant, comme la grille des solveurs)
Couleurs : 0 = ♠, 1 = ♥, 2 = ♦, 3 = ♣
Carte : ``rank * 4 + suit`` ∈ [0, 52)
Combo : index canonique dans l'énumération triée des 1326 paires
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import numpy.typing as npt

__all__ = [
    "RANKS", "SUITS", "N_CARDS", "N_COMBOS", "N_GROUPS",
    "card_str", "combo_index", "combo_cards", "group_index", "group_name",
    "COMBO_TO_GROUP", "GROUP_COMBO_COUNT",
    "HandClass", "Range", "parse_range", "GTO_PRESETS",
    "PREFLOP_POSITIONS", "opening_range", "opening_fraction",
]

F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]

RANKS: str = "AKQJT98765432"
SUITS: str = "shdc"
N_CARDS: int = 52
N_COMBOS: int = 1326
N_GROUPS: int = 169


class RangeError(ValueError):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# INDEXATION CANONIQUE
# ═══════════════════════════════════════════════════════════════════════════


def card_str(card: int) -> str:
    if not (0 <= card < N_CARDS):
        raise RangeError(f"carte {card} hors domaine.")
    return f"{RANKS[card // 4]}{SUITS[card % 4]}"


@lru_cache(maxsize=1)
def _combo_table() -> tuple[I32, dict[tuple[int, int], int]]:
    """Énumération canonique des 1326 combos, triée par (carte_haute, carte_basse)."""
    pairs: list[tuple[int, int]] = []
    for a in range(N_CARDS):
        for b in range(a + 1, N_CARDS):
            pairs.append((a, b))
    arr = np.array(pairs, dtype=np.int32)
    lookup = {p: i for i, p in enumerate(pairs)}
    return arr, lookup


def combo_index(card_a: int, card_b: int) -> int:
    _, lookup = _combo_table()
    key = (min(card_a, card_b), max(card_a, card_b))
    if key[0] == key[1]:
        raise RangeError("un combo exige deux cartes distinctes.")
    try:
        return lookup[key]
    except KeyError as exc:  # pragma: no cover
        raise RangeError(f"combo invalide : {key}") from exc


def combo_cards(combo: int) -> tuple[int, int]:
    arr, _ = _combo_table()
    if not (0 <= combo < N_COMBOS):
        raise RangeError(f"combo {combo} hors domaine.")
    return int(arr[combo, 0]), int(arr[combo, 1])


def group_index(hi_rank: int, lo_rank: int, suited: bool) -> int:
    """Indice dans la grille 13×13. Diagonale = paires, au-dessus = suited."""
    if hi_rank == lo_rank:
        return hi_rank * 13 + hi_rank
    hi, lo = min(hi_rank, lo_rank), max(hi_rank, lo_rank)
    return hi * 13 + lo if suited else lo * 13 + hi


def group_name(group: int) -> str:
    r, c = divmod(group, 13)
    if r == c:
        return RANKS[r] * 2
    if r < c:
        return f"{RANKS[r]}{RANKS[c]}s"
    return f"{RANKS[c]}{RANKS[r]}o"


@lru_cache(maxsize=1)
def _combo_to_group() -> I32:
    arr, _ = _combo_table()
    out = np.empty(N_COMBOS, dtype=np.int32)
    for i in range(N_COMBOS):
        a, b = int(arr[i, 0]), int(arr[i, 1])
        ra, sa = divmod(a, 4)
        rb, sb = divmod(b, 4)
        out[i] = group_index(ra, rb, suited=(sa == sb))
    return out


COMBO_TO_GROUP: I32 = _combo_to_group()

GROUP_COMBO_COUNT: I32 = np.bincount(COMBO_TO_GROUP, minlength=N_GROUPS).astype(np.int32)
"""6 pour les paires, 4 pour les suited, 12 pour les offsuit."""


# ═══════════════════════════════════════════════════════════════════════════
# CATÉGORIES DE MAINS (filtres du Range Explorer)
# ═══════════════════════════════════════════════════════════════════════════


class HandClass(str, Enum):
    PAIR = "pair"
    SUITED = "suited"
    OFFSUIT = "offsuit"
    PREMIUM = "premium"
    BROADWAY = "broadway"
    SUITED_CONNECTOR = "suited_connector"
    SUITED_GAPPER = "suited_gapper"
    SUITED_ACE = "suited_ace"
    SMALL_PAIR = "small_pair"
    TRASH = "trash"


def _group_classes(g: int) -> frozenset[HandClass]:
    r, c = divmod(g, 13)
    out: set[HandClass] = set()
    if r == c:
        out.add(HandClass.PAIR)
        if r >= 8:
            out.add(HandClass.SMALL_PAIR)
        if r <= 3:
            out.add(HandClass.PREMIUM)
        return frozenset(out)

    suited = r < c
    hi, lo = (r, c) if suited else (c, r)
    out.add(HandClass.SUITED if suited else HandClass.OFFSUIT)
    gap = lo - hi

    if hi <= 3 and lo <= 4:
        out.add(HandClass.BROADWAY)
    if hi == 0 and suited:
        out.add(HandClass.SUITED_ACE)
    if suited and gap == 1:
        out.add(HandClass.SUITED_CONNECTOR)
    if suited and gap in (2, 3):
        out.add(HandClass.SUITED_GAPPER)
    if (hi == 0 and lo <= 2) or (hi <= 1 and lo <= 2):
        out.add(HandClass.PREMIUM)
    if not suited and gap >= 4 and hi >= 4:
        out.add(HandClass.TRASH)
    return frozenset(out)


GROUP_CLASSES: tuple[frozenset[HandClass], ...] = tuple(
    _group_classes(g) for g in range(N_GROUPS)
)


# ═══════════════════════════════════════════════════════════════════════════
# LA RANGE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class Range:
    """Poids ∈ [0,1] sur les 1326 combos.

    Toutes les fusions consomment ou produisent cet objet :
      - F1/F3 le mettent à jour (inférence bayésienne, filtre particulaire)
      - F4 en calcule l'entropie et le gain d'information par sizing
      - F5 en mesure la distance de Fisher à une référence GTO
      - F6 le compresse en règles mémorisables
      - F8/F13 en dérivent la stratégie
    """

    weights: F64

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64).ravel()
        if w.size != N_COMBOS:
            raise RangeError(f"attendu {N_COMBOS} poids, reçu {w.size}.")
        if np.any(w < -1e-12) or np.any(w > 1.0 + 1e-12):
            raise RangeError("les poids doivent être dans [0, 1].")
        self.weights = np.clip(w, 0.0, 1.0)

    # ── constructeurs ────────────────────────────────────────────────────
    @classmethod
    def empty(cls) -> "Range":
        return cls(np.zeros(N_COMBOS))

    @classmethod
    def full(cls) -> "Range":
        return cls(np.ones(N_COMBOS))

    @classmethod
    def from_groups(cls, group_weights: Mapping[str, float] | F64) -> "Range":
        """Depuis la grille 169 — chaque combo hérite du poids de son groupe."""
        g = np.zeros(N_GROUPS, dtype=np.float64)
        if isinstance(group_weights, Mapping):
            name_to_idx = {group_name(i): i for i in range(N_GROUPS)}
            for name, w in group_weights.items():
                if name not in name_to_idx:
                    raise RangeError(f"groupe inconnu : {name!r}")
                g[name_to_idx[name]] = float(w)
        else:
            arr = np.asarray(group_weights, dtype=np.float64).ravel()
            if arr.size != N_GROUPS:
                raise RangeError(f"attendu {N_GROUPS} groupes, reçu {arr.size}.")
            g = arr
        return cls(g[COMBO_TO_GROUP])

    # ── vues ─────────────────────────────────────────────────────────────
    def to_groups(self) -> F64:
        """Agrège en 169 groupes (moyenne pondérée par le nombre de combos)."""
        acc = np.bincount(COMBO_TO_GROUP, weights=self.weights, minlength=N_GROUPS)
        return acc / np.maximum(GROUP_COMBO_COUNT, 1)

    @property
    def n_combos(self) -> float:
        """Nombre effectif de combos (somme des poids)."""
        return float(self.weights.sum())

    @property
    def fraction(self) -> float:
        """Part des mains jouées, dans [0,1]."""
        return self.n_combos / N_COMBOS

    def normalised(self) -> F64:
        """Distribution de probabilité sur les combos."""
        s = self.weights.sum()
        if s <= 0.0:
            raise RangeError("range vide : pas de distribution définie.")
        return self.weights / s

    # ── algèbre ──────────────────────────────────────────────────────────
    def __and__(self, other: "Range") -> "Range":
        return Range(np.minimum(self.weights, other.weights))

    def __or__(self, other: "Range") -> "Range":
        return Range(np.minimum(1.0, self.weights + other.weights))

    def __sub__(self, other: "Range") -> "Range":
        return Range(np.maximum(0.0, self.weights - other.weights))

    def scaled(self, factor: float) -> "Range":
        return Range(np.clip(self.weights * factor, 0.0, 1.0))

    def filtered(self, classes: Iterable[HandClass]) -> "Range":
        """Ne conserve que les combos appartenant à l'une des classes."""
        wanted = set(classes)
        mask = np.array(
            [bool(GROUP_CLASSES[g] & wanted) for g in range(N_GROUPS)], dtype=bool
        )
        return Range(self.weights * mask[COMBO_TO_GROUP])

    def remove_blockers(self, dead_cards: Sequence[int]) -> "Range":
        """Retire tous les combos utilisant une carte morte (board, cartes hero).

        C'est l'effet blocker, et il n'est correct qu'à la résolution 1326 :
        c'est précisément ce que la grille 169 ne peut pas représenter.
        """
        dead = set(int(c) for c in dead_cards)
        if not dead:
            return Range(self.weights.copy())
        arr, _ = _combo_table()
        mask = ~(np.isin(arr[:, 0], list(dead)) | np.isin(arr[:, 1], list(dead)))
        return Range(self.weights * mask)

    # ── métriques informationnelles ──────────────────────────────────────
    @property
    def entropy_bits(self) -> float:
        r""":math:`H(R) = -\sum p\log_2 p` sur les combos normalisés."""
        s = self.weights.sum()
        if s <= 0.0:
            return 0.0
        p = self.weights[self.weights > 0.0] / s
        return float(-np.sum(p * np.log2(p)))

    @property
    def max_entropy_bits(self) -> float:
        n = int(np.count_nonzero(self.weights))
        return float(math.log2(n)) if n > 1 else 0.0

    @property
    def normalised_entropy(self) -> float:
        """H / H_max ∈ [0,1]. Proche de 1 = range dispersée, proche de 0 = polarisée."""
        hm = self.max_entropy_bits
        return self.entropy_bits / hm if hm > 0 else 0.0

    def shape_label(self) -> str:
        r = self.normalised_entropy
        if r > 0.97:
            return "uniforme"
        if r > 0.90:
            return "équilibrée"
        if r > 0.75:
            return "condensée"
        return "polarisée"

    # ── avantages structurels ────────────────────────────────────────────
    def nut_advantage(self, equities: F64, threshold: float = 0.90) -> float:
        """Part du poids de range dans le top décile d'équité."""
        eq = np.asarray(equities, dtype=np.float64).ravel()
        if eq.size != N_COMBOS:
            raise RangeError("equities doit couvrir les 1326 combos.")
        s = self.weights.sum()
        if s <= 0.0:
            return 0.0
        return float(np.sum(self.weights * (eq >= threshold)) / s)

    def range_advantage(self, equities: F64) -> float:
        """Équité moyenne pondérée de la range."""
        eq = np.asarray(equities, dtype=np.float64).ravel()
        s = self.weights.sum()
        if s <= 0.0:
            return 0.0
        return float(np.sum(self.weights * eq) / s)

    def is_capped(self, equities: F64, nut_threshold: float = 0.95,
                  max_share: float = 0.02) -> bool:
        """Range « cappée » : quasiment aucune main premium."""
        return self.nut_advantage(equities, nut_threshold) < max_share

    # ── inférence bayésienne (F3/F4) ─────────────────────────────────────
    def bayes_update(self, action_likelihood: F64) -> "Range":
        r""":math:`r'(c) \propto r(c)\,P(a \mid c)`.

        C'est la brique élémentaire de la mise à jour de range street par
        street. Les poids restent dans [0,1] après renormalisation au max.
        """
        lik = np.asarray(action_likelihood, dtype=np.float64).ravel()
        if lik.size != N_COMBOS:
            raise RangeError("action_likelihood doit couvrir les 1326 combos.")
        if np.any(lik < 0.0):
            raise RangeError("vraisemblance négative.")
        post = self.weights * lik
        m = post.max()
        return Range(post / m if m > 0 else post)

    def information_gain(self, action_likelihood: F64) -> float:
        """Bits gagnés sur la range en observant une action binaire."""
        lik = np.asarray(action_likelihood, dtype=np.float64).ravel()
        s = self.weights.sum()
        if s <= 0.0:
            return 0.0
        r = self.weights / s
        p_yes = float(np.sum(r * lik))
        p_no = 1.0 - p_yes
        h_before = self.entropy_bits

        def _h(w: F64) -> float:
            t = w.sum()
            if t <= 0:
                return 0.0
            p = w[w > 0] / t
            return float(-np.sum(p * np.log2(p)))

        h_after = 0.0
        if p_yes > 1e-12:
            h_after += p_yes * _h(r * lik)
        if p_no > 1e-12:
            h_after += p_no * _h(r * (1.0 - lik))
        return float(max(0.0, h_before - h_after))

    # ── affichage ────────────────────────────────────────────────────────
    def grid_text(self, cell: str = "{:3.0f}") -> str:
        """Grille 13×13 en texte — pour le débogage et les tests de régression."""
        g = self.to_groups() * 100.0
        rows = ["    " + " ".join(f"{r:>3}" for r in RANKS)]
        for i, r in enumerate(RANKS):
            cells = " ".join(cell.format(g[i * 13 + j]) for j in range(13))
            rows.append(f"{r}   {cells}")
        return "\n".join(rows)

    def top_groups(self, k: int = 10) -> list[tuple[str, float]]:
        g = self.to_groups()
        idx = np.argsort(-g)[:k]
        return [(group_name(int(i)), float(g[i])) for i in idx if g[i] > 0]

    def __repr__(self) -> str:
        return (
            f"Range({self.n_combos:.1f}/1326 combos, {self.fraction * 100:.1f}%, "
            f"H={self.entropy_bits:.2f} bits, {self.shape_label()})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PARSING TEXTE (format standard des solveurs)
# ═══════════════════════════════════════════════════════════════════════════


def _expand_token(tok: str) -> list[int]:
    """« AA », « AKs », « 77+ », « A2s+ », « KQo », « T9s-65s » → indices de groupe."""
    tok = tok.strip()
    if not tok:
        return []
    rank_idx = {r: i for i, r in enumerate(RANKS)}

    if "-" in tok:
        lo_tok, hi_tok = tok.split("-", 1)
        a, b = _expand_token(lo_tok.strip()), _expand_token(hi_tok.strip())
        return sorted(set(a) | set(b))

    plus = tok.endswith("+")
    core = tok[:-1] if plus else tok

    if len(core) == 2 and core[0] == core[1]:
        r = rank_idx[core[0]]
        rs = range(0, r + 1) if plus else [r]
        return [group_index(i, i, True) for i in rs]

    if len(core) < 3:
        raise RangeError(f"token illisible : {tok!r}")
    hi, lo, kind = rank_idx[core[0]], rank_idx[core[1]], core[2].lower()
    if kind not in ("s", "o"):
        raise RangeError(f"suffixe inconnu dans {tok!r} (attendu 's' ou 'o').")
    suited = kind == "s"
    los = range(hi + 1, lo + 1) if plus else [lo]
    return [group_index(hi, l, suited) for l in los]


def parse_range(spec: str, weight: float = 1.0) -> Range:
    """Parse une range au format solveur : ``"77+, AJs+, KQs, A5s-A2s, AQo+"``.

    Supporte aussi les poids par token : ``"KQo:0.7"``.
    """
    g = np.zeros(N_GROUPS, dtype=np.float64)
    for raw in spec.replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        w = weight
        if ":" in raw:
            raw, w_txt = raw.split(":", 1)
            w = float(w_txt)
        if not (0.0 <= w <= 1.0):
            raise RangeError(f"poids hors [0,1] : {w}")
        for gi in _expand_token(raw):
            g[gi] = max(g[gi], w)
    return Range.from_groups(g)


GTO_PRESETS: dict[str, str] = {
    # Ouvertures 6-max 100bb sans ante — approximations de référence.
    #
    # Les fréquences mixtes de la frontière sont explicites (``:0.7``) : une
    # vraie solution de solveur n'est jamais binaire, et ce sont précisément
    # ces spots-là qui coûtent le plus cher et qu'aucun chart statique ne
    # transmet. Elles alimentent le mode « spots mixtes uniquement » du drill.
    #
    # ⚠️ À recalibrer sur tes propres solutions : ce sont des points de
    # comparaison, pas des vérités.
    "UTG": "88+, 77:0.5, ATs+, A5s:0.5, KTs+, QTs+, JTs, T9s:0.5, 98s:0.3, "
           "AQo+, AJo:0.5, KQo:0.3",
    "MP": "77+, 66:0.6, 55:0.3, A9s+, A5s-A2s:0.6, KTs+, K9s:0.5, QTs+, JTs, "
          "T9s, 98s:0.7, 87s:0.4, AJo+, ATo:0.5, KQo:0.7",
    "CO": "55+, 44:0.7, 33:0.4, A5s+, A4s-A2s:0.7, K9s+, K8s:0.5, Q9s+, Q8s:0.5, "
          "J9s+, J8s:0.6, T8s+, 97s+, 87s, 76s, 65s:0.6, "
          "ATo+, A9o:0.5, KJo+, KTo:0.6, QJo, QTo:0.4, JTo:0.4",
    "BTN": "22+, A2s+, K5s+, K4s-K2s:0.7, Q7s+, Q6s:0.5, J8s+, J7s:0.6, T7s+, "
           "T6s:0.5, 96s+, 95s:0.4, 85s+, 75s+, 64s+, 54s, 43s:0.4, "
           "A2o+, K9o+, K8o:0.5, Q9o+, Q8o:0.4, J9o+, T9o, 98o:0.7, 87o:0.4",
    "SB": "22+, A2s+, K2s+, Q4s+, Q3s-Q2s:0.6, J6s+, J5s:0.5, T6s+, T5s:0.5, "
          "95s+, 94s:0.4, 84s+, 74s+, 63s+, 53s+, 43s, "
          "A2o+, K5o+, K4o-K2o:0.5, Q8o+, Q7o:0.5, J8o+, J7o:0.4, T8o+, "
          "97o+, 87o, 76o:0.5",
}


# ═══════════════════════════════════════════════════════════════════════════
# ACCÈS CANONIQUE AUX RANGES D'OUVERTURE (ajout — conseiller de défense)
# ═══════════════════════════════════════════════════════════════════════════

PREFLOP_POSITIONS: tuple[str, ...] = ("UTG", "MP", "CO", "BTN", "SB", "BB")
"""Ordre de parole préflop à une table 6-max — la big blind parle en dernier.

C'est l'ordre qu'exploite le modèle de défense du conseiller : face à une
ouverture, l'ouvreur est nécessairement assis AVANT le héros dans cette
séquence (la BB n'ouvre jamais, elle est déjà servie).
"""


@lru_cache(maxsize=8)
def opening_range(position: str) -> Range:
    """Range d'ouverture de référence de ``position`` (:data:`GTO_PRESETS` parsée).

    Mémoïsée parce que les presets sont des constantes ; l'objet rendu est
    donc PARTAGÉ — copier ``weights`` avant toute modification en place.

    >>> opening_range("UTG").fraction < opening_range("BTN").fraction
    True
    >>> opening_range("XX")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RangeError: pas de chart d'ouverture pour « XX » ...
    """
    if position not in GTO_PRESETS:
        raise RangeError(f"pas de chart d'ouverture pour « {position} » "
                         f"(positions couvertes : {', '.join(GTO_PRESETS)}).")
    return parse_range(GTO_PRESETS[position])


def opening_fraction(position: str) -> float:
    """Part des mains que la chart de ``position`` ouvre (moyenne des 1326 poids).

    C'est exactement le chiffre « chart » que le banc Pluribus imprime en
    section 6 : la fréquence d'ouverture prescrite, fréquences mixtes
    comprises.

    >>> 0.05 < opening_fraction("UTG") < opening_fraction("SB") < 0.60
    True
    """
    return opening_range(position).fraction
