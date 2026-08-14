r"""
L6 — Population mining privé : le prior des inconnus, tiré de TES mains.

Statut (audit du 14 août 2026) — en attente de données réelles, NON branché
---------------------------------------------------------------------------
Hors tests (``tests/test_population.py``), l'unique consommateur est
``pfs.analysis.inference_check`` — lui-même un banc non branché. Pourquoi :
dans l'application, les trackers F1 démarrent sur Jeffreys ; injecter un
prior de population suppose d'avoir miné un corpus de SES mains, et ce
corpus n'est pas encore constitué en volume utile (le plafond
``n0_eff = min(n0, n_pop/10)`` rend d'ailleurs le module quasi muet sous
~300 occasions par stat — assumé). Ce qui existe déjà : ingestion iPoker,
persistance SQLite locale, priors plafonnés, ``stake_band`` — testés.
Accroche naturelle si on branche : ``recuperer_mains.py`` pour l'ingestion,
puis ``pfs/engine.py:FusionEngine`` (créer les ``DynamicBetaTracker`` d'un
inconnu sur ``PopulationMiner.prior_for`` au lieu de Jeffreys).
Règle du projet : aucun module fusionné sans route + UI + test de bout en
bout.

Hand2Note vend l'analyse de population ; GTO Wizard vend des solutions
« vs population » agrégées mondialement. Ici : la même idée, locale, privée,
sur les hand histories de Pierre — les tendances du POOL qu'il affronte
réellement, par tranche d'enjeux.

Usage dans la chaîne de fusion : quand un joueur est inconnu (0 main), F1/F3
démarrent aujourd'hui sur Jeffreys (non-informatif). Avec ce module, ils
démarrent sur le postérieur DE LA POPULATION, pondéré en pseudo-observations —
c'est l'empirical Bayes de Robbins (1956) / Efron & Morris (1975), la même
famille que le rétrécissement de James-Stein déjà utilisé pour SharkScope.

Garanties :
- même règle que F1 : une stat n'est comptée QUE s'il y a eu occasion
  (les clés absentes de ``stat_observations`` ne créent aucun comptage) ;
- le héros est EXCLU par défaut (on modélise le pool, pas soi-même) ;
- la force du prior est plafonnée : ``n0_eff = min(n0, n_pop/10)`` — un pool
  de 40 observations ne fabrique pas un prior de 30 mains ;
- persistance SQLite locale, zéro réseau (même contrat que player_notes).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from pfs.data.hand_history import ParsedHand, iter_hands

__all__ = [
    "PopulationError", "PopulationMiner", "PopulationPrior", "stake_band",
]

JEFFREYS = 0.5
STATS = ("vpip", "pfr", "three_bet", "fold_to_cbet", "wtsd")


class PopulationError(ValueError):
    pass


def stake_band(big_blind: float, is_tournament: bool) -> str:
    """Tranche d'enjeux : « MTT » ou « NLx » (x = 100·bb, convention cash).

    >>> stake_band(0.02, False), stake_band(0.5, False), stake_band(100, True)
    ('NL2', 'NL50', 'MTT')
    """
    if is_tournament:
        return "MTT"
    if big_blind <= 0:
        raise PopulationError("big blind <= 0.")
    return f"NL{max(1, round(big_blind * 100))}"


@dataclass(frozen=True, slots=True)
class PopulationPrior:
    """Prior Beta prêt à injecter dans DynamicBetaTracker.

    ``alpha``/``beta`` incluent Jeffreys + n0_eff pseudo-observations au taux
    de la population. ``n_pop`` = occasions réellement observées dans le pool.
    """
    stat: str
    band: str
    alpha: float
    beta: float
    n_pop: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def strength(self) -> float:
        """Pseudo-observations au-delà de Jeffreys."""
        return self.alpha + self.beta - 2 * JEFFREYS


class PopulationMiner:
    """Agrège les hand histories en fréquences de population par tranche.

    Examples
    --------
    >>> m = PopulationMiner()
    >>> m._add("NL2", "vpip", True); m._add("NL2", "vpip", True)
    >>> m._add("NL2", "vpip", False); m._add("NL2", "vpip", False)
    >>> m.rate("NL2", "vpip")
    0.5
    """

    def __init__(self) -> None:
        # (band, stat) → [succès, occasions]
        self._counts: dict[tuple[str, str], list[int]] = {}
        self._hands = 0

    # ── ingestion ─────────────────────────────────────────────────────────

    def _add(self, band: str, stat: str, success: bool) -> None:
        c = self._counts.setdefault((band, stat), [0, 0])
        c[0] += int(success)
        c[1] += 1

    def ingest(self, hand: ParsedHand, exclude_hero: bool = True) -> int:
        """Compte toutes les occasions de tous les joueurs de la main."""
        band = stake_band(hand.big_blind, hand.is_tournament)
        n = 0
        for seat in hand.seats:
            if exclude_hero and seat.is_hero:
                continue
            for stat, success in hand.stat_observations(seat.player).items():
                self._add(band, stat, success)
                n += 1
        self._hands += 1
        return n

    def ingest_text(self, text: str, salt: str = "",
                    exclude_hero: bool = True) -> tuple[int, int]:
        """Parse un bloc de hand histories brut. → (mains, observations)."""
        hands = obs = 0
        for hand in iter_hands(text, salt=salt):
            obs += self.ingest(hand, exclude_hero=exclude_hero)
            hands += 1
        return hands, obs

    def ingest_files(self, paths: Iterable[str | Path], salt: str = "",
                     exclude_hero: bool = True) -> tuple[int, int]:
        hands = obs = 0
        for p in paths:
            h, o = self.ingest_text(Path(p).read_text(encoding="utf-8",
                                                      errors="replace"),
                                    salt=salt, exclude_hero=exclude_hero)
            hands += h
            obs += o
        return hands, obs

    # ── lecture ───────────────────────────────────────────────────────────

    @property
    def n_hands(self) -> int:
        return self._hands

    def occasions(self, band: str, stat: str) -> int:
        return self._counts.get((band, stat), [0, 0])[1]

    def rate(self, band: str, stat: str) -> float:
        k, n = self._counts.get((band, stat), [0, 0])
        if n == 0:
            raise PopulationError(f"aucune occasion pour {stat}@{band}.")
        return k / n

    def bands(self) -> list[str]:
        return sorted({b for b, _ in self._counts})

    def prior_for(self, band: str, stat: str,
                  n0: float = 30.0) -> PopulationPrior:
        """Prior Beta empirical-Bayes, force plafonnée par l'information réelle.

        ``n0`` = force cible en pseudo-mains ; effective :
        ``n0_eff = min(n0, n_pop / 10)`` — il faut 300 occasions de pool pour
        mériter un prior de 30. Sans données : Jeffreys pur (α=β=½).
        """
        if n0 < 0:
            raise PopulationError("n0 >= 0 requis.")
        k, n = self._counts.get((band, stat), [0, 0])
        if n == 0:
            return PopulationPrior(stat, band, JEFFREYS, JEFFREYS, 0)
        p_hat = (k + JEFFREYS) / (n + 2 * JEFFREYS)   # postérieur du pool
        n0_eff = min(n0, n / 10.0)
        return PopulationPrior(stat, band,
                               JEFFREYS + n0_eff * p_hat,
                               JEFFREYS + n0_eff * (1.0 - p_hat), n)

    def table(self) -> list[dict]:
        """Rapport lisible : taux + IC à 95 % (quantiles Beta) par cellule."""
        from scipy import stats as sps
        rows = []
        for (band, stat), (k, n) in sorted(self._counts.items()):
            a, b = k + JEFFREYS, n - k + JEFFREYS
            rows.append({
                "band": band, "stat": stat, "n": n,
                "rate": k / n,
                "ci_low": float(sps.beta.ppf(0.025, a, b)),
                "ci_high": float(sps.beta.ppf(0.975, a, b)),
            })
        return rows

    # ── persistance locale ────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        con = sqlite3.connect(str(path))
        try:
            con.execute("CREATE TABLE IF NOT EXISTS population "
                        "(band TEXT, stat TEXT, successes INTEGER, "
                        "occasions INTEGER, PRIMARY KEY (band, stat))")
            con.execute("CREATE TABLE IF NOT EXISTS meta "
                        "(key TEXT PRIMARY KEY, value TEXT)")
            con.execute("DELETE FROM population")
            con.executemany(
                "INSERT INTO population VALUES (?, ?, ?, ?)",
                [(b, s, c[0], c[1]) for (b, s), c in self._counts.items()])
            con.execute("INSERT OR REPLACE INTO meta VALUES ('n_hands', ?)",
                        (str(self._hands),))
            con.commit()
        finally:
            con.close()

    @classmethod
    def load(cls, path: str | Path) -> "PopulationMiner":
        m = cls()
        con = sqlite3.connect(str(path))
        try:
            for band, stat, k, n in con.execute(
                    "SELECT band, stat, successes, occasions FROM population"):
                m._counts[(band, stat)] = [int(k), int(n)]
            row = con.execute(
                "SELECT value FROM meta WHERE key='n_hands'").fetchone()
            m._hands = int(row[0]) if row else 0
        finally:
            con.close()
        return m
