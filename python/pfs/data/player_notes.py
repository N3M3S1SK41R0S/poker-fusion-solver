"""
Base locale de profils adverses — enrichissement hors ligne, lookup en direct.

═══════════════════════════════════════════════════════════════════════════
LE PATTERN
═══════════════════════════════════════════════════════════════════════════

C'est exactement l'architecture du blueprint, appliquée aux adversaires :

    HORS LIGNE (quand tu veux, sans contrainte de temps)
        tu consultes SharkScope / OPR toi-même, ou tu exportes un CSV
                            │
                            ▼
        import dans la base locale, chiffrée, indexée par pseudo haché
                            │
    ═══════════════════════ │ ═══════════════════════════════════════════
                            ▼
    EN DIRECT (contrainte : < 1 ms, zéro réseau)
        la vision lit le pseudo à l'écran → hash → lookup local O(1)
                            │
                            ▼
        prior d'archétype (F3) + propension à s'adapter ρ (F13)

Le pseudo lu à l'écran sert de **clé de lookup**, jamais de requête sortante.

═══════════════════════════════════════════════════════════════════════════
LES DEUX VERROUS DE L'ENRICHISSEMENT
═══════════════════════════════════════════════════════════════════════════

Un enrichissement automatique naïf — lire les pseudos à l'écran et interroger
une base tierce pendant la main — se heurte à deux objections sérieuses :

* **PokerStars** limite la collecte d'informations sur les joueurs aux mains
  auxquelles tu participes.
* La **traçabilité** côté opérateur repose sur la corrélation temporelle entre
  un board distribué et une requête sortante — c'est le mécanisme du « Fair
  Play Check » de GTO Wizard, qui joint les deux logs.

:class:`EnrichmentQueue` répond aux deux, littéralement :

  **Verrou (a) — hors main.** Rien ne part tant qu'une main est vivante à une
  table quelconque. La corrélation temporelle est cassée à la source.

  **Verrou (b) — participation.** Un pseudo n'entre dans la file que si ce
  joueur a été dans un pot avec toi. Un simple spectateur, ou un joueur d'une
  table observée, est refusé — et le refus est tracé.

S'y ajoutent une cadence minimale entre deux traitements (pour éviter une
rafale reconnaissable) et un **journal d'audit local** de tout ce qui a été
tenté, accepté ou refusé.

Le noyau reste **sans aucun accès réseau**, vérifié par un test. Brancher un
fournisseur de ratings est un acte explicite : tu implémentes le protocole
:class:`RatingProvider` avec ta propre clé d'abonnement. La file applique les
verrous quel que soit le fournisseur — y compris le tien.

═══════════════════════════════════════════════════════════════════════════
CE QUE LE PRIOR PÈSE RÉELLEMENT
═══════════════════════════════════════════════════════════════════════════

Un prior externe est **écrasé par les observations en une cinquantaine de
mains** — c'est la propriété recherchée, pas un défaut. Il sert à ne pas
partir de zéro sur un inconnu, pas à décider. La méthode
:meth:`PlayerNotes.prior_weight` calcule explicitement son poids résiduel.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from pfs.data.hand_history import player_key
from pfs.fusion.particle import ARCHETYPE_PRIORS, Archetype
from pfs.fusion.skill_prior import (
    ExternalRating,
    GameFormat,
    RatingSource,
    SkillEstimate,
    adaptation_propensity_from_skill,
    archetype_prior_from_skill,
    estimate_skill,
)

__all__ = [
    "RatingProvider",
    "ManualProvider",
    "EnrichmentQueue",
    "EnrichmentEvent",
    "PlayerProfile",
    "PlayerNotes",
    "STALE_AFTER_DAYS",
]

STALE_AFTER_DAYS: float = 90.0
"""Au-delà, un rating est signalé comme périmé : les joueurs évoluent."""


class PlayerNotesError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlayerProfile:
    """Ce que la base sait d'un adversaire, avant la première main."""

    key: str
    source: RatingSource
    fmt: GameFormat
    n_tournaments: int
    observed_roi: float
    skill: float
    shrunk_roi: float
    significant: bool
    rho: float
    archetype_prior: dict[Archetype, float]
    note: str
    updated_at: float

    @property
    def age_days(self) -> float:
        return (time.time() - self.updated_at) / 86400.0

    @property
    def is_stale(self) -> bool:
        return self.age_days > STALE_AFTER_DAYS

    def prior_weight(self, hands_observed: int, half_life: int = 50) -> float:
        r"""Poids résiduel du prior après ``hands_observed`` mains observées.

        .. math::
            w = 2^{-n/H}

        Avec H = 50, le prior externe pèse encore 100 % à la première main,
        50 % à 50 mains, 6 % à 200 mains. **C'est voulu** : ce que tu observes
        toi-même est toujours meilleur qu'un ROI de tournoi pour prédire un
        fold-to-cbet.
        """
        if hands_observed < 0:
            raise PlayerNotesError("hands_observed doit être >= 0.")
        base = 2.0 ** (-hands_observed / max(1, half_life))
        return base * (0.5 if self.is_stale else 1.0)

    def summary(self) -> str:
        flag = "  ⚠ périmé" if self.is_stale else ""
        sig = "significatif" if self.significant else "trop bruité"
        top = sorted(self.archetype_prior.items(), key=lambda kv: -kv[1])[:3]
        return (
            f"{self.key[:8]} · {self.source.value} · {self.n_tournaments} tournois "
            f"· ROI {self.observed_roi * 100:+.0f}% → {self.shrunk_roi * 100:+.1f}% "
            f"({sig}) · ρ={self.rho:.2f}{flag}\n"
            f"    prior : " + "  ".join(f"{k.value} {v * 100:.0f}%" for k, v in top)
            + (f"\n    note : {self.note}" if self.note else "")
        )


class PlayerNotes:
    """Store SQLite local. Aucun réseau, aucun pseudo en clair.

    Parameters
    ----------
    path
        Chemin du fichier. ``":memory:"`` pour une base éphémère (tests).
        En production : ``%LOCALAPPDATA%\\PokerFusion\\players.db`` — jamais
        ``%APPDATA%``, qui est synchronisé par les profils itinérants.
    salt
        Sel de hachage des pseudos. Généré à l'installation, ne quitte jamais
        la machine. Deux installations avec des sels différents produisent des
        bases non recoupables.
    """

    __slots__ = ("_db", "_salt")

    def __init__(self, path: str | Path = ":memory:", salt: str = "pfs-local") -> None:
        self._salt = salt
        p = str(path)
        if p != ":memory:":
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(p, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS players (
                key            TEXT PRIMARY KEY,
                source         TEXT NOT NULL,
                fmt            TEXT NOT NULL,
                n_tournaments  INTEGER NOT NULL,
                observed_roi   REAL NOT NULL,
                skill          REAL NOT NULL,
                shrunk_roi     REAL NOT NULL,
                significant    INTEGER NOT NULL,
                rho            REAL NOT NULL,
                archetypes     TEXT NOT NULL,
                note           TEXT NOT NULL DEFAULT '',
                updated_at     REAL NOT NULL
            )""")
        self._db.commit()

    # ── écriture ─────────────────────────────────────────────────────────
    def upsert(
        self,
        nickname: str,
        rating: ExternalRating,
        note: str = "",
        strength: float = 1.0,
    ) -> PlayerProfile:
        """Enregistre (ou met à jour) un profil. Le pseudo est haché à l'entrée."""
        est: SkillEstimate = estimate_skill(rating)
        prior = archetype_prior_from_skill(est.skill, strength)
        prof = PlayerProfile(
            key=player_key(nickname, self._salt),
            source=rating.source,
            fmt=rating.fmt,
            n_tournaments=rating.n_tournaments,
            observed_roi=rating.observed_roi,
            skill=est.skill,
            shrunk_roi=est.shrunk_roi,
            significant=est.significant,
            rho=adaptation_propensity_from_skill(est.skill),
            archetype_prior=prior,
            note=note,
            updated_at=time.time(),
        )
        self._db.execute(
            "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (prof.key, prof.source.value, prof.fmt.value, prof.n_tournaments,
             prof.observed_roi, prof.skill, prof.shrunk_roi, int(prof.significant),
             prof.rho, json.dumps({k.value: v for k, v in prior.items()}),
             prof.note, prof.updated_at),
        )
        self._db.commit()
        return prof

    # ── lecture — le chemin critique en direct ───────────────────────────
    def lookup(self, nickname: str) -> PlayerProfile | None:
        """Lookup O(1) par pseudo. **C'est ce qui tourne pendant la main.**

        Aucun accès réseau, aucune latence perceptible : un index SQLite sur
        clé primaire répond en dizaines de microsecondes.
        """
        return self.lookup_key(player_key(nickname, self._salt))

    def lookup_key(self, key: str) -> PlayerProfile | None:
        row = self._db.execute(
            "SELECT * FROM players WHERE key = ?", (key,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def lookup_many(self, nicknames: Sequence[str]) -> dict[str, PlayerProfile]:
        """Toute une table d'un coup — l'appel type après une capture d'écran."""
        keys = {player_key(n, self._salt): n for n in nicknames}
        if not keys:
            return {}
        marks = ",".join("?" * len(keys))
        rows = self._db.execute(
            f"SELECT * FROM players WHERE key IN ({marks})", tuple(keys)
        ).fetchall()
        out: dict[str, PlayerProfile] = {}
        for row in rows:
            prof = self._row_to_profile(row)
            out[keys[prof.key]] = prof
        return out

    @staticmethod
    def _row_to_profile(row: tuple) -> PlayerProfile:
        arch = {Archetype(k): v for k, v in json.loads(row[9]).items()}
        return PlayerProfile(
            key=row[0], source=RatingSource(row[1]), fmt=GameFormat(row[2]),
            n_tournaments=row[3], observed_roi=row[4], skill=row[5],
            shrunk_roi=row[6], significant=bool(row[7]), rho=row[8],
            archetype_prior=arch, note=row[10], updated_at=row[11],
        )

    # ── import / export hors ligne ───────────────────────────────────────
    CSV_FIELDS = ("nickname", "source", "format", "tournaments", "roi", "note")

    def import_csv(self, path: str | Path, strength: float = 1.0) -> tuple[int, list[str]]:
        """Importe un CSV exporté depuis SharkScope, OPR, ou tapé à la main.

        Colonnes attendues : ``nickname, source, format, tournaments, roi, note``.
        Le ROI est accepté en fraction (0.12) ou en pourcentage (12).

        Returns
        -------
        (nombre importé, liste des lignes rejetées avec leur motif)
        """
        imported, errors = 0, []
        with Path(path).open(encoding="utf-8-sig", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh), start=2):
                try:
                    nick = (row.get("nickname") or "").strip()
                    if not nick:
                        raise PlayerNotesError("pseudo vide")
                    raw_roi = str(row.get("roi") or "0").replace("%", "").replace(",", ".").strip()
                    roi = float(raw_roi or "0")
                    if abs(roi) > 5.0:      # saisi en pourcentage
                        roi /= 100.0
                    self.upsert(
                        nick,
                        ExternalRating(
                            source=RatingSource((row.get("source") or "manual").strip().lower()),
                            fmt=GameFormat((row.get("format") or "mtt_large").strip().lower()),
                            n_tournaments=int(float(row.get("tournaments") or 0)),
                            observed_roi=roi,
                        ),
                        note=(row.get("note") or "").strip(),
                        strength=strength,
                    )
                    imported += 1
                except Exception as exc:
                    errors.append(f"ligne {i} : {type(exc).__name__} — {exc}")
        return imported, errors

    def export_csv(self, path: str | Path) -> int:
        """Export **anonymisé** : les clés hachées, jamais les pseudos.

        Une base exportée ne permet donc pas d'identifier les joueurs — c'est
        volontaire, et c'est aussi ce qui la rend inutilisable pour du
        *data sharing*.
        """
        rows = self._db.execute("SELECT * FROM players").fetchall()
        with Path(path).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(("key", "source", "format", "tournaments", "roi",
                        "skill", "shrunk_roi", "rho", "note", "updated_at"))
            for r in rows:
                w.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[8], r[10], r[11]))
        return len(rows)

    # ── divers ───────────────────────────────────────────────────────────
    def blended_prior(
        self, nickname: str, hands_observed: int
    ) -> dict[Archetype, float]:
        """Prior d'archétype mélangé avec le prior neutre selon l'ancienneté.

        C'est ce que consomme le filtre particulaire à l'ouverture d'une main.
        """
        prof = self.lookup(nickname)
        if prof is None:
            return dict(ARCHETYPE_PRIORS)
        w = prof.prior_weight(hands_observed)
        out = {k: (1 - w) * ARCHETYPE_PRIORS[k] + w * prof.archetype_prior.get(k, 0.0)
               for k in ARCHETYPE_PRIORS}
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}

    def stats(self) -> dict[str, float]:
        n = self._db.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        if not n:
            return {"n": 0, "significant": 0, "stale": 0, "mean_skill": 0.5}
        sig = self._db.execute(
            "SELECT COUNT(*) FROM players WHERE significant = 1").fetchone()[0]
        cutoff = time.time() - STALE_AFTER_DAYS * 86400
        stale = self._db.execute(
            "SELECT COUNT(*) FROM players WHERE updated_at < ?", (cutoff,)).fetchone()[0]
        mean = self._db.execute("SELECT AVG(skill) FROM players").fetchone()[0]
        return {"n": n, "significant": sig, "stale": stale, "mean_skill": mean}

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM players").fetchone()[0]

    def close(self) -> None:
        self._db.close()


# ═══════════════════════════════════════════════════════════════════════════
# FILE D'ENRICHISSEMENT DIFFÉRÉ
# ═══════════════════════════════════════════════════════════════════════════
#
# Conception issue d'une contrainte posée explicitement : n'enrichir que
#   (a) **hors main** — aucune main en cours à aucune table, et
#   (b) pour des joueurs **avec qui tu as effectivement joué une main**.
#
# Ces deux verrous répondent aux deux objections sérieuses :
#
#   * PokerStars limite la collecte d'informations sur les joueurs aux mains
#     auxquelles tu participes. Le verrou (b) l'implémente littéralement : un
#     joueur qui n'a jamais été dans un pot avec toi n'entre jamais dans la
#     file.
#   * La traçabilité côté opérateur repose sur la **corrélation temporelle**
#     entre un board distribué et une requête sortante. Le verrou (a) casse
#     cette corrélation : rien ne part tant qu'une main est vivante.
#
# Ce qui reste de ta responsabilité : le fournisseur de ratings lui-même.
# Le noyau reste sans réseau ; brancher un fournisseur est un acte explicite,
# désactivé par défaut, et tracé dans un journal d'audit local.


from typing import Protocol, runtime_checkable


@runtime_checkable
class RatingProvider(Protocol):
    """Fournisseur de ratings. Implémentation laissée à l'utilisateur.

    Le noyau n'en fournit **aucune** qui accède au réseau. Pour brancher
    SharkScope, écris une classe conforme à ce protocole avec ta propre clé
    d'API et l'endpoint de ton abonnement, puis passe-la à
    :meth:`EnrichmentQueue.flush`. La file applique les verrous quel que soit
    le fournisseur.
    """

    def fetch(self, nickname: str) -> ExternalRating | None:
        """Retourne un rating, ou ``None`` si introuvable."""
        ...


@dataclass(slots=True)
class ManualProvider:
    """Fournisseur hors ligne : un dictionnaire que tu remplis toi-même.

    C'est l'implémentation par défaut, et elle couvre le cas réel : tu
    consultes SharkScope dans ton navigateur entre deux sessions, tu colles
    les chiffres, la file fait le reste.
    """

    ratings: dict[str, ExternalRating] = field(default_factory=dict)

    def fetch(self, nickname: str) -> ExternalRating | None:
        return self.ratings.get(nickname.strip().lower())


@dataclass(frozen=True, slots=True)
class EnrichmentEvent:
    """Une entrée du journal d'audit — ce qui est parti, quand, et pourquoi."""

    nickname_key: str
    at: float
    outcome: str
    reason: str = ""


class EnrichmentQueue:
    """File différée, doublement verrouillée.

    Parameters
    ----------
    notes
        Base locale de destination.
    min_interval_s
        Intervalle minimal entre deux enrichissements, pour éviter une rafale
        reconnaissable. 2 s par défaut.
    """

    __slots__ = ("_notes", "_pending", "_participated", "_live_hands",
                 "_log", "_min_interval", "_last_flush")

    def __init__(self, notes: PlayerNotes, min_interval_s: float = 2.0) -> None:
        self._notes = notes
        self._pending: dict[str, str] = {}      # nickname → pseudo original
        self._participated: set[str] = set()
        self._live_hands: set[str] = set()
        self._log: list[EnrichmentEvent] = []
        self._min_interval = float(min_interval_s)
        self._last_flush = 0.0

    # ── état des tables ──────────────────────────────────────────────────
    def hand_started(self, table_id: str) -> None:
        self._live_hands.add(table_id)

    def hand_ended(self, table_id: str, participants: Iterable[str]) -> None:
        """Fin de main : les joueurs présents deviennent éligibles.

        ``participants`` = les pseudos qui étaient **dans la main avec toi**.
        C'est le verrou (b), et il est appliqué ici, à la source.
        """
        self._live_hands.discard(table_id)
        for nick in participants:
            n = nick.strip()
            if n:
                self._participated.add(n.lower())

    @property
    def has_live_hand(self) -> bool:
        return bool(self._live_hands)

    # ── file ─────────────────────────────────────────────────────────────
    def enqueue(self, nickname: str) -> bool:
        """Met un pseudo en file. Retourne ``False`` s'il est refusé.

        Refusé si : jamais joué contre toi, ou déjà connu en base.
        """
        n = nickname.strip()
        if not n:
            return False
        if n.lower() not in self._participated:
            self._log.append(EnrichmentEvent(player_key(n, self._notes._salt),
                                             time.time(), "refusé",
                                             "aucune main jouée ensemble"))
            return False
        if self._notes.lookup(n) is not None:
            return False
        self._pending[n.lower()] = n
        return True

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._pending.values())

    # ── exécution ────────────────────────────────────────────────────────
    def flush(
        self, provider: RatingProvider, max_items: int | None = None
    ) -> tuple[int, str]:
        """Traite la file — **uniquement si aucune main n'est en cours**.

        Returns
        -------
        (nombre enrichi, message)
        """
        if self.has_live_hand:
            return 0, ("Main en cours : rien n'est envoyé. La file sera traitée "
                       "dès que toutes les tables seront entre deux mains.")
        if not self._pending:
            return 0, "File vide."
        now = time.time()
        if now - self._last_flush < self._min_interval:
            return 0, f"Cadence limitée ({self._min_interval:.0f} s entre deux traitements)."

        done = 0
        items = list(self._pending.items())
        if max_items:
            items = items[:max_items]
        for lower, original in items:
            rating = provider.fetch(original)
            self._pending.pop(lower, None)
            if rating is None:
                self._log.append(EnrichmentEvent(
                    player_key(original, self._notes._salt), time.time(),
                    "introuvable"))
                continue
            self._notes.upsert(original, rating)
            self._log.append(EnrichmentEvent(
                player_key(original, self._notes._salt), time.time(), "enrichi"))
            done += 1

        self._last_flush = now
        return done, f"{done} profil(s) enrichi(s), {len(self._pending)} en attente."

    # ── audit ────────────────────────────────────────────────────────────
    @property
    def audit_log(self) -> tuple[EnrichmentEvent, ...]:
        """Journal local : tout ce qui a été tenté, accepté ou refusé."""
        return tuple(self._log)

    def audit_summary(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._log:
            out[e.outcome] = out.get(e.outcome, 0) + 1
        return out
