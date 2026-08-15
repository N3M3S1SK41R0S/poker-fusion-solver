"""
Blueprint flop — l'infrastructure du pré-calcul Phase 2 : classes, magasin,
manifeste de reprise, requête re-projetée.

Statut (14 août 2026) — BRANCHÉ
-------------------------------
Premier consommateur applicatif des deux briques Phase 2 jusqu'ici en
bibliothèque : ``pfs.solver.isomorphism`` (les 1 755 classes) et
``pfs.solver.abstraction`` (les buckets EHS). Consommé par
``banc_blueprint.py`` (script livré, banc de dimensionnement) ; le CALCUL
par lots — remplir le magasin pour les 1 755 classes — viendra sur décision,
chiffré par le banc. Ce module rend ce calcul possible, REPRENABLE après
interruption, et interrogeable depuis n'importe quel flop réel.

Dénombrement exact (posé ici, testé dans ``tests/test_blueprint.py``)
---------------------------------------------------------------------
- **Côté flops bruts** : C(52,3) = 22 100 flops, partitionnés par l'action
  du groupe S₄ des couleurs en **1 755 orbites** (Burnside :
  (22 100 + 6·2 938 + 8·299)/24 = 42 120/24 = 1 755, détail dans
  ``pfs.solver.isomorphism``). La somme des poids des classes rendues par
  ``enumerate_classes`` vaut donc EXACTEMENT 22 100.
- **Côté mains fixées** : pour toute main héros (2 cartes), il reste
  C(50,3) = 19 600 flops compatibles. Les cartes du héros brisent la
  symétrie des couleurs, donc les orbites ne coupent PAS ce sous-ensemble
  uniformément — mais chaque flop compatible appartient à exactement une
  des 1 755 classes : Σ_classes |orbite ∩ {flops évitant la main}| = 19 600.
  C'est pourquoi 1 755 solves suffisent à desservir TOUTES les mains : la
  requête transporte la main par la permutation de couleurs de SON flop.

Architecture du magasin
-----------------------
::

    racine/
      v1/                       ← version du FORMAT de stockage
        <empreinte-réglages>/   ← sha256 court des réglages (12 hex)
          manifest.json         ← réglages complets + classes faites
          classes/
            AsKsQh.npz          ← np.savez_compressed, un fichier par classe

- **Un np.savez compressé par classe** : zéro dépendance nouvelle. LMDB ou
  un format monolithique ne se justifieraient que par une mesure (latence
  d'ouverture de ~1 755 petits fichiers) que ``banc_blueprint.py`` rend
  possible — à ce stade, l'accès est « une classe à la fois » (requête d'un
  flop) et np.load d'un npz unique est déjà en dessous de la milliseconde.
- **Reprise sur interruption** : le manifeste est réécrit ATOMIQUEMENT
  (fichier temporaire + ``os.replace``) après chaque classe sauvée ; le
  ``.npz`` lui-même est écrit en temporaire puis renommé. Relancer avec les
  mêmes réglages recharge le manifeste et ``pending()`` liste ce qui reste.
  Une entrée du manifeste dont le fichier a disparu est ignorée (la classe
  redevient à faire) : le manifeste ne peut pas mentir plus d'un fichier.
- **Invalidation propre** : les réglages (itérations, buckets, version du
  solveur…) déterminent le RÉPERTOIRE (empreinte sha256). Un réglage
  différent ouvre un répertoire vierge — rien d'un ancien réglage n'est
  jamais compté comme fait, et l'ancien calcul n'est pas détruit. Si un
  manifeste présent contredit ses propres réglages (édition manuelle,
  collision), l'ouverture échoue explicitement.

Requête (aller-retour flop réel ↔ classe)
-----------------------------------------
``query_flop`` : flop réel → ``canonical_board`` (classe) → solution chargée
→ stratégie re-projetée par la permutation des couleurs. Les tableaux
déclarés « indexés combo » (dernier axe = 1 326) sont réindexés par
``combo_permutation`` : ``sortie[..., i] = stocké[..., σ(i)]`` où σ envoie le
combo réel sur son image canonique. L'invariance stratégique sous S₄
(Waugh, 2013) garantit que la stratégie re-projetée est EXACTEMENT celle
qu'un solve direct du flop réel aurait produite.

Pipeline d'abstraction (classe → buckets EHS → entrée du solveur)
-----------------------------------------------------------------
``bucketed_inputs`` assemble, pour un flop et deux ranges, ce que le CFR
bucketisé consomme à la place des 1 326 combos : combos vivants, poids,
EHS (``pfs.solver.abstraction``, mémoïsé par classe canonique) et étiquette
de bucket par combo. Deux flops isomorphes rendent des entrées strictement
identiques à la permutation des couleurs près.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from pfs.core.range_model import (
    N_COMBOS,
    RANKS,
    SUITS,
    Range,
    combo_cards,
    combo_index,
)
from pfs.solver.abstraction import bucket_assignments, expected_hand_strength
from pfs.solver.isomorphism import (
    canonical_board,
    enumerate_canonical_flops,
    suit_mapping,
)

__all__ = [
    "BlueprintError",
    "BlueprintQuery",
    "BlueprintSettings",
    "BlueprintSolution",
    "BlueprintStore",
    "BucketedInputs",
    "FlopClass",
    "PlayerBuckets",
    "FORMAT_VERSION",
    "N_FLOP_CLASSES",
    "TOTAL_FLOPS",
    "bucketed_inputs",
    "class_key",
    "combo_permutation",
    "enumerate_classes",
    "query_flop",
]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

FORMAT_VERSION: int = 1
"""Version du FORMAT de stockage (répertoire ``v1/``) — pas des réglages."""

N_FLOP_CLASSES: int = 1755
"""Nombre de classes canoniques de flops (Burnside, vérifié par énumération)."""

TOTAL_FLOPS: int = 22100
"""C(52,3) — la somme des poids des 1 755 classes, testée."""

_METHODS: tuple[str, ...] = ("percentile", "kmeans")


class BlueprintError(ValueError):
    """Entrée, réglage ou état de magasin invalide."""


# ═══════════════════════════════════════════════════════════════════════════
# 1. ÉNUMÉRATION DES CLASSES AVEC POIDS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FlopClass:
    """Une classe canonique de flops.

    Attributes
    ----------
    board : tuple[int, int, int]
        Le représentant canonique (trié croissant, minimal sous S₄).
    weight : int
        Taille de l'orbite — nombre de flops bruts que la classe représente.
        Σ weight sur les 1 755 classes = C(52,3) = 22 100.
    key : str
        Clé lisible et sûre pour un nom de fichier : ``"AsKsQh"``.
    """

    board: tuple[int, int, int]
    weight: int
    key: str


def class_key(cards: Sequence[int]) -> str:
    """Clé canonique d'un flop : représentant minimal, cartes concaténées.

    Deux flops isomorphes ont la même clé ; la clé est un nom de fichier
    valide (lettres et chiffres seulement).

    >>> class_key([2, 7, 9])     # Ad Kc Qh → classe arc-en-ciel AKQ
    'AsKhQd'
    """
    canon = canonical_board(cards)
    if len(canon) != 3:
        raise BlueprintError(
            f"blueprint FLOP : 3 cartes attendues, reçu {len(canon)}."
        )
    return "".join(f"{RANKS[c // 4]}{SUITS[c % 4]}" for c in canon)


@lru_cache(maxsize=1)
def _classes() -> tuple[FlopClass, ...]:
    items = sorted(enumerate_canonical_flops().items())
    return tuple(
        FlopClass(board=b, weight=w, key=class_key(b)) for b, w in items
    )


@lru_cache(maxsize=1)
def _class_by_key() -> dict[str, FlopClass]:
    return {c.key: c for c in _classes()}


def enumerate_classes() -> tuple[FlopClass, ...]:
    """Les 1 755 classes canoniques de flops, poids inclus, ordre déterministe.

    L'ordre est l'ordre lexicographique des représentants canoniques —
    stable d'un processus à l'autre, c'est l'ordre de travail du calcul par
    lots. Le tuple rendu est immuable et partagé (mémoïsé).

    >>> classes = enumerate_classes()
    >>> len(classes), sum(c.weight for c in classes)
    (1755, 22100)
    """
    return _classes()


# ═══════════════════════════════════════════════════════════════════════════
# 2. RÉGLAGES (l'identité d'un calcul de blueprint)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BlueprintSettings:
    """Le réglage complet d'un calcul de blueprint — son identité de reprise.

    Deux réglages égaux partagent le même répertoire de magasin (reprise) ;
    deux réglages différents n'ont AUCUN état en commun (invalidation
    propre par empreinte).

    Attributes
    ----------
    iterations : int
        Itérations CFR par classe.
    n_buckets : int
        Nombre de buckets EHS de l'abstraction.
    method : str
        ``"percentile"`` ou ``"kmeans"`` (cf. ``pfs.solver.abstraction``).
    n_rollouts : int | None
        Sous-échantillonnage des runouts de l'EHS (``None`` = exact).
    seed : int
        Graine du sous-échantillonnage (sans effet en mode exact).
    solver_version : str
        Version du solveur qui produit les solutions — changer de solveur
        invalide le calcul, exactement comme changer d'itérations.
    """

    iterations: int
    n_buckets: int
    method: str = "percentile"
    n_rollouts: int | None = None
    seed: int = 0
    solver_version: str = "turn-proxy-v1"

    def __post_init__(self) -> None:
        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations, int
        ) or self.iterations < 1:
            raise BlueprintError("iterations : entier ≥ 1 requis.")
        if isinstance(self.n_buckets, bool) or not isinstance(
            self.n_buckets, int
        ) or self.n_buckets < 1:
            raise BlueprintError("n_buckets : entier ≥ 1 requis.")
        if self.method not in _METHODS:
            raise BlueprintError(
                f"method inconnue : {self.method!r} — attendu "
                f"{' ou '.join(map(repr, _METHODS))}."
            )
        if self.n_rollouts is not None and (
            isinstance(self.n_rollouts, bool)
            or not isinstance(self.n_rollouts, int)
            or self.n_rollouts < 1
        ):
            raise BlueprintError("n_rollouts : entier ≥ 1 ou None requis.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise BlueprintError("seed : entier requis.")
        if not isinstance(self.solver_version, str) or not self.solver_version:
            raise BlueprintError("solver_version : chaîne non vide requise.")

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "n_buckets": self.n_buckets,
            "method": self.method,
            "n_rollouts": self.n_rollouts,
            "seed": self.seed,
            "solver_version": self.solver_version,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "BlueprintSettings":
        try:
            return cls(**{k: d[k] for k in (
                "iterations", "n_buckets", "method", "n_rollouts", "seed",
                "solver_version")})
        except (KeyError, TypeError) as exc:
            raise BlueprintError(f"réglages illisibles : {d!r}") from exc

    def digest(self) -> str:
        """Empreinte courte (12 hex) — le nom du répertoire de ce réglage."""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return sha256(blob.encode("utf-8")).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAGASIN DE SOLUTIONS (np.savez par classe + manifeste de reprise)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BlueprintSolution:
    """Solution stockée d'une classe, telle que relue du magasin."""

    board: tuple[int, int, int]
    arrays: dict[str, np.ndarray]
    combo_keys: tuple[str, ...]
    meta: dict


class BlueprintStore:
    """Magasin de solutions par classe canonique, avec manifeste de reprise.

    Parameters
    ----------
    root : str | Path
        Racine du magasin. Le répertoire réel est
        ``root/v{FORMAT_VERSION}/{settings.digest()}``.
    settings : BlueprintSettings
        L'identité du calcul — voir la docstring du module pour la
        sémantique de reprise et d'invalidation.

    Raises
    ------
    BlueprintError
        Manifeste présent mais incompatible (format inconnu, réglages qui
        contredisent l'empreinte du répertoire) — jamais d'écrasement muet.
    """

    def __init__(self, root: str | Path, settings: BlueprintSettings) -> None:
        if not isinstance(settings, BlueprintSettings):
            raise BlueprintError("settings doit être un BlueprintSettings.")
        self.settings = settings
        self.directory = Path(root) / f"v{FORMAT_VERSION}" / settings.digest()
        self._classes_dir = self.directory / "classes"
        self._manifest_path = self.directory / "manifest.json"
        self._classes_dir.mkdir(parents=True, exist_ok=True)
        self._by_key = _class_by_key()
        self._done: dict[str, dict] = {}
        if self._manifest_path.exists():
            self._load_manifest()
        else:
            self._write_manifest()

    # ── manifeste ─────────────────────────────────────────────────────────

    def _load_manifest(self) -> None:
        try:
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BlueprintError(
                f"manifeste illisible : {self._manifest_path}"
            ) from exc
        if data.get("format") != FORMAT_VERSION:
            raise BlueprintError(
                f"format de manifeste {data.get('format')!r} : "
                f"attendu {FORMAT_VERSION}."
            )
        stored = data.get("settings")
        if stored != self.settings.to_dict():
            raise BlueprintError(
                "réglages du manifeste incompatibles avec le répertoire — "
                "magasin corrompu ou édité à la main : "
                f"{stored!r} ≠ {self.settings.to_dict()!r}."
            )
        done = data.get("done", {})
        if not isinstance(done, dict):
            raise BlueprintError("manifeste corrompu : 'done' n'est pas un objet.")
        kept: dict[str, dict] = {}
        for key, entry in done.items():
            if key not in self._by_key:
                raise BlueprintError(f"clé de classe inconnue au manifeste : {key!r}.")
            if self._solution_path(key).exists():
                kept[key] = dict(entry)
            # fichier absent → la classe redevient À FAIRE, silencieusement :
            # le manifeste ne peut pas prétendre plus que le disque.
        self._done = kept

    def _write_manifest(self) -> None:
        payload = {
            "format": FORMAT_VERSION,
            "settings": self.settings.to_dict(),
            "updated": _now_iso(),
            "n_classes": N_FLOP_CLASSES,
            "done": self._done,
        }
        fd, tmp = tempfile.mkstemp(
            dir=self.directory, prefix=".manifest-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
            os.replace(tmp, self._manifest_path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ── état ──────────────────────────────────────────────────────────────

    def _solution_path(self, key: str) -> Path:
        return self._classes_dir / f"{key}.npz"

    def is_done(self, board: Sequence[int]) -> bool:
        """La classe de ce flop (réel ou canonique) est-elle résolue ?"""
        return class_key(board) in self._done

    def done_keys(self) -> frozenset[str]:
        return frozenset(self._done)

    def pending(self) -> tuple[FlopClass, ...]:
        """Les classes restant à résoudre, dans l'ordre de travail."""
        return tuple(c for c in enumerate_classes() if c.key not in self._done)

    def progress(self) -> dict:
        """Avancement : classes faites et POIDS couvert (flops desservis)."""
        weight_done = sum(self._by_key[k].weight for k in self._done)
        return {
            "n_done": len(self._done),
            "n_classes": N_FLOP_CLASSES,
            "weight_done": weight_done,
            "weight_total": TOTAL_FLOPS,
            "fraction": weight_done / TOTAL_FLOPS,
        }

    # ── écriture / lecture ────────────────────────────────────────────────

    def save_solution(
        self,
        board: Sequence[int],
        arrays: Mapping[str, np.ndarray],
        combo_keys: Sequence[str] = (),
        meta: Mapping | None = None,
        elapsed_s: float | None = None,
    ) -> Path:
        """Sauve la solution d'une classe (atomique) et marque la classe faite.

        Parameters
        ----------
        board : Sequence[int]
            Flop réel ou canonique — canonisé ici : sauver sous n'importe
            quelle image de l'orbite marque LA classe.
        arrays : Mapping[str, ndarray]
            Les tableaux de la solution, nommés librement (pas de préfixe
            ``_`` — réservé aux métadonnées internes).
        combo_keys : Sequence[str]
            Noms des tableaux dont le DERNIER axe est indexé par les 1 326
            combos canoniques — les seuls que ``query_flop`` re-projette.
        meta : Mapping | None
            Métadonnées JSON-sérialisables (exploitabilité, EV…).
        elapsed_s : float | None
            Durée du solve, consignée au manifeste (matière du banc).

        Returns
        -------
        Path
            Le fichier ``.npz`` écrit.
        """
        key = class_key(board)
        canon = canonical_board(board)
        if not arrays:
            raise BlueprintError("arrays vide : rien à sauver.")
        for name, arr in arrays.items():
            if not isinstance(name, str) or not name or name.startswith("_"):
                raise BlueprintError(
                    f"nom de tableau invalide : {name!r} (préfixe '_' réservé)."
                )
            if not isinstance(arr, np.ndarray):
                raise BlueprintError(f"{name!r} n'est pas un ndarray.")
        combo_keys = tuple(combo_keys)
        for name in combo_keys:
            if name not in arrays:
                raise BlueprintError(f"combo_keys : {name!r} absent de arrays.")
            if arrays[name].ndim < 1 or arrays[name].shape[-1] != N_COMBOS:
                raise BlueprintError(
                    f"{name!r} : dernier axe {arrays[name].shape} ≠ "
                    f"{N_COMBOS} — un tableau « combo » couvre les 1 326 "
                    "combos canoniques (combos morts inclus)."
                )
        meta_d = dict(meta) if meta is not None else {}
        try:
            meta_json = json.dumps(meta_d, ensure_ascii=False, sort_keys=True)
        except TypeError as exc:
            raise BlueprintError("meta non sérialisable en JSON.") from exc

        final = self._solution_path(key)
        fd, tmp = tempfile.mkstemp(
            dir=self._classes_dir, prefix=f".{key}-", suffix=".tmp.npz"
        )
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    _board=np.array(canon, dtype=np.int64),
                    _combo_keys=np.array(list(combo_keys), dtype=np.str_),
                    _meta=np.array(meta_json, dtype=np.str_),
                    **arrays,
                )
            os.replace(tmp, final)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        entry: dict = {
            "multiplicity": self._by_key[key].weight,
            "size_bytes": final.stat().st_size,
            "saved_at": _now_iso(),
        }
        if elapsed_s is not None:
            entry["elapsed_s"] = round(float(elapsed_s), 6)
        self._done[key] = entry
        self._write_manifest()
        return final

    def load_solution(self, board: Sequence[int]) -> BlueprintSolution:
        """Recharge la solution de la classe de ce flop (telle que sauvée).

        Les tableaux rendus sont indexés par les combos CANONIQUES — pour la
        vue re-projetée sur un flop réel, passer par ``query_flop``.
        """
        key = class_key(board)
        path = self._solution_path(key)
        if key not in self._done or not path.exists():
            raise BlueprintError(
                f"classe {key} absente du magasin ({len(self._done)}/"
                f"{N_FLOP_CLASSES} faites) — la résoudre d'abord."
            )
        with np.load(path, allow_pickle=False) as data:
            combo_keys = tuple(str(x) for x in data["_combo_keys"])
            meta = json.loads(str(data["_meta"]))
            board_stored = tuple(int(x) for x in data["_board"])
            arrays = {
                name: data[name].copy()
                for name in data.files
                if not name.startswith("_")
            }
        if board_stored != canonical_board(board):
            raise BlueprintError(
                f"fichier {path.name} : board stocké {board_stored} ≠ classe "
                f"attendue {canonical_board(board)} — magasin corrompu."
            )
        return BlueprintSolution(
            board=board_stored, arrays=arrays, combo_keys=combo_keys, meta=meta
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════════════════
# 4. REQUÊTE : flop réel → classe → stratégie re-projetée
# ═══════════════════════════════════════════════════════════════════════════


@lru_cache(maxsize=32)
def _combo_perm(perm: tuple[int, int, int, int]) -> I64:
    out = np.empty(N_COMBOS, dtype=np.int64)
    for i in range(N_COMBOS):
        a, b = combo_cards(i)
        out[i] = combo_index(
            (a // 4) * 4 + perm[a % 4], (b // 4) * 4 + perm[b % 4]
        )
    out.flags.writeable = False
    return out


def combo_permutation(mapping: Mapping[int, int]) -> I64:
    """Permutation induite sur les 1 326 combos par une permutation de couleurs.

    ``out[i]`` est l'index du combo image de ``i`` par ``mapping``
    (``{ancienne_couleur: nouvelle_couleur}``, bijection de {0,1,2,3}).
    Pour re-projeter un tableau canonique sur un flop réel dont
    ``suit_mapping`` vaut ``mapping`` : ``reel = canonique[..., out]``.

    Returns
    -------
    ndarray
        (1326,) int64 en lecture seule (mémoïsé par permutation).
    """
    try:
        perm = tuple(int(mapping[s]) for s in range(4))
    except (KeyError, TypeError, ValueError) as exc:
        raise BlueprintError(f"mapping illisible : {mapping!r}") from exc
    if sorted(perm) != [0, 1, 2, 3]:
        raise BlueprintError(
            f"mapping non bijectif sur les 4 couleurs : {mapping!r}"
        )
    return _combo_perm(perm)


@dataclass(frozen=True, slots=True)
class BlueprintQuery:
    """Résultat d'une requête : la solution de la classe, VUE depuis le flop réel.

    Attributes
    ----------
    board : tuple[int, int, int]
        Le flop réel demandé (trié croissant).
    canonical : tuple[int, int, int]
        Le représentant canonique de sa classe.
    mapping : dict[int, int]
        La permutation de couleurs flop réel → canonique (``suit_mapping``).
    weight : int
        Poids de la classe (taille d'orbite).
    arrays : dict[str, ndarray]
        Les tableaux de la solution ; ceux listés dans ``combo_keys`` sont
        RE-PROJETÉS : ``arrays[k][..., i]`` concerne le combo réel ``i``.
    combo_keys : tuple[str, ...]
        Les tableaux indexés combo (donc re-projetés).
    meta : dict
        Métadonnées de la solution, inchangées.
    """

    board: tuple[int, int, int]
    canonical: tuple[int, int, int]
    mapping: dict[int, int]
    weight: int
    arrays: dict[str, np.ndarray]
    combo_keys: tuple[str, ...]
    meta: dict


def query_flop(store: BlueprintStore, board: Sequence[int]) -> BlueprintQuery:
    """Flop réel → classe canonique → solution chargée → stratégie re-projetée.

    L'API de service du blueprint : quel que soit le flop demandé (les
    22 100), la solution provient d'UNE des 1 755 classes, et les tableaux
    indexés combo sont réindexés par la permutation de couleurs pour que
    l'index ``i`` désigne le combo RÉEL ``i``. Sur un flop déjà canonique,
    la permutation est l'identité et les tableaux sont rendus tels quels.

    Raises
    ------
    BlueprintError
        Flop invalide (via l'isomorphisme), ou classe non résolue.
    """
    canon = canonical_board(board)
    if len(canon) != 3:
        raise BlueprintError("query_flop : un FLOP (3 cartes) est attendu.")
    mapping = suit_mapping(board)
    sol = store.load_solution(canon)
    perm = combo_permutation(mapping)
    out: dict[str, np.ndarray] = {}
    for name, arr in sol.arrays.items():
        out[name] = arr[..., perm] if name in sol.combo_keys else arr.copy()
    cls = _class_by_key()[class_key(canon)]
    return BlueprintQuery(
        board=tuple(sorted(int(c) for c in board)),
        canonical=canon,
        mapping=dict(mapping),
        weight=cls.weight,
        arrays=out,
        combo_keys=sol.combo_keys,
        meta=sol.meta,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. PIPELINE D'ABSTRACTION : classe → buckets EHS → entrée du solveur
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class PlayerBuckets:
    """L'entrée « cartes » d'un joueur pour un CFR bucketisé.

    Attributes
    ----------
    cards : ndarray
        (n, 2) int64 — cartes des combos vivants (blockers du board retirés).
    combo_indices : ndarray
        (n,) int64 — index canonique 0–1325 de chaque combo vivant.
    weights : ndarray
        (n,) float64 — poids de range des combos vivants.
    ehs : ndarray
        (n,) float64 — force espérée (``expected_hand_strength``).
    buckets : ndarray
        (n,) int64 — bucket ∈ {0, …, K−1} de chaque combo, croissant avec
        la force (``bucket_assignments``, masses pondérées par la range).
    """

    cards: I64
    combo_indices: I64
    weights: F64
    ehs: F64
    buckets: I64


@dataclass(frozen=True, slots=True)
class BucketedInputs:
    """Entrée complète du solveur bucketisé pour UNE classe de flop."""

    board: tuple[int, int, int]
    canonical: tuple[int, int, int]
    mapping: dict[int, int]
    n_buckets: int
    oop: PlayerBuckets
    ip: PlayerBuckets


def _player_buckets(
    range_: Range, board: Sequence[int], settings: BlueprintSettings
) -> PlayerBuckets:
    live = range_.remove_blockers(board)
    idx = np.nonzero(live.weights > 0.0)[0]
    if idx.size == 0:
        raise BlueprintError("range vide sur ce flop après retrait des blockers.")
    cards = np.array([combo_cards(int(i)) for i in idx], dtype=np.int64)
    weights = live.weights[idx].astype(np.float64)
    ehs = expected_hand_strength(
        cards, board, n_rollouts=settings.n_rollouts, seed=settings.seed
    )
    buckets = bucket_assignments(
        ehs, n_buckets=settings.n_buckets, method=settings.method,
        weights=weights,
    )
    return PlayerBuckets(
        cards=cards, combo_indices=idx.astype(np.int64), weights=weights,
        ehs=ehs, buckets=buckets,
    )


def bucketed_inputs(
    board: Sequence[int],
    oop_range: Range,
    ip_range: Range,
    settings: BlueprintSettings,
) -> BucketedInputs:
    """Assemble l'entrée d'un CFR bucketisé : combos vivants, EHS, buckets.

    Le pipeline Phase 2 au complet, pour une classe : l'EHS transite par la
    table mémoïsée de la classe CANONIQUE (``pfs.solver.abstraction``), donc
    appeler cette fonction sur les 24 images d'un même flop ne paie le
    calcul qu'une fois — et rend des valeurs strictement identiques à la
    permutation des couleurs près.

    Parameters
    ----------
    board : Sequence[int]
        Flop (3 cartes) — réel ou canonique.
    oop_range, ip_range : Range
        Les deux ranges du spot.
    settings : BlueprintSettings
        ``n_buckets``, ``method``, ``n_rollouts``, ``seed`` sont consommés
        ici (``iterations`` et ``solver_version`` concernent le solve).

    Raises
    ------
    BlueprintError
        Board non-flop, range invalide ou vide après blockers.
    """
    canon = canonical_board(board)
    if len(canon) != 3:
        raise BlueprintError("bucketed_inputs : un FLOP (3 cartes) est attendu.")
    if not isinstance(settings, BlueprintSettings):
        raise BlueprintError("settings doit être un BlueprintSettings.")
    for name, r in (("oop_range", oop_range), ("ip_range", ip_range)):
        if not isinstance(r, Range):
            raise BlueprintError(f"{name} doit être une Range.")
    return BucketedInputs(
        board=tuple(sorted(int(c) for c in board)),
        canonical=canon,
        mapping=dict(suit_mapping(board)),
        n_buckets=settings.n_buckets,
        oop=_player_buckets(oop_range, board, settings),
        ip=_player_buckets(ip_range, board, settings),
    )
