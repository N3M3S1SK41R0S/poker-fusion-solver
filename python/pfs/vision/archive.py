"""Archive des captures — et surtout des ÉCHECS de reconnaissance.

Tout ce qui a été mesuré jusqu'ici l'a été sur des tables synthétiques
composées pour le test. C'est ce qui a fait perdre le plus de temps : les
défauts réels — cadrage qui ne tombait pas sur la carte, HUD masquant le bas
des cartes, seuils calibrés sur des images parfaites — ne se voyaient pas.

Ce module conserve donc la matière réelle au moment où elle passe :
  * chaque capture collée dans l'application, entière ;
  * chaque **découpe refusée ou seulement proposée**, avec son diagnostic
    (écart, marge, meilleur candidat). Ce sont les échecs qui instruisent :
    une reconnaissance réussie n'apprend rien qu'on ne sache déjà.

Rien ne sort de la machine : l'archive vit dans le dossier de données local
de l'utilisateur, et n'est jamais versionnée.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "dossier_archive",
    "enregistrer_capture",
    "enregistrer_echec",
    "lister_echecs",
    "EchecArchive",
]

_HORODATAGE = "%Y%m%d-%H%M%S"


def dossier_archive() -> Path:
    """Racine de l'archive, créée au besoin, **chemin réel**.

    Sous ``%LOCALAPPDATA%`` : hors du dépôt, donc jamais versionnée, et
    conservée d'une réinstallation du logiciel à l'autre.

    Le ``realpath`` final n'est pas cosmétique. Quand l'interpréteur vient
    du Microsoft Store — ce qui est le cas ici, ``sys.base_prefix`` pointant
    sous ``C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation...`` —
    Windows redirige silencieusement les écritures du processus vers le
    ``LocalCache`` du paquet, **tout en laissant** ``os.environ["LOCALAPPDATA"]``
    et ``os.path.abspath`` afficher le chemin d'origine. Le logiciel
    annonçait donc un dossier qui n'existe pas : ``Test-Path`` répondait
    ``False`` depuis PowerShell là où Python voyait ses fichiers, et
    l'utilisateur ne pouvait pas retrouver ses captures dans l'explorateur.

    ``realpath`` traverse la redirection et rend le chemin qu'on peut coller
    dans une barre d'adresse. Les tests ne pouvaient pas voir le défaut :
    ils remplacent ``LOCALAPPDATA`` par un dossier temporaire, ce qui
    court-circuite la virtualisation — d'où
    ``tests/test_archive_chemin_reel.py``, qui compare les deux.
    """
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    d = Path(base) / "PokerFusionSolver" / "captures"
    (d / "echecs").mkdir(parents=True, exist_ok=True)
    return Path(os.path.realpath(d))


def _decoder(image_b64: str) -> bytes:
    """Octets PNG d'une image encodée, préfixe « data: » toléré."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return base64.b64decode(image_b64)


def _horodate(dossier: Path, prefixe: str, suffixe: str) -> Path:
    """Chemin horodaté, garanti libre.

    L'horodatage seul ne suffit pas : une lecture de table archive jusqu'à
    huit découpes dans la même seconde, et les noms entraient en collision —
    les échecs s'écrasaient silencieusement. Une session de calibration ne
    conservait qu'une découpe par seconde et par statut, ce qui vide le banc
    d'essai de sa substance sans qu'aucune erreur ne se manifeste.
    """
    base = f"{datetime.now().strftime(_HORODATAGE)}_{prefixe}"
    chemin = dossier / f"{base}{suffixe}"
    n = 1
    while chemin.exists():
        chemin = dossier / f"{base}-{n}{suffixe}"
        n += 1
    return chemin


def enregistrer_capture(image_b64: str, note: str = "") -> Path:
    """Archive une capture entière. Renvoie le chemin écrit."""
    d = dossier_archive()
    chemin = _horodate(d, "capture", ".png")
    chemin.write_bytes(_decoder(image_b64))
    if note:
        chemin.with_suffix(".txt").write_text(note, encoding="utf-8")
    return chemin


def enregistrer_echec(image_b64: str, diagnostic: dict,
                      dossier: Path | None = None) -> Path:
    """Archive une découpe qui n'a pas été lue avec certitude.

    Parameters
    ----------
    image_b64 : str
        La découpe telle qu'elle a été soumise au recogniseur — pas la
        capture entière : c'est l'entrée exacte qui a échoué qu'il faut
        pouvoir rejouer.
    diagnostic : dict
        Ce que le recogniseur a renvoyé (statut, distance, marge,
        best_guess), plus le contexte utile (cible héros/board).
    dossier : Path, optional
        Racine d'archive DÉJÀ résolue. Sert à l'archivage différé (thread) :
        le dossier est fixé au moment de la requête, pas au moment de
        l'écriture — sans quoi un environnement restauré entre-temps (les
        tests détournent ``LOCALAPPDATA``) ferait écrire ailleurs. Absent,
        comportement historique : ``dossier_archive()``.

    Returns
    -------
    Path
        Chemin de l'image écrite ; le diagnostic est à côté en ``.json``.
    """
    d = (dossier if dossier is not None else dossier_archive()) / "echecs"
    d.mkdir(parents=True, exist_ok=True)
    statut = str(diagnostic.get("statut", "refus"))
    chemin = _horodate(d, statut, ".png")
    chemin.write_bytes(_decoder(image_b64))
    chemin.with_suffix(".json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=1), encoding="utf-8")
    return chemin


@dataclass(frozen=True, slots=True)
class EchecArchive:
    """Une découpe archivée et son diagnostic."""

    image: Path
    diagnostic: dict

    @property
    def statut(self) -> str:
        return str(self.diagnostic.get("statut", "?"))

    @property
    def verite(self) -> str | None:
        """Carte réelle, si elle a été renseignée après coup.

        Un échec sans vérité-terrain permet de mesurer un taux ; avec elle,
        il permet de corriger. C'est le champ à remplir pour transformer
        l'archive en banc d'essai.
        """
        v = self.diagnostic.get("verite")
        return str(v) if v else None


def lister_echecs(limite: int | None = None) -> list[EchecArchive]:
    """Les découpes archivées, de la plus récente à la plus ancienne."""
    d = dossier_archive() / "echecs"
    if not d.is_dir():
        return []
    out: list[EchecArchive] = []
    for png in sorted(d.glob("*.png"), reverse=True):
        meta = png.with_suffix(".json")
        try:
            diag = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            diag = {}
        out.append(EchecArchive(image=png, diagnostic=diag))
        if limite is not None and len(out) >= limite:
            break
    return out
