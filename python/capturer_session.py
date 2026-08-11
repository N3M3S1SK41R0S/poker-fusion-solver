#!/usr/bin/env python
"""Constitue un jeu de captures RÉELLES pendant que l'utilisateur joue.

    python capturer_session.py                 toutes les tables, 30 min
    python capturer_session.py --minutes 90    plus longtemps
    python capturer_session.py --dossier D     ailleurs que dans l'archive

Ce que ça fait, et pourquoi
--------------------------
Tout ce que la perception sait d'elle-même a été mesuré sur des tables
**synthétiques**, et chaque confrontation au réel a démenti une hypothèse :
le bandeau d'équité qui masque le tiers bas d'une carte, un jeton de mise
posé sur le board, une étiquette de prime collée contre la carte du siège.
Aucune de ces trois occultations n'existait dans le banc.

Ce script ne corrige rien. Il **collecte** — plusieurs tables en parallèle,
pendant que la session se déroule — pour qu'on cesse de deviner.

Deux précautions qui font toute la différence :

* **Seules les images qui CHANGENT sont gardées.** Une table entre deux
  actions est figée ; sans ce filtre on archive quarante fois la même scène
  et le jeu de données ment sur sa propre diversité.
* **Rien n'est interprété ici.** Le script n'appelle ni le détecteur ni le
  recogniseur : il enregistre. Analyser en même temps qu'on collecte,
  c'est risquer de ne garder que ce que l'on sait déjà lire.

Le mode calibration (`calibrer.py`) lit et diagnostique ; celui-ci amasse la
matière. Les deux sont complémentaires et ne doivent pas être confondus.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision.archive import dossier_archive  # noqa: E402
from pfs.vision.live import (  # noqa: E402
    CaptureImpossible,
    SondeIntrouvable,
    capturer_fenetre,
    fenetres_disponibles,
)

#: Une fenêtre de TABLE porte les blindes dans son titre (« 75/150 »). Le
#: lobby et les fenêtres annexes du client n'en ont pas — c'est le seul
#: discriminant fiable observé, les titres de tables variant beaucoup.
_TABLE = re.compile(r"\b\d+\s*/\s*\d+\b")

#: Nom de dossier tiré du titre : on veut pouvoir retrouver de quelle table
#: vient une capture six mois plus tard, sans le titre complet en nom de
#: fichier (illisible, et plein de caractères interdits par Windows).
_PROPRE = re.compile(r"[^\w\-]+")


def tables_ouvertes() -> list[str]:
    """Titres des fenêtres qui sont réellement des tables de jeu."""
    return [t for t in fenetres_disponibles() if _TABLE.search(t)]


def _dossier(base: Path, titre: str) -> Path:
    court = _PROPRE.sub("_", titre.split("|")[0].strip())[:48]
    d = base / court
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="durée de la collecte (défaut 30)")
    ap.add_argument("--pause", type=float, default=2.5, metavar="S",
                    help="secondes entre deux tours de table (défaut 2,5)")
    ap.add_argument("--dossier", metavar="D",
                    help="où écrire ; défaut : <archive>/sessions")
    a = ap.parse_args()

    base = Path(a.dossier) if a.dossier else dossier_archive() / "sessions"
    base.mkdir(parents=True, exist_ok=True)

    try:
        tables = tables_ouvertes()
    except SondeIntrouvable as e:
        print(f"✗ {e}")
        return 2
    if not tables:
        print("Aucune table ouverte. Lance une partie, puis relance ce script.")
        return 1

    print(f"{len(tables)} table(s) suivie(s), {a.minutes:g} min :")
    for t in tables:
        print(f"   {t[:88]}")
    print(f"\nÉcriture dans {base}\n")

    vues: dict[str, set[str]] = {t: set() for t in tables}
    gardees: Counter[str] = Counter()
    echecs: Counter[str] = Counter()
    fin = time.monotonic() + a.minutes * 60

    try:
        while time.monotonic() < fin:
            for titre in tables:
                try:
                    png = capturer_fenetre(titre, timeout_s=6)
                except CaptureImpossible:
                    echecs[titre] += 1
                    continue
                # L'empreinte évite de garder deux fois la même scène : une
                # table figée entre deux actions produit des octets
                # identiques, et gonflerait le jeu de données sans rien y
                # ajouter.
                empreinte = hashlib.sha1(png).hexdigest()
                if empreinte in vues[titre]:
                    continue
                vues[titre].add(empreinte)
                n = gardees[titre]
                (_dossier(base, titre) / f"{n:04d}.png").write_bytes(png)
                gardees[titre] = n + 1
            reste = fin - time.monotonic()
            total = sum(gardees.values())
            print(f"\r  {total} frame(s) distincte(s) · "
                  f"{reste / 60:.1f} min restantes   ", end="", flush=True)
            time.sleep(a.pause)
    except KeyboardInterrupt:
        print("\n  interrompu.")

    print("\n")
    for titre in tables:
        print(f"  {gardees[titre]:>4} frames · {echecs[titre]:>3} échecs · "
              f"{titre[:70]}")
    print(f"\n{sum(gardees.values())} frames au total dans {base}")
    print("Ensuite : python calibrer.py, ou rejoue-les contre le détecteur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
