#!/usr/bin/env python
"""Calibration en direct du recogniseur, sur une vraie table à l'écran.

    python calibrer.py --fenetres          liste les fenêtres capturables
    python calibrer.py                     une lecture (détection auto)
    python calibrer.py --fenetre "PMU"     une lecture d'une fenêtre nommée
    python calibrer.py --boucle 20         20 lectures espacées de 3 s

Chaque lecture affiche UNIQUEMENT ce que la machine a vu, avec la confiance
de chaque carte. Aucune recommandation n'est produite : ce script est un
banc d'essai de la perception, pas une assistance de jeu (voir l'en-tête de
`pfs/vision/live.py`, et `tests/test_live_sans_conseil.py` qui le vérifie).

Les cartes mal lues sont archivées automatiquement. Une fois la session
finie :

    python banc_echecs.py                  ce qui a échoué
    python banc_echecs.py --annoter Ah <fichier>
    python banc_echecs.py --rejouer        après correction, mesure le gain
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision.archive import dossier_archive  # noqa: E402
from pfs.vision.live import (  # noqa: E402
    CaptureImpossible,
    SondeIntrouvable,
    fenetres_disponibles,
    lire_ecran,
)

_COULEUR = {"sure": "\x1b[32m", "propose": "\x1b[33m", "refus": "\x1b[31m"}
_RAZ = "\x1b[0m"


def _lister() -> int:
    fenetres = fenetres_disponibles()
    if not fenetres:
        print("aucune fenêtre capturable.")
        return 1
    print(f"{len(fenetres)} fenêtre(s) — les tables probables d'abord :\n")
    for t in fenetres[:40]:
        print(f"  {t}")
    return 0


def _une_lecture(fenetre: str | None, numero: int | None = None) -> bool:
    """Une capture + lecture. Renvoie False si la capture a échoué."""
    t0 = time.perf_counter()
    try:
        lecture = lire_ecran(fenetre)
    except CaptureImpossible as e:
        print(f"\x1b[31m✗\x1b[0m {e}")
        return False
    ms = (time.perf_counter() - t0) * 1000

    entete = f"[{numero}] " if numero is not None else ""
    print(f"\n{entete}{lecture.fenetre}  "
          f"{lecture.largeur}×{lecture.hauteur}  {ms:.0f} ms")
    if not lecture.cartes:
        print("  aucune carte détectée — le détecteur n'a rien trouvé qui "
              "ressemble à une carte.")
        return True

    for c in lecture.cartes:
        couleur = _COULEUR.get(c.statut, "")
        lu = c.carte or f"— (candidat {c.candidat})"
        print(f"  {couleur}{c.statut:<8}{_RAZ} {c.role:<6} {lu:<22}"
              f" écart {c.ecart}  marge {c.marge}"
              f"   @{c.boite[0]},{c.boite[1]} {c.boite[2]}×{c.boite[3]}")
    print(f"  → {lecture.sures}/{len(lecture.cartes)} lues avec certitude "
          f"({lecture.taux:.0%})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fenetres", action="store_true",
                    help="liste les fenêtres capturables et sort")
    ap.add_argument("--fenetre", metavar="TITRE",
                    help="sous-chaîne du titre ; défaut : détection auto")
    ap.add_argument("--boucle", type=int, metavar="N", default=1,
                    help="nombre de lectures (défaut 1)")
    ap.add_argument("--pause", type=float, default=3.0, metavar="S",
                    help="secondes entre deux lectures (défaut 3)")
    a = ap.parse_args()

    try:
        if a.fenetres:
            return _lister()

        if a.boucle <= 1:
            return 0 if _une_lecture(a.fenetre) else 1

        print(f"{a.boucle} lectures, une toutes les {a.pause:g} s. "
              "Ctrl-C pour arrêter.")
        echecs = 0
        for i in range(1, a.boucle + 1):
            if not _une_lecture(a.fenetre, i):
                echecs += 1
                if echecs >= 3:
                    print("\n3 captures de suite ont échoué — j'arrête.")
                    return 1
            else:
                echecs = 0
            if i < a.boucle:
                time.sleep(a.pause)
        print(f"\nÉchecs archivés dans {dossier_archive() / 'echecs'}")
        print("Ensuite : python banc_echecs.py")
        return 0

    except SondeIntrouvable as e:
        print(f"\x1b[31m✗\x1b[0m {e}")
        return 2
    except KeyboardInterrupt:
        print("\ninterrompu.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
