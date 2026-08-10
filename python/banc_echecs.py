#!/usr/bin/env python
"""Banc d'essai des ÉCHECS réels de reconnaissance.

L'archive (`pfs.vision.archive`) conserve chaque découpe que le recogniseur
n'a pas lue avec certitude, telle qu'elle lui a été soumise. Ce script en
fait un banc de mesure — le seul qui porte sur des captures réelles.

    python banc_echecs.py                  état de l'archive
    python banc_echecs.py --rejouer        re-soumet chaque échec au
                                           recogniseur actuel et compare
    python banc_echecs.py --annoter Ah 20260810-123005_refus.png
                                           inscrit la vérité-terrain

Annoter est ce qui transforme l'archive en mesure : sans vérité, on connaît
le taux de refus ; avec elle, on sait si le refus était justifié.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision import identify_card  # noqa: E402
from pfs.vision.archive import dossier_archive, lister_echecs  # noqa: E402


def etat() -> int:
    echecs = lister_echecs()
    d = dossier_archive()
    captures = list(d.glob("*.png"))
    print(f"Archive : {d}")
    print(f"  captures entières   : {len(captures)}")
    print(f"  découpes non lues   : {len(echecs)}")
    if not echecs:
        print("\n  Rien à travailler pour l'instant. Colle des captures dans")
        print("  l'onglet « Ma main » : tout ce qui échoue atterrit ici.")
        return 0
    refus = [e for e in echecs if e.statut == "refus"]
    prop = [e for e in echecs if e.statut == "propose"]
    annotes = [e for e in echecs if e.verite]
    print(f"    · refusées        : {len(refus)}")
    print(f"    · proposées       : {len(prop)}")
    print(f"    · annotées        : {len(annotes)}  "
          f"(vérité-terrain renseignée)")
    print("\n  Dernières :")
    for e in echecs[:12]:
        v = f"  vérité {e.verite}" if e.verite else ""
        print(f"    {e.image.name:<34} {e.statut:<8} "
              f"candidat {str(e.diagnostic.get('best_guess')):>4} "
              f"écart {e.diagnostic.get('distance')} "
              f"marge {e.diagnostic.get('margin')}{v}")
    if not annotes:
        print("\n  Aucune n'est annotée : lance")
        print("    python banc_echecs.py --annoter <carte> <fichier>")
        print("  pour dire ce que c'était réellement (ex. « Ah »).")
    return 0


def rejouer() -> int:
    """Re-soumet chaque échec au recogniseur actuel."""
    echecs = lister_echecs()
    if not echecs:
        print("Archive vide — rien à rejouer.")
        return 0
    annotes = [e for e in echecs if e.verite]
    print(f"{len(echecs)} découpe(s) rejouée(s) "
          f"({len(annotes)} avec vérité-terrain)\n")
    print(f"{'fichier':<34} {'avant':<8} {'maintenant':<10} {'verdict'}")
    mieux = pire = juste = faux = 0
    for e in echecs:
        m = identify_card(e.image)
        avant = e.statut
        apres = m.statut
        note = ""
        if e.verite:
            if m.card == e.verite:
                juste += 1
                note = "✓ lue correctement"
            elif m.best_guess == e.verite:
                note = f"~ bon candidat ({apres})"
            elif m.card is not None:
                faux += 1
                note = f"✗ LUE FAUX : {m.card} au lieu de {e.verite}"
            else:
                note = "— toujours refusée"
        rang = {"refus": 0, "propose": 1, "sure": 2}
        if rang[apres] > rang[avant]:
            mieux += 1
        elif rang[apres] < rang[avant]:
            pire += 1
        print(f"{e.image.name:<34} {avant:<8} {apres:<10} {note}")
    print(f"\n  progressent : {mieux}   ·   régressent : {pire}")
    if annotes:
        print(f"  sur les {len(annotes)} annotées : {juste} lues correctement, "
              f"{faux} LUES FAUX")
    return 0


def annoter(carte: str, fichier: str) -> int:
    d = dossier_archive() / "echecs"
    png = d / fichier
    if not png.exists():
        print(f"introuvable : {png}")
        return 1
    meta = png.with_suffix(".json")
    diag = {}
    if meta.exists():
        diag = json.loads(meta.read_text(encoding="utf-8"))
    diag["verite"] = carte
    meta.write_text(json.dumps(diag, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"{fichier} : vérité-terrain = {carte}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rejouer", action="store_true")
    ap.add_argument("--annoter", nargs=2, metavar=("CARTE", "FICHIER"))
    a = ap.parse_args()
    if a.annoter:
        return annoter(*a.annoter)
    if a.rejouer:
        return rejouer()
    return etat()


if __name__ == "__main__":
    raise SystemExit(main())
