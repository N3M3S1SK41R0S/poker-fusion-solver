#!/usr/bin/env python
"""Une recherche locale du cadrage est-elle sûre, ou fabrique-t-elle des faux ?

    python banc_cadrage.py            le verdict
    python banc_cadrage.py --grille 5 grille plus fine (plus lent)

Le problème
-----------
Sur la première capture réelle d'une table PMU, le détecteur rend une boîte
de 126 px de large pour une carte de 121, décalée de 7 px. Cette boîte-là se
lit « refus » à un écart de 681 ; la boîte exacte, elle, donne « 3h » à 209
avec 149 de marge. Sept pixels séparent la lecture parfaite du refus, et
`autocrop_card` n'y peut rien : il isole une carte ENTOURÉE de feutre, alors
qu'ici la découpe est déjà presque toute la carte — il n'a plus d'anneau de
référence et rend ``None``.

La tentation est d'essayer plusieurs cadrages voisins et de garder le
meilleur. Mais prendre le MINIMUM sur N essais, c'est N chances de tomber
par hasard sous le seuil : c'est le problème des comparaisons multiples, et
il se paie en cartes inventées — exactement ce que `DISTANCE_SURE` vient
d'éliminer.

Ce banc mesure les deux côtés avant de décider :
  * ce que la recherche RÉCUPÈRE sur les deux vraies cartes de la capture ;
  * ce qu'elle COÛTE, en mesurant le plancher de bruit sous la même
    recherche. Si le minimum sur la grille descend sous le seuil, la
    recherche est à rejeter, quel que soit son gain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision.archive import dossier_archive  # noqa: E402
from pfs.vision.card_recognizer import (  # noqa: E402
    DISTANCE_SURE,
    identify_card,
)

#: Vérité-terrain lue à l'œil sur la capture du 10 août 2026.
CARTES_REELLES = {
    "5c": (816, 902, 126, 115),   # boîte rendue par le détecteur
    "3h": (934, 902, 126, 115),
}
FEUTRES = ((24, 86, 52), (38, 110, 74), (18, 32, 70),
           (92, 26, 38), (84, 84, 90), (16, 16, 20))


def _grille(pas: int):
    """Décalages et redimensionnements essayés autour de la boîte."""
    d = [-2 * pas, -pas, 0, pas, 2 * pas]
    t = [-2 * pas, 0, 2 * pas]
    return [(dx, dy, dw, dh) for dx in d for dy in d for dw in t for dh in t]


def _meilleur(image, boite, grille):
    """Meilleure lecture sur la grille : (écart, carte, marge, essais)."""
    x, y, w, h = boite
    best = None
    for dx, dy, dw, dh in grille:
        xx, yy = x + dx, y + dy
        ww, hh = w + dw, h + dh
        if ww < 30 or hh < 30 or xx < 0 or yy < 0:
            continue
        if xx + ww > image.width or yy + hh > image.height:
            continue
        m = identify_card(image.crop((xx, yy, xx + ww, yy + hh)))
        if best is None or m.distance < best[0]:
            best = (m.distance, m.best_guess or m.card, m.margin, m.statut)
    return best


def _non_cartes(n: int):
    """Découpes qui ne sont pas des cartes, sur fond de feutre."""
    rng = np.random.default_rng(20260810)
    for i in range(n):
        h, w = int(rng.integers(90, 140)), int(rng.integers(90, 140))
        if i % 3 == 0:
            a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            img = Image.fromarray(a)
        elif i % 3 == 1:
            c = FEUTRES[i % len(FEUTRES)]
            a = np.tile(np.array(c, dtype=np.uint8), (h, w, 1))
            a = np.clip(a.astype(int) + rng.normal(0, 5, a.shape), 0, 255)
            img = Image.fromarray(a.astype(np.uint8))
        else:
            img = Image.new("RGB", (w, h),
                            tuple(int(v) for v in rng.integers(60, 200, 3)))
            d = ImageDraw.Draw(img)
            for k in range(0, w + h, 7):
                d.line([(k, 0), (0, k)],
                       fill=tuple(int(v) for v in rng.integers(0, 120, 3)),
                       width=2)
        yield img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pas", type=int, default=3,
                    help="pas de la grille en pixels (défaut 3)")
    ap.add_argument("--faux", type=int, default=60,
                    help="nombre de non-cartes testées (défaut 60)")
    a = ap.parse_args()

    grille = _grille(a.pas)
    print(f"grille de {len(grille)} cadrages (pas {a.pas} px), "
          f"seuil d'affirmation {DISTANCE_SURE}\n")

    capture = sorted(dossier_archive().glob("*.png"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not capture:
        print("aucune capture archivée — rien à mesurer.")
        return 1
    image = Image.open(capture[0]).convert("RGB")
    print(f"capture : {capture[0].name} {image.width}×{image.height}\n")

    print("── CE QUE LA RECHERCHE RÉCUPÈRE (vraies cartes)")
    gagnees = 0
    for attendu, boite in CARTES_REELLES.items():
        direct = identify_card(image.crop(
            (boite[0], boite[1], boite[0] + boite[2], boite[1] + boite[3])))
        best = _meilleur(image, boite, grille)
        juste = best is not None and best[1] == attendu
        gagnees += bool(juste and direct.card != attendu)
        print(f"  {attendu} : direct {str(direct.card or direct.best_guess):>4} "
              f"[{direct.statut}] écart {direct.distance}"
              f"   →   recherche {str(best[1]):>4} [{best[3]}] écart {best[0]} "
              f"marge {best[2]}   {'✓' if juste else '✗'}")

    print(f"\n── CE QU'ELLE COÛTE ({a.faux} non-cartes, "
          f"minimum sur la même grille)")
    planchers, marges, affirmees = [], [], 0
    for img in _non_cartes(a.faux):
        w, h = img.size
        best = _meilleur(img, (0, 0, w, h), grille)
        if not best:
            continue
        planchers.append(best[0])
        marges.append(best[2])
        # Le critère qui compte vraiment : cette non-carte serait-elle
        # AFFIRMÉE ? La distance seule ne suffit pas à le dire, la marge
        # doit aussi franchir MARGE_SURE.
        if best[0] <= DISTANCE_SURE and best[3] == "sure":
            affirmees += 1
    p, mg = np.array(planchers), np.array(marges)
    sous = int((p <= DISTANCE_SURE).sum())
    print(f"  écart minimal : {p.min()}   p5 {int(np.percentile(p, 5))}   "
          f"médiane {int(np.median(p))}")
    print(f"  marge au minimum : max {mg.max()}   p95 "
          f"{int(np.percentile(mg, 95))}   médiane {int(np.median(mg))}")
    print(f"  non-cartes sous le seuil {DISTANCE_SURE} : {sous} / {len(p)}")
    print(f"  non-cartes AFFIRMÉES (distance ET marge) : "
          f"{affirmees} / {len(p)}")

    print()
    if affirmees:
        print("  VERDICT : recherche à REJETER — elle affirme des cartes")
        print("  sur des découpes qui n'en sont pas.")
        return 1
    if sous:
        print("  VERDICT : recherche à REJETER. Elle fabrique des lectures")
        print("  affirmées sur des découpes qui ne sont pas des cartes.")
        return 1
    if gagnees:
        print(f"  VERDICT : recherche RECEVABLE — {gagnees} carte(s) "
              "récupérée(s), plancher de bruit intact.")
        return 0
    print("  VERDICT : recherche sans gain mesuré ici ; ne pas l'ajouter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
