#!/usr/bin/env python
"""Banc de la LOCALISATION : faut-il réduire l'image avant de chercher ?

    python banc_localisation.py                 le balayage de référence
    python banc_localisation.py --large         + thèmes, bruits, tailles
    python banc_localisation.py --villains      avec des adversaires à table

Pourquoi ce fichier existe
--------------------------
Les chiffres de `LARGEUR_DETECTION` (pfs/vision/live.py) ont d'abord été
produits par un script jetable, hors dépôt. Ils étaient donc invérifiables,
et deux d'entre eux étaient faux — dont un qui justifiait une décision de
conception. Un chiffre publié sans le banc qui le produit n'est pas une
mesure, c'est une affirmation.

Définitions, qui décident du résultat
-------------------------------------
* **localisée** : une carte de la vérité-terrain (héros + board) recouverte
  à ≥ 50 % par une boîte détectée.
* **rôle juste** : localisée par une boîte portant le bon rôle.
* **fantôme** : une boîte qui ne recouvre **aucune carte du tableau**, dos
  adverses compris. Comparer aux seules cartes héros/board compterait les
  dos des adversaires comme des fantômes — sur une table réaliste, cela
  gonfle le compte de plusieurs dizaines et rend la mesure absurde.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision.live import _agrandir, _reduire  # noqa: E402
from pfs.vision.synth_table import (  # noqa: E402
    FELTS,
    TableSpec,
    render_table,
)
from pfs.vision.table_detector import CardBox, read_table  # noqa: E402

TAILLE = (2560, 1529)
LARGEURS = (640, 800, 960, 1280, 1600, 0)   # 0 = pleine échelle
MAINS = (("Ah", "Kd"), ("7c", "7s"), ("Qs", "Jh"))
BOARDS = ((), ("Ts", "4h", "9d"), ("Ts", "4h", "9d", "2c", "Kh"))


@dataclass(frozen=True, slots=True)
class Config:
    """Une famille de tables : c'est l'axe que le balayage doit couvrir."""

    nom: str
    theme: str = "pmu_solid"
    carte: tuple[int, int] = (68, 92)
    noise: float = 0.0
    blur: float = 0.0
    villains: int = 0


REFERENCE = (Config("référence (pmu_solid 68×92)"),)

LARGE = REFERENCE + (
    Config("deck classique", theme="pmu_deck"),
    Config("bruité (σ6, flou 0,6)", noise=6.0, blur=0.6),
    Config("petites cartes 52×70", carte=(52, 70)),
    Config("grandes cartes 80×108", carte=(80, 108)),
)


def _boite(v) -> CardBox:
    if isinstance(v, CardBox):
        return v
    x, y, w, h = v
    return CardBox(x=x, y=y, w=w, h=h)


def _recouvre(a: CardBox, b: CardBox) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    return inter / min(a.w * a.h, b.w * b.h) if inter else 0.0


def _tables(cfg: Config, graine: int):
    for i, main in enumerate(MAINS):
        for j, board in enumerate(BOARDS):
            for k, felt in enumerate(FELTS):
                yield render_table(TableSpec(
                    hero=main, board=board, felt=felt, size=TAILLE,
                    theme=cfg.theme, card_w=cfg.carte[0], card_h=cfg.carte[1],
                    noise=cfg.noise, blur=cfg.blur, villains=cfg.villains,
                    decor=True, seed=graine + i * 17 + j * 5 + k))


def _mesurer(cfg: Config, largeur: int, graine: int) -> dict:
    attendues = trouvees = roles = fantomes = boites_tot = 0
    duree = 0.0
    n = 0

    for st in _tables(cfg, graine):
        n += 1
        image = st.image
        verite = [("hero", _boite(v)) for v in st.hero_boxes]
        verite += [("board", _boite(v)) for v in st.board_boxes]
        # Les dos adverses ne sont pas à reconnaître, mais une boîte posée
        # dessus n'est pas un fantôme pour autant : c'est bien une carte.
        toutes = [b for _, b in verite]
        toutes += [_boite(v) for v in getattr(st, "villain_boxes", []) or []]
        attendues += len(verite)

        t0 = time.perf_counter()
        if largeur:
            reduite, k = _reduire(image, largeur)
            lu = read_table(reduite)
            remise = lambda b: _agrandir(b, k, image.width, image.height)  # noqa: E731,B023
        else:
            lu = read_table(image)
            remise = lambda b: b  # noqa: E731
        duree += time.perf_counter() - t0

        boites = [(r, remise(b)) for r in ("hero", "board", "others")
                  for b in getattr(lu, r)]
        boites_tot += len(boites)

        prises: set[int] = set()
        for role_vrai, cible in verite:
            meilleur, score = None, 0.0
            for idx, (_, bb) in enumerate(boites):
                if idx in prises:
                    continue
                s = _recouvre(cible, bb)
                if s > score:
                    meilleur, score = idx, s
            if meilleur is not None and score >= 0.5:
                prises.add(meilleur)
                trouvees += 1
                if boites[meilleur][0] == role_vrai:
                    roles += 1

        fantomes += sum(
            1 for _, bb in boites
            if max((_recouvre(v, bb) for v in toutes), default=0.0) < 0.5)

    return {"tables": n, "cartes": attendues,
            "loc": trouvees / attendues if attendues else 0.0,
            "roles": roles / attendues if attendues else 0.0,
            "fantomes": fantomes, "boites": boites_tot,
            "ms": duree / n * 1000 if n else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large", action="store_true",
                    help="balaie aussi thèmes, bruits et tailles de carte")
    ap.add_argument("--villains", type=int, default=0, metavar="N",
                    help="dessine N adversaires (cartes face cachée)")
    ap.add_argument("--graine", type=int, default=20260810)
    a = ap.parse_args()

    configs = LARGE if a.large else REFERENCE
    if a.villains:
        configs = tuple(Config(f"{c.nom} + {a.villains} adversaires",
                               c.theme, c.carte, c.noise, c.blur, a.villains)
                        for c in configs)

    print(f"tables {TAILLE[0]}×{TAILLE[1]}, graine {a.graine}, "
          f"{len(MAINS)}×{len(BOARDS)}×{len(FELTS)} = "
          f"{len(MAINS) * len(BOARDS) * len(FELTS)} tables par configuration\n")

    for cfg in configs:
        print(f"── {cfg.nom}")
        print(f"{'largeur':>9}{'carte':>8}{'localisées':>12}{'rôles':>9}"
              f"{'fantômes':>10}{'boîtes':>8}{'temps':>10}")
        for largeur in LARGEURS:
            r = _mesurer(cfg, largeur, a.graine)
            px = round(cfg.carte[1] * largeur / TAILLE[0]) if largeur \
                else cfg.carte[1]
            nom = f"{largeur}" if largeur else "pleine"
            print(f"{nom:>9}{px:>7} px{r['loc']:>11.1%}{r['roles']:>9.1%}"
                  f"{r['fantomes']:>10}{r['boites']:>8}{r['ms']:>7.0f} ms")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
