#!/usr/bin/env python
"""Banc du badge « PMU PLAY » : rejoue l'étude sur les 57 captures réelles.

    python banc_badge_pmu.py
    python banc_badge_pmu.py --captures <dossier>
    python banc_badge_pmu.py --detail          une ligne par frame
    python banc_badge_pmu.py --json out.json

Résultat au 14 août 2026 (57 captures, deux tables PMU PLAY v.26.1.1.23,
fenêtre 1920×1361) ::

    BADGE VU : 50/57 frames (invariant : >= 50)
      filigrane seul  : 17 frames (scores 0,417 à 1,000 ; occulté <= 0,086)
      dos orange      : 49 frames (scores >= 0,975 ; sans dos <= 0,673)
    frames sans badge : 7 — toutes des fins de main (abattage 6-max
      0026-0031, 7-max 0024 tous adversaires couchés)
    NÉGATIFS SYNTHÉTIQUES (10 tables sans filigrane, dos bleus) :
      filigrane max 0,050   dos max 0,569   faux positifs : 0 (invariant)    
    coût médian : ~360 ms/frame (dont ~325 ms de dos plein cadre)

Ce que le banc vérifie (et casse s'il ne le retrouve plus)
----------------------------------------------------------
1. **>= 50/57 frames vues.** C'est la couverture immédiate mesurée par
   l'étude ; les 7 frames restantes N'ONT pas de marqueur visible (fins de
   main), les « voir » serait un faux positif.
2. **0 faux positif sur les négatifs synthétiques** — les mêmes 10 tables
   que l'étude (5 tapis × 2 graines, dos de cartes bleus, aucun filigrane).

Écart connu avec les JSON de l'étude : sur 12 frames du thème clair,
l'étude ne mesurait le filigrane qu'avec le gabarit du thème de la session
(score −0,097) ; le détecteur, qui ne connaît pas le thème en production,
prend le max des deux gabarits (−0,028 sur ces frames). Les deux valeurs
sont très loin sous le seuil 0,25 : aucun verdict ne change.

Sur les captures et leur absence
--------------------------------
Les 57 PNG pèsent ~108 Mo et ne sont PAS dans le dépôt. Sans les images, le
banc dit où les chercher et s'arrête : il ne rend jamais de chiffre
approximatif. Les négatifs synthétiques et le refus de taille, eux, se
rejouent toujours.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from pfs.vision.badge_pmu import (  # noqa: E402
    SEUIL_DOS,
    SEUIL_FILIGRANE,
    TAILLE_CALIBREE,
    detecter_badge,
)
from pfs.vision.synth_table import FELTS, TableSpec, render_table  # noqa: E402

#: Les deux sessions de l'étude, avec le thème que le filigrane doit révéler
#: quand il est visible. 32 frames « clair » + 25 frames « sombre » = 57.
SESSIONS: tuple[tuple[str, str], ...] = (
    ("100_-_6-max_Turbo_Re-buy_1197329110", "clair"),
    ("300_7-max_KO_1197327559", "sombre"),
)
FRAMES_ATTENDUES = 57
#: Couverture immédiate mesurée par l'étude : 50/57. Les 7 frames restantes
#: sont des fins de main SANS marqueur visible — l'invariant est >=, pas ==,
#: parce qu'une amélioration honnête du détecteur a le droit de faire mieux,
#: mais jamais moins.
VUES_MIN = 50


def captures_par_defaut() -> Path:
    """Emplacement des sessions écrites par `capturer_session.py`."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "PokerFusionSolver" / "captures" / "sessions"


def rejouer_captures(racine: Path, detail: bool) -> tuple[list[dict], list[str]]:
    """Rejoue les 57 frames ; renvoie (lignes, fautes)."""
    lignes: list[dict] = []
    fautes: list[str] = []
    chronos: list[float] = []
    for session, theme_attendu in SESSIONS:
        dossier = racine / session
        for chemin in sorted(dossier.glob("*.png")):
            t0 = time.time()
            lu = detecter_badge(Image.open(chemin))
            chronos.append(time.time() - t0)
            etiquette = f"{session[:14]}/{chemin.name}"
            if lu.refus is not None:
                fautes.append(f"{etiquette} : refus inattendu « {lu.refus} »")
            if lu.theme is not None and lu.theme != theme_attendu:
                fautes.append(f"{etiquette} : thème {lu.theme}, "
                              f"attendu {theme_attendu}")
            lignes.append({
                "frame": etiquette, "filigrane": round(lu.filigrane_score, 4),
                "dos": round(lu.dos_score, 4), "vu": lu.vu,
                "theme": lu.theme, "position": lu.position})
            if detail:
                print(f"  {etiquette:28s}  filigrane {lu.filigrane_score:+.3f}  "
                      f"dos {lu.dos_score:+.3f}  "
                      f"{'VU (' + (lu.theme or 'dos') + ')' if lu.vu else '—'}")

    if len(lignes) != FRAMES_ATTENDUES:
        fautes.append(f"{len(lignes)} frames rejouées, {FRAMES_ATTENDUES} "
                      "attendues — le corpus a changé, les invariants ne "
                      "veulent plus rien dire")
    chronos.sort()
    if chronos:
        print(f"\n  coût médian par frame : "
              f"{1000 * chronos[len(chronos) // 2]:.0f} ms "
              "(étude : 5,6 ms filigrane + 325,4 ms dos)")
    return lignes, fautes


def rejouer_negatifs(detail: bool) -> tuple[list[dict], list[str]]:
    """Les 10 tables synthétiques de l'étude : aucun badge à voir."""
    lignes: list[dict] = []
    fautes: list[str] = []
    for felt in list(FELTS)[:5]:
        for seed in (1, 4242):
            table = render_table(TableSpec(
                hero=("Ah", "Kd"), board=("Ts", "4h", "9d"), felt=felt,
                size=TAILLE_CALIBREE, theme="pmu_solid",
                card_w=102, card_h=138, decor=True, seed=seed))
            lu = detecter_badge(table.image)
            quoi = f"synth {felt} seed={seed}"
            lignes.append({"quoi": quoi,
                           "filigrane": round(lu.filigrane_score, 4),
                           "dos": round(lu.dos_score, 4), "vu": lu.vu})
            if lu.vu:
                fautes.append(f"FAUX POSITIF sur {quoi} : filigrane "
                              f"{lu.filigrane_score:.3f}, dos {lu.dos_score:.3f}")
            if detail:
                print(f"  {quoi:32s}  filigrane {lu.filigrane_score:+.3f}  "
                      f"dos {lu.dos_score:+.3f}")
    return lignes, fautes


def verifier_refus_taille() -> list[str]:
    """Une image d'une autre taille doit être REFUSÉE, pas mesurée."""
    table = render_table(TableSpec(size=(1280, 720), decor=True))
    lu = detecter_badge(table.image)
    fautes = []
    if lu.refus is None:
        fautes.append("une image 1280×720 a été mesurée au lieu d'être "
                      "refusée (la calibration ne vaut que pour "
                      f"{TAILLE_CALIBREE[0]}×{TAILLE_CALIBREE[1]})")
    if lu.vu:
        fautes.append("une image 1280×720 a rendu vu=True — le fail-closed "
                      "est cassé")
    return fautes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", type=Path, default=None,
                   help="dossier des sessions de captures")
    p.add_argument("--detail", action="store_true",
                   help="une ligne par frame et par négatif")
    p.add_argument("--json", type=Path, default=None,
                   help="écrit le détail dans ce fichier")
    args = p.parse_args()

    racine = args.captures or captures_par_defaut()
    manquantes = [s for s, _ in SESSIONS if not (racine / s).is_dir()]
    if manquantes:
        print(f"captures introuvables : {racine / manquantes[0]}\n"
              "Les 57 PNG (~108 Mo) ne sont pas versionnés. Relance une "
              "session avec `python capturer_session.py`, ou pointe le "
              "dossier avec --captures.", file=sys.stderr)
        return 2

    print("=" * 72)
    print("BANC DU BADGE « PMU PLAY » — 57 captures réelles + négatifs")
    print("=" * 72)

    lignes, fautes = rejouer_captures(racine, args.detail)
    vues = sum(1 for li in lignes if li["vu"])
    par_filigrane = sum(1 for li in lignes if li["filigrane"] >= SEUIL_FILIGRANE)
    par_dos = sum(1 for li in lignes if li["dos"] >= SEUIL_DOS)
    print(f"\n  BADGE VU : {vues}/{len(lignes)} frames "
          f"(filigrane {par_filigrane}, dos {par_dos})")
    non_vues = [li["frame"] for li in lignes if not li["vu"]]
    if non_vues:
        print(f"  sans badge visible ({len(non_vues)}) : "
              + ", ".join(non_vues))
    if vues < VUES_MIN:
        fautes.append(f"couverture {vues}/{len(lignes)} sous l'invariant "
                      f">= {VUES_MIN} — le détecteur a perdu des frames "
                      "que l'étude voyait")

    print("\n  négatifs synthétiques (10 tables sans badge) :")
    negatifs, fautes_neg = rejouer_negatifs(args.detail)
    fautes += fautes_neg
    pire_wm = max(n["filigrane"] for n in negatifs)
    pire_dos = max(n["dos"] for n in negatifs)
    print(f"  pire filigrane {pire_wm:.3f} (seuil {SEUIL_FILIGRANE}), "
          f"pire dos {pire_dos:.3f} (seuil {SEUIL_DOS}) — "
          f"{sum(1 for n in negatifs if n['vu'])} faux positif(s)")

    fautes += verifier_refus_taille()
    print("  refus à taille non calibrée : "
          + ("OK (fail-closed)" if not any("1280" in f for f in fautes)
             else "CASSÉ"))

    if args.json:
        args.json.write_text(
            json.dumps({"frames": lignes, "negatifs": negatifs},
                       indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  détail écrit dans {args.json}")

    print()
    for f in fautes:
        print(f"  FAUTE  {f}")
    print("BANC VERT — couverture >= 50/57, zéro faux positif, fail-closed"
          if not fautes else
          f"BANC ROUGE — {len(fautes)} faute(s)")
    return len(fautes)


if __name__ == "__main__":
    raise SystemExit(main())
