#!/usr/bin/env python
"""Reconnaît des cartes dans une image — et, si tu donnes le contexte du
spot, enchaîne sur le conseil « que fallait-il faire ? ».

Une seule carte :

    python reconnaitre.py --card capture_carte.png

Plusieurs cartes par régions d'intérêt (x,y,largeur,hauteur ; séparées par
« ; ») dans une capture :

    python reconnaitre.py --image table.png --rois "120,300,60,80; 190,300,60,80"

Reconnaître le héros + le board puis conseiller (les ROI héros d'abord, board
ensuite ; ajoute le contexte du pot) :

    python reconnaitre.py --image table.png \
        --hero-rois "120,300,60,80; 190,300,60,80" \
        --board-rois "400,180,60,80; 470,180,60,80; 540,180,60,80" \
        --pot 100 --bet 75 --bb 10

⚠️ Les coordonnées des ROI dépendent de la room et de la résolution : à
mesurer une fois sur une vraie capture (le recogniseur, lui, est calibré sur
le deck PMU et robuste à l'échelle).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision import identify_card, recognize_cards  # noqa: E402


def _parse_rois(spec: str) -> list[tuple[int, int, int, int]]:
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [int(v) for v in chunk.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError(f"ROI « {chunk} » : attendu x,y,largeur,hauteur")
        out.append(tuple(vals))
    return out


def _show(matches, labels=None) -> None:
    for i, m in enumerate(matches):
        tag = f"{labels[i]} " if labels else ""
        if m.card:
            print(f"  {tag}{m.card}  (distance {m.distance}, "
                  f"confiance {m.confidence:.0%})")
        else:
            print(f"  {tag}?? illisible  (meilleur : {m.runner_up}, "
                  f"distance {m.distance}, marge {m.margin})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Reconnaissance de cartes depuis une image.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--card", help="image d'UNE carte")
    ap.add_argument("--image", help="image contenant plusieurs cartes")
    ap.add_argument("--rois", help="ROI génériques « x,y,w,h; ... »")
    ap.add_argument("--hero-rois", help="ROI des cartes du héros")
    ap.add_argument("--board-rois", help="ROI des cartes du board")
    ap.add_argument("--pot", type=float, default=0.0)
    ap.add_argument("--bet", type=float, default=0.0)
    ap.add_argument("--stack", type=float, default=0.0)
    ap.add_argument("--bb", type=float, default=1.0)
    ap.add_argument("--position", default="BTN")
    ap.add_argument("--villain", default="moyenne")
    args = ap.parse_args()

    if args.card:
        print("Carte reconnue :")
        _show([identify_card(args.card)])
        return 0

    if not args.image:
        ap.print_help()
        return 1

    if args.hero_rois and args.board_rois:
        hero = recognize_cards(args.image, _parse_rois(args.hero_rois))
        board = recognize_cards(args.image, _parse_rois(args.board_rois))
        print("Héros :")
        _show(hero)
        print("Board :")
        _show(board)
        if any(m.card is None for m in hero + board):
            print("\n⚠️ Une carte n'a pas été lue avec certitude — vérifie les ROI.")
            return 2
        # enchaîne sur le conseil
        from pfs.analysis import Spot, advise
        spot = Spot(
            hero=" ".join(m.card for m in hero),
            board=" ".join(m.card for m in board),
            pot=args.pot, bet=args.bet, stack=args.stack,
            big_blind=args.bb, position=args.position, villain=args.villain,
        )
        print()
        print(advise(spot).explain())
        return 0

    if args.rois:
        print("Cartes reconnues :")
        _show(recognize_cards(args.image, _parse_rois(args.rois)))
        return 0

    print("Image entière traitée comme une seule carte :")
    _show([identify_card(args.image)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
