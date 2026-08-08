#!/usr/bin/env python
"""Analyse d'une main jouée — « qu'est-ce qu'il fallait faire ? »

Pensé pour dépouiller une pile de captures d'écran : tu recopies ce que
l'image montre, le solveur rend son verdict. Deux usages.

**Une main, en une ligne :**

    python analyser_main.py --hero "Ah Kd" --board "Qs 7d 2c" \
        --pot 100 --bet 75 --stack 300 --bb 10

**En rafale, sans réécrire les options** — lance sans argument, puis une
main par ligne au format court :

    Ah Kd | Qs 7d 2c | pot 100 | bet 75 | bb 10
    As Ac | | stack 10 | bb 1
    (ligne vide ou « q » pour quitter)

Les cartes s'écrivent comme sur la table : « Ah Kd », « A♠ K♦ », « 10h »,
majuscules ou non.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.analysis import Spot, advise  # noqa: E402

JOURNAL = Path(__file__).resolve().parent / "mains_analysees.jsonl"


def _log(spot: Spot, advice, note: str = "") -> None:
    """Ajoute la main au journal — les captures dépouillées s'accumulent.

    Une main analysée puis oubliée ne vaut rien : le journal permet de
    relire l'ensemble et d'y voir les récurrences (``--recap``).
    """
    entry = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hero": spot.hero, "board": spot.board, "pot": spot.pot,
        "bet": spot.bet, "stack": spot.stack, "big_blind": spot.big_blind,
        "position": spot.position, "villain": spot.villain,
        "hand": advice.hand, "action": advice.action,
        "regime": advice.regime, "equity": advice.equity,
        "required": advice.required, "ev_bb": advice.ev_bb,
        "note": note,
    }
    with JOURNAL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _recap() -> int:
    """Synthèse des mains déjà analysées : verdicts et récurrences."""
    if not JOURNAL.exists():
        print("Aucune main analysée pour l'instant "
              f"(journal attendu : {JOURNAL.name}).")
        return 0
    rows = [json.loads(line) for line in
            JOURNAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("Journal vide.")
        return 0

    verdicts = Counter(r["action"].split(" (")[0] for r in rows)
    hands = Counter(r["hand"] for r in rows)
    streets = Counter(
        {0: "préflop", 3: "flop", 4: "turn", 5: "river"}.get(
            len([c for c in (r["board"] or "") if c.isalnum()]) // 2, "?")
        for r in rows
    )
    print("──────────────────────────────────────────────────────")
    print(f"  {len(rows)} mains analysées  ·  journal {JOURNAL.name}")
    print("──────────────────────────────────────────────────────")
    print("  Verdicts :")
    for action, n in verdicts.most_common():
        print(f"    {action:<28} {n:3d}")
    print("  Par rue :")
    for street, n in streets.most_common():
        print(f"    {street:<28} {n:3d}")
    print("  Mains les plus revues :")
    for hand, n in hands.most_common(6):
        print(f"    {hand:<28} {n:3d}")
    evs = [r["ev_bb"] for r in rows if r.get("ev_bb") is not None]
    if evs:
        print(f"  EV moyenne des spots jugés : {sum(evs) / len(evs):+.2f} bb")
    print("──────────────────────────────────────────────────────")
    return 0


def _spot_from_line(line: str) -> Spot:
    """Parse la notation courte : « cartes | board | clé valeur | … ».

    Les deux premiers champs sont les cartes du héros et le board (le
    board peut être vide) ; les suivants sont des paires « clé valeur »
    parmi pot, bet, stack, bb, position, villain, players.
    """
    parts = [p.strip() for p in line.split("|")]
    spot = Spot(hero=parts[0], board=parts[1] if len(parts) > 1 else "")
    alias = {"bb": "big_blind", "pos": "position", "vs": "villain"}
    for field in parts[2:]:
        if not field:
            continue
        try:
            key, value = field.split(None, 1)
        except ValueError:
            raise ValueError(f"champ illisible : « {field} » "
                             "(attendu « clé valeur », ex. « pot 100 »)")
        key = alias.get(key.lower(), key.lower())
        if not hasattr(spot, key):
            raise ValueError(f"clé inconnue : « {key} ».")
        current = getattr(spot, key)
        setattr(spot, key, type(current)(value) if not isinstance(current, str)
                else value)
    return spot


def _run_batch() -> None:
    print(__doc__.split("**En rafale")[0].strip())
    print("\nUne main par ligne (ligne vide pour quitter) :\n")
    while True:
        try:
            line = input("main> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line or line.lower() in ("q", "quit", "exit"):
            return
        try:
            spot = _spot_from_line(line)
            advice = advise(spot)
            print(advice.explain())
            _log(spot, advice)
        except Exception as exc:
            print(f"  ✗ {type(exc).__name__}: {exc}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Que fallait-il faire dans cette main ?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--hero", help="tes deux cartes, ex. « Ah Kd »")
    ap.add_argument("--board", default="", help="cartes communes, ex. « Qs 7d 2c »")
    ap.add_argument("--pot", type=float, default=0.0,
                    help="pot AVANT la mise adverse")
    ap.add_argument("--bet", type=float, default=0.0, help="mise à payer")
    ap.add_argument("--stack", type=float, default=0.0, help="tapis effectif")
    ap.add_argument("--bb", type=float, default=1.0, dest="big_blind",
                    help="valeur de la big blind")
    ap.add_argument("--position", default="BTN", help="BTN, SB, BB, CO, MP, UTG")
    ap.add_argument("--villain", default="moyenne",
                    help="range adverse : large | moyenne | serree | « TT+, AQs+ »")
    ap.add_argument("--players", type=int, default=2, help="joueurs en jeu")
    ap.add_argument("--note", default="", help="note libre (contexte du screen)")
    ap.add_argument("--recap", action="store_true",
                    help="synthèse des mains déjà analysées, puis quitte")
    args = ap.parse_args()

    if args.recap:
        return _recap()
    if not args.hero:
        _run_batch()
        return 0

    spot = Spot(hero=args.hero, board=args.board, pot=args.pot, bet=args.bet,
                stack=args.stack, big_blind=args.big_blind,
                position=args.position, villain=args.villain,
                players=args.players)
    try:
        advice = advise(spot)
        print(advice.explain())
        _log(spot, advice, note=args.note)
    except Exception as exc:
        print(f"✗ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
