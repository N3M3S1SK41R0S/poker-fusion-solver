#!/usr/bin/env python
"""Récupère TOUTES les mains jouées, où qu'elles soient, et les analyse.

    python recuperer_mains.py                 tous les comptes connus
    python recuperer_mains.py --detail        + le bilan par tournoi
    python recuperer_mains.py --dossier D     un dossier précis

Pourquoi ce fichier existe
--------------------------
Retrouver ses propres mains s'est révélé étonnamment piégeux. Trois écueils,
tous rencontrés sur une session réelle, chacun perdant des mains en silence :

1. **Les fichiers ne sont pas là où on croit.** Le client range les
   historiques sous ``History/Data/Tournaments`` mais garde aussi des copies
   sous ``History/TempData/<horodatage>/...`` pour les tournois en cours. Ne
   lire que le premier dossier fait manquer les parties récentes.

2. **Les copies TempData sont PARTIELLES.** Le même identifiant de main
   existe en version tronquée (instantané pris pendant la partie) et en
   version complète (écrite à la fin). Dédoublonner naïvement, en gardant la
   première rencontrée, remplace la bonne par la mauvaise selon l'ordre du
   parcours : mesuré, un tournoi de 171 mains rendait des statistiques
   entièrement nulles. On garde donc la version la plus RICHE.

3. **Un montant en devise interrompait la lecture.** Sur les tournois à
   primes, le client écrit « 120€ » dans des champs numériques ; la lecture
   étant un générateur, l'exception emportait toutes les mains suivantes du
   fichier, sans message. Corrigé dans `pfs.data.hand_history` — mesuré :
   427 mains récupérées avant, 721 après.

Le fait que la première version de cette récupération ait été écrite à la
main, hors dépôt, est précisément ce qui a laissé passer l'écueil n° 2.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.analysis import review_hands  # noqa: E402
from pfs.data.hand_history import ParsedHand, iter_hands  # noqa: E402

#: Emplacements connus des historiques PMU, argent fictif puis argent réel.
DOSSIERS_CONNUS = (
    r"C:\Users\pierr\AppData\Local\PMU PLAY 100% Poker\data",
    r"C:\Users\pierr\AppData\Local\PMU Poker\data",
)

_TABLE = re.compile(r"<tablename>([^<]+)</tablename>")


def _richesse(m: ParsedHand) -> tuple[int, int, int]:
    """De quoi comparer deux versions d'une même main.

    Une copie prise en cours de partie s'arrête au milieu : elle a moins
    d'actions, parfois pas les cartes du héros. Trier là-dessus fait
    remonter la version complète, quel que soit l'ordre des fichiers.
    """
    return (len(m.actions), len(m.seats), len(m.hero_cards or ()))


def collecter(racines) -> tuple[dict[str, ParsedHand], dict[str, list[str]],
                                list[tuple[str, str]]]:
    """Toutes les mains uniques, leur tournoi, et les fichiers illisibles."""
    mains: dict[str, ParsedHand] = {}
    tournoi: dict[str, str] = {}
    echecs: list[tuple[str, str]] = []

    for racine in racines:
        base = Path(racine)
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.xml")):
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                echecs.append((p.name, f"lecture : {e}"))
                continue
            t = _TABLE.search(txt)
            nom = t.group(1).strip() if t else "tournoi inconnu"
            try:
                lues = list(iter_hands(txt))
            except Exception as e:                      # noqa: BLE001
                echecs.append((p.name, f"{type(e).__name__} : {e}"))
                continue
            for h in lues:
                ancienne = mains.get(h.hand_id)
                if ancienne is None or _richesse(h) > _richesse(ancienne):
                    mains[h.hand_id] = h
                    tournoi[h.hand_id] = nom

    par: dict[str, list[str]] = defaultdict(list)
    for hid, nom in tournoi.items():
        par[nom].append(hid)
    return mains, par, echecs


def bilan(titre: str, lot: list[ParsedHand]) -> None:
    """Profil, variance aux tapis, et ce qui reste imputable au jeu."""
    if not lot:
        return
    r = review_hands(lot)
    p = r.profile
    if not p.vpip_opp:
        print(f"{titre}\n   {p.n_hands} mains — aucune décision du héros "
              "(il n'est pas assis dans ces mains)\n")
        return

    reel = attendu = 0.0
    for s in r.allin_spots:
        bb = s.big_blind or 1
        reel += s.realized / bb
        attendu += (s.equity * s.pot) / bb
    variance = reel - attendu
    vpip = p.vpip_yes / p.vpip_opp
    pfr = p.pfr_yes / max(1, p.pfr_opp)

    print(f"{titre}")
    print(f"   {p.n_hands} mains   net {p.net_bb:+.1f} bb")
    print(f"   VPIP {vpip:.1%}   PFR {pfr:.1%}   écart {100 * (vpip - pfr):.1f} pts"
          f"   3-bet {p.three_bet_yes / max(1, p.three_bet_opp):.1%}"
          f"   WTSD {p.wtsd_yes / max(1, p.wtsd_opp):.1%}")
    print(f"   tapis : {len(r.allin_spots)}/{r.n_allin_total} mesurés — "
          f"encaissé {reel:+.1f}, espérance {attendu:+.1f}, "
          f"variance {variance:+.1f} bb")
    print(f"   → imputable au JEU : {p.net_bb - variance:+.1f} bb\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dossier", action="append", metavar="D",
                    help="dossier à fouiller (répétable) ; défaut : les "
                         "emplacements PMU connus")
    ap.add_argument("--detail", action="store_true",
                    help="bilan tournoi par tournoi")
    ap.add_argument("--min", type=int, default=15, metavar="N",
                    help="avec --detail, seuil de mains par tournoi (15)")
    a = ap.parse_args()

    racines = a.dossier or DOSSIERS_CONNUS
    mains, par, echecs = collecter(racines)
    if not mains:
        print("Aucune main trouvée. Vérifie les dossiers d'historique.")
        return 1

    print(f"{len(mains)} mains uniques dans {len(par)} tournoi(s)")
    if echecs:
        print(f"{len(echecs)} fichier(s) illisibles :")
        for nom, motif in echecs[:8]:
            print(f"   {nom} : {motif}")
    print()

    bilan(f"TOUT — {len(mains)} mains", list(mains.values()))

    if a.detail:
        gros = sorted(par.items(), key=lambda kv: -len(kv[1]))
        for nom, ids in gros:
            if len(ids) < a.min:
                continue
            bilan(f"{nom}  ({len(ids)} mains)", [mains[i] for i in ids])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
