#!/usr/bin/env python
"""Extrait les JEUX DE CARTES du client PMU PLAY vers des gabarits pHash.

    python scripts/extraire_jeux_pmu.py --lister
    python scripts/extraire_jeux_pmu.py --extraire 5 4c_2 --vers python/pfs/vision/templates

Pourquoi ce fichier existe
--------------------------
Les gabarits `templates/pmu_deck/` et `templates/pmu_solid/` avaient été
extraits lors d'une session précédente, sans laisser de trace de la méthode :
impossible d'en ajouter un, ni même de savoir d'où ils venaient. Ce script
rétablit le chemin complet, et il est vérifiable — `--lister` redonne la carte
des ressources sans rien écrire.

Ce qu'on trouve dans le client (mesuré le 10 août 2026, v26.1.1.23)
------------------------------------------------------------------
`data/PokerCommonWidgetsQRC.rcc` est une archive de ressources Qt (magie
« qres ») de 79 Mo. Elle contient 3 565 PNG, dont **12 jeux complets de 52
cartes en 65 × 88** — la taille exacte à laquelle la table les dessine
(`BaseCardDelegate.qml` : ``width: 65; height: 88``). Ils sont rangés à plat
sous ``/Images/<préfixe>_<enseigne><rang>`` :

    « 0 » … « 5 »          les six habillages du réglage « jeu de cartes »
    « 4c_0 » … « 4c_5 »    les mêmes en variante QUATRE COULEURS

Les décors « 5 » et « 4c_5 » sont le même fichier (l'habillage à fond plein
code déjà l'enseigne par la couleur du carton) — et c'est exactement lui que
`templates/pmu_solid/` contient, au bit près : la vérification est dans
`--verifier`.

Deux autres jeux de 52 cartes existent, à d'autres tailles et pour d'autres
écrans : ``/Images/HHCards/hh_*`` en 15 × 20 (historique des mains — c'est la
source de `templates/pmu_deck/`) et ``/Images/AllInOutsCards/*`` en 26 × 30
(pastilles d'outs). Ils ne servent pas à lire une table.

Format .rcc, juste ce qu'il faut
--------------------------------
En-tête ``qres`` + version + trois décalages (arbre, données, noms). L'arbre
est une suite de nœuds de taille fixe ; un nœud porte un décalage de nom, des
drapeaux (0x02 = dossier, 0x01 = zlib, 0x04 = zstd) et deux entiers. Les noms
sont en UTF-16BE précédés de leur longueur et d'un hachage. Les données sont
précédées de leur longueur. Aucun outil Qt n'est nécessaire.
"""

from __future__ import annotations

import argparse
import io
import re
import struct
import sys
import zlib
from collections import defaultdict
from pathlib import Path

FLAG_COMPRESSED = 0x01
FLAG_DIRECTORY = 0x02
FLAG_ZSTD = 0x04

RCC_PAR_DEFAUT = Path(r"C:\Users\pierr\AppData\Local\PMU PLAY 100% Poker\data"
                      r"\PokerCommonWidgetsQRC.rcc")

#: ``/Images/4c_2_h10`` → préfixe « 4c_2 », enseigne « h », rang « 10 ».
MOTIF_CARTE = re.compile(r"^/Images/(.+)_([cdhs])(10|[2-9]|[ajqk])$")
#: Le dépôt nomme les cartes « Ah », « Td », « 5c » ; le client « ha », « d10 ».
RANGS = {"10": "T", "a": "A", "j": "J", "q": "Q", "k": "K"}
TAILLE_TABLE = (65, 88)


class Rcc:
    """Archive de ressources Qt, lue en mémoire."""

    def __init__(self, chemin: Path) -> None:
        self.raw = chemin.read_bytes()
        if self.raw[:4] != b"qres":
            raise ValueError(f"{chemin} n'est pas une archive Qt (magie « qres »)")
        (self.version, self.tree_off, self.data_off,
         self.name_off) = struct.unpack_from(">IIII", self.raw, 4)
        # v1 : nom(4) + drapeaux(2) + deux entiers(8). v2+ : + date sur 8 octets.
        self.node_size = 14 + (8 if self.version >= 2 else 0)

    def _noeud(self, i: int) -> tuple[int, int, int, int]:
        o = self.tree_off + i * self.node_size
        name_off, flags = struct.unpack_from(">IH", self.raw, o)
        a, b = struct.unpack_from(">II", self.raw, o + 6)
        return name_off, flags, a, b

    def _nom(self, off: int) -> str:
        n, = struct.unpack_from(">H", self.raw, self.name_off + off)
        debut = self.name_off + off + 6          # +2 longueur, +4 hachage
        return self.raw[debut:debut + n * 2].decode("utf-16-be")

    def parcours(self, idx: int = 0, prefixe: str = ""):
        """Itère ``(chemin, drapeaux, décalage_données)`` sur tous les fichiers."""
        name_off, flags, a, b = self._noeud(idx)
        nom = "" if idx == 0 else self._nom(name_off)
        chemin = f"{prefixe}/{nom}" if nom else prefixe
        if flags & FLAG_DIRECTORY:
            for k in range(a):
                yield from self.parcours(b + k, chemin)
        else:
            yield chemin, flags, b

    def lire(self, flags: int, off: int) -> bytes:
        n, = struct.unpack_from(">I", self.raw, self.data_off + off)
        blob = self.raw[self.data_off + off + 4:self.data_off + off + 4 + n]
        if flags & FLAG_COMPRESSED:
            return zlib.decompress(blob[4:])     # 4 octets de taille en clair
        if flags & FLAG_ZSTD:                    # pragma: no cover - non rencontré
            import zstandard
            return zstandard.ZstdDecompressor().decompress(blob)
        return blob


def jeux_de_table(rcc: Rcc) -> dict[str, dict[str, bytes]]:
    """Les jeux de 52 cartes à la taille de la table, par préfixe du client."""
    from PIL import Image

    trouves: dict[str, dict[str, bytes]] = defaultdict(dict)
    for chemin, flags, off in rcc.parcours():
        m = MOTIF_CARTE.match(chemin)
        if not m:
            continue
        prefixe, enseigne, rang = m.groups()
        blob = rcc.lire(flags, off)
        if not blob.startswith(b"\x89PNG"):
            continue
        if Image.open(io.BytesIO(blob)).size != TAILLE_TABLE:
            continue
        trouves[prefixe][f"{RANGS.get(rang, rang.upper())}{enseigne}"] = blob
    # un jeu incomplet ne sert à rien : on ne garde que les 52 cartes pleines
    return {p: c for p, c in trouves.items() if len(c) == 52}


def _nom_de_theme(prefixe: str) -> str:
    """« 4c_2 » → « pmu_4c_2 », « 3 » → « pmu_t3 » (un dossier par habillage)."""
    return f"pmu_4c_{prefixe[3:]}" if prefixe.startswith("4c_") else f"pmu_t{prefixe}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rcc", type=Path, default=RCC_PAR_DEFAUT,
                   help="archive PokerCommonWidgetsQRC.rcc du client")
    p.add_argument("--lister", action="store_true",
                   help="montre les jeux trouvés, n'écrit rien")
    p.add_argument("--extraire", nargs="*", metavar="PREFIXE",
                   help="préfixes à extraire (vide = tous)")
    p.add_argument("--vers", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "python" / "pfs" / "vision" / "templates",
                   help="racine des gabarits")
    p.add_argument("--verifier", metavar="THEME",
                   help="compare un thème déjà livré aux jeux du client")
    args = p.parse_args()

    if not args.rcc.exists():
        print(f"archive introuvable : {args.rcc}", file=sys.stderr)
        return 2
    jeux = jeux_de_table(Rcc(args.rcc))
    print(f"{len(jeux)} jeux complets de 52 cartes en {TAILLE_TABLE[0]}×"
          f"{TAILLE_TABLE[1]} dans {args.rcc.name}")
    for prefixe in sorted(jeux):
        print(f"   « {prefixe} »  → {_nom_de_theme(prefixe)}")

    if args.verifier:
        import numpy as np
        from PIL import Image

        ref = args.vers / args.verifier
        if not ref.is_dir():
            print(f"thème absent : {ref}", file=sys.stderr)
            return 2
        for prefixe, cartes in sorted(jeux.items()):
            ecarts = []
            for f in sorted(ref.glob("*.png")):
                b = cartes.get(f.stem)
                if b is None:
                    continue
                a1 = np.asarray(Image.open(f).convert("RGBA")).astype(int)
                a2 = np.asarray(Image.open(io.BytesIO(b)).convert("RGBA")).astype(int)
                if a1.shape == a2.shape:
                    ecarts.append(float(np.abs(a1 - a2).mean()))
            if ecarts:
                print(f"   {args.verifier} vs « {prefixe} » : écart moyen "
                      f"{sum(ecarts) / len(ecarts):.2f}")

    if args.extraire is None:
        return 0
    voulus = set(args.extraire) or set(jeux)
    for prefixe in sorted(voulus):
        if prefixe not in jeux:
            print(f"   ignoré (inconnu) : {prefixe}", file=sys.stderr)
            continue
        d = args.vers / _nom_de_theme(prefixe)
        d.mkdir(parents=True, exist_ok=True)
        for carte, blob in sorted(jeux[prefixe].items()):
            (d / f"{carte}.png").write_bytes(blob)
        print(f"   écrit {len(jeux[prefixe])} cartes dans {d}")
    print("\nPensez à ajouter les thèmes à `_THEMES` (pfs/vision/card_recognizer.py)\n"
          "puis à régénérer les signatures : "
          "python -c \"from pfs.vision.card_recognizer import build_all_themes;"
          " build_all_themes()\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
