#!/usr/bin/env python
"""Banc de VÉRITÉ-TERRAIN : rappel, précision, lectures fausses affirmées.

    python banc_verite_captures.py
    python banc_verite_captures.py --captures <dossier>
    python banc_verite_captures.py --detail          une ligne par frame
    python banc_verite_captures.py --json out.json
    python banc_verite_captures.py --quiet-sides 2   ablation du détecteur

Résultat au 11 août 2026 (57 captures, deux tables PMU) ::

    cartes réellement présentes (pleinement visibles) : 258
    RAPPEL DE LOCALISATION :  76,7 %  (198/258)
    RAPPEL DE LECTURE      :  76,7 %  (198/258)
      dont bon rôle        :  65,1 %  (168/258)
    PRÉCISION              : 100,0 %  (199/199 lectures affirmées)
    ABSTENTION             :  38,8 %  (126/325 boîtes)
    cartes JAMAIS localisées : 60      lectures fausses affirmées : 0
    cartes inventées         :  0      rôles faux affirmés        : 30

Le banc reste ROUGE sur les 30 rôles faux : ce défaut est réel, diagnostiqué
(voir plus bas) et non corrigé. Un banc qui passerait au vert en fermant les
yeux dessus ne vaudrait rien.

Pourquoi ce banc remplace le « 95 % »
-------------------------------------
Le dépôt a annoncé, sur ces mêmes 57 captures :

    cartes qui comptent (héros + board) : 199 lues avec certitude sur 209 → 95 %

Ce chiffre est faux **deux fois**, et les deux erreurs vont dans le même sens
— celui qui flatte :

1. **Le dénominateur était conditionnel à la détection.** 209 = 108 boîtes
   étiquetées « hero » + 101 boîtes étiquetées « board ». Une carte que le
   détecteur n'a jamais vue ne figurait NULLE PART, ni au numérateur ni au
   dénominateur. Le vrai dénominateur est le nombre de cartes RÉELLEMENT
   PRÉSENTES : 258 pleinement visibles (112 héros + 146 board), plus 5 en
   transition. Le rappel, lui, n'avait jamais été mesuré.
2. **Il n'y avait aucune vérité-terrain.** « 199 lues avec certitude » voulait
   dire « 199 non refusées ». Une lecture fausse et AFFIRMÉE comptait comme un
   succès. C'est précisément le mode d'échec le plus grave de cet outil, et la
   métrique était construite pour ne pas le voir.

Ce banc corrige les deux. Il part du relevé fait à l'œil sur les 57 captures
(`tests/donnees/verite_captures.json`), rejoue la chaîne de production —
`pfs.vision.live.lire_image`, c'est-à-dire exactement ce que fait le mode
calibration une fois la capture obtenue — et apparie chaque boîte détectée à
l'emplacement de carte qu'elle recouvre.

Ce qu'il rend
-------------
* **RAPPEL DE LOCALISATION** — part des cartes réellement présentes qu'une
  boîte recouvre. C'est le chiffre qui manquait.
* **RAPPEL DE LECTURE** — part des cartes réellement présentes qui sont lues,
  JUSTES et affirmées. C'est le seul taux qui mérite d'être annoncé.
* **PRÉCISION** — part des lectures affirmées qui sont justes.
* **ABSTENTION** — part des boîtes détectées que la chaîne refuse de lire.
* **LECTURES FAUSSES AFFIRMÉES** — le chiffre qui doit valoir 0 : une carte
  nommée, sans réserve, alors qu'elle est autre chose. Le banc échoue si ce
  nombre n'est pas nul.
* **RÔLES FAUX AFFIRMÉS** — une carte juste, affirmée, mais rangée dans le
  mauvais rôle (une carte du board présentée comme une carte du héros). Pour
  un solveur, c'est aussi grave qu'une carte fausse.

Ce que le banc a trouvé, et qui n'était pas cherché
---------------------------------------------------
* **60 cartes ne sont jamais vues.** 45 sont des cartes du HÉROS, dont les 42
  de la table 7-max — jamais une seule sur 24 des 25 captures. Ce n'est ni
  une occultation ni un siège vide : la carte est parfaitement visible, ses
  deux arêtes verticales sont trouvées, le recalage donne le bon rapport
  (1,017, dans la plage « carte coupée par le bas »). C'est la règle
  « 3 abords calmes sur 4 » (`table_detector.QUIET_SIDES`) qui les rejette —
  l'habillage « KO » de cette table borde le siège d'un rail lumineux et pose
  une pastille de prime « 2,25 € » à cheval sur la carte de gauche. Les 15
  autres sont des cartes du board sous la pile de jetons du pot.
  Ablation : `--quiet-sides 2` porte le rappel à 96,9 % sans une seule carte
  inventée ; sur les 144 tables synthétiques, la même ablation coûte 1,9 % de
  fantômes. Les deux mesures s'opposent, et seule la seconde existait quand
  la constante a été choisie.
* **30 cartes du board sont présentées comme cartes du héros**, conséquence
  mécanique : `read_table` prend la rangée la plus basse pour la main du
  héros ; quand le siège n'est pas détecté, le board hérite du rôle.
* **Une lecture fausse et affirmée existait bien** : le 6♣ du flop de
  `300_7-max_KO/0003`, saisi en pleine animation de retournement, sortait
  « Kc » avec le statut « sure ». Le contrôle de dispersion du lecteur à fond
  plein refusait la découpe, mais son refus était muet et la chaîne passait
  la main au hachage. Corrigé (refus franc), compteur à 0.

Sur les captures et leur absence
--------------------------------
Les 57 PNG pèsent ~108 Mo et ne sont PAS dans le dépôt. Seule la
vérité-terrain l'est. Sans les images, le banc dit où les chercher et
s'arrête : il ne rend jamais de chiffre approximatif.

La preuve que les tests associés peuvent ÉCHOUER est elle aussi un banc :
`banc_mutations_verite.py` casse onze fois ce qu'ils protègent — à commencer
par le dénominateur conditionnel, c'est-à-dire le « 95 % » refabriqué — et
exige qu'ils tombent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.vision.live import lire_image  # noqa: E402

VERITE = Path(__file__).resolve().parent / "tests" / "donnees" / "verite_captures.json"

def captures_par_defaut() -> Path:
    """Emplacement des sessions écrites par `capturer_session.py`.

    Hors du dépôt, et volontairement : 57 PNG pèsent ~108 Mo.
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "PokerFusionSolver" / "captures" / "sessions"


#: Recouvrement minimal (rapporté à la plus petite des deux boîtes) au-delà
#: duquel une boîte détectée est considérée POSÉE SUR un emplacement de carte.
#:
#: 0,50 et non l'IoU : une boîte de carte du héros déborde volontiers sur sa
#: voisine (le détecteur garde la plus grande des boîtes concurrentes, cf.
#: `table_detector`), et l'IoU d'une boîte 126×115 sur un emplacement 117×106
#: tombe sous 0,5 sans que la boîte soit « ailleurs ». Ce qui compte ici est :
#: cette boîte désigne-t-elle CETTE carte ? Le recouvrement relatif le dit.
RECOUVREMENT_MIN = 0.50


@dataclass
class Compte:
    """Un décompte, et les cas qui le composent (pour pouvoir les relire)."""

    n: int = 0
    cas: list[str] = field(default_factory=list)

    def ajoute(self, quoi: str) -> None:
        self.n += 1
        self.cas.append(quoi)


@dataclass
class Bilan:
    """Tout ce que le banc mesure, sur l'ensemble des frames."""

    cartes_reelles: int = 0
    localisees: Compte = field(default_factory=Compte)
    lues_justes: Compte = field(default_factory=Compte)
    lues_justes_bon_role: Compte = field(default_factory=Compte)
    non_localisees: Compte = field(default_factory=Compte)
    localisees_non_lues: Compte = field(default_factory=Compte)

    boites: int = 0
    affirmations: int = 0
    #: Affirmations dont la carte est celle qui se trouve à cet endroit,
    #: cartes en transition COMPRISES. C'est le numérateur de la précision :
    #: nommer juste une carte en cours de retournement reste une lecture
    #: juste, même si le rappel ne la compte pas (elle n'est pas pleinement
    #: visible, et la refuser est un comportement légitime).
    affirmations_justes: int = 0
    fausses_affirmees: Compte = field(default_factory=Compte)
    inventees: Compte = field(default_factory=Compte)
    roles_faux: Compte = field(default_factory=Compte)

    transitoires: int = 0
    transitoires_affirmees: Compte = field(default_factory=Compte)


def _recouvrement(a: tuple[int, int, int, int],
                  b: tuple[int, int, int, int]) -> float:
    """Intersection rapportée à la plus PETITE des deux boîtes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / min(aw * ah, bw * bh) if ix and iy else 0.0


def cartes_attendues(frame: dict, emplacements: dict
                     ) -> list[tuple[str, tuple[int, int, int, int], str, bool]]:
    """(rôle, boîte, carte, transitoire) pour toutes les cartes d'une frame.

    Les cartes du board occupent les emplacements de gauche à droite, celles
    du héros les deux emplacements du siège. Les cartes en TRANSITION
    (retournement en cours, fondu de sortie) occupent les mêmes emplacements ;
    elles sont marquées, car un lecteur honnête a le droit de les refuser.
    """
    out = []
    for role, cle in (("board", "board"), ("hero", "hero")):
        stables = frame[cle]
        transito = frame[f"{cle}_transition"]
        for i, carte in enumerate(list(stables) + list(transito)):
            out.append((role, tuple(emplacements[cle][i]), carte,
                        i >= len(stables)))
    return out


def mesure_frame(chemin: Path, frame: dict, emplacements: dict,
                 bilan: Bilan) -> dict:
    """Rejoue la chaîne sur une capture et impute chaque lecture."""
    from PIL import Image

    image = Image.open(chemin).convert("RGB")
    # `archiver=False` : ces découpes sont déjà sur disque, les rejouer dans
    # l'archive de l'utilisateur n'apprendrait rien.
    lecture = lire_image(image, fenetre=chemin.parent.name, archiver=False)

    attendues = cartes_attendues(frame, emplacements)
    etiquette = f"{chemin.parent.name[:14]}/{chemin.stem}"

    # Chaque boîte détectée est rattachée à l'emplacement qu'elle recouvre le
    # plus, ou à rien du tout.
    rattachee: list[tuple[object, int | None]] = []
    for c in lecture.cartes:
        meilleur, score = None, RECOUVREMENT_MIN
        for i, (_, boite, _, _) in enumerate(attendues):
            s = _recouvrement(boite, tuple(c.boite))
            if s > score:
                meilleur, score = i, s
        rattachee.append((c, meilleur))

    # --- ce que chaque carte réelle est devenue ---------------------------
    for i, (role_vrai, _, carte_vraie, transitoire) in enumerate(attendues):
        dessus = [c for c, j in rattachee if j == i]
        quoi = f"{etiquette} {role_vrai} {carte_vraie}"
        if transitoire:
            bilan.transitoires += 1
            for c in dessus:
                if c.statut == "sure":
                    verdict = "JUSTE" if c.carte == carte_vraie else "FAUSSE"
                    bilan.transitoires_affirmees.ajoute(
                        f"{quoi} (en transition) affirmée {c.carte} [{verdict}]")
            continue

        bilan.cartes_reelles += 1
        if not dessus:
            bilan.non_localisees.ajoute(quoi)
            continue
        bilan.localisees.ajoute(quoi)
        justes = [c for c in dessus
                  if c.statut == "sure" and c.carte == carte_vraie]
        if justes:
            bilan.lues_justes.ajoute(quoi)
            if any(c.role == role_vrai for c in justes):
                bilan.lues_justes_bon_role.ajoute(quoi)
            else:
                bilan.roles_faux.ajoute(
                    f"{quoi} lue {justes[0].carte} mais rangée dans "
                    f"« {justes[0].role} »")
        else:
            meilleure = dessus[0]
            bilan.localisees_non_lues.ajoute(
                f"{quoi} → statut {meilleure.statut}, candidat "
                f"{meilleure.candidat} (écart {meilleure.ecart}, "
                f"marge {meilleure.marge})")

    # --- ce que chaque lecture affirmée vaut ------------------------------
    for c, i in rattachee:
        bilan.boites += 1
        if c.statut != "sure":
            continue
        bilan.affirmations += 1
        if i is None:
            bilan.inventees.ajoute(
                f"{etiquette} boîte {tuple(c.boite)} rôle « {c.role} » "
                f"affirme {c.carte} — aucune carte à cet endroit")
        elif c.carte == attendues[i][2]:
            bilan.affirmations_justes += 1
        else:
            role_vrai, _, carte_vraie, transitoire = attendues[i]
            etat = " (carte en transition)" if transitoire else ""
            bilan.fausses_affirmees.ajoute(
                f"{etiquette} {role_vrai} {carte_vraie}{etat} affirmée "
                f"« {c.carte} » (écart {c.ecart}, marge {c.marge})")

    return {
        "session": frame["session"], "frame": frame["frame"],
        "attendues": len(attendues),
        "boites": len(lecture.cartes),
        "sures": lecture.sures,
        "lues": [f"{c.role}:{c.carte or '-'}" for c in lecture.cartes],
    }


def _pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f} %" if d else "   n/a"


def rapport(bilan: Bilan, verbeux: bool) -> int:
    """Imprime les vrais chiffres. Renvoie le nombre de fautes graves."""
    print("\n" + "=" * 72)
    print("RÉSULTATS SUR VÉRITÉ-TERRAIN")
    print("=" * 72)
    n = bilan.cartes_reelles
    print(f"  cartes réellement présentes (pleinement visibles) : {n}")
    print(f"  cartes en transition (retournement, fondu)        : "
          f"{bilan.transitoires}")
    print(f"  boîtes rendues par le détecteur                   : {bilan.boites}")
    print()
    print(f"  RAPPEL DE LOCALISATION : {_pct(bilan.localisees.n, n)}  "
          f"({bilan.localisees.n}/{n})")
    print(f"  RAPPEL DE LECTURE      : {_pct(bilan.lues_justes.n, n)}  "
          f"({bilan.lues_justes.n}/{n})  ← le taux à annoncer")
    print(f"    dont bon rôle        : {_pct(bilan.lues_justes_bon_role.n, n)}  "
          f"({bilan.lues_justes_bon_role.n}/{n})")
    print(f"  PRÉCISION              : "
          f"{_pct(bilan.affirmations_justes, bilan.affirmations)}  "
          f"({bilan.affirmations_justes}/{bilan.affirmations} "
          "lectures affirmées)")
    print(f"  ABSTENTION             : "
          f"{_pct(bilan.boites - bilan.affirmations, bilan.boites)}  "
          f"({bilan.boites - bilan.affirmations}/{bilan.boites} boîtes)")
    print()
    print(f"  cartes JAMAIS localisées      : {bilan.non_localisees.n}")
    print(f"  localisées mais non lues      : {bilan.localisees_non_lues.n}")
    print(f"  LECTURES FAUSSES AFFIRMÉES    : {bilan.fausses_affirmees.n}"
          "   ← doit valoir 0")
    print(f"  cartes INVENTÉES (sur du vide): {bilan.inventees.n}"
          "   ← doit valoir 0")
    print(f"  RÔLES FAUX AFFIRMÉS           : {bilan.roles_faux.n}"
          "   ← doit valoir 0")
    print(f"  affirmations sur carte en transition : "
          f"{bilan.transitoires_affirmees.n}")

    for titre, compte in (("LECTURES FAUSSES AFFIRMÉES", bilan.fausses_affirmees),
                          ("CARTES INVENTÉES", bilan.inventees),
                          ("RÔLES FAUX AFFIRMÉS", bilan.roles_faux),
                          ("AFFIRMATIONS SUR CARTE EN TRANSITION",
                           bilan.transitoires_affirmees)):
        if compte.n:
            print(f"\n  --- {titre} ({compte.n}) ---")
            for c in compte.cas:
                print(f"    {c}")

    if bilan.non_localisees.n:
        print(f"\n  --- CARTES JAMAIS LOCALISÉES ({bilan.non_localisees.n}) ---")
        resume = Counter(c.split(" ", 1)[1] for c in bilan.non_localisees.cas)
        for quoi, k in resume.most_common():
            print(f"    ×{k:3d}  {quoi}")
        if verbeux:
            for c in bilan.non_localisees.cas:
                print(f"      {c}")

    if bilan.localisees_non_lues.n and verbeux:
        print(f"\n  --- LOCALISÉES MAIS NON LUES ({bilan.localisees_non_lues.n}) ---")
        for c in bilan.localisees_non_lues.cas:
            print(f"    {c}")

    # Le verdict nomme les trois fautes séparément : elles n'ont ni la même
    # cause ni le même chantier, et un « BANC ROUGE » global cacherait
    # laquelle des trois est à zéro.
    fautes = (bilan.fausses_affirmees.n + bilan.inventees.n
              + bilan.roles_faux.n)
    print()
    for nom, k in (("lecture(s) fausse(s) affirmée(s)", bilan.fausses_affirmees.n),
                   ("carte(s) inventée(s)", bilan.inventees.n),
                   ("rôle(s) faux affirmé(s)", bilan.roles_faux.n)):
        print(f"  {'OK  ' if k == 0 else 'FAUTE'}  {k} {nom}")
    print("BANC VERT — aucune affirmation fausse" if fautes == 0 else
          f"BANC ROUGE — {fautes} affirmation(s) fausse(s) au total")
    return fautes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", type=Path, default=None,
                   help="dossier des sessions de captures")
    p.add_argument("--detail", action="store_true",
                   help="une ligne par frame, et le détail des échecs")
    p.add_argument("--json", type=Path, default=None,
                   help="écrit le détail par frame dans ce fichier")
    p.add_argument("--quiet-sides", type=int, default=None, metavar="N",
                   help="ABLATION : nombre d'abords calmes exigés par le "
                        "détecteur (défaut 3). C'est le test qui rejette les "
                        "cartes du héros de la table 7-max ; le desserrer se "
                        "paie en cartes inventées, que ce banc compte.")
    args = p.parse_args()

    if args.quiet_sides is not None:
        from pfs.vision import table_detector
        table_detector.QUIET_SIDES = args.quiet_sides
        print(f"ABLATION : QUIET_SIDES = {args.quiet_sides} "
              f"(valeur du dépôt : 3)")

    doc = json.loads(VERITE.read_text(encoding="utf-8"))
    racine = args.captures or captures_par_defaut()
    if not racine.is_dir():
        print(f"captures introuvables : {racine}\n"
              "Les 57 PNG (~108 Mo) ne sont pas versionnés. Relance une "
              "session avec `python capturer_session.py`, ou pointe le "
              "dossier avec --captures.", file=sys.stderr)
        return 2

    bilan = Bilan()
    lignes = []
    manquantes = []
    for frame in doc["frames"]:
        chemin = racine / frame["session"] / f"{frame['frame']}.png"
        if not chemin.exists():
            manquantes.append(str(chemin))
            continue
        ligne = mesure_frame(chemin, frame, doc["emplacements"], bilan)
        lignes.append(ligne)
        if args.detail:
            print(f"{ligne['session'][:14]}/{ligne['frame']} : "
                  f"{ligne['attendues']} attendues, {ligne['boites']} boîtes, "
                  f"{ligne['sures']} sûres — {' '.join(ligne['lues'])}")

    if manquantes:
        print(f"{len(manquantes)} capture(s) absente(s), à commencer par "
              f"{manquantes[0]}", file=sys.stderr)
        return 2

    fautes = rapport(bilan, args.detail)
    if args.json:
        args.json.write_text(json.dumps(lignes, indent=1, ensure_ascii=False),
                             encoding="utf-8")
        print(f"détail par frame écrit dans {args.json}")
    return fautes


if __name__ == "__main__":
    raise SystemExit(main())
