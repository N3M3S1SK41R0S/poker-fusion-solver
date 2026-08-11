#!/usr/bin/env python
"""Banc de MUTATION : les tests du chantier « vérité-terrain » tombent-ils ?

    python banc_mutations_verite.py

Pourquoi ce fichier existe
--------------------------
Ce dépôt a déjà payé le prix d'un test qui ne pouvait pas échouer : deux
gabarits de cartes intervertis ont survécu six semaines sous 1 057 tests
verts, parce que chaque gabarit n'était comparé qu'à lui-même. « Vert » ne
veut rien dire tant qu'on n'a pas montré que rouge est atteignable.

Ce banc casse, une par une, les choses que `tests/test_verite_captures.py`
prétend protéger — dans une COPIE du dépôt, jamais dans le dépôt lui-même —
et exige que les tests visés échouent. Onze mutations : sept sur le code
(l'arithmétique du banc et le garde-fou de la carte retournée), quatre sur le
fichier de vérité-terrain (les fautes de saisie plausibles quand on relève
263 cartes à la main).

La plus importante est la première : **remettre le dénominateur conditionnel
à la détection**, c'est-à-dire refabriquer exactement le « 95 % ». Si ce test
cessait de tomber, la métrique pourrait redevenir mensongère sans que rien ne
passe au rouge.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent
FICHIER_TESTS = "tests/test_verite_captures.py"

#: (nom, fichier, motif exact, remplacement, tests qui doivent tomber)
MUTATIONS: tuple[tuple[str, str, str, str, list[str]], ...] = (
    ("le rappel ne compte plus que les cartes localisées "
     "(c'est le « 95 % » refabriqué)",
     "banc_verite_captures.py",
     "        bilan.cartes_reelles += 1\n        if not dessus:\n"
     "            bilan.non_localisees.ajoute(quoi)\n            continue",
     "        if not dessus:\n            bilan.non_localisees.ajoute(quoi)\n"
     "            continue\n        bilan.cartes_reelles += 1",
     ["test_une_carte_jamais_detectee_reste_au_denominateur"]),

    ("« affirmée » redevient synonyme de « juste »",
     "banc_verite_captures.py",
     "        justes = [c for c in dessus\n"
     "                  if c.statut == \"sure\" and c.carte == carte_vraie]",
     "        justes = [c for c in dessus if c.statut == \"sure\"]",
     ["test_une_lecture_affirmee_fausse_est_comptee_fausse"]),

    ("le rôle n'est plus vérifié",
     "banc_verite_captures.py",
     "            if any(c.role == role_vrai for c in justes):",
     "            if True:",
     ["test_un_role_faux_est_signale_meme_quand_la_carte_est_juste"]),

    ("une affirmation posée sur du vide n'est plus signalée",
     "banc_verite_captures.py",
     "        if i is None:\n            bilan.inventees.ajoute(",
     "        if i is None:\n            _ = (\n",
     ["test_une_affirmation_sur_du_vide_est_une_carte_inventee",
      "test_une_boite_qui_rate_la_carte_ne_la_valide_pas"]),

    ("les cartes en transition ne sont plus jugées du tout",
     "banc_verite_captures.py",
     "        elif c.carte == attendues[i][2]:",
     "        elif c.carte == attendues[i][2] or attendues[i][3]:",
     ["test_une_carte_en_transition_ne_gonfle_pas_le_rappel"]),

    ("le refus franc du lecteur à fond plein est retiré",
     "pfs/vision/card_recognizer.py",
     "    if lu.carte is None and lu.du_jeu and \"uniforme\" in lu.motif:",
     "    if False:",
     ["test_une_carte_en_retournement_n_est_jamais_affirmee_par_la_chaine"]),

    ("le refus franc s'étend à tout, deck classique compris",
     "pfs/vision/card_recognizer.py",
     "    if lu.carte is None and lu.du_jeu and \"uniforme\" in lu.motif:",
     "    if lu.carte is None:",
     ["test_le_refus_franc_ne_s_applique_qu_au_jeu_a_fond_plein"]),
)


def _change(doc: dict, frame: str, cle: str, i: int, valeur: str) -> None:
    for f in doc["frames"]:
        if f["frame"] == frame and f["session"].startswith("100"):
            f[cle][i] = valeur
            return
    raise SystemExit(f"frame {frame} introuvable dans la vérité-terrain")


#: Fautes de saisie plausibles dans un relevé fait à la main.
MUTATIONS_DONNEES = (
    ("une carte du board recopiée de travers au turn",
     lambda d: _change(d, "0020", "board", 2, "3s"),
     ["test_le_board_ne_recule_jamais_dans_une_main"]),
    ("la même carte relevée deux fois sur une frame",
     lambda d: _change(d, "0030", "board", 4, "2c"),
     ["test_aucune_carte_en_double_dans_une_frame"]),
    ("une carte oubliée dans le relevé",
     lambda d: d["frames"][10]["board"].pop(),
     ["test_les_totaux_sont_ceux_annonces"]),
    ("une notation invalide (« 10c » au lieu de « Tc »)",
     lambda d: _change(d, "0000", "hero", 0, "10c"),
     ["test_chaque_carte_est_bien_formee"]),
)


def _essaie(nom: str, applique, tests: list[str]) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "python"
        shutil.copytree(RACINE, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache"))
        if not applique(dst):
            print(f"  MOTIF ABSENT  {nom}")
            print("      Le code a changé : remets ce banc à jour AVANT de "
                  "conclure quoi que ce soit sur les tests.")
            return False
        r = subprocess.run(
            [sys.executable, "-m", "pytest", FICHIER_TESTS, "-q",
             "-p", "no:cacheprovider", "-k", " or ".join(tests)],
            cwd=dst, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(dst)})
        tombe = r.returncode != 0
        print(f"  {'OK  ' if tombe else 'RATÉ'}  {nom}")
        if not tombe:
            print(re.sub(r"^", "      ", r.stdout[-1200:], flags=re.M))
        return tombe


def main() -> int:
    fautes = 0
    print("=== mutations du CODE ===")
    for nom, fichier, motif, remplacement, tests in MUTATIONS:
        def applique(dst, f=fichier, m=motif, rp=remplacement):
            p = dst / f
            t = p.read_text(encoding="utf-8")
            if m not in t:
                return False
            p.write_text(t.replace(m, rp, 1), encoding="utf-8")
            return True
        fautes += not _essaie(nom, applique, tests)

    print("=== mutations de la VÉRITÉ-TERRAIN ===")
    for nom, muter, tests in MUTATIONS_DONNEES:
        def applique(dst, mu=muter):
            p = dst / "tests" / "donnees" / "verite_captures.json"
            d = json.loads(p.read_text(encoding="utf-8"))
            mu(d)
            p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            return True
        fautes += not _essaie(nom, applique, tests)

    total = len(MUTATIONS) + len(MUTATIONS_DONNEES)
    print("\n" + (f"BANC VERT — {total}/{total} mutations attrapées"
                  if fautes == 0 else
                  f"BANC ROUGE — {fautes}/{total} mutation(s) inaperçue(s)"))
    return fautes


if __name__ == "__main__":
    raise SystemExit(main())
