#!/usr/bin/env python
"""Banc de MUTATION : les tests du banc corpus tombent-ils quand on le casse ?

    python banc_mutations_corpus.py

Pourquoi ce fichier existe
--------------------------
``banc_corpus_pluribus.py`` produit les chiffres d'un rapport : taux d'accord
par régime, familles de spots où le désaccord est systématique, axes de
caractérisation. Une erreur silencieuse y donnerait un rapport **faux mais
crédible** — le pire des résultats, parce que personne n'a de raison de le
relire.

Le fichier ``tests/test_banc_corpus_pluribus.py`` prétend protéger le
classement des spots et l'arithmétique du banc. Ce banc-ci le vérifie : il
casse, une par une, les décisions dont dépendent les chiffres — dans une
COPIE du dépôt, jamais dans le dépôt lui-même — et exige que le test visé
échoue. Un test qui reste vert sous sa mutation ne protège rien.

Les dix-sept mutations couvrent les cinq endroits où une faute déplacerait un
chiffre sans qu'aucun total ne bouge :

1. **le régime** — seuil du tapis court, séparation des rues postflop ;
2. **la profondeur** — tapis effectif contre tapis brut, big blind nulle ;
3. **la caractérisation** — texture du board (as bas, rues vues, seuil de
   couleur) et type de main (meilleure combinaison de cinq cartes) ;
4. **la statistique** — critère de disjonction des intervalles, comptage et
   fusion des axes ;
5. **la couverture annoncée** — le classement des moteurs qui fait dire à la
   section 9 quelle partie du conseiller a été confrontée.

Deux mutations comptent plus que les autres. Remplacer la disjonction des
intervalles de Wilson par une simple comparaison de taux fait la différence
entre « ce défaut est mesuré » et « ce motif est apparu sur 10 000 mains
parce qu'on a regardé assez d'axes ». Inverser la priorité des motifs de
classement fait dire au rapport que l'ICM a été éprouvé — sur un corpus de
cash game.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent
FICHIER_TESTS = "tests/test_banc_corpus_pluribus.py"
FICHIER_MUTE = "banc_corpus_pluribus.py"

#: (nom, motif exact, remplacement, test qui doit tomber)
MUTATIONS: tuple[tuple[str, str, str, str], ...] = (
    # ── 1. le régime ────────────────────────────────────────────────────────
    ("seuil du tapis court : « < » devient « <= »",
     "return REGIME_COURT if prof_bb < SEUIL_TAPIS_COURT_BB "
     "else REGIME_PROFOND",
     "return REGIME_COURT if prof_bb <= SEUIL_TAPIS_COURT_BB "
     "else REGIME_PROFOND",
     "TestRegimeDuSpot::test_la_frontiere_du_tapis_court_est_a_quinze_bb"),

    ("les trois rues postflop fondues en un seul régime",
     'return f"postflop · {d.street.value}"',
     'return "postflop · flop"',
     "TestRegimeDuSpot::test_le_postflop_est_separe_rue_par_rue"),

    # ── 2. la profondeur ────────────────────────────────────────────────────
    ("profondeur lue sur le tapis du joueur, pas sur le tapis effectif",
     "return d.stack_effectif / bb if bb > 0 else math.inf",
     "return d.stack / bb if bb > 0 else math.inf",
     "TestProfondeurBb::test_c_est_le_tapis_effectif_qui_compte"),

    ("big blind nulle : 0 bb au lieu d'infini (fabrique un tapis court)",
     "return d.stack_effectif / bb if bb > 0 else math.inf",
     "return d.stack_effectif / bb if bb > 0 else 0.0",
     "TestProfondeurBb::test_big_blind_nulle_ne_fabrique_pas_un_tapis_court"),

    ("la profondeur entre dans le nom de famille (dilue les effectifs)",
     'return (regime, f"préflop · ouverture propre · {d.position}")',
     'return (regime, f"préflop · ouverture propre · {d.position} '
     '· {prof_bb}")',
     "TestFamilleDuSpot::test_la_famille_ne_depend_pas_de_la_profondeur"),

    # ── 3. la caractérisation ───────────────────────────────────────────────
    ("l'as ne compte plus comme carte basse dans la connexité",
     "    if 12 in valeurs:\n        valeurs = [-1, *valeurs]",
     "    if False:\n        valeurs = [-1, *valeurs]",
     "TestTextureBoard::test_l_as_compte_aussi_comme_carte_basse"),

    ("texture lue sur le flop seul, turn et river ignorés",
     'cartes = parse_cards(" ".join(board))\n    rangs',
     'cartes = parse_cards(" ".join(board))[:3]\n    rangs',
     "TestTextureBoard::test_appariement_vu_au_turn_et_a_la_river"),

    ("seuil de couleur porté à 4 cartes au lieu de 3",
     "if max(Counter(couleurs).values()) >= 3",
     "if max(Counter(couleurs).values()) >= 4",
     "TestTextureBoard::test_couleur"),

    ("main postflop évaluée sur les 5 premières cartes au lieu de la "
     "meilleure combinaison",
     "    sous = np.asarray(list(itertools.combinations(cartes, 5)), "
     "dtype=np.int64)",
     "    sous = np.asarray([cartes[:5]], dtype=np.int64)",
     "TestTypeDeMain::test_la_meilleure_main_est_prise_au_turn"),

    ("mains assorties et dépareillées confondues",
     'return "assortie" if cartes[0] & 3 == cartes[1] & 3 '
     'else "dépareillée"',
     'return "dépareillée"',
     "TestTypeDeMain::test_preflop_trois_formes"),

    ("l'axe « profondeur » prend un seuil différent du régime",
     '"profondeur": (f"< {SEUIL_TAPIS_COURT_BB:.0f} bb"\n'
     '                       if prof_bb < SEUIL_TAPIS_COURT_BB',
     '"profondeur": (f"< {SEUIL_TAPIS_COURT_BB:.0f} bb"\n'
     '                       if prof_bb < 10.0',
     "TestAxesDuSpot::test_l_axe_profondeur_suit_le_meme_seuil_que_le_regime"),

    # ── 4. la statistique ───────────────────────────────────────────────────
    ("disjonction des intervalles remplacée par un écart de taux de 3 points",
     "    return h1 < b2 or h2 < b1",
     "    return abs(k1 / max(n1, 1) - k2 / max(n2, 1)) > 0.03",
     "TestDisjonctionDesIntervalles::test_un_ecart_de_bruit_ne_l_est_pas"),

    ("les axes comptent tout du côté « accord »",
     "            self.axes[nom][valeur][0 if accord else 1] += 1",
     "            self.axes[nom][valeur][0] += 1",
     "TestComptageDesAxes::test_chaque_axe_totalise_les_spots_tranches"),

    ("la fusion des axes écrase au lieu d'ajouter",
     "                cible[0] += a\n                cible[1] += d",
     "                cible[0] = a\n                cible[1] = d",
     "TestComptageDesAxes::test_la_fusion_conserve_les_totaux"),

    # ── 5. la couverture annoncée (section 9) ───────────────────────────────
    ("priorité des motifs inversée : « (ICM) » cesse de gagner",
     "    for motif, nom, _ in COMPOSANTS:",
     "    for motif, nom, _ in reversed(COMPOSANTS):",
     "TestComposantsSollicites::test_le_motif_le_plus_specifique_gagne"),

    ("un moteur inconnu est rangé dans un composant plausible",
     '    return "« non classé »"',
     "    return COMPOSANTS[-1][1]",
     "TestComposantsSollicites::test_un_moteur_inconnu_reste_visible"),

    ("les composants sont comptés sans être consommés (double compte)",
     "    sortie = [(nom, par_composant.pop(nom, 0), portee)",
     "    sortie = [(nom, par_composant.get(nom, 0), portee)",
     "TestComposantsSollicites::"
     "test_le_total_par_composant_recolle_au_total_des_moteurs"),
)


def _essaie(nom: str, motif: str, remplacement: str, test: str) -> bool:
    """Applique une mutation dans une copie du dépôt et lance le test visé."""
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "python"
        shutil.copytree(RACINE, dst, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache"))
        cible = dst / FICHIER_MUTE
        texte = cible.read_text(encoding="utf-8")
        if texte.count(motif) != 1:
            print(f"  MOTIF ABSENT OU AMBIGU  {nom}")
            print(f"      {texte.count(motif)} occurrence(s) : le code a "
                  "changé. Remets ce banc à jour AVANT de conclure quoi que "
                  "ce soit sur les tests.")
            return False
        cible.write_text(texte.replace(motif, remplacement, 1),
                         encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", f"{FICHIER_TESTS}::{test}",
             "-q", "-p", "no:cacheprovider"],
            cwd=dst, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(dst)})
        tombe = r.returncode != 0
        print(f"  {'OK  ' if tombe else 'RATÉ'}  {nom}")
        if not tombe:
            print(f"      {test} est resté vert sous cette mutation : il ne "
                  "protège pas ce qu'il prétend protéger.")
            print(re.sub(r"^", "      ", r.stdout[-1000:], flags=re.M))
        return tombe


def main() -> int:
    fautes = 0
    print("=== mutations du banc corpus Pluribus ===")
    for nom, motif, remplacement, test in MUTATIONS:
        fautes += not _essaie(nom, motif, remplacement, test)
    total = len(MUTATIONS)
    print("\n" + (f"BANC VERT — {total}/{total} mutations attrapées"
                  if fautes == 0 else
                  f"BANC ROUGE — {fautes}/{total} mutation(s) inaperçue(s)"))
    return fautes


if __name__ == "__main__":
    raise SystemExit(main())
