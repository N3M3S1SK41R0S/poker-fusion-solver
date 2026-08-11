#!/usr/bin/env python
r"""Banc REPÈRES — d'où viennent les bornes affichées par la revue de session.

    python banc_reperes_corpus.py               le rapport complet
    python banc_reperes_corpus.py --verifier    les constantes gelées sont-elles justes ?
    python banc_reperes_corpus.py --geler       réimprime le littéral Python à coller
    python banc_reperes_corpus.py --situer      où se situe l'utilisateur
    python banc_reperes_corpus.py --corpus-pluribus CHEMIN --corpus-wsop CHEMIN

Durée mesurée sur la machine de développement : **12 s** pour le rapport
complet (10 000 fichiers Pluribus lus deux fois — une par mélange de
positions — plus les 83 fichiers WSOP). Aucune équité n'est calculée ici :
le banc ne fait que compter des occasions.

Ce que ce banc établit
----------------------
La revue de session affichait « zone saine 28–50 % » pour le VPIP : un chiffre
de manuel, sans dénominateur ni corpus. Ce banc le remplace par des valeurs
recalculées depuis ``phh-dataset`` (licence MIT, université de Toronto), avec
le **même code de comptage** que celui qui mesure l'utilisateur —
:meth:`~pfs.data.hand_history.ParsedHand.stat_observations`. C'est ce partage
qui rend la comparaison licite : une fréquence de poker n'existe pas sans sa
convention de dénominateur.

``--verifier`` relit le corpus et compare aux constantes de
:data:`pfs.analysis.reperes.REPERES`, chiffre par chiffre, à 10⁻³ près. Il
sort en code 1 au premier écart : les constantes gelées ne peuvent pas
diverger silencieusement du corpus.

Ce que ce banc NE démontre PAS
------------------------------
1. **Pluribus joue du cash 6-max à 100 bb sans ante.** L'utilisateur joue des
   tournois. Aucun chiffre d'ici ne dit ce qu'est un bon VPIP de tournoi.
2. **Les 83 mains WSOP sont un tournoi**, mais 18 seulement sont du hold'em
   (championnat mixte) : 90 mains-joueurs. Le banc les publie avec leurs
   effectifs et **refuse d'en conclure** ; l'intervalle de Wilson du VPIP y
   couvre 19 points.
3. **Un écart au repère n'est pas une faute.** C'est une divergence avec une
   stratégie d'équilibre mesurée dans un AUTRE format. Le banc dit de combien,
   pas si c'est grave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pfs.analysis.reperes import (  # noqa: E402
    CORPUS_PLURIBUS,
    CORPUS_WSOP_2023,
    N_MIN_JOUEUR,
    POSITIONS_TABLE_COURTE,
    REPERES,
    JeuDeReperes,
    ReperesError,
    calculer_jeu,
    jeu_par_defaut,
    situer,
)

_TRAIT = "─" * 78

#: Profil mesuré de l'utilisateur, tel que la revue de session l'a rendu sur
#: ses 721 mains de tournoi (PMU, format court). Recopié ici pour que
#: ``--situer`` soit rejouable sans accès à ses historiques, qui ne sont pas
#: dans le dépôt. Ce n'est PAS une mesure de ce banc : c'est la sortie de
#: ``pfs.analysis.session_review`` sur ses fichiers.
PROFIL_UTILISATEUR = {
    "vpip": 63.1,
    "pfr": 31.3,
    "ecart_vpip_pfr": 31.8,
    "three_bet": 5.4,
    "wtsd": 44.9,
}
MAINS_UTILISATEUR = 721


def _libelle(stat: str) -> str:
    return {
        "vpip": "VPIP",
        "pfr": "PFR",
        "ecart_vpip_pfr": "écart VPIP−PFR",
        "three_bet": "3-bet",
        "fold_to_cbet": "fold to c-bet",
        "wtsd": "WTSD",
        "af_postflop": "AF postflop",
    }.get(stat, stat)


def _titre(txt: str) -> None:
    print()
    print(_TRAIT)
    print(f"  {txt}")
    print(_TRAIT)


def _tableau(jeu: JeuDeReperes) -> None:
    print(f"  source  : {jeu.source}")
    print(f"  lecture : {jeu.n_mains} mains, {jeu.n_mains_joueurs} mains-joueurs"
          + (f", positions {'/'.join(jeu.positions)}" if jeu.positions else ""))
    if not jeu.concluant:
        print("  ⚠ EFFECTIF INSUFFISANT — publié pour dire ce que le corpus")
        print("    contient, pas pour servir de repère.")
    print()
    print(f"  {'statistique':<16}{'valeur':>9}{'occasions':>11}"
          f"{'IC 95 %':>18}{'Q1':>8}{'méd':>8}{'Q3':>8}{'joueurs':>9}")
    for stat, r in jeu.reperes.items():
        ic = (f"[{r.ic_bas:.1f} ; {r.ic_haut:.1f}]"
              if r.ic_bas != r.ic_haut else "—")
        val = f"{r.valeur:.2f}{r.unite}"
        disp = (f"{r.q1:7.1f} {r.mediane:7.1f} {r.q3:7.1f}"
                if r.n_joueurs else f"{'—':>7} {'—':>7} {'—':>7}")
        print(f"  {_libelle(stat):<16}{val:>9}{r.occasions:>11}{ic:>18}"
              f"{disp}{r.n_joueurs:>9}")
    print()
    for lim in jeu.limites:
        print(f"  · {lim}")


def rapport(pluribus: Path, wsop: Path) -> int:
    """Recalcule tout et l'imprime. Rend 0."""
    _titre("1. PLURIBUS — 6-max complet")
    _tableau(calculer_jeu(pluribus, "pluribus_6max",
                          REPERES["pluribus_6max"].source,
                          REPERES["pluribus_6max"].limites))

    _titre("2. PLURIBUS — restreint aux positions d'une table à trois")
    _tableau(calculer_jeu(pluribus, "pluribus_tardives",
                          REPERES["pluribus_tardives"].source,
                          REPERES["pluribus_tardives"].limites,
                          positions=POSITIONS_TABLE_COURTE))
    print()
    print("  L'écart entre les sections 1 et 2 est le biais de mélange de")
    print("  positions : comparer un joueur de table courte à l'agrégat 6-max")
    print("  lui reproche l'UTG et le MP qu'il n'a jamais à jouer.")

    _titre("3. WSOP 2023 — le seul TOURNOI du corpus")
    _tableau(calculer_jeu(wsop, "wsop2023_holdem",
                          REPERES["wsop2023_holdem"].source,
                          REPERES["wsop2023_holdem"].limites,
                          is_tournament=True))

    _titre("4. CE QUE LE CORPUS NE PERMET PAS")
    for ligne in (
        "Le seul tournoi disponible tient en 18 mains de hold'em. Il n'existe",
        "donc AUCUN repère de tournoi mesurable ici. Les repères Pluribus sont",
        "du cash : antes, ICM et tapis courts changent l'équilibre, et de",
        "beaucoup — un VPIP de tournoi n'est pas un VPIP de cash.",
        "",
        "La comparaison la moins malhonnête disponible est la section 2 :",
        "même code de comptage, même mélange de positions qu'une table à",
        "trois. Elle reste du cash, et elle est une BORNE BASSE (à trois, le",
        "bouton affronte deux adversaires et non cinq, donc ouvre plus large).",
    ):
        print(f"  {ligne}")
    print(_TRAIT)
    return 0


def verifier(pluribus: Path, wsop: Path) -> int:
    """Recalcule et compare aux constantes gelées. Rend 1 au premier écart."""
    attendus = {
        "pluribus_6max": lambda: calculer_jeu(
            pluribus, "pluribus_6max", REPERES["pluribus_6max"].source,
            REPERES["pluribus_6max"].limites),
        "pluribus_tardives": lambda: calculer_jeu(
            pluribus, "pluribus_tardives", REPERES["pluribus_tardives"].source,
            REPERES["pluribus_tardives"].limites,
            positions=POSITIONS_TABLE_COURTE),
        "wsop2023_holdem": lambda: calculer_jeu(
            wsop, "wsop2023_holdem", REPERES["wsop2023_holdem"].source,
            REPERES["wsop2023_holdem"].limites, is_tournament=True),
    }
    ecarts = 0
    for cle, fabrique in attendus.items():
        gele = REPERES[cle]
        frais = fabrique()
        if (frais.n_mains, frais.n_mains_joueurs) != (gele.n_mains,
                                                      gele.n_mains_joueurs):
            print(f"  ✗ {cle} : {frais.n_mains} mains / "
                  f"{frais.n_mains_joueurs} mains-joueurs, gelé "
                  f"{gele.n_mains} / {gele.n_mains_joueurs}")
            ecarts += 1
        if set(frais.reperes) != set(gele.reperes):
            print(f"  ✗ {cle} : statistiques {sorted(frais.reperes)} ≠ "
                  f"{sorted(gele.reperes)}")
            ecarts += 1
            continue
        for stat, r in frais.reperes.items():
            g = gele[stat]
            for champ in ("valeur", "occasions", "succes", "ic_bas", "ic_haut",
                          "q1", "mediane", "q3", "n_joueurs"):
                a, b = getattr(r, champ), getattr(g, champ)
                if abs(float(a) - float(b)) > 1e-3:
                    print(f"  ✗ {cle}.{stat}.{champ} : recalculé {a}, "
                          f"gelé {b}")
                    ecarts += 1
        if not ecarts:
            print(f"  ✓ {cle} — {len(frais.reperes)} repères, "
                  f"{frais.n_mains_joueurs} mains-joueurs")
    if ecarts:
        print(f"\n  {ecarts} écart(s) : la table gelée ne correspond plus au "
              "corpus. Relancer --geler et recoller.")
        return 1
    print("\n  Les constantes gelées de pfs/analysis/reperes.py sont exactement")
    print("  ce que le corpus donne aujourd'hui.")
    return 0


def geler(pluribus: Path, wsop: Path) -> int:
    """Réimprime le littéral Python des constantes, prêt à coller."""
    jeux = [
        calculer_jeu(pluribus, "pluribus_6max", "…", ()),
        calculer_jeu(pluribus, "pluribus_tardives", "…", (),
                     positions=POSITIONS_TABLE_COURTE),
        calculer_jeu(wsop, "wsop2023_holdem", "…", (), is_tournament=True),
    ]
    for jeu in jeux:
        print(f'    "{jeu.cle}": JeuDeReperes(')
        print(f'        cle="{jeu.cle}", source=…,')
        print(f'        n_mains={jeu.n_mains}, '
              f'n_mains_joueurs={jeu.n_mains_joueurs},')
        print(f'        positions={jeu.positions!r},')
        print('        reperes={')
        for stat, r in jeu.reperes.items():
            print(f'            "{stat}": Repere("{stat}", {r.valeur:.4f}, '
                  f'{r.occasions}, {r.succes}, {r.ic_bas:.4f}, '
                  f'{r.ic_haut:.4f},')
            print(f'                           {r.q1:.4f}, {r.mediane:.4f}, '
                  f'{r.q3:.4f}, {r.n_joueurs}),')
        print('        },\n        limites=…,\n    ),')
    return 0


def situer_utilisateur() -> int:
    """Place le profil mesuré de l'utilisateur face à chaque jeu de repères."""
    _titre(f"OÙ SE SITUE L'UTILISATEUR ({MAINS_UTILISATEUR} mains de tournoi)")
    print("  Profil mesuré par pfs.analysis.session_review sur ses historiques :")
    for stat, val in PROFIL_UTILISATEUR.items():
        unite = REPERES["pluribus_tardives"][stat].unite
        print(f"    {_libelle(stat):<16}{val:6.1f} {unite}")
    print()
    print(f"  Jeu retenu pour une table à trois : « {jeu_par_defaut(3)} ».")

    for cle in ("pluribus_tardives", "pluribus_6max", "wsop2023_holdem"):
        jeu = REPERES[cle]
        print()
        print(f"  ── face à « {cle} » "
              f"({jeu.n_mains_joueurs} mains-joueurs"
              + ("" if jeu.concluant else ", NON CONCLUANT") + ") ──")
        print(f"  {'statistique':<16}{'lui':>8}{'repère':>9}{'écart':>9}"
              f"{'×repère':>9}  unité / lecture")
        for e in situer(PROFIL_UTILISATEUR, jeu):
            marque = "  ⚠ NON COMPARABLE" if not e.comparable else ""
            print(f"  {_libelle(e.stat):<16}{e.valeur:7.1f} {e.repere:8.1f} "
                  f"{e.ecart:+8.1f} {e.ecart_relatif:8.2f}  "
                  f"{e.unite or 'ratio'}{marque}")
            if not e.comparable:
                print(f"      {e.pourquoi}.")
    print()
    print("  Rappel : ces repères sont du CASH. L'écart mesuré mélange une")
    print("  différence de style et une différence de format. Ce qu'on peut")
    print("  dire honnêtement, c'est qu'aucune différence de format ne rend")
    print("  un VPIP deux fois plus haut nécessaire.")
    print(_TRAIT)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus-pluribus", type=Path, default=CORPUS_PLURIBUS)
    p.add_argument("--corpus-wsop", type=Path, default=CORPUS_WSOP_2023)
    p.add_argument("--verifier", action="store_true",
                   help="comparer les constantes gelées au corpus")
    p.add_argument("--geler", action="store_true",
                   help="réimprimer le littéral Python")
    p.add_argument("--situer", action="store_true",
                   help="placer le profil de l'utilisateur")
    a = p.parse_args(argv)

    try:
        if a.situer:
            return situer_utilisateur()
        if a.verifier:
            return verifier(a.corpus_pluribus, a.corpus_wsop)
        if a.geler:
            return geler(a.corpus_pluribus, a.corpus_wsop)
        return rapport(a.corpus_pluribus, a.corpus_wsop)
    except ReperesError as exc:
        print(f"  corpus indisponible : {exc}", file=sys.stderr)
        print("  Le corpus phh-dataset vit hors du dépôt ; --situer et les",
              file=sys.stderr)
        print("  constantes gelées fonctionnent sans lui.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
