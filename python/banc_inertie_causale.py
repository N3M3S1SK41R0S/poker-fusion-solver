#!/usr/bin/env python
"""Quels calculs sont TRAVERSÉS mais n'ont AUCUN effet sur la sortie ?

    python banc_inertie_causale.py                  les deux niveaux
    python banc_inertie_causale.py --entrees        niveau 1 seul
    python banc_inertie_causale.py --symboles       niveau 2 seul
    python banc_inertie_causale.py --sortie action  inertie POUR LA DÉCISION
    python banc_inertie_causale.py --detail         la matrice complète
    python banc_inertie_causale.py --defaut stacks  réintroduit le défaut n°1
    python banc_inertie_causale.py --json           sortie machine

La sixième catégorie de code
----------------------------
Un module peut être importé (donc « vivant » pour l'analyse statique) et
traversé (donc « couvert » pour `coverage.py`) tout en n'ayant **aucune
influence** sur ce que l'utilisateur lit. C'était le cas de `/api/advise`, qui
construisait `stacks` et `payouts` puis les jetait : la route s'exécutait,
la couverture était verte, et le régime ICM ne pouvait pas s'atteindre. Le
verdict rendu sur une bulle était l'INVERSE du bon.

Ni l'import ni la couverture ne voient ce défaut : les deux mesurent le
passage, aucune ne mesure la **causalité**. D'où ce banc, qui pose la seule
question qui la mesure :

    si je modifie une entrée, la sortie finale bouge-t-elle ?

Si elle ne bouge jamais, alors tout le calcul qui consommait cette entrée est
causalement inerte sur ce chemin — quelle que soit sa couverture.

Deux niveaux, deux portées
--------------------------
**Niveau 1 — perturbation d'entrée (exhaustif sur les champs testés).**
Chaque champ du formulaire est modifié, un par un, sur une batterie de spots
couvrant les quatre régimes ; on compare la réponse HTTP entière. Un champ
qui ne fait bouger la sortie sur **aucun** spot est le signal exact du défaut
n°1. Ce niveau est complet et automatique.

**Niveau 2 — perturbation de sortie interne (échantillon nommé).**
Chaque symbole d'une liste explicite est enveloppé : sa valeur de retour est
perturbée, la batterie rejouée, la réponse comparée. Quatre verdicts, et c'est
le dernier qui est la catégorie recherchée :

  * `NON APPELÉ PAR CE NOM` — ce lien de nom n'a pas servi sur le chemin ;
  * `NON PERTURBABLE` — le type de retour échappe aux règles de `_perturber` :
    on ne conclut rien, surtout pas l'inertie ;
  * `SENSIBLE` — au moins une réponse a bougé : le calcul sert ;
  * `INERTE` — appelé, perturbé, et **rien n'a bougé** : le résultat est
    calculé puis abandonné.

Le choix de la sortie change le verdict, et il est donc explicite
-----------------------------------------------------------------
`--sortie complete` (défaut) compare la réponse HTTP entière : un calcul qui
ne déplace qu'une phrase d'explication est déclaré branché, parce qu'il l'est.
`--sortie action` ne compare que le verdict appliqué par l'utilisateur, et
répond à une question plus dure : *ce calcul change-t-il la décision ?*
Mesure du 11 août 2026 : 0 symbole inerte au sens `complete`, **7 sur 13** au
sens `action` — les mêmes calculs déplacent l'explication sans renverser le
verdict **sur cette batterie**. Ce n'est pas un défaut : c'est la distance des
six spots à leur seuil de bascule. Un spot posé sur le seuil les rendrait
sensibles. Le nombre `action` mesure la batterie autant que le code, et se lit
avec elle.

Le piège de la mémoïsation, constaté puis corrigé
-------------------------------------------------
Première version de ce banc : `solve_hu_pushfold` et `equity_matrix_169`
rapportés « 0 appel ». Faux. Les valeurs étaient en cache (`lru_cache` pour la
première, singleton `_EQUITY_MATRIX` écrit à la main pour la seconde) depuis la
mesure de référence, et l'enveloppe n'était jamais atteinte. Un banc d'inertie
sans vidage de caches produit des **faux « jamais appelé »** — exactement le
genre de vert trompeur que ce dépôt traque. Cf. `_vider_caches`.

Jusqu'où ce banc va, et où il s'arrête (NEMESIS)
-----------------------------------------------
Ce qu'il prouve : pour les champs et symboles **listés ici**, sur les spots
**listés ici**, via la route `/api/advise`.

Ce qu'il ne prouve pas, et qu'il ne faut pas lui faire dire :

  * il ne rend **aucun pourcentage de lignes**. L'inertie est mesurée par
    champ et par symbole, pas par ligne : une même ligne sert plusieurs
    chemins, et lui attribuer une causalité demanderait un découpage que ce
    banc ne fait pas. Le nombre publié en (c) est donc un **compte de
    symboles**, pas une part du code, et il n'est comparable ni à (a) ni à (b) ;
  * il ne couvre qu'une route sur trente-deux. Les trente et une autres n'ont
    pas été soumises à perturbation ;
  * un symbole `SENSIBLE` n'est pas prouvé **juste** — seulement branché ;
  * un symbole `INERTE` ne l'est que sur cette batterie : l'absence d'effet
    observé n'est pas une preuve d'absence d'effet ;
  * un symbole dont la valeur de retour n'est pas perturbable par les règles
    de `_perturber` est déclaré `NON PERTURBABLE`, jamais `INERTE`. Confondre
    les deux serait exactement le travers que ce banc existe pour corriger.

Preuve que ce banc n'est pas tautologique
-----------------------------------------
`--defaut stacks` réinstalle le défaut n°1 (la route reçoit le champ et le
jette). Mesuré le 11 août 2026 : **0 champ inerte sur 12 sans le défaut, 3
avec** — `stacks`, mais aussi `payouts` et `ante`, qui n'avaient plus de
régime ICM où agir. Un champ jeté en éteint donc trois : c'est la forme exacte
qu'avait le défaut d'origine, et la raison pour laquelle il inversait un
verdict sans rien casser d'apparent.
`tests/test_trois_nombres_distincts.py` rejoue les deux sens.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import socket
import sys
import tempfile
import threading
import urllib.error
import urllib.request

JETON = "jeton-du-banc-inertie"

# ═══════════════════════════════════════════════════════════════════════════
# LA BATTERIE DE SPOTS
# ═══════════════════════════════════════════════════════════════════════════

#: Les quatre régimes de `advise`, repris des spots du parcours complet : un
#: banc d'inertie ne vaut que par la variété des chemins qu'il déclenche. Avec
#: un seul spot préflop, `board` paraîtrait inerte alors qu'il ne l'est pas.
SPOTS: list[tuple[str, dict]] = [
    ("chipEV AA 10bb",
     {"hero": "Ah Ad", "stack": 10, "big_blind": 1, "players": 2,
      "position": "SB"}),
    ("chipEV 95o 20bb",
     {"hero": "9h 5d", "stack": 20, "big_blind": 1, "players": 2,
      "position": "SB"}),
    ("ICM bulle A7o",
     {"hero": "Ah 7d", "big_blind": 1, "stack": 10, "players": 3,
      "position": "SB", "pot": 1.5, "bet": 0,
      "stacks": "10, 40, 1", "payouts": "50, 30, 20"}),
    ("postflop brelan",
     {"hero": "7s 7h", "board": "Qs 7d 2c", "pot": 100, "bet": 75,
      "stack": 300, "big_blind": 10, "position": "BTN"}),
    ("postflop AJ tournoi",
     {"hero": "Ad Jh", "board": "Qs 9d 4c 2h", "pot": 100, "bet": 50,
      "stack": 300, "big_blind": 10, "position": "BTN", "players": 3,
      "stacks": "300, 900, 60", "payouts": "50, 30, 20"}),
    ("profond AKo BTN",
     {"hero": "Ah Kd", "stack": 100, "big_blind": 1, "players": 6,
      "position": "BTN"}),
]

#: Valeur de remplacement de chaque champ, et pourquoi elle est choisie ainsi.
#: Le principe : rester dans le domaine valide du champ, pour que la sortie
#: qui ne bouge pas soit une information sur le câblage et non sur un refus.
PERTURBATIONS: dict[str, object] = {
    # une autre main, de force très différente
    "hero": "2c 7d",
    # un board complet là où il y en a un, un flop là où il n'y en avait pas
    "board": "As Ks Qs",
    "pot": 250.0,
    "bet": 5.0,
    "stack": 47.0,
    "big_blind": 4.0,
    "position": "UTG",
    "villain": "serré",
    "players": 5,
    # bulle très marquée : si le régime ICM est atteignable, cela se voit
    "stacks": "5, 200, 200",
    "payouts": "80, 15, 5",
    "ante": 0.25,
}

#: Symboles perturbés au niveau 2. Ils sont nommés **dans l'espace de noms où
#: ils sont liés** (`spot_advisor` fait `from … import …` : remplacer
#: `pfs.core.icm.bubble_factor` n'aurait aucun effet, le nom y est déjà résolu).
SYMBOLES: list[tuple[str, str]] = [
    ("pfs.analysis.spot_advisor", "bubble_factor"),
    ("pfs.analysis.spot_advisor", "icm_required_equity"),
    ("pfs.analysis.spot_advisor", "risk_premium"),
    ("pfs.analysis.spot_advisor", "equity_vs_range"),
    ("pfs.analysis.spot_advisor", "required_equity"),
    ("pfs.analysis.spot_advisor", "minimum_defence_frequency"),
    ("pfs.analysis.spot_advisor", "solve_hu_pushfold"),
    ("pfs.analysis.spot_advisor", "equity_matrix_169"),
    # Un cran plus bas, dans les modules de calcul eux-mêmes : `bubble_factor`
    # appelle `icm_equities` par son nom local, la perturbation l'atteint donc.
    ("pfs.core.icm", "icm_equities"),
    ("pfs.core.icm", "icm_dollar_ev"),
    ("pfs.core.icm", "fgs_equities"),
    ("pfs.core.bluffcatch", "d_prime"),
    ("pfs.core.range_model", "parse_range"),
]

#: Mémoïsations écrites à la main — invisibles à `cache_clear`. Sans cette
#: remise à zéro, `equity_matrix_169` était rapporté « JAMAIS APPELÉ » alors
#: qu'il alimente les deux solutions de Nash : le singleton `_EQUITY_MATRIX`
#: était rempli au premier appel du processus et plus jamais recalculé.
#: (module, attribut, valeur neutre)
SINGLETONS: list[tuple[str, str, object]] = [
    ("pfs.analysis.spot_advisor", "_EQUITY_MATRIX", None),
]


# ═══════════════════════════════════════════════════════════════════════════
# LE SERVEUR ET LE CLIENT
# ═══════════════════════════════════════════════════════════════════════════


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _poster(base: str, payload: dict) -> tuple[int, dict]:
    """POST /api/advise, et la réponse telle que la page la reçoit."""
    req = urllib.request.Request(
        f"{base}/api/advise", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-PFS-Token": JETON})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def _vider_caches() -> int:
    """Vide toutes les mémoïsations de `pfs`, et rend leur nombre.

    Sans cela, le niveau 2 ment. `_nash_solution`, `_villain_range`,
    `load_templates`… sont `lru_cache` : après la mesure de référence, la
    valeur est en cache et l'enveloppe posée sur le symbole **n'est jamais
    appelée**. Le banc concluait alors « JAMAIS APPELÉ » sur des fonctions qui
    sont au contraire au cœur du chemin — un faux négatif exactement du type
    que ce dépôt traque. Constaté puis corrigé le 11 août 2026 :
    `solve_hu_pushfold` et `equity_matrix_169` passaient de 0 appel à des
    appels réels une fois les caches vidés.
    """
    vides = 0
    for nom, module in list(sys.modules.items()):
        if not (nom == "pfs" or nom.startswith("pfs.")) or module is None:
            continue
        for attribut in list(vars(module).values()):
            vider = getattr(attribut, "cache_clear", None)
            if callable(vider):
                vider()
                vides += 1
    for module_nom, attribut, neutre in SINGLETONS:
        module = sys.modules.get(module_nom)
        if module is not None and hasattr(module, attribut):
            setattr(module, attribut, neutre)
            vides += 1
    return vides


#: Quelle « sortie » compte comme la sortie ? Le choix change le verdict, donc
#: il est explicite et imprimé dans le rapport.
#:   * `complete` : la réponse HTTP entière. Un calcul qui ne déplace qu'une
#:     phrase d'explication est déclaré branché — il l'est.
#:   * `action` : le seul verdict que l'utilisateur applique. Un calcul qui ne
#:     le déplace jamais est inerte **pour la décision**, ce qui est une
#:     question différente et plus dure.
PORTEES = ("complete", "action")


def _empreinte(code: int, rep: dict, portee: str = "complete") -> str:
    """La sortie observable, sous une forme comparable."""
    if portee == "action":
        rep = {"action": rep.get("action"), "error": rep.get("error")}
    return f"{code}|{json.dumps(rep, sort_keys=True, ensure_ascii=False)}"


# ═══════════════════════════════════════════════════════════════════════════
# NIVEAU 1 — PERTURBATION D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════


def _mesure(base: str, payload: dict, portee: str) -> str:
    """Une réponse, caches vidés : chaque appel repart du même état."""
    _vider_caches()
    code, rep = _poster(base, payload)
    return _empreinte(code, rep, portee)


def niveau1(base: str, portee: str) -> dict:
    references = {nom: _mesure(base, p, portee) for nom, p in SPOTS}

    par_champ: dict[str, list[str]] = {c: [] for c in PERTURBATIONS}
    ignores: dict[str, list[str]] = {c: [] for c in PERTURBATIONS}
    for nom, payload in SPOTS:
        for champ, valeur in PERTURBATIONS.items():
            if payload.get(champ) == valeur:
                ignores[champ].append(nom)      # rien à perturber : on le dit
                continue
            perturbe = dict(payload)
            perturbe[champ] = valeur
            if _mesure(base, perturbe, portee) != references[nom]:
                par_champ[champ].append(nom)

    inertes = sorted(c for c, spots in par_champ.items() if not spots)
    return {
        "portee": portee,
        "spots": [n for n, _ in SPOTS],
        "champs": sorted(PERTURBATIONS),
        "bouge_sur": {c: sorted(v) for c, v in par_champ.items()},
        "ignores": {c: v for c, v in ignores.items() if v},
        "champs_inertes": inertes,
        "essais": sum(len(SPOTS) - len(ignores[c]) for c in PERTURBATIONS),
    }


# ═══════════════════════════════════════════════════════════════════════════
# NIVEAU 2 — PERTURBATION DE SORTIE INTERNE
# ═══════════════════════════════════════════════════════════════════════════


def _perturber(v):
    """Renvoie (valeur perturbée, perturbation réussie ?).

    Les flottants de [0, 1] — équités, fréquences — sont décalés modulo 1 pour
    rester dans leur domaine : un banc qui les ferait sortir du domaine
    mesurerait la levée d'exception, pas l'inertie.
    """
    if isinstance(v, bool):
        return (not v), True
    if isinstance(v, float):
        return ((v + 0.37) % 1.0, True) if 0.0 <= v <= 1.0 else (v * 1.37 + 0.11, True)
    if isinstance(v, int):
        return v + 1, True
    if isinstance(v, (tuple, list)):
        sortie, touche = [], False
        for x in v:
            y, ok = _perturber(x)
            sortie.append(y)
            touche |= ok
        return (type(v)(sortie) if isinstance(v, list) else tuple(sortie)), touche
    if isinstance(v, dict):
        sortie, touche = {}, False
        for k, x in v.items():
            y, ok = _perturber(x)
            sortie[k] = y
            touche |= ok
        return sortie, touche
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        # `equity_vs_range` rend un `EquityResult` : sans ce cas, le banc le
        # déclarait NON PERTURBABLE et n'apprenait rien du maillon principal.
        remplaces, touche = {}, False
        for champ in dataclasses.fields(v):
            y, ok = _perturber(getattr(v, champ.name))
            if ok:
                remplaces[champ.name] = y
                touche = True
        if not touche:
            return v, False
        try:
            return dataclasses.replace(v, **remplaces), True
        except Exception:            # noqa: BLE001 — validation en __post_init__
            return v, False
    try:                                  # tableaux numpy et assimilés
        import numpy as np
        if isinstance(v, np.ndarray) and v.size and np.issubdtype(v.dtype, np.number):
            return (v + 0.37) % 1.0 if v.dtype.kind == "f" else v + 1, True
    except ImportError:                   # pragma: no cover — numpy est requis
        pass
    return v, False


def niveau2(base: str, portee: str) -> dict:
    references = {nom: _mesure(base, p, portee) for nom, p in SPOTS}

    resultats: dict[str, dict] = {}
    for module_nom, attribut in SYMBOLES:
        module = sys.modules.get(module_nom) or __import__(
            module_nom, fromlist=["_"])
        original = getattr(module, attribut, None)
        cle = f"{module_nom}.{attribut}"
        if original is None or not callable(original):
            resultats[cle] = {"verdict": "ABSENT", "appels": 0, "bouge_sur": []}
            continue

        compteur = {"appels": 0, "perturbes": 0}

        def enveloppe(*a, _o=original, _c=compteur, **k):
            _c["appels"] += 1
            v = _o(*a, **k)
            perturbe, ok = _perturber(v)
            _c["perturbes"] += int(ok)
            return perturbe if ok else v

        setattr(module, attribut, enveloppe)
        try:
            bouge = [nom for nom, p in SPOTS
                     if _mesure(base, p, portee) != references[nom]]
        finally:
            setattr(module, attribut, original)
            _vider_caches()      # ne pas laisser une valeur perturbée en cache

        if compteur["appels"] == 0:
            # « par ce nom » : `spot_advisor` fait ses imports en tête de
            # module, donc perturber `pfs.core.range_model.parse_range` ne
            # touche pas le lien déjà résolu chez lui. Le banc ne dit pas que
            # la fonction n'a pas tourné — il dit que CE lien n'a pas servi.
            verdict = "NON APPELÉ PAR CE NOM"
        elif compteur["perturbes"] == 0:
            verdict = "NON PERTURBABLE"
        elif bouge:
            verdict = "SENSIBLE"
        else:
            verdict = "INERTE"
        resultats[cle] = {"verdict": verdict, "appels": compteur["appels"],
                          "perturbes": compteur["perturbes"],
                          "bouge_sur": bouge}
    return resultats


# ═══════════════════════════════════════════════════════════════════════════


def mesurer(faire1: bool, faire2: bool, defaut: str | None,
            portee: str = "complete") -> dict:
    from pfs.app import server as srv_mod
    from pfs.app.server import create_server

    # `ROUTES` est un dictionnaire de MODULE : y écrire modifie le serveur pour
    # tout le processus. La première version de ce banc n'y remettait jamais la
    # route d'origine — sous pytest, les tests exécutés APRÈS lui recevaient
    # une route amputée et neuf tests du parcours complet tombaient sans que
    # rien n'ait changé dans `pfs/`. Le défaut n'apparaissait que selon l'ordre
    # alphabétique des fichiers de test. D'où ce rétablissement inconditionnel.
    original = srv_mod.ROUTES["advise"]
    if defaut:
        def advise_ampute(p: dict, _o=original, _c=defaut) -> dict:
            """Le défaut n°1 réinstallé : le champ arrive et il est jeté."""
            q = dict(p)
            q.pop(_c, None)
            return _o(q)

        srv_mod.ROUTES["advise"] = advise_ampute

    srv, _ = create_server(_port_libre(), token=JETON)
    fil = threading.Thread(target=srv.serve_forever, daemon=True)
    fil.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        sortie: dict = {"defaut_injecte": defaut, "portee": portee}
        if faire1:
            sortie["niveau1"] = niveau1(base, portee)
        if faire2:
            sortie["niveau2"] = niveau2(base, portee)
        return sortie
    finally:
        srv_mod.ROUTES["advise"] = original
        _vider_caches()
        srv.shutdown()
        fil.join(timeout=5)
        srv.server_close()


def rapport(r: dict, detail: bool) -> None:
    print()
    print("═" * 74)
    print("  (c) CALCULS TRAVERSÉS MAIS SANS EFFET SUR LA SORTIE")
    print("      mesurés par PERTURBATION, route /api/advise")
    portee = r.get("portee", "complete")
    print(f"      sortie observée : {portee}"
          + ("  (réponse HTTP entière)" if portee == "complete"
             else "  (le seul verdict — inerte = « ne change pas la décision »)"))
    if r.get("defaut_injecte"):
        print(f"  ⚠ DÉFAUT INJECTÉ : le champ « {r['defaut_injecte']} » est jeté")
    print("═" * 74)

    if "niveau1" in r:
        n1 = r["niveau1"]
        print(f"  NIVEAU 1 — perturbation d'entrée ({len(n1['spots'])} spots, "
              f"{len(n1['champs'])} champs, {n1['essais']} essais)")
        print("─" * 74)
        for champ in n1["champs"]:
            spots = n1["bouge_sur"][champ]
            etat = "INERTE (aucun spot)" if not spots else f"sensible sur {len(spots)}"
            print(f"  {champ:<14s} {etat}")
            if detail and spots:
                print(f"                 {', '.join(spots)}")
        print(f"  → champs causalement inertes : {len(n1['champs_inertes'])} "
              f"sur {len(n1['champs'])}"
              + (f"  {n1['champs_inertes']}" if n1["champs_inertes"] else ""))
        print()

    if "niveau2" in r:
        n2 = r["niveau2"]
        print(f"  NIVEAU 2 — perturbation de sortie interne ({len(n2)} symboles)")
        print("─" * 74)
        for cle, v in n2.items():
            print(f"  {v['verdict']:<22s} {cle:<46s} "
                  f"{v['appels']:4d} appels")
        inertes = [c for c, v in n2.items() if v["verdict"] == "INERTE"]
        jamais = [c for c, v in n2.items()
                  if v["verdict"] == "NON APPELÉ PAR CE NOM"]
        muets = [c for c, v in n2.items() if v["verdict"] == "NON PERTURBABLE"]
        print(f"  → INERTES : {len(inertes)}")
        for c in inertes:
            print(f"      {c}  ({n2[c]['appels']} appels, sortie immobile)")
        print(f"  → non appelés par ce nom : {len(jamais)} · "
              f"non perturbables : {len(muets)}")
        print()

    print("═" * 74)
    print("  Ce banc compte des CHAMPS et des SYMBOLES, jamais des lignes :")
    print("  son nombre n'est comparable ni à (a) ni à (b).")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entrees", action="store_true", help="niveau 1 seul")
    ap.add_argument("--symboles", action="store_true", help="niveau 2 seul")
    ap.add_argument("--detail", action="store_true", help="matrice complète")
    ap.add_argument("--defaut", metavar="CHAMP",
                    help="réinstalle le défaut n°1 sur ce champ (preuve de chute)")
    ap.add_argument("--sortie", choices=PORTEES, default="complete",
                    help="ce qui compte comme la sortie (défaut : réponse entière)")
    ap.add_argument("--json", action="store_true", help="sortie machine")
    args = ap.parse_args()

    # L'archive des échecs vit sous %LOCALAPPDATA% : la détourner évite qu'un
    # banc noie les vrais échecs de l'utilisateur dans les siens.
    os.environ["LOCALAPPDATA"] = tempfile.mkdtemp(prefix="pfs-banc-inertie-")

    faire1 = args.entrees or not args.symboles
    faire2 = args.symboles or not args.entrees
    r = mesurer(faire1, faire2, args.defaut, args.sortie)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        rapport(r, args.detail)


if __name__ == "__main__":
    main()
