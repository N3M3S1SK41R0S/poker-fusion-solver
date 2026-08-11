"""Interdit de republier une équivalence entre les trois mesures de code mort.

Pourquoi ce fichier existe
--------------------------
Le README et `MISSION_CONSEIL_TOUR_3.md` ont affirmé une « coïncidence
troublante » : 31,7 % de code mort par recensement statique, 31,7 % de
couverture par `coverage.py`, « obtenus par deux méthodes entièrement
différentes ». La phrase était fausse deux fois.

  1. **Arithmétiquement.** 31,7 % de lignes *traversées* implique 68,3 % de
     lignes *non traversées*. Pour qu'une couverture confirme un taux de
     mortalité de 31,7 %, il aurait fallu lire 68,3 %. Les deux 31,7 %
     désignaient les deux moitiés **complémentaires** du fichier, jamais le
     même ensemble de lignes.
  2. **Factuellement.** Le banc de couverture compte 20 modules jamais
     atteints là où le recensement à la main en annonçait 12 : il ne le
     confirmait pas, il le révisait à la hausse.

C'est la famille de défauts que ce dépôt traque : une conclusion plus forte
que la mesure qui la porte — comme le « 97 % des pertes viennent des
décisions » déjà corrigé. La correction ne tient que si rien ne la réécrit :
d'où ce garde-fou.

Ce que ce fichier vérifie
-------------------------
  * aucun document publié ne réaffirme d'équivalence entre (a) l'analyse
    statique, (b) la couverture et (c) l'inertie causale ;
  * le détecteur **tombe** sur la phrase retirée et sur ses paraphrases —
    sinon il ne prouverait rien ;
  * le README publie bien trois lignes, trois bancs, trois valeurs, et ne
    republie pas le même pourcentage pour deux mesures différentes ;
  * les trois bancs existent et se lancent ;
  * importer n'est pas exécuter (le point aveugle de (a)) ;
  * la perturbation voit le défaut n°1 quand on le réinstalle, et ne le voit
    pas quand il est absent (le point aveugle de (a) **et** de (b)).

Limite assumée (NEMESIS)
------------------------
Le détecteur est lexical. Il repère une équivalence affirmée entre deux
familles de mesures dans une même phrase, hors citation et hors négation. Une
équivalence formulée sans aucun de ces mots, ou étalée sur trois paragraphes,
lui échapperait : il rend le défaut coûteux à réintroduire, pas impossible.
Le texte entre guillemets français est explicitement exempté — citer une
affirmation pour la retirer n'est pas l'affirmer — ce qui laisse ouverte la
possibilité de la réaffirmer entre guillemets. C'est un trou connu, borné, et
préféré à un détecteur qui interdirait de citer ses propres erreurs.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

RACINE_PY = Path(__file__).resolve().parents[1]
DEPOT = RACINE_PY.parent

#: Les documents publiés qui parlent de code mort. Un document absent de cette
#: liste n'est pas surveillé : l'ajouter est le prix d'une nouvelle publication.
DOCUMENTS = (
    "README.md",
    "MISSION_CONSEIL_TOUR_3.md",
    "BRIEF_CONSEIL_DES_MODELES.md",
    "MESSAGE_ACCOMPAGNEMENT.md",
    "REPONSE_CONSEIL_TOUR_1.md",
)

BANCS = {
    "a": "banc_atteignabilite_statique.py",
    "b": "banc_couverture_parcours.py",
    "c": "banc_inertie_causale.py",
}

# ── Le détecteur ───────────────────────────────────────────────────────────

#: Trois familles de mesure. Une phrase qui n'en cite qu'une ne peut pas
#: affirmer d'équivalence entre deux d'entre elles.
FAMILLES: dict[str, tuple[str, ...]] = {
    "statique": (r"jamais import", r"aucun import", r"code mort",
                 r"recensement", r"atteignabilit[ée] statique",
                 r"banc_atteignabilite_statique"),
    "couverture": (r"couvert", r"couverture", r"coverage", r"travers[ée]",
                   r"banc_couverture_parcours"),
    "inertie": (r"inerti", r"inerte", r"perturbation",
                r"banc_inertie_causale"),
}

#: Les tournures qui posent une égalité ou une confirmation mutuelle.
EQUIVALENCE: tuple[str, ...] = (
    r"co[ïi]ncidence",
    r"le m[êe]me\s+(?:chiffre|taux|pourcentage|nombre|r[ée]sultat|\d)",
    r"m[êe]me\s+(?:chiffre|taux|pourcentage|valeur|proportion)",
    r"deux m[ée]thodes\s+(?:\w+\s+)?diff[ée]rentes",
    r"confirm",           # confirme, confirmait, confirmation
    r"corrobor",
    r"recoup",
    r"convergen?t? vers le",
    r"[ée]quivalen",
    r"identique",
    r"se rejoign",
    r"valide (?:le|la|notre|ce)",
)

#: Une négation dans la même proposition renverse le sens : « il ne confirme
#: pas » est le contraire de « il confirme ». On ne l'accepte que si elle
#: précède immédiatement la tournure — sinon un « ne » en début de paragraphe
#: blanchirait n'importe quelle affirmation plus loin.
NEGATIONS = re.compile(r"\b(?:ne|n'|pas|aucune?|jamais|ni|sans|non)\b",
                       re.IGNORECASE)
PORTEE_NEGATION = 60          # caractères, mesurés avant la tournure

#: Le texte cité entre guillemets français est retiré avant analyse.
CITATION = re.compile(r"«[^»]*»")


def _unites(texte: str) -> list[tuple[int, str]]:
    """Découpe en paragraphes, les lignes de tableau restant séparées.

    Le paragraphe est la bonne maille : l'affirmation retirée tenait sur deux
    phrases (« Coïncidence troublante : le même 31,7 % … » puis « Un tiers du
    code est traversé… »), et une découpe à la phrase l'aurait laissée passer
    — vérifié, le détecteur rendait alors une liste vide.

    Les lignes de tableau Markdown font exception et restent des unités
    isolées : sans cela, trois lignes voisines publiant honnêtement trois
    mesures différentes fusionneraient en une fausse équivalence.
    """
    unites: list[tuple[int, str]] = []
    bloc: list[str] = []
    debut = 1

    def _fermer() -> None:
        # Les guillemets sont retirés APRÈS l'assemblage : une citation
        # française court souvent sur deux lignes, et les nettoyer ligne à
        # ligne les laissait passer intactes.
        if bloc:
            unites.append((debut, CITATION.sub(" ", " ".join(bloc))))
        bloc.clear()

    for n, brute in enumerate(texte.splitlines(), start=1):
        nue = brute.lstrip()
        if not nue:
            _fermer()
            continue
        if nue.startswith("|"):              # ligne de tableau : unité isolée
            _fermer()
            unites.append((n, CITATION.sub(" ", nue)))
            continue
        if re.match(r"(?:[*+-]|\d+[.)])\s", nue):   # nouvelle puce de liste
            _fermer()
        if not bloc:
            debut = n
        bloc.append(nue.rstrip())
    _fermer()
    return unites


def equivalences(texte: str) -> list[tuple[int, str, str]]:
    """Les affirmations d'équivalence entre deux familles : (ligne, motif, texte)."""
    trouvees: list[tuple[int, str, str]] = []
    for ligne, unite in _unites(texte):
        familles = {f for f, motifs in FAMILLES.items()
                    if any(re.search(m, unite, re.IGNORECASE) for m in motifs)}
        if len(familles) < 2:
            continue
        for motif in EQUIVALENCE:
            for trouve in re.finditer(motif, unite, re.IGNORECASE):
                avant = unite[max(0, trouve.start() - PORTEE_NEGATION):trouve.start()]
                if NEGATIONS.search(avant):
                    continue
                trouvees.append((ligne, motif, unite))
                break
    return trouvees


# ── 1. Les documents publiés ───────────────────────────────────────────────


@pytest.mark.parametrize("document", DOCUMENTS)
def test_aucun_document_ne_reaffirme_l_equivalence(document: str) -> None:
    chemin = DEPOT / document
    if not chemin.is_file():
        pytest.skip(f"{document} absent du dépôt")
    fautes = equivalences(chemin.read_text(encoding="utf-8"))
    assert not fautes, (
        f"{document} réaffirme une équivalence entre deux mesures de code "
        f"mort. Ces mesures sont indépendantes et ne se confirment pas :\n"
        + "\n".join(f"  ligne {n} (motif « {m} ») : {t}" for n, m, t in fautes))


# ── 2. Preuve que le détecteur n'est pas décoratif ─────────────────────────


#: La phrase exacte retirée de `MISSION_CONSEIL_TOUR_3.md` le 11 août 2026.
PHRASE_RETIREE = (
    "Coïncidence troublante : le même 31,7 %, obtenu par une méthode "
    "entièrement différente. Un tiers du code est traversé quand "
    "l'utilisateur s'en sert. Le recensement statique est donc confirmé."
)

#: Des reformulations : un détecteur qui ne verrait que la phrase d'origine
#: serait un test de chaîne de caractères, pas un garde-fou.
PARAPHRASES = (
    "Les deux bancs donnent le même taux : la couverture corrobore le "
    "recensement statique.",
    "Le pourcentage de couverture est identique au pourcentage de code mort "
    "trouvé sans aucun import.",
    "La mesure d'inertie confirme la couverture, et les deux valident le "
    "recensement.",
    "Code mort et couverture se recoupent exactement.",
)


def test_le_detecteur_tombe_sur_la_phrase_retiree() -> None:
    assert equivalences(PHRASE_RETIREE), (
        "le détecteur laisse passer la phrase même qu'il existe pour "
        "interdire : il ne prouve rien")


@pytest.mark.parametrize("phrase", PARAPHRASES)
def test_le_detecteur_tombe_sur_les_paraphrases(phrase: str) -> None:
    assert equivalences(phrase), (
        f"reformulation non détectée, le garde-fou se contourne : {phrase}")


def test_le_detecteur_laisse_passer_le_constat_correct() -> None:
    """Un garde-fou qui refuse aussi la formulation juste est inutilisable."""
    correct = (
        "Ces trois nombres ne sont ni égaux ni comparables, et aucun ne "
        "confirme les autres. La couverture ne confirmait pas le recensement : "
        "elle le révisait à la hausse. Le code jamais importé et le code "
        "jamais traversé sont deux ensembles distincts, mesurés par deux bancs "
        "différents."
    )
    assert not equivalences(correct), (
        "le détecteur refuse la formulation correcte : il est trop large")


def test_le_detecteur_exige_deux_familles() -> None:
    """« confirme » seul ne suffit pas : sans deux mesures, pas d'équivalence."""
    innocent = ("Le banc de couverture confirme que la route est traversée "
                "à 42 %.")
    assert not equivalences(innocent)


# ── 3. Ce que le README doit publier ───────────────────────────────────────


def _lignes_du_tableau() -> dict[str, str]:
    """Les trois lignes « (a) », « (b) », « (c) » du tableau du README."""
    readme = (DEPOT / "README.md").read_text(encoding="utf-8")
    lignes: dict[str, str] = {}
    for ligne in readme.splitlines():
        trouve = re.match(r"\|\s*\*\*\((a|b|c)\)\*\*\s*\|", ligne)
        if trouve:
            lignes[trouve.group(1)] = ligne
    return lignes


def test_le_readme_publie_trois_mesures_et_trois_bancs() -> None:
    lignes = _lignes_du_tableau()
    assert set(lignes) == {"a", "b", "c"}, (
        "le README doit publier les trois mesures séparément, une ligne par "
        f"méthode ; trouvé : {sorted(lignes)}")
    for cle, banc in BANCS.items():
        assert banc in lignes[cle], (
            f"la ligne ({cle}) doit nommer son banc rejouable {banc} — "
            "un chiffre sans banc n'est pas vérifiable")


def test_le_readme_ne_republie_pas_le_meme_pourcentage_pour_deux_mesures() -> None:
    """Le cœur du défaut retiré : un pourcentage servant deux mesures."""
    lignes = _lignes_du_tableau()
    pourcentages = {cle: set(re.findall(r"\d+[,.]\d+\s*%", ligne))
                    for cle, ligne in lignes.items()}
    for gauche, droite in (("a", "b"), ("a", "c"), ("b", "c")):
        commun = pourcentages[gauche] & pourcentages[droite]
        assert not commun, (
            f"le même pourcentage {sorted(commun)} est publié pour ({gauche}) "
            f"et ({droite}). C'est exactement le rapprochement retiré : deux "
            "mesures indépendantes qui affichent la même valeur doivent être "
            "expliquées, pas présentées comme une confirmation.")


# ── 4. Les trois bancs existent et se lancent ──────────────────────────────


@pytest.mark.parametrize("banc", sorted(BANCS.values()))
def test_chaque_banc_est_rejouable(banc: str) -> None:
    chemin = RACINE_PY / banc
    assert chemin.is_file(), f"{banc} annoncé par le README et absent"
    env = dict(os.environ, PYTHONPATH=str(RACINE_PY), PYTHONIOENCODING="utf-8")
    fini = subprocess.run([sys.executable, str(chemin), "--help"],
                          capture_output=True, text=True, timeout=120,
                          cwd=str(RACINE_PY), env=env)
    assert fini.returncode == 0, f"{banc} --help échoue :\n{fini.stderr}"


# ── 5. Le point aveugle de (a) : importer n'est pas exécuter ───────────────


def test_importer_n_est_pas_executer() -> None:
    """`pfs.engine` est atteignable par import et n'est pas chargé pour autant.

    C'est la raison pour laquelle (a) et (b) ne peuvent pas se confirmer : le
    banc statique voit le lien d'import de `API.resolve` vers `pfs.engine` et
    déclare le module vivant ; à l'exécution, l'import est **différé** dans le
    corps de la route, et rien ne le charge tant que personne n'appelle
    `/api/resolve`. La couverture, elle, mesure 0 ligne sur 200.
    """
    import banc_atteignabilite_statique as statique

    graphe = statique.construire_graphe(statique._fichiers_paquet())
    vus = statique.atteignables(graphe, statique.ENTREES_APP)
    assert "pfs.engine" in vus, (
        "pfs.engine n'est plus atteignable par import : si la route resolve a "
        "disparu, ce test doit être réécrit sur un autre module à import "
        "différé, pas supprimé")

    fini = subprocess.run(
        [sys.executable, "-c",
         "import sys; import pfs.app.server; "
         "print('pfs.engine' in sys.modules)"],
        capture_output=True, text=True, timeout=180, cwd=str(RACINE_PY),
        env=dict(os.environ, PYTHONPATH=str(RACINE_PY),
                 PYTHONIOENCODING="utf-8"))
    assert fini.returncode == 0, fini.stderr
    assert fini.stdout.strip() == "False", (
        "pfs.engine est désormais chargé à l'import du serveur : l'écart entre "
        "(a) et (b) que ce test épingle a changé de nature, réécrire le test")


# ── 6. Le point aveugle de (a) ET de (b) : l'inertie causale ───────────────


@pytest.fixture
def banc_reduit(monkeypatch, tmp_path):
    """Le banc d'inertie sur deux spots et deux champs — assez pour trancher."""
    import banc_inertie_causale as banc

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(banc, "SPOTS", [
        ("ICM bulle A7o",
         {"hero": "Ah 7d", "big_blind": 1, "stack": 10, "players": 3,
          "position": "SB", "pot": 1.5, "bet": 0,
          "stacks": "10, 40, 1", "payouts": "50, 30, 20"}),
        ("chipEV AA 10bb",
         {"hero": "Ah Ad", "stack": 10, "big_blind": 1, "players": 2,
          "position": "SB"}),
    ])
    monkeypatch.setattr(banc, "PERTURBATIONS", {
        "stacks": "5, 200, 200",
        "hero": "2c 7d",
    })
    return banc


def test_sans_defaut_les_tapis_de_tournoi_atteignent_le_verdict(banc_reduit) -> None:
    r = banc_reduit.mesurer(faire1=True, faire2=False, defaut=None)
    assert r["niveau1"]["champs_inertes"] == [], (
        "un champ du formulaire n'atteint plus la sortie : "
        f"{r['niveau1']['champs_inertes']}")


def test_avec_le_defaut_n1_reinstalle_le_banc_declare_les_tapis_inertes(
        banc_reduit) -> None:
    """Preuve que la perturbation voit ce que couverture et import ne voient pas.

    La route reçoit `stacks` et le jette — le défaut d'origine, à la virgule
    près. Toutes les lignes restent traversées, tous les modules restent
    importés : ni (a) ni (b) ne bougent d'un pouce. Seule la perturbation
    tombe.
    """
    r = banc_reduit.mesurer(faire1=True, faire2=False, defaut="stacks")
    assert "stacks" in r["niveau1"]["champs_inertes"], (
        "le défaut n°1 est réinstallé et le banc ne le voit pas : "
        "il ne prouve rien")
    assert r["niveau1"]["bouge_sur"]["hero"], (
        "la main du héros doit rester sensible : si plus rien ne bouge, le "
        "banc mesure une panne du serveur, pas une inertie causale")


def test_le_banc_rend_la_route_intacte_apres_avoir_injecte_le_defaut(
        banc_reduit) -> None:
    """Le banc ne doit pas laisser le défaut installé derrière lui.

    `pfs.app.server.ROUTES` est un dictionnaire de module : l'injection y
    survit à la fin du banc si personne ne la retire. Constaté le 11 août
    2026 — sans rétablissement, **neuf tests de `test_parcours_complet.py`
    tombaient**, mais seulement quand pytest les exécutait après ce fichier.
    Un défaut qui dépend de l'ordre alphabétique des fichiers de test est
    précisément le vert trompeur que ce dépôt traque, alors on l'épingle.
    """
    from pfs.app import server as srv_mod

    avant = srv_mod.ROUTES["advise"]
    banc_reduit.mesurer(faire1=True, faire2=False, defaut="stacks")
    assert srv_mod.ROUTES["advise"] is avant, (
        "le banc a laissé la route amputée en place : tout test exécuté "
        "ensuite recevra un verdict de cash game sur un spot de tournoi")
