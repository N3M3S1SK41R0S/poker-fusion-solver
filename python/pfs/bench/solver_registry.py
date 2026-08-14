"""
Registre des paramètres solveur — la taxonomie complète, vivante dans le code.

Statut (audit du 14 août 2026) — banc/registre, NON branché (assumé)
--------------------------------------------------------------------
Ce module est de la documentation exécutable : seul
``tests/test_solver_registry.py`` l'importe, et c'est sa place — il n'a pas
vocation à être appelé par l'application. En revanche, l'audit a montré
qu'il MENTAIT par omission : « couvert » signifiait « le code existe et il
est testé », sans dire si ce code est branché dans l'application. D'où le
champ ``wired`` : ``coverage`` mesure l'implémentation, ``wired`` mesure le
branchement (route/UI/moteur). ``library_only()`` liste les paramètres
implémentés et testés mais que rien n'exécute pour le joueur.

Ce module encode **tous les paramètres** que prennent en compte les solveurs
réputés (PioSOLVER, GTO+, Simple, TexasSolver, GTO Wizard AI, MonkerSolver,
HRC, ICMIZER…), organisés selon la taxonomie A→I, et — pour chacun — l'état de
couverture dans le Poker Fusion Solver.

But : que la question « est-ce que tout ça est dans notre logiciel ? » ait une
réponse **exécutable**, pas une affirmation. `coverage_report()` la calcule.

Sources (relevés août 2026) : doc PioSOLVER (UPI/tree), aide GTO Wizard (custom
solving / rake / ICM / nodelock 2.0), SimplePoker, TexasSolver (DCFR/JSON),
HoldemResources, ICMIZER, PT4/HM3/Hand2Note. Références complètes dans le
benchmark livré à part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Category",
    "Coverage",
    "Parameter",
    "SolverProfile",
    "PARAMETERS",
    "SOLVERS",
    "coverage_report",
    "gaps",
    "differentiators",
    "library_only",
]


class Category(str, Enum):
    GAME = "A. Structure de jeu"
    BET_TREE = "B. Arbre de mise"
    RANGES = "C. Ranges"
    BOARD = "D. Board / cartes"
    SOLVING = "E. Solving"
    ICM = "F. ICM / tournoi"
    EXPLOIT = "G. Node locking / exploitation"
    OUTPUT = "H. Métriques de sortie"
    OPPONENT = "I. Adversaire (angle mort des solveurs)"


class Coverage(str, Enum):
    COVERED = "couvert"
    PARTIAL = "partiel"
    ABSENT = "absent"
    PLANNED = "planifié"       # spécifié dans le Plan Directeur, non codé


@dataclass(frozen=True, slots=True)
class Parameter:
    """Un paramètre atomique, et où on en est."""

    name: str
    category: Category
    exposed_by: tuple[str, ...]
    """Solveurs qui exposent ce paramètre."""
    coverage: Coverage
    module: str = ""
    """Module/fusion qui le couvre chez nous (vide si absent)."""
    note: str = ""
    wired: bool = True
    """Branché dans l'application (route/UI/moteur ou script livré).

    ``False`` = implémenté et TESTÉ, mais aucun chemin d'exécution
    utilisateur n'y passe : bibliothèque en attente de branchement (audit du
    14 août 2026). ``coverage`` dit « le code existe », ``wired`` dit « le
    joueur en profite ». Les deux axes sont orthogonaux et le score de
    couverture ne mélange pas les deux — c'est ``library_only()`` qui rend
    l'écart visible."""

    @property
    def is_industry_standard(self) -> bool:
        """Exposé par au moins trois solveurs de référence."""
        return len(self.exposed_by) >= 3


# Raccourcis de nommage des solveurs.
PIO, GTOP, SIMPLE, TEXAS, GTOW, MONKER = (
    "PioSOLVER", "GTO+", "Simple", "TexasSolver", "GTO Wizard AI", "MonkerSolver")
HRC, ICMIZER, DEEPS = "HRC", "ICMIZER", "DeepSolver"
ALL_POSTFLOP = (PIO, GTOP, SIMPLE, TEXAS, GTOW, MONKER, DEEPS)


# ═══════════════════════════════════════════════════════════════════════════
# LE REGISTRE — taxonomie A→I
# ═══════════════════════════════════════════════════════════════════════════

_P = Parameter
_C = Category
_COV, _PAR, _ABS, _PLA = (
    Coverage.COVERED, Coverage.PARTIAL, Coverage.ABSENT, Coverage.PLANNED)

PARAMETERS: tuple[Parameter, ...] = (
    # ── A. Structure de jeu ──────────────────────────────────────────────
    _P("Variante (NLHE)", _C.GAME, ALL_POSTFLOP, _COV, "range_model", "NLHE natif"),
    _P("Variante (PLO / Short Deck)", _C.GAME, (SIMPLE, MONKER, GTOW), _ABS,
       note="hors périmètre — NLHE uniquement"),
    _P("Format (cash / MTT / SNG / Spin)", _C.GAME, (GTOW, SIMPLE, HRC, ICMIZER), _COV,
       "engine", "resolve_spot(game_format=) : cash→rake, mtt→sans rake + ICM ; "
       "push/fold ICM pour l'endgame SNG/MTT"),
    _P("Nombre de joueurs / joueurs restants", _C.GAME, ALL_POSTFLOP, _COV,
       "engine", "multi-joueurs suivis individuellement"),
    _P("Positions (UTG…BB)", _C.GAME, ALL_POSTFLOP, _COV, "range_model.GTO_PRESETS"),
    _P("Stack effectif (bb)", _C.GAME, ALL_POSTFLOP, _COV, "engine"),
    _P("Stacks individuels asymétriques", _C.GAME, (PIO, GTOW, MONKER), _COV,
       "engine", "hero_stack/villain_stack → tapis effectif par paire (HU exact)"),
    _P("SPR de départ", _C.GAME, ALL_POSTFLOP, _COV, "bet_sizing"),
    _P("Blinds", _C.GAME, ALL_POSTFLOP, _COV, "hand_history"),
    _P("Antes / BB ante", _C.GAME, (GTOW, HRC, ICMIZER, SIMPLE), _COV, "hand_history"),
    _P("Straddle (simple/double/Mississippi)", _C.GAME, (GTOW,), _COV,
       "hand_history", "posts parsés, effective_bb, jamais compté VPIP"),
    _P("Rake (% + cap)", _C.GAME, (PIO, GTOW, GTOP, HRC), _COV, "core.rake",
       "min(cap, pct·pot) au terminal ; β* = b/net(P+2b) vérifié en forme close"),
    _P("Rake no-flop-no-drop", _C.GAME, (), _ABS,
       note="aucun solveur ne l'expose nativement — parité"),

    # ── B. Arbre de mise ─────────────────────────────────────────────────
    _P("Sizings par street (flop/turn/river séparés)", _C.BET_TREE,
       (PIO, GTOP, TEXAS, GTOW, SIMPLE), _COV, "bet_sizing.sizing_table"),
    _P("Sizings IP vs OOP séparés", _C.BET_TREE, (PIO, GTOP, TEXAS, GTOW), _COV,
       "solver.postflop", "P1 : oop_bet_fracs / ip_bet_fracs"),
    _P("Expression % du pot", _C.BET_TREE, ALL_POSTFLOP, _COV, "bet_sizing"),
    _P("Expression géométrique (e)", _C.BET_TREE, (GTOW, PIO), _COV, "bet_sizing",
       "sizing géométrique r,r²,r³ — Plan §F8"),
    _P("Donk bets", _C.BET_TREE, (PIO, TEXAS, GTOW), _COV, "solver.postflop",
       "natif : l'OOP peut ouvrir la mise à tout nœud racine"),
    _P("Raises (nombre + tailles)", _C.BET_TREE, ALL_POSTFLOP, _COV,
       "solver.postflop", "raise_fracs dédiées + plafond max_bets"),
    _P("Préflop : RFI / 3bet / 4bet / squeeze", _C.BET_TREE,
       (PIO, GTOW, SIMPLE, MONKER), _COV, "range_model", "presets + tracking"),
    _P("All-in threshold", _C.BET_TREE, (PIO, TEXAS, JES := "Jesolver"), _COV,
       "solver.postflop", "allin_threshold : mise ≥ t·cap → all-in (convention Pio)"),
    _P("Add all-in", _C.BET_TREE, (PIO, TEXAS), _COV, "solver.postflop",
       "add_allin=True : l'all-in comme action systématique"),
    _P("Betting cap (nb relances/street)", _C.BET_TREE, (PIO, TEXAS), _COV,
       "solver.postflop", "max_bets"),
    _P("Merge / seuil de fusion des tailles", _C.BET_TREE, (GTOW,), _COV,
       "solver.postflop", "simplify_report : tailles à ε-EV près déclarées fusionnables"),
    _P("Contraintes de nœud (force bet/check)", _C.BET_TREE, (PIO, GTOW), _COV,
       "solver.postflop", "lock_node à fréquence 1 = action forcée"),
    _P("Mode sizing auto / dynamique", _C.BET_TREE, (GTOW, DEEPS), _COV, "bet_sizing",
       "notre λ dynamique EST un sizing dynamique fondé sur l'information"),

    # ── C. Ranges ────────────────────────────────────────────────────────
    _P("Range préflop par position/action", _C.RANGES, ALL_POSTFLOP, _COV,
       "range_model.parse_range"),
    _P("Poids par combo (fréquences mixtes)", _C.RANGES, ALL_POSTFLOP, _COV,
       "range_model", "1326 combos, poids [0,1]"),
    _P("Combos exacts vs classes", _C.RANGES, ALL_POSTFLOP, _COV, "range_model"),
    _P("Card removal / blockers", _C.RANGES, ALL_POSTFLOP, _COV,
       "range_model.Range.remove_blockers", "effet blocker à la résolution 1326"),
    _P("Bunching (retrait des folds)", _C.RANGES, (GTOW,), _COV,
       "core.bunching", "MC jointe + approximation pairwise (corr. 0.995) — "
       "seul GTO Wizard AI le fait aussi", wired=False),
    _P("Range vs range + normalisation", _C.RANGES, ALL_POSTFLOP, _COV, "range_model"),
    _P("Import/export de ranges", _C.RANGES, ALL_POSTFLOP, _COV,
       "range_model.parse_range", "format solveur standard"),
    _P("Inférence de range en direct", _C.RANGES, (), _COV, "particle",
       "★ AUCUN solveur ne le fait — filtre particulaire sur la stratégie"),

    # ── D. Board ─────────────────────────────────────────────────────────
    _P("Board de départ", _C.BOARD, ALL_POSTFLOP, _COV, "range_model"),
    _P("Texture (connectivité, paires…)", _C.BOARD, ALL_POSTFLOP, _COV, "range_model"),
    _P("Isomorphisme de couleurs", _C.BOARD, (PIO, TEXAS, GTOP), _COV,
       "solver.isomorphism",
       "L8 : 22 100 → 1 755 flops (Burnside vérifié) ; turn 16 432, river "
       "134 459 — brique du blueprint flop Phase 2", wired=False),
    _P("Runouts (énumération turn/river)", _C.BOARD, (PIO, GTOW), _COV, "core.equity",
       "hotness grid + Monte-Carlo"),
    _P("Board clustering (familles de flops)", _C.BOARD, (GTOW, SIMPLE), _PAR,
       "topology", "TDA sur familles — angle différent", wired=False),

    # ── E. Solving ───────────────────────────────────────────────────────
    _P("Algorithme CFR+/DCFR", _C.SOLVING, ALL_POSTFLOP, _COV, "solver.dcfr",
       "DCFR alterné + Hyperparameter Schedules 2024"),
    _P("Cible d'exploitabilité (% pot)", _C.SOLVING, (PIO, TEXAS, GTOP), _COV,
       "solver.dcfr", "exploitabilité exacte, information imparfaite"),
    _P("Critère d'arrêt (temps/itér/seuil)", _C.SOLVING, ALL_POSTFLOP, _COV,
       "solver.dcfr"),
    _P("Abstraction / bucketing", _C.SOLVING, (SIMPLE, MONKER, GTOW), _COV,
       "solver.abstraction", "EHS exact mémoïsé par classe isomorphe + "
       "distribution-aware (Johanson 2013) — validé contre oracle au turn ; "
       "consommateur (blueprint flop) = Phase 2", wired=False),
    _P("Search à profondeur limitée + value net", _C.SOLVING, (GTOW, DEEPS), _COV,
       "solver.postflop", "P3 : leaf_model rollout (somme-exacte) / eqr "
       "(valeur apprise L7, directionnelle — pas un réseau profond, assumé). "
       "Les deux variantes sont atteignables par /api/postflop "
       "(leaf_model=) ; eqr entraîne fusion.eqr une fois puis mémoïse"),
    _P("Nested / subgame re-solving", _C.SOLVING, (GTOW,), _COV, "engine",
       "★ resolve_spot : re-solve du sous-jeu depuis la range INFÉRÉE (F3) — "
       "personne ne re-solve depuis une range apprise en direct"),
    _P("Discount weights (α,β,γ)", _C.SOLVING, (PIO, TEXAS), _COV, "solver.dcfr",
       "α=1.5 β=0 γ=2 + schedules"),
    _P("Solving postflop range-vs-range (river/turn)", _C.SOLVING, ALL_POSTFLOP,
       _COV, "solver.postflop",
       "L1 : river EXACTE (1326 combos, card removal O(n log n)), turn par "
       "chance node — validé sur la solution analytique de polarisation ; "
       "flop complet = Phase 2"),
    _P("Ressources (threads/GPU/RAM)", _C.SOLVING, ALL_POSTFLOP, _PAR,
       "solver.postflop", "vectorisation NumPy + BLAS multi-thread ; threads "
       "dédiés/GPU = Phase 2 (wgpu) — PARTIEL assumé, pas de fausse promesse"),

    # ── F. ICM / tournoi ─────────────────────────────────────────────────
    _P("Modèle ICM (Malmuth-Harville)", _C.ICM, (PIO, HRC, ICMIZER, MONKER, GTOW), _COV,
       "core.icm", "F14 : Harville exact mémoïsé ≤12 joueurs, MC au-delà"),
    _P("Prize pool / payouts", _C.ICM, (HRC, ICMIZER), _COV, "core.icm",
       "payouts arbitraires, $EV par joueur"),
    _P("Bubble factor", _C.ICM, (HRC, ICMIZER), _COV, "core.icm",
       "convention ICMIZER/HRC + prime de risque BF/(1+BF)−½"),
    _P("Future Game Simulation (FGS)", _C.ICM, (HRC, ICMIZER), _PAR,
       "core.icm", "L4 : FGS léger (érosion walk, ordre des blindes) — "
       "pas encore le FGS à équilibres push-fold de HRC"),
    _P("Bounties / PKO", _C.ICM, (ICMIZER, HRC), _COV,
       "core.icm", "L5 : ½ cash + ½ sur sa propre prime × P(1er) Harville — "
       "équité requise exacte par différences de $EV"),
    _P("Équilibres push/fold Nash (jam/call, ICM)", _C.ICM, (HRC, ICMIZER), _COV,
       "solver.pushfold", "fictitious play affaibli (Leslie & Collins 2006), "
       "matrice 169×169 en cache ; HU 10bb : jam 58.7 % (réf ~58)"),
    _P("$EV vs chipEV", _C.ICM, (HRC, ICMIZER, GTOW), _COV, "core.icm",
       "α_ICM = BF·b/(P+b+BF·b) ; analyse_icm_spot() rend le verdict"),

    # ── G. Node locking / exploitation ───────────────────────────────────
    _P("Node locking complet", _C.EXPLOIT, (PIO, GTOW, DEEPS), _COV,
       "solver.postflop", "P2 : lock_node + re-solve libre autour (nodelock 2.0)"),
    _P("Node locking partiel (combos)", _C.EXPLOIT, (PIO, GTOW), _COV,
       "solver.postflop", "combos= : sous-ensemble verrouillé, le reste re-solve"),
    _P("Override de range adverse", _C.EXPLOIT, (PIO, GTOW, MONKER), _COV, "particle",
       "le filtre particulaire EST un override probabiliste continu"),
    _P("Réponse maximalement exploitante", _C.EXPLOIT, (PIO,), _COV, "arbiter",
       "σ_BR dans F13"),
    _P("Exploit BORNÉ en exploitabilité", _C.EXPLOIT, (), _COV, "arbiter",
       "★ safe exploitation Ganzfried-Sandholm — personne ne le fait"),

    # ── H. Métriques de sortie ───────────────────────────────────────────
    _P("EV (par action/combo/range)", _C.OUTPUT, ALL_POSTFLOP, _COV, "bet_sizing"),
    _P("Equity", _C.OUTPUT, ALL_POSTFLOP, _COV, "core.equity"),
    _P("Equity realization (EQR)", _C.OUTPUT, (PIO, GTOW), _COV, "fusion.eqr",
       "L7 : ridge apprise de NOS solves river — branchée le 14 août 2026 : "
       "/api/postflop leaf_model='eqr' entraîne train_eqr() une fois "
       "(mémoïsé) et republie le R² mesuré (~0,6) avec sa limite : valeur "
       "directionnelle, la somme des EV dérive"),
    _P("Fréquences d'action", _C.OUTPUT, ALL_POSTFLOP, _COV, "arbiter"),
    _P("Range advantage", _C.OUTPUT, (PIO, GTOW), _COV, "range_model.Range.range_advantage"),
    _P("Nut advantage", _C.OUTPUT, (PIO, GTOW), _COV, "range_model.Range.nut_advantage"),
    _P("Exploitabilité de la solution", _C.OUTPUT, (PIO, TEXAS), _COV, "solver.dcfr"),
    _P("MDF / alpha / pot odds", _C.OUTPUT, (PIO, GTOW), _COV, "bluffcatch"),
    _P("Entropie de range (bits)", _C.OUTPUT, (), _COV, "range_model.Range.entropy_bits",
       "★ Shannon sur la range — inédit"),
    _P("Gain d'information par sizing", _C.OUTPUT, (), _COV, "bet_sizing.information_gain",
       "★ I(X;Y) par mise — inédit"),
    _P("Distance de Fisher au GTO", _C.OUTPUT, (), _COV, "geometry.fisher_rao_distance",
       "★ métrique invariante — inédit"),
    _P("Compression en règles (Info Bottleneck)", _C.OUTPUT, (), _COV,
       "bottleneck.compress_range", "★ 169 combos → poignée de règles — inédit"),

    # ── I. Adversaire — l'angle mort ─────────────────────────────────────
    _P("Profil / archétype (clustering)", _C.OPPONENT, (), _COV, "particle",
       "★ 6 archétypes, filtre particulaire"),
    _P("Stats HUD avec confiance", _C.OPPONENT, (), _COV, "dynamic_beta",
       "★ Beta-Binomial : θ̂ ± IC95, pas un point"),
    _P("Tendances par street/sizing", _C.OPPONENT, (), _COV, "dynamic_beta"),
    _P("Historique joueur-vs-joueur", _C.OPPONENT, (), _COV, "player_notes"),
    _P("État mental (tilt/fatigue)", _C.OPPONENT, (), _COV, "hmm",
       "★ HMM 3 états, détection avant les stats"),
    _P("Skill / ROI externe (SharkScope)", _C.OPPONENT, (), _COV, "skill_prior",
       "★ prior rétréci + propension à s'adapter"),
    _P("Non-stationnarité (dérive adverse)", _C.OPPONENT, (), _COV, "dynamic_beta",
       "★ facteur d'oubli δ — suit l'adaptation"),
    _P("Patterns exploitables (TDA)", _C.OPPONENT, (), _COV, "topology",
       "★ homologie persistante + test de permutation", wired=False),
    _P("Incertitude → n'exploiter que si sûr", _C.OPPONENT, (), _COV, "arbiter",
       "★ λ dérivé de σ_θ — le cœur du système"),
    _P("Tells temporels (latence de décision)", _C.OPPONENT, (), _COV,
       "fusion.timing",
       "L2 : log-normal + Welford + CUSUM + tells validés par IC de Wilson. "
       "Volontairement NON compté comme différenciateur absolu : Hand2Note "
       "stocke des moyennes de temps statiques (la surprise séquentielle en "
       "direct, elle, n'existe nulle part)", wired=False),
    _P("Tendances de population (privées)", _C.OPPONENT, (GTOW,), _COV,
       "data.population",
       "L6 : empirical Bayes sur SES PROPRES HH, par tranche d'enjeux — "
       "GTO Wizard vend l'agrégat mondial, Hand2Note l'analyse cloud ; "
       "ici : local, privé, plafonné par l'information réelle", wired=False),
    _P("Aversion aux pertes personnelle (λ)", _C.OPPONENT, (), _COV, "drill",
       "★ λ de Kahneman mesuré sur tes décisions"),
)


# ═══════════════════════════════════════════════════════════════════════════
# PROFILS DES SOLVEURS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SolverProfile:
    name: str
    price_eur: str
    model: str
    algorithm: str
    multiway: str
    node_lock: bool
    icm: bool
    opponent_modeling: bool
    scriptable: bool
    local: bool
    strengths: tuple[str, ...] = ()
    weaknesses: tuple[str, ...] = ()


SOLVERS: tuple[SolverProfile, ...] = (
    SolverProfile(
        "PioSOLVER Edge", "≈800 € (licence perpétuelle)", "local", "CFR+",
        "Heads-up seulement", True, True, False, True, True,
        ("Référence de vitesse locale", "Node-locking le plus fin", "UPI scriptable",
         "Préflop (Edge)"),
        ("Windows only", "64 Go RAM", "Aucune modélisation adverse", "Interface ardue")),
    SolverProfile(
        "GTO Wizard AI", "≈139 €/mois (Elite, par format)", "cloud", "CFR + réseaux de valeur",
        "Préflop 9-max, postflop 3-way", True, True, False, False, False,
        ("Solve ~6 s (vs 81 min Pio)", "Node-locking 2.0", "53 000 sims ICM", "Écosystème complet"),
        ("Abonnement par format", "Multiway postflop plafonné 3-way", "Aucune modélisation adverse",
         "Requêtes traçables")),
    SolverProfile(
        "GTO+", "75 € (perpétuel)", "local", "CFR (Nash exact)",
        "HU + 3-way", False, False, False, False, True,
        ("Rapport qualité/prix imbattable", "Nash exact", "Fichiers compacts"),
        ("Windows only", "Pas de node-lock", "Pas d'ICM", "Aucune modélisation adverse")),
    SolverProfile(
        "MonkerSolver", "≈499 € (perpétuel)", "local", "CFR + abstraction",
        "★ Multiway profond 4+ / PLO", False, True, False, False, True,
        ("Seul vrai multiway profond", "PLO", "Arbre libre"),
        ("64 Go+ RAM", "Peu maintenu", "Interface austère", "Aucune modélisation adverse")),
    SolverProfile(
        "TexasSolver", "0 € (AGPL)", "local", "DCFR",
        "Heads-up seulement", False, False, False, True, True,
        ("Gratuit, open-source", "Bet-tree très granulaire", "Multi-OS"),
        ("Pas de préflop/multiway", "JSON lourd", "Aucun support", "Aucune modélisation adverse")),
    SolverProfile(
        "Poker Fusion Solver", "0 € (privé)", "local",
        "CFR range-vs-range (river exact, turn, profondeur limitée EQR) + 14 fusions",
        "Équité multiway N ranges (solving HU)", True, True, True, True, True,
        ("★ RE-SOLVE depuis la range inférée en direct (personne ne le fait)",
         "★ Node-locking complet+partiel avec re-solve (nodelock 2.0)",
         "★ Push/fold Nash ICM (particularité HRC) + PKO + FGS léger",
         "★ Arbre complet : tailles OOP/IP/raise, all-in threshold, merge",
         "★ Bunching + abstraction EHS isomorphe + tells temporels",
         "★ Modélisation adverse individuelle, exploit borné",
         "Zéro réseau, zéro coût"),
        ("Flop complet en un solve = Phase 2 (classes isomorphes prêtes)",
         "PLO/Short Deck hors périmètre (choix)", "GPU = Phase 2",
         # Audit du 14 août 2026 : la force et la faiblesse ensemble, sinon
         # le tableau ment — voir library_only(). (EQR retiré de la liste le
         # 14 août 2026 : branché via /api/postflop leaf_model='eqr'.)
         "Bunching, abstraction EHS, isomorphisme, tells temporels, "
         "TDA, population : bibliothèques testées NON branchées dans l'app",
         # v4.4 lit une vraie table (calibration live, lecture seule) et
         # v4.5 lit les montants d'une capture collée : la faiblesse
         # restante est le rappel de la détection, pas son absence.
         "Perception : rappel 76,7 % sur captures réelles (7-max en défaut)")),
)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSE
# ═══════════════════════════════════════════════════════════════════════════


def coverage_report() -> dict[str, dict[str, int]]:
    """Compte les paramètres par catégorie et par état de couverture."""
    out: dict[str, dict[str, int]] = {}
    for cat in Category:
        row = {c.value: 0 for c in Coverage}
        row["total"] = 0
        for p in PARAMETERS:
            if p.category is cat:
                row[p.coverage.value] += 1
                row["total"] += 1
        out[cat.value] = row
    return out


def coverage_score() -> dict[str, float]:
    """Score global : couvert=1, partiel/planifié=0.5, absent=0."""
    weight = {Coverage.COVERED: 1.0, Coverage.PARTIAL: 0.5,
              Coverage.PLANNED: 0.5, Coverage.ABSENT: 0.0}
    total = len(PARAMETERS)
    got = sum(weight[p.coverage] for p in PARAMETERS)
    industry = [p for p in PARAMETERS if p.is_industry_standard]
    got_ind = sum(weight[p.coverage] for p in industry)
    return {
        "parameters": total,
        "score": got / total,
        "industry_parameters": len(industry),
        "industry_score": got_ind / len(industry) if industry else 0.0,
    }


def gaps() -> list[Parameter]:
    """Paramètres absents qu'au moins un solveur de référence expose — le travail restant."""
    return sorted(
        (p for p in PARAMETERS if p.coverage is Coverage.ABSENT and p.exposed_by),
        key=lambda p: (-len(p.exposed_by), p.category.value),
    )


def library_only() -> list[Parameter]:
    """Paramètres implémentés et testés mais que RIEN n'exécute pour le joueur.

    L'écart honnête entre « le code existe » (``coverage``) et « le joueur en
    profite » (``wired``) — recensé par l'audit du 14 août 2026. Chaque module
    pointé ici porte une section « Statut » en tête de docstring qui dit
    pourquoi il n'est pas branché et où l'accrocher si on le branche.
    """
    return [p for p in PARAMETERS
            if p.coverage in (Coverage.COVERED, Coverage.PARTIAL)
            and not p.wired]


def differentiators() -> list[Parameter]:
    """Ce que NOUS faisons et qu'AUCUN solveur de référence ne fait.

    Repérés par : couverts chez nous, exposés par aucun solveur listé, et
    marqués ★ dans la note.
    """
    return [p for p in PARAMETERS
            if p.coverage is Coverage.COVERED and not p.exposed_by and "★" in p.note]
