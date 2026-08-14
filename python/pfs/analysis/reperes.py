r"""Repères MESURÉS sur corpus public, en remplacement des fourchettes de manuel.

Pourquoi ce module existe
-------------------------
La revue de session affichait « zone saine 28–50 % » pour le VPIP. Ce chiffre
vient de la littérature de poker, pas d'une mesure : personne dans le projet
ne pouvait dire sur quelles mains il avait été calculé, ni avec quelle
définition du dénominateur. Ici, chaque borne est **recalculée à la commande**
depuis un corpus public sous licence MIT (``phh-dataset``, université de
Toronto), avec **exactement le même code de comptage** que celui qui mesure
l'utilisateur — :meth:`~pfs.data.hand_history.ParsedHand.stat_observations`.

C'est ce partage du code de comptage qui rend la comparaison licite. Une
fréquence de poker n'a pas de sens sans son dénominateur, et les conventions
diffèrent d'un tracker à l'autre :

* **VPIP / PFR** — occasion à chaque main où le joueur est assis.
* **3-bet** — occasion à chaque main où **au moins une relance préflop** a eu
  lieu, pour **tous** les joueurs assis, y compris ceux déjà couchés. Ce
  n'est PAS le « 3-bet vs open » des trackers (qui conditionne au fait de
  faire face à une relance) : le dénominateur d'ici est plus large, donc le
  taux plus bas. Repère et mesure utilisateur partagent la convention, ce qui
  est la seule chose qui compte pour les comparer.
* **Fold to c-bet** — occasion quand le joueur agit après une mise adverse au
  flop.
* **WTSD** — occasion à chaque main **où un flop est tombé**, pour tous les
  joueurs assis, y compris ceux couchés préflop. Le WTSD classique conditionne
  au fait d'avoir vu le flop ; ici non. Conséquence : **ce taux dépend
  mécaniquement du nombre de sièges** — à six, quatre joueurs se couchent
  préflop et pèsent au dénominateur ; à trois, presque personne. Comparer un
  WTSD de table courte à un repère 6-max est donc invalide, et le module le
  dit dans :attr:`JeuDeReperes.limites` plutôt que de laisser croire à un
  écart de comportement.
* **AF postflop** — (mises + relances + tapis agressifs) / suivis, sur flop,
  turn et river. Ratio, pas pourcentage.

Ce que le corpus permet, et ce qu'il ne permet pas
--------------------------------------------------
**Pluribus** (Brown & Sandholm, *Science*, 2019) : 10 000 mains, cash game
6-max, tapis remis à 100 bb à chaque main, **sans ante**, sans ICM. Les
valeurs qu'on en tire sont un repère d'**équilibre** — ce que joue une
stratégie proche de l'optimal — et non une moyenne de population humaine. Un
écart à ce repère ne dit pas « tu joues autrement que les autres », il dit
« tu joues autrement qu'une stratégie d'équilibre dans CE format ».

**WSOP 2023, event #43** (Poker Players Championship à 50 000 $) : 83 mains,
dont **18 seulement sont du hold'em** — c'est un championnat *mixte*, les 65
autres sont de l'Omaha hi-lo, du stud, du razz, du triple draw, et le lecteur
PHH les refuse au lieu de les lire de travers. Ces 18 mains sont un tournoi
avec antes et tapis inégaux, donc bien plus proches du format de
l'utilisateur — mais 18 mains × 5 joueurs = 90 mains-joueurs. À cet effectif,
l'intervalle de Wilson d'un VPIP couvre une vingtaine de points : le jeu est
publié avec ses effectifs, et marqué :attr:`JeuDeReperes.concluant` à False.
Aucune conclusion n'en est tirée, et la revue de session ne s'en sert pas.

La comparaison la plus honnête disponible
------------------------------------------
L'utilisateur joue des tables courtes. Agréger un VPIP sur six positions puis
le comparer au sien, qui n'agrège que trois positions tardives (BTN, SB, BB),
compare des mélanges de positions différents — l'UTG et le MP tirent
mécaniquement le repère 6-max vers le bas. Le jeu ``pluribus_tardives``
restreint donc le même calcul aux trois positions d'une table à trois, ce qui
retire ce biais-là. Il en reste un, énoncé : à trois, le bouton affronte deux
adversaires et non cinq, donc l'équilibre y est **plus large encore** que ce
que Pluribus joue au bouton d'une table à six. Le repère tardives est de ce
fait une **borne basse** de ce que vaudrait un équilibre à trois — ce qui suffit
pour lire un dépassement, jamais pour valider un chiffre « dans la zone ».

Rejouabilité
------------
``python banc_reperes_corpus.py --verifier`` recalcule tout depuis les fichiers
``.phh`` et compare aux constantes gelées ci-dessous, chiffre par chiffre.
Aucun aléa : lecture ordonnée par :func:`~pfs.data.phh.iter_phh_files`,
comptage entier, taux dérivés au moment de l'affichage.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pfs.data.hand_history import ActionType, Street
from pfs.data.phh import (
    PhhError,
    iter_phh_files,
    parse_phh_file,
    positions_par_siege,
)

__all__ = [
    "CORPUS_PLURIBUS",
    "CORPUS_WSOP_2023",
    "POSITIONS_TABLE_COURTE",
    "REPERES",
    "Ecart",
    "JeuDeReperes",
    "Repere",
    "ReperesError",
    "calculer_jeu",
    "compter_agression",
    "jeu_par_defaut",
    "racine_corpus",
    "situer",
    "wilson",
]


class ReperesError(ValueError):
    """Corpus introuvable, ou repère demandé qui n'existe pas."""


def racine_corpus() -> Path:
    """Racine du dossier ``corpus`` — celui qui contient ``phh-dataset``.

    Résolution, dans l'ordre :

    1. la variable d'environnement ``PFS_CORPUS``, pour une machine où le
       corpus vit ailleurs ;
    2. ``<parent du dépôt>/corpus`` — le corpus pèse 500 Mo et vit
       volontairement HORS du dépôt, en frère de celui-ci.

    L'existence du dossier n'est PAS vérifiée ici : c'est
    :func:`calculer_jeu` qui lève :class:`ReperesError` avec le chemin fautif
    dans le message, et les consommateurs sans corpus (la table gelée
    :data:`REPERES`, la route ``/api/reperes``) ne touchent jamais le disque.
    Vérifier ici casserait leur import sur toute machine sans corpus.
    """
    env = os.environ.get("PFS_CORPUS", "").strip()
    if env:
        return Path(env)
    # reperes.py = <dépôt>/python/pfs/analysis/reperes.py → parents[3] = dépôt.
    return Path(__file__).resolve().parents[3].parent / "corpus"


CORPUS_PLURIBUS = racine_corpus() / "phh-dataset" / "data" / "pluribus"
CORPUS_WSOP_2023 = racine_corpus() / "phh-dataset" / "data" / "wsop" / "2023"

POSITIONS_TABLE_COURTE = ("BTN", "SB", "BB")
"""Les trois positions d'une table à trois joueurs.

Servent à restreindre un corpus 6-max au mélange de positions d'une table
courte — voir la section « comparaison la plus honnête » du module.
"""

#: Statistiques de proportion : un succès sur une occasion, donc un intervalle
#: de Wilson a un sens. ``ecart_vpip_pfr`` et ``af_postflop`` n'en sont pas et
#: sont traités à part.
STATS_PROPORTION = ("vpip", "pfr", "three_bet", "fold_to_cbet", "wtsd")

#: Occasions minimales pour qu'un joueur entre dans la dispersion inter-joueurs.
#: En dessous, son taux est du bruit et déplacerait les quartiles sans rien
#: apprendre.
N_MIN_JOUEUR = 100

#: Occasions en dessous desquelles un jeu de repères est publié mais marqué
#: non concluant : à 90 mains-joueurs, l'IC d'un VPIP fait 19 points de large.
N_MIN_CONCLUANT = 1000


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Intervalle de Wilson à 95 % pour ``k`` succès sur ``n`` occasions.

    Choisi plutôt que l'intervalle normal parce qu'il reste dans [0, 1] et
    reste correct près des bords — ce qui arrive ici (3-bet à 4 %).

    >>> lo, hi = wilson(50, 100)
    >>> round(lo, 4), round(hi, 4)
    (0.4038, 0.5962)
    >>> wilson(0, 0)
    (0.0, 1.0)
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    demi = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return ((centre - demi) / d, (centre + demi) / d)


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    """(Q1, médiane, Q3) par rang, sans interpolation.

    Sans interpolation parce qu'un quartile interpolé sur 14 joueurs suggère
    une précision que 14 points n'ont pas.

    >>> _quartiles([1.0, 2.0, 3.0, 4.0])
    (2.0, 3.0, 4.0)
    """
    q = sorted(xs)
    n = len(q)
    return (q[n // 4], q[n // 2], q[min(n - 1, 3 * n // 4)])


@dataclass(frozen=True, slots=True)
class Repere:
    """Une statistique mesurée sur un corpus, avec de quoi la relativiser.

    Attributes
    ----------
    stat : str
        Clé de la statistique (« vpip », « af_postflop »…).
    valeur : float
        Taux en pourcent, sauf ``af_postflop`` qui est un ratio.
    occasions : int
        Dénominateur réel. Une valeur sans son dénominateur n'est pas une
        mesure.
    succes : int
        Numérateur, pour que le chiffre soit recalculable à la main. Pour
        ``af_postflop`` c'est le nombre d'actions agressives et ``occasions``
        le nombre de suivis : le numérateur y est donc plus GRAND que le
        dénominateur, l'AF étant un ratio et non une proportion.
    ic_bas, ic_haut : float
        Bornes de Wilson à 95 %, en pourcent. Valent ``valeur`` quand la
        statistique n'est pas une proportion (l'écart, l'AF) : l'incertitude
        y existe, mais Wilson ne la décrit pas, et inventer un intervalle
        serait pire que de ne pas en donner.
    q1, mediane, q3 : float
        Dispersion **entre joueurs** du corpus (pas entre mains). C'est elle
        qui dit si le repère est un point ou un nuage.
    n_joueurs : int
        Joueurs retenus dans cette dispersion (≥ :data:`N_MIN_JOUEUR`).
    """

    stat: str
    valeur: float
    occasions: int
    succes: int
    ic_bas: float
    ic_haut: float
    q1: float
    mediane: float
    q3: float
    n_joueurs: int

    @property
    def unite(self) -> str:
        """« % », « pt » (une différence de deux taux) ou « » (un ratio).

        Distinguer les trois n'est pas cosmétique : un écart VPIP−PFR de
        16,5 est en POINTS de pourcentage, pas en pourcent, et l'AF n'a
        aucune unité. Les afficher tous en « % » ferait dire au tableau des
        choses fausses.
        """
        if self.stat == "af_postflop":
            return ""
        return "pt" if self.stat == "ecart_vpip_pfr" else "%"


@dataclass(frozen=True, slots=True)
class JeuDeReperes:
    """Un corpus, un mélange de positions, et les repères qui en sortent."""

    cle: str
    source: str
    n_mains: int
    n_mains_joueurs: int
    positions: tuple[str, ...]
    reperes: dict[str, Repere]
    limites: tuple[str, ...]

    @property
    def concluant(self) -> bool:
        """Faux quand l'effectif interdit de conclure — jamais masqué."""
        return self.n_mains_joueurs >= N_MIN_CONCLUANT

    def __getitem__(self, stat: str) -> Repere:
        try:
            return self.reperes[stat]
        except KeyError as exc:
            raise ReperesError(
                f"pas de repère « {stat} » dans « {self.cle} » ; "
                f"disponibles : {sorted(self.reperes)}.") from exc


@dataclass(frozen=True, slots=True)
class Ecart:
    """Position d'une valeur mesurée chez l'utilisateur face à un repère."""

    stat: str
    valeur: float
    repere: float
    occasions_utilisateur: int
    ecart: float           # valeur − repère, en points (ou en ratio pour l'AF)
    unite: str
    comparable: bool       # False = la définition interdit la comparaison
    pourquoi: str          # raison de l'incomparabilité, sinon ""

    @property
    def ecart_relatif(self) -> float:
        """Écart rapporté au repère : 2.0 = « deux fois le repère »."""
        return self.valeur / self.repere if self.repere else math.inf


# ═══════════════════════════════════════════════════════════════════════════
# CALCUL DEPUIS LE CORPUS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _Compteurs:
    """Accumulateurs bruts d'un passage sur le corpus."""

    mains: int = 0
    refusees: int = 0
    mains_joueurs: int = 0
    total: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0]))
    par_joueur: dict[str, dict[str, list[int]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    # joueur → [agressives, suivis] postflop
    agression: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0]))


def compter_agression(hand, joueur: str) -> tuple[int, int]:
    """(actions agressives, suivis) du joueur **après le préflop**.

    L'AF classique des trackers. Le préflop est exclu parce qu'une relance
    d'ouverture y est déjà comptée par le PFR : l'inclure ferait dire deux
    fois la même chose à deux statistiques censées être indépendantes.
    """
    agressives = suivis = 0
    for a in hand.actions:
        if a.player != joueur or a.street is Street.PREFLOP:
            continue
        if a.action in (ActionType.BET, ActionType.RAISE, ActionType.ALLIN):
            agressives += 1
        elif a.action is ActionType.CALL:
            suivis += 1
    return agressives, suivis


def calculer_jeu(racine: Path | str, cle: str, source: str,
                 limites: tuple[str, ...],
                 positions: tuple[str, ...] | None = None,
                 is_tournament: bool = False,
                 mains_max: int = 0) -> JeuDeReperes:
    """Recalcule un jeu de repères depuis un dossier de fichiers ``.phh``.

    Parameters
    ----------
    racine : Path or str
        Dossier parcouru récursivement.
    cle, source, limites
        Étiquettes portées par le :class:`JeuDeReperes` produit.
    positions : tuple of str, optional
        Restreint le comptage à ces positions (``POSITIONS_TABLE_COURTE``
        pour un mélange de table à trois). ``None`` : toute la table.
    is_tournament : bool
        Passé au lecteur PHH, qui ne le déduit pas du fichier.
    mains_max : int
        0 = tout le corpus. Sert aux tests, qui n'ont pas à lire 10 000
        fichiers pour vérifier une mécanique.

    Raises
    ------
    ReperesError
        Dossier absent, ou aucune main lisible dedans.
    """
    dossier = Path(racine)
    if not dossier.is_dir():
        raise ReperesError(f"corpus introuvable : {dossier}")

    c = _Compteurs()
    for fichier in iter_phh_files(dossier):
        if mains_max and c.mains >= mains_max:
            break
        try:
            main = parse_phh_file(fichier, is_tournament=is_tournament)
        except (PhhError, OSError, UnicodeDecodeError):
            # Les variantes non hold'em du corpus WSOP tombent ici : refusées
            # par le lecteur, comptées, jamais lues de travers.
            c.refusees += 1
            continue
        c.mains += 1
        pos_par_siege = positions_par_siege(main)
        for siege in main.seats:
            pos = pos_par_siege.get(siege.seat, "")
            if positions is not None and pos not in positions:
                continue
            c.mains_joueurs += 1
            for stat, valeur in main.stat_observations(siege.player).items():
                c.total[stat][0] += int(valeur)
                c.total[stat][1] += 1
                c.par_joueur[siege.player][stat][0] += int(valeur)
                c.par_joueur[siege.player][stat][1] += 1
            agr, sui = compter_agression(main, siege.player)
            c.agression[siege.player][0] += agr
            c.agression[siege.player][1] += sui

    if not c.mains:
        raise ReperesError(f"aucune main hold'em lisible sous {dossier}.")

    return _assembler(c, cle, source, positions or (), limites)


def _repere_proportion(stat: str, k: int, n: int,
                       taux_joueurs: list[float]) -> Repere:
    lo, hi = wilson(k, n)
    q1, med, q3 = (_quartiles(taux_joueurs) if taux_joueurs
                   else (100.0 * k / n,) * 3 if n else (0.0, 0.0, 0.0))
    return Repere(stat=stat, valeur=100.0 * k / n if n else 0.0,
                  occasions=n, succes=k,
                  ic_bas=100.0 * lo, ic_haut=100.0 * hi,
                  q1=q1, mediane=med, q3=q3, n_joueurs=len(taux_joueurs))


def _assembler(c: _Compteurs, cle: str, source: str,
               positions: tuple[str, ...],
               limites: tuple[str, ...]) -> JeuDeReperes:
    """Transforme les compteurs bruts en repères publiables."""
    reperes: dict[str, Repere] = {}

    for stat in STATS_PROPORTION:
        k, n = c.total.get(stat, [0, 0])
        if not n:
            continue
        taux = [100.0 * v[0] / v[1]
                for st in c.par_joueur.values()
                for v in (st.get(stat, [0, 0]),)
                if v[1] >= N_MIN_JOUEUR]
        reperes[stat] = _repere_proportion(stat, k, n, taux)

    # Écart VPIP − PFR : une différence de deux proportions sur le MÊME
    # dénominateur. Pas de Wilson là-dessus — les bornes valent la valeur, et
    # la dispersion inter-joueurs prend le relais pour dire la variabilité.
    kv, nv = c.total.get("vpip", [0, 0])
    kp, np_ = c.total.get("pfr", [0, 0])
    if nv and np_:
        val = 100.0 * (kv / nv - kp / np_)
        ecarts = [100.0 * (st["vpip"][0] / st["vpip"][1]
                           - st["pfr"][0] / st["pfr"][1])
                  for st in c.par_joueur.values()
                  if st.get("vpip", [0, 0])[1] >= N_MIN_JOUEUR
                  and st.get("pfr", [0, 0])[1] >= N_MIN_JOUEUR]
        q1, med, q3 = _quartiles(ecarts) if ecarts else (val, val, val)
        reperes["ecart_vpip_pfr"] = Repere(
            stat="ecart_vpip_pfr", valeur=val, occasions=nv, succes=kv - kp,
            ic_bas=val, ic_haut=val, q1=q1, mediane=med, q3=q3,
            n_joueurs=len(ecarts))

    # AF postflop : un ratio, dénominateur = les suivis.
    agr = sum(v[0] for v in c.agression.values())
    sui = sum(v[1] for v in c.agression.values())
    if sui:
        ratios = [v[0] / v[1] for v in c.agression.values() if v[1] >= 20]
        q1, med, q3 = (_quartiles(ratios) if ratios
                       else (agr / sui, agr / sui, agr / sui))
        reperes["af_postflop"] = Repere(
            stat="af_postflop", valeur=agr / sui, occasions=sui, succes=agr,
            ic_bas=agr / sui, ic_haut=agr / sui,
            q1=q1, mediane=med, q3=q3, n_joueurs=len(ratios))

    return JeuDeReperes(cle=cle, source=source, n_mains=c.mains,
                        n_mains_joueurs=c.mains_joueurs,
                        positions=positions, reperes=reperes, limites=limites)


# ═══════════════════════════════════════════════════════════════════════════
# LA TABLE GELÉE
# ═══════════════════════════════════════════════════════════════════════════
#
# Produite par ``python banc_reperes_corpus.py --geler`` et vérifiée chiffre
# par chiffre par ``--verifier``. Elle est gelée parce que le corpus vit HORS
# du dépôt (500 Mo de fichiers .phh sous ``corpus/``) : sans elle, le logiciel
# n'aurait aucun repère sur une machine où le corpus n'est pas installé, et il
# retomberait sur les fourchettes de manuel que ce module remplace.

REPERES: dict[str, JeuDeReperes] = {
    "pluribus_6max": JeuDeReperes(
        cle="pluribus_6max",
        source="Pluribus (Brown & Sandholm, Science 2019) — 10 000 mains, "
               "cash 6-max, 100 bb, sans ante, 13 professionnels + le bot",
        n_mains=10000, n_mains_joueurs=60000,
        positions=(),
        reperes={
            "vpip": Repere("vpip", 26.3167, 60000, 15790, 25.9658, 26.6705,
                           25.1096, 26.3736, 27.2414, 14),
            "pfr": Repere("pfr", 17.5517, 60000, 10531, 17.2494, 17.8581,
                          16.8750, 17.6225, 18.0949, 14),
            "three_bet": Repere("three_bet", 4.3731, 51222, 2240, 4.1994,
                                4.5537, 4.0555, 4.6414, 5.2093, 14),
            "fold_to_cbet": Repere("fold_to_cbet", 46.8651, 3461, 1622,
                                   45.2070, 48.5301, 43.3803, 47.1976,
                                   49.5575, 9),
            "wtsd": Repere("wtsd", 10.6220, 32028, 3402, 10.2892, 10.9641,
                           9.6252, 10.7906, 11.5385, 14),
            "ecart_vpip_pfr": Repere("ecart_vpip_pfr", 8.7650, 60000, 5259,
                                     8.7650, 8.7650, 7.0147, 8.8852,
                                     9.6189, 14),
            "af_postflop": Repere("af_postflop", 2.2669, 3364, 7626, 2.2669,
                                  2.2669, 1.9323, 2.1250, 2.8598, 14),
        },
        limites=(
            "Cash game, pas tournoi : ni antes, ni ICM, ni tapis court.",
            "Tapis remis à 100 bb à chaque main : aucun spot de push/fold.",
            "Six sièges : l'agrégat mélange UTG et MP, que n'a pas une table "
            "courte. Pour une table à trois, lire « pluribus_tardives ».",
            "Repère d'équilibre, pas moyenne de population : 13 joueurs "
            "invités de haut niveau et un bot surhumain.",
        ),
    ),
    "pluribus_tardives": JeuDeReperes(
        cle="pluribus_tardives",
        source="Pluribus, mêmes 10 000 mains, restreint aux positions BTN, "
               "SB et BB — le mélange de positions d'une table à trois",
        n_mains=10000, n_mains_joueurs=30000,
        positions=POSITIONS_TABLE_COURTE,
        reperes={
            "vpip": Repere("vpip", 32.1900, 30000, 9657, 31.6636, 32.7209,
                           29.0598, 32.5118, 34.1026, 14),
            "pfr": Repere("pfr", 15.6700, 30000, 4701, 15.2630, 16.0857,
                          15.1515, 16.1206, 16.7147, 14),
            "three_bet": Repere("three_bet", 6.7276, 25611, 1723, 6.4272,
                                7.0409, 6.0976, 6.9532, 7.6220, 14),
            "fold_to_cbet": Repere("fold_to_cbet", 47.2547, 2823, 1334,
                                   45.4180, 49.0988, 41.3919, 47.4157,
                                   52.5122, 8),
            "wtsd": Repere("wtsd", 14.6934, 16014, 2353, 14.1535, 15.2502,
                           13.5089, 14.8162, 15.6010, 14),
            "ecart_vpip_pfr": Repere("ecart_vpip_pfr", 16.5200, 30000, 4956,
                                     16.5200, 16.5200, 12.5199, 16.6889,
                                     17.9413, 14),
            "af_postflop": Repere("af_postflop", 1.9529, 2443, 4771, 1.9529,
                                  1.9529, 1.7778, 1.9214, 2.2055, 13),
        },
        limites=(
            "Cash game, pas tournoi : ni antes, ni ICM, ni tapis court.",
            "BORNE BASSE, pas cible : au bouton d'une table à SIX, Pluribus "
            "affronte cinq adversaires ; à trois il n'en affronterait que "
            "deux et ouvrirait plus large. Un dépassement de ce repère est "
            "donc réel ; y être « dedans » ne prouve rien.",
        ),
    ),
    "wsop2023_holdem": JeuDeReperes(
        cle="wsop2023_holdem",
        source="WSOP 2023 event #43 (Poker Players Championship, 50 000 $), "
               "les 18 mains de hold'em des 83 publiées — 5 joueurs, "
               "tournoi avec antes",
        n_mains=18, n_mains_joueurs=90,
        positions=(),
        reperes={
            "vpip": Repere("vpip", 32.2222, 90, 29, 23.4668, 42.4332,
                           32.2222, 32.2222, 32.2222, 0),
            "pfr": Repere("pfr", 24.4444, 90, 22, 16.7328, 34.2484,
                          24.4444, 24.4444, 24.4444, 0),
            "three_bet": Repere("three_bet", 9.3333, 75, 7, 4.5949, 18.0347,
                                9.3333, 9.3333, 9.3333, 0),
            "fold_to_cbet": Repere("fold_to_cbet", 12.5000, 8, 1, 2.2417,
                                   47.0888, 12.5000, 12.5000, 12.5000, 0),
            "wtsd": Repere("wtsd", 17.7778, 45, 8, 9.2944, 31.3298,
                           17.7778, 17.7778, 17.7778, 0),
            "ecart_vpip_pfr": Repere("ecart_vpip_pfr", 7.7778, 90, 7, 7.7778,
                                     7.7778, 7.7778, 7.7778, 7.7778, 0),
            "af_postflop": Repere("af_postflop", 1.6667, 12, 20, 1.6667,
                                  1.6667, 1.6667, 1.6667, 1.6667, 0),
        },
        limites=(
            "NON CONCLUANT : 18 mains, 90 mains-joueurs. L'IC de Wilson du "
            "VPIP couvre 23,5–42,4 % : ce jeu ne peut ni confirmer ni "
            "infirmer quoi que ce soit, il est publié pour dire ce que le "
            "corpus contient réellement.",
            "65 des 83 mains sont d'une autre variante (Omaha hi-lo, stud, "
            "razz, triple draw) : le championnat est mixte et le lecteur "
            "PHH les refuse au lieu de les lire de travers.",
            "Cinq joueurs seulement, chacun sur 18 mains : aucune dispersion "
            "inter-joueurs n'est publiée (n_joueurs = 0), le seuil étant de "
            f"{N_MIN_JOUEUR} occasions.",
            "Une table finale à 50 000 $ n'est pas un tournoi en ligne : "
            "l'ICM y est extrême et les tapis très inégaux.",
        ),
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# SITUER UN JOUEUR
# ═══════════════════════════════════════════════════════════════════════════

#: Statistiques dont le dénominateur dépend du nombre de sièges, et qui ne
#: peuvent donc pas être comparées d'un format à l'autre. Voir la définition
#: du WTSD en tête de module : à six, quatre joueurs couchés préflop pèsent au
#: dénominateur ; à trois, presque aucun.
_DEPEND_DU_NOMBRE_DE_SIEGES = {
    "wtsd": "le dénominateur compte tous les joueurs assis quand un flop "
            "tombe, couchés préflop compris : il baisse mécaniquement quand "
            "la table s'agrandit",
}


def situer(mesures: dict[str, float], jeu: JeuDeReperes,
           occasions: dict[str, int] | None = None) -> list[Ecart]:
    """Place les mesures d'un joueur face à un jeu de repères.

    Parameters
    ----------
    mesures : dict
        ``{"vpip": 63.1, "pfr": 31.3, ...}`` en pourcent (ratio pour l'AF).
        Les clés inconnues du jeu sont ignorées sans bruit : un profil peut
        porter des statistiques qu'aucun corpus ne mesure.
    jeu : JeuDeReperes
        Le repère de comparaison.
    occasions : dict, optional
        Dénominateurs de l'utilisateur, reportés tels quels.

    Returns
    -------
    list of Ecart
        Trié par écart absolu décroissant, comparables d'abord : le premier
        élément est la plus grande divergence réellement interprétable. Les
        statistiques de :data:`_DEPEND_DU_NOMBRE_DE_SIEGES` sortent avec
        ``comparable=False`` et leur raison — leur écart est calculé quand
        même, mais l'appelant sait qu'il n'a pas le droit de le lire comme un
        écart de comportement.
    """
    occ = occasions or {}
    sorties: list[Ecart] = []
    for stat, valeur in mesures.items():
        if stat not in jeu.reperes:
            continue
        rep = jeu[stat]
        raison = _DEPEND_DU_NOMBRE_DE_SIEGES.get(stat, "")
        sorties.append(Ecart(
            stat=stat, valeur=float(valeur), repere=rep.valeur,
            occasions_utilisateur=int(occ.get(stat, 0)),
            ecart=float(valeur) - rep.valeur, unite=rep.unite,
            comparable=not raison, pourquoi=raison))
    sorties.sort(key=lambda e: (e.comparable, abs(e.ecart)), reverse=True)
    return sorties


def jeu_par_defaut(n_sieges: int) -> str:
    """Clé du jeu de repères adapté à une table de ``n_sieges`` joueurs.

    Une table à trois ou moins se compare au mélange de positions d'une table
    à trois (``pluribus_tardives``), pas à un agrégat 6-max qui inclut l'UTG
    et le MP — voir la section « comparaison la plus honnête » du module.

    >>> jeu_par_defaut(3), jeu_par_defaut(6)
    ('pluribus_tardives', 'pluribus_6max')
    """
    return "pluribus_tardives" if n_sieges <= 3 else "pluribus_6max"
