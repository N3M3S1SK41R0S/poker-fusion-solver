"""Drills ciblés — les fuites mesurées de tes mains, rejouées à l'entraînement.

Ce module ferme la boucle entre les deux moitiés du projet : l'analyse
(``pfs.analysis``) trouve les décisions ratées et les chiffre en big blinds,
l'entraînement (``pfs.train.drill``) les fait revenir jusqu'à ce qu'elles
soient acquises. Ici on transforme les unes en les autres.

Ce qu'un drill contient
-----------------------
Le spot tel qu'il a été joué (cartes, pot, mise, tapis), les options, et **la
bonne réponse tranchée par le solveur** — jamais une opinion : c'est
l'équilibre de Nash jam/fold de ``pfs.solver.pushfold``, résolu deux fois (la
revue avec l'ante de la main, le conseiller sans) et retenu seulement si les
deux tranchent dans le même sens.

Périmètre couvert (honnêteté NEMESIS)
-------------------------------------
Un drill n'est fabriqué QUE si la bonne réponse est certaine. En pratique cela
restreint la génération au seul régime où le conseiller renvoie
``confidence == "certain"`` : le **préflop heads-up en tapis court**, où la
théorie des jeux tranche exactement. Le domaine effectivement atteignable est
``bb + ante < tapis effectif ≤ 25 bb`` : en deçà, ``solve_hu_pushfold`` refuse
un jeu dégénéré (mesuré : à 1,0 bb effectif pile, sans ante, c'est la revue
elle-même qui lève ``PushFoldError`` avant d'atteindre ce module).

Sont écartés, comptés à part et jamais approximés :

- les spots encore multiway quand le héros parle (le solveur est heads-up) ;
- les tapis > 25 bb (l'arbre limp/raise-fold compte : hors modèle) ;
- les relances non-jam (hors du modèle push/fold) ;
- les limps que le jam n'aurait pas améliorés (leur coût vaudrait zéro par
  construction : rien à enseigner) ;
- les spots où le solve **avec ante** (revue) et le solve **sans ante**
  (conseiller) ne désignent pas la même action : le verdict n'est alors pas
  robuste à la structure exacte, on ne l'enseigne pas.

Un drill dont la « bonne réponse » serait fausse apprendrait une erreur : le
postflop (range adverse supposée) et le préflop profond (charts déclarées
approximatives) restent donc volontairement hors du générateur.

Ce qui est mesuré, et ce qui ne l'est pas
-----------------------------------------
Le coût d'un **jam** ou d'un **fold** erroné est une mesure : les deux actions
existent dans le modèle, ``EV(jam) − EV(fold)`` sort du solveur, et l'écart est
exactement l'EV sacrifiée.

Le coût d'un **limp** n'en est pas une. Le modèle push/fold ne définit pas
``EV(limp)`` ; ``pushfold_review._judge`` le facture ``max(EV(jam) − EV(fold),
0)``, c'est-à-dire qu'il valorise le limp au prix du fold. Ce chiffre est une
convention, exacte si et seulement si limper vaut passer.
``SpotDrill.cout_mesure`` le déclare, ``SpotDrill.bornes_du_cout`` en donne
l'encadrement, ``explication()`` refuse de l'imprimer comme un fait, et le
rapport n'annonce jamais un total sans en isoler la part non mesurée.

Mesuré sur le corpus de développement (138 mains iPoker) : le total affiché
vaut 0,6936 bb, dont 0,5885 bb (84,9 %) viennent d'un unique limp QJo à 12 bb.
Le coût réel de l'ensemble est encadré par [0,11 ; 1,19] bb.

Priorisation
------------
La priorité d'un drill est le **coût total en bb** de la fuite qu'il cible,
c'est-à-dire la somme des coûts de toutes ses occurrences. Ce seul nombre
règle l'arbitrage demandé entre gravité et fréquence : vingt erreurs à
0,037 bb (0,73 bb au total) passent devant une erreur isolée à 0,49 bb.

Le tri hérite de la convention ci-dessus : un limp y est classé au prix du
fold qu'il abandonne. C'est ce qui le met en tête du corpus de développement,
et le rapport l'écrit à côté du chiffre plutôt que de le laisser croire mesuré.

Planification
-------------
Les drills sont branchés sur la répétition espacée SM-2 existante
(``pfs.train.drill.SpacedRepetition``, Wozniak 1990) : un spot raté revient
immédiatement, un spot acquis s'espace exponentiellement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from pfs.analysis.pushfold_review import (
    PushFoldReview,
    ShoveSpot,
    review_pushfold_hands,
)
from pfs.analysis.session_review import HeroProfile, review_hands
from pfs.analysis.spot_advisor import Spot, advise
from pfs.core.range_model import RANKS, card_str
from pfs.data.hand_history import ParsedHand, parse_file
from pfs.train.drill import Grade, SpacedRepetition

__all__ = [
    "SpotDrill",
    "ReponseDrill",
    "RapportDrills",
    "SessionDrillsCibles",
    "generer_drills",
    "generer_drills_mains",
    "generer_drills_dossier",
]

# Structure du modèle push/fold : le héros est au bouton (= small blind en
# heads-up) et complète 0,5 bb pour voir la big blind.
_SB_BB = 0.5
_BB_BB = 1.0
_POSITION = "BTN/SB (heads-up)"
_OPTIONS = ("JAM", "FOLD")

# Le solve chipEV de la revue (``solve_hu_pushfold`` sans ``payouts``) bâtit un
# vecteur de DEUX tapis : quelle que soit la table, il ne fait payer que deux
# antes. Le pot affiché suit ce modèle, sans quoi la question serait posée dans
# une structure que la réponse n'a pas résolue.
_JOUEURS_DU_MODELE = 2

# Familles de fuites, avec ce qu'elles coûtent *structurellement*. Le libellé
# de gauche est le verdict brut de ``pushfold_review``.
_FUITES: dict[str, tuple[str, str]] = {
    "jam trop lâche": (
        "Jam trop lâche",
        "tu pousses des mains que l'équilibre couche : chaque jam de trop "
        "est de l'EV jetée, et c'est la traduction directe d'un VPIP large.",
    ),
    "fold trop serré": (
        "Fold trop serré",
        "tu couches des mains dont le jam est rentable : l'EV abandonnée ne "
        "se voit jamais dans le résultat, seul le solveur la chiffre.",
    ),
    "limp": (
        "Limp en tapis court",
        "à ce tapis, seules deux actions ont une EV définie ; limper les "
        "abandonne toutes les deux et laisse la main se jouer sans initiative.",
    ),
}

_FUITES_NON_MESUREES = frozenset({"limp"})
"""Familles dont le coût est une convention, pas une mesure (voir le module)."""

_ECART_MULTIWAY = "multiway (hors solveur heads-up)"
_ECART_PROFOND = "tapis > 25 bb (hors push/fold)"
_ECART_RELANCE = "relance non-jam (hors push/fold)"
_ECART_LIMP_NUL = "limp sans coût mesurable"
_ECART_INCERTAIN = "réponse non certaine (hors périmètre du solveur)"
_ECART_ANTE = "verdict retourné par l'ante (non robuste)"

_RAISONS_ECART = (
    _ECART_MULTIWAY,
    _ECART_PROFOND,
    _ECART_RELANCE,
    _ECART_LIMP_NUL,
    _ECART_INCERTAIN,
    _ECART_ANTE,
)

_ECARTS_DEFENSIFS = frozenset({_ECART_INCERTAIN})
"""Raisons que le pipeline ne peut structurellement pas déclencher.

``review_pushfold_hands`` ne produit que des spots à deux joueurs vivants et à
``1 ≤ eff_bb ≤ 25``, domaine sur lequel ``advise`` renvoie toujours
``confidence == "certain"`` (``MAX_PUSHFOLD_BB ≥ MAX_EFF_BB``, testé). Le
compteur reste comme garde-fou d'une revue construite à la main, mais un zéro
en face n'est pas un filtrage observé — c'est une tautologie, et le rapport le
dit plutôt que de l'exhiber comme une mesure.
"""


def _cartes_representatives(groupe: str) -> str:
    """Combo canonique d'un groupe de mains, quand les cartes réelles manquent.

    Parameters
    ----------
    groupe : str
        Nom de groupe de la grille 13 × 13 : « AA », « A5s », « A5o ».

    Returns
    -------
    str
        Les deux cartes, prêtes pour ``spot_advisor.parse_cards`` : paire →
        ♠♥, suited → ♠♠, offsuit → ♠♥.

    Raises
    ------
    ValueError
        Si le nom n'est pas un groupe valide.

    Notes
    -----
    Le choix du représentant est neutre : par symétrie des couleurs, l'équité
    préflop d'un combo contre une range fermée par couleur ne dépend pas du
    représentant — c'est la propriété sur laquelle repose déjà la matrice
    169 × 169 du solveur. La réponse Nash du drill est donc identique à celle
    des cartes réellement tenues.

    Examples
    --------
    >>> _cartes_representatives("A5o")
    'As 5h'
    """
    if len(groupe) == 2 and groupe[0] == groupe[1] and groupe[0] in RANKS:
        r = RANKS.index(groupe[0])
        return f"{card_str(r * 4)} {card_str(r * 4 + 1)}"
    if len(groupe) == 3 and groupe[0] in RANKS and groupe[1] in RANKS:
        hi, lo = RANKS.index(groupe[0]), RANKS.index(groupe[1])
        if groupe[2] == "s":
            return f"{card_str(hi * 4)} {card_str(lo * 4)}"
        if groupe[2] == "o":
            return f"{card_str(hi * 4)} {card_str(lo * 4 + 1)}"
    raise ValueError(f"groupe de mains invalide : {groupe!r}.")


def _memes_verdicts(ev_conseil: float, ev_revue: float) -> bool:
    """Les deux solves désignent-ils la même action ?

    Parameters
    ----------
    ev_conseil : float
        ``EV(jam) − EV(fold)`` du conseiller, qui solve **sans ante**.
    ev_revue : float
        La même quantité côté revue, qui solve **avec** l'ante de la main.

    Returns
    -------
    bool
        Vrai si les deux sont du même côté de zéro. Faux ⇒ l'ante retourne la
        décision : la bonne réponse dépend d'un détail de structure que le
        drill n'afficherait pas, on n'enseigne pas ce spot.

    Notes
    -----
    Seul le SIGNE est confronté, jamais la magnitude — les deux solves ne
    modélisent pas la même structure. Mesuré sur ce dépôt (heads-up, 10 et
    12 bb, ante de 0 à 0,25 bb) : l'écart relatif entre les deux EV vaut 0 %
    à ante nulle (c'est alors le même appel), 4,3 % pour 72o à 10 bb avec
    0,10 bb d'ante, 15,0 % à 0,20 bb, et jusqu'à 76,7 % pour QJo à 12 bb avec
    0,25 bb d'ante ; le signe lui-même bascule pour Q7o à 12 bb dès 0,15 bb
    d'ante (−0,0776 sans ante, +0,0116 avec). C'est pourquoi le drill affiche
    l'EV de la revue — celle qui a produit son coût et son pot — et jamais
    celle du conseiller, qui ne sert ici que de contre-épreuve de signe.
    """
    return (ev_conseil >= 0.0) == (ev_revue >= 0.0)


@dataclass(frozen=True, slots=True)
class SpotDrill:
    """Un exercice rejouant un spot réellement raté.

    Attributes
    ----------
    cle : str
        Clé stable de planification SM-2 (main, tapis arrondi, bonne réponse).
    fuite : str
        Famille de fuite ciblée (verdict de ``pushfold_review``).
    main, cartes : str
        Le groupe (« A5o ») et les deux cartes affichées (« As 5h »).
    tapis_bb, pot_bb, mise_bb, ante_bb : float
        Tapis effectif, pot avant l'action du héros, complément à payer, et
        ante par joueur — tous dans la structure que le solveur a résolue.
    options : tuple of str
        Les réponses admises.
    bonne_reponse : str
        Tranchée par le solveur, jamais par une heuristique.
    action_jouee : str
        Ce que le héros a réellement fait (« jam », « fold », « limp »).
    ev_jam_moins_fold : float
        ``EV(jam) − EV(fold)`` à l'équilibre, en bb (chipEV), issue du **même
        solve que le coût et le pot** (celui de la revue, avec l'ante).
        Positif ⇒ jam.
    ev_sans_ante_bb : float
        La même quantité au solve sans ante du conseiller — contre-épreuve de
        signe, pas la même magnitude (voir ``_memes_verdicts``).
    cout_unitaire_bb, cout_total_bb : float
        Coût d'UNE occurrence, et coût cumulé de toutes les occurrences.
    cout_mesure : bool
        Faux pour un limp : le coût affiché n'est alors pas une mesure mais la
        valeur qu'il aurait si limper valait passer (voir ``bornes_du_cout``).
    occurrences : int
        Nombre de fois où cette erreur exacte a été commise.
    mains_ids : tuple of str
        Identifiants des mains d'origine, une par occurrence.
    regime : str
        Le solve qui a produit ``ev_jam_moins_fold``, ante comprise.
    cartes_reelles : bool
        Faux si les cartes affichées sont le représentant canonique du groupe
        (historique non fourni) — la réponse reste exacte, pas l'anecdote ; le
        pot est alors celui d'une structure sans ante, faute de main à lire.
    """

    cle: str
    fuite: str
    main: str
    cartes: str
    tapis_bb: float
    pot_bb: float
    mise_bb: float
    ante_bb: float
    options: tuple[str, ...]
    bonne_reponse: str
    action_jouee: str
    ev_jam_moins_fold: float
    ev_sans_ante_bb: float
    cout_unitaire_bb: float
    cout_total_bb: float
    cout_mesure: bool
    occurrences: int
    mains_ids: tuple[str, ...]
    regime: str
    cartes_reelles: bool

    @property
    def priorite(self) -> float:
        """Coût total de la fuite en bb — l'unique critère de tri.

        Le classement hérite de la convention des limps : un limp y est rangé
        au prix du fold qu'il abandonne, donc sur un coût non mesuré (voir
        ``cout_mesure``). Sur le corpus de développement, c'est ce qui place un
        limp en tête — ``explain()`` le dit à côté du chiffre.
        """
        return self.cout_total_bb

    @property
    def bornes_du_cout(self) -> tuple[float, float]:
        """Encadrement du coût total réel, en bb.

        Pour un jam ou un fold, les deux actions du modèle ont une EV : le
        coût est mesuré, l'encadrement est ponctuel.

        Pour un limp, ``cout_total_bb`` n'est pas une mesure. L'encadrement
        retenu est :

        - **bas** : 0 bb — si limper valait le jam, l'erreur ne coûte rien ;
        - **haut** : ``cout_total_bb + occurrences × mise_bb``. La pire suite
          possible après un limp est de ne plus jamais remettre un jeton : le
          héros perd alors le 1 bb engagé au lieu des ``_SB_BB`` d'un fold
          immédiat, soit exactement ``mise_bb`` de plus. Toute autre suite fait
          mieux, donc ``EV(limp) ≥ EV(fold) − mise_bb`` sans hypothèse.
        """
        if self.cout_mesure:
            return self.cout_total_bb, self.cout_total_bb
        return 0.0, self.cout_total_bb + self.occurrences * self.mise_bb

    def question(self) -> str:
        """L'énoncé présenté au joueur, sans jamais souffler la réponse."""
        return (
            f"{_POSITION} · préflop · {self.tapis_bb:.1f} bb effectifs\n"
            f"  Ta main : {self.cartes} ({self.main})\n"
            f"  Pot     : {self.pot_bb:.1f} bb · il t'en coûte "
            f"{self.mise_bb:.1f} bb pour suivre\n"
            f"  {' ou '.join(self.options)} ?"
        )

    def explication(self) -> str:
        """Le corrigé : la réponse, sa source chiffrée, et la fuite visée."""
        if self.cout_mesure:
            cout = (f"{self.cout_total_bb:.2f} bb perdues "
                    f"({self.cout_unitaire_bb:.2f} bb par occurrence)")
        else:
            bas, haut = self.bornes_du_cout
            cout = (f"{self.cout_total_bb:.2f} bb NON MESURÉES : ce modèle ne "
                    f"définit pas l'EV d'un limp, ce chiffre est ce que "
                    f"l'erreur coûterait si limper valait passer — coût réel "
                    f"encadré par [{bas:.2f} ; {haut:.2f}] bb")
        lignes = [
            f"Bonne réponse : {self.bonne_reponse}",
            f"  {self.regime} : EV(jam) − EV(fold) = "
            f"{self.ev_jam_moins_fold:+.2f} bb.",
            f"  Dans tes mains tu as choisi « {self.action_jouee} » "
            f"{self.occurrences} fois → {cout}.",
        ]
        if self.ante_bb > 0.0:
            lignes.append(
                f"  Contre-épreuve sans ante : {self.ev_sans_ante_bb:+.2f} bb "
                "— même action, magnitude différente : les deux solves ne sont "
                "confrontés que sur le signe."
            )
        lignes.append(f"  Fuite ciblée : {_FUITES[self.fuite][0]} — "
                      f"{_FUITES[self.fuite][1]}")
        return "\n".join(lignes)


@dataclass(frozen=True, slots=True)
class ReponseDrill:
    """Une réponse donnée à un drill ciblé."""

    drill: SpotDrill
    reponse: str
    correcte: bool
    grade: Grade
    cout_bb: float
    """bb qu'aurait coûté cette réponse dans la main d'origine (0 si correcte).

    Hérite de ``SpotDrill.cout_unitaire_bb`` : pour un limp, ce n'est donc pas
    une mesure (``drill.cout_mesure`` est faux).
    """
    secondes: float


@dataclass(slots=True)
class RapportDrills:
    """Les drills générés et la comptabilité honnête de ce qui a été écarté.

    L'objet se comporte comme la liste de drills (``len``, itération,
    indexation) tout en portant le rapport ``explain()``.

    La comptabilité est bouclée et vérifiable :
    ``n_decisions_vues == n_spots_juges + n_ecartes``, sans recouvrement — une
    décision est soit jugée (correcte ou erreur chiffrée), soit écartée pour
    une raison et une seule.
    """

    drills: list[SpotDrill] = field(default_factory=list)
    n_mains: int = 0
    n_spots_corrects: int = 0
    n_erreurs: int = 0
    ecartes: dict[str, int] = field(default_factory=dict)
    profil: HeroProfile | None = None
    n_fichiers_illisibles: int = 0
    n_fichiers_sans_main: int = 0

    def __len__(self) -> int:
        return len(self.drills)

    def __iter__(self) -> Iterator[SpotDrill]:
        return iter(self.drills)

    def __getitem__(self, index: int | slice) -> SpotDrill | list[SpotDrill]:
        return self.drills[index]

    @property
    def cout_cible_bb(self) -> float:
        """bb affichées pour l'ensemble des fuites ciblées.

        Somme brute des coûts, part non mesurée comprise : c'est le nombre
        qu'affiche ``explain()``, et il n'est jamais présenté seul.
        """
        return sum(d.cout_total_bb for d in self.drills)

    @property
    def cout_mesure_bb(self) -> float:
        """La part de ``cout_cible_bb`` qui sort du solveur (jams et folds)."""
        return sum(d.cout_total_bb for d in self.drills if d.cout_mesure)

    @property
    def cout_non_mesure_bb(self) -> float:
        """La part conventionnelle de ``cout_cible_bb`` (limps)."""
        return sum(d.cout_total_bb for d in self.drills if not d.cout_mesure)

    @property
    def bornes_du_cout(self) -> tuple[float, float]:
        """Encadrement du coût réel de l'ensemble des fuites ciblées, en bb."""
        bornes = [d.bornes_du_cout for d in self.drills]
        return sum(b for b, _ in bornes), sum(h for _, h in bornes)

    @property
    def n_ecartes(self) -> int:
        """Décisions écartées, toutes raisons confondues (sans recouvrement)."""
        return sum(self.ecartes.values())

    @property
    def n_spots_juges(self) -> int:
        """Décisions effectivement jugées : correctes + erreurs chiffrées."""
        return self.n_spots_corrects + self.n_erreurs

    @property
    def n_decisions_vues(self) -> int:
        """Décisions préflop parcourues par la revue, jugées ou écartées."""
        return self.n_spots_juges + self.n_ecartes

    @property
    def fuites(self) -> list[tuple[str, int, float]]:
        """``(famille, nombre de drills, coût total bb)``, les plus chères d'abord."""
        agg: dict[str, tuple[int, float]] = {}
        for d in self.drills:
            n, cout = agg.get(d.fuite, (0, 0.0))
            agg[d.fuite] = (n + 1, cout + d.cout_total_bb)
        return sorted(
            ((f, n, c) for f, (n, c) in agg.items()),
            key=lambda t: -t[2],
        )

    def explain(self) -> str:
        """Rapport : quelles fuites sont ciblées, pourquoi, et ce qui est écarté."""
        bas, haut = self.bornes_du_cout
        lines = [
            "══════════════════════════════════════════════════════════",
            "  DRILLS CIBLÉS — tes fuites mesurées, remises à l'entraînement",
            "══════════════════════════════════════════════════════════",
            f"  Mains fournies              : {self.n_mains}",
            f"  Décisions préflop vues      : {self.n_decisions_vues}",
            f"    · jugées                  : {self.n_spots_juges}"
            f"   ({self.n_spots_corrects} correctes, {self.n_erreurs} erreurs)",
            f"    · écartées                : {self.n_ecartes}",
            f"  Drills générés              : {len(self.drills)}"
            f"   ({self.cout_cible_bb:.2f} bb affichées)",
        ]
        if self.n_fichiers_illisibles:
            lines.append(f"  Fichiers illisibles ignorés : "
                         f"{self.n_fichiers_illisibles}")
        if self.n_fichiers_sans_main:
            lines.append(f"  Fichiers sans main lisible  : "
                         f"{self.n_fichiers_sans_main}")
        if self.cout_non_mesure_bb > 0.0:
            part = self.cout_non_mesure_bb / self.cout_cible_bb * 100.0
            lines += [
                f"      dont mesuré (jam/fold)  : {self.cout_mesure_bb:.2f} bb",
                f"      dont limps, NON MESURÉ  : {self.cout_non_mesure_bb:.2f} bb"
                f"   ({part:.0f} % du total affiché)",
                f"      Coût réel encadré par   : [{bas:.2f} ; {haut:.2f}] bb",
                "      Un limp n'a pas d'EV dans ce modèle : il y est valorisé "
                "au prix du fold.",
                "      Le total affiché est donc une convention, pas une mesure.",
            ]

        lines += ["", f"  Spots écartés (aucun drill fabriqué) : {self.n_ecartes}"]
        for raison, n in self.ecartes.items():
            marque = "   (garde-fou)" if raison in _ECARTS_DEFENSIFS else ""
            lines.append(f"    · {raison:<48} {n:>4}{marque}")
        lines += [
            "    Une bonne réponse incertaine apprendrait une erreur : ces "
            "spots sont comptés, pas devinés.",
            "    « garde-fou » : raison que le pipeline ne peut structurellement "
            "pas atteindre (la revue",
            "    borne déjà le tapis et le nombre de joueurs) — un 0 en face "
            "n'est pas un filtrage observé.",
        ]

        if self.profil is not None:
            p = self.profil
            ecart = p.vpip - p.pfr
            lines += [
                "",
                "  Profil observé (revue de session) :",
                f"    VPIP {p.vpip:5.1f} %  ·  PFR {p.pfr:5.1f} %  ·  "
                f"écart {ecart:.1f} points",
                f"    → {ecart:.1f} % des mains sont jouées sans relancer ; "
                "c'est là que vivent limps et suivis dominés.",
                "    Ces fréquences localisent la fuite ; seules les décisions "
                "ci-dessous sont chiffrées par le solveur.",
            ]

        if self.drills:
            lines += ["", "  Fuites ciblées, les plus coûteuses d'abord :"]
            for i, (fuite, n, cout) in enumerate(self.fuites, 1):
                titre, pourquoi = _FUITES[fuite]
                marque = "" if fuite not in _FUITES_NON_MESUREES else " (non mesuré)"
                lines.append(f"    {i}. {titre:<22} {n:>3} drill(s) · "
                             f"{cout:6.2f} bb{marque}")
                lines.append(f"       {pourquoi}")
            lines += ["", "  Ordre de révision :"]
            for i, d in enumerate(self.drills, 1):
                mesure = "" if d.cout_mesure else " non mesuré"
                lines.append(
                    f"    {i:>2}. {d.main:>4} @ {d.tapis_bb:4.1f} bb  → "
                    f"{d.bonne_reponse:<4}  (tu {d.action_jouee}s ×"
                    f"{d.occurrences}, {d.cout_total_bb:.2f} bb{mesure})"
                )
        else:
            lines += [
                "",
                "  Aucune fuite chiffrable dans ce corpus : soit les décisions "
                "jugeables étaient correctes, soit elles sortent du périmètre "
                "où le solveur tranche avec certitude.",
            ]
        lines.append("══════════════════════════════════════════════════════════")
        return "\n".join(lines)


def _contexte_du_pot(main: ParsedHand | None) -> tuple[float, float, float]:
    """Pot avant l'action du héros, complément à payer, et ante — en bb.

    L'ante comptée est celle de ``_JOUEURS_DU_MODELE`` joueurs, pas de toute la
    table : afficher le pot réel d'une table à six poserait la question dans une
    structure que la réponse n'a pas résolue. Écart mesuré à 0,075 bb d'ante et
    six sièges : 1,95 bb pour la table, 1,65 bb pour le modèle.

    L'ante est arrondie au centième de bb, la granularité exacte de la clé de
    mémoïsation du solve de la revue — le pot affiché est alors celui-là même
    qui a produit l'EV et le coût.

    Main absente ⇒ ante inconnue ⇒ pot sans ante ; ``SpotDrill.cartes_reelles``
    signale ce cas.
    """
    mise = _BB_BB - _SB_BB
    if main is None or main.big_blind <= 0 or main.ante <= 0:
        return _SB_BB + _BB_BB, mise, 0.0
    ante = round(main.ante / main.big_blind * 100.0) / 100.0
    return _SB_BB + _BB_BB + _JOUEURS_DU_MODELE * ante, mise, ante


def _cartes_du_spot(
    spot: ShoveSpot, index: dict[str, ParsedHand]
) -> tuple[str, bool, ParsedHand | None]:
    """Cartes affichées : celles réellement tenues si l'historique est fourni."""
    main = index.get(spot.hand_id)
    if main is not None and len(main.hero_cards) == 2:
        return " ".join(main.hero_cards), True, main
    return _cartes_representatives(spot.hand), False, main


def _regime(eff_bb: float, ante_bb: float) -> str:
    """Nom du solve qui a produit l'EV affichée — l'ante y figure si elle existe."""
    if ante_bb > 0.0:
        return f"Nash push/fold {eff_bb:.1f} bb (ante {ante_bb:.2f} bb)"
    return f"Nash push/fold {eff_bb:.1f} bb"


@dataclass(frozen=True, slots=True)
class _Occurrence:
    """Une erreur retenue, avant regroupement des répétitions."""

    spot: ShoveSpot
    cartes: str
    cartes_reelles: bool
    main: ParsedHand | None
    ev_sans_ante: float


def generer_drills(
    revue: PushFoldReview,
    mains: Sequence[ParsedHand] = (),
    *,
    profil: HeroProfile | None = None,
    max_drills: int | None = None,
) -> RapportDrills:
    """Fabrique les drills ciblés d'une revue shove/fold.

    Parameters
    ----------
    revue : PushFoldReview
        Sortie de ``pfs.analysis.pushfold_review`` : chaque décision préflop
        de tapis court déjà confrontée au Nash, avec son coût en bb.
    mains : sequence of ParsedHand, optional
        Les mains d'origine. Fournies, les drills affichent les cartes
        réellement tenues et le pot exact (ante comprise) ; sinon le
        représentant canonique du groupe et un pot sans ante (la bonne réponse
        est la même, voir ``_cartes_representatives``).
    profil : HeroProfile, optional
        Profil statistique (``SessionReport.profile``) — sert au rapport, pas
        à la génération : aucune fuite n'est déduite d'une fréquence.
    max_drills : int, optional
        Ne garder que les ``max_drills`` fuites les plus coûteuses. La coupe
        porte sur l'affichage, jamais sur la comptabilité : ``n_erreurs`` et
        les écartés restent ceux du corpus entier.

    Returns
    -------
    RapportDrills
        Les drills triés par coût total décroissant, et la comptabilité des
        spots écartés.

    Raises
    ------
    ValueError
        Si ``max_drills`` est fourni et < 1.
    """
    if max_drills is not None and max_drills < 1:
        raise ValueError("max_drills doit être >= 1.")

    index = {h.hand_id: h for h in mains}
    ecartes = {r: 0 for r in _RAISONS_ECART}
    ecartes[_ECART_MULTIWAY] = revue.n_skipped_multiway
    ecartes[_ECART_PROFOND] = revue.n_skipped_deep

    n_corrects = 0
    groupes: dict[tuple[str, int, str], list[_Occurrence]] = {}

    for spot in revue.spots:
        if spot.verdict == "ok":
            if spot.decision == "raise":
                ecartes[_ECART_RELANCE] += 1
            else:
                n_corrects += 1
            continue
        if spot.verdict == "limp" and spot.cost_bb <= 0.0:
            ecartes[_ECART_LIMP_NUL] += 1
            continue

        cartes, reelles, main = _cartes_du_spot(spot, index)
        conseil = advise(
            Spot(hero=cartes, stack=spot.eff_bb, big_blind=1.0,
                 position="BTN", players=2)
        )
        if conseil.confidence != "certain":
            ecartes[_ECART_INCERTAIN] += 1
            continue
        if not _memes_verdicts(float(conseil.ev_bb), spot.ev_jam_minus_fold):
            ecartes[_ECART_ANTE] += 1
            continue

        bonne = "JAM" if conseil.action.startswith("JAM") else "FOLD"
        cle = (spot.hand, round(spot.eff_bb), bonne)
        groupes.setdefault(cle, []).append(
            _Occurrence(spot, cartes, reelles, main, float(conseil.ev_bb))
        )

    n_erreurs = sum(len(occ) for occ in groupes.values())
    drills: list[SpotDrill] = []
    for (nom, tapis_arrondi, bonne), occurrences in groupes.items():
        # Le spot exposé est le plus coûteux du paquet : à tapis quasi
        # identiques, c'est celui dont l'erreur se paie le plus cher. Le total
        # peut agréger des structures d'ante différentes ; le corrigé affiche
        # celle de ce spot-là, jamais une moyenne qu'aucune main n'a jouée.
        pire = max(occurrences, key=lambda o: o.spot.cost_bb)
        pot, mise, ante = _contexte_du_pot(pire.main)
        drills.append(
            SpotDrill(
                cle=f"{nom}|{tapis_arrondi}bb|{bonne}",
                fuite=pire.spot.verdict,
                main=nom,
                cartes=pire.cartes,
                tapis_bb=pire.spot.eff_bb,
                pot_bb=pot,
                mise_bb=mise,
                ante_bb=ante,
                options=_OPTIONS,
                bonne_reponse=bonne,
                action_jouee=pire.spot.decision,
                ev_jam_moins_fold=pire.spot.ev_jam_minus_fold,
                ev_sans_ante_bb=pire.ev_sans_ante,
                cout_unitaire_bb=pire.spot.cost_bb,
                cout_total_bb=sum(o.spot.cost_bb for o in occurrences),
                cout_mesure=pire.spot.verdict not in _FUITES_NON_MESUREES,
                occurrences=len(occurrences),
                mains_ids=tuple(o.spot.hand_id for o in occurrences),
                regime=_regime(pire.spot.eff_bb, ante),
                cartes_reelles=pire.cartes_reelles,
            )
        )

    drills.sort(key=lambda d: (-d.cout_total_bb, -d.occurrences, d.cle))
    if max_drills is not None:
        drills = drills[:max_drills]

    return RapportDrills(
        drills=drills,
        n_mains=len(mains),
        n_spots_corrects=n_corrects,
        n_erreurs=n_erreurs,
        ecartes=ecartes,
        profil=profil,
    )


def generer_drills_mains(
    mains: Sequence[ParsedHand],
    hero: str | None = None,
    *,
    profil: HeroProfile | None = None,
    max_drills: int | None = None,
) -> RapportDrills:
    """Revue shove/fold puis génération des drills, en un appel."""
    revue = review_pushfold_hands(mains, hero)
    return generer_drills(revue, mains, profil=profil, max_drills=max_drills)


def generer_drills_dossier(
    path: str | Path,
    salt: str = "",
    *,
    avec_profil: bool = True,
    max_drills: int | None = None,
) -> RapportDrills:
    """Lit tous les XML iPoker d'un dossier et en tire les drills ciblés.

    Parameters
    ----------
    path : str or Path
        Dossier d'historiques (emplacement PMU typique :
        ``%LocalAppData%\\PMU PLAY 100%% Poker\\data\\<pseudo>\\History``).
    salt : str
        Sel de hachage des pseudos, identique à celui des autres revues.
    avec_profil : bool, default True
        Calcule aussi le profil statistique (VPIP/PFR…) pour le rapport. Le
        coût est celui de la revue de session : Monte-Carlo d'équité sur les
        tapis abattus.
    max_drills : int, optional
        Ne garder que les fuites les plus coûteuses.

    Returns
    -------
    RapportDrills
        Deux compteurs disent ce que la lecture a perdu, plutôt que de laisser
        un corpus tronqué passer pour complet : ``n_fichiers_illisibles``
        (l'accès au fichier a échoué) et ``n_fichiers_sans_main``
        (``iter_hands`` n'en a tiré aucune main — XML cassé ou session vide,
        il les avale sans lever).
    """
    root = Path(path)
    mains: list[ParsedHand] = []
    illisibles = 0
    sans_main = 0
    for f in sorted(root.rglob("*.xml")):
        try:
            lues = parse_file(f, salt)
        except Exception:
            illisibles += 1
            continue
        if not lues:
            sans_main += 1
        mains.extend(lues)
    profil = review_hands(mains).profile if avec_profil else None
    rapport = generer_drills_mains(mains, profil=profil, max_drills=max_drills)
    rapport.n_fichiers_illisibles = illisibles
    rapport.n_fichiers_sans_main = sans_main
    return rapport


class SessionDrillsCibles:
    """Enchaîne les drills ciblés en réutilisant la planification SM-2.

    La sélection diffère de ``drill.DrillSession`` sur un point, qui est tout
    l'objet de ce module : à défaut d'item dû, ce n'est pas un spot tiré au
    hasard qui sort, mais **la fuite la plus coûteuse encore jamais révisée**.

    Parameters
    ----------
    drills : iterable of SpotDrill
        Les drills à réviser.
    srs : SpacedRepetition, optional
        Planificateur existant, pour poursuivre un historique de révisions.

    Raises
    ------
    ValueError
        Si la liste de drills est vide, ou si deux drills partagent une clé.
    """

    __slots__ = ("srs", "drills", "reponses", "_par_cle")

    SECONDES_LENTES = 8.0
    """Au-delà, un rappel correct est dégradé — même seuil que ``DrillSession``,
    pour que les deux modes d'entraînement notent de la même façon."""

    def __init__(
        self,
        drills: Sequence[SpotDrill],
        srs: SpacedRepetition | None = None,
    ) -> None:
        items = list(drills)
        if not items:
            raise ValueError("aucun drill à enchaîner.")
        self.drills = sorted(items, key=lambda d: (-d.priorite, d.cle))
        self._par_cle = {d.cle: d for d in self.drills}
        if len(self._par_cle) != len(self.drills):
            raise ValueError("deux drills partagent la même clé SM-2.")
        self.srs = srs if srs is not None else SpacedRepetition()
        self.reponses: list[ReponseDrill] = []

    def prochain(self) -> SpotDrill | None:
        """Le prochain drill : dû d'abord, sinon la fuite la plus chère.

        Returns
        -------
        SpotDrill or None
            ``None`` quand tous les drills ont été vus et qu'aucun n'est dû —
            il faut alors avancer le temps logique (``srs.advance``).
        """
        for carte in self.srs.due(limit=len(self.drills)):
            drill = self._par_cle.get(carte.key)
            if drill is not None:
                return drill
        for drill in self.drills:
            if drill.cle not in self.srs.cards:
                return drill
        return None

    def repondre(
        self, drill: SpotDrill, reponse: str, secondes: float = 0.0
    ) -> ReponseDrill:
        """Note une réponse et replanifie le spot.

        Une réponse fausse vaut ``Grade.BLACKOUT`` : le choix est binaire, il
        n'y a pas de crédit partiel à accorder — et SM-2 remet alors
        l'intervalle à zéro, donc le spot raté revient immédiatement.

        Raises
        ------
        ValueError
            Si la réponse n'est pas l'une des options du drill.
        """
        rep = reponse.strip().upper()
        if rep not in drill.options:
            raise ValueError(
                f"réponse {reponse!r} inconnue ; options : "
                f"{', '.join(drill.options)}."
            )
        correcte = rep == drill.bonne_reponse
        if not correcte:
            grade = Grade.BLACKOUT
        elif secondes > self.SECONDES_LENTES:
            grade = Grade.GOOD
        else:
            grade = Grade.PERFECT
        self.srs.review(drill.cle, grade)
        note = ReponseDrill(
            drill=drill,
            reponse=rep,
            correcte=correcte,
            grade=grade,
            cout_bb=0.0 if correcte else drill.cout_unitaire_bb,
            secondes=secondes,
        )
        self.reponses.append(note)
        return note

    def score(self) -> dict[str, float]:
        """Exactitude, bb encore lâchées, et maîtrise SM-2.

        Returns
        -------
        dict of str to float
            Toujours les mêmes cinq clés — ``n``, ``exactitude``,
            ``bb_manquees``, ``bb_ciblees``, ``maitrise`` — y compris sur une
            session vierge : un appelant n'a pas à connaître l'état interne
            pour savoir quelles clés existent.

            ``bb_manquees`` hérite des coûts unitaires : la part venant de
            limps n'est pas une mesure (voir ``SpotDrill.cout_mesure``).
        """
        n = len(self.reponses)
        justes = sum(1 for r in self.reponses if r.correcte)
        return {
            "n": float(n),
            "exactitude": justes / n if n else 0.0,
            "bb_manquees": float(sum(r.cout_bb for r in self.reponses)),
            "bb_ciblees": float(sum(d.priorite for d in self.drills)),
            "maitrise": self.srs.mastery(),
        }
