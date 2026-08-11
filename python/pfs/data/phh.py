r"""Lecteur PHH — le format d'échange des corpus publics de poker.

PHH (*poker hand history*, https://phh.readthedocs.io/) est le format
déclaratif que la recherche universitaire utilise pour publier des mains.
Savoir le lire ouvre les corpus MIT du dépôt ``phh-dataset``.

Ce que ça donne **exactement**, mesuré (``banc_corpus_pluribus.py
--verifier``) :

* **Pluribus** (Brown & Sandholm, *Science*, 2019) : 10 000 mains lues sur
  10 000. Tout le corpus.
* **WSOP 2023**, table finale du Poker Players Championship : 18 mains lues
  sur 83. Les 65 autres ne sont pas du hold'em — c'est un championnat
  *mixte* : Omaha hi-lo, stud, razz, triple draw. Elles sont refusées, pas
  mal lues.
* mains historiques isolées, dont **Dwan-Ivey 2009**, dont le déroulé est
  vérifiable à la main et sert de golden aux tests.

Le format est du TOML :

.. code-block:: toml

    variant = "NT"                       # no-limit Texas hold'em
    antes = [0, 0, 0, 0, 0, 0]
    blinds_or_straddles = [50, 100, 0, 0, 0, 0]
    starting_stacks = [10000, 10000, 10000, 10000, 10000, 10000]
    actions = ['d dh p1 JsAh', ..., 'p5 cbr 250', 'p6 cbr 737', 'p1 f']
    finishing_stacks = [9950, 9900, 10000, 10000, 9750, 10400]

Sémantique des actions, **vérifiée sur les tapis de fin** (voir
:func:`parse_phh`, section « comptabilité ») :

===============  ==========================================================
``d dh pN XxYy``  distribution des cartes fermées du joueur N
``d db XxYyZz``   cartes communes (3 = flop, puis 1 = turn, puis 1 = river)
``pN f``          couche
``pN cc``         checke ou suit
``pN cbr X``      **monte à X** — X est le total misé par N SUR CETTE RUE,
                  pas l'incrément payé
``pN sm XxYy``    abat ou jette ; n'apporte rien ici, ``d dh`` donnant déjà
                  toutes les cartes
===============  ==========================================================

Ce que ce lecteur donne de PLUS qu'un hand history de room
----------------------------------------------------------
Les cartes de **tous** les joueurs, y compris ceux qui se sont couchés :
``d dh`` les déclare toutes. Un HH réel ne montre que le héros et les mains
abattues. C'est ce qui permet de chiffrer *exactement* le résultat d'un call
que le joueur n'a pas fait (voir ``banc_corpus_pluribus.py``).

Ce que ce lecteur NE donne PAS
------------------------------
* **Le tournoi n'est pas déclaré** par le format : ``is_tournament`` est un
  paramètre de l'appelant, faute de quoi il vaut ``False``. Pluribus et
  Dwan-Ivey sont du cash game ; s'en remettre au fichier serait deviner.
* **Les variantes non hold'em** sont refusées explicitement plutôt que
  mal lues : seuls ``NT`` (no-limit hold'em) et ``FT`` (limit hold'em)
  partagent la grammaire d'actions et l'évaluateur du projet. Un badugi ou
  un 2-7 triple draw lève :class:`PhhError` en nommant la variante.
* ``ante_trimming_status`` n'est pas modélisé : les antes sont postées telles
  que déclarées. Sur les corpus lus ici elles sont uniformes (Pluribus : 0
  partout ; Dwan-Ivey : 500 pour les trois), donc le rognage n'a rien à
  faire.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pfs.data.hand_history import (
    ActionType,
    HandAction,
    HandHistoryError,
    ParsedHand,
    PlayerSeat,
    Room,
    Street,
    player_key,
)

__all__ = [
    "PhhError",
    "Decision",
    "VARIANTES_SUPPORTEES",
    "parse_phh",
    "parse_phh_file",
    "phh_players",
    "iter_phh_files",
    "positions_par_siege",
    "iter_decisions",
]


class PhhError(HandHistoryError):
    """Fichier PHH illisible, ou variante hors du domaine du projet."""


VARIANTES_SUPPORTEES = frozenset({"NT", "FT"})
"""Variantes lues : hold'em no-limit et hold'em limit.

Ce sont les seules dont la grammaire d'actions (``f`` / ``cc`` / ``cbr``) et
l'évaluateur 7 cartes du projet valent. Les variantes de draw ajoutent
``sd``/``dd``, Omaha change le nombre de cartes fermées, le badugi et le
lowball changent le classement des mains : les refuser est la seule lecture
honnête possible aujourd'hui.
"""

_CARTE = re.compile(r"([2-9TJQKA])([shdc])")

#: Noms de position en partant du siège **qui suit le bouton**, par nombre de
#: joueurs. En heads-up le bouton EST la small blind, d'où la table à deux
#: entrées qui commence quand même par « SB ». Les noms sont ceux de
#: ``core.range_model.GTO_PRESETS`` (UTG, MP, CO, BTN, SB) pour que le
#: conseiller retrouve ses charts sans traduction ; au-delà de six joueurs le
#: format « UTG1 », « MP1 » désigne les sièges intermédiaires, qui n'ont pas
#: de chart.
_POSITIONS: dict[int, tuple[str, ...]] = {
    2: ("SB", "BB"),
    3: ("SB", "BB", "BTN"),
    4: ("SB", "BB", "UTG", "BTN"),
    5: ("SB", "BB", "UTG", "CO", "BTN"),
    6: ("SB", "BB", "UTG", "MP", "CO", "BTN"),
    7: ("SB", "BB", "UTG", "UTG1", "MP", "CO", "BTN"),
    8: ("SB", "BB", "UTG", "UTG1", "MP", "MP1", "CO", "BTN"),
    9: ("SB", "BB", "UTG", "UTG1", "UTG2", "MP", "MP1", "CO", "BTN"),
}


def _cartes(tok: str) -> tuple[str, ...]:
    """Découpe « Ac2d » → ``('Ac', '2d')``.

    Les jetons inconnus (« ???? », qu'un corpus utilise pour une main jamais
    montrée) ne rendent rien : le lecteur ne fabrique pas de carte.

    >>> _cartes("Jc3d5c")
    ('Jc', '3d', '5c')
    >>> _cartes("????")
    ()
    """
    return tuple(r + s for r, s in _CARTE.findall(tok))


def _indice_joueur(tok: str, n: int) -> int:
    """« p3 » → 2 (indice 0-based), avec contrôle de bornes."""
    if not tok.startswith("p") or not tok[1:].isdigit():
        raise PhhError(f"identifiant de joueur illisible : « {tok} ».")
    i = int(tok[1:]) - 1
    if not (0 <= i < n):
        raise PhhError(f"« {tok} » hors de la table de {n} joueurs.")
    return i


def phh_players(text: str) -> tuple[str, ...]:
    """Noms de table déclarés par le fichier, dans l'ordre p1, p2, …

    Ils sont rendus **en clair**, contrairement à ceux de :class:`ParsedHand`
    qui sont hachés par :func:`~pfs.data.hand_history.player_key`. Le corpus
    PHH est public et sous licence MIT : « Pluribus », « MrBlue »,
    « Phil Ivey » sont des étiquettes de publication, pas des données
    personnelles collectées. La fonction existe pour que les bancs puissent
    nommer les joueurs dans leurs rapports ; le chemin ``ParsedHand``, lui,
    reste haché comme partout ailleurs dans le projet.
    """
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PhhError(f"TOML illisible : {exc}") from exc
    return tuple(str(x) for x in doc.get("players", []))


def parse_phh(text: str, salt: str = "", *, hero: str | int | None = None,
              hand_id: str = "", is_tournament: bool = False) -> ParsedHand:
    """Lit une main PHH → :class:`ParsedHand`.

    Parameters
    ----------
    text : str
        Contenu du fichier ``.phh`` (TOML).
    salt : str
        Sel de :func:`~pfs.data.hand_history.player_key`, comme les autres
        parseurs.
    hero : str or int or None
        Joueur à marquer comme héros : son **nom de table** (« Pluribus »)
        ou son numéro PHH 1-based (``3`` pour ``p3``). ``None`` : aucun héros
        — PHH est un format d'observateur, il n'en désigne pas.
    hand_id : str
        Identifiant à porter dans ``ParsedHand.hand_id``. **Prioritaire sur
        la clé ``hand`` du fichier**, qui n'est unique que dans sa partie :
        le corpus Pluribus étale ses 10 000 mains sur 92 parties et n'a que
        292 numéros distincts — ``hand = 30`` revient 65 fois.
    is_tournament : bool
        Le format ne le déclare pas ; c'est donc à l'appelant de le dire.

    Returns
    -------
    ParsedHand
        Avec, en plus de ce qu'un HH de room donne : les cartes de **tous**
        les joueurs dans ``seats[i].cards``.

    Raises
    ------
    PhhError
        TOML illisible, variante hors :data:`VARIANTES_SUPPORTEES`, tableaux
        de longueurs incohérentes, action inconnue, ou **comptabilité
        incohérente** (voir ci-dessous).

    Notes
    -----
    **Comptabilité.** ``cbr X`` monte le total de la rue à ``X`` ; l'incrément
    réellement payé vaut donc ``X − (déjà misé sur la rue)``. Cette lecture
    n'est pas une convention choisie : elle est **vérifiée** quand le fichier
    porte ``finishing_stacks``. Le lecteur en déduit ce que chaque joueur a
    encaissé,

    .. math:: \\text{encaissé}_i = \\text{fin}_i - \\text{départ}_i
              + \\text{misé}_i

    et refuse la main dès que l'un de ces montants est **négatif**, ou
    **non nul chez un joueur qui s'est couché** — un couché ne gagne rien,
    jamais. C'est la seconde condition qui a du mordant : quand tout le monde
    passe, le rendu de la surmise absorbe l'écart entre les deux lectures de
    ``cbr`` et le total redevient juste par accident. Il faut une relance
    *payée* par un joueur qui se couche ensuite pour que l'erreur affleure —
    et là elle affleure toujours.

    Sur les 10 000 mains de Pluribus la vérification passe pour les 60 000
    joueurs-mains, dont 48 271 couchés (``banc_corpus_pluribus.py
    --verifier``).

    **All-in.** Une action agressive qui laisse le joueur à zéro jeton devient
    ``ActionType.ALLIN`` ; un *suivi* qui met à tapis reste ``CALL`` — même
    convention que :func:`~pfs.data.hand_history.parse_ipoker`, pour que
    ``preflop_raise`` et ``three_bet`` comptent la même chose d'un parseur à
    l'autre.

    Examples
    --------
    >>> main = parse_phh('''
    ... variant = "NT"
    ... antes = [0, 0]
    ... blinds_or_straddles = [50, 100]
    ... starting_stacks = [10000, 10000]
    ... actions = ["d dh p1 AsAh", "d dh p2 7d2c", "p1 cbr 300", "p2 f"]
    ... finishing_stacks = [10100, 9900]
    ... ''')
    >>> main.big_blind, main.pot      # 50 + 100 + 50 payés : la relance à
    (100.0, 200.0)                    # 300 n'a jamais été suivie
    >>> main.button_seat          # heads-up : la small blind EST le bouton
    1
    """
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PhhError(f"TOML illisible : {exc}") from exc

    variante = str(doc.get("variant", ""))
    if variante not in VARIANTES_SUPPORTEES:
        raise PhhError(
            f"variante « {variante or '?' } » hors domaine : ce lecteur ne "
            f"couvre que {sorted(VARIANTES_SUPPORTEES)} (hold'em). Lire une "
            "autre variante avec cette grammaire donnerait des mains "
            "silencieusement fausses.")

    depart = [float(x) for x in doc.get("starting_stacks", [])]
    n = len(depart)
    if n < 2:
        raise PhhError("moins de deux joueurs : rien à jouer.")
    blindes = [float(x) for x in doc.get("blinds_or_straddles", [0.0] * n)]
    antes = [float(x) for x in doc.get("antes", [0.0] * n)]
    if len(blindes) != n or len(antes) != n:
        raise PhhError(
            f"tableaux de longueurs incohérentes : {n} tapis, "
            f"{len(blindes)} blindes, {len(antes)} antes.")

    noms = [str(x) for x in doc.get("players", [])] or [f"p{i + 1}" for i in range(n)]
    if len(noms) != n:
        raise PhhError(f"{len(noms)} noms pour {n} tapis.")
    cles = [player_key(nom, salt) for nom in noms]

    cle_hero: str | None = None
    if isinstance(hero, int):
        cle_hero = cles[_indice_joueur(f"p{hero}", n)]
    elif isinstance(hero, str) and hero:
        if hero not in noms:
            raise PhhError(f"héros « {hero} » absent de la table {noms}.")
        cle_hero = cles[noms.index(hero)]

    bb = blindes[1] if n > 1 and blindes[1] > 0 else (
        float(doc.get("min_bet", 0.0)) or max(blindes, default=0.0))
    main = ParsedHand(
        room=Room.PHH,
        hand_id=hand_id or str(doc.get("hand", "")),
        is_real_money="currency" in doc,
        is_tournament=is_tournament,
        big_blind=bb,
        ante=max(antes) if antes else 0.0,
        table=str(doc.get("event", "")),
        # À trois joueurs et plus le bouton est le dernier de l'ordre PHH (qui
        # part de la small blind) ; en heads-up la small blind EST le bouton.
        button_seat=1 if n == 2 else n,
        hero=cle_hero,
        raw=text,
    )

    reste = list(depart)
    mise_totale = [0.0] * n         # tout ce que le joueur a mis dans la main
    mise_rue = [0.0] * n            # misé sur la rue en cours, antes exclues

    def _payer(i: int, voulu: float) -> float:
        """Débite au plus le tapis restant, et rend l'incrément réel."""
        inc = min(max(voulu, 0.0), reste[i])
        reste[i] -= inc
        mise_totale[i] += inc
        return inc

    def _rendre_non_paye() -> None:
        """Rend au dernier agresseur la part de sa mise que personne n'a payée.

        Sans ça le pot serait faux dès qu'une relance fait coucher tout le
        monde — cas le plus fréquent du corpus Pluribus (« p6 cbr 737 » puis
        cinq folds : 737 jetons sortent et reviennent). Et faux aussi sur
        Dwan-Ivey 2009, où Ivey monte à 1 067 100 alors que Dwan ne peut
        payer que 495 000 sur la rue : 572 100 lui reviennent, et le pot
        tombe de 1 681 600 à 1 109 500.

        La règle : sur la rue qui se termine, l'unique plus gros miseur
        récupère l'écart avec le deuxième. Égalité en tête → rien à rendre.
        Appelée à chaque changement de rue et en fin de main ; quand elle
        rend quelque chose, plus personne ne peut miser (celui qui suivait
        était à tapis, ou tous les autres se sont couchés), donc elle ne
        perturbe aucune décision ultérieure.
        """
        if n < 2:
            return
        rang = sorted(range(n), key=lambda i: mise_rue[i], reverse=True)
        exces = mise_rue[rang[0]] - mise_rue[rang[1]]
        if exces <= 0.0:
            return
        i0 = rang[0]
        reste[i0] += exces
        mise_totale[i0] -= exces
        mise_rue[i0] -= exces

    # ── argent mort et blindes ────────────────────────────────────────────
    for i, a in enumerate(antes):
        if a > 0:
            # L'ante n'entre PAS dans mise_rue : « monter à 300 » veut dire
            # 300 de mise, ante à part. C'est la même règle que le parseur
            # iPoker applique à sa baseline « raise to ».
            main.actions.append(
                HandAction(cles[i], Street.PREFLOP, ActionType.POST, _payer(i, a)))
    for i, b in enumerate(blindes):
        if b > 0:
            inc = _payer(i, b)
            mise_rue[i] += inc
            main.actions.append(
                HandAction(cles[i], Street.PREFLOP, ActionType.POST, inc,
                           total_bet=mise_rue[i]))
            if i >= 2:
                # Au-delà de la big blind, une entrée non nulle est un straddle.
                main.straddles.append((cles[i], inc))

    # ── déroulé ───────────────────────────────────────────────────────────
    fermees: list[tuple[str, ...]] = [() for _ in range(n)]
    board: list[str] = []
    rue = Street.PREFLOP
    # La rue préflop est déjà ouverte par la big blind : la première action
    # agressive y est une RELANCE, pas un bet — sans quoi la stat PFR
    # manquerait tous les open-raises.
    rue_ouverte = any(b > 0 for b in blindes)
    couche = [False] * n

    for brute in doc.get("actions", []):
        toks = str(brute).split()
        if not toks:
            continue

        if toks[0] == "d":
            if len(toks) < 3:
                raise PhhError(f"action croupier incomplète : « {brute} ».")
            # Les cartes sont recollées avant analyse : « d db Jc3d5c » et
            # « d db Jc 3d 5c » désignent le même flop, et ne lire que le
            # premier jeton perdrait deux cartes SANS rien signaler.
            if toks[1] == "dh":
                i = _indice_joueur(toks[2], n)
                fermees[i] = _cartes("".join(toks[3:]))
            elif toks[1] == "db":
                _rendre_non_paye()
                board.extend(_cartes("".join(toks[2:])))
                rue = {3: Street.FLOP, 4: Street.TURN,
                       5: Street.RIVER}.get(len(board), rue)
                mise_rue = [0.0] * n
                rue_ouverte = False
            else:
                raise PhhError(
                    f"action croupier « {toks[1]} » inconnue en hold'em "
                    f"(« {brute} ») : le lecteur préfère s'arrêter plutôt que "
                    "de deviner.")
            continue

        i = _indice_joueur(toks[0], n)
        verbe = toks[1] if len(toks) > 1 else ""

        if verbe == "sm":
            # Abat ou jette : aucune information neuve, « d dh » a déjà donné
            # toutes les cartes fermées.
            continue
        if verbe == "f":
            couche[i] = True
            main.actions.append(HandAction(cles[i], rue, ActionType.FOLD, 0.0))
            continue
        if verbe == "cc":
            besoin = max(mise_rue) - mise_rue[i]
            inc = _payer(i, besoin)
            mise_rue[i] += inc
            genre = ActionType.CHECK if besoin <= 0 else ActionType.CALL
            main.actions.append(
                HandAction(cles[i], rue, genre, inc, total_bet=mise_rue[i]))
            continue
        if verbe == "cbr":
            if len(toks) < 3:
                raise PhhError(f"« cbr » sans montant : « {brute} ».")
            try:
                monte_a = float(toks[2])
            except ValueError as exc:
                raise PhhError(f"montant illisible dans « {brute} ».") from exc
            inc = _payer(i, monte_a - mise_rue[i])
            if inc <= 0.0:
                raise PhhError(
                    f"« {brute} » ne monte rien (déjà {mise_rue[i]:g} sur la "
                    "rue) : lecture du fichier incohérente.")
            mise_rue[i] += inc
            genre = ActionType.RAISE if rue_ouverte else ActionType.BET
            if reste[i] <= 0.0:
                genre = ActionType.ALLIN
            rue_ouverte = True
            main.actions.append(
                HandAction(cles[i], rue, genre, inc, total_bet=mise_rue[i]))
            continue

        raise PhhError(
            f"action joueur « {verbe} » inconnue en hold'em (« {brute} ») : "
            "le lecteur préfère s'arrêter plutôt que de deviner.")

    _rendre_non_paye()
    main.board = tuple(board)
    main.seats = [
        PlayerSeat(seat=i + 1, player=cles[i], stack=depart[i],
                   is_hero=(cles[i] == cle_hero), cards=fermees[i])
        for i in range(n)
    ]
    if cle_hero is not None:
        main.hero_cards = fermees[cles.index(cle_hero)]

    # ── comptabilité : les tapis de fin valident la lecture de « cbr » ────
    fin = [float(x) for x in doc.get("finishing_stacks", [])]
    if fin:
        if len(fin) != n:
            raise PhhError(f"{len(fin)} tapis de fin pour {n} joueurs.")
        for i in range(n):
            encaisse = fin[i] - depart[i] + mise_totale[i]
            if encaisse < -1e-6 or (couche[i] and abs(encaisse) > 1e-6):
                raise PhhError(
                    f"comptabilité incohérente pour {noms[i]} : tapis "
                    f"{depart[i]:g} → {fin[i]:g} alors que le déroulé lu lui "
                    f"fait miser {mise_totale[i]:g}, soit un encaissement de "
                    f"{encaisse:g}"
                    + (" alors qu'il s'est COUCHÉ." if couche[i] else
                       " (négatif).")
                    + " Le fichier ou la lecture des montants est faux — "
                    "mieux vaut refuser la main que la compter.")
            if encaisse > 1e-6:
                main.winners[cles[i]] = encaisse
    main.pot = sum(mise_totale)
    return main


def parse_phh_file(path: str | Path, salt: str = "", **kw) -> ParsedHand:
    """Lit un fichier ``.phh``. L'identifiant par défaut est ``dossier/nom``.

    Les mains de Pluribus sont numérotées par partie (``100/30.phh``) : la
    clé ``hand = 30`` du fichier revient dans 65 parties, d'où le préfixe du
    dossier — sans lui, deux mains différentes porteraient le même
    identifiant et un rapport agrégé par main serait faux.
    """
    p = Path(path)
    kw.setdefault("hand_id", f"{p.parent.name}/{p.stem}")
    return parse_phh(p.read_text(encoding="utf-8"), salt, **kw)


def iter_phh_files(root: str | Path) -> Iterator[Path]:
    """Tous les ``.phh`` sous ``root``, dans un ordre stable (donc rejouable).

    Le tri est fait sur les composants du chemin, avec les segments
    entièrement numériques triés comme des nombres : ``100/2`` vient donc
    avant ``100/10``, et un banc tronqué à N mains lit toujours les mêmes.
    """
    def cle(p: Path) -> tuple:
        parts = p.relative_to(Path(root)).parts
        return tuple((0, int(x), "") if x.isdigit() else (1, 0, x)
                     for x in (*parts[:-1], p.stem))

    return iter(sorted(Path(root).rglob("*.phh"), key=cle))


def positions_par_siege(hand: ParsedHand) -> dict[int, str]:
    """Siège → nom de position, déduit de ``button_seat``.

    La numérotation part du siège **qui suit** le bouton (la small blind) et
    remonte jusqu'au bouton, selon :data:`_POSITIONS`. **Sauf en heads-up**,
    où le bouton EST la small blind : à deux, partir du siège suivant
    intervertirait les deux joueurs, et le conseiller appliquerait au bouton
    la chart de la big blind. Au-delà de neuf joueurs, ou si ``button_seat``
    ne désigne aucun siège, la fonction rend un dictionnaire vide plutôt
    qu'un nom inventé : un mauvais nom de position ferait chercher une chart
    qui ne correspond à rien.

    >>> from pfs.data.hand_history import ParsedHand, PlayerSeat, Room
    >>> h = ParsedHand(Room.PHH, "x", False, False, 100, 0, "", 6,
    ...                seats=[PlayerSeat(i, f"j{i}", 100) for i in range(1, 7)])
    >>> positions_par_siege(h)[1], positions_par_siege(h)[6]
    ('SB', 'BTN')
    """
    sieges = sorted(s.seat for s in hand.seats)
    n = len(sieges)
    noms = _POSITIONS.get(n)
    if noms is None or hand.button_seat not in sieges:
        return {}
    bouton = sieges.index(hand.button_seat)
    depart = bouton if n == 2 else (bouton + 1) % n
    return {sieges[(depart + k) % n]: noms[k] for k in range(n)}


@dataclass(frozen=True, slots=True)
class Decision:
    """Un point de décision du rejeu, avec l'état de la table AVANT l'action.

    Attributes
    ----------
    hand_id : str
        Main d'où vient la décision.
    ordre : int
        Rang de l'action volontaire dans la main (0 = première).
    street : Street
        Rue de la décision.
    player : str
        Clé hachée du joueur qui parle.
    seat : int
        Son siège.
    position : str
        Sa position (« BTN », « SB »…), vide si indéterminable.
    cards : tuple of str
        Ses cartes fermées.
    board : tuple of str
        Cartes communes **visibles à cet instant** — pas celles de la fin de
        main.
    pot : float
        Jetons déjà au pot, antes et blindes comprises, avant l'action.
    pot_gagnable : float
        Le même pot **plafonné à ce que ce joueur peut réellement gagner** :
        la part de la mise adverse qui dépasse ce qu'il peut couvrir lui
        reviendra. C'est ce montant-là, pas ``pot``, qui donne les bonnes
        cotes. Les deux ne diffèrent que dans les spots « à tapis pour
        moins » — mais l'écart y est énorme : sur le dernier call de
        Dwan-Ivey 2009, ``pot`` vaut 1 419 200 et ``pot_gagnable`` 847 100.
    to_call : float
        À ajouter pour suivre (0 = personne n'a misé sur la rue), plafonné au
        tapis restant.
    stack : float
        Jetons restant DERRIÈRE le joueur avant l'action.
    stack_effectif : float
        ``min(stack, plus gros tapis restant parmi les adversaires encore en
        jeu)`` — ce qui peut encore être misé face à quelqu'un.
    engage : float
        Ce que le joueur a déjà mis dans la main.
    actifs : int
        Joueurs non couchés, lui compris.
    relances_avant : int
        Mises et relances déjà faites sur cette rue. Zéro préflop signifie
        « personne n'a ouvert », le seul cas où la chart d'ouverture du
        conseiller a un sens.
    action : ActionType
        Ce qui a été joué.
    montant : float
        Incrément réellement payé.
    """

    hand_id: str
    ordre: int
    street: Street
    player: str
    seat: int
    position: str
    cards: tuple[str, ...]
    board: tuple[str, ...]
    pot: float
    pot_gagnable: float
    to_call: float
    stack: float
    stack_effectif: float
    engage: float
    actifs: int
    relances_avant: int
    action: ActionType
    montant: float


_AGRESSIVES = (ActionType.BET, ActionType.RAISE, ActionType.ALLIN)


def iter_decisions(hand: ParsedHand) -> Iterator[Decision]:
    """Rejoue une main et rend l'état de la table avant chaque action.

    Conçu pour les mains produites par :func:`parse_phh`, dont les incréments
    (``HandAction.amount``) sont exacts et dont les tapis de départ sont ceux
    de ``seats``. Les parseurs texte (Winamax, PokerStars) n'offrent pas cette
    garantie sur le montant d'une relance : les y appliquer donnerait des pots
    faux. C'est une limite de leur format d'origine, pas de cette fonction.

    Les ``POST`` (antes, blindes, straddles) sont appliqués mais ne produisent
    pas de :class:`Decision` : personne n'y décide rien.
    """
    tapis = {s.player: s.stack for s in hand.seats}
    siege = {s.player: s.seat for s in hand.seats}
    cartes = {s.player: s.cards for s in hand.seats}
    pos = positions_par_siege(hand)

    reste = dict(tapis)
    engage = {p: 0.0 for p in tapis}
    mise_rue = {p: 0.0 for p in tapis}
    couche: set[str] = set()
    rue = Street.PREFLOP
    relances = 0
    ordre = 0

    for a in hand.actions:
        if a.street is not rue:
            rue = a.street
            mise_rue = {p: 0.0 for p in tapis}
            relances = 0

        if a.action is ActionType.POST:
            reste[a.player] = reste.get(a.player, 0.0) - a.amount
            engage[a.player] = engage.get(a.player, 0.0) + a.amount
            # Une ante n'est pas une mise vivante : seules les blindes, qui
            # portent un total_bet, entrent dans la baseline de la rue.
            if a.total_bet > 0.0:
                mise_rue[a.player] = a.total_bet
            continue

        adversaires = [p for p in tapis if p != a.player and p not in couche]
        eff = min(reste.get(a.player, 0.0),
                  max((reste.get(p, 0.0) for p in adversaires), default=0.0))
        du = max(mise_rue.values(), default=0.0) - mise_rue.get(a.player, 0.0)
        a_payer = min(max(du, 0.0), reste.get(a.player, 0.0))
        board_vu = {Street.PREFLOP: 0, Street.FLOP: 3,
                    Street.TURN: 4, Street.RIVER: 5}.get(rue, 5)
        # Plafond de gain : après un suivi, ce joueur aura engagé
        # `engage + a_payer` ; il ne peut disputer à chacun que ce montant.
        plafond = engage.get(a.player, 0.0) + a_payer

        yield Decision(
            hand_id=hand.hand_id,
            ordre=ordre,
            street=rue,
            player=a.player,
            seat=siege.get(a.player, 0),
            position=pos.get(siege.get(a.player, 0), ""),
            cards=cartes.get(a.player, ()),
            board=hand.board[:board_vu],
            pot=sum(engage.values()),
            pot_gagnable=sum(min(v, plafond) for v in engage.values()),
            to_call=a_payer,
            stack=reste.get(a.player, 0.0),
            stack_effectif=eff,
            engage=engage.get(a.player, 0.0),
            actifs=len(adversaires) + 1,
            relances_avant=relances,
            action=a.action,
            montant=a.amount,
        )
        ordre += 1

        if a.action is ActionType.FOLD:
            couche.add(a.player)
        else:
            reste[a.player] = reste.get(a.player, 0.0) - a.amount
            engage[a.player] = engage.get(a.player, 0.0) + a.amount
            mise_rue[a.player] = mise_rue.get(a.player, 0.0) + a.amount
            if a.action in _AGRESSIVES:
                relances += 1
