"""Lexique — tout le vocabulaire du poker et du solveur, expliqué.

Chaque terme employé par le logiciel doit être explicable sans renvoyer à
un autre jargon. Une définition qui exige d'en lire trois autres n'apprend
rien : ici chaque entrée tient debout seule, et donne quand elle existe la
raison pour laquelle la notion compte à la table.

Le lexique est une DONNÉE, pas du texte d'interface : l'application le sert
tel quel (``/api/lexique``), et les rapports peuvent y renvoyer.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Terme", "LEXIQUE", "chercher", "definir"]


@dataclass(frozen=True, slots=True)
class Terme:
    """Une entrée du lexique."""

    nom: str
    categorie: str
    definition: str
    pourquoi: str = ""
    aussi: tuple[str, ...] = ()


def _t(nom, categorie, definition, pourquoi="", aussi=()):
    return Terme(nom, categorie, definition, pourquoi, tuple(aussi))


LEXIQUE: tuple[Terme, ...] = (
    # ── les bases ────────────────────────────────────────────────────────
    _t("bb (big blind)", "bases",
       "L'unité de compte du poker : la grosse blinde. Un tapis de 400 jetons "
       "à des blindes 10/20 vaut 20 bb.",
       "Compter en bb rend les situations comparables entre niveaux de "
       "blindes : « 10 bb » veut dire la même chose au niveau 1 et au niveau 9.",
       ("tapis effectif",)),
    _t("tapis effectif", "bases",
       "Le plus petit des deux tapis en jeu. Si tu as 50 bb et ton adversaire "
       "12 bb, vous ne jouez que 12 bb.",
       "C'est la seule profondeur qui compte : on ne peut pas gagner plus que "
       "ce que l'adversaire peut perdre.",
       ("bb (big blind)",)),
    _t("position", "bases",
       "L'ordre de parole. Être « en position » signifie parler APRÈS "
       "l'adversaire sur toutes les rues suivantes.",
       "Parler en dernier, c'est décider en sachant ce que l'autre a fait. "
       "C'est l'avantage structurel le plus rentable du jeu."),
    _t("préflop / flop / turn / river", "bases",
       "Les quatre tours d'enchères : avant toute carte commune (préflop), "
       "puis après 3 cartes (flop), une 4ᵉ (turn), une 5ᵉ (river)."),
    _t("board", "bases",
       "Les cartes communes posées au centre, partagées par tous les joueurs."),
    _t("assorti / dépareillé (suited / offsuit)", "bases",
       "Deux cartes de la même couleur (assorties, notées « s ») ou non "
       "(dépareillées, notées « o »). « A5s » = as et 5 assortis.",
       "Une main assortie vaut nettement plus qu'elle n'en a l'air : la "
       "possibilité de couleur ajoute de l'équité et de la jouabilité."),
    _t("SPR (stack-to-pot ratio)", "bases",
       "Rapport entre le tapis restant et le pot. SPR 2 = il reste deux fois "
       "le pot derrière.",
       "Il dicte l'engagement : à SPR bas, une top paire suffit souvent à "
       "jouer son tapis ; à SPR élevé, il faut bien plus."),

    # ── les actions ──────────────────────────────────────────────────────
    _t("limp", "actions",
       "Entrer dans le coup en payant simplement la grosse blinde, sans "
       "relancer.",
       "En tapis court, c'est une action DOMINÉE : le modèle push/fold montre "
       "qu'il vaut mieux pousser ou passer. Le logiciel te les signale."),
    _t("jam / shove / tapis", "actions",
       "Miser tout son tapis d'un coup.",
       "En dessous de ~25 bb, c'est souvent la seule bonne mise : relancer "
       "petit t'engage sans te laisser de porte de sortie."),
    _t("c-bet (continuation bet)", "actions",
       "Mise au flop par celui qui avait relancé préflop.",
       "C'est l'action la plus fréquente du poker moderne : l'initiative "
       "préflop se prolonge, que le flop ait aidé ou non."),
    _t("3-bet", "actions",
       "La deuxième relance préflop : quelqu'un ouvre, tu re-relances.",
       "Un 3-bet trop rare (sous 8 %) signale un jeu passif, facilement "
       "volé par les adversaires attentifs."),
    _t("bluff-catch", "actions",
       "Payer avec une main qui ne bat que les bluffs — elle perd contre "
       "toute main de valeur.",
       "La décision ne dépend alors pas de la force de ta main mais de la "
       "FRÉQUENCE de bluff de l'adversaire, comparée aux cotes du pot."),

    # ── les mesures ──────────────────────────────────────────────────────
    _t("équité", "mesures",
       "Ta part du pot en pourcentage, si la main allait à l'abattage sans "
       "aucune mise supplémentaire. AA contre KK préflop : 81,5 %.",
       "C'est la brique de toute décision : on la compare à ce que coûte de "
       "continuer.",
       ("cotes du pot", "équité requise")),
    _t("cotes du pot / équité requise", "mesures",
       "Le pourcentage minimum d'équité qui rend un call rentable. Payer 75 "
       "dans un pot de 100 exige 75/(100+75+75) = 30 %.",
       "C'est le seuil de bascule d'une décision : au-dessus tu paies, en "
       "dessous tu passes. Le logiciel te le donne à chaque verdict.",
       ("équité",)),
    _t("EV (espérance de gain)", "mesures",
       "Le gain moyen d'une décision si on la répétait un très grand nombre "
       "de fois. Exprimée en bb.",
       "Une décision se juge à son EV, jamais à son résultat : perdre un "
       "coup à +3 bb d'EV reste la bonne décision."),
    _t("MDF (fréquence de défense minimale)", "mesures",
       "La part de ta range que tu dois continuer à jouer face à une mise "
       "pour ne pas être exploitable par le bluff. Face à une mise de b dans "
       "un pot P : 1 − b/(P+b).",
       "Si tu te couches plus souvent que la MDF, l'adversaire gagne en "
       "misant n'importe quoi."),
    _t("VPIP", "mesures",
       "Part des mains où tu mets volontairement de l'argent préflop "
       "(les blindes ne comptent pas).",
       "L'indicateur n°1 de sélection. Repère MESURÉ, pas fourchette de "
       "manuel : sur les 10 000 mains publiques de Pluribus restreintes aux "
       "positions d'une table à trois, l'équilibre joue 32,2 % (30 000 "
       "mains-joueurs) — et c'est une borne basse, à trois on ouvre plus "
       "large qu'au bouton d'une table à six. Le double de ce chiffre est "
       "la fuite la plus coûteuse et la plus fréquente."),
    _t("PFR", "mesures",
       "Part des mains où tu RELANCES préflop.",
       "Comparé au VPIP, il dit si tu prends l'initiative ou si tu suis. "
       "L'écart VPIP−PFR de Pluribus vaut 16,5 points sur les positions "
       "d'une table à trois, 8,8 sur les six : au-delà, tu suis au lieu de "
       "relancer."),
    _t("WTSD", "mesures",
       "Part des mains vues au flop qui vont jusqu'à l'abattage.",
       "Attention au dénominateur : celui du logiciel compte tous les "
       "joueurs assis quand un flop tombe, couchés préflop compris. Il "
       "baisse donc mécaniquement quand la table s'agrandit, et un WTSD de "
       "table courte ne se compare PAS à un repère 6-max."),
    _t("fold-to-cbet", "mesures",
       "Fréquence à laquelle tu te couches face à une mise de continuation.",
       "Au-delà de ~60 %, tu es exploitable : miser contre toi devient "
       "rentable quelles que soient les cartes."),
    _t("variance", "mesures",
       "L'écart entre ce que l'équité te promettait et ce que tu as "
       "réellement encaissé.",
       "La séparer du niveau de jeu est indispensable : le logiciel calcule "
       "ton équité AU MOMENT où les jetons entrent, pas d'après le résultat.",
       ("all-in adjusted",)),
    _t("all-in adjusted", "mesures",
       "Résultat recalculé en remplaçant l'issue réelle des tapis par ton "
       "équité au moment où les jetons sont entrés.",
       "C'est la mesure qui répond à « ai-je mal joué, ou mal fini ? »."),

    # ── ranges et solveur ────────────────────────────────────────────────
    _t("range", "solveur",
       "L'ensemble des mains qu'un joueur peut avoir dans une situation "
       "donnée, chacune avec sa fréquence.",
       "On ne joue jamais contre une main précise mais contre une "
       "distribution : c'est le cœur du raisonnement moderne."),
    _t("GTO (équilibre de Nash)", "solveur",
       "Une stratégie que personne ne peut exploiter : même en la montrant à "
       "l'adversaire, il ne peut pas faire mieux que l'égalité.",
       "C'est un maximin — une garantie de ne pas perdre, pas une promesse "
       "de gagner le plus. Contre un adversaire faible, s'en écarter "
       "rapporte davantage.",
       ("exploitation",)),
    _t("exploitation", "solveur",
       "S'écarter volontairement de l'équilibre pour punir un défaut précis "
       "de l'adversaire.",
       "Plus rentable que le GTO contre un joueur imparfait, mais te rend "
       "toi-même exploitable — d'où l'idée de borner le risque pris."),
    _t("exploitabilité", "solveur",
       "Ce qu'un adversaire parfait gagnerait contre ta stratégie, en bb "
       "pour 100 mains. Zéro = équilibre exact.",
       "C'est la mesure de qualité d'une solution de solveur."),
    _t("push/fold", "solveur",
       "Le jeu simplifié des tapis courts : la seule question préflop est "
       "« pousser tout son tapis, ou passer ? ».",
       "Dans cette zone, l'équilibre se calcule EXACTEMENT. C'est pour ça "
       "que le logiciel y annonce des verdicts « certains »."),
    _t("nodelock", "solveur",
       "Figer la stratégie de l'adversaire à un endroit de l'arbre, puis "
       "re-résoudre le reste contre cette contrainte.",
       "C'est la façon d'étudier « et s'il jouait comme ÇA ? »."),
    _t("blocker", "solveur",
       "Une carte de ta main qui retire des combinaisons de la range "
       "adverse. Avoir l'as de pique retire ses couleurs à l'as de pique.",
       "Un blocker change les fréquences réelles, donc les bonnes décisions "
       "— surtout à la river."),
    _t("EQR (equity realization)", "solveur",
       "La part de ton équité brute que tu encaisses réellement, une fois "
       "les mises jouées. Une main hors de position en réalise moins.",
       "Explique pourquoi deux mains de même équité ne valent pas pareil."),

    # ── tournoi ──────────────────────────────────────────────────────────
    _t("ICM", "tournoi",
       "Modèle qui convertit des JETONS en argent réel, selon la grille de "
       "gains et les tapis de tous les joueurs.",
       "En tournoi, les jetons ne sont pas linéaires : les derniers jetons "
       "que tu risques valent plus que les premiers que tu gagnes. Ignorer "
       "l'ICM est l'erreur la plus chère près des places payées."),
    _t("bubble factor", "tournoi",
       "De combien un affrontement te coûte plus qu'il ne te rapporte, en "
       "monnaie ICM. Un facteur 2 signifie que perdre coûte deux fois ce "
       "que gagner rapporte.",
       "Il dicte à quel point resserrer : plus il est élevé, moins on prend "
       "de risques.",
       ("ICM",)),
    _t("prime de risque", "tournoi",
       "L'équité supplémentaire exigée par l'ICM au-delà des simples cotes "
       "du pot.",
       "C'est la traduction concrète du bubble factor en seuil de décision."),
    _t("PKO / bounty", "tournoi",
       "Format où éliminer un joueur rapporte immédiatement une prime, "
       "généralement moitié encaissée, moitié ajoutée à ta propre tête."),
    _t("FGS (future game simulation)", "tournoi",
       "Raffinement de l'ICM qui simule les tours de blindes à venir au lieu "
       "de figer la situation."),

    # ── lecture d'adversaire ─────────────────────────────────────────────
    _t("filtre particulaire", "adversaire",
       "Méthode qui maintient un nuage d'hypothèses sur le style de "
       "l'adversaire et les repondère à chaque action observée.",
       "C'est ce qui permet d'apprendre un joueur EN DIRECT plutôt que de "
       "supposer qu'il joue comme la moyenne."),
    _t("prior / rétrécissement", "adversaire",
       "Le prior est ce qu'on croit avant d'observer ; le rétrécissement "
       "tire une estimation bruitée vers cette croyance.",
       "Sur 20 mains, une fréquence observée ne vaut rien seule. Le "
       "rétrécissement évite de conclure trop vite — et c'est mesuré : sans "
       "lui, le modèle fait PIRE que le prior."),
    _t("ESS (taille d'échantillon efficace)", "adversaire",
       "Combien d'hypothèses restent réellement vivantes dans le filtre.",
       "Quand elle s'effondre, la lecture n'est plus fiable et le logiciel "
       "revient à la range de référence."),
    _t("λ (lambda) — confiance", "adversaire",
       "Le curseur entre jouer l'équilibre et exploiter la lecture, dérivé "
       "de l'incertitude sur cette lecture.",
       "On n'exploite qu'à hauteur de ce qu'on sait vraiment."),
)


def chercher(motif: str) -> list[Terme]:
    """Entrées dont le nom ou la définition contient ``motif``.

    La recherche ignore la casse et les accents ne sont pas normalisés :
    on cherche du vocabulaire, pas du texte libre.
    """
    m = (motif or "").strip().lower()
    if not m:
        return list(LEXIQUE)
    return [t for t in LEXIQUE
            if m in t.nom.lower() or m in t.definition.lower()
            or m in t.categorie.lower()]


def definir(nom: str) -> Terme | None:
    """Entrée dont le nom correspond, au plus proche."""
    n = (nom or "").strip().lower()
    for t in LEXIQUE:
        if t.nom.lower() == n:
            return t
    for t in LEXIQUE:
        if n and n in t.nom.lower():
            return t
    return None
