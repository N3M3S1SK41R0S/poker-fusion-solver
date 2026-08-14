"""Emplacements connus des historiques de mains PMU sur cette machine.

Cette détection vivait dans le script ``recuperer_mains.py``, hors du paquet :
le serveur ne pouvait donc pas préremplir le champ « dossier d'historiques »
sans recopier la logique. Elle est ici pour être importée par les deux —
le script en ligne de commande et la route ``/api/emplacements`` — sans
duplication.

Piège documenté sur cette machine : le Python du Microsoft Store VIRTUALISE
les ÉCRITURES sous ``Packages/.../LocalCache/Local`` — mais il ne touche ni à
la variable d'environnement ``%LOCALAPPDATA%`` ni à la LECTURE des dossiers
qui existent au vrai chemin, et ce module ne fait que lire. Le serveur trouve
donc bien les historiques au vrai ``%LOCALAPPDATA%``.

Les chemins sont dérivés de l'environnement plutôt que codés en dur, pour que
la détection fonctionne sur n'importe quel compte Windows. Ce sont des
fonctions, pas des constantes de module : l'environnement peut changer entre
deux appels (les tests s'en servent), et une constante figée au premier
import mentirait ensuite.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["appdata_local", "dossiers_connus", "dossiers_detectes"]

#: Clients PMU connus, argent fictif puis argent réel. Les historiques vivent
#: sous ``<client>/data/<pseudo>/History`` ; on renvoie ``<client>/data`` et
#: la lecture récursive fait le reste.
_CLIENTS: tuple[str, ...] = ("PMU PLAY 100% Poker", "PMU Poker")


def appdata_local() -> Path:
    """``%LOCALAPPDATA%`` réel, avec repli sur ``~/AppData/Local``.

    Returns
    -------
    Path
        Le dossier de données local de l'utilisateur courant.

    Notes
    -----
    Le repli ne couvre que le cas d'un environnement lancé sans
    ``LOCALAPPDATA`` (service, shell minimal) ; il vaut mieux qu'un chemin
    d'un autre utilisateur codé en dur, qui ne trouverait jamais rien.
    """
    env = os.environ.get("LOCALAPPDATA", "").strip()
    return Path(env) if env else Path.home() / "AppData" / "Local"


def dossiers_connus() -> tuple[str, ...]:
    """Emplacements candidats des historiques PMU, existants ou non.

    Returns
    -------
    tuple of str
        Un chemin par client connu, dans l'ordre de ``_CLIENTS``. Aucune
        vérification d'existence : c'est la liste des endroits où CHERCHER,
        pas de ce qui a été TROUVÉ (voir ``dossiers_detectes``).
    """
    return tuple(str(appdata_local() / client / "data") for client in _CLIENTS)


def dossiers_detectes() -> tuple[str, ...]:
    """Ceux des emplacements connus qui existent réellement sur le disque.

    Returns
    -------
    tuple of str
        Sous-ensemble ordonné de ``dossiers_connus()``. Vide si aucun client
        PMU n'est installé — l'appelant doit alors demander le chemin plutôt
        que d'en inventer un.
    """
    return tuple(d for d in dossiers_connus() if Path(d).is_dir())
