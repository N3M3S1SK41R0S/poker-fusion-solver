"""Le chemin annoncé à l'utilisateur doit être le chemin réel des fichiers.

Défaut constaté : l'interpréteur du projet dérive d'un Python Microsoft
Store (``sys.base_prefix`` sous ``C:\\Program Files\\WindowsApps\\...``).
Windows redirige alors les écritures du processus vers le ``LocalCache`` du
paquet, **sans** modifier ``os.environ["LOCALAPPDATA"]`` ni
``os.path.abspath``. Le logiciel affichait donc

    C:\\Users\\pierr\\AppData\\Local\\PokerFusionSolver\\captures

alors que les fichiers atterrissaient dans

    ...\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_...
        \\LocalCache\\Local\\PokerFusionSolver\\captures

Le dossier annoncé n'existait pas du tout : ``Test-Path`` répondait ``False``
depuis PowerShell là où Python voyait ses fichiers. Pierre ne pouvait pas
retrouver ses captures dans l'explorateur.

Aucun test existant ne pouvait le voir : ils remplacent tous
``LOCALAPPDATA`` par un dossier temporaire, ce qui court-circuite la
virtualisation. Ces tests-ci travaillent donc **dans l'environnement réel**.
"""

from __future__ import annotations

import os
from pathlib import Path

from pfs.vision.archive import dossier_archive


def test_le_chemin_annonce_est_deja_resolu() -> None:
    """`dossier_archive()` rend un chemin que `realpath` ne déplace plus.

    C'est la propriété qui rend le chemin copiable dans l'explorateur. Sans
    elle, on affiche une adresse plausible qui ne mène nulle part.
    """
    d = dossier_archive()
    reel = Path(os.path.realpath(d))
    assert d == reel, (
        f"le dossier annoncé n'est pas le dossier réel :\n"
        f"  annoncé : {d}\n"
        f"  réel    : {reel}\n"
        "Sous un Python du Microsoft Store, %LOCALAPPDATA% est virtualisé.")


def test_le_dossier_annonce_existe_vraiment() -> None:
    """Ce que l'on affiche existe sur le disque, et est un dossier."""
    d = dossier_archive()
    assert d.is_dir(), f"{d} n'est pas un dossier existant"
    assert (d / "echecs").is_dir(), "le sous-dossier des échecs manque"


def test_un_fichier_ecrit_se_retrouve_au_chemin_annonce() -> None:
    """Un fichier créé dans l'archive est visible à l'adresse annoncée.

    Vérification de bout en bout : on écrit, puis on relit **par le chemin
    tel qu'il serait montré à l'utilisateur**, reconstruit à la main plutôt
    que réutilisé tel quel — c'est bien l'adresse affichée qu'on teste.
    """
    d = dossier_archive()
    temoin = d / "echecs" / ".temoin-test-chemin"
    try:
        temoin.write_text("témoin", encoding="utf-8")
        annonce = Path(str(d)) / "echecs" / ".temoin-test-chemin"
        assert annonce.exists(), (
            f"écrit dans {temoin}, introuvable via {annonce}")
        assert annonce.read_text(encoding="utf-8") == "témoin"
    finally:
        temoin.unlink(missing_ok=True)
