"""Un second serveur ne doit JAMAIS pouvoir se lier au même port.

Ce test protège contre un défaut qui a coûté des heures de diagnostic.
`HTTPServer` active `allow_reuse_address` par défaut ; sur Windows, cette
option laisse un second processus se lier à un port déjà en écoute. Les deux
serveurs répondent alors en alternance, et après une modification du code
les routes nouvellement ajoutées renvoient « route inconnue » une fois sur
deux — le défaut paraît venir du routeur alors qu'il vient du démarrage.
"""

from __future__ import annotations

import socket

import pytest

from pfs.app.server import create_server


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_le_port_ne_peut_pas_etre_pris_deux_fois() -> None:
    """Le second `create_server` sur le même port échoue franchement."""
    port = _port_libre()
    premier, _ = create_server(port, token="jeton-de-test")
    try:
        with pytest.raises(OSError):
            second, _ = create_server(port, token="jeton-de-test")
            second.server_close()   # au cas où le garde-fou aurait sauté
    finally:
        premier.server_close()


def test_le_port_est_rendu_a_la_fermeture() -> None:
    """Après fermeture, le port se réutilise sans attendre.

    Le garde-fou doit empêcher la COHABITATION, pas rendre le redémarrage
    impossible : sans cette garantie, chaque relance demanderait d'attendre
    la fin du TIME_WAIT.
    """
    port = _port_libre()
    premier, _ = create_server(port, token="jeton-de-test")
    premier.server_close()

    second, _ = create_server(port, token="jeton-de-test")
    second.server_close()


def test_reuse_address_est_bien_desactive() -> None:
    """L'intention est inscrite dans la classe, pas seulement dans un test."""
    port = _port_libre()
    srv, _ = create_server(port, token="jeton-de-test")
    try:
        assert srv.allow_reuse_address is False
    finally:
        srv.server_close()
