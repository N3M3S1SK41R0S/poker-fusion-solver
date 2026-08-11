"""Le régime ICM doit être atteignable depuis l'API, pas seulement en Python.

`advise()` bascule en régime ICM sur la présence de ``stacks`` et
``payouts``. La route ``/api/advise`` construisait le `Spot` **sans ces deux
champs** : le régime ICM, pourtant implémenté, documenté et couvert par
`test_advisor_icm.py`, était donc inatteignable depuis l'interface.

Ce n'est pas une fonctionnalité manquante, c'est un **verdict faux**. Sur un
spot de bulle mesuré, la route rendait « OUVRIR (relance) » là où le régime
ICM rend « FOLD » — l'inverse, sur exactement le genre de main qui coûte un
tournoi.

Leçon de méthode : les tests du régime ICM appelaient `advise()` en direct.
Ils passaient tous, et ne pouvaient rien dire du chemin réellement emprunté
par l'utilisateur. Un test qui court-circuite l'interface ne prouve rien sur
l'interface.
"""

from __future__ import annotations

from pfs.app.server import API

#: Le spot de bulle de référence : 10 bb en SB, trois joueurs, structure de
#: gains plate. Le chipEV veut ouvrir, l'ICM veut coucher.
SPOT = {
    "hero": "Ah 7d", "big_blind": 1, "stack": 10, "players": 3,
    "position": "SB", "pot": 1.5, "bet": 0,
}


def test_sans_tapis_le_regime_reste_chipev() -> None:
    """Comportement inchangé quand on ne décrit pas un tournoi."""
    r = API.advise(dict(SPOT))
    assert r["bubble"] is None
    assert "ICM" not in r["regime"]


def test_avec_tapis_et_gains_le_regime_devient_icm() -> None:
    """C'est ce que la route jetait."""
    r = API.advise(dict(SPOT, stacks="10, 40, 1", payouts="50, 30, 20"))
    assert "ICM" in r["regime"], (
        f"régime « {r['regime']} » : les tapis n'ont pas été transmis")
    assert r["bubble"] is not None and r["bubble"] > 1.0
    assert r["ev_icm"] is not None


def test_le_verdict_est_bien_inverse_par_l_icm() -> None:
    """Le cœur du défaut : la pression de bulle change la décision.

    Si ces deux verdicts redevenaient identiques, ce serait soit que la
    transmission a re-cassé, soit que le régime ICM a changé — dans les deux
    cas il faut regarder, pas ajuster le test.
    """
    cash = API.advise(dict(SPOT))
    icm = API.advise(dict(SPOT, stacks="10, 40, 1", payouts="50, 30, 20"))
    assert cash["action"] != icm["action"], (
        f"même verdict avec et sans ICM ({cash['action']}) : la bascule "
        "n'opère plus")
    assert icm["action"].upper().startswith("FOLD")


def test_les_tapis_s_acceptent_en_liste_ou_en_texte() -> None:
    """L'interface envoie du texte, un script enverra une liste."""
    texte = API.advise(dict(SPOT, stacks="10, 40, 1", payouts="50, 30, 20"))
    liste = API.advise(dict(SPOT, stacks=[10, 40, 1], payouts=[50, 30, 20]))
    assert texte["action"] == liste["action"]
    assert texte["bubble"] == liste["bubble"]


def test_la_reponse_expose_de_quoi_juger_le_regime() -> None:
    """Sans `bubble` ni `ev_icm` renvoyés, l'interface ne peut rien montrer."""
    r = API.advise(dict(SPOT, stacks="10, 40, 1", payouts="50, 30, 20"))
    for champ in ("bubble", "ev_icm", "regime", "action", "assumptions"):
        assert champ in r, f"champ « {champ} » absent de la réponse"
