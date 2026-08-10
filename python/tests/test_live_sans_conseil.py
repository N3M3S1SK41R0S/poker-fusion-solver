"""La calibration en direct lit l'écran, elle ne conseille jamais.

Pourquoi un test et pas seulement une intention
-----------------------------------------------
`pfs.vision.live` capture la fenêtre d'un client poker pendant que la partie
est en cours. C'est légitime pour une seule raison : il n'en sort que le
constat de ce que la machine a *vu*. Dès qu'un verdict, une équité ou un
seuil de bascule franchirait cette frontière, l'outil cesserait d'être un
banc d'essai pour devenir une assistance en temps réel — ce que le projet
refuse.

Une intention ne se relit pas dans six mois, et un refactoring ne la
consulte pas. Ces tests, si.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from pfs.vision import live

#: Modules qui calculent une recommandation. Aucun ne doit être joignable
#: depuis la calibration, ni directement ni par un import transitif écrit
#: dans le fichier.
CONSEIL = (
    "spot_advisor",
    "pushfold_review",
    "session_review",
    "simulateur",
    "solver",
    "equity",
    "icm",
    "nash",
)

#: Termes d'un verdict. Un champ de sortie qui porterait l'un d'eux
#: signalerait que la recommandation a fuité dans la structure de données.
VERDICTS = ("verdict", "action", "conseil", "recommand", "esperance",
            "equite", "equity", "bascule", "ev_", "_ev", "fold", "call",
            "raise", "shove", "jam", "miser", "coucher", "suivre")


def _source() -> str:
    return Path(inspect.getfile(live)).read_text(encoding="utf-8")


def test_aucun_import_de_conseil() -> None:
    """Le module n'importe aucun calculateur de décision."""
    arbre = ast.parse(_source())
    importes: list[str] = []
    for n in ast.walk(arbre):
        if isinstance(n, ast.Import):
            importes += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            importes.append(n.module)
            importes += [f"{n.module}.{a.name}" for a in n.names]

    fautifs = [i for i in importes
               if any(c in i.lower() for c in CONSEIL)]
    assert not fautifs, (
        f"la calibration en direct importe du conseil : {fautifs}. "
        "Elle doit rendre ce qui a été LU, jamais ce qu'il faut FAIRE.")


def test_la_lecture_ne_porte_aucun_verdict() -> None:
    """Aucun champ de `LectureLive` / `CarteLue` ne nomme une décision."""
    for classe in (live.LectureLive, live.CarteLue):
        champs = set(classe.__dataclass_fields__)
        champs |= {n for n, v in vars(classe).items()
                   if isinstance(v, property)}
        fautifs = [c for c in champs
                   if any(v in c.lower() for v in VERDICTS)]
        assert not fautifs, (
            f"{classe.__name__} expose {fautifs} : une recommandation a "
            "franchi la frontière lecture / conseil.")


def test_la_lecture_se_resume_sans_rien_recommander() -> None:
    """Le résumé destiné à l'utilisateur ne contient pas de verdict."""
    lecture = live.LectureLive(
        fenetre="table", largeur=1280, hauteur=800,
        cartes=[
            live.CarteLue(role="hero", boite=(0, 0, 40, 60), carte="Ah",
                          candidat="Ah", statut="sure", ecart=12, marge=90),
            live.CarteLue(role="hero", boite=(50, 0, 40, 60), carte=None,
                          candidat="Ks", statut="refus", ecart=940, marge=3),
        ])
    texte = lecture.resume().lower()
    assert not any(v in texte for v in VERDICTS), (
        f"le résumé recommande quelque chose : {texte!r}")
    assert "1/2" in texte, "le résumé doit dire combien de lectures sont sûres"


def test_taux_de_lecture() -> None:
    """`taux` mesure la part de cartes lues avec certitude."""
    vide = live.LectureLive(fenetre="t", largeur=1, hauteur=1)
    assert vide.taux == 0.0, "une lecture vide ne vaut pas 100 %"

    carte = live.CarteLue(role="board", boite=(0, 0, 1, 1), carte="2c",
                          candidat="2c", statut="sure", ecart=5, marge=80)
    pleine = live.LectureLive(fenetre="t", largeur=1, hauteur=1,
                              cartes=[carte, carte])
    assert pleine.taux == 1.0


def test_sonde_absente_dit_quoi_faire() -> None:
    """Un binaire manquant produit un message actionnable, pas un traceback.

    Le message doit contenir la commande de compilation : sur une machine
    fraîche, c'est la seule chose à faire, autant l'écrire.
    """
    try:
        live.chemin_sonde()
    except live.SondeIntrouvable as e:
        assert "cargo build" in str(e)
    else:
        pytest.skip("la sonde est compilée sur cette machine")
