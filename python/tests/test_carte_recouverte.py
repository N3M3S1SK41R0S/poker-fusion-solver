"""Une carte dont le bas est RECOUVERT doit rester lisible.

Sur la capture PMU PLAY du 10 août 2026, le bandeau d'équité du client
(« 0 % 0 % 12 % ») mange le tiers bas des cartes du siège. Comparées aux
gabarits ENTIERS, les deux cartes du héros sortaient à 705 et 731 — au-dessus
du plancher de bruit, donc refusées. Le premier diagnostic fut « habillage
absent des gabarits » ; il était faux. L'habillage est `pmu_solid`, déjà
livré : c'est la COMPARAISON qui était fautive.

La vérité-terrain est lue à l'œil sur la capture : **5♣** et **3♥**. Les deux
découpes sont figées dans `donnees/`, aux bords relevés au pixel (cartons de
122 px de large, 109 px encore visibles sur les 165 attendus).

Ce fichier verrouille trois choses :
  1. les deux cartes sont lues, et AFFIRMÉES (pas seulement proposées) ;
  2. la fraction visible se déduit du seul rapport de la découpe — aucune
     constante n'est calée sur cette capture ;
  3. le mécanisme ne coûte rien aux découpes de forme normale, et ne fait pas
     descendre le plancher de bruit sous le seuil.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pfs.vision.card_recognizer import (
    CARD_RATIO,
    DISTANCE_SURE,
    _rank,
    _TEMPLATE_ROOT,
    _THEMES,
    _visible_fraction,
    identify_card,
    load_templates,
)

DONNEES = Path(__file__).resolve().parent / "donnees"

#: (fichier, carte attendue) — vérité-terrain lue à l'œil sur la capture.
HEROS = (("pmu_play_hero_5c.png", "5c"), ("pmu_play_hero_3h.png", "3h"))


def _charge(nom: str) -> Image.Image:
    f = DONNEES / nom
    if not f.exists():  # pragma: no cover - dépôt incomplet
        pytest.skip(f"découpe de référence absente : {f}")
    return Image.open(f).convert("RGB")


@pytest.mark.parametrize("nom,attendu", HEROS)
def test_la_carte_recouverte_est_lue_et_affirmee(nom: str, attendu: str) -> None:
    """Le seul critère qui compte : la carte du héros est lue juste."""
    m = identify_card(_charge(nom))
    assert m.card == attendu, (
        f"{nom} : attendu {attendu}, lu {m.card} (pari {m.best_guess}, "
        f"écart {m.distance}, marge {m.margin}, statut {m.statut})")
    assert m.statut == "sure", (
        f"{nom} : lecture juste mais non affirmée (statut {m.statut})")
    assert m.distance < DISTANCE_SURE


@pytest.mark.parametrize("nom,attendu", HEROS)
def test_les_gabarits_entiers_seuls_ne_suffisaient_pas(nom: str,
                                                       attendu: str) -> None:
    """Le défaut corrigé doit rester REPRODUCTIBLE, sinon le correctif est du
    folklore : comparée aux seuls gabarits entiers, la même découpe échoue."""
    d, carte, _, _ = _rank(_charge(nom), load_templates())
    assert carte != attendu or d > DISTANCE_SURE, (
        f"{nom} se lit désormais sans gabarit amputé ({carte} à {d}) : "
        "ce test ne mesure plus rien, revoir la justification du mécanisme.")


@pytest.mark.parametrize("nom,attendu", HEROS)
def test_la_fraction_visible_se_deduit_du_rapport(nom: str, attendu: str) -> None:
    """Aucune constante calée à la main : la forme de la découpe dit tout."""
    img = _charge(nom)
    w, h = img.size
    frac = _visible_fraction(img)
    assert frac is not None
    assert frac == pytest.approx(CARD_RATIO * h / w, abs=0.01)
    # et cette fraction est bien celle qui lit juste
    assert identify_card(img).card == attendu


def test_un_gabarit_entier_n_active_pas_le_mecanisme() -> None:
    """Sur une découpe de forme normale, rien ne doit changer.

    C'est ce qui garantit que les 104 gabarits livrés continuent de se relire
    exactement comme avant : leur rapport est celui d'un carton entier.
    """
    for theme in _THEMES:
        d = _TEMPLATE_ROOT / theme
        if not d.is_dir():  # pragma: no cover - thème non livré
            continue
        for png in sorted(d.glob("*.png")):
            assert _visible_fraction(Image.open(png)) is None, (
                f"{png} déclenche la comparaison aux gabarits amputés")
            m = identify_card(png)
            assert m.card == png.stem and m.statut == "sure"


def test_le_plancher_de_bruit_survit_aux_gabarits_ampute() -> None:
    """Multiplier les gabarits multiplie les occasions de ressembler.

    Mesuré : sur les 240 découpes qui ne sont pas des cartes, 99 sont assez
    trapues pour déclencher la comparaison amputée, et le plancher ne bouge
    pas (672, seuil 625). Si ce test tombe, c'est le seuil qu'il faut
    re-mesurer — pas ce test.
    """
    from test_faux_positifs_surs import _non_cartes

    lus = [identify_card(i) for i in _non_cartes()]
    plancher = min(m.distance for m in lus)
    assert plancher > DISTANCE_SURE, (
        f"une non-carte descend à {plancher}, sous DISTANCE_SURE="
        f"{DISTANCE_SURE} : les gabarits amputés ont refermé le vide.")
    assert not [m for m in lus if m.statut == "sure"]
