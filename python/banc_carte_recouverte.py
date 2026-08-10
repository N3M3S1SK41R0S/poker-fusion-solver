#!/usr/bin/env python
"""Banc de la carte RECOUVERTE : lire un carton dont le bas est masqué.

    python banc_carte_recouverte.py              le banc de référence
    python banc_carte_recouverte.py --balayage   + le balayage des fractions

Pourquoi ce fichier existe
--------------------------
Le premier diagnostic porté sur la capture PMU PLAY du 10 août 2026 était :
« les cartes du héros appartiennent à un TROISIÈME habillage, absent des
gabarits ». Ce diagnostic est FAUX, et ce banc le montre : le même gabarit
`pmu_solid` déjà livré lit les deux cartes du héros dès que la découpe couvre
le carton ENTIER. Ce qui manquait, ce n'était pas un habillage, c'était la
prise en compte du bandeau d'équité du client, qui recouvre le bas des cartes
du siège.

Trois mesures, dans l'ordre où elles tranchent :

1. **La capture réelle, découpe entière contre découpe visible.** Une même
   carte, cadrée sur toute sa hauteur (le bandeau compris) ou seulement sur sa
   partie visible, ne donne pas du tout le même écart. C'est le fait qui
   invalide l'hypothèse de l'habillage manquant.
2. **Le balayage de la fraction visible.** On recoupe les gabarits à toutes
   les fractions et on regarde laquelle lit juste : le creux est net, et il
   tombe exactement sur la fraction que le RAPPORT de la découpe prédit — donc
   aucune constante n'a besoin d'être calée à la main.
3. **Le plancher de bruit.** Ajouter des gabarits amputés donne à chaque
   découpe des chances supplémentaires de ressembler à quelque chose. On
   revérifie donc que les non-cartes restent au-dessus du seuil.

Vérité-terrain : les deux cartes du héros de
`captures/20260810-203641_capture.png` sont **5♣** et **3♥**, lues à l'œil.
Les découpes sont figées dans `tests/donnees/pmu_play_hero_5c.png` et
`pmu_play_hero_3h.png` — le banc ne dépend donc pas d'un fichier hors dépôt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

from pfs.vision.card_recognizer import (  # noqa: E402
    CARD_RATIO,
    DISTANCE_SURE,
    MAX_ACCEPT_DISTANCE,
    _rank,
    _truncated_templates,
    _visible_fraction,
    identify_card,
    load_templates,
)

DONNEES = Path(__file__).resolve().parent / "tests" / "donnees"

#: Les deux cartes du héros, telles que découpées sur la capture réelle.
#: (fichier, carte attendue) — la découpe s'arrête à la dernière ligne visible.
HEROS = (("pmu_play_hero_5c.png", "5c"), ("pmu_play_hero_3h.png", "3h"))

#: Géométrie lue au pixel sur la capture (voir la docstring du module).
#: Le carton mesure 122 px de large ; à ce rapport il ferait 165 px de haut,
#: mais le bandeau d'équité en masque le bas dès la ligne 1013.
LARGEUR_CARTON = 122
HAUTEUR_VISIBLE = 109


def _charge(nom: str) -> Image.Image:
    f = DONNEES / nom
    if not f.exists():                       # pragma: no cover - dépôt incomplet
        raise SystemExit(f"découpe absente : {f}")
    return Image.open(f).convert("RGB")


def mesure_capture() -> int:
    """Les deux cartes du héros sont-elles lues, et à quel écart ?"""
    print("=== 1. capture réelle PMU PLAY (20260810-203641) ===")
    hauteur_entiere = round(LARGEUR_CARTON / CARD_RATIO)
    print(f"    carton {LARGEUR_CARTON}×{hauteur_entiere} px attendu, "
          f"{HAUTEUR_VISIBLE} px visibles "
          f"({HAUTEUR_VISIBLE / hauteur_entiere:.0%})")
    fautes = 0
    for nom, attendu in HEROS:
        img = _charge(nom)
        frac = _visible_fraction(img)
        m = identify_card(img)
        etat = "OK " if m.card == attendu else "NON"
        if m.card != attendu:
            fautes += 1
        print(f"    {etat} {nom:24s} attendu {attendu} → lu {m.card} "
              f"({m.statut}) écart {m.distance} marge {m.margin} "
              f"— fraction déduite {frac}")
        # le même carton comparé aux gabarits ENTIERS : c'est la mesure qui
        # montre que le refus venait de l'amputation, pas de l'habillage
        d, c, marge, _ = _rank(img, load_templates())
        print(f"        contre gabarits entiers : {c} à {d} (marge {marge}) "
              f"— seuil {MAX_ACCEPT_DISTANCE}")
    return fautes


def balayage() -> None:
    """Quelle fraction de gabarit lit juste ? Le creux doit être unique."""
    print("\n=== 2. balayage de la fraction visible ===")
    for nom, attendu in HEROS:
        img = _charge(nom)
        w, h = img.size
        predite = round(CARD_RATIO * h / w, 2)
        print(f"    {nom} (rapport {w / h:.3f}, fraction prédite {predite})")
        for k in range(13):
            frac = round(0.56 + 0.02 * k, 2)
            d, c, marge, _ = _rank(img, _truncated_templates(frac))
            marque = "  <-- juste" if c == attendu else ""
            print(f"        {frac:.2f} : {c} à {d} (marge {marge}){marque}")


def plancher_de_bruit() -> int:
    """Les non-cartes restent-elles au-dessus du seuil, gabarits amputés compris ?"""
    print("\n=== 3. plancher de bruit (240 découpes qui ne sont pas des cartes) ===")
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    from test_faux_positifs_surs import _non_cartes  # noqa: E402

    ecarts = [identify_card(i) for i in _non_cartes()]
    plancher = min(m.distance for m in ecarts)
    sures = [m for m in ecarts if m.statut == "sure"]
    print(f"    écart minimal {plancher} (seuil {DISTANCE_SURE})")
    print(f"    non-cartes affirmées : {len(sures)}")
    return 0 if (plancher > DISTANCE_SURE and not sures) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--balayage", action="store_true",
                   help="ajoute le balayage des fractions (plus lent)")
    args = p.parse_args()

    fautes = mesure_capture()
    if args.balayage:
        balayage()
    fautes += plancher_de_bruit()
    print("\n" + ("BANC VERT" if fautes == 0 else f"BANC ROUGE ({fautes})"))
    return fautes


if __name__ == "__main__":
    raise SystemExit(main())
