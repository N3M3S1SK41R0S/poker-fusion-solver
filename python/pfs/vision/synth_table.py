"""Générateur de tables synthétiques — le banc d'essai de la perception.

Sert à MESURER la détection et la lecture avant d'avoir des captures réelles :
on compose des tables dont on connaît la vérité-terrain (cartes posées, leurs
positions, les montants), sur des tapis et des habillages variés.

Le DÉCOR n'est pas cosmétique : c'est lui qui rend la mesure honnête. Une
table nue ne contient que des cartes, donc n'importe quel détecteur y annonce
0 % de fausses cartes. On dessine donc ce qui, dans un vrai client, a la forme
d'une carte sans en être une : piles de jetons vues de côté, avatars de siège,
plaques de nom, boutons d'action, bouton donneur. Deux mesures faites sur ce
décor sont ce qui a imposé au détecteur ses deux bornes hautes :

  · les trois piles de jetons tombent DANS la fenêtre de rapport, pas au-delà
    (boîtes mesurées 24×26, 24×30 et 24×34, soit 0,92, 0,80 et 0,71 : deux
    des trois sont des rapports parfaitement acceptables pour une carte). Ce
    n'est donc pas leur forme qui les écarte mais leurs stries : 0,833
    d'arêtes sur une pile isolée, 0,769 à 0,775 sur la boîte que le détecteur
    cadre réellement au milieu d'une table. C'est cette dernière valeur qui
    compte, et elle tombe dans la distribution des vraies cartes ;
  · les avatars de siège sont des carrés, mais leur boîte une fois recalée sur
    les arêtes mesure 0,846 à 0,872 — c'est cette valeur-là, pas le carré
    théorique, qui fixe la borne haute de rapport à 0,82.

Les cartes face cachée des adversaires, elles, SONT des cartes : elles ont
leur propre vérité-terrain (``villain_boxes``) et doivent être trouvées sans
jamais être attribuées au héros ni au board. Les montants (``texts``) sont
DESSINÉS sur l'image : une vérité-terrain pour du texte absent ne servirait à
rien à la phase B.

Honnêteté : une table synthétique n'est pas une capture réelle (pas de
compression, pas de dégradé de feutre, pas d'ombres portées). Elle valide la
LOGIQUE de détection, pas la tolérance au rendu d'un vrai client — d'où le
bruit et le flou optionnels, et la validation ultérieure sur vraies captures.
Le trou le plus large est chiffré dans `table_detector` : recompressé en JPEG,
ce banc perd la localisation de 98,8 % (PNG) à 88,1 % (q = 75).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from pfs.vision.card_recognizer import _TEMPLATE_ROOT

__all__ = ["TableSpec", "SynthTable", "render_table", "FELTS"]

FELTS: dict[str, tuple[int, int, int]] = {
    "feutre vert": (20, 83, 45),
    "tapis clair": (203, 207, 214),
    "bleu nuit": (16, 24, 56),
    "bordeaux": (92, 18, 30),
    "gris moyen": (120, 122, 128),
    "noir": (12, 12, 14),
}


@dataclass(slots=True)
class TableSpec:
    """Ce qu'on veut dessiner."""

    hero: tuple[str, str] = ("Ah", "Kd")
    board: tuple[str, ...] = ()
    theme: str = "pmu_solid"
    felt: str = "feutre vert"
    card_w: int = 68
    card_h: int = 92
    size: tuple[int, int] = (900, 560)
    pot: float = 1.5
    hero_stack: float = 10.0
    villain_stacks: tuple[float, ...] = (9.5, 9.0)
    to_call: float = 1.0
    noise: float = 0.0
    blur: float = 0.0
    seed: int = 0
    decor: bool = True
    villains: int = 0


@dataclass(slots=True)
class SynthTable:
    """L'image produite ET sa vérité-terrain."""

    image: Image.Image
    hero_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    board_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    villain_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    hero: tuple[str, ...] = ()
    board: tuple[str, ...] = ()
    texts: dict[str, str] = field(default_factory=dict)

    @property
    def all_boxes(self) -> list[tuple[int, int, int, int]]:
        """Les cartes du héros et du board — celles qu'il faut LIRE."""
        return self.hero_boxes + self.board_boxes

    @property
    def every_card(self) -> list[tuple[int, int, int, int]]:
        """Toutes les cartes présentes, y compris les faces cachées."""
        return self.hero_boxes + self.board_boxes + self.villain_boxes


# Sièges des adversaires, en fractions de l'image (coin haut-gauche de la
# plaque). Aucun n'est aussi bas que le héros : c'est la disposition de toutes
# les tables en ligne, et c'est ce qui rend l'attribution des rôles possible.
_SEATS: tuple[tuple[float, float], ...] = (
    (0.055, 0.24), (0.055, 0.55), (0.40, 0.04), (0.755, 0.24), (0.755, 0.55),
)


def _plaque(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
            felt: tuple[int, int, int], label: str, amount: str) -> None:
    """Plaque de siège : avatar carré + bandeau nom/tapis."""
    dark = tuple(max(0, c - 40) for c in felt)
    d.rounded_rectangle([x + h, y, x + w, y + h], radius=h // 4, fill=dark,
                        outline=tuple(min(255, c + 40) for c in dark), width=1)
    d.rectangle([x, y, x + h, y + h], fill=tuple(min(255, c + 70) for c in dark))
    d.text((x + h + 6, y + 3), label, fill=(235, 235, 235))
    d.text((x + h + 6, y + h // 2), amount, fill=(210, 210, 160))


def _chip_stack(d: ImageDraw.ImageDraw, x: int, y: int, w: int, n: int) -> None:
    """Pile de jetons vue de côté : `n` jetons empilés, donc `n` fixe le rapport.

    Le banc en dessine trois de hauteurs différentes pour balayer la fenêtre du
    détecteur par le haut plutôt que de viser un seul rapport : à w = 23, les
    boîtes mesurent 0,92, 0,80 et 0,71 pour n = 6, 7 et 8.
    """
    eh = max(3, w // 4)
    for i in range(n):
        yy = y - i * (eh - 1)
        col = (196, 60, 60) if i % 2 else (240, 240, 240)
        d.ellipse([x, yy, x + w, yy + eh], fill=col, outline=(40, 40, 40))


def _card_back(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    """Carte face cachée : carton bleu à liseré clair, aux proportions d'une carte."""
    d.rounded_rectangle([x, y, x + w - 1, y + h - 1], radius=max(2, w // 10),
                        fill=(28, 52, 122), outline=(232, 232, 240), width=2)
    d.rounded_rectangle([x + w // 6, y + h // 6, x + w - 1 - w // 6, y + h - 1 - h // 6],
                        radius=max(1, w // 12), outline=(96, 122, 196), width=2)


def _draw_decor(d: ImageDraw.ImageDraw, spec: TableSpec, W: int, H: int,
                felt: tuple[int, int, int]) -> list[tuple[int, int, int, int]]:
    """Dessine sièges, jetons et boutons ; renvoie les boîtes des faces cachées."""
    pw, ph = int(W * 0.145), int(H * 0.058)
    backs: list[tuple[int, int, int, int]] = []
    bw, bh = int(spec.card_w * 0.62), int(spec.card_h * 0.62)

    for i, (fx, fy) in enumerate(_SEATS):
        x, y = int(W * fx), int(H * fy)
        _plaque(d, x, y, pw, ph, felt, f"Joueur {i + 1}", f"{9.5 - i:.2f} BB")
        if i < spec.villains:
            cx = x + pw // 2 - bw - 2
            cy = y - bh - 4 if fy > 0.2 else y + ph + 4
            for k in range(2):
                bx = cx + k * (bw + 3)
                _card_back(d, bx, cy, bw, bh)
                backs.append((bx, cy, bw, bh))

    # jetons au centre : plusieurs piles verticales, le piège à faux positifs
    for k, (fx, fy) in enumerate(((0.44, 0.62), (0.52, 0.63), (0.60, 0.61))):
        _chip_stack(d, int(W * fx), int(H * fy), max(10, int(W * 0.026)), 6 + k)

    # bouton donneur et boutons d'action
    r = max(7, int(W * 0.013))
    d.ellipse([int(W * 0.36), int(H * 0.60), int(W * 0.36) + r, int(H * 0.60) + r],
              fill=(245, 245, 245), outline=(60, 60, 60), width=1)
    aw, ah = int(W * 0.11), int(H * 0.055)
    for k, txt in enumerate(("FOLD", "CALL", "RAISE")):
        ax = int(W * 0.62) + k * (aw + 8)
        d.rounded_rectangle([ax, int(H * 0.90), ax + aw, int(H * 0.90) + ah],
                            radius=6, fill=(48, 52, 60), outline=(150, 150, 150))
        d.text((ax + 10, int(H * 0.90) + ah // 3), txt, fill=(240, 240, 240))
    return backs


def _card(theme: str, card: str, w: int, h: int) -> Image.Image:
    p = _TEMPLATE_ROOT / theme / f"{card}.png"
    return Image.open(p).convert("RGBA").resize((w, h), Image.LANCZOS)


def render_table(spec: TableSpec) -> SynthTable:
    """Dessine une table et renvoie l'image + la vérité-terrain."""
    W, H = spec.size
    felt = FELTS.get(spec.felt, FELTS["feutre vert"])
    img = Image.new("RGB", (W, H), felt)
    d = ImageDraw.Draw(img)

    # ovale de table, légèrement plus clair — comme un vrai client
    lighter = tuple(min(255, c + 26) for c in felt)
    d.ellipse([W * 0.10, H * 0.16, W * 0.90, H * 0.80], fill=lighter,
              outline=tuple(min(255, c + 60) for c in felt), width=5)

    villain_boxes = _draw_decor(d, spec, W, H, felt) if spec.decor else []

    cw, ch = spec.card_w, spec.card_h
    gap = max(6, cw // 8)

    # board centré
    board_boxes: list[tuple[int, int, int, int]] = []
    if spec.board:
        total = len(spec.board) * cw + (len(spec.board) - 1) * gap
        x0 = (W - total) // 2
        y0 = int(H * 0.38)
        for i, c in enumerate(spec.board):
            x = x0 + i * (cw + gap)
            img.paste(_card(spec.theme, c, cw, ch), (x, y0),
                      _card(spec.theme, c, cw, ch))
            board_boxes.append((x, y0, cw, ch))

    # cartes du héros, en bas, un peu chevauchantes (comme à l'écran)
    hero_boxes: list[tuple[int, int, int, int]] = []
    hx = (W - (2 * cw + gap // 2)) // 2
    hy = int(H * 0.72)
    for i, c in enumerate(spec.hero):
        x = hx + i * (cw + gap // 2)
        img.paste(_card(spec.theme, c, cw, ch), (x, hy),
                  _card(spec.theme, c, cw, ch))
        hero_boxes.append((x, hy, cw, ch))

    # Montants : DESSINÉS, sinon ``texts`` serait une vérité-terrain pour du
    # texte absent de l'image. Ils sont placés hors des cartes (au-dessus du
    # board, sous la main du héros, au-dessus des boutons) pour ne pas masquer
    # une arête que la détection doit trouver.
    texts = {
        "pot": f"Pot: {spec.pot:.2f} BB".replace(".", ","),
        "hero_stack": f"{spec.hero_stack:.2f} BB".replace(".", ","),
        "to_call": f"CALL {spec.to_call:.2f} BB".replace(".", ","),
    }
    # 16 px sous la main du héros, pas 6 : à 6 px le libellé tombe DANS la bande
    # d'abord basse que la détection exige calme, et la localisation du banc
    # chute de 98,8 % à 97,0 %. C'est une limite du détecteur, mesurée ici.
    ink = (238, 238, 210)
    d.text((W // 2 - 30, int(H * 0.30)), texts["pot"], fill=ink)
    d.text((W // 2 - 24, hy + ch + 16), texts["hero_stack"], fill=ink)
    d.text((int(W * 0.62), int(H * 0.90) - 14), texts["to_call"], fill=ink)

    if spec.blur:
        img = img.filter(ImageFilter.GaussianBlur(spec.blur))
    if spec.noise:
        a = np.asarray(img).astype(np.int16)
        a = np.clip(a + np.random.default_rng(spec.seed).normal(
            0, spec.noise, a.shape), 0, 255)
        img = Image.fromarray(a.astype(np.uint8))

    return SynthTable(image=img, hero_boxes=hero_boxes, board_boxes=board_boxes,
                      villain_boxes=villain_boxes,
                      hero=tuple(spec.hero), board=tuple(spec.board),
                      texts=texts)
