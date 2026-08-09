"""Génère l'icône PKS — deux as (♥ ♠) sur feutre vert, jetons en dessous.

Tout est dessiné en géométrie (pas de police, pas d'image externe) : l'icône
reste nette à toutes les tailles et se régénère à l'identique. Le rendu se
fait à ×4 puis est réduit — c'est ce sur-échantillonnage qui donne des bords
propres sur les diagonales et les courbes.

    python scripts/make_icon.py [chemin/pks.ico]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

S = 4                      # sur-échantillonnage
N = 256                    # taille logique
W = N * S

FELT_HI = (26, 120, 66)    # feutre vert, centre
FELT_LO = (9, 58, 33)      # feutre vert, bords
CARD = (252, 252, 250)
CARD_EDGE = (196, 200, 205)
RED = (208, 32, 44)
BLACK = (22, 24, 30)


def rounded(draw, box, r, fill, outline=None, width=0):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def heart(draw, cx, cy, w, h, fill):
    """Cœur : deux lobes circulaires + une pointe triangulaire."""
    r = w / 2.0
    lobe = r * 0.56
    draw.ellipse([cx - r, cy - h * 0.5, cx - r + 2 * lobe, cy - h * 0.5 + 2 * lobe], fill=fill)
    draw.ellipse([cx + r - 2 * lobe, cy - h * 0.5, cx + r, cy - h * 0.5 + 2 * lobe], fill=fill)
    draw.polygon([(cx - r + 0.06 * w, cy - h * 0.5 + lobe * 0.95),
                  (cx + r - 0.06 * w, cy - h * 0.5 + lobe * 0.95),
                  (cx, cy + h * 0.5)], fill=fill)


def spade(draw, cx, cy, w, h, fill):
    """Pique : un cœur renversé + un pied trapézoïdal."""
    r = w / 2.0
    lobe = r * 0.56
    body_b = cy + h * 0.22                     # bas des lobes
    draw.ellipse([cx - r, body_b - 2 * lobe, cx - r + 2 * lobe, body_b], fill=fill)
    draw.ellipse([cx + r - 2 * lobe, body_b - 2 * lobe, cx + r, body_b], fill=fill)
    draw.polygon([(cx - r + 0.06 * w, body_b - lobe * 0.95),
                  (cx + r - 0.06 * w, body_b - lobe * 0.95),
                  (cx, cy - h * 0.5)], fill=fill)
    draw.polygon([(cx - w * 0.20, cy + h * 0.5),
                  (cx + w * 0.20, cy + h * 0.5),
                  (cx + w * 0.055, body_b - h * 0.04),
                  (cx - w * 0.055, body_b - h * 0.04)], fill=fill)


def letter_a(draw, cx, cy, w, h, fill):
    """Un « A » dessiné au trait : lisible même à 16 px, sans police."""
    t = w * 0.22
    draw.polygon([(cx - w / 2, cy + h / 2), (cx - w / 2 + t, cy + h / 2),
                  (cx, cy - h / 2), (cx - t / 2, cy - h / 2)], fill=fill)
    draw.polygon([(cx + w / 2, cy + h / 2), (cx + w / 2 - t, cy + h / 2),
                  (cx, cy - h / 2), (cx + t / 2, cy - h / 2)], fill=fill)
    draw.rectangle([cx - w * 0.27, cy + h * 0.10, cx + w * 0.27, cy + h * 0.10 + t * 0.85],
                   fill=fill)


def make_card(cw, ch, suit, colour):
    """Une carte sur son propre calque (permet la rotation antialiasée)."""
    card = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    rad = int(cw * 0.11)
    rounded(d, [0, 0, cw - 1, ch - 1], rad, CARD, CARD_EDGE, max(1, int(2 * S)))
    # index en haut à gauche : le A puis le petit symbole
    letter_a(d, cw * 0.27, ch * 0.20, cw * 0.30, ch * 0.19, colour)
    suit(d, cw * 0.27, ch * 0.375, cw * 0.24, ch * 0.15, colour)
    # grand symbole : centré horizontalement, pour rester lisible même quand
    # la carte est partiellement recouverte par sa voisine
    suit(d, cw * 0.50, ch * 0.66, cw * 0.50, ch * 0.33, colour)
    return card


def build() -> Image.Image:
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # fond : feutre vert, dégradé radial approché par cercles concentriques
    rounded(d, [0, 0, W - 1, W - 1], int(W * 0.22), FELT_LO)
    steps = 90
    for i in range(steps, 0, -1):
        f = i / steps
        col = tuple(int(FELT_LO[k] + (FELT_HI[k] - FELT_LO[k]) * (1 - f) ** 1.5)
                    for k in range(3))
        rr = int(W * 0.78 * f)
        d.ellipse([W * 0.5 - rr, W * 0.40 - rr, W * 0.5 + rr, W * 0.40 + rr], fill=col)
    # masque pour garder les coins arrondis
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, W - 1],
                                           radius=int(W * 0.22), fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)

    # ── les deux as, légèrement écartés en éventail ───────────────────────
    cw, ch = int(W * 0.36), int(W * 0.50)
    shadow = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)

    for suit, colour, angle, dx in ((spade, BLACK, 14, -0.135), (heart, RED, -14, 0.135)):
        card = make_card(cw, ch, suit, colour)
        rot = card.rotate(angle, expand=True, resample=Image.BICUBIC)
        px = int(W * 0.5 + dx * W - rot.width / 2)
        py = int(W * 0.40 - rot.height / 2)
        sd.rounded_rectangle([px + int(W * .012), py + int(W * .018),
                              px + rot.width + int(W * .012),
                              py + rot.height + int(W * .018)],
                             radius=int(cw * 0.11), fill=(0, 0, 0, 70))
        img.alpha_composite(rot, (px, py))
        # (ombre composée avant la carte suivante pour rester derrière)
    img = Image.alpha_composite(
        Image.alpha_composite(Image.new("RGBA", (W, W), (0, 0, 0, 0)), shadow), img)
    d = ImageDraw.Draw(img)

    # ── jetons empilés, en dessous ────────────────────────────────────────
    def chip(cx, cy, r, body, edge):
        d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=(0, 0, 0, 90))
        d.ellipse([cx - r, cy - r, cx + r, cy + r * 0.72], fill=body,
                  outline=edge, width=max(1, int(2.5 * S)))
        # marques de bord : ce qui fait « lire » un jeton de poker
        import math
        for k in range(8):
            a = math.radians(k * 45 + 11)
            x0, y0 = cx + math.cos(a) * r * 0.80, cy + math.sin(a) * r * 0.66
            x1, y1 = cx + math.cos(a) * r * 0.99, cy + math.sin(a) * r * 0.82
            d.line([(x0, y0), (x1, y1)], fill=edge, width=max(1, int(4 * S)))
        rr = r * 0.46
        d.ellipse([cx - rr, cy - rr * 0.92, cx + rr, cy + rr * 0.62],
                  fill=None, outline=edge, width=max(1, int(2.5 * S)))

    R = W * 0.115
    chip(W * 0.31, W * 0.815, R, (198, 40, 48), (245, 236, 232))   # rouge
    chip(W * 0.69, W * 0.815, R, (28, 32, 44), (238, 234, 228))    # noir
    chip(W * 0.50, W * 0.845, R * 1.10, (232, 232, 236), (176, 30, 40))  # blanc devant

    return img.resize((N, N), Image.LANCZOS)


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pks.ico")
    out.parent.mkdir(parents=True, exist_ok=True)
    icon = build()
    icon.save(out, format="ICO",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                     (128, 128), (256, 256)])
    icon.save(out.with_suffix(".png"))          # aperçu
    print(f"icône écrite : {out}  (+ aperçu {out.with_suffix('.png').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
