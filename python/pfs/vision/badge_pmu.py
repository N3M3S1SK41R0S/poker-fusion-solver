"""Détecteur du badge « PMU PLAY » — la preuve d'identité du client play-money.

Ce module LIT une capture et dit si la marque « PMU PLAY » (argent fictif)
y est visible. Il ne décide RIEN d'éthique : il rend une lecture
(:class:`BadgeLu`), et c'est le gate de conformité (`pfs.compliance.gate`,
chantier séparé) qui en tirera une décision. Un détecteur qui déciderait
lui-même mélangerait la mesure et la politique — les deux doivent pouvoir
être auditées séparément.

Deux marqueurs, mesurés sur les 57 captures réelles des 10-11 août 2026
(deux tables PMU PLAY v.26.1.1.23, fenêtre 1920×1361) :

1. **Filigrane central « PMU PLAY »** — ZNCC en niveaux de gris à POSITION
   FIXE (le board recouvre le filigrane en cours de main, chercher ailleurs
   ne sert à rien) : boîte (716, 584, 1156, 655) sur le thème clair,
   (736, 634, 1174, 704) sur le thème sombre, jitter ±4 px.
   Mesuré : 17/57 frames visibles (scores 0,417 à 1,000), 40/57 occultées
   par le board (≤ 0,086), négatifs synthétiques ≤ 0,050.
2. **Dos de carte orange « PMU PLAY »** — ZNCC plein cadre sur le canal
   R − B (l'orange du dos se détache de tout feutre), plancher de variance
   σ ≥ 5 niveaux (sans lui, une fenêtre quasi plate fait exploser la
   normalisation : les négatifs synthétiques montaient à 44 991 au lieu
   de 0,580). Mesuré : 49/57 frames à ≥ 0,975, frames sans dos ≤ 0,673,
   négatifs synthétiques ≤ 0,580.

Combinés : 50/57 frames portent une preuve immédiate. Les 7 restantes sont
des fins de main (abattage avec board sur le filigrane, ou tous les
adversaires couchés) — c'est au consommateur de la lecture de tenir une
mémoire courte, pas à ce module de l'inventer.

Coûts mesurés : filigrane 5,6 ms/frame, dos plein cadre ~325 ms/frame.

Limites honnêtes (NEMESIS)
--------------------------
* CALIBRÉ pour une fenêtre 1920×1361 uniquement. Autre taille ⇒ échelle
  différente ⇒ **refus explicite** (fail-closed) : la robustesse
  multi-échelles n'a PAS été mesurée, on ne l'improvise pas.
* Les 57 frames viennent de DEUX tables du même client, même version.
* Aucune capture du client PMU argent réel n'existe ici : « le client réel
  n'affiche pas PMU PLAY » est plausible, pas prouvé. D'où le sens unique
  de la lecture : badge vu = play money ; badge non vu = NON CONCLUANT.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.signal import fftconvolve

__all__ = [
    "BOITE_DOS_SOURCE",
    "BOITE_FILIGRANE_CLAIR",
    "BOITE_FILIGRANE_SOMBRE",
    "JITTER_PX",
    "SEUIL_DOS",
    "SEUIL_FILIGRANE",
    "SIGMA_MIN_DOS",
    "TAILLE_CALIBREE",
    "BadgeLu",
    "detecter_badge",
]

_F32 = npt.NDArray[np.float32]

#: Taille de fenêtre sur laquelle TOUT est calibré (boîtes, sprites, seuils).
#: Les 57 captures de l'étude font toutes cette taille ; aucune autre n'a été
#: mesurée, donc aucune autre n'est acceptée.
TAILLE_CALIBREE: tuple[int, int] = (1920, 1361)

#: Boîte du filigrane central (x0, y0, x1, y1) — thème « feutre vert clair »
#: (disposition 6-max). Source : sessions/100_-_6-max_Turbo_Re-buy/0000.png.
BOITE_FILIGRANE_CLAIR: tuple[int, int, int, int] = (716, 584, 1156, 655)

#: Boîte du filigrane central — thème « feutre violet sombre » (7-max KO).
#: Source : sessions/300_7-max_KO/0010.png. Le centre de table diffère de
#: ~50 px entre les deux dispositions, d'où deux boîtes.
BOITE_FILIGRANE_SOMBRE: tuple[int, int, int, int] = (736, 634, 1174, 704)

#: Boîte d'origine du sprite de dos (documentaire : la recherche du dos est
#: PLEIN CADRE, les sièges variant d'une disposition à l'autre).
BOITE_DOS_SOURCE: tuple[int, int, int, int] = (950, 106, 1058, 210)

#: Dérive de position tolérée autour de la boîte du filigrane. Les 57 frames
#: n'ont montré AUCUNE dérive (rendu statique, 16 frames au pixel près) ;
#: ±4 px absorbent un arrondi de cadrage sans ouvrir la porte au reste.
JITTER_PX = 4

#: Seuil du filigrane. Pire positif visible mesuré : 0,417 ; pire score sur
#: filigrane occulté par le board : 0,086 ; pire négatif synthétique : 0,050.
#: 0,25 sépare d'un rapport 8,3× le pire positif du pire négatif — zéro faux
#: positif sur l'étude.
SEUIL_FILIGRANE = 0.25

#: Seuil du dos orange. Pires positifs mesurés : 0,975 (49 frames à ≥ 0,975) ;
#: pire frame SANS dos : 0,673 (pic sur du contenu orange quelconque) ; pire
#: négatif synthétique : 0,580. 0,80 laisse 0,175 de marge sous le pire
#: positif et 0,127 au-dessus du pire négatif.
SEUIL_DOS = 0.80

#: Plancher de variance du dos : une fenêtre dont l'écart-type R−B est
#: inférieur à 5 niveaux est quasi plate — sa ZNCC est numériquement
#: instable (mesuré : scores > 44 000 sur feutre uni sans ce plancher).
SIGMA_MIN_DOS = 5.0

_DOSSIER_SPRITES = Path(__file__).parent / "templates" / "pmu_play"


@dataclass(frozen=True, slots=True)
class BadgeLu:
    """Lecture du badge sur UNE image — une mesure, pas une décision.

    Attributes
    ----------
    filigrane_score : float
        Meilleure ZNCC du filigrane central sur les deux thèmes (−1 à 1 ;
        0.0 quand la fenêtre est plate ou l'image refusée).
    dos_score : float
        Pic de ZNCC plein cadre du dos orange (canal R−B, plancher σ ≥ 5).
    vu : bool
        Au moins un des deux marqueurs dépasse son seuil. ``False`` ne
        prouve PAS l'absence du badge (7/57 frames de l'étude sont des fins
        de main sans marqueur visible) : c'est un « non concluant ».
    position : tuple[int, int] | None
        Coin haut-gauche du marqueur qui porte la détection — la boîte du
        filigrane s'il est vu (prioritaire), sinon le pic du dos. ``None``
        si rien n'est vu.
    theme : str | None
        ``"clair"`` ou ``"sombre"`` si le filigrane est vu (c'est lui qui
        révèle le thème), ``None`` sinon.
    refus : str | None
        Non ``None`` si l'image n'a pas été mesurée du tout (taille non
        calibrée). Dans ce cas les scores valent 0.0 et ``vu`` est
        ``False`` — fail-closed.
    """

    filigrane_score: float
    dos_score: float
    vu: bool
    position: tuple[int, int] | None
    theme: str | None
    refus: str | None = None


@lru_cache(maxsize=1)
def _gabarits() -> tuple[_F32, _F32, _F32]:
    """(filigrane clair L, filigrane sombre L, dos R−B) — chargés une fois."""
    clair = np.asarray(
        Image.open(_DOSSIER_SPRITES / "sprite_pmu_play_clair.png").convert("L"),
        dtype=np.float32)
    sombre = np.asarray(
        Image.open(_DOSSIER_SPRITES / "sprite_pmu_play_sombre.png").convert("L"),
        dtype=np.float32)
    rgb = np.asarray(
        Image.open(_DOSSIER_SPRITES / "sprite_dos_pmu_play.png").convert("RGB"),
        dtype=np.float32)
    return clair, sombre, rgb[..., 0] - rgb[..., 2]


def _zncc_fixe(img: _F32, tpl: _F32, x0: int, y0: int) -> float:
    """ZNCC du gabarit contre ``img`` à (x0, y0), meilleur score sur ±JITTER_PX.

    Les fenêtres plates (norme nulle : feutre uni) sont ignorées ; si aucune
    fenêtre n'est mesurable, le score vaut 0.0 — pas −1, qui est un vrai
    score d'anticorrélation.
    """
    th, tw = tpl.shape
    tn = tpl - tpl.mean()
    tss = float(np.sqrt((tn ** 2).sum()))
    if tss == 0.0:
        return 0.0
    best: float | None = None
    for dy in range(-JITTER_PX, JITTER_PX + 1):
        for dx in range(-JITTER_PX, JITTER_PX + 1):
            y, x = y0 + dy, x0 + dx
            if y < 0 or x < 0 or y + th > img.shape[0] or x + tw > img.shape[1]:
                continue
            w = img[y:y + th, x:x + tw]
            wn = w - w.mean()
            d = float(np.sqrt((wn ** 2).sum())) * tss
            if d > 0:
                s = float((wn * tn).sum()) / d
                best = s if best is None else max(best, s)
    return 0.0 if best is None else best


def _zncc_plein_cadre(img: _F32, tpl: _F32) -> tuple[float, tuple[int, int] | None]:
    """Pic de ZNCC de ``tpl`` partout dans ``img`` ; renvoie (score, (x, y)).

    Corrélation par FFT + sommes locales par images intégrales (c'est ce qui
    tient le plein cadre 1920×1361 à ~325 ms). Les fenêtres dont l'écart-type
    est sous :data:`SIGMA_MIN_DOS` sont exclues du pic : sans ce plancher, la
    division par une variance quasi nulle fabrique des scores absurdes
    (mesuré : 44 991 sur un feutre bordeaux uni).
    """
    th, tw = tpl.shape
    n = th * tw
    tn = tpl - tpl.mean()
    tss = float(np.sqrt((tn ** 2).sum()))
    if tss == 0.0:
        return 0.0, None
    corr = fftconvolve(img, tn[::-1, ::-1], mode="valid")
    ii = np.pad(np.cumsum(np.cumsum(img.astype(np.float64), 0), 1), ((1, 0), (1, 0)))
    ii2 = np.pad(np.cumsum(np.cumsum(img.astype(np.float64) ** 2, 0), 1),
                 ((1, 0), (1, 0)))
    h, w = img.shape
    s1 = (ii[th:h + 1, tw:w + 1] - ii[:h - th + 1, tw:w + 1]
          - ii[th:h + 1, :w - tw + 1] + ii[:h - th + 1, :w - tw + 1])
    s2 = (ii2[th:h + 1, tw:w + 1] - ii2[:h - th + 1, tw:w + 1]
          - ii2[th:h + 1, :w - tw + 1] + ii2[:h - th + 1, :w - tw + 1])
    var = np.maximum(s2 - s1 ** 2 / n, 0.0)
    mesurable = var >= n * SIGMA_MIN_DOS ** 2
    if not bool(mesurable.any()):
        return 0.0, None
    z = np.where(mesurable, corr / (np.sqrt(np.maximum(var, 1e-6)) * tss), -2.0)
    iy, ix = np.unravel_index(int(np.argmax(z)), z.shape)
    return float(z[iy, ix]), (int(ix), int(iy))


def detecter_badge(image: Image.Image) -> BadgeLu:
    """Lit les deux marqueurs « PMU PLAY » sur une capture 1920×1361.

    Parameters
    ----------
    image : PIL.Image.Image
        La capture pleine fenêtre du client, telle que rendue par la sonde.

    Returns
    -------
    BadgeLu
        La lecture. Si l'image n'est pas à la taille calibrée, ``refus``
        explique pourquoi et rien n'a été mesuré (fail-closed) : la
        robustesse multi-échelles n'est pas mesurée, on ne l'improvise pas.
    """
    if image.size != TAILLE_CALIBREE:
        return BadgeLu(
            filigrane_score=0.0, dos_score=0.0, vu=False, position=None,
            theme=None,
            refus=(f"badge non calibré à cette taille : fenêtre "
                   f"{image.size[0]}×{image.size[1]}, calibration "
                   f"{TAILLE_CALIBREE[0]}×{TAILLE_CALIBREE[1]} uniquement"))

    tpl_clair, tpl_sombre, tpl_dos = _gabarits()
    gris = np.asarray(image.convert("L"), dtype=np.float32)
    candidats = (
        (_zncc_fixe(gris, tpl_clair, *BOITE_FILIGRANE_CLAIR[:2]),
         "clair", BOITE_FILIGRANE_CLAIR),
        (_zncc_fixe(gris, tpl_sombre, *BOITE_FILIGRANE_SOMBRE[:2]),
         "sombre", BOITE_FILIGRANE_SOMBRE),
    )
    filigrane_score, theme_candidat, boite = max(candidats, key=lambda c: c[0])

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    dos_score, dos_pos = _zncc_plein_cadre(rgb[..., 0] - rgb[..., 2], tpl_dos)

    filigrane_vu = filigrane_score >= SEUIL_FILIGRANE
    dos_vu = dos_score >= SEUIL_DOS
    if filigrane_vu:
        position: tuple[int, int] | None = (boite[0], boite[1])
    elif dos_vu:
        position = dos_pos
    else:
        position = None
    return BadgeLu(
        filigrane_score=filigrane_score, dos_score=dos_score,
        vu=filigrane_vu or dos_vu, position=position,
        theme=theme_candidat if filigrane_vu else None)
