"""Calibration en direct — lire une vraie table, à l'écran, tout de suite.

Pourquoi ce module existe
-------------------------
Tout ce qui a été mesuré jusqu'ici l'a été sur des tables **synthétiques**.
C'est ce qui a coûté le plus de temps : les défauts réels (HUD qui masque le
bas des cartes, anti-crénelage du client, échelle de la fenêtre, thème du
tapis) ne se voient pas sur une image fabriquée pour le test. Et la boucle
« je colle une capture → ça rate → je corrige → je recolle » est lente et
coûteuse : un aller-retour complet par image.

Ce module supprime l'aller-retour. Il capture la fenêtre du client à
l'instant présent, la donne au détecteur puis au recogniseur, et rend
**uniquement ce qui a été lu**, avec la confiance de chaque lecture. Une
session de calibration donne des dizaines de lectures réelles en quelques
minutes, chaque échec étant archivé (`pfs.vision.archive`) pour être rejoué
plus tard contre un recogniseur corrigé.

Ce que ce module ne fait pas, et ne fera pas
--------------------------------------------
Il ne produit **aucune recommandation de jeu**. Pas de verdict, pas
d'équité, pas de seuil de bascule, aucun appel au conseiller. Il dit ce que
la machine a *vu*, jamais ce qu'il faudrait *faire*.

Ce n'est pas une limite technique, c'est la frontière du logiciel : lire son
propre écran pour vérifier qu'un programme reconnaît des images ne retire
rien à personne — recevoir une recommandation calculée pendant une main
d'argent réel la retire aux adversaires, qui ignorent qu'ils affrontent une
machine. La première moitié est un banc d'essai, la seconde une triche. La
séparation est ici, dans le code, et `tests/test_live_sans_conseil.py` la
vérifie.

Le conseil, lui, reste disponible sur les mains **terminées** (onglet « Ma
main », `pfs.analysis.spot_advisor`) et sur l'entraînement
(`pfs.train.simulateur`) — là où il ne coûte rien à personne.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from pfs.vision.archive import enregistrer_echec
from pfs.vision.card_recognizer import identify_card
from pfs.vision.table_detector import CardBox, read_table

__all__ = [
    "chemin_sonde",
    "fenetres_disponibles",
    "capturer_fenetre",
    "lire_ecran",
    "CarteLue",
    "LectureLive",
    "SondeIntrouvable",
    "CaptureImpossible",
]

#: Codes de sortie de la sonde Rust (voir `rust/crates/pfs-capture/src/bin/probe.rs`).
_SORTIE_FENETRE_INTROUVABLE = 2
_SORTIE_ECHEC_CAPTURE = 3
_SORTIE_TIMEOUT = 4

#: Au-delà, on considère que le client ne rend pas de frame : fenêtre figée,
#: minimisée, ou capture bloquée par le client. 8 s laisse le temps à une
#: table calme (aucune animation) de rafraîchir au moins une fois.
_TIMEOUT_S = 8


class SondeIntrouvable(RuntimeError):
    """La sonde Rust n'est pas compilée."""


class CaptureImpossible(RuntimeError):
    """La fenêtre n'existe pas, ou n'a rendu aucune frame."""


def chemin_sonde() -> Path:
    """Emplacement de `probe.exe`, compilé en release.

    Raises
    ------
    SondeIntrouvable
        Si le binaire n'existe pas — le message dit quoi lancer.
    """
    racine = Path(__file__).resolve().parents[3]
    exe = racine / "rust" / "target" / "release" / "probe.exe"
    if not exe.exists():
        raise SondeIntrouvable(
            f"sonde absente : {exe}\n"
            "compile-la : cargo build --release --manifest-path rust/Cargo.toml"
        )
    return exe


def fenetres_disponibles() -> list[str]:
    """Titres des fenêtres capturables, les clients poker d'abord.

    Le tri met en tête ce qui ressemble à une table : c'est la seule chose
    qu'on cherche, et la liste brute de Windows en compte des dizaines.
    """
    r = subprocess.run([str(chemin_sonde()), "--list"],
                       capture_output=True, text=True, timeout=20,
                       encoding="utf-8", errors="replace")
    titres = [t.strip() for t in (r.stdout or "").splitlines() if t.strip()]
    indices = ("pmu", "winamax", "pokerstars", "partypoker", "unibet",
               "betclic", "888", "ggpoker", "ipoker", "hold", "omaha",
               "table", "twister", "expresso")
    exclus = ("fusion", "pfs-", "probe")

    def rang(t: str) -> int:
        b = t.lower()
        if any(e in b for e in exclus):
            return 2
        return 0 if any(i in b for i in indices) else 1

    return sorted(titres, key=lambda t: (rang(t), t.lower()))


def _forcer_redessin(titre: str | None) -> int:
    """Demande à la fenêtre cible de se redessiner. Renvoie le nombre touché.

    Windows Graphics Capture n'émet une frame **que** lorsque la fenêtre se
    redessine. Une table entre deux actions, un client en pause, une image
    ouverte dans une visionneuse : rien ne bouge, aucune frame n'arrive, et
    la capture expire alors que la fenêtre est parfaitement visible. C'est
    le mode d'échec le plus fréquent, et il n'a rien à voir avec un client
    qui bloquerait la capture.

    `RedrawWindow` avec `RDW_INVALIDATE | RDW_UPDATENOW` invalide la zone
    cliente et force le rendu immédiatement, y compris depuis un autre
    processus. La fenêtre n'est ni déplacée, ni activée, ni mise au premier
    plan : l'utilisateur ne voit rien.

    L'échec est silencieux et sans conséquence : certaines fenêtres
    (rendu GPU exclusif) ignorent la demande, et la capture est alors
    simplement tentée telle quelle.
    """
    if not sys.platform.startswith("win"):
        return 0
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    RDW_INVALIDATE, RDW_UPDATENOW, RDW_ALLCHILDREN = 0x0001, 0x0100, 0x0080

    besoin = (titre or "").lower()
    touchees = 0

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _visiter(hwnd, _):
        nonlocal touchees
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        tampon = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, tampon, n + 1)
        # Sans cible nommée on ne redessine rien : invalider toutes les
        # fenêtres du bureau pour capturer l'une d'elles serait grossier.
        if besoin and besoin in tampon.value.lower():
            user32.RedrawWindow(hwnd, None, None,
                                RDW_INVALIDATE | RDW_UPDATENOW
                                | RDW_ALLCHILDREN)
            touchees += 1
        return True

    user32.EnumWindows(_visiter, 0)
    return touchees


def capturer_fenetre(titre: str | None = None,
                     timeout_s: int = _TIMEOUT_S) -> bytes:
    """Capture une fenêtre et renvoie les octets PNG.

    Parameters
    ----------
    titre : str, optional
        Sous-chaîne du titre de la fenêtre. ``None`` déclenche la détection
        automatique d'un client poker connu (``probe --auto``).
    timeout_s : int
        Délai au-delà duquel on abandonne. Windows Graphics Capture n'émet
        une frame que lorsque la fenêtre se redessine : une table sans
        animation peut rester muette plusieurs secondes.

    Returns
    -------
    bytes
        Le PNG de la fenêtre entière.

    Raises
    ------
    CaptureImpossible
        Fenêtre absente, aucune frame, ou capture refusée par le client.
        Le message distingue les trois cas — ils appellent des gestes
        différents (ouvrir la table / cliquer dedans / rien à faire).
    """
    exe = chemin_sonde()
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "frame.png"
        cible = ["--auto"] if titre is None else [titre]
        # 1 frame suffit : la sonde écrit l'instantané dès la première.
        cmd = [str(exe), *cible, "1", "--snap", str(png),
               "--timeout", str(timeout_s)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
        # La session de capture met un instant à s'ouvrir : redessiner
        # avant qu'elle n'écoute ne servirait à rien. On insiste donc
        # pendant toute l'attente, jusqu'à ce que la frame tombe.
        fin = time.monotonic() + timeout_s
        octets: bytes | None = None
        while proc.poll() is None and time.monotonic() < fin:
            # Dès que l'instantané est écrit, l'attente n'a plus d'objet :
            # la sonde poursuit sa comptabilité, nous avons notre image.
            if png.exists() and png.stat().st_size > 0:
                lu = png.read_bytes()
                # Le PNG est écrit en une passe ; on vérifie tout de même
                # le marqueur de fin de fichier avant de s'en servir.
                if lu.endswith(b"IEND\xaeB`\x82"):
                    octets = lu
                    break
            _forcer_redessin(titre)
            time.sleep(0.1)

        try:
            sortie_err = proc.communicate(timeout=5)[1]
        except subprocess.TimeoutExpired:
            proc.kill()
            sortie_err = proc.communicate()[1]

        if octets is not None:
            return octets
        if png.exists() and png.stat().st_size > 0:
            return png.read_bytes()

        detail = (sortie_err or "").strip().splitlines()
        detail = detail[-1] if detail else "aucun message"
        if proc.returncode == _SORTIE_FENETRE_INTROUVABLE:
            raise CaptureImpossible(
                f"fenêtre introuvable ({detail}). Ouvre la table, puis "
                "recharge la liste des fenêtres.")
        if proc.returncode == _SORTIE_TIMEOUT:
            raise CaptureImpossible(
                f"la fenêtre n'a rendu aucune image en {timeout_s} s, "
                "malgré une demande de redessin. Si elle est minimisée, "
                "restaure-la ; sinon le client bloque la capture.")
        raise CaptureImpossible(f"capture refusée par le client ({detail}).")


@dataclass(frozen=True, slots=True)
class CarteLue:
    """Une carte détectée à l'écran et le degré de certitude de sa lecture.

    Attributes
    ----------
    role : str
        « hero », « board » ou « ? » — d'où vient la carte sur la table.
    carte : str or None
        La carte lue, notation courte (« Ah »). ``None`` si le recogniseur a
        refusé : mieux vaut rien qu'une carte fausse.
    candidat : str or None
        Ce qu'il aurait répondu sans le garde-fou. Sert au diagnostic : un
        candidat systématiquement juste veut dire que le seuil est trop
        strict, un candidat faux qu'il est bien placé.
    ecart, marge : int
        Distance de Hamming au meilleur gabarit, et écart au deuxième. La
        marge est le vrai signal de confiance : un écart faible avec deux
        candidats à égalité ne vaut rien.
    """

    role: str
    boite: tuple[int, int, int, int]
    carte: str | None
    candidat: str | None
    statut: str
    ecart: int | None
    marge: int | None

    @property
    def sure(self) -> bool:
        return self.statut == "sure"


@dataclass(frozen=True, slots=True)
class LectureLive:
    """Ce que la machine a vu — et rien d'autre.

    Aucun champ ne porte de recommandation : cette classe est le contrat qui
    garde la calibration séparée du conseil (voir l'en-tête du module).
    """

    fenetre: str
    largeur: int
    hauteur: int
    cartes: list[CarteLue] = field(default_factory=list)
    image_b64: str = ""

    @property
    def sures(self) -> int:
        return sum(1 for c in self.cartes if c.sure)

    @property
    def taux(self) -> float:
        """Part des cartes détectées qui ont été lues avec certitude."""
        return self.sures / len(self.cartes) if self.cartes else 0.0

    def resume(self) -> str:
        """Une ligne lisible, pour le terminal."""
        if not self.cartes:
            return "aucune carte détectée"
        par_role: dict[str, list[str]] = {}
        for c in self.cartes:
            par_role.setdefault(c.role, []).append(c.carte or "?")
        bouts = [f"{r} {' '.join(v)}" for r, v in sorted(par_role.items())]
        return f"{' | '.join(bouts)}  ({self.sures}/{len(self.cartes)} sûres)"


#: Réduire l'image avant de localiser : optimisation ESSAYÉE PUIS REJETÉE.
#:
#: Le détecteur raisonne en fractions de l'image (`MIN_AREA_FRAC`), donc
#: insensible à l'échelle, alors que son coût croît avec le nombre de
#: pixels. Réduire avant de chercher paraissait donc gratuit. Une première
#: mesure, sur une seule famille de tables, l'a semblé confirmer — et j'ai
#: publié « 1280 est un optimum ». C'était faux.
#:
#: Le banc complet (`banc_localisation.py --large`, 54 tables décorées
#: 2560×1529 par configuration) dit l'inverse — localisation / rôles ::
#:
#:     largeur      référence   deck classique   bruité   52×70   80×108
#:        640      15,5 / 14,3      2,4 /  0,0  16,3/11,9  0,0/0,0  50,0/21,4
#:        800      70,2 / 42,9     31,0 / 14,3  16,7/11,1  2,4/2,4  69,0/44,0
#:        960      70,2 / 42,9     65,5 / 51,2  62,7/38,1 33,3/6,0  78,6/66,3
#:       1280     100,0 /100,0     56,0 / 22,6  79,8/63,9 70,2/41,7 84,5/84,5
#:       1600     100,0 /100,0     97,6 / 95,2  96,4/94,0 97,6/95,2 100,0/100,0
#:     pleine     100,0 /100,0    100,0 /100,0 100,0/100,0 100,0/100,0 100,0/100,0
#:
#: Trois enseignements :
#:
#: * **La pleine échelle est la seule largeur qui tient partout** : 100 % de
#:   localisation ET de rôles dans les cinq configurations, sans exception.
#: * **1280 n'est pas un optimum**, ni même monotone : sur le deck classique
#:   il fait 56,0 % là où 960 fait 65,5 %. Le 100 % initial n'était qu'un
#:   accord fortuit avec la taille de carte (68×92) de mon premier banc.
#: * Le gain de temps est réel mais faible au regard de ce qu'il coûte :
#:   ~230 ms à 1280 contre ~700 ms en pleine échelle sur une table, pour
#:   jusqu'à 44 % des cartes perdues.
#:
#: La localisation se fait donc **à pleine échelle** par défaut. La réduction
#: reste accessible par le paramètre `largeur_detection` de `lire_ecran`,
#: pour le seul cas où elle se justifie : une fenêtre de bureau très chargée
#: (milliers d'arêtes verticales), où la localisation atteint 7,6 s. Une
#: table de poker, presque tout en feutre uni, coûte dix fois moins.
#:
#: Contrepartie assumée de la pleine échelle : elle produit ~0,83 boîte
#: fantôme par table (45 sur 54), là où 1280 n'en produit aucune. Toutes
#: atterrissent dans `others` — aucune n'est promue carte du héros ou du
#: board — et `lire_ecran` n'archive plus les échecs de `others`, pour que
#: le banc ne se remplisse pas de découpes de feutre.
LARGEUR_SANS_REDUCTION = 0


def _reduire(image, largeur: int):
    """Image ramenée à `largeur` au plus, et le facteur pour revenir.

    Returns
    -------
    tuple
        L'image réduite et le facteur multiplicatif qui ramène ses
        coordonnées vers l'image d'origine (1.0 si aucune réduction).
    """
    if image.width <= largeur:
        return image, 1.0
    from PIL import Image

    k = largeur / image.width
    reduite = image.resize((largeur, max(1, round(image.height * k))),
                           Image.LANCZOS)
    return reduite, image.width / reduite.width


def _agrandir(b: CardBox, k: float, largeur: int, hauteur: int) -> CardBox:
    """Boîte détectée sur l'image réduite, ramenée à l'échelle d'origine.

    La marge d'un pixel absorbe l'arrondi ; sans elle, une bordure de carte
    peut être rognée, ce qui suffit à faire échouer la reconnaissance.
    """
    x = max(0, round(b.x * k) - 1)
    y = max(0, round(b.y * k) - 1)
    w = min(largeur - x, round(b.w * k) + 2)
    h = min(hauteur - y, round(b.h * k) + 2)
    return CardBox(x=x, y=y, w=w, h=h, role=b.role)


def _lire_boites(image, boites: list[CardBox], role: str,
                 fenetre: str) -> list[CarteLue]:
    """Reconnaît chaque boîte, et archive celles qui échouent — sauf `others`.

    Une découpe de rôle inconnu qui n'est pas lue n'apprend rien : ce peut
    être une carte, mais aussi un dos d'adversaire (illisible par nature) ou
    une boîte fantôme posée sur du feutre. En archiver revient à remplir le
    banc d'essai de bruit, jusqu'à noyer les vrais échecs. Seuls les rôles
    identifiés — héros et board — méritent d'être conservés pour rejeu.
    """
    archivable = role in ("hero", "board")
    lues: list[CarteLue] = []
    for b in boites:
        decoupe = image.crop((b.x, b.y, b.x + b.w, b.y + b.h))
        m = identify_card(decoupe)
        lues.append(CarteLue(
            role=role, boite=(b.x, b.y, b.w, b.h),
            carte=m.card, candidat=m.best_guess, statut=m.statut,
            ecart=m.distance, marge=m.margin))
        if m.statut != "sure" and archivable:
            # L'échec est archivé tel qu'il a été soumis : c'est cette
            # découpe exacte qu'il faudra rejouer après correction.
            tampon = io.BytesIO()
            decoupe.save(tampon, format="PNG")
            enregistrer_echec(
                base64.b64encode(tampon.getvalue()).decode("ascii"),
                {"statut": m.statut, "distance": m.distance,
                 "margin": m.margin, "best_guess": m.best_guess,
                 "role": role, "origine": "live", "fenetre": fenetre,
                 "boite": [b.x, b.y, b.w, b.h]})
    return lues


def lire_ecran(titre: str | None = None,
               timeout_s: int = _TIMEOUT_S,
               largeur_detection: int = LARGEUR_SANS_REDUCTION) -> LectureLive:
    """Capture la fenêtre, localise les cartes, les lit, archive les échecs.

    C'est la boucle de calibration complète en un appel. Elle ne conseille
    rien : voir l'en-tête du module.

    Parameters
    ----------
    titre : str, optional
        Sous-chaîne du titre de la fenêtre ; ``None`` pour la détection
        automatique.
    largeur_detection : int
        Largeur à laquelle réduire l'image AVANT de localiser les cartes.
        ``0`` (défaut) = pas de réduction, seul réglage qui localise 100 %
        des cartes dans toutes les configurations mesurées. Ne réduire que
        sur une fenêtre de bureau très chargée, où la localisation à pleine
        échelle atteint plusieurs secondes — et en sachant ce que ça coûte :
        voir le tableau de `LARGEUR_SANS_REDUCTION`.

    Returns
    -------
    LectureLive
        Les cartes vues, leur rôle et la confiance de chaque lecture, plus
        la capture en base64 pour l'affichage.
    """
    from PIL import Image

    png = capturer_fenetre(titre, timeout_s)
    png_b64 = base64.b64encode(png).decode("ascii")
    image = Image.open(io.BytesIO(png)).convert("RGB")

    if largeur_detection:
        cherchee, k = _reduire(image, largeur_detection)
    else:
        cherchee, k = image, 1.0
    table = read_table(cherchee)

    cartes: list[CarteLue] = []
    for role, boites in (("hero", table.hero), ("board", table.board),
                         ("?", table.others)):
        # La reconnaissance se fait toujours sur la capture d'origine : les
        # boîtes sont remises à l'échelle avant découpe, pour ne perdre
        # aucun détail du glyphe.
        pleines = ([_agrandir(b, k, image.width, image.height)
                    for b in boites] if largeur_detection else list(boites))
        cartes.extend(_lire_boites(image, pleines, role, titre or "auto"))

    return LectureLive(fenetre=titre or "auto",
                       largeur=image.width, hauteur=image.height,
                       cartes=cartes, image_b64=png_b64)
