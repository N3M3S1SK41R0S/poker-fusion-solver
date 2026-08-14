"""Mode Live — perception et garde-fou du conseil en direct (argent fictif).

Ce module fait TOUT ce que le mode Live exige AVANT le conseil, et rien du
conseil lui-même : résolution de la fenêtre, capture, preuve d'identité
PMU PLAY (badge pixel), décision du gate de conformité, lecture des cartes
et des montants. Il n'importe AUCUN module de recommandation — c'est
vérifiable d'un grep, et c'est voulu : la route qui l'appelle ne convoque le
conseiller QU'APRÈS un verdict ``mode == "live"``, jamais avant.

La frontière éthique, telle qu'elle est câblée ici :

- les routes historiques ``live/fenetres`` et ``live/lire`` restent SANS
  conseil (verrouillées par ``tests/test_live_sans_conseil.py``) ;
- le conseil en direct passe par la nouvelle route ``live/table``, derrière
  ``ComplianceGate.profil_pmu_play()`` : armement manuel par fenêtre (avec
  une confirmation explicite exigée ici même) ET badge « PMU PLAY » vu —
  fail-closed, toute panne ou doute rend REVIEW ;
- le badge a une mémoire de 60 s liée au couple (fenêtre, titre) : il est
  souvent caché par le board en cours de main (mesuré : 50/57 frames avec
  preuve immédiate, les 7 autres sont des fins de main), mais tout
  changement de fenêtre OU de titre purge la mémoire immédiatement ;
- le titre et les montants lus sont JOURNALISÉS dans la réponse — sur
  PMU PLAY ils affichent des euros fictifs et ne votent pas (voir la
  docstring du profil du gate), mais ils restent traçables.
"""

from __future__ import annotations

import ctypes
import io
import time
from dataclasses import dataclass

from pfs.compliance.gate import ComplianceGate, GateDecision, TableObservation
from pfs.vision import live as vision_live
from pfs.vision.badge_pmu import BadgeLu, detecter_badge
from pfs.vision.table_detector import CardBox, TableRead
from pfs.vision.zones_montants import lire_montants

__all__ = ["ModeLive", "BADGE_MEMOIRE_S"]

#: Durée de validité d'un badge vu, lié au couple (hwnd, titre). 60 s couvre
#: les mains où le board occulte le filigrane (mesuré sur les 57 captures :
#: aucune fenêtre de plus de 40 s sans l'un des deux marqueurs), sans jamais
#: survivre à un changement de fenêtre ou de titre.
BADGE_MEMOIRE_S = 60.0


def _hwnd_par_titre(titre: str) -> int:
    """Le handle de la première fenêtre visible dont le titre contient `titre`.

    Même mécanique que ``vision_live._forcer_redessin`` (EnumWindows par
    ctypes) : on résout un VRAI handle système, pas une chaîne — l'armement
    du gate est lié au hwnd, donc à la fenêtre physique, et meurt avec elle.
    """
    user32 = ctypes.windll.user32
    trouve: list[int] = []
    proto = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def visite(hwnd, _param):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        tampon = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, tampon, n + 1)
        if titre.lower() in tampon.value.lower():
            trouve.append(int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0))
            return False
        return True

    user32.EnumWindows(proto(visite), None)
    if not trouve:
        raise vision_live.CaptureImpossible(
            f"aucune fenêtre visible dont le titre contient {titre!r}")
    return trouve[0]


@dataclass(slots=True)
class _Souvenir:
    titre: str
    horodatage: float
    badge: BadgeLu


class ModeLive:
    """L'état du mode Live : gate, armements, mémoire du badge."""

    def __init__(self) -> None:
        self.gate = ComplianceGate.profil_pmu_play()
        self._souvenirs: dict[int, _Souvenir] = {}
        self._titres_armes: dict[int, str] = {}

    # ── armement ─────────────────────────────────────────────────────────
    def armer(self, titre: str, confirme: bool) -> dict:
        """Arme une fenêtre — exige la confirmation explicite de l'utilisateur.

        La case à cocher côté interface porte l'hypothèse que ni le badge ni
        le titre ne peuvent prouver : « je confirme que cette table PMU PLAY
        est en argent fictif ». Sans elle, refus — le gate n'est jamais armé
        en silence.
        """
        if confirme is not True:
            raise ValueError(
                "armement refusé : coche « Je confirme : table d'argent "
                "fictif (PMU PLAY) » — l'armement ne se fait jamais en "
                "silence.")
        hwnd = _hwnd_par_titre(titre)
        self.gate.arm_window(hwnd)          # lève GateLockedError si verrouillé
        self._titres_armes[hwnd] = titre
        return {"fenetre": titre, "hwnd": hwnd, "armee": True}

    def desarmer(self, titre: str | None = None) -> dict:
        """Désarme une fenêtre (ou toutes) ; purge aussi la mémoire du badge."""
        if titre is None:
            for h in list(self._titres_armes):
                self.gate.disarm_window(h)
            self._titres_armes.clear()
            self._souvenirs.clear()
            return {"armee": False, "toutes": True}
        cibles = [h for h, t in self._titres_armes.items()
                  if titre.lower() in t.lower()]
        for h in cibles:
            self.gate.disarm_window(h)
            self._titres_armes.pop(h, None)
            self._souvenirs.pop(h, None)
        return {"armee": False, "fenetres": len(cibles)}

    # ── badge avec mémoire ───────────────────────────────────────────────
    def _badge(self, hwnd: int, titre: str, image) -> tuple[bool | None, BadgeLu, float]:
        """Le badge de CETTE frame, ou son souvenir frais pour CETTE fenêtre.

        Rend (valeur pour le gate, lecture brute, âge en secondes). ``None``
        signifie « pas de preuve » — jamais « preuve d'absence ».
        """
        lu = detecter_badge(image)
        maintenant = time.monotonic()
        if lu.vu:
            self._souvenirs[hwnd] = _Souvenir(titre, maintenant, lu)
            return True, lu, 0.0
        s = self._souvenirs.get(hwnd)
        if s is not None and s.titre == titre \
                and maintenant - s.horodatage <= BADGE_MEMOIRE_S:
            return True, lu, maintenant - s.horodatage
        # Souvenir périmé ou titre changé : on l'oublie, fail-closed.
        if s is not None and s.titre != titre:
            self._souvenirs.pop(hwnd, None)
        return None, lu, float("inf")

    # ── la table, sans conseil ───────────────────────────────────────────
    def table(self, titre: str) -> dict:
        """Capture, prouve, lit — et rend un verdict de gate. AUCUN conseil.

        La réponse est à contrat clos, en trois blocs : ``gate`` (le verdict
        et chaque signal), ``lecture`` (cartes + montants, ou ``None`` si la
        capture échoue), et l'appelant décide s'il a le droit d'y adjoindre
        un conseil — uniquement si ``gate["mode"] == "live"``.
        """
        from PIL import Image

        hwnd = next((h for h, t in self._titres_armes.items()
                     if titre.lower() in t.lower()), 0)
        titre_reel = self._titres_armes.get(hwnd, titre)

        png = vision_live.capturer_fenetre(titre)
        image = Image.open(io.BytesIO(png)).convert("RGB")

        badge_gate, badge_lu, age = self._badge(hwnd, titre_reel, image)
        decision: GateDecision = self.gate.evaluate(TableObservation(
            hwnd=hwnd, window_title=titre_reel,
            play_money_badge_detected=badge_gate))

        # PAS de ``png=`` : ce paramètre ne sert qu'à remplir
        # ``lecture.image_b64``, que cette route ne renvoie pas (le contrat
        # de `live/table` n'a pas d'image — vérifié par le test de contrat
        # clos). L'encoder à chaque cycle serait du travail jeté.
        lecture = vision_live.lire_image(image, fenetre=titre_reel)
        table = TableRead(
            hero=[CardBox(c.boite[0], c.boite[1], c.boite[2], c.boite[3], "hero")
                  for c in lecture.cartes if c.role == "hero"],
            board=[CardBox(c.boite[0], c.boite[1], c.boite[2], c.boite[3], "board")
                   for c in lecture.cartes if c.role == "board"],
            others=[])
        montants = lire_montants(image, table)

        return {
            "gate": {
                "mode": decision.mode.value,
                "raison": decision.reason,
                "verrouille": self.gate.is_locked,
                "badge_vu": bool(badge_gate),
                "badge_filigrane": round(badge_lu.filigrane_score, 3),
                "badge_dos": round(badge_lu.dos_score, 3),
                "badge_age_s": None if age == float("inf") else round(age, 1),
                "badge_refus": badge_lu.refus,
                "signaux": [{"source": r.source, "verdict": r.verdict.value,
                             "detail": r.detail} for r in decision.readings],
            },
            "lecture": {
                "fenetre": lecture.fenetre,
                "largeur": lecture.largeur, "hauteur": lecture.hauteur,
                "sures": lecture.sures, "total": len(lecture.cartes),
                "cartes": [{"role": c.role, "carte": c.carte,
                            "statut": c.statut, "boite": list(c.boite)}
                           for c in lecture.cartes],
                "main": [c.carte for c in lecture.cartes
                         if c.role == "hero" and c.carte],
                "tableau": [c.carte for c in lecture.cartes
                            if c.role == "board" and c.carte],
                "montants": montants,
            },
        }
