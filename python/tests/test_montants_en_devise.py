"""Un montant en devise ne doit pas interrompre la lecture d'un historique.

Sur les tournois à primes, le client PMU écrit des montants suffixés de leur
devise (« 120€ ») dans des champs par ailleurs numériques. `float()` levait
alors `ValueError` — et comme `iter_hands` est un générateur, l'exception
emportait **toutes les mains suivantes du fichier**, sans le moindre message.

Constaté sur une session réelle : 5 fichiers interrompus, dont un de
171 mains. La récupération complète est passée de 427 à 721 mains une fois
les deux conversions durcies (`_num` et `_ipoker_amount`).

C'est le mode d'échec le plus coûteux d'un parseur : il ne se plaint pas, il
rend moins.
"""

from __future__ import annotations

import pytest

from pfs.data.hand_history import _ipoker_amount, _num


@pytest.mark.parametrize("brut,attendu", [
    ("120€", 120.0),
    ("1 355", 1355.0),          # espace fine insécable = milliers
    ("1 234,50", 1234.5),       # virgule décimale
    ("10€", 10.0),
    ("$40", 40.0),
    ("-25,5", -25.5),
    ("0", 0.0),
])
def test_les_montants_se_lisent(brut, attendu) -> None:
    assert _num(brut) == pytest.approx(attendu)
    assert _ipoker_amount(brut) == pytest.approx(attendu)


@pytest.mark.parametrize("brut", [None, "", "  ", "€", "abc", "-", "."])
def test_un_montant_illisible_vaut_zero_sans_lever(brut) -> None:
    """Perdre un champ vaut mieux que perdre le reste du fichier."""
    assert _num(brut) == 0.0
    assert _ipoker_amount(brut) == 0.0


def test_la_lecture_d_un_fichier_survit_a_un_montant_en_devise() -> None:
    """Le cas complet : une prime en euros au milieu d'un historique iPoker.

    Sans le durcissement, la première main passait et la seconde emportait
    tout. Le test vérifie donc qu'on lit bien les DEUX.
    """
    from pfs.data.hand_history import iter_hands

    xml = """<?xml version="1.0" encoding="utf-8"?>
<session sessioncode="1">
 <general><mode>real</mode><gametype>Holdem NL</gametype>
  <tablename>BANC, 1</tablename><nickname>heros</nickname>
  <bets>120€</bets></general>
 <game gamecode="1">
  <general><startdate>2026-08-11 00:00:00</startdate>
   <players>
    <player seat="1" name="heros" chips="1000" dealer="1" win="0" bet="10"/>
    <player seat="2" name="vilain" chips="1000" dealer="0" win="20" bet="10"
            bounty="120€"/>
   </players></general>
  <round no="1">
   <action no="1" player="heros" type="3" sum="10"/>
   <action no="2" player="vilain" type="4" sum="0"/>
  </round>
 </game>
 <game gamecode="2">
  <general><startdate>2026-08-11 00:01:00</startdate>
   <players>
    <player seat="1" name="heros" chips="990" dealer="0" win="0" bet="10"
            bounty="150€"/>
    <player seat="2" name="vilain" chips="1010" dealer="1" win="20" bet="10"/>
   </players></general>
  <round no="1">
   <action no="1" player="heros" type="0" sum="0"/>
  </round>
 </game>
</session>"""
    mains = list(iter_hands(xml))
    assert len(mains) == 2, (
        f"{len(mains)} main(s) lue(s) au lieu de 2 : un montant en devise "
        "interrompt encore la lecture")
