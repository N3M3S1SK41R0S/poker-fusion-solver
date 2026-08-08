"""Point d'entrée : ``python -m pfs``."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pfs",
        description="Poker Fusion Solver — interface locale (127.0.0.1 uniquement).",
    )
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--no-browser", action="store_true",
                   help="ne pas ouvrir le navigateur automatiquement")
    p.add_argument("--demo", action="store_true",
                   help="exécuter la démonstration des fusions en console")
    p.add_argument("--selftest", action="store_true",
                   help="vérifier les valeurs golden puis quitter")
    a = p.parse_args(argv)

    if a.demo:
        from demo import main as demo_main  # type: ignore
        demo_main()
        return 0

    if a.selftest:
        from pfs.selftest import run_selftest
        return 0 if run_selftest() else 1

    from pfs.app.server import run
    run(port=a.port, open_browser=not a.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
