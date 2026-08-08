#!/usr/bin/env bash
# Poker Fusion Solver — bootstrap Phase 0
set -euo pipefail

echo "> 1/4  Verification des prerequis"
command -v cargo >/dev/null || { echo "  X Rust absent -> https://rustup.rs"; exit 1; }
command -v uv    >/dev/null || { echo "  > Installation de uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

echo "> 2/4  Environnement Python"
(cd python && uv sync --extra dev)

echo "> 3/4  Tests (doivent etre verts AVANT d'ecrire quoi que ce soit)"
(cd python && uv run pytest -q)
(cd rust   && cargo test --workspace --quiet)

echo "> 4/4  Demonstration"
(cd python && uv run python demo.py)

cat <<'MSG'

----------------------------------------------------------------
  Phase 0 operationnelle.

  Rappels du Plan Directeur v2.0 :
   * Passe ton bureau Windows a 144 Hz  ->  -8 ms de latence, gratuit
   * Tranche la §1 (mode d'operation) et la §6.1 (licence)
   * Phase 1.1 : capturer une fenetre Winamax OCCULTEE, p95 < 1 ms
     -> c'est le test de faisabilite du projet entier
   * Construis le harnais vision<->hand-history AVANT la vision
----------------------------------------------------------------
MSG
