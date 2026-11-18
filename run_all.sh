#!/usr/bin/env bash
# Full pipeline, from the raw StatsBomb files to figures, tables and the interactive page.
# Usage: ./run_all.sh [statsbomb-open-data checkout]   (run ./fetch_data.sh first)
# Runtime on 2 CPU cores: about 1.5 hours (action-value model ~17 min, Bayesian fits ~1 hour). 8 GB RAM recommended.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SB="${1:-$HERE/statsbomb-open-data}"
PY="${PYTHON:-python3}"
cd "$HERE"
"$PY" src/verify_data.py "$SB"
"$PY" src/actions.py "$SB/data"          # actions, minutes, matches, player names
"$PY" src/vaep.py | tee results/vaep.log   # action-value model, leave-one-league-out, first halves only
"$PY" src/ratings.py | tee results/ratings.log   # Bayesian ratings, first half -> second half validation
"$PY" src/publish_ratings.py | tee results/publish.log   # full-season ratings (model chosen by the rule)
"$PY" src/figures.py
"$PY" explorer/export.py
"$PY" explorer/build.py
"$PY" -m pytest -q
