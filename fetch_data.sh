#!/usr/bin/env bash
# Download exactly the StatsBomb Open Data files this project uses, at a pinned commit, and verify every file
# against its git blob hash (data_manifest.tsv). About 4.3 GB on disk.
# Usage: ./fetch_data.sh [destination]   (default: ./statsbomb-open-data)
# Please read and accept the StatsBomb Public Data User Agreement (LICENSE.pdf in the download) before use.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$HERE/statsbomb-open-data}"
COMMIT=4b73468fc5b0f1950f9f66fada70ad3a4f9327cb
PY="${PYTHON:-python3}"

if [ ! -d "$DEST/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/statsbomb/open-data.git "$DEST"
fi
cd "$DEST"
git sparse-checkout init --no-cone
grep -v '^#' "$HERE/data_manifest.tsv" | cut -f1 | sed 's|^|/|' | git sparse-checkout set --no-cone --stdin
git -c advice.detachedHead=false checkout "$COMMIT"
"$PY" "$HERE/src/verify_data.py" "$DEST"
