#!/usr/bin/env bash
# Creates a project-local .venv and installs runtime dependencies.
# Usage (from repo root):
#   bash env/setup_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
REQ_PATH="$REPO_ROOT/env/requirements.txt"

if [[ ! -f "$REQ_PATH" ]]; then
  echo "Could not find env/requirements.txt" >&2
  exit 1
fi

echo "==> Repo root: $REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on PATH. Install Python 3.11+ and retry." >&2
  exit 1
fi

VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "==> Detected Python $VERSION"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "==> Creating virtual environment at .venv"
  python3 -m venv "$VENV_PATH"
else
  echo "==> Reusing existing .venv (delete .venv to recreate from scratch)"
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

echo "==> Upgrading pip"
python -m pip install --upgrade pip

echo "==> Installing requirements from env/requirements.txt"
python -m pip install -r "$REQ_PATH"

echo "==> Installing project package (editable)"
python -m pip install -e "$REPO_ROOT"

echo "==> Verifying key imports"
python -c "import pandas, openpyxl, requests, sklearn, rapidfuzz; print('OK: pandas, openpyxl, requests, sklearn, rapidfuzz')"

cat <<'EOF'

Environment ready.
Next steps:
  1. Activate:  source .venv/bin/activate
  2. (Optional) Add On3 recruiting CSVs under data/raw/recruiting/
  3. Compile:    python -m rookie_ppr.compile
  4. Outputs:    data/output/rookie_ppr_master.xlsx and data/output/csv/
  5. Scorer UI:  python -m rookie_ppr.ui
     (or:        rookie-ppr-ui)
EOF
