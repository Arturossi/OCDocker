#!/usr/bin/env bash

# OCDocker — Coverage Runner
# Usage: ./run_coverage.sh [pytest-args]
# Example: ./run_coverage.sh -q -k db_export

set -euo pipefail

# Pick a python interpreter (allow override via PYTHON_BIN)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Python not found in PATH. Activate your venv first." >&2
    exit 1
  fi
fi

require_module() {
  local module="$1"
  local install_hint="$2"
  if ! "$PYTHON_BIN" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)"; then
    echo "Missing Python module: $module" >&2
    echo "Install it with: $install_hint" >&2
    exit 1
  fi
}

# Resolve repo root (directory containing this script)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Ensure a temp folder for SQLite (kept out of /tmp to avoid sandbox purges)
TMP_COV_DIR=".coverage_tmp"
SQLITE_DIR="$TMP_COV_DIR/sqlite"
mkdir -p "$SQLITE_DIR"
export OCDOCKER_DB_BACKEND=sqlite
export OCDOCKER_SQLITE_PATH="$SQLITE_DIR/ocdocker.db"

# Avoid heavy model/bootstrap work during tests
export OCDOCKER_SKIP_ODDT=1
export OC_BUILD_DOCS=0

# Ensure required tools are present in the active interpreter
DEV_INSTALL_HINT="${PYTHON_BIN} -m pip install -e '.[dev]'"
require_module coverage "$DEV_INSTALL_HINT"
require_module pytest "$DEV_INSTALL_HINT"

# Make sure coverage uses the project rcfile
RCFILE=".coveragerc"
if [[ ! -f "$RCFILE" ]]; then
  echo "[run]" > "$RCFILE"
  echo "branch = True" >> "$RCFILE"
  echo "source =\n    OCDocker" >> "$RCFILE"
fi

echo "==> Erasing previous coverage data"
"$PYTHON_BIN" -m coverage erase || true

echo "==> Running tests under coverage"
# Pass any extra pytest args through
"$PYTHON_BIN" -m coverage run --rcfile "$RCFILE" -m pytest "$@"

echo "==> Combining coverage data (safe if single file)"
"$PYTHON_BIN" -m coverage combine || true

echo "==> Generating terminal report"
"$PYTHON_BIN" -m coverage report -m

echo "==> Generating HTML report (htmlcov/)"
"$PYTHON_BIN" -m coverage html -d htmlcov

echo "==> Generating XML report (coverage.xml)"
"$PYTHON_BIN" -m coverage xml -o coverage.xml

echo "Done. Open htmlcov/index.html for a detailed report."
