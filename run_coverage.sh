#!/usr/bin/env bash

# OCDocker — Coverage Runner
# Usage: ./run_coverage.sh [pytest-args]
# Example: ./run_coverage.sh -q -k db_export

set -euo pipefail

# Resolve repo root (directory containing this script)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

# Ensure a temp folder for SQLite (kept out of /tmp to avoid sandbox purges)
TMP_COV_DIR=".coverage_tmp"
SQLITE_DIR="$TMP_COV_DIR/sqlite"
mkdir -p "$SQLITE_DIR"
export OCDOCKER_USE_SQLITE=1
export OCDOCKER_SQLITE_PATH="$SQLITE_DIR/ocdocker.db"

# Avoid heavy model/bootstrap work during tests
export OCDOCKER_SKIP_ODDT=1
export OC_BUILD_DOCS=0

# Make sure coverage uses the project rcfile
RCFILE=".coveragerc"
if [[ ! -f "$RCFILE" ]]; then
  echo "[run]" > "$RCFILE"
  echo "branch = True" >> "$RCFILE"
  echo "source =\n    OCDocker" >> "$RCFILE"
fi

echo "==> Erasing previous coverage data"
coverage erase || true

echo "==> Running tests under coverage"
# Pass any extra pytest args through
coverage run --rcfile "$RCFILE" -m pytest "$@"

echo "==> Combining coverage data (safe if single file)"
coverage combine || true

echo "==> Generating terminal report"
coverage report -m

echo "==> Generating HTML report (htmlcov/)"
coverage html -d htmlcov

echo "==> Generating XML report (coverage.xml)"
coverage xml -o coverage.xml

echo "Done. Open htmlcov/index.html for a detailed report."
