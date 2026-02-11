#!/usr/bin/env bash

# Description
###############################################################################
# OCDocker installer (Ubuntu-like systems)
# - Installs MGLTools, AutoDock Vina, Miniconda (in $HOME/miniconda)
# - Installs DSSP and a SQL server (PostgreSQL by default; MySQL optional) via apt
# - Creates the conda environment defined in environment.yml
# - Creates SQL user/databases for OCDocker

set -uo pipefail
IFS=$'\n\t'

# Utilities
###############################################################################
info() { echo "[INFO] $*"; }

# Run a step and, on failure, prompt whether to continue.
# Usage: step "Description" <command...>
step() {
  local desc="$1"; shift
  info "$desc"
  # Run the command block under pipefail to catch failures in pipelines
  bash -o pipefail -c "$*"
  local ec=$?
  if [ $ec -ne 0 ]; then
    echo "[WARN] Step failed: $desc (exit $ec)" >&2
    read -r -p "Continue anyway? [y/N] " ans
    case "$ans" in
      y|Y|yes|YES) echo "[INFO] Continuing after failure of: $desc";;
      *) echo "[ABORT] Stopping at failed step: $desc"; exit $ec;;
    esac
  fi
}

# Resolve script directory to find environment.yml reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set the Database user, password, and database name
export DB_USER=ocdocker
export DB_NAME=ocdocker
export DB_PASS=ocdocker
export DB_NAME_OPTIMIZATION=optimization

# Decide DB mode.
# Backend precedence: env OCDOCKER_DB_BACKEND/DB_BACKEND, then config DB_BACKEND, then postgresql.
normalize_backend() {
  case "${1:-}" in
    postgresql|postgres|pgsql) echo "postgresql" ;;
    mysql|mariadb) echo "mysql" ;;
    sqlite|sqlite3) echo "sqlite" ;;
    *) echo "" ;;
  esac
}
cfg_get() {
  local key="$1"
  [[ -f "OCDocker.cfg" ]] || return 0
  grep -E "^\s*${key}\s*=\s*" OCDocker.cfg | tail -n1 | awk -F= '{print $2}' | xargs
}

SELECTED_DB_BACKEND="postgresql"
raw_backend="${OCDOCKER_DB_BACKEND:-${DB_BACKEND:-}}"
if [[ -z "${raw_backend}" ]]; then
  raw_backend="$(cfg_get DB_BACKEND)"
fi
norm_backend="$(normalize_backend "${raw_backend}")"
if [[ -n "${norm_backend}" ]]; then
  SELECTED_DB_BACKEND="${norm_backend}"
fi

info "Starting the installation process..."
if [[ "${SELECTED_DB_BACKEND}" == "sqlite" ]]; then
  info "Database mode: SQLite"
else
  info "Database backend: ${SELECTED_DB_BACKEND}"
fi

# Step 1: Download and install MGLTools
step "Downloading and installing MGLTools..." \
  "wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz && \
   [[ -s mgltools.tar.gz ]] && \
   mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz && \
   pushd mgltools >/dev/null && source ./install.sh && popd >/dev/null"
info "MGLTools installation step finished."

# Step 2: Download and install Autodock Vina
step "Downloading and setting up AutoDock Vina..." \
  "mkdir -p vina && \
   wget -O vina/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 && \
   chmod +x vina/vina && \
   sudo install -m 0755 vina/vina /usr/bin/vina"
info "AutoDock Vina setup step finished."

# Step 3: Download and install Miniconda
step "Downloading and installing Miniconda..." \
  "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh && \
   [[ -s Miniconda3.sh ]] && chmod +x Miniconda3.sh && \
   ./Miniconda3.sh -b -p \"$HOME/miniconda\""
info "Miniconda installation step finished."

# Step 4: Initialize conda
step "Initializing conda..." \
  "source \"$HOME/miniconda/etc/profile.d/conda.sh\""

# Step 5: Install necessary system packages
if [[ "${SELECTED_DB_BACKEND}" == "sqlite" ]]; then
  step "Installing required system packages (openbabel, libopenbabel-dev, swig, dssp)..." \
    "sudo apt-get update -y && sudo apt-get install -y openbabel libopenbabel-dev swig dssp"
  info "SQLite mode selected — skipping SQL server installation."
elif [[ "${SELECTED_DB_BACKEND}" == "mysql" ]]; then
  step "Installing required system packages (openbabel, libopenbabel-dev, swig, dssp, mysql-server)..." \
    "sudo apt-get update -y && sudo apt-get install -y openbabel libopenbabel-dev swig dssp mysql-server"
else
  step "Installing required system packages (openbabel, libopenbabel-dev, swig, dssp, postgresql, postgresql-contrib)..." \
    "sudo apt-get update -y && sudo apt-get install -y openbabel libopenbabel-dev swig dssp postgresql postgresql-contrib"
fi

# Step 6: Install mamba
step "Installing mamba..." \
  "conda install -y -n base -c conda-forge mamba"

# Step 7: Create the environment from the YAML file
if conda env list | grep -q "^ocdocker\s"; then
    info "Conda env 'ocdocker' already exists; skipping creation."
else
    step "Creating conda environment 'ocdocker' from environment.yml..." \
      "mamba env create -f \"$SCRIPT_DIR/environment.yml\""
fi

# Step 8: Configure SQL backend
if [[ "${SELECTED_DB_BACKEND}" == "sqlite" ]]; then
  info "SQLite mode selected — skipping SQL user/database configuration."
elif [[ "${SELECTED_DB_BACKEND}" == "mysql" ]]; then
  step "Configuring MySQL: create user and databases..." \
    "sudo mysql -u root -e \"CREATE USER IF NOT EXISTS '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';\" && \
     sudo mysql -u root -e \"CREATE DATABASE IF NOT EXISTS ${DB_NAME};\" && \
     sudo mysql -u root -e \"GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';\" && \
     sudo mysql -u root -e \"CREATE DATABASE IF NOT EXISTS ${DB_NAME_OPTIMIZATION};\" && \
     sudo mysql -u root -e \"GRANT ALL PRIVILEGES ON ${DB_NAME_OPTIMIZATION}.* TO '${DB_USER}'@'localhost';\" && \
     sudo mysql -u root -e \"FLUSH PRIVILEGES;\""
  info "MySQL configuration step finished."
else
  step "Configuring PostgreSQL: create role and databases..." \
    "sudo -u postgres psql -v ON_ERROR_STOP=1 -c \"DO \\\$\\\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASS}'; END IF; END \\\$\\\$;\" && \
     if ! sudo -u postgres psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'\" | grep -qx 1; then sudo -u postgres psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\"; fi && \
     if ! sudo -u postgres psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME_OPTIMIZATION}'\" | grep -qx 1; then sudo -u postgres psql -v ON_ERROR_STOP=1 -c \"CREATE DATABASE ${DB_NAME_OPTIMIZATION} OWNER ${DB_USER};\"; fi"
  info "PostgreSQL configuration step finished."
fi

# Step 9: Activate the environment
step "Activating the conda environment..." \
  "conda activate ocdocker"
info "Conda environment activation step finished."
info "Installation process finished (with prompts on failures)."
