#!/usr/bin/env bash

# Description
###############################################################################
# OCDocker installer (Ubuntu-like systems)
# - Installs MGLTools, AutoDock Vina, Miniconda (in $HOME/miniconda)
# - Installs DSSP and MySQL server via apt
# - Creates the conda environment defined in environment.yml
# - Creates MySQL user/database for OCDocker

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
export DB_NAME_OPTIMIZATION=ocdocker

info "Starting the installation process..."

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
step "Installing required system packages (dssp, mysql-server)..." \
  "sudo apt-get update -y && sudo apt-get install -y dssp mysql-server"

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

# Step 8: Configure MySQL
step "Configuring MySQL: create user and databases..." \
  "sudo mysql -u root -e \"CREATE USER IF NOT EXISTS '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';\" && \
   sudo mysql -u root -e \"CREATE DATABASE IF NOT EXISTS ${DB_NAME};\" && \
   sudo mysql -u root -e \"GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';\" && \
   sudo mysql -u root -e \"CREATE DATABASE IF NOT EXISTS ${DB_NAME_OPTIMIZATION};\" && \
   sudo mysql -u root -e \"GRANT ALL PRIVILEGES ON ${DB_NAME_OPTIMIZATION}.* TO '${DB_USER}'@'localhost';\" && \
   sudo mysql -u root -e \"FLUSH PRIVILEGES;\""
info "MySQL configuration step finished."

# Step 9: Activate the environment
step "Activating the conda environment..." \
  "conda activate ocdocker"
info "Conda environment activation step finished."
info "Installation process finished (with prompts on failures)."
