#!/usr/bin/env bash

# Description
###############################################################################
# OCDocker installer (Ubuntu-like systems)
# - Installs MGLTools, AutoDock Vina, Miniconda (in $HOME/miniconda)
# - Installs DSSP and MySQL server via apt
# - Creates the conda environment defined in environment.yml
# - Creates MySQL user/database for OCDocker

set -euo pipefail
IFS=$'\n\t'

# Utilities
###############################################################################
die() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }

# Resolve script directory to find environment.yml reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set the Database user, password, and database name
export DB_USER=ocdocker
export DB_PASS=ocdocker
export DB_NAME=ocdocker

info "Starting the installation process..."

# Step 1: Download and install MGLTools
info "Downloading and installing MGLTools..."
wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz
[[ -s mgltools.tar.gz ]] || die "Failed to download MGLTools."

mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz
[[ -d mgltools ]] || die "Failed to extract MGLTools."

pushd mgltools >/dev/null
source ./install.sh || die "MGLTools installer failed"
popd >/dev/null
info "MGLTools installation complete."

# Step 2: Download and install Autodock Vina
info "Downloading and setting up AutoDock Vina..."
mkdir -p vina
wget -O vina/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 || die "Failed to download AutoDock Vina"
chmod +x vina/vina
sudo install -m 0755 vina/vina /usr/bin/vina || die "Failed to install /usr/bin/vina"
info "AutoDock Vina setup complete."

# Step 3: Download and install Miniconda
info "Downloading and installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
[[ -s Miniconda3.sh ]] || die "Failed to download Miniconda"
chmod +x Miniconda3.sh
./Miniconda3.sh -b -p "$HOME/miniconda" || die "Failed to install Miniconda"
info "Miniconda installation complete."

# Step 4: Initialize conda
info "Initializing conda..."
source "$HOME/miniconda/etc/profile.d/conda.sh" || die "Failed to initialize conda"

# Step 5: Install necessary system packages
info "Installing required system packages (dssp, mysql-server)..."
sudo apt-get update -y
sudo apt-get install -y dssp mysql-server || die "Failed to install system packages"

# Step 6: Install mamba
info "Installing mamba..."
conda install -y -n base -c conda-forge mamba || die "Failed to install mamba"

# Step 7: Create the environment from the YAML file
info "Creating the environment from the environment.yml file..."
if conda env list | grep -q "^ocdocker\s"; then
    info "Conda env 'ocdocker' already exists; skipping creation."
else
    mamba env create -f "$SCRIPT_DIR/environment.yml" || die "Failed to create conda environment"
fi

# Step 8: Configure MySQL
info "Configuring MySQL, creating user and database..."
sudo mysql -u root -e "CREATE USER IF NOT EXISTS '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';" || die "Failed to create MySQL user"
sudo mysql -u root -e "CREATE DATABASE IF NOT EXISTS ${DB_NAME};" || die "Failed to create MySQL database"
sudo mysql -u root -e "GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';" || die "Failed to grant privileges"
sudo mysql -u root -e "FLUSH PRIVILEGES;" || die "Failed to flush MySQL privileges"
info "MySQL configuration complete."

# Step 9: Activate the environment
info "Activating the conda environment..."
conda activate ocdocker || die "Failed to activate conda environment"
info "Conda environment activated."
info "Installation process finished successfully!"
