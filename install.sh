#!/usr/bin/sh

# Set the Database user, password, and database name
export DB_USER=ocdocker
export DB_PASS=ocdocker
export DB_NAME=ocdocker

echo "Starting the installation process..."

# Step 1: Download and install MGLTools
echo "Downloading and installing MGLTools..."
wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz
if [ $? -ne 0 ]; then
    echo "Failed to download MGLTools."
    exit 1
fi

mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz
if [ $? -ne 0 ]; then
    echo "Failed to extract MGLTools."
    exit 1
fi

cd mgltools && source ./install.sh
echo "MGLTools installation complete."

# Step 2: Download and install Autodock Vina
echo "Downloading and setting up Autodock Vina..."
mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina
if [ $? -ne 0 ]; then
    echo "Failed to download Autodock Vina."
    exit 1
fi

sudo cp vina/vina /usr/bin/vina
sudo chmod +x /usr/bin/vina
if [ $? -ne 0 ]; then
    echo "Failed to set up Autodock Vina."
    exit 1
fi

echo "Autodock Vina setup complete."

# Step 3: Download and install Miniconda
echo "Downloading and installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh
if [ $? -ne 0 ]; then
    echo "Failed to download Miniconda."
    exit 1
fi

sudo chmod +x Miniconda3.sh && ./Miniconda3.sh -b -p $HOME/miniconda

if [ $? -ne 0 ]; then
    echo "Failed to install Miniconda."
    exit 1
fi

echo "Miniconda installation complete."

# Step 4: Initialize conda
echo "Initializing conda..."
source "$HOME/miniconda/etc/profile.d/conda.sh"
if [ $? -ne 0 ]; then
    echo "Failed to initialize conda."
    exit 1
fi

# Step 5: Install necessary system packages
echo "Installing required system packages (dssp, mysql-server)..."
sudo apt install -y dssp mysql-server
if [ $? -ne 0 ]; then
    echo "Failed to install system packages."
    exit 1
fi

# Step 6: Install mamba
echo "Installing mamba..."
conda install -y conda-forge::mamba
if [ $? -ne 0 ]; then
    echo "Failed to install mamba."
    exit 1
fi

# Step 7: Create the environment from the YAML file
echo "Creating the environment from the environment_11.yml file..."
mamba env create -f environment_11.yml
if [ $? -ne 0 ]; then
    echo "Failed to create conda environment."
    exit 1
fi

# Step 8: Configure MySQL
echo "Configuring MySQL, creating user and database..."
sudo mysql -u root -e "CREATE USER '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASS}';"
if [ $? -ne 0 ]; then
    echo "Failed to create MySQL user."
    exit 1
fi

sudo mysql -u root -e "CREATE DATABASE ${DB_NAME};"
if [ $? -ne 0 ]; then
    echo "Failed to create MySQL database."
    exit 1
fi

sudo mysql -u root -e "GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';"
if [ $? -ne 0 ]; then
    echo "Failed to grant privileges on MySQL database."
    exit 1
fi

sudo mysql -u root -e "FLUSH PRIVILEGES;"
if [ $? -ne 0 ]; then
    echo "Failed to flush MySQL privileges."
    exit 1
fi

echo "MySQL configuration complete."

# Step 9: Activate the environment
echo "Activating the conda environment..."
conda activate ocdocker
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment."
    exit 1
fi
echo "Conda environment activated."
echo "Installation process finished successfully!"
