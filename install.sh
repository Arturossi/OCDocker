#!/usr/bin/sh

# Download MGLtools, untar-it, create new dir if doesn't exist, untar the downloaded tar into created folder, delete tar, enter the folder and then source the installation file
wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz && mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz && cd mgltools && source ./install.sh

# OBS: The scripts used to prepare ligand/receptor will be in the following dir: ``installation_dir/mgltools/MGLToolsPckgs/AutoDockTools``

# Download Autodock Vina, create new dir if doesn't exist, untar the downloaded tar into created folder and delete tar
#wget http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz -O vina.tgz && mkdir -p vina && tar -xvzf vina.tgz -C vina --strip-components=1 && rm vina.tgz
mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina && sudo cp vina/vina /usr/bin/vina

# OBS: The vina executable will be in the following dir: ``installation_dir/vina/bin``
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh && sudo chmod 777 Miniconda3.sh && ./Miniconda3.sh

conda activate ocdocker
conda update -n base -c defaults conda
conda install numpy matplotlib pandas scikit-learn tqdm pip
conda install -c conda-forge rdkit sklearn openbabel biopython

sudo apt install dssp
