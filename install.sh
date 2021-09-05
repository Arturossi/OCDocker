#!/usr/bin/sh

# Download MGLtools, untar-it, create new dir if doesn't exist, untar the downloaded tar into created folder, delete tar, enter the folder and then source the installation file
wget https://ccsb.scripps.edu/download/532/ -O mgltools.tar.gz && mkdir -p mgltools && tar -xvzf mgltools.tar.gz -C mgltools --strip-components=1 && rm mgltools.tar.gz && cd mgltools && source ./install.sh 

# OBS: The scripts used to prepare ligand/receptor will be in the following dir: ``installation_dir/mgltools/MGLToolsPckgs/AutoDockTools`` 

# Download Autodock Vina, create new dir if doesn't exist, untar the downloaded tar into created folder and delete tar
wget http://vina.scripps.edu/download/autodock_vina_1_1_2_linux_x86.tgz -O vina.tgz && mkdir -p vina && tar -xvzf vina.tgz -C vina --strip-components=1 && rm vina.tgz

# OBS: The vina executable will be in the following dir: ``installation_dir/vina/bin`` 