![OCDocker](./OCDocker.png "OCDocker")

OCDocker installation
=====================

Simplest methods
----------------

Conda
-----

OCDocker is a conda package, so the simplest way to install it is to use conda. If you do not have conda installed, please follow the instructions at https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html.
If you have conda installed, you can install OCDocker with the following command:

	```bash
	$ conda install arturossi/label/prealpha::ocdocker 
	```

If you have mamba installed, you can install OCDocker with the following command:

	```bash
	$ mamba install arturossi/label/prealpha::ocdocker 
	```

pip
---

If you prefer to use pip, you can install OCDocker with the following command:

	```bash
	$ pip install ocdocker
	```

From source
-----------

Download the source code from the GitHub repository:

	```bash
	$ git clone https://github.com/Arturossi/OCDocker
	```

Go to OCDocker dir and execute the install.sh file with the following command. (Yes, need to be sudo... sorry :/)

	```bash
	$ sudo sh ./install.sh
	```

Log out then in (or open a new shell env) to activate conda env (base).

Install the conda OCDocker env

	```bash
	$ conda env create -f environment.yml
	```

Or perform each software installation manually with the below steps.

Download and install MGLTools (or ADFR, which is still being tested)
------------------------------------------------------------------------------------

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

	```bash
	$ wget https://ccsb.scripps.edu/download/292/ --no-check-certificate -O mgltools_install
	```

* Option 2 (For those who love to follow each step)

	- Download the file

	```bash
	$ wget https://ccsb.scripps.edu/download/532/ --no-check-certificate -O mgltools_install.tar.gz
	```

	- Untar it:

	```bash
	$ tar -xvzf mgltools_install.tar.gz
	```

	- cd into created dir

	```bash
	$ cd mgltools_x86_64Linux2_1.5.X
	```

	- source the install.sh

	```bash
	$ source ./install.sh
	```

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

```bash
$ wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz --no-check-certificate && mkdir -p mgltools && tar -xvzf mgltools_install.tar.gz -C mgltools --strip-components=1 && rm mgltools_install.tar.gz && cd mgltools && source ./install.sh
```

OBS: The scripts used to prepare ligand/receptor will be in the following dir: ``<installation_dir>/mgltools/MGLToolsPckgs/AutoDockTools``

> :warning: **Still cannot run MGLTools?**: If you are facing some shady problems such as the numpy one, you might have to compile MGLTools from source. You can download it at https://github.com/genome-vendor/MGLtools (Still not sure about its version... I do not know if it is 1.5.6 or 1.5.4)

Download and install ADFRtools (in test)
----------------------------------------

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

	```bash
	$ wget https://ccsb.scripps.edu/adfr/download/1028/ --no-check-certificate -O adfr_install
	```

* Option 2 (For those who love to follow each step)

	- Download the file

	```bash
	$ wget https://ccsb.scripps.edu/adfr/download/1038/ --no-check-certificate -O adfr_install.tar.gz
	```

	- Untar it and rename it (to look nicer):

	```bash
	$ tar -xvzf adfr_install.tar.gz -C ADFRsuite
	```

	- cd into created dir

	```bash
	$ cd ADFRsuite
	```

	- source the install.sh

	```bash
	$ source ./install.sh
	```

	- export the variable to the path
	```bash
	$ echo "PATH=`pwd`/bin:"'$PATH' >> ~/.bashrc
	```

	- source the bashrc
	```bash
	$ source ~/.bashrc
	```

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

```bash
$ wget https://ccsb.scripps.edu/adfr/download/1028/ --no-check-certificate -O adfr_install && mkdir -p mgltools && tar -xvzf adfr_install.tar.gz -C ADFRsuite --strip-components=1 && rm adfr_install.tar.gz && cd ADFRsuite && source ./install.sh && echo "PATH=`pwd`/bin:"'$PATH' >> ~/.bashrc && source ~/.bashrc
```


Install DSSP
---------------

To install DSSP in Ubuntu 18.04+:

```bash
$ sudo apt install dssp
```

As default, the dssp path will be '/usr/bin/dssp'.


Download and install Autodock VINA
---------------

To install it, you have 2 options:

* Option 1 (For those who love to follow each step)

	- Go to the website http://vina.scripps.edu/download.html and download the Linux installer (tgz)
	- Untar it:

	```bash
	$ tar -xvzf autodock_vina_1_1_2_linux_x86.tgz
	```

* Option 2 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

	```bash
	$ mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina && sudo cp vina/vina /usr/bin/vina
	```

OBS: The vina executable will be in the following dir: ``installation_dir/vina/bin``


Download and install SMINA
---------------

First of all make sure that you have all required libs installed (openbabel must be v3+).

```bash
$ sudo apt install git libboost-all-dev libopenbabel-dev build-essential libeigen3-dev openbabel
```

Now clone the smina repo then enter it, create a build folder, enter the build folder, perform the cmake using the parent folder as the source and finally use the make with 12 jobs (you can increase/decrease the number of jobs if you want, but 12 is what is written in smina's doc).

```bash
$ git clone https://git.code.sf.net/p/smina/code smina-code && cd smina-code && mkdir build && cd build && cmake .. && make -j12
```


Download and install PLANTS
---------------

Go to http://www.tcd.uni-konstanz.de/plants_download/ and demand a license

Download and install GNINA
---------------

Go to https://github.com/gnina/gnina/releases and download the latest build.

If somehow you need or want to build your own code, follow the instructions from this link: https://github.com/gnina/gnina


EXPLAINING THE OCDOCKER FILE STRUCTURE
============

OCDocker has been designed to use the following structure of files:

```
└── receptor
    └── compounds
        ├── candidates
        │   ├── molecule_1
        │   └── molecule_2
        ├── decoys
        │   ├── molecule_A
        │   └── Molecule_B
        └── ligands
            ├── molecule_a
            └── molecule_b
```

| Folder | Description |
| ------------ | ------------ |
| receptor | Contains the receptor file (.pdb). |
| compounds | Used to keep things organized. Contains just the next three folders. |
| candidates | Any folder inside this folder will be flagged as a candidate compound, which means that it is not known the nature of its interaction with the receptor. (In a real world VS, only this folder will be populated.) |
| decoys | Any folder inside this folder will be flagged as a decoy. (This folder is used to validate ML results, probably not being used for real VS.) |
| ligands | Any folder inside this folder will be flagged as a ligand. (This folder is used to train and validate ML results, probably not being used for real VS.) |

USAGE
======

> :warning: To perform docking using the OCDocker library docking functions you must first install the abovementioned software.

In OCDocker the docking routines are oriented towards a Receptor and a Ligand, therefore, first of all, it is needed to create the receptor and ligand objects.

Here is an example of receptor and multiple ligand creations using files found in test_files folder:

```python
# Receptor import and creation
import OCDocker.Receptor as ocr
receptor = ocr.Receptor("./test_files/receptor.pdb", name="Receptor")

# Ligand import and creation
import OCDocker.Ligand as ocl
ligand = ocl.Ligand("./test_files/compounds/ligands/ligand/ligand.smi", name="Ligand")
decoy =  ocl.Ligand("./test_files/compounds/decoys/ZINC000000000015/ligand.smi", name="ZINC000000000015")
decoy2 =  ocl.Ligand("./test_files/compounds/decoys/ZINC000000000024/ligand.smi", name="ZINC000000000024")
decoy3 =  ocl.Ligand("./test_files/compounds/decoys/ZINC000000000030/ligand.smi", name="ZINC000000000030")
```

Now we can create the docking objects, here how is it done:

Pre steps
-------

```python
# Parameterize the path to make easier
ligandPath = f"./test_files/compounds/ligands/ligand"
```

GNINA
------
```python3
# Import
import OCDocker.Docking.Gnina as ocgnina

# Create object
gnina_ligand = ocgnina.Gnina(f"{ligandPath}/gninaFiles/conf_gnina.txt", f"{ligandPath}boxes/box.pdb", receptor, f"./test_files/prepared_receptor.pdbqt", ligand, f"{ligandPath}/prepared_ligand.pdbqt", f"{ligandPath}/gninaFiles/gnina.log", f"{ligandPath}/gninaFiles/gnina.pdbqt", name=f"Gnina receptor-ligand")

# Prepare receptor
gnina_ligand.run_prepare_receptor()

# Prepare ligand
gnina_ligand.run_prepare_ligand()

# Run docking
gnina_ligand.run_dock()
```

SMINA
------
```python3
# Import
import OCDocker.Docking.Smina as ocsmina

# Create object
smina_ligand = ocgnina.Smina(f"{ligandPath}/sminaFiles/conf_smina.txt", f"{ligandPath}boxes/box.pdb", receptor, f"./test_files/prepared_receptor.pdbqt", ligand, f"{ligandPath}/prepared_ligand.pdbqt", f{ligandPath}/sminaFiles/smina.log", f"{ligandPath}/sminaFiles/smina.pdbqt", name=f"Smina receptor-ligand")

# Prepare receptor
smina_ligand.run_prepare_receptor()

# Prepare ligand
smina_ligand.run_prepare_ligand()

# Run docking
smina_ligand.run_dock()
```

Vina
------
```python3
# Import
import OCDocker.Docking.Vina as ocvina

# Create object
vina_ligand = ocvina.Vina(f"{ligandPath}/vinaFiles/conf_vina.txt", f"{ligandPath}boxes/box.pdb", receptor, f"./test_files/prepared_receptor.pdbqt", ligand, f"{ligandPath}/prepared_ligand.pdbqt", f"{ligandPath}/vinaFiles/vina.log", f"{ligandPath}/vinaFiles/vina.pdbqt", name=f"Vina receptor-ligand")

# Prepare receptor
vina_ligand.run_prepare_receptor()

# Prepare ligand
vina_ligand.run_prepare_ligand()

# Run docking
vina_ligand.run_dock()
```

These steps will be the same for any pairs receptor-ligand!

## License

OCDocker is dual-licensed:

- **Open Source License**: Apache License 2.0 — free for academic, research, and non-commercial use.
- **Commercial License**: Required for any commercial or for-profit use.

See the `LICENSE` and `COMMERCIAL.txt` files for full terms.
