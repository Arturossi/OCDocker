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
conda install arturossi/label/prealpha::ocdocker 
```

If you have mamba installed, you can install OCDocker with the following command:

```bash
mamba install arturossi/label/prealpha::ocdocker 
```

pip
---

If you prefer to use pip, you can install OCDocker with the following command:

```bash
pip install ocdocker
```

From source
-----------

Download the source code from the GitHub repository:

```bash
git clone https://github.com/Arturossi/OCDocker
```

Go to the OCDocker directory and execute the installer with:

```bash
bash ./install.sh
```

Prerequisites
-------------

- Ubuntu/Debian-like system with internet access
- sudo privileges (required to install: DSSP, MySQL server, and place Vina in `/usr/bin`)
- ~15–20 GB of free disk space for conda env + tools + caches
- bash shell (the installer uses `bash` and `conda.sh`)

Automated installer details
---------------------------

The installer performs the following actions on Ubuntu-like systems:

- Installs MGLTools locally under `./mgltools`.
- Downloads AutoDock Vina and installs it to `/usr/bin/vina` (requires sudo).
- Installs Miniconda under `$HOME/miniconda` (no sudo).
- Installs DSSP and MySQL server via `apt-get` (requires sudo).
- Creates the conda environment defined by `environment.yml` (name: `ocdocker`).
- Creates a MySQL user and database named `ocdocker` (configurable via environment variables `DB_USER`, `DB_PASS`, `DB_NAME` before running the script).

Notes
-----

- You will be prompted for sudo privileges when needed (system packages and `/usr/bin/vina`).
- The script is idempotent for major steps: it skips environment creation if `ocdocker` already exists and uses `IF NOT EXISTS` for MySQL user/database.
- After completion, activate the environment with:

```bash
conda activate ocdocker
```

Log out then log in (or open a new shell) if you need to initialize conda’s base shell integration.

Troubleshooting
---------------

- MGLTools issues (e.g., NumPy import errors):
  - Consider reinstalling MGLTools from source or using the official archives; ensure system Python/conda paths don’t shadow MGLTools’ bundled Python.
  - Verify the `pythonsh` and `prepare_*` paths configured in `OCDocker.cfg`.

- Conda not found after install:
  - Open a new shell, or run `source "$HOME/miniconda/etc/profile.d/conda.sh"` before `conda activate ocdocker`.

- MySQL authentication errors:
  - Ensure `mysql-server` service is running (`sudo systemctl status mysql`).
  - Re-run the user/database creation commands shown in `install.sh`, or set `DB_USER/DB_PASS/DB_NAME` and re-run the script.

- DSSP not found:
  - Install via `sudo apt-get install -y dssp`, or adjust the `dssp` path in `OCDocker.cfg` to match your system.

GPU (optional)
--------------

OCDocker can leverage NVIDIA GPUs for PyTorch-based components (e.g., OCScore DNN/SHAP pipelines). The provided environment pins:

- PyTorch 2.4.1 with CUDA 12.1 (`pytorch-cuda=12.1`)
- cuDNN bundled via conda

### Requirements

- Recent NVIDIA driver compatible with CUDA 12.1 (recommended ≥ 535)
- No system CUDA toolkit is strictly required; the conda packages ship the CUDA runtime

### Quick checks

```bash
# Driver + GPU visible?
nvidia-smi

# PyTorch sees the GPU?
conda activate ocdocker
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### Troubleshooting GPU

- If `torch.cuda.is_available()` is False:
  - Ensure the NVIDIA driver is installed and loaded (e.g., `sudo ubuntu-drivers autoinstall` then reboot)
  - Verify driver ≥ 535 for CUDA 12.1
  - Make sure you activated the correct conda env (`ocdocker`)
  - Avoid mixing system CUDA with conda CUDA unless you know what you’re doing

Or perform each software installation manually with the below steps.

Download and install MGLTools
------------------------------------------------------------------------------------

To install it, you have 3 options:

* Option 1 (For those who loves GUI)

```bash
wget https://ccsb.scripps.edu/download/292/ --no-check-certificate -O mgltools_install
```

* Option 2 (For those who love to follow each step)

	- Download the file

```bash
wget https://ccsb.scripps.edu/download/532/ --no-check-certificate -O mgltools_install.tar.gz
```

	- Untar it:

```bash
tar -xvzf mgltools_install.tar.gz
```

	- cd into created dir

```bash
cd mgltools_x86_64Linux2_1.5.X
```

	- source the install.sh

```bash
source ./install.sh
```

* Option 3 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

```bash
wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz --no-check-certificate && mkdir -p mgltools && tar -xvzf mgltools_install.tar.gz -C mgltools --strip-components=1 && rm mgltools_install.tar.gz && cd mgltools && source ./install.sh
```

Note: The scripts used to prepare ligand/receptor are located at ``<installation_dir>/mgltools/MGLToolsPckgs/AutoDockTools``.

> :warning: **Still cannot run MGLTools?**: If you are facing some shady problems such as the numpy one, you might have to compile MGLTools from source. You can download it at https://github.com/genome-vendor/MGLtools (Still not sure about its version... I do not know if it is 1.5.6 or 1.5.4)

Install DSSP
---------------

To install DSSP in Ubuntu 18.04+:

```bash
sudo apt install dssp
```

By default, the DSSP path is '/usr/bin/dssp'.


Download and install AutoDock Vina
---------------

To install it, you have 2 options:

* Option 1 (Step-by-step)

	- Go to the website http://vina.scripps.edu/download.html and download the Linux installer (tgz)
	- Untar it:

```bash
tar -xvzf autodock_vina_1_1_2_linux_x86.tgz
```

* Option 2 (Use this all-in-one command. It seems to be more complicated, but its easier than option 2 and its easy to automate-it)

```bash
mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina && sudo cp vina/vina /usr/bin/vina
```

Note: The Vina executable will be located in ``installation_dir/vina/bin``.


Download and install SMINA
---------------

First of all make sure that you have all required libs installed (openbabel must be v3+).

```bash
sudo apt install git libboost-all-dev libopenbabel-dev build-essential libeigen3-dev openbabel
```

Now clone the smina repo then enter it, create a build folder, enter the build folder, perform the cmake using the parent folder as the source and finally use the make with 12 jobs (you can increase/decrease the number of jobs if you want, but 12 is what is written in smina's doc).

```bash
git clone https://git.code.sf.net/p/smina/code smina-code && cd smina-code && mkdir build && cd build && cmake .. && make -j12
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


SMINA
-----
```python
# Import
import OCDocker.Docking.Smina as ocsmina

# Create object
smina_ligand = ocsmina.Smina(
    f"{ligandPath}/sminaFiles/conf_smina.txt",
    f"{ligandPath}/boxes/box.pdb",
    receptor,
    f"./test_files/prepared_receptor.pdbqt",
    ligand,
    f"{ligandPath}/prepared_ligand.pdbqt",
    f"{ligandPath}/sminaFiles/smina.log",
    f"{ligandPath}/sminaFiles/smina.pdbqt",
    name=f"Smina receptor-ligand",
)

# Prepare receptor
smina_ligand.run_prepare_receptor()

# Prepare ligand
smina_ligand.run_prepare_ligand()

# Run docking
smina_ligand.run_docking()
```

Vina
----
```python
# Import
import OCDocker.Docking.Vina as ocvina

# Create object
vina_ligand = ocvina.Vina(
    f"{ligandPath}/vinaFiles/conf_vina.txt",
    f"{ligandPath}/boxes/box.pdb",
    receptor,
    f"./test_files/prepared_receptor.pdbqt",
    ligand,
    f"{ligandPath}/prepared_ligand.pdbqt",
    f"{ligandPath}/vinaFiles/vina.log",
    f"{ligandPath}/vinaFiles/vina.pdbqt",
    name=f"Vina receptor-ligand",
)

# Prepare receptor
vina_ligand.run_prepare_receptor()

# Prepare ligand
vina_ligand.run_prepare_ligand()

# Run docking
vina_ligand.run_docking()
```

These steps will be the same for any pairs receptor-ligand!

## License

This software is proprietary and owned by the Federal University of Rio de Janeiro (UFRJ). See the `LICENSE` file for full terms.

Summary:

- All proprietary rights belong to UFRJ.
- Use, reproduction, modification, and distribution are restricted and subject to formal authorization from the institution (Law No. 9,609/1998 — Brazil).
- For authorization and commercial licensing inquiries, contact: `arturossi10@gmail.com`.
