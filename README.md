![OCDocker](./OCDocker.png "OCDocker")

OCDocker
========

Project Description
-------------------

OCDocker is a Python toolkit and CLI for automated molecular docking, virtual
screening, and AI‑assisted consensus scoring. It streamlines end‑to‑end flows
from preparation through docking, pose clustering and rescoring, with optional
database persistence and analysis utilities.

Key capabilities:

- Multi‑engine docking: AutoDock Vina, Smina, PLANTS (future: Gnina, others)
- Pipelines: run engines, cluster poses by RMSD (medoid), rescore and export
- Rescoring: built‑in engine rescoring and ODDT models (RFScore, NNScore, PLEC)
- OCScore analytics: DNN/XGBoost/Transformer optimizers, ranking metrics, SHAP
- Database integration: MySQL (default) or SQLite fallback for dev/tests
- CLI and Python API: doctor diagnostics, timeouts, binary checks, reproducible configs
- Packaging: conda/pip, Dockerfiles for engines, docs and examples

Installation
------------

Simplest methods
----------------

Conda / Mamba
-------------

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

**Important:** Before installing via pip, you must install the required system dependencies:

```bash
sudo apt-get install openbabel libopenbabel-dev swig
```

Then install OCDocker:

```bash
pip install ocdocker
```

**Installing from source with pip:**

For development or if you want to install from source using pip:

```bash
# Clone the repository
git clone https://github.com/Arturossi/OCDocker
cd OCDocker

# Install system dependencies first (REQUIRED)
sudo apt-get install openbabel libopenbabel-dev swig

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

**Note on chemistry packages (rdkit, openbabel):**

Some packages like `rdkit` and `openbabel` are easier to install via conda due to their system dependencies. Before installing, ensure you have the required system packages:

```bash
# Install system dependencies first (REQUIRED)
sudo apt-get install openbabel libopenbabel-dev swig
```

If you encounter installation issues with pip, you can install these via conda first:

```bash
# Install chemistry packages via conda
conda install -c conda-forge rdkit openbabel

# Then install the rest via pip
pip install -r requirements.txt
pip install -e .
```

Alternatively, you can use a hybrid approach: install the chemistry packages via conda, then use pip for the rest of the dependencies.

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

- Python 3.9+
- Ubuntu/Debian-like system with internet access
- sudo privileges (required to install: OpenBabel, DSSP, MySQL server, and place Vina in `/usr/bin`)
- ~15–20 GB of free disk space for conda env + tools + caches
- bash shell (the installer uses `bash` and `conda.sh`)

**System Dependencies (Required Before Installation):**

Before installing OCDocker, you must install the following system packages:

```bash
sudo apt-get install openbabel libopenbabel-dev swig
```

These packages are required for building and using OpenBabel Python bindings, which are essential for OCDocker's molecular processing capabilities.

MySQL setup (quick tutorial)
----------------------------

OCDocker stores docking and optimization results in MySQL by default. If you don't already have a MySQL server, install it and create a user/database:

1) Install and start MySQL (Ubuntu/Debian)

```bash
sudo apt-get update && sudo apt-get install -y mysql-server
sudo systemctl enable --now mysql
```

2) Create a database and user (local-only access)

```sql
-- Enter the MySQL shell
sudo mysql

-- Create databases (adjust name as desired)
CREATE DATABASE ocdocker CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE optimization CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- Create user for local connections only
CREATE USER 'ocdocker'@'localhost' IDENTIFIED BY 'strong_password_here';
GRANT ALL PRIVILEGES ON ocdocker.* TO 'ocdocker'@'localhost';
GRANT ALL PRIVILEGES ON optimization.* TO 'ocdocker'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

3) Optional: allow remote connections (use strong passwords and firewalls)

```sql
-- In the MySQL shell
CREATE USER 'ocdocker'@'%' IDENTIFIED BY 'strong_password_here';
GRANT ALL PRIVILEGES ON optimization.* TO 'ocdocker'@'%';
FLUSH PRIVILEGES;
```

If you enable remote access, also edit `mysqld.cnf` to listen externally:

```bash
sudo sed -i "s/^bind-address.*/bind-address = 0.0.0.0/" /etc/mysql/mysql.conf.d/mysqld.cnf
sudo systemctl restart mysql
```

4) Test connectivity from Python

```python
from sqlalchemy import create_engine
from urllib.parse import quote_plus

user = "ocdocker"
password = quote_plus("strong_password_here")
host = "localhost"  # or server IP
port = 3306
db   = "optimization"

engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")
with engine.connect() as conn:
    print(conn.execute("SELECT 1").scalar())
```

Notes:

- The SQLAlchemy URL uses the PyMySQL driver (`mysql+pymysql://...`). Ensure `pymysql` is installed (present in the provided `environment.yml`).
- For CI/tests or local experiments, set `OCDOCKER_USE_SQLITE=1` to bypass MySQL.
 - You can also set SQLite via config (`USE_SQLITE = yes`) and choose a custom file via `SQLITE_PATH`.

Automated installer details
---------------------------

The installer performs the following actions on Ubuntu-like systems:

- Installs MGLTools locally under `./mgltools`.
- Downloads AutoDock Vina and installs it to `/usr/bin/vina` (requires sudo).
- Installs Miniconda under `$HOME/miniconda` (no sudo).
- Installs system packages via `apt-get` (requires sudo): openbabel, libopenbabel-dev, swig, DSSP, and MySQL server (if not using SQLite).
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
-----------------------------

Use either the step‑by‑step install or the single all‑in‑one command below.

- Option 1 (Step‑by‑step)

  - Download the archive

    ```bash
    wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz
    ```

  - Extract it

    ```bash
    tar -xvzf mgltools_install.tar.gz
    ```

  - Enter the created directory and run the installer

    ```bash
    cd mgltools_x86_64Linux2_1.5.X
    source ./install.sh
    ```

- Option 2 (All‑in‑one, easy to automate)

  ```bash
  wget https://ccsb.scripps.edu/download/532/ -O mgltools_install.tar.gz \
    && mkdir -p mgltools \
    && tar -xvzf mgltools_install.tar.gz -C mgltools --strip-components=1 \
    && rm mgltools_install.tar.gz \
    && cd mgltools \
    && source ./install.sh
  ```

Note: The `prepare_*` scripts are located at `<installation_dir>/mgltools/MGLToolsPckgs/AutoDockTools`.

If you still can’t run MGLTools (e.g., NumPy errors), consider reinstalling from source and ensure your environment paths don’t shadow the MGLTools Python.

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

Usage Overview
--------------

- CLI: `ocdocker` exposes subcommands for docking, pipelines, SHAP analysis, diagnostics, and an interactive console.
- Programmatic: importing modules auto‑bootstraps once by default (see Bootstrap below). You can opt out via an env var and call `bootstrap()` explicitly.

Bootstrap & Configuration
-------------------------

- Auto‑bootstrap on import: when you import OCDocker modules, the environment initializes once (config, DB, dirs). This is skipped during docs/tests.
- Configuration file: set `OCDOCKER_CONFIG` to point to your `OCDocker.cfg` or place `OCDocker.cfg` in the working directory.
- Disable auto‑bootstrap: set `OCDOCKER_NO_AUTO_BOOTSTRAP=1` and call `bootstrap()` explicitly:

```python
from OCDocker.Initialise import bootstrap
import argparse
bootstrap(argparse.Namespace(
    multiprocess=True,
    update=False,
    config_file='OCDocker.cfg',
    output_level=2,
    overwrite=False,
))
```

SQLite Fallback (optional)
--------------------------

- For development/tests, you can bypass MySQL entirely by setting `OCDOCKER_USE_SQLITE=1` before import or running the CLI.
- This creates/uses a local `ocdocker.db` under the module directory.

Installer behavior with SQLite
------------------------------

- To skip installing and configuring MySQL during `install.sh`, enable SQLite mode before running it:

```bash
export OCDOCKER_USE_SQLITE=1                # select SQLite backend
export OCDOCKER_SQLITE_PATH=/path/ocdocker.db  # optional custom path
bash ./install.sh
```

- Alternatively, if you already have an `OCDocker.cfg` in the project directory, you can set in the file:
  - `USE_SQLITE = yes`
  - `SQLITE_PATH = /path/to/ocdocker.db` (optional)

In both cases, the installer will:
- Install only `dssp` (skips `mysql-server`)
- Skip MySQL user/database creation
- Proceed with the remaining steps normally

Important note about SQLite
---------------------------

- SQLite is convenient for development and tests but has limitations for concurrent writes and larger workloads.
- For production use, performance, and concurrency, a full MySQL installation is strongly recommended.

Diagnostics: `ocdocker doctor`
--------------------------------

Run a quick environment report:

```bash
ocdocker doctor --conf OCDocker.cfg
```

It checks:

- Config path in use
- Binaries: `vina`, `smina`, `plants` (presence on PATH or configured paths)
- Python deps: rdkit, Biopython, ODDT, SQLAlchemy
- DB connectivity (opens/closes a connection)

Docking: Quick Examples
-----------------------

Single engine (Vina) with timeout, storing to DB:

```bash
ocdocker vs \
  --engine vina \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.smi \
  --box path/to/box.pdb \
  --timeout 600 \
  --store-db
```

Pipeline across engines with clustering and rescoring:

```bash
ocdocker pipeline \
  --receptor path/to/receptor.pdb \
  --ligand path/to/ligand.sdf \
  --box path/to/box.pdb \
  --engines vina,smina,plants \
  --store-db
```

Notes:

- `--timeout` limits external tool runtime (also via `OCDOCKER_TIMEOUT`).
- `--store-db` auto‑creates tables and stores minimal metadata (name) in the DB.

Timeouts & External Tools
-------------------------

- You can prevent hangs by defining a timeout:
  - CLI: `--timeout <seconds>` (for `vs` and `pipeline`)
  - Env: `OCDOCKER_TIMEOUT=<seconds>`

Binary Checks
-------------

- The CLI validates presence of required binaries (`vina`/`smina`/`plants`) before running and errors early if missing. Use `ocdocker doctor` to see what’s available.

Interactive Console
-------------------

```bash
ocdocker console --conf OCDocker.cfg
```

This opens an interactive namespace with common OCDocker utilities imported.

Running Python Scripts
----------------------

Run Python scripts with all OCDocker libraries pre-loaded:

```bash
ocdocker script --conf OCDocker.cfg script.py [script_args...]
```

This command bootstraps the OCDocker environment, loads all modules (ocl, ocr, ocvina, etc.),
and executes your script. All OCDocker classes and functions are available without imports.

Example script:
```python
# script.py - All OCDocker modules are pre-loaded!
receptor = ocr.Receptor("receptor.pdb")
ligand = ocl.Ligand("ligand.smi")
vina = ocvina.Vina(...)
# ... use OCDocker functionality
```

See `examples/13_cli_script_example.py` for a complete example.

Environment Variables (reference)
---------------------------------

- `OCDOCKER_CONFIG`: path to `OCDocker.cfg` (config file with external tool paths and parameters).
- `OCDOCKER_NO_AUTO_BOOTSTRAP`: if set to `1/true/yes`, disables auto‑bootstrap on import; call `bootstrap()` manually.
- `OCDOCKER_USE_SQLITE`: if set to `1/true/yes`, uses a local SQLite DB instead of MySQL.
- `OCDOCKER_TIMEOUT`: default timeout (seconds) for external tools when not provided via CLI.

Python Support
--------------

- Requires Python 3.10+.

```bash
mkdir vina && wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 -O vina/vina && sudo cp vina/vina /usr/bin/vina
```

Note: The Vina executable will be located in ``installation_dir/vina/bin``.

Testing
=======

This repository ships a test suite under `tests/` that exercises the core library (Toolbox, Docking helpers, DB minimal, parsing, etc.).

Quick start
-----------

```bash
conda activate ocdocker
pytest -q
```

Useful commands
---------------

- Run a specific test file:

  ```bash
  pytest tests/test_Vina.py -q
  ```

- Show test names while running:

  ```bash
  pytest -q -k vina -vv
  ```

- Coverage (if `pytest-cov` is present):

  ```bash
  pytest --cov=OCDocker --cov-report=term-missing
  ```

Notes for testing
-----------------

- The tests operate on sample data under `test_files/` and do not require external binaries to actually run (they validate parsing/IO helpers, config generation, log readers, etc.).
- If you want to run end‑to‑end docking locally, ensure you’ve installed external tools (MGLTools, Vina, Smina/PLANTS where applicable) and set paths in `OCDocker.cfg`.
- Some modules (e.g., Initialise) perform environment bootstrapping; the test suite avoids heavy side effects, but for interactive usage consider setting `OCDOCKER_CONFIG=./OCDocker.cfg`.


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
decoy  = ocl.Ligand("./test_files/compounds/decoys/ZINC000000000015/ligand.smi", name="ZINC000000000015")
decoy2 = ocl.Ligand("./test_files/compounds/decoys/ZINC000000000024/ligand.smi", name="ZINC000000000024")
decoy3 = ocl.Ligand("./test_files/compounds/decoys/ZINC000000000044/ligand.smi", name="ZINC000000000044")
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

Testing
=======
This repository ships a test suite under `tests/` that exercises the core library (Toolbox, Docking helpers, DB minimal, parsing, etc.).
Quick start
-----------
```bash
conda activate ocdocker
pytest -q
```
Useful commands
---------------
- Run a specific test file:
  ```bash
  pytest tests/test_Vina.py -q
  ```
- Show test names while running:
  ```bash
  pytest -q -k vina -vv
  ```
- Coverage (if `pytest-cov` is present):
  ```bash
  pytest --cov=OCDocker --cov-report=term-missing
  ```
Notes for testing
-----------------
- The tests operate on sample data under `test_files/` and do not require external binaries to actually run (they validate parsing/IO helpers, config generation, log readers, etc.).
- If you want to run end‑to‑end docking locally, ensure you’ve installed external tools (MGLTools, Vina, Smina/PLANTS where applicable) and set paths in `OCDocker.cfg`.
- Some modules (e.g., Initialise) perform environment bootstrapping; the test suite avoids heavy side effects, but for interactive usage consider setting `OCDOCKER_CONFIG=./OCDocker.cfg`.
