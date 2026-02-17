[![codecov](https://codecov.io/gh/Arturossi/OCDocker/graph/badge.svg)](https://codecov.io/gh/Arturossi/OCDocker)
![CI](https://img.shields.io/github/actions/workflow/status/Arturossi/OCDocker/type-check.yml)
![Python](https://img.shields.io/pypi/pyversions/ocdocker)
![PyPI](https://img.shields.io/pypi/v/ocdocker)
![Issues](https://img.shields.io/github/issues/Arturossi/OCDocker)
![Last commit](https://img.shields.io/github/last-commit/Arturossi/OCDocker)

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

- Multi‑engine docking: AutoDock Vina, Smina, Gnina, PLANTS
- Pipelines: run engines, cluster poses by RMSD (medoid), rescore and export
- Rescoring: built‑in engine rescoring and ODDT models (RFScore, NNScore, PLEC)
- OCScore analytics: DNN/XGBoost/Transformer optimizers, ranking metrics, SHAP
- Database integration: PostgreSQL (default), MySQL support, or SQLite fallback for dev/tests
- CLI and Python API: doctor diagnostics, timeouts, binary checks, reproducible configs
- Packaging: pip (recommended inside a conda/mamba env), Dockerfiles for engines, docs and examples

Community
---------

- Code of Conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Security: [SECURITY.md](SECURITY.md)
- Collaborators: [COLLABORATORS.md](COLLABORATORS.md)

Documentation
-------------

- Manual (GitHub): [MANUAL.md](MANUAL.md)
- Sphinx docs: `docs/` (install docs deps first; then run `make -C docs html`)
- Error handling guide: [docs/ERROR_HANDLING.md](docs/ERROR_HANDLING.md)

Installation
------------

Quickstart (minimal, SQLite)
----------------------------

If you want the fastest path without setting up PostgreSQL/MySQL, use SQLite (local file DB):

1) Install system dependencies (see [System dependencies](#system-dependencies)).
2) Create a conda env with Python 3.11 (prefer `mamba`) and install OCDocker with pip.
3) Run with SQLite enabled:

```bash
export OCDOCKER_DB_BACKEND=sqlite
ocdocker doctor
```

SQLite is recommended for quick experiments and development. For multi-user or long-running workloads, use PostgreSQL (default backend) or MySQL.

Recommended method (mamba + pip)
--------------------------------

**Important:** Install the required system dependencies first (see [System dependencies](#system-dependencies)).

If `mamba` is not installed yet:

```bash
conda install -n base -c conda-forge mamba
```

Then create the environment and install OCDocker from PyPI:

```bash
mamba create -n ocdocker python=3.11 -y
conda activate ocdocker
pip install ocdocker
```

`pip install ocdocker` installs the core package only. To include every optional runtime stack, use `pip install "ocdocker[all]"`.

Install optional feature stacks as needed:

```bash
# Docking workflows
pip install "ocdocker[docking]"

# Docking + DB support
pip install "ocdocker[docking,db]"

# ML workflows (PyTorch/XGBoost/Optuna)
pip install "ocdocker[ml]"

# All optional runtime features
pip install "ocdocker[all]"
```

**Installing from source with pip:**

For development, install from source with pip inside the same conda environment. Ensure the system dependencies are installed first (see [System dependencies](#system-dependencies)).

```bash
# Clone the repository
git clone https://github.com/Arturossi/OCDocker
cd OCDocker

# Create and activate conda env (if not already active)
mamba create -n ocdocker python=3.11 -y
conda activate ocdocker

# Install the package in development mode
pip install -e .

# Optional: install feature extras in editable mode
pip install -e ".[docking,db,ml]"
```

**Note on chemistry packages (`rdkit`, `openbabel`):**

These packages may require system libraries. Install the system dependencies first (see [System dependencies](#system-dependencies)). If pip installation fails, verify your compiler/toolchain and OpenBabel system packages are installed.

Prerequisites
-------------

- Python 3.11+
- Conda (Miniconda/Anaconda) and mamba
- pip (inside the conda environment)
- NVIDIA driver/runtime compatible with CUDA 12.8 (required for Gnina CUDA builds)
- Ubuntu/Debian-like system with internet access
- sudo privileges (needed for system packages, and optional PostgreSQL/MySQL/Vina installs)
- ~10-15 GB of free disk space for dependencies, tools, and caches (minimal installs use less)
- bash shell (used in command examples and helper scripts)

System dependencies
-------------------

Before installing OCDocker, you must install the following system packages:

```bash
sudo apt-get install openbabel libopenbabel-dev swig cmake g++
```

These packages are required for building and using OpenBabel Python bindings, which are essential for OCDocker's molecular processing capabilities.

PostgreSQL setup (quick tutorial)
---------------------------------

This section is optional. Skip it if you are using SQLite (see [Quickstart](#quickstart-minimal-sqlite)).

OCDocker stores docking and optimization results in PostgreSQL by default.

1) Install and start PostgreSQL (Ubuntu/Debian)

```bash
sudo apt-get update && sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

2) Create a role and databases

```bash
sudo -u postgres psql
```

```sql
CREATE ROLE ocdocker LOGIN PASSWORD 'strong_password_here';
CREATE DATABASE ocdocker OWNER ocdocker;
CREATE DATABASE optimization OWNER ocdocker;
\q
```

3) Configure `OCDocker.cfg`

```ini
DB_BACKEND = postgresql
HOST = localhost
PORT = 5432
USER = ocdocker
PASSWORD = strong_password_here
DATABASE = ocdocker
OPTIMIZEDB = optimization
```

4) Test connectivity from Python

```python
from sqlalchemy import create_engine
from urllib.parse import quote_plus

user = "ocdocker"
password = quote_plus("strong_password_here")
host = "localhost"
port = 5432
db = "optimization"

engine = create_engine(f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}")
with engine.connect() as conn:
    print(conn.exec_driver_sql("SELECT 1").scalar())
```

MySQL remains supported:

- Set `DB_BACKEND = mysql` (or `OCDOCKER_DB_BACKEND=mysql`).
- Use `PORT = 3306`.
- SQLAlchemy URL format: `mysql+pymysql://...`.

Notes:

- For CI/tests or local experiments, set `OCDOCKER_DB_BACKEND=sqlite` to bypass server DBs.
- You can also set SQLite via config (`DB_BACKEND = sqlite`) and choose a custom file via `SQLITE_PATH`.

Troubleshooting
---------------

- MGLTools issues (e.g., NumPy import errors):
  - Consider reinstalling MGLTools from source or using the official archives; ensure system Python/conda paths don’t shadow MGLTools’ bundled Python.
  - Verify the `pythonsh` and `prepare_*` paths configured in `OCDocker.cfg`.

- Database authentication errors:
  - PostgreSQL: ensure service is running (`sudo systemctl status postgresql`) and role/database exist.
  - MySQL: ensure service is running (`sudo systemctl status mysql`) and user/database grants exist.

- DSSP not found:
  - Install via `sudo apt-get install -y dssp`, or adjust the `dssp` path in `OCDocker.cfg` to match your system.

GPU (optional)
--------------

OCDocker can leverage NVIDIA GPUs for PyTorch-based components (e.g., OCScore DNN/SHAP pipelines).

### Requirements

- Recent NVIDIA driver compatible with your installed PyTorch CUDA build (for torch 2.4.x, a modern 535+ driver is a good baseline)

### Quick checks

```bash
# Driver + GPU visible?
nvidia-smi

# PyTorch sees the GPU?
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### Troubleshooting GPU

- If `torch.cuda.is_available()` is False:
  - Ensure the NVIDIA driver is installed and loaded (e.g., `sudo ubuntu-drivers autoinstall` then reboot)
  - Verify your driver version and installed torch CUDA build are compatible
  - Make sure you activated the correct conda environment (`ocdocker`)
  - Avoid mixing multiple CUDA toolkits unless you intentionally need that setup

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

* Option 2 (Use this all-in-one command. It seems to be more complicated, but it’s easier than option 1 and its easy to automate-it)

```bash
mkdir -p vina \
  && wget -O vina/vina https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.3/vina_1.2.3_linux_x86_64 \
  && chmod +x vina/vina \
  && sudo install -m 0755 vina/vina /usr/bin/vina
```

Download and install Gnina (CUDA 12.8)
---------------

OCDocker uses the Gnina CUDA 12.8 build. To run it correctly, ensure:

- NVIDIA driver is compatible with CUDA 12.8
- cuDNN 9 runtime is available

Step-by-step:

```bash
mkdir -p gnina
wget -O gnina/gnina.1.3.2.cuda12.8 https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8
chmod +x gnina/gnina.1.3.2.cuda12.8
sudo install -m 0755 gnina/gnina.1.3.2.cuda12.8 /usr/bin/gnina
```

Verify installation:

```bash
gnina --version
```

All-in-one command:

```bash
mkdir -p gnina \
  && wget -O gnina/gnina.1.3.2.cuda12.8 https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8 \
  && chmod +x gnina/gnina.1.3.2.cuda12.8 \
  && sudo install -m 0755 gnina/gnina.1.3.2.cuda12.8 /usr/bin/gnina
```

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

- For development/tests, you can bypass PostgreSQL/MySQL entirely by setting `OCDOCKER_DB_BACKEND=sqlite` before import or running the CLI.
- This creates/uses a local `ocdocker.db` under the module directory.

Installer behavior with SQLite
------------------------------

- To skip installing and configuring PostgreSQL/MySQL during `install.sh`, enable SQLite mode before running it:

```bash
export OCDOCKER_DB_BACKEND=sqlite           # select SQLite backend
export OCDOCKER_SQLITE_PATH=/path/ocdocker.db  # optional custom path
bash ./install.sh
```

- Alternatively, if you already have an `OCDocker.cfg` in the project directory, you can set in the file:
  - `DB_BACKEND = sqlite`
  - `SQLITE_PATH = /path/to/ocdocker.db` (optional)

In both cases, the installer will:
- Install only `dssp` (skips SQL server packages)
- Skip SQL user/database creation
- Proceed with the remaining steps normally
- Install Gnina CUDA 12.8 (`gnina.1.3.2.cuda12.8`) and register it as `/usr/bin/gnina`

Important note about SQLite
---------------------------

- SQLite is convenient for development and tests but has limitations for concurrent writes and larger workloads.
- For production use, performance, and concurrency, a full PostgreSQL installation is strongly recommended (MySQL is also supported).

Diagnostics: `ocdocker doctor`
--------------------------------

Run a quick environment report:

```bash
ocdocker doctor --conf OCDocker.cfg
```

It checks:

- Config path in use
- Binaries: `vina`, `smina`, `plants` (presence on PATH or configured paths)
- External tool metadata: resolved executable and version (`vina`, `smina`, `plants`, `gnina`, `pythonsh`, `dssp`, `obabel`, `spores`)
- Python deps: rdkit, Biopython, ODDT, SQLAlchemy
- DB backend/driver metadata, client version, server version (when queryable), connectivity, and current/expected user+database checks

Reproducibility: `ocdocker manifest`
------------------------------------

Generate a JSON manifest with OCDocker/Python versions, external tool versions,
platform metadata, git metadata (when available), and installed Python packages:

```bash
ocdocker manifest --conf OCDocker.cfg --output reproducibility_manifest.json
```

From Python code:

```python
import OCDocker.Toolbox.Reproducibility as ocrepro

manifest = ocrepro.generate_reproducibility_manifest(include_python_packages=False)
_ = ocrepro.write_reproducibility_manifest("reproducibility_manifest.json")
```

Docking: Quick Examples
-----------------------

Install docking dependencies first if needed:

```bash
pip install "ocdocker[docking]"
```

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

For ``--store-db``, install DB dependencies too:

```bash
pip install "ocdocker[db]"
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
- `--store-db` auto-creates tables and stores receptor/ligand descriptors plus supported rescoring columns in the DB.

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
ocdocker script --conf OCDocker.cfg --allow-unsafe-exec script.py [script_args...]
```

Security note: in-process script execution requires explicit opt-in via
`--allow-unsafe-exec` (or `OCDOCKER_ALLOW_SCRIPT_EXEC=1`).

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

Container wrappers (Docker, Podman and Singularity)
---------------------------------------------------

OCDocker includes helper scripts that auto-mount likely host paths:

- Docker: `containers/docker/ocdocker.sh`
- Podman: `containers/podman/ocdocker.sh`
- Singularity/Apptainer: `containers/singularity/ocdocker.sh`

All wrappers can:

- parse explicit `--mount` flags
- read mount lists from env vars (`OCDOCKER_DOCKER_MOUNTS` / `OCDOCKER_SINGULARITY_MOUNTS`)
- auto-detect absolute paths passed in CLI arguments
- parse `OCDocker.cfg` paths and add their parent directories as bind mounts

Docker/Podman backend selection:

- Default is PostgreSQL.
- Set `OCDOCKER_DB_BACKEND=mysql` (or `DB_BACKEND=mysql`) to use the MySQL compose override and MySQL container config.

Singularity helper extras:

- `--cfg-source /path/to/OCDocker.cfg` to force which config is parsed for bind hints
- `--dry-run` to print the resolved `apptainer/singularity exec` command without executing it

Singularity SQL sidecars:

- PostgreSQL (default): `containers/singularity/postgresql.sh`
- MySQL (optional): `containers/singularity/mysql.sh`

```bash
# Start local PostgreSQL (default backend)
containers/singularity/postgresql.sh start

# Start local MySQL (optional backend)
containers/singularity/mysql.sh start
```

Default PostgreSQL config matches `containers/singularity/OCDocker.cfg.singularity`:

- `DB_BACKEND=postgresql`
- `HOST=localhost`
- `PORT=5432`
- `USER=ocdocker`
- `PASSWORD=ocdocker_pass`
- `DATABASE=ocdocker`

For MySQL, use `containers/singularity/OCDocker.cfg.singularity.mysql` (or set `DB_BACKEND=mysql` and port `3306`).

Recommended pattern for dynamic script paths:

1. Create one project/work root (for example `/data/your_project`).
2. Keep all inputs/outputs under that root.
3. Bind that root once, instead of many scattered folders.

Singularity example:

```bash
export OCDOCKER_SINGULARITY_IMAGE=/path/to/ocdocker.sif
containers/singularity/ocdocker.sh \
  --workdir /data/your_project \
  script --allow-unsafe-exec /data/your_project/run.py
```

Environment Variables (reference)
---------------------------------

- `OCDOCKER_CONFIG`: path to `OCDocker.cfg` (config file with external tool paths and parameters).
- `OCDOCKER_DB_BACKEND` / `DB_BACKEND`: database backend override (`postgresql`, `mysql`, or `sqlite`).
- `OCDOCKER_NO_AUTO_BOOTSTRAP`: if set to `1/true/yes`, disables auto‑bootstrap on import; call `bootstrap()` manually.
- `OCDOCKER_SQLITE_PATH`: optional explicit SQLite database file path (used when backend is `sqlite`).
- `OCDOCKER_TIMEOUT`: default timeout (seconds) for external tools when not provided via CLI.

Python Support
--------------

- Requires Python 3.11+.

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
  pytest tests/docking/test_vina.py -q
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
