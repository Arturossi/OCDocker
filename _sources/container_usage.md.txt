# Container Usage

OCDocker includes Docker, Podman, and Singularity/Apptainer helpers. Containers
are useful when you want a reproducible runtime, but host data paths still need
to be mounted explicitly or kept under a mounted workspace.

## Docker

Install the short `ocd` wrapper:

```bash
containers/docker/install_docker.sh --install-ocd
```

Default backend is PostgreSQL. Set a database password before using the wrapper:

```bash
export OCDOCKER_DB_PASS='<db_password>'
ocd doctor
```

Use MySQL instead:

```bash
export OCDOCKER_DB_BACKEND=mysql
export OCDOCKER_DB_PASS='<db_password>'
export MYSQL_ROOT_PASSWORD='<root_password>'
ocd doctor
```

The wrapper auto-mounts likely host paths from CLI arguments. You can also mount
paths explicitly:

```bash
ocd --mount /data/project pipeline \
  --receptor /data/project/receptor.pdb \
  --ligand /data/project/ligand.sdf \
  --box /data/project/box.pdb \
  --outdir /data/project/output
```

Recommended pattern: keep all inputs and outputs under one project directory and
mount that directory once.

## Docker database behavior

The committed Docker config files do not contain a real password. The entrypoint
creates a runtime config in `/tmp/ocdocker/OCDocker.cfg` using the environment
values passed to compose.

Relevant variables:

- `OCDOCKER_DB_BACKEND`: `postgresql` or `mysql`.
- `OCDOCKER_DB_PASS`: required for Docker database services.
- `MYSQL_ROOT_PASSWORD`: required when using the MySQL service.
- `OCDOCKER_DATABASE`: defaults to `ocdocker`.
- `OCDOCKER_OPTIMIZEDB`: defaults to `optimization`.
- `OCDOCKER_DB_USER`: defaults to `ocdocker`.

## Singularity/Apptainer

Wrapper:

```bash
containers/singularity/ocdocker.sh --help
```

Sidecar database helpers:

```bash
containers/singularity/postgresql.sh start
containers/singularity/mysql.sh start
```

Useful options:

- `--cfg-source /path/to/OCDocker.cfg`: parse config for bind hints.
- `--dry-run`: print the resolved command without executing it.
- `--workdir /path/to/project`: run from a mounted project directory.

Example:

```bash
export OCDOCKER_SINGULARITY_IMAGE=/path/to/ocdocker.sif
containers/singularity/ocdocker.sh \
  --workdir /data/project \
  script --allow-unsafe-exec /data/project/run.py
```

## Scheduler workflows

For workflow managers, prefer explicit resources and stable output markers:

```bash
ocdocker \
  --threads 4 \
  --tmp-dir tmp/sample_001 \
  pipeline \
  --receptor input/sample_001/receptor.pdbqt \
  --ligand input/sample_001/ligand.pdbqt \
  --box input/sample_001/box.txt \
  --outdir results/sample_001 \
  --engines vina,smina,plants \
  --strict-engines \
  --done-marker results/sample_001/done.json \
  --log-file logs/sample_001.log \
  --no-stdout-log
```

Examples:

- `examples/19_Snakefile_ocdocker_pipeline.smk`
- `examples/20_Snakefile_ocdocker_granular_pipeline.smk`
