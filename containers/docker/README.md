# OCDocker Docker Stack

This directory contains the Docker entrypoint for running OCDocker with a locked
workspace mount and a managed SQL database.

## Defaults

- Database: PostgreSQL (`postgres:16`)
- Workspace: host `./workspace` mounted as container `/workspace`
- OCDocker source: installed inside the image at `/opt/ocdocker`
- Container config: `/etc/ocdocker/OCDocker.cfg.postgresql`
- Architecture/output/data files should live under `/workspace`

## Quick Start

```bash
containers/docker/install_docker.sh \
  --workspace /path/to/ocdocker-project \
  --gnina /path/to/downloaded/gnina \
  --shell
```

Inside the shell:

```bash
ocdocker --help
```

Run a command directly:

```bash
containers/docker/install_docker.sh \
  --workspace /path/to/ocdocker-project \
  --gnina /path/to/downloaded/gnina \
  -- ocdocker ocscore train --help
```


## Optional `ocd` Wrapper

Install a host-side wrapper named `ocd`:

```bash
containers/docker/install_docker.sh --install-ocd
```

By default this writes `~/.local/bin/ocd`. It refuses to overwrite an existing
file unless `--force-ocd` is passed. Use `--wrapper-dir /path/to/bin` to choose
a different install directory.

The wrapper runs the containerized `ocdocker` command using the current directory
as the workspace unless `OCDOCKER_WORKSPACE` is set:

```bash
OCDOCKER_GNINA_HOST_PATH=/path/to/downloaded/gnina ocd --help
OCDOCKER_GNINA_HOST_PATH=/path/to/downloaded/gnina ocd ocscore train --help
```

For MySQL through the wrapper:

```bash
OCDOCKER_DB_BACKEND=mysql \
OCDOCKER_GNINA_HOST_PATH=/path/to/downloaded/gnina \
ocd --help
```


## Snakemake Jobs Through Docker

The `ocd` wrapper can be used as the command inside Snakemake rules. The host working directory becomes the Docker workspace unless `OCDOCKER_WORKSPACE` is set. Keep Snakemake inputs, outputs, logs, and temporary directories under that workspace so host and container paths match.

Example using `examples/19_Snakefile_ocdocker_pipeline.smk`:

```bash
OCDOCKER_GNINA_HOST_PATH=/path/to/downloaded/gnina \
snakemake -s examples/19_Snakefile_ocdocker_pipeline.smk \
  --cores 4 \
  --config ocdocker_command=ocd
```

The Compose files forward `OCDOCKER_THREADS`, `SNAKEMAKE_THREADS`, and `OCDOCKER_TMP_DIR` into the container. OCDocker also accepts `--threads` and `--tmp-dir` directly, so each Snakemake rule can use job-local resources without relying on host CPU autodetection or shared temporary directories.

For more granular per-engine scheduling, use `examples/20_Snakefile_ocdocker_granular_pipeline.smk` with the same `--config ocdocker_command=ocd` setting.

## MySQL Instead of PostgreSQL

PostgreSQL is the default. To force MySQL:

```bash
containers/docker/install_docker.sh \
  --db mysql \
  --workspace /path/to/ocdocker-project \
  --gnina /path/to/downloaded/gnina \
  --shell
```

This selects the standalone `docker-compose.mysql.yml`, switches `OCDOCKER_DB_BACKEND=mysql`,
and uses `/etc/ocdocker/OCDocker.cfg.mysql` inside the container.

## Workspace Locking

Docker mode mounts one host directory as `/workspace`. Keep input and output
paths under `/workspace` to avoid host/container path confusion. Advanced users
can add extra mounts to Compose if needed.

## Docking Binaries

The main OCDocker image uses the Linux executables shipped in this repository:

- `bin/docking/vina/vina`
- `bin/docking/vina/vina_split`
- `bin/docking/smina/smina`
- `bin/docking/plants/PLANTS1.2_64bit`
- `bin/docking/plants/SPORES_64bit`

The image also installs system tools from the OS package manager:
- Open Babel
- DSSP

Some tools may require external/licensed installation or user-provided binaries:

- GNINA is required and must be provided by the user because the correct binary depends on CPU/GPU/CUDA compatibility.
- MGLTools/AutoDockTools may be user-provided if your workflow needs AutoDockTools preparation scripts.

Download GNINA from <https://github.com/gnina/gnina>, then pass it to the installer:

```bash
containers/docker/install_docker.sh \
  --workspace /path/to/ocdocker-project \
  --gnina /path/to/downloaded/gnina \
  --shell
```

The installer copies it to:

```text
/path/to/ocdocker-project/software/docking/gnina/gnina
```

Inside the container OCDocker uses:

```text
/workspace/software/docking/gnina/gnina
```

PLANTS/SPORES are shipped in this repository under `bin/docking/plants`, but can also be replaced by workspace binaries or a custom `OCDOCKER_CONFIG`.

## Compose Directly

PostgreSQL:

```bash
OCDOCKER_WORKSPACE=/path/to/project \
docker compose -f containers/docker/docker-compose.yml run --rm ocdocker bash
```

MySQL:

```bash
OCDOCKER_WORKSPACE=/path/to/project \
docker compose -f containers/docker/docker-compose.mysql.yml run --rm ocdocker bash
```
