#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

backend="postgresql"
workspace="${OCDOCKER_WORKSPACE:-${repo_root}/workspace}"
command="shell"
build="true"
pull="false"
gnina_host_path="${OCDOCKER_GNINA_HOST_PATH:-}"
wrapper_dir="${OCDOCKER_WRAPPER_DIR:-${HOME}/.local/bin}"
force_ocd="false"
compose_args=()
compose_cmd=()

usage() {
  cat <<'USAGE'
Usage: containers/docker/install_docker.sh [options] [-- <ocdocker command>]

Build and run the OCDocker Docker stack. PostgreSQL is the default database.

Options:
  --db postgresql|mysql     Database backend. Default: postgresql.
  --workspace PATH          Host directory mounted as /workspace.
  --gnina PATH              GNINA executable downloaded by the user. Required unless already in workspace.
  --no-build                Do not build the OCDocker image before running.
  --pull                    Pull database base images before running.
  --up                      Start database and OCDocker services detached.
  --down                    Stop services and remove containers.
  --shell                   Open an interactive shell in the OCDocker container. Default.
  --install-ocd             Install the Docker wrapper command named 'ocd' and exit.
  --wrapper-dir PATH        Directory for --install-ocd. Default: ~/.local/bin.
  --force-ocd               Overwrite an existing ocd wrapper.
  --help                    Show this help.

Examples:
  containers/docker/install_docker.sh --workspace /data/my_project --gnina /path/to/gnina --shell
  containers/docker/install_docker.sh --install-ocd
  containers/docker/install_docker.sh --db mysql --workspace /data/my_project --gnina /path/to/gnina --up
  containers/docker/install_docker.sh -- ocdocker --help
  containers/docker/install_docker.sh --workspace /data/my_project -- ocdocker ocscore train --help
USAGE
}

install_ocd_wrapper() {
  local target_dir="$1"
  local target="${target_dir}/ocd"

  mkdir -p "${target_dir}"
  if [[ -e "${target}" && "${force_ocd}" != "true" ]]; then
    echo "Refusing to overwrite existing wrapper: ${target}" >&2
    echo "Use --force-ocd if you intentionally want to replace it." >&2
    exit 2
  fi

  cat > "${target}" <<WRAPPER
#!/usr/bin/env bash
set -euo pipefail

installer="${script_dir}/install_docker.sh"
workspace="\${OCDOCKER_WORKSPACE:-\${PWD}}"
backend="\${OCDOCKER_DB_BACKEND:-postgresql}"
args=("--db" "\${backend}" "--workspace" "\${workspace}")

if [[ -n "\${OCDOCKER_GNINA_HOST_PATH:-}" ]]; then
  args+=("--gnina" "\${OCDOCKER_GNINA_HOST_PATH}")
fi

exec "\${installer}" "\${args[@]}" -- ocdocker "\$@"
WRAPPER
  chmod +x "${target}"
  echo "Installed OCDocker Docker wrapper: ${target}"

  case ":${PATH}:" in
    *":${target_dir}:"*) ;;
    *)
      echo "Add this directory to PATH if needed: ${target_dir}" >&2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      backend="${2:-}"
      shift 2
      ;;
    --workspace)
      workspace="${2:-}"
      shift 2
      ;;
    --gnina)
      gnina_host_path="${2:-}"
      shift 2
      ;;
    --no-build)
      build="false"
      shift
      ;;
    --pull)
      pull="true"
      shift
      ;;
    --up)
      command="up"
      shift
      ;;
    --down)
      command="down"
      shift
      ;;
    --shell)
      command="shell"
      shift
      ;;
    --install-ocd)
      command="install_ocd"
      build="false"
      shift
      ;;
    --wrapper-dir)
      wrapper_dir="${2:-}"
      shift 2
      ;;
    --force-ocd)
      force_ocd="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      command="run"
      compose_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${backend,,}" in
  postgresql|postgres|pgsql)
    backend="postgresql"
    compose_files=(-f docker-compose.yml)
    export OCDOCKER_DB_BACKEND=postgresql
    export OCDOCKER_DB_PORT="${OCDOCKER_DB_PORT:-5432}"
    ;;
  mysql|mariadb)
    backend="mysql"
    compose_files=(-f docker-compose.mysql.yml)
    export OCDOCKER_DB_BACKEND=mysql
    export OCDOCKER_DB_PORT="${OCDOCKER_DB_PORT:-3306}"
    ;;
  *)
    echo "Unsupported --db '${backend}'. Use postgresql or mysql." >&2
    exit 2
    ;;
esac

if [[ "${command}" == "install_ocd" ]]; then
  install_ocd_wrapper "${wrapper_dir}"
  exit 0
fi

mkdir -p "${workspace}"
export OCDOCKER_WORKSPACE="$(cd "${workspace}" && pwd)"

if [[ "${command}" != "down" ]]; then
  gnina_workspace_path="${OCDOCKER_WORKSPACE}/software/docking/gnina/gnina"
  if [[ -x "${gnina_workspace_path}" ]]; then
    :
  elif [[ -n "${gnina_host_path}" ]]; then
    if [[ ! -f "${gnina_host_path}" ]]; then
      echo "GNINA executable not found: ${gnina_host_path}" >&2
      exit 2
    fi
    mkdir -p "$(dirname "${gnina_workspace_path}")"
    cp "${gnina_host_path}" "${gnina_workspace_path}"
    chmod +x "${gnina_workspace_path}"
    echo "Installed GNINA executable into workspace: ${gnina_workspace_path}"
  else
    cat >&2 <<'GNINA_ERROR'
GNINA is required for the Docker workspace but was not found.

Download the correct GNINA executable for your machine/CUDA stack from:
  https://github.com/gnina/gnina

Then run one of:
  containers/docker/install_docker.sh --workspace /path/to/project --gnina /path/to/gnina --shell
  OCDOCKER_GNINA_HOST_PATH=/path/to/gnina containers/docker/install_docker.sh --workspace /path/to/project --shell

The installer will copy it to:
  /path/to/project/software/docking/gnina/gnina

Inside the container OCDocker uses:
  /workspace/software/docking/gnina/gnina
GNINA_ERROR
    exit 2
  fi
fi

cd "${script_dir}"

if docker compose version >/dev/null 2>&1; then
  compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  compose_cmd=(docker-compose)
else
  echo "Docker Compose is required but was not found." >&2
  echo "Install the Docker Compose plugin ('docker compose') or legacy docker-compose." >&2
  exit 127
fi

if [[ "${pull}" == "true" ]]; then
  "${compose_cmd[@]}" "${compose_files[@]}" pull db
fi

if [[ "${build}" == "true" && "${command}" != "down" ]]; then
  "${compose_cmd[@]}" "${compose_files[@]}" build ocdocker
fi

case "${command}" in
  up)
    "${compose_cmd[@]}" "${compose_files[@]}" up -d
    ;;
  down)
    "${compose_cmd[@]}" "${compose_files[@]}" down
    ;;
  shell)
    "${compose_cmd[@]}" "${compose_files[@]}" run --rm ocdocker bash
    ;;
  run)
    if [[ ${#compose_args[@]} -eq 0 ]]; then
      echo "No command provided after --." >&2
      exit 2
    fi
    "${compose_cmd[@]}" "${compose_files[@]}" run --rm ocdocker "${compose_args[@]}"
    ;;
  *)
    echo "Unsupported command mode '${command}'." >&2
    exit 2
    ;;
esac
