#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

runtime_cmd=()
runtime_note=""

resolve_runtime_cmd() {
  if command -v apptainer >/dev/null 2>&1; then
    runtime_cmd=(apptainer)
    return 0
  fi
  if command -v singularity >/dev/null 2>&1; then
    runtime_cmd=(singularity)
    return 0
  fi
  return 1
}

ensure_runtime() {
  if ! resolve_runtime_cmd; then
    echo "error: neither 'apptainer' nor 'singularity' was found in PATH." >&2
    exit 127
  fi
}

usage() {
  cat <<'EOF'
Usage:
  containers/singularity/mysql.sh <command> [options]

Commands:
  start    Start MySQL instance (default image: docker://mysql:8.4)
  stop     Stop MySQL instance
  status   Show whether the instance is running
  ping     Wait for and verify MySQL readiness

Options:
  --name NAME          Instance name (default: ocdocker-mysql)
  --image IMAGE        SIF path or OCI reference (default: docker://mysql:8.4)
  --data-dir DIR       Host dir for MySQL data (default: ./tmp/singularity-mysql)
  --init-dir DIR       Host dir mounted at /docker-entrypoint-initdb.d
  --port PORT          MySQL port (default: 3306)
  --root-password PWD  Root password (default: MYSQL_ROOT_PASSWORD or rootpass)
  --db NAME            Initial DB name (default: ocdocker)
  --user NAME          DB user (default: ocdocker)
  --password PWD       DB user password (default: OCDOCKER_DB_PASS or ocdocker_pass)
  --fakeroot           Start instance with fakeroot
  --dry-run            Print resolved command without executing it
  -h, --help           Show this help
EOF
}

instance_name="${OCDOCKER_MYSQL_INSTANCE_NAME:-ocdocker-mysql}"
image="${OCDOCKER_MYSQL_SINGULARITY_IMAGE:-docker://mysql:8.4}"
data_dir="${OCDOCKER_MYSQL_DATA_DIR:-${repo_root}/tmp/singularity-mysql}"
init_dir="${OCDOCKER_MYSQL_INIT_DIR:-${script_dir}/mysql}"
root_password="${MYSQL_ROOT_PASSWORD:-rootpass}"
db_name="${OCDOCKER_MYSQL_DATABASE:-ocdocker}"
db_user="${OCDOCKER_MYSQL_USER:-ocdocker}"
db_password="${OCDOCKER_DB_PASS:-${OCDOCKER_MYSQL_PASSWORD:-ocdocker_pass}}"
port="${OCDOCKER_MYSQL_PORT:-3306}"
use_fakeroot=0
dry_run=0

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
  usage
  exit 2
fi
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      shift
      [[ $# -gt 0 ]] || { echo "error: --name requires a value" >&2; exit 2; }
      instance_name="$1"
      shift
      ;;
    --name=*)
      instance_name="${1#--name=}"
      shift
      ;;
    --image)
      shift
      [[ $# -gt 0 ]] || { echo "error: --image requires a value" >&2; exit 2; }
      image="$1"
      shift
      ;;
    --image=*)
      image="${1#--image=}"
      shift
      ;;
    --data-dir)
      shift
      [[ $# -gt 0 ]] || { echo "error: --data-dir requires a value" >&2; exit 2; }
      data_dir="$1"
      shift
      ;;
    --data-dir=*)
      data_dir="${1#--data-dir=}"
      shift
      ;;
    --init-dir)
      shift
      [[ $# -gt 0 ]] || { echo "error: --init-dir requires a value" >&2; exit 2; }
      init_dir="$1"
      shift
      ;;
    --init-dir=*)
      init_dir="${1#--init-dir=}"
      shift
      ;;
    --port)
      shift
      [[ $# -gt 0 ]] || { echo "error: --port requires a value" >&2; exit 2; }
      port="$1"
      shift
      ;;
    --port=*)
      port="${1#--port=}"
      shift
      ;;
    --root-password)
      shift
      [[ $# -gt 0 ]] || { echo "error: --root-password requires a value" >&2; exit 2; }
      root_password="$1"
      shift
      ;;
    --root-password=*)
      root_password="${1#--root-password=}"
      shift
      ;;
    --db)
      shift
      [[ $# -gt 0 ]] || { echo "error: --db requires a value" >&2; exit 2; }
      db_name="$1"
      shift
      ;;
    --db=*)
      db_name="${1#--db=}"
      shift
      ;;
    --user)
      shift
      [[ $# -gt 0 ]] || { echo "error: --user requires a value" >&2; exit 2; }
      db_user="$1"
      shift
      ;;
    --user=*)
      db_user="${1#--user=}"
      shift
      ;;
    --password)
      shift
      [[ $# -gt 0 ]] || { echo "error: --password requires a value" >&2; exit 2; }
      db_password="$1"
      shift
      ;;
    --password=*)
      db_password="${1#--password=}"
      shift
      ;;
    --fakeroot)
      use_fakeroot=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage
      exit 2
      ;;
  esac
done

normalize_path() {
  local p="$1"
  [[ "${p}" == "~/"* ]] && p="${HOME}/${p#~/}"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -m "$p" 2>/dev/null || printf '%s\n' "$p"
  else
    printf '%s\n' "$p"
  fi
}

data_dir="$(normalize_path "${data_dir}")"
init_dir="$(normalize_path "${init_dir}")"

runtime_env=(
  "APPTAINERENV_MYSQL_ROOT_PASSWORD=${root_password}"
  "APPTAINERENV_MYSQL_DATABASE=${db_name}"
  "APPTAINERENV_MYSQL_USER=${db_user}"
  "APPTAINERENV_MYSQL_PASSWORD=${db_password}"
  "SINGULARITYENV_MYSQL_ROOT_PASSWORD=${root_password}"
  "SINGULARITYENV_MYSQL_DATABASE=${db_name}"
  "SINGULARITYENV_MYSQL_USER=${db_user}"
  "SINGULARITYENV_MYSQL_PASSWORD=${db_password}"
)

instance_start_cmd() {
  local cmd_arr=("${runtime_cmd[@]}" instance start)
  if [[ "${use_fakeroot}" == "1" ]]; then
    cmd_arr+=("--fakeroot")
  fi
  cmd_arr+=("--bind" "${data_dir}:/var/lib/mysql")
  if [[ -d "${init_dir}" ]]; then
    cmd_arr+=("--bind" "${init_dir}:/docker-entrypoint-initdb.d:ro")
  fi
  cmd_arr+=("${image}" "${instance_name}" "--port=${port}")
  printf '%s\0' "${cmd_arr[@]}"
}

instance_running() {
  ensure_runtime
  if "${runtime_cmd[@]}" instance list 2>/dev/null | awk '{print $1}' | grep -qx "${instance_name}"; then
    return 0
  fi
  return 1
}

ping_instance() {
  ensure_runtime
  local i
  for i in $(seq 1 90); do
    if "${runtime_cmd[@]}" exec "instance://${instance_name}" \
      mysqladmin --protocol=tcp --host=127.0.0.1 --port="${port}" \
      --user=root --password="${root_password}" ping --silent >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_instance() {
  if ! resolve_runtime_cmd; then
    if [[ "${dry_run}" == "1" ]]; then
      runtime_cmd=(apptainer)
      runtime_note="(runtime binary not found; using placeholder for dry-run)"
    else
      echo "error: neither 'apptainer' nor 'singularity' was found in PATH." >&2
      exit 127
    fi
  fi
  mkdir -p "${data_dir}"
  if [[ ! -d "${init_dir}" ]]; then
    echo "warning: init directory not found: ${init_dir}. Starting without init scripts." >&2
  fi

  if [[ "${dry_run}" != "1" ]] && instance_running; then
    echo "MySQL instance '${instance_name}' is already running."
    return 0
  fi

  local -a cmd_arr=()
  while IFS= read -r -d '' x; do
    cmd_arr+=("$x")
  done < <(instance_start_cmd)

  if [[ "${dry_run}" == "1" ]]; then
    echo "Runtime: ${runtime_cmd[*]}"
    [[ -n "${runtime_note}" ]] && echo "Runtime note: ${runtime_note}"
    echo "Image: ${image}"
    echo "Data dir: ${data_dir}"
    echo "Init dir: ${init_dir}"
    echo "Instance: ${instance_name}"
    echo "Command:"
    printf '  env'
    local env_kv
    for env_kv in "${runtime_env[@]}"; do
      printf ' %q' "${env_kv}"
    done
    for part in "${cmd_arr[@]}"; do
      printf ' %q' "${part}"
    done
    printf '\n'
    return 0
  fi

  env "${runtime_env[@]}" "${cmd_arr[@]}"

  echo "Started MySQL instance '${instance_name}'. Waiting for readiness..."
  if ping_instance; then
    echo "MySQL is ready on localhost:${port}"
    echo "Use DB_BACKEND=mysql, HOST=localhost, PORT=${port}, USER=${db_user}, PASSWORD=${db_password} in OCDocker.cfg."
  else
    echo "error: MySQL instance started but did not become ready in time." >&2
    exit 1
  fi
}

stop_instance() {
  ensure_runtime
  if ! instance_running; then
    echo "MySQL instance '${instance_name}' is not running."
    return 0
  fi
  "${runtime_cmd[@]}" instance stop "${instance_name}"
  echo "Stopped MySQL instance '${instance_name}'."
}

status_instance() {
  ensure_runtime
  if instance_running; then
    echo "MySQL instance '${instance_name}': running"
    return 0
  fi
  echo "MySQL instance '${instance_name}': stopped"
  return 1
}

case "${cmd}" in
  start)
    start_instance
    ;;
  stop)
    stop_instance
    ;;
  status)
    status_instance
    ;;
  ping)
    if ping_instance; then
      echo "MySQL instance '${instance_name}' is reachable."
    else
      echo "MySQL instance '${instance_name}' is not reachable." >&2
      exit 1
    fi
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "error: unknown command '${cmd}'" >&2
    usage
    exit 2
    ;;
esac
