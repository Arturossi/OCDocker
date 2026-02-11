#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
compose_file="${script_dir}/docker-compose.yml"
compose_mysql_override="${script_dir}/docker-compose.mysql.yml"

# Default to the parent of the repo root if HOST_OCDOCKER_ROOT isn't set.
if [[ -z "${HOST_OCDOCKER_ROOT:-}" ]]; then
  export HOST_OCDOCKER_ROOT
  HOST_OCDOCKER_ROOT="$(cd "${repo_root}/.." && pwd)"
fi

declare -A seen_mounts=()
extra_mounts=()

normalize_backend() {
  case "${1:-}" in
    postgresql|postgres|pgsql) printf 'postgresql\n' ;;
    mysql|mariadb) printf 'mysql\n' ;;
    sqlite|sqlite3) printf 'sqlite\n' ;;
    *) printf 'postgresql\n' ;;
  esac
}

extract_backend_from_cfg() {
  local cfg="$1"
  [[ -f "${cfg}" ]] || return 0
  grep -E "^\s*DB_BACKEND\s*=" "${cfg}" | tail -n1 | awk -F= '{print $2}' | xargs
}

add_mount() {
  local m="$1"
  [[ -z "${m}" ]] && return 0

  # Normalize if possible; tolerate non-existent paths for planned outputs.
  if command -v realpath >/dev/null 2>&1; then
    m="$(realpath -m "${m}")"
  elif command -v readlink >/dev/null 2>&1; then
    m="$(readlink -m "${m}" 2>/dev/null || echo "${m}")"
  fi

  if [[ -n "${seen_mounts[${m}]:-}" ]]; then
    return 0
  fi
  seen_mounts["${m}"]=1
  extra_mounts+=("-v" "${m}:${m}")
}

# Parse explicit --mount flags (repeatable).
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mount)
      shift
      [[ $# -gt 0 ]] || { echo "error: --mount needs a path" >&2; exit 2; }
      add_mount "$1"
      shift
      ;;
    --mount=*)
      add_mount "${1#--mount=}"
      shift
      ;;
    --)
      shift
      args+=("$@")
      break
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# Add mounts from env (colon-separated list).
if [[ -n "${OCDOCKER_DOCKER_MOUNTS:-}" ]]; then
  IFS=':' read -r -a env_mounts <<< "${OCDOCKER_DOCKER_MOUNTS}"
  for m in "${env_mounts[@]}"; do
    [[ -n "${m}" ]] && add_mount "${m}"
  done
fi

# Auto-detect absolute paths in args (including --key=/path).
for a in "${args[@]}"; do
  path=""
  if [[ "${a}" == /* ]]; then
    path="${a}"
  elif [[ "${a}" == *=/* ]]; then
    path="${a#*=}"
  fi

  [[ -z "${path}" ]] && continue

  if [[ -e "${path}" ]]; then
    if [[ -f "${path}" ]]; then
      add_mount "$(dirname "${path}")"
    else
      add_mount "${path}"
    fi
  else
    # For output paths that don't exist yet, mount the parent directory if it does.
    parent="$(dirname "${path}")"
    [[ -d "${parent}" ]] && add_mount "${parent}"
  fi
done

selected_backend_raw="${OCDOCKER_DB_BACKEND:-${DB_BACKEND:-}}"
if [[ -z "${selected_backend_raw}" ]]; then
  if [[ -n "${OCDOCKER_CONFIG:-}" && -f "${OCDOCKER_CONFIG}" ]]; then
    selected_backend_raw="$(extract_backend_from_cfg "${OCDOCKER_CONFIG}")"
  elif [[ -f "${repo_root}/OCDocker.cfg" ]]; then
    selected_backend_raw="$(extract_backend_from_cfg "${repo_root}/OCDocker.cfg")"
  fi
fi
selected_backend="$(normalize_backend "${selected_backend_raw:-postgresql}")"
compose_args=(-f "${compose_file}")
if [[ "${selected_backend}" == "mysql" ]]; then
  compose_args+=(-f "${compose_mysql_override}")
fi

exec docker compose "${compose_args[@]}" run --rm \
  "${extra_mounts[@]}" \
  --entrypoint ocdocker \
  ocdocker "${args[@]}"
