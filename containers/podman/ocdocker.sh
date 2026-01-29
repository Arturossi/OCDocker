#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
compose_file="${script_dir}/podman-compose.yml"

if [[ -z "${HOST_OCDOCKER_ROOT:-}" ]]; then
  export HOST_OCDOCKER_ROOT
  HOST_OCDOCKER_ROOT="$(cd "${repo_root}/.." && pwd)"
fi

compose_cmd=()
if command -v podman-compose >/dev/null 2>&1; then
  compose_cmd=(podman-compose)
else
  compose_cmd=(podman compose)
fi

declare -A seen_mounts=()
extra_mounts=()

add_mount() {
  local m="$1"
  [[ -z "${m}" ]] && return 0

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

if [[ -n "${OCDOCKER_PODMAN_MOUNTS:-${OCDOCKER_DOCKER_MOUNTS:-}}" ]]; then
  IFS=':' read -r -a env_mounts <<< "${OCDOCKER_PODMAN_MOUNTS:-${OCDOCKER_DOCKER_MOUNTS:-}}"
  for m in "${env_mounts[@]}"; do
    [[ -n "${m}" ]] && add_mount "${m}"
  done
fi

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
    parent="$(dirname "${path}")"
    [[ -d "${parent}" ]] && add_mount "${parent}"
  fi
done

exec "${compose_cmd[@]}" -f "${compose_file}" run --rm \
  "${extra_mounts[@]}" \
  --entrypoint ocdocker \
  ocdocker "${args[@]}"
