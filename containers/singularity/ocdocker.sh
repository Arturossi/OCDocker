#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

if [[ -z "${OCDOCKER_SINGULARITY_IMAGE:-}" ]]; then
  if [[ -f "${repo_root}/ocdocker.sif" ]]; then
    OCDOCKER_SINGULARITY_IMAGE="${repo_root}/ocdocker.sif"
  elif [[ -f "${script_dir}/ocdocker.sif" ]]; then
    OCDOCKER_SINGULARITY_IMAGE="${script_dir}/ocdocker.sif"
  else
    OCDOCKER_SINGULARITY_IMAGE=""
  fi
fi

runtime_cmd=()
runtime_note=""

declare -A seen_mounts=()
extra_mounts=()
args=()
cfg_source=""
workdir=""
dry_run=0

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

normalize_path() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p"
  elif command -v readlink >/dev/null 2>&1; then
    readlink -m "$p" 2>/dev/null || printf '%s\n' "$p"
  else
    printf '%s\n' "$p"
  fi
}

add_mount() {
  local m="$1"
  [[ -z "${m}" ]] && return 0

  [[ "${m}" == "~/"* ]] && m="${HOME}/${m#~/}"
  m="$(normalize_path "$m")"

  if [[ -n "${seen_mounts[${m}]:-}" ]]; then
    return 0
  fi
  seen_mounts["${m}"]=1
  extra_mounts+=("--bind" "${m}:${m}")
}

mount_path_or_parent() {
  local p="$1"
  [[ -z "${p}" ]] && return 0

  [[ "${p}" == "~/"* ]] && p="${HOME}/${p#~/}"
  p="$(normalize_path "$p")"

  if [[ -e "${p}" ]]; then
    if [[ -f "${p}" ]]; then
      add_mount "$(dirname "${p}")"
    else
      add_mount "${p}"
    fi
  else
    local parent
    parent="$(dirname "${p}")"
    if [[ -d "${parent}" ]]; then
      add_mount "${parent}"
    fi
  fi

  return 0
}

extract_mounts_from_cfg() {
  local cfg="$1"
  [[ -f "${cfg}" ]] || return 0

  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    local line key value
    line="${raw_line%%#*}"
    line="$(trim "${line}")"
    [[ -z "${line}" || "${line}" != *=* ]] && continue

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    [[ -z "${value}" ]] && continue

    case "${value}" in
      /*|~/*)
        mount_path_or_parent "${value}"
        ;;
      *)
        ;;
    esac
    # Keep shellcheck happy when set but currently unused in pattern-only parsing.
    : "${key}"
  done < "${cfg}"
}

resolve_cfg_source() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    if [[ -f "${script_dir}/OCDocker.cfg.singularity" ]]; then
      printf '%s\n' "${script_dir}/OCDocker.cfg.singularity"
    fi
    return 0
  fi

  if [[ "${candidate}" == "~/"* ]]; then
    candidate="${HOME}/${candidate#~/}"
  fi

  if [[ -f "${candidate}" ]]; then
    printf '%s\n' "$(normalize_path "${candidate}")"
    return 0
  fi

  # If this points to an in-container path (e.g., /opt/ocdocker/OCDocker.cfg), ignore.
  if [[ "${candidate}" == /opt/* || "${candidate}" == /workspace/* ]]; then
    return 0
  fi

  if [[ -f "${PWD}/${candidate}" ]]; then
    printf '%s\n' "$(normalize_path "${PWD}/${candidate}")"
    return 0
  fi
}

extract_conf_arg() {
  local i tok next
  for ((i=0; i<${#args[@]}; i++)); do
    tok="${args[$i]}"
    if [[ "${tok}" == "--conf" ]]; then
      if (( i + 1 < ${#args[@]} )); then
        next="${args[$((i + 1))]}"
        printf '%s\n' "${next}"
        return 0
      fi
    elif [[ "${tok}" == --conf=* ]]; then
      printf '%s\n' "${tok#--conf=}"
      return 0
    fi
  done
  return 0
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      shift
      [[ $# -gt 0 ]] || { echo "error: --image needs a .sif path" >&2; exit 2; }
      OCDOCKER_SINGULARITY_IMAGE="$1"
      shift
      ;;
    --image=*)
      OCDOCKER_SINGULARITY_IMAGE="${1#--image=}"
      shift
      ;;
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
    --workdir)
      shift
      [[ $# -gt 0 ]] || { echo "error: --workdir needs a path" >&2; exit 2; }
      workdir="$1"
      shift
      ;;
    --workdir=*)
      workdir="${1#--workdir=}"
      shift
      ;;
    --cfg-source)
      shift
      [[ $# -gt 0 ]] || { echo "error: --cfg-source needs a file path" >&2; exit 2; }
      cfg_source="$1"
      shift
      ;;
    --cfg-source=*)
      cfg_source="${1#--cfg-source=}"
      shift
      ;;
    --dry-run)
      dry_run=1
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

if [[ -z "${OCDOCKER_SINGULARITY_IMAGE}" ]]; then
  cat >&2 <<'EOF'
error: Singularity image not found.
Set OCDOCKER_SINGULARITY_IMAGE or pass --image /path/to/ocdocker.sif.
EOF
  exit 2
fi

add_mount "${PWD}"

if [[ -n "${workdir}" ]]; then
  mount_path_or_parent "${workdir}"
fi

if [[ -n "${OCDOCKER_SINGULARITY_MOUNTS:-${OCDOCKER_DOCKER_MOUNTS:-}}" ]]; then
  IFS=':' read -r -a env_mounts <<< "${OCDOCKER_SINGULARITY_MOUNTS:-${OCDOCKER_DOCKER_MOUNTS:-}}"
  for m in "${env_mounts[@]}"; do
    [[ -n "${m}" ]] && add_mount "${m}"
  done
fi

for a in "${args[@]}"; do
  path=""
  if [[ "${a}" == /* || "${a}" == ~/* ]]; then
    path="${a}"
  elif [[ "${a}" == *=/* || "${a}" == *=~/* ]]; then
    path="${a#*=}"
  fi
  [[ -z "${path}" ]] || mount_path_or_parent "${path}"
done

conf_from_args="$(extract_conf_arg)"
if [[ -z "${cfg_source}" ]]; then
  if [[ -n "${conf_from_args}" ]]; then
    cfg_source="${conf_from_args}"
  elif [[ -n "${OCDOCKER_CONFIG:-}" ]]; then
    cfg_source="${OCDOCKER_CONFIG}"
  fi
fi

resolved_cfg="$(resolve_cfg_source "${cfg_source}")"
if [[ -n "${resolved_cfg}" ]]; then
  mount_path_or_parent "${resolved_cfg}"
  extract_mounts_from_cfg "${resolved_cfg}"
fi

if ! resolve_runtime_cmd; then
  if [[ "${dry_run}" == "1" ]]; then
    runtime_cmd=(apptainer)
    runtime_note="(runtime binary not found; using placeholder for dry-run)"
  else
    echo "error: neither 'apptainer' nor 'singularity' was found in PATH." >&2
    exit 127
  fi
fi

exec_cmd=("${runtime_cmd[@]}" exec)
if [[ -n "${workdir}" ]]; then
  exec_cmd+=("--pwd" "$(normalize_path "${workdir}")")
fi
exec_cmd+=("${extra_mounts[@]}" "${OCDOCKER_SINGULARITY_IMAGE}" ocdocker "${args[@]}")

if [[ "${dry_run}" == "1" ]]; then
  printf 'Runtime: %s\n' "${runtime_cmd[*]}"
  [[ -n "${runtime_note}" ]] && printf 'Runtime note: %s\n' "${runtime_note}"
  printf 'Image: %s\n' "${OCDOCKER_SINGULARITY_IMAGE}"
  if [[ -n "${resolved_cfg:-}" ]]; then
    printf 'Config source: %s\n' "${resolved_cfg}"
  fi
  printf 'Command:\n'
  printf '  %q' "${exec_cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${exec_cmd[@]}"
