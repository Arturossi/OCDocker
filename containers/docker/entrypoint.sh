#!/usr/bin/env bash
set -euo pipefail

backend="${OCDOCKER_DB_BACKEND:-postgresql}"
case "${backend,,}" in
  postgresql|postgres|pgsql)
    export OCDOCKER_DB_BACKEND=postgresql
    default_config=/etc/ocdocker/OCDocker.cfg.postgresql
    ;;
  mysql|mariadb)
    export OCDOCKER_DB_BACKEND=mysql
    default_config=/etc/ocdocker/OCDocker.cfg.mysql
    ;;
  sqlite)
    export OCDOCKER_DB_BACKEND=sqlite
    default_config=/etc/ocdocker/OCDocker.cfg.postgresql
    ;;
  *)
    echo "Unsupported OCDOCKER_DB_BACKEND='${backend}'. Use postgresql, mysql, or sqlite." >&2
    exit 2
    ;;
esac

export OCDOCKER_CONFIG="${OCDOCKER_CONFIG:-${default_config}}"

set_config_value() {
  local cfg="$1"
  local key="$2"
  local value="$3"

  [[ -n "${value}" ]] || return 0
  if grep -Eq "^[[:space:]]*${key}[[:space:]]*=" "${cfg}"; then
    sed -i -E "s|^[[:space:]]*${key}[[:space:]]*=.*|${key} = ${value}|" "${cfg}"
  else
    printf '%s = %s\n' "${key}" "${value}" >> "${cfg}"
  fi
}

if [[ "${OCDOCKER_DB_BACKEND}" != "sqlite" ]]; then
  if [[ -z "${OCDOCKER_DB_PASS:-}" ]]; then
    echo "OCDOCKER_DB_PASS is required when Docker uses ${OCDOCKER_DB_BACKEND}." >&2
    exit 2
  fi

  runtime_config="/tmp/ocdocker/OCDocker.cfg"
  mkdir -p "$(dirname "${runtime_config}")"
  cp "${OCDOCKER_CONFIG}" "${runtime_config}"
  export OCDOCKER_CONFIG="${runtime_config}"

  set_config_value "${OCDOCKER_CONFIG}" "DB_BACKEND" "${OCDOCKER_DB_BACKEND}"
  set_config_value "${OCDOCKER_CONFIG}" "USER" "${OCDOCKER_DB_USER:-ocdocker}"
  set_config_value "${OCDOCKER_CONFIG}" "PASSWORD" "${OCDOCKER_DB_PASS}"
  set_config_value "${OCDOCKER_CONFIG}" "DATABASE" "${OCDOCKER_DATABASE:-ocdocker}"
  set_config_value "${OCDOCKER_CONFIG}" "OPTIMIZEDB" "${OCDOCKER_OPTIMIZEDB:-optimization}"
fi

if [[ "${OCDOCKER_LOCK_WORKSPACE:-true}" == "true" ]]; then
  for candidate in "${OCDOCKER_INPUT_DIR:-}" "${OCDOCKER_OUTPUT_DIR:-}"; do
    [[ -z "${candidate}" ]] && continue
    case "${candidate}" in
      /workspace|/workspace/*) ;;
      *)
        echo "WARNING: '${candidate}' is outside /workspace. Docker mode only mounts /workspace by default." >&2
        echo "         Mount an extra path explicitly or use /workspace-relative paths." >&2
        ;;
    esac
  done
fi

if [[ "$#" -eq 0 ]]; then
  exec bash
fi

exec micromamba run -n ocdocker "$@"
