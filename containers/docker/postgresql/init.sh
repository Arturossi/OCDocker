#!/usr/bin/env bash
set -euo pipefail

primary_db="${POSTGRES_DB:-ocdocker}"
optimize_db="${OCDOCKER_OPTIMIZEDB:-optimization}"

if [[ -z "${optimize_db}" || "${optimize_db}" == "${primary_db}" ]]; then
  exit 0
fi

psql -v ON_ERROR_STOP=1 \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set=optimizedb="${optimize_db}" <<'SQL'
SELECT format('CREATE DATABASE %I', :'optimizedb')
WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = :'optimizedb'
)\gexec
SQL
