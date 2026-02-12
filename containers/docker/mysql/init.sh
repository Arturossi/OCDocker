#!/usr/bin/env bash
set -euo pipefail

optimize_db="${OCDOCKER_OPTIMIZEDB:-optimization}"
mysql_user="${MYSQL_USER:-ocdocker}"

if [[ -z "${optimize_db}" ]]; then
  exit 0
fi

if [[ ! "${optimize_db}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Invalid OCDOCKER_OPTIMIZEDB value: '${optimize_db}'" >&2
  exit 1
fi

if [[ ! "${mysql_user}" =~ ^[A-Za-z0-9_]+$ ]]; then
  echo "Invalid MYSQL_USER value: '${mysql_user}'" >&2
  exit 1
fi

mysql --protocol=socket -uroot -p"${MYSQL_ROOT_PASSWORD}" <<SQL
CREATE DATABASE IF NOT EXISTS \`${optimize_db}\`
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON \`${optimize_db}\`.* TO '${mysql_user}'@'%';
FLUSH PRIVILEGES;
SQL
