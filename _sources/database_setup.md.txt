# Database Setup

OCDocker supports PostgreSQL, MySQL, and SQLite.

Use SQLite for development, tests, and quick local experiments. Use PostgreSQL
or MySQL for persistent, concurrent, or long-running workloads.

## SQLite

```bash
export OCDOCKER_DB_BACKEND=sqlite
ocdocker doctor
```

Config equivalent:

```ini
DB_BACKEND = sqlite
SQLITE_PATH = /path/to/ocdocker.db
```

If `SQLITE_PATH` is omitted, OCDocker uses its default local SQLite path.

## PostgreSQL

PostgreSQL is the default server backend.

Install and start PostgreSQL on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

Create the user and databases:

```bash
sudo -u postgres psql
```

```text
CREATE USER ocdocker WITH PASSWORD '<db_password>';
CREATE DATABASE ocdocker OWNER ocdocker;
CREATE DATABASE optimization OWNER ocdocker;
GRANT ALL PRIVILEGES ON DATABASE ocdocker TO ocdocker;
GRANT ALL PRIVILEGES ON DATABASE optimization TO ocdocker;
\q
```

Config example:

```ini
DB_BACKEND = postgresql
HOST = localhost
PORT = 5432
USER = ocdocker
PASSWORD = <db_password>
DATABASE = ocdocker
OPTIMIZEDB = optimization
```

Install Python DB dependencies when needed:

```bash
pip install "ocdocker[db]"
```

## MySQL

Install and start MySQL on Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y mysql-server
sudo systemctl enable --now mysql
```

Create the user and databases:

```bash
sudo mysql
```

```sql
CREATE DATABASE IF NOT EXISTS ocdocker
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE IF NOT EXISTS optimization
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'ocdocker'@'localhost' IDENTIFIED BY '<db_password>';
GRANT ALL PRIVILEGES ON ocdocker.* TO 'ocdocker'@'localhost';
GRANT ALL PRIVILEGES ON optimization.* TO 'ocdocker'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

If OCDocker connects from a container or another host, create the matching host
grant, for example `'ocdocker'@'%'`, and harden the network configuration.

Config example:

```ini
DB_BACKEND = mysql
HOST = localhost
PORT = 3306
USER = ocdocker
PASSWORD = <db_password>
DATABASE = ocdocker
OPTIMIZEDB = optimization
```

## Notes

- PostgreSQL/MySQL require `HOST`, `USER`, `PASSWORD`, `DATABASE`, and integer
  `PORT` values when database initialization is requested.
- Missing PostgreSQL/MySQL databases are created only through explicit setup
  intent, such as CLI paths that initialize DB storage or application code that
  passes `create_db_if_missing=True`.
- For CI and local unit tests, prefer SQLite unless the test specifically targets
  PostgreSQL or MySQL.
