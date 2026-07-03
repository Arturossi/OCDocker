#!/usr/bin/env python3

# Description
###############################################################################
"""
Sets of classes and functions that are used for setting up the database.

Usage:

import OCDocker.DB.DBMinimal as ocdbmin
"""

# Imports
###############################################################################
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

try:
    from sqlalchemy import create_engine as sqlalchemy_create_engine
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.engine.url import URL, make_url
    from sqlalchemy.orm import scoped_session, sessionmaker
except ModuleNotFoundError as exc:  # pragma: no cover - exercised with import stubs
    SQLALCHEMY_IMPORT_ERROR: Optional[ModuleNotFoundError] = exc
    sqlalchemy_create_engine = None
    Engine = Any
    URL = Any
    make_url = None
    scoped_session = Any
    sessionmaker = None
else:
    SQLALCHEMY_IMPORT_ERROR = None

try:
    from sqlalchemy.exc import NoSuchModuleError, OperationalError, SQLAlchemyError
except ModuleNotFoundError:  # pragma: no cover - test stubs may omit sqlalchemy.exc
    NoSuchModuleError = OperationalError = SQLAlchemyError = Exception

import OCDocker.Error as ocerror

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################


class DatabaseError(RuntimeError):
    """Base exception for database setup and access failures."""


class MissingDatabaseDependencyError(DatabaseError, ImportError):
    """Raised when an optional database dependency is not installed."""


class DatabaseConfigurationError(DatabaseError, ValueError):
    """Raised when database configuration is invalid or incomplete."""


class DatabaseCreationNotAllowedError(DatabaseError):
    """Raised when a missing remote database would be created implicitly."""


class DatabaseCreationError(DatabaseError):
    """Raised when explicit database creation fails."""


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection/check fails."""


@dataclass(frozen=True)
class DatabaseSettings:
    """Validated database connection settings."""

    backend: str
    host: str = ""
    user: str = ""
    password: str = ""
    database: str = ""
    optimizedb: str = ""
    port: Optional[int] = None
    sqlite_path: str = ""


# Functions
###############################################################################
## Private ##

SUPPORTED_BACKENDS = ("postgresql", "mysql", "sqlite")
OPTIONAL_DB_INSTALL_HINT = 'pip install "ocdocker[db]"'


def _missing_dependency_message(module_name: str) -> str:
    return (
        f"Missing optional database dependency '{module_name}'. "
        f"Install database support with: {OPTIONAL_DB_INSTALL_HINT}."
    )


def _ensure_sqlalchemy_available() -> None:
    if SQLALCHEMY_IMPORT_ERROR is not None:
        missing = getattr(SQLALCHEMY_IMPORT_ERROR, "name", None) or "sqlalchemy"
        raise MissingDatabaseDependencyError(
            _missing_dependency_message(missing)
        ) from SQLALCHEMY_IMPORT_ERROR
    if make_url is None or sqlalchemy_create_engine is None or sessionmaker is None:
        raise MissingDatabaseDependencyError(_missing_dependency_message("sqlalchemy"))


def _load_sqlalchemy_utils() -> Tuple[Any, Any]:
    try:
        from sqlalchemy_utils import create_database as sqla_create_database
        from sqlalchemy_utils import database_exists as sqla_database_exists
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or "sqlalchemy_utils"
        raise MissingDatabaseDependencyError(
            _missing_dependency_message(missing)
        ) from exc

    return sqla_create_database, sqla_database_exists


def normalize_db_backend(raw_backend: str) -> Optional[str]:
    '''Normalize backend names and SQLAlchemy driver names.'''

    value = str(raw_backend or "").strip().lower()
    if not value:
        return None

    base_driver = value.split("+", 1)[0]
    if base_driver in ("postgresql", "postgres", "pgsql"):
        return "postgresql"
    if base_driver in ("mysql", "mariadb"):
        return "mysql"
    if base_driver in ("sqlite", "sqlite3"):
        return "sqlite"
    return None


def _url_backend(url: URL) -> str:
    backend = normalize_db_backend(getattr(url, "drivername", ""))
    if backend is None:
        raise DatabaseConfigurationError(
            f"Unsupported database backend '{getattr(url, 'drivername', '')}'. "
            "Use one of: postgresql, mysql, sqlite."
        )
    return backend


def _sqlite_path_is_memory(sqlite_path: str) -> bool:
    return sqlite_path in ("", ":memory:") or sqlite_path.startswith("file::memory:")


def _validate_sqlite_path(sqlite_path: str, *, require_sqlite_path: bool) -> str:
    sqlite_path = str(sqlite_path or "").strip()
    if require_sqlite_path and not sqlite_path:
        raise DatabaseConfigurationError(
            "SQLite backend requires SQLITE_PATH or OCDOCKER_SQLITE_PATH when a path is required."
        )

    if not sqlite_path:
        return ":memory:"
    if _sqlite_path_is_memory(sqlite_path):
        return sqlite_path

    path = Path(sqlite_path).expanduser()
    if path.exists() and path.is_dir():
        raise DatabaseConfigurationError(
            f"SQLite path points to a directory, not a database file: {path}"
        )
    if not path.parent.exists():
        raise DatabaseConfigurationError(
            f"SQLite database directory does not exist: {path.parent}"
        )

    return str(path)


def database_exists(url: Union[str, URL]) -> bool:
    '''Return whether a database exists using sqlalchemy-utils.'''

    _ensure_sqlalchemy_available()
    if isinstance(url, str):
        url = make_url(url)
    _, sqla_database_exists = _load_sqlalchemy_utils()
    return bool(sqla_database_exists(url))


def create_database(url: Union[str, URL]) -> None:
    '''Create a database using sqlalchemy-utils.'''

    _ensure_sqlalchemy_available()
    if isinstance(url, str):
        url = make_url(url)
    sqla_create_database, _ = _load_sqlalchemy_utils()
    sqla_create_database(url)


## Public ##


def validate_database_config(
    backend: str,
    *,
    host: str = "",
    user: str = "",
    password: str = "",
    database: str = "",
    optimizedb: str = "",
    port: Union[int, str, None] = None,
    sqlite_path: str = "",
    require_credentials: bool = False,
    require_sqlite_path: bool = False,
) -> DatabaseSettings:
    '''Validate database connection configuration.

    Parameters
    ----------
    backend : str
        Database backend name or alias.
    require_credentials : bool, optional
        Require host/user/password/database for PostgreSQL/MySQL.
    require_sqlite_path : bool, optional
        Require an explicit SQLite path instead of using ``:memory:``.
    '''

    normalized_backend = normalize_db_backend(backend)
    if normalized_backend is None:
        raise DatabaseConfigurationError(
            f"Unsupported database backend '{backend}'. Use one of: postgresql, mysql, sqlite."
        )

    if normalized_backend == "sqlite":
        validated_sqlite_path = _validate_sqlite_path(
            sqlite_path,
            require_sqlite_path=require_sqlite_path,
        )
        return DatabaseSettings(
            backend=normalized_backend,
            database=validated_sqlite_path,
            sqlite_path=validated_sqlite_path,
        )

    if port in (None, ""):
        normalized_port = 5432 if normalized_backend == "postgresql" else 3306
    else:
        try:
            normalized_port = int(port)
        except (TypeError, ValueError) as exc:
            raise DatabaseConfigurationError(
                f"Port for {normalized_backend} must be an integer; got {port!r}."
            ) from exc

    if require_credentials:
        required = {
            "HOST": host,
            "USER": user,
            "PASSWORD": password,
            "DATABASE": database,
        }
        missing = [
            name for name, value in required.items() if not str(value or "").strip()
        ]
        if missing:
            raise DatabaseConfigurationError(
                f"{normalized_backend} database initialization requires: {', '.join(missing)}."
            )

    return DatabaseSettings(
        backend=normalized_backend,
        host=str(host or "").strip(),
        user=str(user or "").strip(),
        password=str(password or ""),
        database=str(database or "").strip(),
        optimizedb=str(optimizedb or "").strip(),
        port=normalized_port,
    )


def build_database_urls(settings: DatabaseSettings) -> Tuple[URL, URL]:
    '''Build primary and optimization SQLAlchemy URLs from validated settings.'''

    _ensure_sqlalchemy_available()

    if settings.backend == "sqlite":
        url = URL.create(
            drivername="sqlite", database=settings.sqlite_path or ":memory:"
        )
        return url, url

    if settings.backend == "mysql":
        drivername = "mysql+pymysql"
    elif settings.backend == "postgresql":
        drivername = "postgresql+psycopg"
    else:
        raise DatabaseConfigurationError(
            f"Unsupported database backend '{settings.backend}'. Use one of: postgresql, mysql, sqlite."
        )

    primary_url = URL.create(
        drivername=drivername,
        host=settings.host,
        username=settings.user,
        password=settings.password,
        database=settings.database,
        port=settings.port,
    )
    optimization_url = URL.create(
        drivername=drivername,
        host=settings.host,
        username=settings.user,
        password=settings.password,
        database=settings.optimizedb or "optimization",
        port=settings.port,
    )
    return primary_url, optimization_url


def get_default_engine() -> Any:
    '''Return the explicitly initialized default engine, if present.'''

    try:
        import OCDocker.Initialise as init
    except ImportError:
        return None
    return getattr(init, "engine", None)


def get_default_session() -> Any:
    '''Return the explicitly initialized default session factory, if present.'''

    try:
        import OCDocker.Initialise as init
    except ImportError:
        return None
    return getattr(init, "session", None)


def cleanup_engine(engine: Optional[Engine]) -> None:
    '''Clean up an engine by disposing of all connections in the pool.

    This function closes all connections in the connection pool and disposes of
    the engine. It's automatically called on application shutdown via atexit handlers.

    Parameters
    ----------
    engine : Engine | None
        The engine to clean up.

    Notes
    -----
    - This is safe to call multiple times (idempotent)
    - Errors during cleanup are silently ignored
    - Typically called automatically on application exit
    - Prevents connection leaks, especially important for pooled DB backends
    '''
    if engine is not None:
        try:
            # Dispose of all connections in the pool
            # close=True ensures connections are properly closed, not just returned to pool
            engine.dispose(close=True)
        except (AttributeError, RuntimeError):
            # Ignore errors during cleanup (engine may already be disposed)
            pass


def cleanup_session(session: Optional[scoped_session]) -> None:
    '''Clean up a scoped session by removing all sessions from the registry.

    This function removes all thread-local session instances from the scoped_session
    registry. It's automatically called on application shutdown via atexit handlers.

    Parameters
    ----------
    session : scoped_session | None
        The scoped session to clean up.

    Notes
    -----
    - This is safe to call multiple times (idempotent)
    - Errors during cleanup are silently ignored
    - Typically called automatically on application exit
    '''
    if session is not None:
        try:
            # Remove all thread-local sessions from the registry
            # This closes all active sessions and releases connections
            session.remove()
        except (AttributeError, RuntimeError):
            # Ignore errors during cleanup (session may already be closed or removed)
            pass


def create_database_if_not_exists(
    url: Union[str, URL],
    *,
    create_if_missing: bool = False,
) -> bool:
    '''Create the database only when explicitly allowed.

    Parameters
    ----------
    url : str | sqlalchemy.engine.url.URL
        The database url (string or URL object).
    create_if_missing : bool, optional
        If True, create a missing database. The default is False to avoid
        silently creating remote PostgreSQL/MySQL databases.

    Returns
    -------
    bool
        True if this call created a database, False otherwise.
    '''

    _ensure_sqlalchemy_available()
    if isinstance(url, str):
        url = make_url(url)

    backend = _url_backend(url)
    if backend == "sqlite":
        database = getattr(url, "database", None) or ""
        _ = _validate_sqlite_path(database, require_sqlite_path=False)
        return False

    try:
        exists = database_exists(url)
    except OperationalError as exc:
        raise DatabaseConnectionError(
            f"Could not check whether {backend} database '{url.database}' exists. "
            "Check host, port, credentials, and server availability."
        ) from exc
    except SQLAlchemyError as exc:
        raise DatabaseConnectionError(
            f"Could not check whether {backend} database '{url.database}' exists: {exc}"
        ) from exc

    if exists:
        return False

    if not create_if_missing:
        raise DatabaseCreationNotAllowedError(
            f"{backend} database '{url.database}' does not exist. "
            "Pass create_if_missing=True to create it explicitly."
        )

    try:
        create_database(url)
    except OperationalError as exc:
        raise DatabaseCreationError(
            f"Could not create {backend} database '{url.database}'. "
            "Check permissions, credentials, and server availability."
        ) from exc
    except SQLAlchemyError as exc:
        raise DatabaseCreationError(
            f"Could not create {backend} database '{url.database}': {exc}"
        ) from exc

    print(f"Created {backend} database '{url.database}'.")
    return True


def create_engine(
    url: Union[str, URL],
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: int = 30,
    pool_recycle: int = 3600,
) -> Engine:
    '''Create the engine with connection pooling.

    Parameters
    ----------
    url : str | sqlalchemy.engine.url.URL
        The database url (string or URL object).
    echo : bool
        Echo the SQL commands.
    pool_size : int, optional
        Number of connections to maintain in the pool. Default is 5.
    max_overflow : int, optional
        Maximum number of connections to allow beyond pool_size. Default is 10.
    pool_timeout : int, optional
        Seconds to wait before giving up on getting a connection. Default is 30.
    pool_recycle : int, optional
        Seconds after which a connection is recreated. Default is 3600 (1 hour).

    Returns
    -------
    Engine : sqlalchemy.engine.base.Engine
        The engine with connection pooling configured.
    '''

    _ensure_sqlalchemy_available()
    if isinstance(url, str):
        url = make_url(url)
    backend = _url_backend(url)

    # Create the engine with connection pooling configuration
    # For SQLite, pooling is handled differently, so we only apply pooling for non-SQLite databases
    try:
        if backend == "sqlite":
            # SQLite doesn't need connection pooling in the same way
            engine = sqlalchemy_create_engine(url, echo=echo, pool_pre_ping=True)
        else:
            # PostgreSQL/MySQL and other client/server databases: configure connection pooling
            engine = sqlalchemy_create_engine(
                url,
                echo=echo,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # Verify connections before using them
            )
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or "database driver"
        raise MissingDatabaseDependencyError(
            _missing_dependency_message(missing)
        ) from exc
    except NoSuchModuleError as exc:
        raise MissingDatabaseDependencyError(
            f"Missing SQLAlchemy database driver for '{url.drivername}'. "
            f"Install database support with: {OPTIONAL_DB_INSTALL_HINT}."
        ) from exc
    except OperationalError as exc:
        raise DatabaseConnectionError(
            f"Could not create engine for {backend} database '{url.database}'. "
            "Check host, port, credentials, and server availability."
        ) from exc
    except SQLAlchemyError as exc:
        raise DatabaseError(
            f"Could not create database engine for '{url.drivername}': {exc}"
        ) from exc

    # Return the engine (despite the lint flagging as a MockConnection, it is an Engine)
    return engine


def create_session(engine: Optional[Engine]) -> Optional[scoped_session]:
    '''Create a scoped session for database operations.

    Parameters
    ----------
    engine : from sqlalchemy.engine.base.Engine | None
        The engine.

    Returns
    -------
    scoped_session : sqlalchemy.orm.scoped_session
        The scoped session factory. Use `with session() as s:` to get a session instance.

    Notes
    -----
    Session Lifecycle:
    - Always use context managers: `with session() as s: ...`
    - The context manager automatically handles commit/rollback and closing
    - The scoped_session registry is cleaned up automatically on application shutdown
    - For manual cleanup, call `cleanup_session(session)` or let atexit handlers run

    Example
    -------
    ::

        session = create_session(engine)
        with session() as s:
            result = s.query(Model).all()
            s.commit()  # Optional - context manager handles this
    '''

    # Check if the engine is defined
    if engine is None:
        # The engine is not defined
        _ = ocerror.Error.engine_not_created(
            "The engine is not defined. Please create the engine first."
        )
        # Return None
        return None

    # Create the session in a scoped session to avoid threading problems
    # scoped_session provides thread-local session instances
    _ensure_sqlalchemy_available()
    session = scoped_session(sessionmaker(bind=engine))

    # Return the session
    return session
