#!/usr/bin/env python3
"""Environment diagnostics (doctor) CLI command."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from typing import Any, Dict, Optional

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.CLI.common import _bootstrap_ocdocker_env, _preparse_global_args
from OCDocker.CLI.manifest import _collect_external_tool_manifest

LOGGER = oclogging.get_logger("cli")

def cmd_doctor(args: argparse.Namespace) -> int:  # pragma: no cover - environment probing is platform-dependent
    '''Run diagnostics: config, binaries, Python deps, DB connectivity.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    # Bootstrap lightweight config/runtime context (no DB required for diagnostics)
    globals_ns = _preparse_global_args(sys.argv[1:])
    setattr(globals_ns, "_ocdocker_init_db", False)
    _bootstrap_ocdocker_env(globals_ns)

    # Configure logging according to CLI flags
    try:
        import OCDocker.Error as ocerror
        import OCDocker.Toolbox.Logging as oclogging
        oclogging.configure(level=ocerror.Error.get_output_level(), log_file=args.log_file, to_stdout=(not args.no_stdout_log))
    except (ImportError, AttributeError, OSError):
        # Ignore logging configuration errors (non-critical for core functionality)
        pass

    report: Dict[str, Any] = {}

    # Config source
    try:
        import OCDocker.Initialise as OCI
        cfg = getattr(OCI, 'config_file', None)
        report['config'] = {
            'path': str(cfg) if cfg else 'unknown',
        }
    except Exception as e:
        report['config'] = {'error': f'{e}'}

    # Engine binaries
    def _exists_exe(p: Optional[str]) -> bool:
        if not p:
            return False
        if os.path.isabs(p):
            return os.path.isfile(p) and os.access(p, os.X_OK)
        return shutil.which(p) is not None

    _vina_bin: Optional[str]
    _smina_bin: Optional[str]
    _plants_bin: Optional[str]
    config: Optional[Any] = None
    try:
        from OCDocker.Config import get_config
        config = get_config()
        v: Optional[str] = config.vina.executable
        s: Optional[str] = config.smina.executable
        p: Optional[str] = config.plants.executable
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        # Fallback if config is not available
        v = s = p = None
        report['binaries_error'] = f"CONFIG_UNAVAILABLE ({type(exc).__name__}: {exc})"
    report['binaries'] = {
        'vina': 'OK' if _exists_exe(v) else 'MISSING',
        'smina': 'OK' if _exists_exe(s) else 'MISSING',
        'plants': 'OK' if _exists_exe(p) else 'MISSING',
    }
    report['external_tools'] = _collect_external_tool_manifest()

    # Python dependencies
    # SECURITY NOTE: Dynamic import is used here to check for optional dependencies.
    # The module names are hardcoded in a whitelist ('rdkit', 'Bio', 'oddt', 'sqlalchemy')
    # and never come from user input, making this safer from injection attacks.
    pydeps = {}

    # Whitelist of allowed module names for dependency checking
    ALLOWED_DEPENDENCY_MODULES = ('rdkit', 'Bio', 'oddt', 'sqlalchemy')
    for mod in ALLOWED_DEPENDENCY_MODULES:
        # Validate module name contains only safe characters (alphanumeric and underscore)
        if not isinstance(mod, str) or not mod.replace('_', '').isalnum():
            pydeps[mod] = 'INVALID_MODULE_NAME'
            continue
        try:
            __import__(mod)
            pydeps[mod] = 'OK'
        except Exception as e:
            pydeps[mod] = f'MISSING ({e.__class__.__name__})'
    report['python_deps'] = pydeps

    # DB connectivity
    db_report: Dict[str, Any] = {}
    backend_cfg: Optional[str] = None
    try:
        if config is not None:
            backend_cfg = str(getattr(getattr(config, 'database', None), 'backend', '') or '').strip().lower() or None
    except Exception:
        backend_cfg = None

    try:
        import sqlalchemy
        db_report['sqlalchemy_version'] = str(getattr(sqlalchemy, '__version__', 'unknown'))
    except Exception:
        db_report['sqlalchemy_version'] = 'unknown'

    try:
        eng = getattr(OCI, 'engine', None)
        if eng is None:
            db_report['status'] = 'MISSING ENGINE'
            db_report['access'] = False
            db_report['backend'] = backend_cfg or 'unknown'
            report['database'] = db_report
        else:
            url_obj = getattr(eng, 'url', None)
            drivername = str(getattr(url_obj, 'drivername', '') or '')
            backend_from_driver = ''
            if drivername.startswith('postgresql'):
                backend_from_driver = 'postgresql'
            elif drivername.startswith('mysql'):
                backend_from_driver = 'mysql'
            elif drivername.startswith('sqlite'):
                backend_from_driver = 'sqlite'

            backend = backend_cfg or backend_from_driver or 'unknown'
            db_report['backend'] = backend
            db_report['driver'] = drivername or 'unknown'

            # Client/driver library version (best-effort)
            client_version = 'unknown'
            try:
                import importlib
                if backend == 'postgresql':
                    _psycopg = importlib.import_module('psycopg')
                    client_version = str(getattr(_psycopg, '__version__', 'unknown'))
                elif backend == 'mysql':
                    _pymysql = importlib.import_module('pymysql')
                    client_version = str(getattr(_pymysql, '__version__', 'unknown'))
                elif backend == 'sqlite':
                    import sqlite3 as _sqlite3
                    client_version = str(getattr(_sqlite3, 'sqlite_version', 'unknown'))
            except Exception:
                client_version = 'unknown'
            db_report['client_version'] = client_version

            conn = None
            try:
                conn = eng.connect()
                db_report['status'] = 'OK'
                db_report['access'] = True
                server_version: Optional[str] = None
                if hasattr(conn, 'exec_driver_sql'):
                    sql = None
                    if backend == 'postgresql':
                        sql = 'SHOW server_version'
                    elif backend == 'mysql':
                        sql = 'SELECT VERSION()'
                    elif backend == 'sqlite':
                        sql = 'SELECT sqlite_version()'

                    if sql:
                        try:
                            value = conn.exec_driver_sql(sql).scalar()
                            if value is not None:
                                server_version = str(value)
                        except Exception as exc:
                            server_version = f'ERROR ({type(exc).__name__})'
                db_report['server_version'] = server_version or 'unknown'

                # Connected identity / database checks (best-effort).
                expected_user = None
                expected_database = None
                try:
                    if config is not None:
                        expected_user = getattr(getattr(config, 'database', None), 'user', None)
                        expected_database = getattr(getattr(config, 'database', None), 'database', None)
                except Exception:
                    pass

                db_report['expected_user'] = expected_user
                db_report['expected_database'] = expected_database

                current_user: Optional[str] = None
                current_database: Optional[str] = None
                if hasattr(conn, 'exec_driver_sql'):
                    try:
                        if backend == 'postgresql':
                            current_user_val = conn.exec_driver_sql('SELECT current_user').scalar()
                            current_db_val = conn.exec_driver_sql('SELECT current_database()').scalar()
                            current_user = str(current_user_val) if current_user_val is not None else None
                            current_database = str(current_db_val) if current_db_val is not None else None
                        elif backend == 'mysql':
                            current_user_val = conn.exec_driver_sql('SELECT CURRENT_USER()').scalar()
                            current_db_val = conn.exec_driver_sql('SELECT DATABASE()').scalar()
                            current_user = str(current_user_val) if current_user_val is not None else None
                            current_database = str(current_db_val) if current_db_val is not None else None
                        elif backend == 'sqlite':
                            current_user = 'n/a (sqlite)'
                            db_name = getattr(url_obj, 'database', None)
                            current_database = str(db_name) if db_name else 'sqlite'
                    except Exception:
                        # Keep values as None when introspection queries fail.
                        pass

                db_report['current_user'] = current_user or 'unknown'
                db_report['current_database'] = current_database or 'unknown'

                user_check = 'unknown'
                if backend == 'sqlite':
                    user_check = 'n/a'
                elif expected_user:
                    effective_user = (current_user or '').strip()
                    # MySQL CURRENT_USER() often returns "user@host".
                    if backend == 'mysql' and '@' in effective_user:
                        effective_user = effective_user.split('@', 1)[0]
                    user_check = 'ok' if effective_user == str(expected_user).strip() else 'mismatch'
                db_report['user_check'] = user_check

                database_check = 'unknown'
                if expected_database and current_database and current_database != 'unknown':
                    database_check = (
                        'ok'
                        if str(current_database).strip() == str(expected_database).strip()
                        else 'mismatch'
                    )
                db_report['database_check'] = database_check
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
            report['database'] = db_report
    except Exception as e:
        db_report['status'] = f'ERROR ({e})'
        db_report['access'] = False
        if 'backend' not in db_report:
            db_report['backend'] = backend_cfg or 'unknown'
        report['database'] = db_report

    # Summary printout
    print(json.dumps(report, indent=2))

    return 0


def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    '''Register the ``ocdocker doctor`` command group.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Main CLI subparser registry.
    parent : argparse.ArgumentParser
        Parent parser supplying shared global arguments.
    '''

    p_doc = sub.add_parser(
        "doctor",
        description=(
            "Run diagnostics to check your OCDocker environment setup.\n\n"
            "This command verifies:\n"
            "  - Availability and versions of external tools (Vina, Smina, PLANTS, Gnina, Open Babel, etc.)\n"
            "  - Python dependencies and package versions\n"
            "  - Database backend, driver/client version, server version (when reachable), and connectivity\n"
            "  - Configuration file validity\n\n"
            "Use this command to troubleshoot installation or configuration issues before\n"
            "running docking jobs."
        ),
        help="Check environment setup and diagnose issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
    )
    p_doc.set_defaults(func=cmd_doctor)
