#!/usr/bin/env python3

# Description
###############################################################################
'''
Logging wrapper that bridges OCDocker logging to Python's logging.

Usage:

import OCDocker.Toolbox.Logging as oclogging
'''

# Imports
###############################################################################
import inspect
import logging
import logging.handlers
import os
import re
import shutil
import sys
import time

from glob import glob
from typing import Any, Optional, Tuple, TypedDict

import OCDocker.Error as ocerror

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

class _State(TypedDict):
    configured: bool
    logger: logging.Logger
    use_rich: bool
    to_stdout: bool
    stream_handler: Optional[logging.Handler]


_STATE: _State = {
    "configured": False,
    "logger": logging.getLogger("ocdocker"),
    "use_rich": True,
    "to_stdout": True,
    "stream_handler": None,
}

_DATEFMT = "%d-%m-%Y|%H:%M:%S"
_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_RICH_TIME_STYLE = "bright_black"


# Functions
###############################################################################
## Private ##

def _build_stream_handler(to_stdout: bool, use_rich: bool) -> tuple[logging.Handler, bool]:
    '''Create the console logging handler.

    Parameters
    ----------
    to_stdout : bool
        Whether to direct output to stdout (True) or stderr (False).
    use_rich : bool
        Whether to attempt Rich handler creation.

    Returns
    -------
    tuple[logging.Handler, bool]
        The handler instance and a flag indicating whether Rich is active.
    '''

    handler: logging.Handler
    if use_rich:
        try:
            from rich.console import Console
            from rich.logging import RichHandler
            from rich.theme import Theme
            theme = Theme({"log.time": _RICH_TIME_STYLE})
            console = Console(file=sys.stdout if to_stdout else sys.stderr, theme=theme)
            kwargs: dict[str, Any] = {
                "console": console,
                "show_time": True,
                "show_level": True,
                "show_path": False,
                "rich_tracebacks": True,
                "log_time_format": _DATEFMT,
            }
            try:
                sig = inspect.signature(RichHandler.__init__)
                supported: dict[str, Any] = {k: v for k, v in kwargs.items() if k in sig.parameters}
                handler = RichHandler(**supported)
            except (ValueError, TypeError):
                handler = RichHandler(console=console)
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler, True
        except Exception:
            pass
    handler = logging.StreamHandler(sys.stdout if to_stdout else sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    return handler, False


def _default_logdir() -> str:
    '''Get the default log directory, using config if available, otherwise fallback.

    Returns
    -------
    str
        The log directory path.
    '''

    try:
        from OCDocker.Config import get_config
        config = get_config()
        if config and config.logdir:
            return config.logdir
    except (ImportError, AttributeError, RuntimeError):
        # Fallback if config not available
        pass
    base = os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir))
    return os.path.join(base, "logs")


def _ensure_configured(to_stdout: bool = True, use_rich: Optional[bool] = None) -> None:
    '''Ensure the logger has a configured console handler.

    Parameters
    ----------
    to_stdout : bool, optional
        Whether to direct output to stdout (True) or stderr (False).
    use_rich : Optional[bool], optional
        Whether to attempt Rich handler creation. If None, keep prior choice.
    '''

    logger = _STATE["logger"]
    if use_rich is None:
        use_rich = _STATE["use_rich"]

    if _STATE["configured"]:
        if _STATE["use_rich"] == use_rich and _STATE["to_stdout"] == to_stdout:
            return
        old_handler = _STATE["stream_handler"]
        if old_handler is not None and old_handler in logger.handlers:
            logger.removeHandler(old_handler)

    logger.setLevel(logging.DEBUG)
    handler, rich_ok = _build_stream_handler(to_stdout, use_rich)
    logger.addHandler(handler)
    _STATE["configured"] = True
    _STATE["use_rich"] = rich_ok
    _STATE["to_stdout"] = to_stdout
    _STATE["stream_handler"] = handler


def _log_limits() -> Tuple[int, int, int]:
    '''Get the log size limits from the config (log_max_mb, log_backups, log_archives).

    Returns
    -------
    Tuple[int, int, int]
        Bytes at which a log rolls over (0 disables it), rolled-over copies kept per log and
        past run reports kept per log in read_log_past (0 keeps all).
    '''

    try:
        from OCDocker.Config import get_config
        config = get_config()
        return int(float(config.log_max_mb) * 1024 * 1024), int(config.log_backups), int(config.log_archives)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        # Without a usable config, keep every log as it is
        return 0, 0, 0


## Public ##

def backup_log(logname: str) -> None:
    '''Backup the current log under <logdir>/read_log_past.

    Parameters
    ----------
    logname : str
        The base name of the log file (without .log extension).
    '''

    logdir = _default_logdir()
    src = os.path.join(logdir, f"{logname}.log")
    if os.path.isfile(src):
        dst_dir = os.path.join(logdir, "read_log_past")
        try:
            os.makedirs(dst_dir, exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors when creating archive directory
            return
        dst = os.path.join(dst_dir, f"{logname}_{time.strftime('%d%m%Y-%H%M%S')}.log")
        # Two backups within the same second would overwrite each other
        copy = 1
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f"{logname}_{time.strftime('%d%m%Y-%H%M%S')}_{copy}.log")
            copy += 1
        try:
            os.rename(src, dst)
        except (OSError, PermissionError):
            # Ignore errors when archiving logs
            return

        # Keep only the newest archived reports of this log
        archives = _log_limits()[2]
        if archives > 0:
            pattern = re.compile(rf"^{re.escape(logname)}_\d{{8}}-\d{{6}}(_\d+)?\.log$")
            past = sorted(
                (os.path.join(dst_dir, name) for name in os.listdir(dst_dir) if pattern.match(name)),
                key=os.path.getmtime,
            )
            for old in past[:-archives]:
                try:
                    os.remove(old)
                except OSError:
                    # Ignore errors when removing old archives
                    pass


def clear_past_logs() -> None:
    '''Clear past logs under the default log directory/past folders.'''

    logdir = _default_logdir()
    for past in [d for d in glob(f"{logdir}/*") if os.path.isdir(d)]:
        if past.endswith("past"):
            try:
                shutil.rmtree(past)
            except (OSError, PermissionError):
                # Ignore errors when cleaning old logs
                pass


def configure(
        level: Optional[ocerror.ReportLevel] = None,
        log_file: Optional[str] = None,
        to_stdout: bool = True,
        use_rich: Optional[bool] = None,
    ) -> None:
    '''Configure the root ocdocker logger.

    Parameters
    ----------
    level : Optional[ocerror.ReportLevel], optional
        The report level to set. If None, uses current ocerror level.
    log_file : Optional[str], optional
        Path to a log file to add as a handler. If None, no file logging is
        added.
    to_stdout : bool, optional
        Whether to direct console output to stdout (True) or stderr (False).
    use_rich : Optional[bool], optional
        Whether to attempt Rich handler creation. If None, keep prior choice.
    '''

    if use_rich is None:
        use_rich = _STATE["use_rich"]
    _ensure_configured(to_stdout=to_stdout, use_rich=use_rich)
    if level is None:
        level = ocerror.Error.get_output_level()
    set_level_from_report(level)
    if log_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        except (OSError, PermissionError):
            # Ignore errors if directory already exists or permission denied
            pass
        # Attach each log file once, otherwise configuring again writes every message again
        log_path = os.path.abspath(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_path for h in _STATE["logger"].handlers):
            max_bytes, backups, _ = _log_limits()
            fh = logging.handlers.RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backups)
            fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
            _STATE["logger"].addHandler(fh)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    '''Return the configured logger (or a child).

    Parameters
    ----------
    name : Optional[str], optional
        Name of the child logger to return. If None, returns the root logger.

    Returns
    -------
    logging.Logger
        The requested logger instance.
    '''

    _ensure_configured(to_stdout=_STATE["to_stdout"], use_rich=None)

    return _STATE["logger"] if not name else _STATE["logger"].getChild(name)


def is_rich_enabled() -> bool:
    '''Check whether Rich is the active console handler.

    Returns
    -------
    bool
        True when Rich is enabled for console logging.
    '''
    return _STATE["use_rich"]


def rotate_log(logfile: str) -> None:
    '''Roll a log file over to <logfile>.1 (shifting older copies) once it reaches log_max_mb.

    Parameters
    ----------
    logfile : str
        The log file about to be appended to.
    '''

    max_bytes, backups, _ = _log_limits()
    if max_bytes <= 0:
        return
    try:
        if not os.path.isfile(logfile) or os.path.getsize(logfile) < max_bytes:
            return
        if backups <= 0:
            os.remove(logfile)
            return
        for index in range(backups - 1, 0, -1):
            if os.path.isfile(f"{logfile}.{index}"):
                os.replace(f"{logfile}.{index}", f"{logfile}.{index + 1}")
        os.replace(logfile, f"{logfile}.1")
    except OSError:
        # Ignore rotation errors (e.g. another process rotated it first)
        pass


def set_level_from_report(level: ocerror.ReportLevel) -> None:
    '''Map ocerror.ReportLevel to logging level and set it.

    Parameters
    ----------
    level : ocerror.ReportLevel
        The report level to set.
    '''

    lvl_map = {
        ocerror.ReportLevel.NONE: logging.CRITICAL + 10,
        ocerror.ReportLevel.ERROR: logging.ERROR,
        ocerror.ReportLevel.WARNING: logging.WARNING,
        ocerror.ReportLevel.INFO: logging.INFO,
        ocerror.ReportLevel.SUCCESS: logging.INFO,
        ocerror.ReportLevel.DEBUG: logging.DEBUG,
    }
    py_level = lvl_map.get(level, logging.INFO)
    logger = _STATE["logger"]
    logger.setLevel(py_level)
    for h in logger.handlers:
        try:
            h.setLevel(py_level)
        except AttributeError:
            # Ignore if handler doesn't support setLevel
            pass
