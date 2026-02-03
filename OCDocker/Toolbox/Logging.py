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
import os
import shutil
import sys
import time

from glob import glob
from typing import Optional

import OCDocker.Error as ocerror
import OCDocker.Toolbox.FilesFolders as ocff

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

_STATE = {
    "configured": False,
    "logger": logging.getLogger("ocdocker"),
    "use_rich": False,
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

    if use_rich:
        try:
            from rich.console import Console  # type: ignore
            from rich.logging import RichHandler  # type: ignore
            from rich.theme import Theme  # type: ignore
            theme = Theme({"log.time": _RICH_TIME_STYLE})
            console = Console(file=sys.stdout if to_stdout else sys.stderr, theme=theme)
            kwargs = {
                "console": console,
                "show_time": True,
                "show_level": True,
                "show_path": False,
                "rich_tracebacks": True,
                "log_time_format": _DATEFMT,
            }
            try:
                sig = inspect.signature(RichHandler.__init__)
                supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
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
        use_rich = _STATE.get("use_rich", False)

    if _STATE["configured"]:
        if _STATE.get("use_rich") == use_rich and _STATE.get("to_stdout") == to_stdout:
            return
        old_handler = _STATE.get("stream_handler")
        if old_handler in logger.handlers:
            logger.removeHandler(old_handler)

    logger.setLevel(logging.DEBUG)
    handler, rich_ok = _build_stream_handler(to_stdout, use_rich)
    logger.addHandler(handler)
    _STATE["configured"] = True
    _STATE["use_rich"] = rich_ok
    _STATE["to_stdout"] = to_stdout
    _STATE["stream_handler"] = handler


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
        if not os.path.isdir(dst_dir):
            ocff.safe_create_dir(dst_dir)
        dst = os.path.join(dst_dir, f"{logname}_{time.strftime('%d%m%Y-%H%M%S')}.log")
        try:
            os.rename(src, dst)
        except (OSError, PermissionError):
            # Ignore errors when archiving logs
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
        use_rich = _STATE.get("use_rich", False)
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
        fh = logging.FileHandler(log_file)
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

    _ensure_configured(to_stdout=_STATE.get("to_stdout", True), use_rich=None)

    return _STATE["logger"] if not name else _STATE["logger"].getChild(name)


def is_rich_enabled() -> bool:
    '''Check whether Rich is the active console handler.

    Returns
    -------
    bool
        True when Rich is enabled for console logging.
    '''
    return bool(_STATE.get("use_rich"))


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
