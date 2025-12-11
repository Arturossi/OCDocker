#!/usr/bin/env python3

"""
Logging wrapper for OCDocker.

Provides a bridge to Python's logging so the project can centrally control
formatting, levels, and handlers while keeping legacy print helpers working.

Usage:
  import OCDocker.Toolbox.Logging as oclogging
  log = oclogging.get_logger()
  log.info("Hello")

Configure:
  oclogging.configure(level=ocerror.ReportLevel.INFO, log_file="path/to/file.log")
  oclogging.set_level_from_report(ocerror.ReportLevel.DEBUG)
"""

import logging
import os
import sys
import shutil
import time
from typing import Optional
from glob import glob

import OCDocker.Error as ocerror
import OCDocker.Toolbox.FilesFolders as ocff

_STATE = {
    "configured": False,
    "logger": logging.getLogger("ocdocker"),
}

_DATEFMT = "%d-%m-%Y|%H:%M:%S"
_FMT = "[%(asctime)s] %(levelname)s: %(message)s"


def _default_logdir() -> str:
    """Get the default log directory, using config if available, otherwise fallback."""
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


def _ensure_configured(to_stdout: bool = True) -> None:
    if _STATE["configured"]:
        return
    logger = _STATE["logger"]
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout if to_stdout else sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logger.addHandler(handler)
    _STATE["configured"] = True


def set_level_from_report(level: ocerror.ReportLevel) -> None:
    """Map ocerror.ReportLevel to logging level and set it."""
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


def configure(level: Optional[ocerror.ReportLevel] = None, log_file: Optional[str] = None, to_stdout: bool = True) -> None:
    """Configure the root ocdocker logger.

    - level: ocerror.ReportLevel to apply (defaults to ocerror.Error.get_output_level())
    - log_file: optional file to write logs to
    - to_stdout: stream to stdout (default) instead of stderr
    """
    _ensure_configured(to_stdout=to_stdout)
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
    """Return the configured logger (or a child)."""
    _ensure_configured()
    return _STATE["logger"] if not name else _STATE["logger"].getChild(name)


def clear_past_logs() -> None:
    """Clear past logs under the default log directory/past folders."""
    logdir = _default_logdir()
    for past in [d for d in glob(f"{logdir}/*") if os.path.isdir(d)]:
        if past.endswith("past"):
            shutil.rmtree(past)


def backup_log(logname: str) -> None:
    """Backup the current log under <logdir>/read_log_past."""
    logdir = _default_logdir()
    src = os.path.join(logdir, f"{logname}.log")
    if os.path.isfile(src):
        dst_dir = os.path.join(logdir, "read_log_past")
        if not os.path.isdir(dst_dir):
            ocff.safe_create_dir(dst_dir)
        dst = os.path.join(dst_dir, f"{logname}_{time.strftime('%d%m%Y-%H%M%S')}.log")
        os.rename(src, dst)
