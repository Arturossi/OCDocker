#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to run commands in the OS.

Usage:

import OCDocker.Toolbox.Running as ocrun
'''

# Imports
###############################################################################
import os
import shlex
import shutil
import subprocess
import time

from typing import Dict, List, Optional, Tuple, Union

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Printing as ocprint


# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################

class SubprocessError(RuntimeError):
    """Exception raised when a subprocess fails and OCDOCKER_RAISE_SUBPROCESS is enabled.

    Parameters
    ----------
    message : str
        Summary of the failure.
    cmd : List[str]
        Command that was executed.
    cwd : str
        Working directory used for the command.
    returncode : int
        Exit code from the subprocess.
    stderr : str
        Captured stderr output.
    stdout_log : str
        Path to the stdout log file.
    report_path : str
        Path to the failure report file.
    """

    def __init__(
        self,
        message: str,
        cmd: List[str],
        cwd: str,
        returncode: int,
        stderr: str,
        stdout_log: str,
        report_path: str,
    ) -> None:
        super().__init__(message)
        self.cmd = cmd
        self.cwd = cwd
        self.returncode = returncode
        self.stderr = stderr
        self.stdout_log = stdout_log
        self.report_path = report_path

# Functions
###############################################################################
## Private ##

def _env_flag(name: str) -> bool:
    '''Check if an environment flag is enabled.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    bool
        True if the value is a truthy flag, False otherwise.
    '''
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    '''Read an environment variable as a positive integer.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Default value if unset or invalid.

    Returns
    -------
    int
        Parsed integer value or the default.
    '''
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except (ValueError, TypeError):
        return default


def _env_snapshot(keys: List[str]) -> Dict[str, str]:
    '''Capture selected environment variables.

    Parameters
    ----------
    keys : List[str]
        Environment variable names to capture.

    Returns
    -------
    Dict[str, str]
        Mapping of keys to values for set variables.
    '''
    snapshot: Dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value:
            snapshot[key] = value
    return snapshot


def _failure_report_path() -> str:
    '''Resolve the path for subprocess failure reports.

    Returns
    -------
    str
        Path to the report file, or empty string if unavailable.
    '''
    try:
        from OCDocker.Config import get_config
        config = get_config()
        base_dir = config.logdir if config and config.logdir else ""
    except Exception:
        base_dir = ""
    if not base_dir:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(ocerror.__file__), os.pardir, "logs"))
    try:
        os.makedirs(base_dir, exist_ok=True)
    except OSError:
        return ""
    return os.path.join(base_dir, "subprocess_failures.log")


def _format_cmd(cmd: List[str]) -> str:
    '''Format a command list into a shell-friendly string.

    Parameters
    ----------
    cmd : List[str]
        Command and arguments.

    Returns
    -------
    str
        Shell-escaped command string.
    '''
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _tail_file(path: str, max_lines: int, max_bytes: int = 65536) -> str:
    '''Return the last N lines of a file.

    Parameters
    ----------
    path : str
        File path.
    max_lines : int
        Maximum number of lines to return.
    max_bytes : int, optional
        Maximum bytes to read from the file tail.

    Returns
    -------
    str
        Tail of the file content, or empty string if unavailable.
    '''
    if not path or path == os.devnull or not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= 0:
                return ""
            read_size = min(size, max_bytes)
            handle.seek(-read_size, os.SEEK_END)
            data = handle.read()
        return _tail_text(data.decode("utf-8", errors="replace"), max_lines)
    except OSError:
        return ""


def _tail_text(text: str, max_lines: int) -> str:
    '''Return the last N lines of a text string.

    Parameters
    ----------
    text : str
        Text to truncate.
    max_lines : int
        Maximum number of lines to return.

    Returns
    -------
    str
        Tail of the input text.
    '''
    if not text:
        return ""
    lines = text.splitlines()
    if max_lines <= 0 or len(lines) <= max_lines:
        return text.strip()
    return "\n".join(lines[-max_lines:])


def _write_failure_report(report: str) -> str:
    '''Append a failure report to the report file.

    Parameters
    ----------
    report : str
        Report content to append.

    Returns
    -------
    str
        Path to the report file, or empty string if unavailable.
    '''
    path = _failure_report_path()
    if not path:
        return ""
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(report)
            if not report.endswith("\n"):
                handle.write("\n")
    except OSError:
        return ""
    return path


## Public ##

def is_tool_available(exe: str) -> bool:
    '''Check if a tool executable is available.
    
    Parameters
    ----------
    exe : str
        Path to the executable (can be absolute or command name)
        
    Returns
    -------
    bool
        True if the executable is available, False otherwise
    '''
    
    if not exe:
        return False
    return (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)


def run(cmd: List[str], logFile: str = "", cwd: str = "", timeout: Optional[int] = None) -> Union[int, Tuple[int, str]]:
    '''Run the given command (generic).

    Parameters
    ----------
    cmd : List[str]
        The command to be run.
    logFile : str, optional
        The file where the output will be saved. Default is "".
    cwd : str, optional
        The current working directory. Default is "".

    Environment
    -----------
    OCDOCKER_SUBPROCESS_TAIL_LINES : int
        Number of lines to include from stderr/stdout tail (default: 20).
    OCDOCKER_DEBUG_SUBPROCESS : bool
        If enabled, include stdout tail and environment snapshot in the failure report.
    OCDOCKER_RAISE_SUBPROCESS : bool
        If enabled, raise SubprocessError on failure instead of returning an error code.

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
    '''

    if not cmd:
        return ocerror.Error.not_set(message = f"The variable cmd is not set or is an empty list!", level = ocerror.ReportLevel.ERROR)

    if not isinstance(cmd, list):
        return ocerror.Error.wrong_type(message = f"The argument cmd has to be a list! Found '{type(cmd)}' instead...", level = ocerror.ReportLevel.ERROR)

    # Print verboosity
    ocprint.printv(f"Running the command '{' '.join(cmd)}'.")

    if logFile == "":
        ocprint.printv(f"No log will be made")
        logFile = os.devnull
    else:
        ocprint.printv(f"Logging into '{logFile}'")

    # Resolve timeout from param or environment variable
    if timeout is None:
        try:
            timeout_env = int(os.getenv("OCDOCKER_TIMEOUT", "0"))
            timeout = timeout_env if timeout_env > 0 else None
        except (ValueError, TypeError):
            # Ignore invalid timeout values
            timeout = None

    # Validate executable availability (avoid PermissionError on empty string)
    exe = str(cmd[0])
    if not exe:
        return ocerror.Error.subprocess(message = "Executable not set (empty). Check your configuration.", level=ocerror.ReportLevel.ERROR)
    if os.path.isabs(exe):
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            return ocerror.Error.subprocess(message = f"Executable not found or not executable: '{exe}'", level=ocerror.ReportLevel.ERROR)
    else:
        if shutil.which(exe) is None:
            return ocerror.Error.subprocess(message = f"Executable not found on PATH: '{exe}'", level=ocerror.ReportLevel.ERROR)

    try:
        if cwd == "":
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, timeout=timeout)
        else:
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout=outfile, cwd=cwd, stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError as e:
        return ocerror.Error.subprocess(message = f"Command not found when executing '{' '.join(cmd)}': {e}", level=ocerror.ReportLevel.ERROR)
    except subprocess.TimeoutExpired as e:
        return ocerror.Error.subprocess(message = f"Timeout expired after {timeout}s for command '{' '.join(cmd)}'", level=ocerror.ReportLevel.ERROR)
    except Exception as e:
        return ocerror.Error.subprocess(message = f"Found a problem while executing the command '{' '.join(cmd)}': {e}", level=ocerror.ReportLevel.ERROR)

    # If the command has not been executed successfully
    if proc.returncode != 0:
        cmd_str = _format_cmd(cmd)
        tail_lines = _env_int("OCDOCKER_SUBPROCESS_TAIL_LINES", 20)
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        stderr_tail = _tail_text(stderr_text, tail_lines)
        stdout_tail = _tail_file(logFile, tail_lines)
        cwd_display = os.path.abspath(cwd) if cwd else os.getcwd()

        summary_lines = [
            f"Subprocess failed (exit code {proc.returncode}).",
            f"Command: {cmd_str}",
            f"CWD: {cwd_display}",
            f"Stdout log: {logFile if logFile != os.devnull else 'disabled'}",
        ]
        if stderr_tail:
            summary_lines.append(f"Stderr tail (last {tail_lines} lines):\n{stderr_tail}")
        else:
            summary_lines.append("Stderr tail: <empty>")
        if _env_flag("OCDOCKER_DEBUG_SUBPROCESS") and stdout_tail:
            summary_lines.append(f"Stdout tail (last {tail_lines} lines):\n{stdout_tail}")

        report_lines = [
            f"[{time.strftime('%d-%m-%Y|%H:%M:%S')}] Subprocess failure",
            f"Command: {cmd_str}",
            f"CWD: {cwd_display}",
            f"Exit code: {proc.returncode}",
            f"Stdout log: {logFile if logFile != os.devnull else 'disabled'}",
        ]
        if stdout_tail:
            report_lines.append(f"Stdout tail (last {tail_lines} lines):")
            report_lines.append(stdout_tail)
        else:
            report_lines.append("Stdout tail: <empty or unavailable>")
        report_lines.append("Stderr:")
        report_lines.append(stderr_text.strip() or "<empty>")
        if _env_flag("OCDOCKER_DEBUG_SUBPROCESS"):
            env_keys = [
                "PATH",
                "LD_LIBRARY_PATH",
                "DYLD_LIBRARY_PATH",
                "PYTHONPATH",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "VIRTUAL_ENV",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "CUDA_PATH",
                "OCDOCKER_CONFIG",
                "OCDOCKER_TIMEOUT",
            ]
            env_snapshot = _env_snapshot(env_keys)
            if env_snapshot:
                report_lines.append("Environment snapshot:")
                for key, value in env_snapshot.items():
                    report_lines.append(f"{key}={value}")
        report_lines.append("-" * 80)
        report_path = _write_failure_report("\n".join(report_lines))
        if report_path:
            summary_lines.append(f"Failure report: {report_path}")

        summary = "\n".join(summary_lines)
        if _env_flag("OCDOCKER_RAISE_SUBPROCESS"):
            raise SubprocessError(
                summary,
                cmd=cmd,
                cwd=cwd_display,
                returncode=proc.returncode,
                stderr=stderr_text,
                stdout_log=logFile,
                report_path=report_path,
            )
        return ocerror.Error.subprocess(message=summary, level=ocerror.ReportLevel.ERROR), stderr_text
    return ocerror.Error.ok()


### Special functions
