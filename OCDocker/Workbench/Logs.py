#!/usr/bin/env python3

# Description
###############################################################################
'''
Bounded read-only log preview helpers for Workbench run manifests.
'''

# Imports
###############################################################################
from __future__ import annotations

from pathlib import Path

from OCDocker.Workbench.IO import read_run_manifest
from OCDocker.Workbench.Models import RunLogFilePreview
from OCDocker.Workbench.Models import RunLogPreview
from OCDocker.Workbench.Registry import RUN_MANIFEST_FILENAMES

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################

DEFAULT_LOG_LINE_LIMIT = 80
DEFAULT_LOG_BYTE_LIMIT = 65536

# Functions
###############################################################################
## Private ##


def _direct_manifest_paths(directory: Path) -> tuple[Path, ...]:
    '''Return direct child run manifest paths in deterministic order.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to inspect.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Matching direct child run manifest paths.
    '''

    return tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.name.lower() in RUN_MANIFEST_FILENAMES
            ),
            key=lambda path: path.name,
        )
    )


def _resolve_run_manifest_path(target: str | Path) -> Path:
    '''Resolve a run manifest path from a manifest file or bundle directory.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or bundle directory.

    Returns
    -------
    pathlib.Path
        Resolved run manifest path.
    '''

    target_path = Path(target)
    if target_path.is_file():
        return target_path
    if not target_path.exists():
        raise FileNotFoundError(f"Workbench log target does not exist: {target_path}")
    if not target_path.is_dir():
        raise ValueError(
            f"Workbench log target is not a file or directory: {target_path}"
        )

    candidates = _direct_manifest_paths(target_path)
    if not candidates:
        raise FileNotFoundError(
            f"No Workbench run manifest found directly in directory: {target_path}"
        )
    return candidates[0]


def _resolve_path(base_path: Path, path: Path) -> Path:
    '''Resolve a manifest-relative path without modifying the filesystem.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative paths.
    path : pathlib.Path
        Path to resolve.

    Returns
    -------
    pathlib.Path
        Absolute artifact path or path relative to the manifest directory.
    '''

    if path.is_absolute():
        return path
    return base_path / path


def _validate_limits(lines: int, max_bytes: int) -> None:
    '''Validate line and byte limits before reading log files.

    Parameters
    ----------
    lines : int
        Maximum returned lines per file.
    max_bytes : int
        Maximum bytes read per file.
    '''

    if lines < 1:
        raise ValueError("lines must be greater than or equal to one.")
    if max_bytes < 1:
        raise ValueError("max_bytes must be greater than or equal to one.")


def _decode_tail(path: Path, *, lines: int, max_bytes: int, encoding: str) -> tuple:
    '''Read a bounded tail preview from one log file.

    Parameters
    ----------
    path : pathlib.Path
        Log file path.
    lines : int
        Maximum returned lines.
    max_bytes : int
        Maximum bytes read from the end of the file.
    encoding : str
        Text encoding used to decode log bytes.

    Returns
    -------
    tuple
        Returned lines, tail text, size bytes, read bytes, and truncation flag.
    '''

    size_bytes = path.stat().st_size
    read_bytes = min(size_bytes, max_bytes)
    with path.open("rb") as handle:
        handle.seek(max(0, size_bytes - read_bytes))
        data = handle.read(read_bytes)
    text = data.decode(encoding, errors="replace")
    decoded_lines = tuple(text.splitlines())
    returned_lines = decoded_lines[-lines:]
    truncated = size_bytes > read_bytes or len(decoded_lines) > len(returned_lines)
    return returned_lines, "\n".join(returned_lines), size_bytes, read_bytes, truncated


## Public ##


def build_log_file_preview(
    base_path: Path,
    path: Path,
    *,
    lines: int = DEFAULT_LOG_LINE_LIMIT,
    max_bytes: int = DEFAULT_LOG_BYTE_LIMIT,
    encoding: str = "utf-8",
) -> RunLogFilePreview:
    '''Build a bounded preview for one log file, absolute or manifest-relative.

    Parameters
    ----------
    base_path : pathlib.Path
        Directory used for relative log paths.
    path : pathlib.Path
        Declared log path.
    lines : int
        Maximum returned lines.
    max_bytes : int
        Maximum bytes read from the end of the file.
    encoding : str
        Text encoding used to decode log bytes.

    Returns
    -------
    RunLogFilePreview
        Read-only log file preview.
    '''

    resolved = _resolve_path(base_path, path)
    exists = resolved.exists()
    is_file = resolved.is_file()
    is_dir = resolved.is_dir()
    if not exists:
        return RunLogFilePreview(
            path=resolved,
            exists=False,
            is_file=False,
            is_dir=False,
            name=resolved.name,
            role="log",
            encoding=encoding,
            error="Log file does not exist.",
        )
    if not is_file:
        return RunLogFilePreview(
            path=resolved,
            exists=True,
            is_file=is_file,
            is_dir=is_dir,
            name=resolved.name,
            role="log",
            encoding=encoding,
            error="Log path is not a file.",
        )

    try:
        returned_lines, text, size_bytes, read_bytes, truncated = _decode_tail(
            resolved,
            lines=lines,
            max_bytes=max_bytes,
            encoding=encoding,
        )
    except OSError as exc:
        return RunLogFilePreview(
            path=resolved,
            exists=True,
            is_file=True,
            is_dir=False,
            name=resolved.name,
            role="log",
            encoding=encoding,
            error=str(exc),
        )

    return RunLogFilePreview(
        path=resolved,
        exists=True,
        is_file=True,
        is_dir=False,
        name=resolved.name,
        role="log",
        encoding=encoding,
        size_bytes=size_bytes,
        read_bytes=read_bytes,
        returned_line_count=len(returned_lines),
        truncated=truncated,
        lines=returned_lines,
        text=text,
    )


def preview_run_logs(
    target: str | Path,
    *,
    lines: int = DEFAULT_LOG_LINE_LIMIT,
    max_bytes: int = DEFAULT_LOG_BYTE_LIMIT,
    encoding: str = "utf-8",
) -> RunLogPreview:
    '''Preview declared Workbench run logs without streaming or controlling runs.

    Parameters
    ----------
    target : str or pathlib.Path
        Run manifest file or prepared bundle directory.
    lines : int
        Maximum returned lines per declared log file.
    max_bytes : int
        Maximum bytes read from the end of each log file.
    encoding : str
        Text encoding used to decode log bytes.

    Returns
    -------
    RunLogPreview
        Read-only bounded log preview.
    '''

    _validate_limits(lines, max_bytes)
    manifest_path = _resolve_run_manifest_path(target)
    manifest = read_run_manifest(manifest_path)
    base_path = manifest_path.parent
    logs = tuple(
        build_log_file_preview(
            base_path,
            path,
            lines=lines,
            max_bytes=max_bytes,
            encoding=encoding,
        )
        for path in manifest.log_files
    )
    return RunLogPreview(
        manifest_path=manifest_path,
        run_id=manifest.run_id,
        spec_type=manifest.spec_type,
        name=manifest.name,
        status=manifest.status,
        line_limit=lines,
        byte_limit=max_bytes,
        encoding=encoding,
        logs=logs,
    )


__all__ = [
    "DEFAULT_LOG_BYTE_LIMIT",
    "DEFAULT_LOG_LINE_LIMIT",
    "build_log_file_preview",
    "preview_run_logs",
]
