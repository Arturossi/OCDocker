#!/usr/bin/env python3
"""Reproducibility manifest generation and manifest/version CLI commands."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from OCDocker.CLI.common import _bootstrap_ocdocker_env, _preparse_global_args

def _extract_executable_token(command: Optional[str]) -> Optional[str]:
    '''Extract the executable token from a command string.

    Parameters
    ----------
    command : Optional[str]
        Command string or executable path.

    Returns
    -------
    Optional[str]
        The executable token, or None when unavailable.
    '''

    if not command:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(command))).strip()
    if not expanded:
        return None
    try:
        parts = shlex.split(expanded)
    except ValueError:
        # Fallback for malformed quoted strings
        parts = [expanded]
    return parts[0] if parts else None


def _resolve_executable(command: Optional[str]) -> Optional[str]:
    '''Resolve an executable from a configured command string.

    Parameters
    ----------
    command : Optional[str]
        Command string or executable path.

    Returns
    -------
    Optional[str]
        Absolute path to executable when found, else None.
    '''

    token = _extract_executable_token(command)
    if not token:
        return None

    if os.path.isabs(token):
        return token if (os.path.isfile(token) and os.access(token, os.X_OK)) else None

    return shutil.which(token)


def _probe_executable_version(executable: Optional[str]) -> str:
    '''Probe an external executable version string.

    Parameters
    ----------
    executable : Optional[str]
        Absolute path or executable name.

    Returns
    -------
    str
        A short version string, or "unknown" when probing fails.
    '''

    if not executable:
        return "unknown"

    probe_args = (
        ("--version",),
        ("-version",),
        ("version",),
        ("-v",),
    )

    for arg_list in probe_args:
        try:
            completed = subprocess.run(
                [executable, *arg_list],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            continue

        raw = "\n".join([completed.stdout or "", completed.stderr or ""]).strip()
        if not raw:
            continue
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), "")
        if first_line:
            return first_line[:240]

    return "unknown"


def _collect_tool_candidates() -> Dict[str, Optional[str]]:
    '''Collect configured command candidates for core external tools.

    Returns
    -------
    Dict[str, Optional[str]]
        Mapping tool name -> configured command/path candidate.
    '''

    candidates: Dict[str, Optional[str]] = {
        "vina": "vina",
        "smina": "smina",
        "plants": "plants",
        "gnina": "gnina",
        "pythonsh": "pythonsh",
        "dssp": "dssp",
        "obabel": "obabel",
        "spores": "spores",
    }

    try:
        from OCDocker.Config import get_config

        cfg = get_config()
        candidates["vina"] = getattr(cfg.vina, "executable", None) or candidates["vina"]
        candidates["smina"] = getattr(cfg.smina, "executable", None) or candidates["smina"]
        candidates["plants"] = getattr(cfg.plants, "executable", None) or candidates["plants"]
        candidates["gnina"] = getattr(cfg.gnina, "executable", None) or candidates["gnina"]
        candidates["pythonsh"] = getattr(cfg.tools, "pythonsh", None) or candidates["pythonsh"]
        candidates["dssp"] = getattr(cfg.tools, "dssp", None) or candidates["dssp"]
        candidates["obabel"] = getattr(cfg.tools, "obabel", None) or candidates["obabel"]
        candidates["spores"] = getattr(cfg.tools, "spores", None) or candidates["spores"]
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError):
        # Fall back to common executable names
        pass

    return candidates


def _collect_external_tool_manifest() -> Dict[str, Dict[str, Union[str, bool, None]]]:
    '''Collect external tool availability and version details.

    Returns
    -------
    Dict[str, Dict[str, Union[str, bool, None]]]
        Per-tool metadata with configured command, resolved path, availability, and version.
    '''

    manifest: Dict[str, Dict[str, Union[str, bool, None]]] = {}
    for tool_name, configured in _collect_tool_candidates().items():
        resolved = _resolve_executable(configured)
        manifest[tool_name] = {
            "configured": configured,
            "resolved": resolved,
            "available": bool(resolved),
            "version": _probe_executable_version(resolved),
        }
    return manifest


def _collect_python_package_versions() -> Dict[str, str]:
    '''Collect installed Python distribution versions.

    Returns
    -------
    Dict[str, str]
        Mapping distribution name -> version.
    '''

    packages: Dict[str, str] = {}
    try:
        from importlib.metadata import distributions
    except (ImportError, AttributeError):
        return packages

    for dist in distributions():
        try:
            name: Optional[str]
            try:
                name = dist.metadata["Name"]
            except KeyError:
                name = getattr(dist, "name", None)
            version = str(dist.version)
            if not name:
                continue
            packages[str(name)] = version
        except (AttributeError, TypeError, ValueError):
            continue

    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _collect_ocdocker_version() -> str:
    '''Collect OCDocker package version from import or metadata fallback.

    Returns
    -------
    str
        OCDocker version string or "unknown".
    '''

    try:
        import OCDocker as _oc

        value = getattr(_oc, "__version__", None)
        if value:
            return str(value)
    except (ImportError, AttributeError):
        pass

    try:
        from importlib.metadata import version as _pkg_version

        return str(_pkg_version("OCDocker"))
    except Exception:
        return "unknown"


def _collect_git_manifest() -> Dict[str, Optional[Union[str, bool]]]:
    '''Collect source control metadata when available.

    Returns
    -------
    Dict[str, Optional[Union[str, bool]]]
        Git metadata (best-effort).
    '''

    repo_root = Path(__file__).resolve().parents[2]

    def _run_git(args: List[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])

    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


## Public ##

def generate_reproducibility_manifest(include_python_packages: bool = True) -> Dict[str, Any]:
    '''Generate a reproducibility manifest with environment and version metadata.

    Parameters
    ----------
    include_python_packages : bool, optional
        Whether to include the full installed Python package list, by default True.

    Returns
    -------
    Dict[str, Any]
        Reproducibility manifest data.
    '''

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ocdocker": {
            "version": _collect_ocdocker_version(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": {
            "OCDOCKER_CONFIG": os.getenv("OCDOCKER_CONFIG"),
            "OCDOCKER_DB_BACKEND": os.getenv("OCDOCKER_DB_BACKEND"),
            "DB_BACKEND": os.getenv("DB_BACKEND"),
            "OCDOCKER_SQLITE_PATH": os.getenv("OCDOCKER_SQLITE_PATH"),
            "OCDOCKER_TIMEOUT": os.getenv("OCDOCKER_TIMEOUT"),
        },
        "external_tools": _collect_external_tool_manifest(),
        "git": _collect_git_manifest(),
    }

    if include_python_packages:
        packages = _collect_python_package_versions()
        manifest["python_packages"] = packages
        manifest["python_package_count"] = len(packages)

    return manifest


def cmd_manifest(args: argparse.Namespace) -> int:
    '''Generate and print a reproducibility manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    bootstrap_status: Dict[str, Optional[str]] = {"status": "skipped", "error": None}
    globals_ns = _preparse_global_args(sys.argv[1:])
    setattr(globals_ns, "_ocdocker_init_db", False)
    try:
        _bootstrap_ocdocker_env(globals_ns)
        bootstrap_status["status"] = "ok"
    except Exception as exc:
        # Manifest generation should still succeed even if config bootstrap fails.
        bootstrap_status["status"] = "error"
        bootstrap_status["error"] = f"{type(exc).__name__}: {exc}"

    manifest = generate_reproducibility_manifest(include_python_packages=(not args.no_packages))
    manifest["bootstrap"] = bootstrap_status

    payload = json.dumps(manifest, indent=2, sort_keys=True)

    output_path = getattr(args, "output", None)
    if output_path:
        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            print(f"Error: failed to write manifest file '{output_path}': {exc}")
            return 1

    print(payload)
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    '''Print package version without bootstrapping the full environment.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    '''

    _ = args
    print(_collect_ocdocker_version())
    return 0

def register_subparsers(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    p_manifest = sub.add_parser(
        "manifest",
        description=(
            "Generate a reproducibility manifest with runtime, platform, and tool versions.\n\n"
            "This command captures OCDocker version, Python/runtime details, external tool paths\n"
            "and versions, git metadata (when available), and optionally all installed Python\n"
            "package versions. Output is JSON suitable for archiving with docking results."
        ),
        help="Generate reproducibility manifest as JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
    )
    p_manifest.add_argument("--output", default=None, help="Optional path to write the manifest JSON file.")
    p_manifest.add_argument("--no-packages", action="store_true", help="Skip full Python package listing for faster/smaller output.")
    p_manifest.set_defaults(func=cmd_manifest)

    p_ver = sub.add_parser(
        "version",
        description="Print the installed version of OCDocker without bootstrapping the full environment.",
        help="Print OCDocker version",
        parents=[parent],
    )
    p_ver.set_defaults(func=cmd_version)
