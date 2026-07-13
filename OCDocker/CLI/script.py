#!/usr/bin/env python3
"""Run a Python script with OCDocker pre-loaded."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

import OCDocker.Toolbox.Logging as oclogging

import OCDocker.Toolbox.Security as ocsec

from OCDocker.CLI.common import (
    _bootstrap_ocdocker_env,
    _preparse_global_args,
    _print_optional_dependency_hint,
    _suggest_extra_for_missing_module,
)

LOGGER = oclogging.get_logger("cli")

def cmd_script(args: argparse.Namespace) -> int:  # pragma: no cover - script execution is user-provided code
    '''Run a Python script with OCDocker libraries pre-loaded.

    Bootstraps the environment, loads all OCDocker modules, and executes
    the provided script file with those modules available in the namespace.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments. Must contain 'script_file' and optionally
        'script_args'. Dynamic execution requires explicit trust via
        '--allow-unsafe-exec' or 'OCDOCKER_ALLOW_SCRIPT_EXEC=1'.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure, 2 for blocked unsafe execution).
    '''

    # Dynamic in-process execution is a trust boundary.
    try:
        ocsec.require_trusted_input(
            trusted=bool(getattr(args, "allow_unsafe_exec", False)),
            operation="dynamic script execution",
            env_var="OCDOCKER_ALLOW_SCRIPT_EXEC",
            source=str(getattr(args, "script_file", "")),
        )
    except PermissionError as e:
        print(f"Security check failed: {e}")
        LOGGER.warning("Security check failed for script mode: %s", e)
        return 2

    # Bootstrap env to ensure Initialise is safe to import
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

    # Validate script file exists
    script_path = Path(args.script_file)
    if not script_path.exists():
        print(f"Error: Script file not found: {script_path}")
        return 1

    if not script_path.is_file():
        print(f"Error: Path is not a file: {script_path}")
        return 1

    # Load OCDocker libraries into namespace (similar to OCDocker.Console)
    script_namespace = {}

    # Import all OCDocker modules
    try:
        # Import Initialise module and add all non-dunder symbols to namespace
        import OCDocker.Initialise as ocinit
        for k, v in vars(ocinit).items():
            if not k.startswith('__'):
                script_namespace[k] = v

        import OCDocker.Toolbox as octools
        script_namespace['octools'] = octools
        script_namespace['ocsec'] = ocsec
        script_namespace['allow_unsafe_runtime'] = ocsec.allow_unsafe_runtime

        import OCDocker.Ligand as ocl
        script_namespace['ocl'] = ocl

        import OCDocker.Receptor as ocr
        script_namespace['ocr'] = ocr

        import OCDocker.Docking.Vina as ocvina
        script_namespace['ocvina'] = ocvina

        import OCDocker.Docking.Smina as ocsmina
        script_namespace['ocsmina'] = ocsmina

        import OCDocker.Docking.PLANTS as ocplants
        script_namespace['ocplants'] = ocplants

        import OCDocker.Processing.Preprocessing.RMSDClustering as ocrmsdclust
        script_namespace['ocrmsdclust'] = ocrmsdclust

        try:
            import OCDocker.Rescoring.ODDT as ocoddt
            script_namespace['ocoddt'] = ocoddt
        except ModuleNotFoundError as exc:
            missing_mod = getattr(exc, 'name', '')
            if missing_mod == 'oddt' or missing_mod.startswith('oddt.'):
                print("Warning: optional dependency 'oddt' is not installed; 'ocoddt' is unavailable in script mode.")
            else:
                raise

        import OCDocker.Toolbox.Conversion as occonversion
        script_namespace['occonversion'] = occonversion

        import OCDocker.Toolbox.MoleculeProcessing as ocmolproc
        script_namespace['ocmolproc'] = ocmolproc

        # Add standard library modules that scripts commonly need
        from glob import glob
        from pprint import pprint
        script_namespace['os'] = os
        script_namespace['sys'] = sys
        script_namespace['Path'] = Path
        script_namespace['glob'] = glob
        script_namespace['pprint'] = pprint

    except ModuleNotFoundError as e:
        extra = _suggest_extra_for_missing_module(getattr(e, "name", ""))
        return _print_optional_dependency_hint(
            feature="script mode preloaded modules",
            extra=extra,
            exc=e,
        )
    except Exception as e:
        print(f"Error loading OCDocker libraries: {e}")
        LOGGER.exception("Error loading OCDocker libraries for script mode")
        return 1

    # Update sys.argv to include script args for the script's use
    original_argv = sys.argv[:]
    try:
        # Set sys.argv to: [script_file, ...script_args]
        sys.argv = [str(script_path)] + (args.script_args or [])

        # Read and execute the script
        script_content = script_path.read_text(encoding='utf-8')

        # Compile the script for better error messages
        try:
            compiled_script = compile(script_content, str(script_path), 'exec')
        except SyntaxError as e:
            print(f"Syntax error in script {script_path}:")
            print(f"  Line {e.lineno}: {e.text}")
            print(f"  {e.msg}")
            return 1

        # Execute the script with the loaded namespace
        exec(compiled_script, script_namespace)

        return 0

    except SystemExit as e:
        # Script called sys.exit(), respect its exit code
        return int(e.code) if e.code is not None else 0
    except KeyboardInterrupt:
        print("\nScript execution interrupted by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error executing script {script_path}:")
        LOGGER.exception("Error executing script %s", script_path)
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv



def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    '''Register the ``ocdocker script`` command group.

    Parameters
    ----------
    sub : argparse._SubParsersAction
        Main CLI subparser registry.
    parent : argparse.ArgumentParser
        Parent parser supplying shared global arguments.
    '''

    p_script = sub.add_parser(
        "script",
        description=(
            "Run a Python script with OCDocker libraries pre-loaded.\n\n"
            "Security note: This executes the script directly in-process. Only run scripts you trust.\n"
            "You must pass --allow-unsafe-exec (or set OCDOCKER_ALLOW_SCRIPT_EXEC=1)."
        ),
        help="Run a Python script with OCDocker libraries pre-loaded",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
    )
    p_script.add_argument("script_file", help="Path to the Python script file to execute.")
    p_script.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments passed to the script.")
    p_script.add_argument(
        "--allow-unsafe-exec",
        action="store_true",
        default=False,
        help="Required opt-in for in-process script execution.",
    )
    p_script.set_defaults(func=cmd_script)
