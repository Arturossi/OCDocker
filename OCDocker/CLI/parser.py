#!/usr/bin/env python3
"""Build the top-level ``ocdocker`` argument parser."""

from __future__ import annotations

import argparse

from OCDocker.CLI import console as cli_console
from OCDocker.CLI import doctor as cli_doctor
from OCDocker.CLI import init_config as cli_init_config
from OCDocker.CLI import manifest as cli_manifest
from OCDocker.CLI import pipeline as cli_pipeline
from OCDocker.CLI import script as cli_script
from OCDocker.CLI import vs as cli_vs
from OCDocker.CLI.ocscore import register_ocscore_subparser


def _make_parent_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--multiprocess", dest="multiprocess", action="store_true", default=argparse.SUPPRESS, help="Enable multiprocessing for supported tasks")
    parent.add_argument("--no-multiprocess", dest="multiprocess", action="store_false", default=argparse.SUPPRESS, help="Disable multiprocessing for supported tasks")
    parent.add_argument("-u", "--update-databases", dest="update", action="store_true", default=argparse.SUPPRESS, help="Update databases on startup")
    parent.add_argument("--conf", dest="config_file", type=str, default=argparse.SUPPRESS, help="Path to OCDocker configuration file (.cfg or .yml)")
    parent.add_argument("--output-level", dest="output_level", type=int, default=argparse.SUPPRESS, help="Logging verbosity level (0-5)")
    parent.add_argument("--overwrite", dest="overwrite", action="store_true", default=argparse.SUPPRESS, help="Allow overwriting existing output files")
    parent.add_argument("--threads", dest="threads", type=int, default=argparse.SUPPRESS, help="Maximum worker threads/processes for scheduler-managed runs")
    parent.add_argument("--tmp-dir", dest="tmp_dir", type=str, default=argparse.SUPPRESS, help="Job-local temporary directory")
    parent.add_argument("--log-file", dest="log_file", type=str, default=argparse.SUPPRESS, help="Write log messages to this file")
    parent.add_argument("--no-stdout-log", dest="no_stdout_log", action="store_true", default=argparse.SUPPRESS, help="Disable logging to stdout")
    parent.add_argument("--no-splash", dest="no_splash", action="store_true", default=argparse.SUPPRESS, help="Disable splash banner on startup")
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser with subcommands."""

    parser = argparse.ArgumentParser(
        prog="ocdocker",
        description=(
            "OCDocker CLI: Unified command-line interface for molecular docking, virtual screening, and analysis.\n\n"
            "Main commands:\n"
            "  vs        - Single-engine docking with rescoring of all poses\n"
            "  pipeline  - Multi-engine consensus docking with clustering and representative pose selection\n"
            "  ocscore   - Staged OCScore ML pipeline (reduce, train, export tools)\n"
            "  console   - Interactive Python console with OCDocker pre-loaded\n"
            "  script    - Run a Python script with OCDocker libraries pre-loaded\n"
            "  doctor    - Environment diagnostics and setup verification\n"
            "  init-config - Create starter configuration file\n"
            "  manifest  - Generate reproducibility manifest with version metadata\n"
            "  version   - Print version information\n\n"
            "Use 'ocdocker <command> --help' for detailed information about each command."
        ),
        epilog=(
            "Note: SQLite backend is intended for development/tests. "
            "For production workloads (performance/concurrency), use PostgreSQL (default) or MySQL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--multiprocess",
        dest="multiprocess",
        action="store_true",
        default=True,
        help="Enable multiprocessing for supported tasks. Allows parallel execution when possible. Default: enabled",
    )
    parser.add_argument(
        "--no-multiprocess",
        dest="multiprocess",
        action="store_false",
        help="Disable multiprocessing for supported tasks. Useful for stability/debugging.",
    )
    parser.add_argument(
        "-u",
        "--update-databases",
        dest="update",
        action="store_true",
        default=False,
        help="Update databases on startup. Runs database schema updates and migrations if needed.",
    )
    parser.add_argument(
        "--conf",
        dest="config_file",
        type=str,
        help="Path to OCDocker configuration file (.cfg or .yml). If not specified, uses default locations or OCDOCKER_CONFIG environment variable.",
    )
    parser.add_argument(
        "--output-level",
        dest="output_level",
        type=int,
        default=1,
        help="Logging verbosity level (0-5). Higher numbers provide more detailed output. 0=silent, 1=normal, 2-5=increasing verbosity. Default: 1",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Allow overwriting existing output files. By default, existing files are preserved to prevent accidental data loss.",
    )
    parser.add_argument(
        "--threads",
        dest="threads",
        type=int,
        default=None,
        help="Maximum worker threads/processes for scheduler-managed runs. Also accepts OCDOCKER_THREADS or SNAKEMAKE_THREADS.",
    )
    parser.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        type=str,
        default=None,
        help="Job-local temporary directory. Also accepts OCDOCKER_TMP_DIR.",
    )
    parser.add_argument(
        "--log-file",
        dest="log_file",
        type=str,
        default=None,
        help="Write log messages to this file in addition to stdout. Useful for saving detailed logs for later analysis.",
    )
    parser.add_argument(
        "--no-stdout-log",
        dest="no_stdout_log",
        action="store_true",
        default=False,
        help="Disable logging to stdout. Only log to file if --log-file is specified. Useful for cleaner console output.",
    )
    parser.add_argument(
        "--no-splash",
        dest="no_splash",
        action="store_true",
        default=False,
        help="Disable splash banner on startup.",
    )

    parent = _make_parent_parser()
    sub = parser.add_subparsers(dest="command", required=True)

    cli_init_config.register_subparser(sub, parent)
    cli_manifest.register_subparsers(sub, parent)
    cli_vs.register_subparser(sub, parent)
    register_ocscore_subparser(sub)
    cli_pipeline.register_subparser(sub, parent)
    cli_console.register_subparser(sub, parent)
    cli_script.register_subparser(sub, parent)
    cli_doctor.register_subparser(sub, parent)

    return parser
