#!/usr/bin/env python3
"""Interactive OCDocker console CLI command."""

from __future__ import annotations

import argparse
import sys

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.CLI.common import _bootstrap_ocdocker_env, _preparse_global_args
from OCDocker.Console.app import run_console

LOGGER = oclogging.get_logger("cli")


def cmd_console(args: argparse.Namespace) -> int:  # pragma: no cover - interactive console
    """Launch the OCDocker interactive console.

    Respects global flags by bootstrapping environment first.

    Parameters
    ----------
    args
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    globals_ns = _preparse_global_args(sys.argv[1:])
    setattr(globals_ns, "_ocdocker_init_db", False)
    _bootstrap_ocdocker_env(globals_ns)

    try:
        return run_console(args)
    except Exception:
        LOGGER.exception("Interactive console exited with error")
        return 1


def register_subparser(sub: argparse._SubParsersAction, parent: argparse.ArgumentParser) -> None:
    p_console = sub.add_parser(
        "console",
        description=(
            "Launch an interactive Python console with OCDocker pre-loaded.\n\n"
            "This provides an interactive environment with tab-completion and command history,\n"
            "allowing you to use OCDocker programmatically. All OCDocker modules are imported\n"
            "and ready to use. Useful for exploratory work, debugging, or custom workflows\n"
            "that do not fit the standard CLI commands."
        ),
        help="Open interactive OCDocker Python console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent],
    )
    p_console.add_argument(
        "--ipython",
        action="store_true",
        default=False,
        help="Use IPython embed when available (default: simple command loop).",
    )
    p_console.set_defaults(func=cmd_console)
