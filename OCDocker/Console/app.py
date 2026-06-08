#!/usr/bin/env python3

# Description
###############################################################################
'''
Console application entrypoint.

Coordinates environment bootstrap (when invoked standalone), logging setup,
banner rendering, namespace construction, and the interactive REPL loop.
'''

from __future__ import annotations

# Imports
###############################################################################
import argparse
import sys
from typing import Optional

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.Console.render import print_welcome_banner
from OCDocker.Console.session import build_namespace, run_interactive

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
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

LOGGER = oclogging.get_logger("cli")

# Functions
###############################################################################
## Public ##


def run_console(args: Optional[argparse.Namespace] = None) -> int:
    '''Start the console after the environment has been bootstrapped.

    Parameters
    ----------
    args : Optional[argparse.Namespace], optional
        Parsed CLI namespace from ``ocdocker console``. When provided, logging
        is configured from CLI flags and ``--ipython`` is honored.

    Returns
    -------
    int
        Exit code (0 for success, non-zero on failure).
    '''

    if args is not None:
        try:
            import OCDocker.Error as ocerror

            oclogging.configure(
                level=ocerror.Error.get_output_level(),
                log_file=getattr(args, "log_file", None),
                to_stdout=False,
                use_rich=True,
            )
        except (ImportError, AttributeError, OSError):
            pass

    print_welcome_banner()

    try:
        namespace = build_namespace()
    except ModuleNotFoundError as exc:
        from OCDocker.CLI.common import _print_optional_dependency_hint, _suggest_extra_for_missing_module

        extra = _suggest_extra_for_missing_module(getattr(exc, "name", ""))
        return _print_optional_dependency_hint(
            feature="interactive console",
            extra=extra,
            exc=exc,
        )
    except Exception:
        LOGGER.exception("Failed to build console namespace")
        return 1

    use_ipython = bool(getattr(args, "ipython", False)) if args is not None else False
    return run_interactive(namespace, use_ipython=use_ipython)


def main(argv: Optional[list[str]] = None) -> int:
    '''Standalone console entry (``python -m OCDocker.Console``).

    Bootstraps OCDocker via ``OCDocker.Initialise`` when ``argv`` is ``None``,
    or via CLI global-flag preparse when a custom argv list is supplied.

    Parameters
    ----------
    argv : Optional[list[str]], optional
        Command-line arguments excluding the program name. Defaults to compatibility
        Initialise parsing when ``None``.

    Returns
    -------
    int
        Exit code from :func:`run_console`.
    '''

    from OCDocker.CLI.common import _bootstrap_ocdocker_env, _preparse_global_args
    from OCDocker.Initialise import argument_parsing, bootstrap

    if argv is None:
        bootstrap(argument_parsing())
    else:
        globals_ns = _preparse_global_args(list(argv))
        setattr(globals_ns, "_ocdocker_init_db", False)
        _bootstrap_ocdocker_env(globals_ns)

    return run_console()
