#!/usr/bin/env python3

# Description
###############################################################################
'''
Side-effect-free interactive console package for OCDocker.

Importing this package does not print a banner, bootstrap configuration, or
start a REPL. Launch the console explicitly via ``ocdocker console`` or
``python -m OCDocker.Console``.

Usage:

import OCDocker.Console

Modules
-------
- __init__: Package entry; lazy export of ``main``.
- app: Console bootstrap and ``run_console`` / ``main`` entrypoints.
- session: Namespace construction, REPL loop, ``print_args``, ``clean_test_files``.
- commands: Built-in console commands (``help``, ``exit``).
- render: Welcome banner rendering.
- __main__: ``python -m OCDocker.Console`` entrypoint.
'''

from __future__ import annotations

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

__all__ = ["main"]

# Functions
###############################################################################
## Public ##


def __getattr__(name: str):
    '''Lazy attribute resolver for ``main`` without import-time side effects.'''

    if name == "main":
        from OCDocker.Console.app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
