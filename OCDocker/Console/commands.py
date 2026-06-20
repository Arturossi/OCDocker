#!/usr/bin/env python3

# Description
###############################################################################
'''
Built-in interactive console commands.

Provides ``help`` and ``exit`` / ``quit`` handling before Python code execution
in the REPL loop.
'''

from __future__ import annotations

# Imports
###############################################################################
from typing import Any, Mapping

# License
###############################################################################
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

HELP_TEXT = """\
Available console commands:
  help              Show this message
  exit, quit        Leave the console

All other input is executed as Python code in the preloaded OCDocker namespace.
Use print_args() to inspect runtime configuration (e.g. print_args('vina')).
"""

# Functions
###############################################################################
## Public ##


def handle_command(line: str, namespace: Mapping[str, Any]) -> bool:
    '''Handle a built-in console command.

    Parameters
    ----------
    line : str
        Raw input line from the user.
    namespace : Mapping[str, Any]
        Interactive namespace (reserved for future built-ins).

    Returns
    -------
    bool
        ``True`` when the line was a built-in command and was handled.
    '''

    _ = namespace
    cmd = line.strip().lower()
    if cmd in ("help", "?"):
        print(HELP_TEXT)
        return True
    if cmd in ("exit", "quit"):
        return True
    return False
