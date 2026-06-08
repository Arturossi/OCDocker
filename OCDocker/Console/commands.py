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
