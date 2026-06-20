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
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
