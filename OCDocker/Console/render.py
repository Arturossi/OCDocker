#!/usr/bin/env python3

# Description
###############################################################################
'''
Console rendering helpers.

Renders the welcome banner once at session start (never at import time).
'''

from __future__ import annotations

# Imports
###############################################################################
import textwrap as tw

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Functions
###############################################################################
## Public ##


def print_welcome_banner() -> None:
    '''Print the OCDocker console welcome banner once at session start.'''

    from OCDocker.Initialise import clrs

    message = tw.dedent(
        f"""{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{clrs["c"]}
                              CONSOLE MODE{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{clrs["n"]}
          Welcome to the OCDocker interactive console!

      This console allows you to interact with the OCDocker pipeline
      step by step.

      {clrs["g"]}TIP{clrs["n"]} It's an interesting way to learn OCDocker, useful
      for debugging, and great for quick API interaction and
      experimentation.

      To check the args variable use print_args() function.{clrs["y"]}
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{clrs["n"]}"""
    )
    print(message)
