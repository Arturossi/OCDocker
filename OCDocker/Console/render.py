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
