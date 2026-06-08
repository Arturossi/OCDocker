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
