#!/usr/bin/env python3

# Description
###############################################################################
'''
OCDocker module entry point.

Delegates to the unified CLI in `OCDocker.CLI`.
'''

# Imports
###############################################################################
from OCDocker.CLI import main as cli_main

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def main():
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
