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
