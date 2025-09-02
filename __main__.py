#!/usr/bin/env python3

"""
OCDocker module entry point.

Delegates to the unified CLI in `OCDocker.CLI`.
"""

# Imports
###############################################################################
from OCDocker.CLI import main as cli_main

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Description
###############################################################################

# Classes
###############################################################################

# Functions
###############################################################################

# Main Function
###############################################################################
def main():
    return cli_main()

# Execute
###############################################################################
if __name__ == "__main__":
    raise SystemExit(main())
