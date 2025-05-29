#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are for basic uses.

They are imported as:

import OCDocker.Toolbox.Basetools as ocbasetools
'''

# Imports
###############################################################################
import contextlib
import inspect

from tqdm import tqdm

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

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##


@contextlib.contextmanager
def redirect_to_tqdm():
    '''Redirects the stdout to tqdm.write()

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''

    # Store builtin print
    old_print = print
    def new_print(*args, **kwargs):
        # If tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, ** kwargs)
    try:
        # Globaly replace print with new_print
        inspect.builtins.print = new_print # type: ignore
        yield
    finally:
        inspect.builtins.print = old_print # type: ignore
