#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are for basic uses.

Usage:

import OCDocker.Toolbox.Basetools as ocbasetools
'''

# Imports
###############################################################################
import builtins
import contextlib

from tqdm import tqdm
from typing import Iterator

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


@contextlib.contextmanager
def redirect_to_tqdm() -> Iterator[None]:
    '''Redirects the stdout to tqdm.write()

    Returns
    -------
    contextlib.AbstractContextManager
        The context manager that redirects the stdout to tqdm.write().
    '''

    # Store builtin print
    old_print = print
    def new_print(*args, **kwargs) -> None:
        '''New print function that redirects the stdout to tqdm.write().

        Parameters
        ----------
        args : Any
            The arguments to be passed to tqdm.write().
        kwargs : Any
            The keyword arguments to be passed to tqdm.write().
        '''

        # If tqdm.write raises error, use builtin print
        try:
            tqdm.write(*args, **kwargs)
        except (OSError, IOError, AttributeError, BrokenPipeError):
            # Fallback to builtin print if tqdm.write fails
            old_print(*args, ** kwargs)

    try:
        # Globally replace built-in print with new_print
        builtins.print = new_print
        yield
    finally:
        builtins.print = old_print
