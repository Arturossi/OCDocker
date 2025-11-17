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

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##


@contextlib.contextmanager
def redirect_to_tqdm() -> contextlib.AbstractContextManager:
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
        # Globaly replace print with new_print
        inspect.builtins.print = new_print # type: ignore
        yield
    finally:
        inspect.builtins.print = old_print # type: ignore
