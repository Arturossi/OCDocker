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
