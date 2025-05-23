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
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
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
