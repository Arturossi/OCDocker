#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to manipulate logs.

They are imported as:

import OCDocker.Toolbox.Logging as oclogging
'''

# Imports
###############################################################################
import os
import shutil
import time

from glob import glob

import OCDocker.Toolbox.FilesFolders as ocff

from OCDocker.Initialise import *

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
def clear_past_logs() -> None:
    '''Clear past logs entries.

    Parameters
    ----------
    None

    Returns
    -------
    None
    '''
    
    # For each dir in the log dir
    for pastLog in [d for d in glob(f"{logdir}/*") if os.path.isdir(d)]:
        # Extra check for avoid wrong deletions
        if pastLog.endswith("past"):
            # Remove all the folder
            shutil.rmtree(pastLog)
    return None

def backup_log(logname: str) -> None:
    '''Backup the current log.

    Parameters
    ----------
    logname : str
        Name of the log to be backed up.

    Returns
    -------
    None
    '''

    if os.path.isfile(f"{logdir}/{logname}.log"):
        if not os.path.isdir(f"{logdir}/read_log_past"):
            ocff.safe_create_dir(f"{logdir}/read_log_past")
        os.rename(f"{logdir}/{logname}.log", f"{logdir}/read_log_past/{logname}_{time.strftime('%d%m%Y-%H%M%S')}.log")
    
    return None
