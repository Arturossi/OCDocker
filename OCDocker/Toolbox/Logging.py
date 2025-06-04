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
