#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to run commands in the OS.

They are imported as:

import OCDocker.Toolbox.Running as ocrun
'''

# Imports
###############################################################################
import os
import subprocess

import OCDocker.Toolbox.Printing as ocprint

from typing import List, Tuple, Union

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
def run(cmd: List[str], logFile: str = "", cwd : str = "") -> Union[int, Tuple[int, str]]:
    '''Run the given command (generic).

    Parameters
    ----------
    cmd : List[str]
        The command to be run.
    logFile : str, optional
        The file where the output will be saved. Default is "".
    cwd : str, optional
        The current working directory. Default is "".

    Returns
    -------
    int | Tuple[int, str]
        The exit code of the command (based on the Error.py code table) or a tuple with the exit code and the stderr of the command.
    '''

    if not cmd:
        return ocerror.Error.not_set(message = f"The variable cmd is not set or is an empty list!", level = ocerror.ReportLevel.ERROR)

    if type(cmd) != list:
        return ocerror.Error.wrong_type(message = f"The argument cmd has to be a list! Found '{type(cmd)}' instead...", level = ocerror.ReportLevel.ERROR)

    # Print verboosity
    ocprint.printv(f"Running the command '{' '.join(cmd)}'.")

    if logFile == "":
        ocprint.printv(f"No log will be made")
        logFile = os.devnull
    else:
        ocprint.printv(f"Logging into '{logFile}'")

    try:
        if cwd == "":
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout = outfile, stderr = subprocess.PIPE)
        else:
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout = outfile, cwd=cwd, stderr = subprocess.PIPE)
    except Exception as e:
        return ocerror.Error.subprocess(message = f"Found a problem while executing the command '{' '.join(cmd)}': {e}", level=ocerror.ReportLevel.ERROR)

    # If the command has not been executed successfully
    if proc.returncode != 0:
        return ocerror.Error.subprocess(message = f"The command '{' '.join(cmd)}' has not been executed successfully!", level = ocerror.ReportLevel.ERROR), proc.stderr.decode("utf-8")
    return ocerror.Error.ok()


### Special functions
