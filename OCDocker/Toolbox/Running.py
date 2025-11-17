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
import shutil
import subprocess

import OCDocker.Toolbox.Printing as ocprint
import OCDocker.Error as ocerror

from typing import List, Tuple, Union, Optional


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
def run(cmd: List[str], logFile: str = "", cwd: str = "", timeout: Optional[int] = None) -> Union[int, Tuple[int, str]]:
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

    if not isinstance(cmd, list):
        return ocerror.Error.wrong_type(message = f"The argument cmd has to be a list! Found '{type(cmd)}' instead...", level = ocerror.ReportLevel.ERROR)

    # Print verboosity
    ocprint.printv(f"Running the command '{' '.join(cmd)}'.")

    if logFile == "":
        ocprint.printv(f"No log will be made")
        logFile = os.devnull
    else:
        ocprint.printv(f"Logging into '{logFile}'")

    # Resolve timeout from param or environment variable
    if timeout is None:
        try:
            timeout_env = int(os.getenv("OCDOCKER_TIMEOUT", "0"))
            timeout = timeout_env if timeout_env > 0 else None
        except (ValueError, TypeError):
            # Ignore invalid timeout values
            timeout = None

    # Validate executable availability (avoid PermissionError on empty string)
    exe = str(cmd[0])
    if not exe:
        return ocerror.Error.subprocess(message = "Executable not set (empty). Check your configuration.", level=ocerror.ReportLevel.ERROR)
    if os.path.isabs(exe):
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            return ocerror.Error.subprocess(message = f"Executable not found or not executable: '{exe}'", level=ocerror.ReportLevel.ERROR)
    else:
        if shutil.which(exe) is None:
            return ocerror.Error.subprocess(message = f"Executable not found on PATH: '{exe}'", level=ocerror.ReportLevel.ERROR)

    try:
        if cwd == "":
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, timeout=timeout)
        else:
            with open(logFile, 'w') as outfile:
                proc = subprocess.run(cmd, stdout=outfile, cwd=cwd, stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError as e:
        return ocerror.Error.subprocess(message = f"Command not found when executing '{' '.join(cmd)}': {e}", level=ocerror.ReportLevel.ERROR)
    except subprocess.TimeoutExpired as e:
        return ocerror.Error.subprocess(message = f"Timeout expired after {timeout}s for command '{' '.join(cmd)}'", level=ocerror.ReportLevel.ERROR)
    except Exception as e:
        return ocerror.Error.subprocess(message = f"Found a problem while executing the command '{' '.join(cmd)}': {e}", level=ocerror.ReportLevel.ERROR)

    # If the command has not been executed successfully
    if proc.returncode != 0:
        return ocerror.Error.subprocess(message = f"The command '{' '.join(cmd)}' has not been executed successfully!", level = ocerror.ReportLevel.ERROR), proc.stderr.decode("utf-8")
    return ocerror.Error.ok()


def is_tool_available(exe: str) -> bool:
    '''Check if a tool executable is available.
    
    Parameters
    ----------
    exe : str
        Path to the executable (can be absolute or command name)
        
    Returns
    -------
    bool
        True if the executable is available, False otherwise
    '''
    
    if not exe:
        return False
    return (os.path.isabs(exe) and os.path.isfile(exe) and os.access(exe, os.X_OK)) or (shutil.which(exe) is not None)


### Special functions
