#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to print informations.

Usage:

import OCDocker.Toolbox.Printing as ocprint
'''

# Imports
###############################################################################
import datetime
import inspect
from typing import Dict

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##
# Local fallback colors (no ANSI by default)
clrs: Dict[str, str] = {"r":"","g":"","y":"","b":"","p":"","c":"","n":""}

def _should_print() -> bool:
    '''Check whether direct prints should be emitted.

    Returns
    -------
    bool
        False when Rich logging is enabled, True otherwise.
    '''

    try:
        return not oclogging.is_rich_enabled()
    except Exception:
        return True


## Public ##

def print_error(message: str, force: bool = False) -> None:
    '''Print error. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.ERROR or force:
        oclogging.configure(level=ocerror.Error.get_output_level())
        log = oclogging.get_logger("printing")
        if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
            msg = f"ERROR: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'."
        else:
            msg = f"ERROR: {message}"
        log.error(msg)
        if _should_print():
            print(msg)
    return

def print_error_log(message: str, logfile: str, mode: str = 'a') -> None:
    '''Function to print error into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is 'a' (append).
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] ERROR: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_info(message: str, force: bool = False) -> None:
    '''Function to print info.

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.INFO or force:
        oclogging.configure(level=ocerror.Error.get_output_level())
        log = oclogging.get_logger("printing")
        if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
            msg = f"INFO: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'."
        else:
            msg = f"INFO: {message}"
        log.info(msg)
        if _should_print():
            print(msg)
    return

def print_info_log(message: str, logfile:str, mode: str = 'a') -> None:
    '''Function to print info into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is 'a' (append).
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] INFO: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_section(n: int, name: str, logName: str = "OCDocker_Progress.out") -> None:
    '''Print the section header and write progress to the progress file.

    Parameters
    ----------
    n : int
        Section number.
    name : str
        Section name (empty string for no log).
    logName : str, optional
        Log file name. Default is "OCDocker_Progress.out".
    '''

    # Print a nice section header
    print(f"\n{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          f"{clrs['r']}| " +
          f"{clrs['y']}S{clrs['r']}|" +
          f"{clrs['y']}E{clrs['r']}|" +
          f"{clrs['y']}C{clrs['r']}|" +
          f"{clrs['y']}T{clrs['r']}|" +
          f"{clrs['y']}I{clrs['r']}|" +
          f"{clrs['y']}O{clrs['r']}|" +
          f"{clrs['y']}N{clrs['r']}|" +
          f"{clrs['c']} {str(n)}{clrs['r']} | " +
          f"{clrs['c']}{str(name)}\n" +
          f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          clrs['n'])
    # Check if the section should be logged
    if name:
        # Check if is the Runtime Arguments section
        if name == "Runtime Arguments":
            with open(logName, 'w') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')}: Starting new OCDocker run\n")
        else:
            with open(logName, 'a') as f:
                f.write(f"\n{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n")
    return

def print_sorry()-> None:
    '''Function to print sorry message.'''

    # Print a nice looking sorry message :/
    print(f"**We are {clrs['y']}t{clrs['r']}e"+
          f"{clrs['y']}r{clrs['r']}r{clrs['y']}i"+
          f"{clrs['r']}b{clrs['y']}l{clrs['r']}y"+
          f"{clrs['n']} sorry... =(\n")
    return None

def print_subsection(n: int, name: str, logName: str = "OCDocker_Progress.out") -> None:
    '''Print the subsection header in progress file.

    Parameters
    ----------
    n : int
        Subsection number.
    name : str
        Subsection name.
    logName : str
        Log file name. Default is "OCDocker_Progress.out".
    '''

    # Print a nice subsection header
    print(f"\n{clrs['r']}|" +
          f"{clrs['y']}S" +
          f"{clrs['y']}u" +
          f"{clrs['y']}b" +
          f"{clrs['y']}s" +
          f"{clrs['y']}e" +
          f"{clrs['y']}c" +
          f"{clrs['y']}t" +
          f"{clrs['y']}o" +
          f"{clrs['y']}i" +
          f"{clrs['y']}n" +
          f"{clrs['c']} {str(n)}{clrs['r']}| " +
          f"{clrs['c']}{str(name)}\n" +
          f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          clrs['n'])

    if name:
        with open("OCDocker_Progress.out", 'a') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n")
    return

def print_success(message: str, force: bool = False) -> None:
    '''Print success. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.SUCCESS or force:
        oclogging.configure(level=ocerror.Error.get_output_level())
        log = oclogging.get_logger("printing")
        if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
            msg = f"SUCCESS: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'."
        else:
            msg = f"SUCCESS: {message}"
        log.info(msg)
        if _should_print():
            print(msg)
    return

def print_success_log(message: str, logfile: str, mode: str = 'a') -> None:
    '''Function to print success into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is 'a' (append).
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] SUCCESS: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_warning(message: str, force: bool = False) -> None:
    '''Function to print warning. [DEPRECATED]

    Parameters
    ----------
    message : str
        Message to be printed.
    force : bool, optional
        Forces the system to print the message, even if output_level is turning it off (USE WITH CAUTION!!!).
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.WARNING or force:
        oclogging.configure(level=ocerror.Error.get_output_level())
        log = oclogging.get_logger("printing")
        if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
            # For logging, don't add "WARNING:" prefix since logger adds it via levelname
            log_msg = f"{message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'."
            # For print, add "WARNING:" prefix
            print_msg = f"WARNING: {log_msg}"
        else:
            # For logging, don't add "WARNING:" prefix since logger adds it via levelname
            log_msg = message
            # For print, add "WARNING:" prefix
            print_msg = f"WARNING: {message}"
        log.warning(log_msg)
        if _should_print():
            print(print_msg)
    return

def print_warning_log(message: str, logfile: str, mode: str = 'a') -> None:
    '''Function to print warning into log.

    Parameters
    ----------
    message : str
        Message to be printed.
    logfile : str
        Log file to be used.
    mode : str, optional
        Mode to open the file. Default is 'a' (append).
    '''

    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] WARNING: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def printv(message: str) -> None:
    '''Function to print if verbosity mode is set.

    Parameters
    ----------
    message : str
        Message to be printed.
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
        # log + plain print for test expectations
        oclogging.configure(level=ocerror.Error.get_output_level())
        oclogging.get_logger("printing").debug(message)
        if _should_print():
            print(message)
    return

def section(n: int, name: str) -> str:
    '''Return the section header.

    Parameters
    ----------
    n : int
        Section number.
    name : str
        Section name.

    Returns
    -------
    str
        Section header.
    '''

    # Create a nice section header to return
    section_string = str(f"\n{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                         f"{clrs['r']}| "+
                         f"{clrs['y']}S{clrs['r']}|" +
                         f"{clrs['y']}E{clrs['r']}|" +
                         f"{clrs['y']}C{clrs['r']}|" +
                         f"{clrs['y']}T{clrs['r']}|" +
                         f"{clrs['y']}I{clrs['r']}|" +
                         f"{clrs['y']}O{clrs['r']}|" +
                         f"{clrs['y']}N{clrs['r']}|" +
                         f"{clrs['c']} {str(n)}{clrs['r']} | " +
                         f"{clrs['c']}{str(name)}\n" +
                         f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                         clrs['n'])

    return section_string

def subsection(n: int, name: str) -> str:
    '''Return the subsection header.

    Parameters
    ----------
    n : int
        Subsection number.
    name : str
        Subsection name.

    Returns
    -------
    str
        Subsection header.
    '''

    # Create a nice subsection header to return
    subsection_string = str(f"\n{clrs['r']}|" +
                            f"{clrs['y']}S" +
                            f"{clrs['y']}u" +
                            f"{clrs['y']}b" +
                            f"{clrs['y']}s" +
                            f"{clrs['y']}e" +
                            f"{clrs['y']}c" +
                            f"{clrs['y']}t" +
                            f"{clrs['y']}i" +
                            f"{clrs['y']}o" +
                            f"{clrs['y']}n" +
                            f"{clrs['c']} {str(n)}{clrs['r']}| " +
                            f"{clrs['c']}{str(name)}\n" +
                            f"{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
                            clrs['n'])

    return subsection_string
