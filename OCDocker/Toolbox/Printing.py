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
import inspect
import os
from typing import Dict
from datetime import datetime

import OCDocker.Error as ocerror
import OCDocker.Toolbox.Logging as oclogging

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
# Local fallback colors (no ANSI by default)
clrs: Dict[str, str] = {"r":"","g":"","y":"","b":"","p":"","c":"","n":""}

def _caller_context(depth: int = 2) -> str:
    '''Safely build caller context for debug messages.

    Parameters
    ----------
    depth : int
        How many frames to walk back (default 2 for caller of the public API).

    Returns
    -------
    str
        Context string or empty string if unavailable.
    '''

    frame = inspect.currentframe()
    for _ in range(depth):
        if frame is None or frame.f_back is None:
            return ""
        frame = frame.f_back

    if frame is None:
        return ""

    return (f"In function '{frame.f_code.co_name}' "
            f"line {frame.f_lineno} "
            f"from file '{frame.f_code.co_filename}'.")


def _write_log(message: str, logfile: str, mode: str) -> None:
    '''Write a message into a log file, creating its directory if needed.

    A log that cannot be written must not break the caller (most callers are
    already reporting an error), so the message is printed as a warning instead.

    Parameters
    ----------
    message : str
        Formatted line to be written.
    logfile : str
        Log file to be used.
    mode : str
        Mode to open the file.
    '''

    try:
        logdir = os.path.dirname(logfile)
        if logdir:
            os.makedirs(logdir, exist_ok=True)
        if "a" in mode:
            oclogging.rotate_log(logfile)
        with open(logfile, mode) as f:
            f.write(message)
    except OSError as e:
        print_warning(f"Could not write to the log file '{logfile}': {e}. Message: {message.rstrip()}")
    return


## Public ##

def print_error(message: str, force: bool = False) -> None:
    '''Print error.

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
            context = _caller_context()
            msg = f"ERROR: {message} {context}".strip()
        else:
            msg = f"ERROR: {message}"
        log.error(msg)
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

    today = datetime.now()
    context = _caller_context()
    suffix = f" {context}" if context else ""
    _write_log(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] ERROR: {message}{suffix}\n", logfile, mode)
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
            context = _caller_context()
            msg = f"INFO: {message} {context}".strip()
        else:
            msg = f"INFO: {message}"
        log.info(msg)
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

    today = datetime.now()
    context = _caller_context()
    suffix = f" {context}" if context else ""
    _write_log(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] INFO: {message}{suffix}\n", logfile, mode)
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
            _write_log(f"{datetime.now().strftime('%H:%M:%S')}: Starting new OCDocker run\n", logName, 'w')
        else:
            _write_log(f"\n{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n", logName, 'a')
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
        _write_log(f"{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n", logName, 'a')
    return

def print_success(message: str, force: bool = False) -> None:
    '''Print success.

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
            context = _caller_context()
            msg = f"SUCCESS: {message} {context}".strip()
        else:
            msg = f"SUCCESS: {message}"
        log.info(msg)
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

    today = datetime.now()
    context = _caller_context()
    suffix = f" {context}" if context else ""
    _write_log(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] SUCCESS: {message}{suffix}\n", logfile, mode)
    return

def print_warning(message: str, force: bool = False) -> None:
    '''Function to print warning.

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
            # Don't add "WARNING:" prefix since the logger adds it via levelname
            context = _caller_context()
            log_msg = f"{message} {context}".strip()
        else:
            log_msg = message
        log.warning(log_msg)
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

    today = datetime.now()
    context = _caller_context()
    suffix = f" {context}" if context else ""
    _write_log(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] WARNING: {message}{suffix}\n", logfile, mode)
    return

def printv(message: str) -> None:
    '''Function to print if verbosity mode is set.

    Parameters
    ----------
    message : str
        Message to be printed.
    '''

    if ocerror.Error.output_level >= ocerror.ReportLevel.DEBUG:
        oclogging.configure(level=ocerror.Error.get_output_level())
        log = oclogging.get_logger("printing")
        log.debug(message)
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
