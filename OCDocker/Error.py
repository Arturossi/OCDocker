#!/usr/lib/python3

# Imports
###############################################################################
import sys
import shutil
import tarfile
import datetime
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

# Description
###############################################################################
'''
Sets of classes and functions that are used to make all return codes in OCDocker
standard.

They are imported as:

import OCDocker.Error as ocerror
'''

# Classes
###############################################################################
class Error:
    """
    Class to handle errors and standarize them across the whole code.
    If any error needs special treatment
    """
    def __init__(self):
        # Common errors
        self.ok                   = 0
        self.unkown               = -666

        # File errors
        self.fileExists           = 100
        self.fileDoNotExist       = 101
        self.readFile             = 102
        self.writeFile            = 103
        self.untarFile            = 104
        self.unsupportedExtension = 105

        # Directory errors
        self.createDirError       = 150

        # Errors with Variables
        self.wrongType            = 200

        # Subprocess errors
        self.

    def __print_msg(self, message, level):
        '''
        Prints a message based on level.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        if message:
            if level == "warn":
                octools.print_error(message)
            else:
                octools.print_error(message)
        return

    def ok(self):
        '''
        Return this when no error appears.
        Input:
          -
        Return:
          -
        '''
        return self.ok

    def unkown(self, message="", level="warn"):
        '''
        Return when the error is unknown.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.fileExists

    def file_exists(self, message="", level="warn"):
        '''
        Return when the file already exists.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.fileExists

    def file_do_not_exist(self, message="", level="warn"):
        '''
        Return this when the file do not exist.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.fileDoNotExist

    def read_file(self, message="", level="warn"):
        '''
        Return this when a file could not be read.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.readFile

    def write_file(self, message="", level="warn"):
        '''
        Return this when a file could not be written.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.writeFile

    def untar_file(self, message="", level="warn"):
        '''
        Return this when the untar action fails.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.untarFile

    def unsupported_extension(self, message="", level="warn"):
        '''
        Return this when the extension is not supported.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.unsupportedExtension

    def create_dir(self, message="", level="warn"):
        '''
        Return this when the directory creation fails.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        __print_msg(message, level)
        return self.createDirError

    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"\t+------------------------------------------+")
        print(f"\t|        OCDocker Return codes         |")
        print(f"\t+------------------------------------------+")

        print(f"\n\t~~~~~~~~~~~~~~ GENERAL ERRORS ~~~~~~~~~~~~~~")
        print(f"\t - No error:                    {self.ok}")
        print(f"\t - Unknown error:               {self.unkown}")

        print(f"\n\t~~~~~~~~~~~~~~~~ FILE ERRORS ~~~~~~~~~~~~~~~")
        print(f"\t - File exists:                 {self.fileExists}")
        print(f"\t - File does not exists:        {self.fileDoNotExist}")
        print(f"\t - Read file error:             {self.readFileError}")
        print(f"\t - Write file error:            {self.writeFileError}")
        print(f"\t - Untar error:                 {self.untarFile}")
        print(f"\t - Unsupported extension error: {self.unsupportedExtension}")

        print(f"\n\t~~~~~~~~~~~~~ DIRECTORY ERRORS ~~~~~~~~~~~~~")
        print(f"\t - Directory creation error:    {self.createDirError}")

        print(f"\n\t~~~~~~~~~~ ERRORS WITH VARIABLES ~~~~~~~~~~~")
        print(f"\t - Wrong type:                  {self.wrongType}")

        print(f"\n\t~~~~~~~~~~~~~~ PROCESS ERRORS ~~~~~~~~~~~~~~")

        return

# Functions
###############################################################################
