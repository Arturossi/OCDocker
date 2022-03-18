#!/usr/lib/python3

# Imports
###############################################################################
import inspect
import datetime

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
    def __init__(self, args):
        # OCDocker arguments
        self.args = args

        # Common errors
        self.okCode                   = 0
        self.unkownCode               = -666

        # File errors
        self.fileExistsCode           = 100
        self.fileDoNotExistCode       = 101
        self.readFileCode             = 102
        self.writeFileCode            = 103
        self.untarFileCode            = 104
        self.unsupportedExtensionCode = 105

        # Directory errors
        self.createDirCode            = 150

        # Variable errors
        self.wrongTypeCode            = 200
        self.notSetCode               = 201

        # Subprocess errors
        self.subprocessCode           = 300

        # Molecule error
        self.parseMoleculeCode        = 400
        self.malformedMoleculeCode    = 401

    ## Private ##
    def __print_success(self, message):
        '''
        Print success.
        Input:
          message [string] - Message to be printed.
        Return:
          -
        '''
        today = datetime.datetime.now()
        if self.args.output_level >= 4:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;92mSUCCSESS\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.")
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033 [1;92mSUCCSESS\033[1;0m: {message}")
        return

    def __print_warning(self, message):
        '''
        Function to print warning.
        Input:
          message [string] - Message to be printed.
        Return:
          -
        '''
        today = datetime.datetime.now()
        if self.args.output_level >= 3:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;93mWARNING\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.")
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;93mWARNING\033[1;0m: {message}")
        return

    def __print_error(self, message):
        '''
        Print error.
        Input:
          message [string] - Message to be printed.
        Return:
          -
        '''
        today = datetime.datetime.now()
        if self.args.output_level >= 3:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;91mERROR\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.")
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;91mERROR\033[1;0m")
        return

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
            if level == "warn" and self.args.output_level >= 1:
                self.__print_warning(message)
            elif level == "error" and self.args.output_level >= 0:
                self.__print_error(message)
            elif level == "success" and self.args.output_level >= 3:
                self.__print_success(message)
        return

    ## Public ##
    # Common errors
    def ok(self, message=""):
        '''
        Return this when no error appears.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
        Return:
          -
        '''
        self.__print_msg(message, "success")
        return self.okCode

    def unkown(self, message="", level="warn"):
        '''
        Return when the error is unknown.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.fileExistsCode

    # File errors
    def file_exists(self, message="", level="warn"):
        '''
        Return when the file already exists.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.fileExistsCode

    def file_do_not_exist(self, message="", level="warn"):
        '''
        Return this when the file do not exist.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.fileDoNotExistCode

    def read_file(self, message="", level="warn"):
        '''
        Return this when a file could not be read.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.readFileCode

    def write_file(self, message="", level="warn"):
        '''
        Return this when a file could not be written.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.writeFileCode

    def untar_file(self, message="", level="warn"):
        '''
        Return this when the untar action fails.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.untarFileCode

    def unsupported_extension(self, message="", level="warn"):
        '''
        Return this when the extension is not supported.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.unsupportedExtensionCode

    # Directory errors
    def create_dir(self, message="", level="warn"):
        '''
        Return this when the directory creation fails.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.createDirCode

    # Variable errors
    def wrong_type(self, message="", level="warn"):
        '''
        Return this when the variable has wrong type.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.wrongTypeCode

    def not_set(self, message="", level="warn"):
        '''
        Return this when the variable is not set.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.notSetCode

    # Subprocess errors
    def subprocess(self, message="", level="warn"):
        '''
        Return this when there is a problem runing a subprocess.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.subprocessCode

    # Molecules errors
    def parseMolecule(self, message="", level="warn"):
        '''
        Return this when a molecule could not be parsed.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.parseMoleculeCode

    def malformedMolecule(self, message="", level="warn"):
        '''
        Return this when a molecule is malformed.
        Input:
          message [string] DEFAULT: ""     - Message to be shown.
          level   [string] DEFAULT: "warn" - Type of message "warn" or "error".
        Return:
          -
        '''
        self.__print_msg(message, level)
        return self.malformedMoleculeCode

    # Debug functions
    def print_attributes(self):
        '''
        Print the class attributes.
        Input:
          -
        Return:
          -
        '''
        print(f"\t+------------------------------------------+")
        print(f"\t|          OCDocker Return codes           |")
        print(f"\t+------------------------------------------+")

        print(f"\n\t~~~~~~~~~~~~~~ GENERAL ERRORS ~~~~~~~~~~~~~~")
        print(f"\t - No error:                    {self.okCode}")
        print(f"\t - Unknown error:               {self.unkownCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~ FILE ERRORS ~~~~~~~~~~~~~~~")
        print(f"\t - File exists:                 {self.fileExistsCode}")
        print(f"\t - File does not exists:        {self.fileDoNotExistCode}")
        print(f"\t - Read file error:             {self.readFileCode}")
        print(f"\t - Write file error:            {self.writeFileCode}")
        print(f"\t - Untar error:                 {self.untarFileCode}")
        print(f"\t - Unsupported extension error: {self.unsupportedExtensionCode}")

        print(f"\n\t~~~~~~~~~~~~~ DIRECTORY ERRORS ~~~~~~~~~~~~~")
        print(f"\t - Directory creation error:    {self.createDirCode}")

        print(f"\n\t~~~~~~~~~~~~~ VARIABLE ERRORS ~~~~~~~~~~~~~~")
        print(f"\t - Wrong type:                  {self.wrongTypeCode}")

        print(f"\n\t~~~~~~~~~~~~~~ PROCESS ERRORS ~~~~~~~~~~~~~~")
        print(f"\t - Subprocess error:            {self.subprocessCode}")

        print(f"\n\t~~~~~~~~~~~~~ MOLECULE ERRORS ~~~~~~~~~~~~~~")
        print(f"\t - Molecule error:              {self.moleculeParseCode}")

        return

# Functions
###############################################################################
## Private ##

## Public ##
