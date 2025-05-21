#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to make all return codes in OCDocker
standard.

They are imported as:

import OCDocker.Error as ocerror
'''

# Imports
###############################################################################
import inspect
import datetime

from enum import IntEnum
from typing import Any, Callable, Dict, Tuple, Union

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
class ErrorMeta(type):
    """ Metaclass to add error methods to the Error class. """

    def __new__(cls, name: str, bases: Tuple[Any], attrs: Dict[str, Any]):
        ''' Add error methods to the Error class.

        Parameters
        ----------
        name : string
            The name of the class.
        bases : tuple
            The base classes of the class.
        attrs : dict
            The attributes of the class.
        '''

        # Test if the class is the Error class
        if name == "Error":
            new_class = super().__new__(cls, name, bases, attrs)
            new_class._add_error_methods() # type: ignore
            return new_class
        return None

class ErrorCode(IntEnum):
    """ Class with all error codes used in OCDocker. """

    # Common errors
    OK = 0
    ABORT = 1
    SKIP = 2
    UNKNOWN = -666

    # File errors
    FILE_EXISTS = 100
    FILE_NOT_EXIST = 101
    READ_FILE = 102
    WRITE_FILE = 103
    UNTAR_FILE = 104
    UNSUPPORTED_EXTENSION = 105
    BROKEN_PIPE = 106
    EMPTY_FILE = 107
    CORRUPTED_FILE = 108

    # Directory errors
    DIR_EXISTS = 150
    CREATE_DIR = 151
    REMOVE_DIR = 152
    DIR_NOT_EXIST = 153
    UNALLOWED_DIR = 154
    EMPTY_DIR = 155

    # Variable errors
    WRONG_TYPE = 200
    NOT_SET = 201
    EMPTY = 202
    VALUE_ERROR = 203

    # Subprocess errors
    SUBPROCESS = 300

    # Molecule error
    PARSE_MOLECULE = 400
    MALFORMED_MOLECULE = 401
    LIGAND_NOT_PREPARED = 402
    RECEPTOR_NOT_PREPARED = 403
    INVALID_MOLECULE_NAME = 404

    # Docking error
    DOCKING_OBJECT_NOT_GENERATED = 500
    RECEPTOR_OR_LIGAND_NOT_GENERATED = 501
    RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST = 502
    NOT_SUPPORTED_DOCKING_ALGORITHM = 503
    BINDING_SITE_NOT_FOUND = 504
    DOCKING_FAILED = 505
    READ_DOCKING_LOG_ERROR = 506

    # Archive error
    NOT_SUPPORTED_ARCHIVE = 600

    # Scoring and rescoring error
    UNSUPPORTED_SCORING_FUNCTION = 700
    RESCORING_FAILED = 701
    MISSING_ODDT_MODELS = 702

    # Clustering error
    UNSUPPORTED_CLUSTERING_ALGORITHM = 750
    CLUSTER_NOT_CONVERGED = 751
    EMPTY_CLUSTER = 752

    # Database error
    DATABASE_NOT_CONNECTED = 800
    DATABASE_NOT_CREATED = 801
    ENGINE_NOT_CREATED = 802
    SESSION_NOT_CREATED = 803
    DATA_NOT_FOUND = 804
    DATA_ALREADY_EXISTS = 805
    MALFORMED_PAYLOAD = 806

class ReportLevel(IntEnum):
    """ Class with all report levels used in OCDocker. """ 

    DEBUG = 5
    SUCCESS = 4
    INFO = 3
    WARNING = 2
    ERROR = 1
    NONE = 0

class ErrorMethodFactory:
    """ Factory to create methods to report errors. """

    @staticmethod
    def create_error_method(code: ErrorCode, description: str, default_level: ReportLevel) -> Callable:
        ''' Create a method to report an error based on the given code.

        Parameters
        ----------
        code : ErrorCode
            The error code.
        description : string
            The description of the error.
        default_level : ReportLevel
            The default level of the message to be printed, options are:
                - ReportLevel.DEBUG
                - ReportLevel.SUCCESS
                - ReportLevel.INFO
                - ReportLevel.WARNING
                - ReportLevel.ERROR
                - ReportLevel.NONE

        Returns
        -------
        Callable
            The method to report the error.
        '''
        
        # Creating the method
        def error_method(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
            '''{docstring}'''

            # If the level is not specified, use the default level
            return Error.report(code, message, level or default_level)

        # Creating dynamic docstring
        error_method.__doc__ = f" Return this when {description}.\n\n        Parameters\n        ----------\n        message : string, optional\n            Message to be printed. Default is \"\".\n        level : ReportLevel, optional\n            Level of message to be printed. Default is ReportLevel.{default_level.name}.\n\n        Returns\n        -------\n        int\n            The code for this error ({code})."
        
        return error_method

class ErrorMessages:
    messages = {
        # Common errors
        ErrorCode.OK: ("no error appears", ReportLevel.SUCCESS),
        ErrorCode.ABORT: ("the process has been aborted", ReportLevel.WARNING),
        ErrorCode.SKIP: ("the process has been skipped", ReportLevel.INFO),
        ErrorCode.UNKNOWN: ("an unknown error has occurred", ReportLevel.ERROR),

        # File errors
        ErrorCode.FILE_EXISTS: ("the file already exists", ReportLevel.WARNING),
        ErrorCode.FILE_NOT_EXIST: ("the file does not exist", ReportLevel.ERROR),
        ErrorCode.READ_FILE: ("error reading from file", ReportLevel.ERROR),
        ErrorCode.WRITE_FILE: ("error writing to file", ReportLevel.ERROR),
        ErrorCode.UNTAR_FILE: ("error extracting the file", ReportLevel.ERROR),
        ErrorCode.UNSUPPORTED_EXTENSION: ("the file extension is not supported", ReportLevel.ERROR),
        ErrorCode.BROKEN_PIPE: ("a broken pipe error has occurred", ReportLevel.ERROR),
        ErrorCode.EMPTY_FILE: ("the file is empty", ReportLevel.WARNING),
        ErrorCode.CORRUPTED_FILE: ("the file is corrupted", ReportLevel.ERROR),

        # Directory errors
        ErrorCode.DIR_EXISTS: ("the directory already exists", ReportLevel.WARNING),
        ErrorCode.CREATE_DIR: ("the directory could not be created", ReportLevel.ERROR),
        ErrorCode.REMOVE_DIR: ("the directory could not be removed", ReportLevel.ERROR),
        ErrorCode.DIR_NOT_EXIST: ("the directory does not exist", ReportLevel.ERROR),
        ErrorCode.UNALLOWED_DIR: ("access to the directory is not allowed", ReportLevel.ERROR),
        ErrorCode.EMPTY_DIR: ("the directory is empty", ReportLevel.ERROR),

        # Variable errors
        ErrorCode.WRONG_TYPE: ("the variable has the wrong type", ReportLevel.ERROR),
        ErrorCode.NOT_SET: ("the variable has not been set", ReportLevel.ERROR),
        ErrorCode.EMPTY: ("the variable is empty", ReportLevel.WARNING),
        ErrorCode.VALUE_ERROR: ("the variable has a value error", ReportLevel.ERROR),

        # Subprocess errors
        ErrorCode.SUBPROCESS: ("there was a problem running a subprocess", ReportLevel.ERROR),

        # Molecule errors
        ErrorCode.PARSE_MOLECULE: ("a molecule could not be parsed", ReportLevel.WARNING),
        ErrorCode.MALFORMED_MOLECULE: ("a molecule is malformed", ReportLevel.WARNING),
        ErrorCode.LIGAND_NOT_PREPARED: ("a ligand could not be prepared", ReportLevel.WARNING),
        ErrorCode.RECEPTOR_NOT_PREPARED: ("a receptor could not be prepared", ReportLevel.WARNING),
        ErrorCode.INVALID_MOLECULE_NAME: ("a molecule has an invalid name", ReportLevel.ERROR),

        # Docking errors
        ErrorCode.DOCKING_OBJECT_NOT_GENERATED: ("a docking object has not been generated", ReportLevel.WARNING),
        ErrorCode.RECEPTOR_OR_LIGAND_NOT_GENERATED: ("a receptor or ligand object has not been generated", ReportLevel.WARNING),
        ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST: ("a receptor or ligand has no descriptor file", ReportLevel.WARNING),
        ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM: ("the docking algorithm is not supported", ReportLevel.ERROR),
        ErrorCode.BINDING_SITE_NOT_FOUND: ("the binding site has not been found", ReportLevel.ERROR),
        ErrorCode.DOCKING_FAILED: ("the docking run has failed", ReportLevel.ERROR),
        ErrorCode.READ_DOCKING_LOG_ERROR: ("the docking log had problems being read", ReportLevel.ERROR),

        # Archive error
        ErrorCode.NOT_SUPPORTED_ARCHIVE: ("the archive format is not supported", ReportLevel.ERROR),

        # Scoring and rescoring errors
        ErrorCode.UNSUPPORTED_SCORING_FUNCTION: ("the scoring function is not supported", ReportLevel.ERROR),
        ErrorCode.RESCORING_FAILED: ("the rescoring process has failed", ReportLevel.ERROR),
        ErrorCode.MISSING_ODDT_MODELS: ("no ODDt models are available", ReportLevel.ERROR),

        # Clustering errors
        ErrorCode.UNSUPPORTED_CLUSTERING_ALGORITHM: ("an unsupported clustering algorithm is specified", ReportLevel.ERROR),
        ErrorCode.CLUSTER_NOT_CONVERGED: ("the clustering process has not converged", ReportLevel.ERROR),
        ErrorCode.EMPTY_CLUSTER: ("the cluster is empty", ReportLevel.ERROR),

        # Database errors
        ErrorCode.DATABASE_NOT_CONNECTED: ("the database is not connected", ReportLevel.ERROR),
        ErrorCode.DATABASE_NOT_CREATED: ("the database has not been created", ReportLevel.ERROR),
        ErrorCode.ENGINE_NOT_CREATED: ("the engine has not been created", ReportLevel.ERROR),
        ErrorCode.SESSION_NOT_CREATED: ("the session has not been created", ReportLevel.ERROR),
        ErrorCode.DATA_NOT_FOUND: ("the data has not been found", ReportLevel.ERROR),
        ErrorCode.DATA_ALREADY_EXISTS: ("the data already exists", ReportLevel.ERROR),
        ErrorCode.MALFORMED_PAYLOAD: ("the payload is malformed", ReportLevel.ERROR),
    }

class Error(metaclass = ErrorMeta):
    '''Class to handle errors and standarize them across the whole code.'''

    # Class attributes
    output_level = ReportLevel.INFO

    color = {
        ReportLevel.INFO: "\033[1;96m",
        ReportLevel.SUCCESS: "\033[1;92m",
        ReportLevel.WARNING: "\033[1;93m",
        ReportLevel.ERROR: "\033[1;91m",
        ReportLevel.DEBUG: "\033[1;95m",
    }

    @classmethod
    def _add_error_methods(cls):
        ''' Add error methods to the Error class.
        '''
        
        # Iterate through the ErrorCode enumeration
        for code in ErrorCode:
            # Get the description and default level of the error code from ErrorMessages
            description, level = ErrorMessages.messages[code]
            
            # Create the dynamic method for each error code
            error_method = ErrorMethodFactory.create_error_method(code, description, level)

            # Convert the method into a static method
            static_error_method = staticmethod(error_method)

            # Add the static method to the Error class
            setattr(cls, f"{code.name.lower()}", static_error_method)

    @classmethod
    def set_output_level(cls, level: Union[ReportLevel, int]):
        ''' Set the output level of the error messages.

        Parameters
        ----------
        level : ReportLevel or int
            The level of the messages to be printed, options are:
                - ReportLevel.DEBUG   (5)
                - ReportLevel.SUCCESS (4)
                - ReportLevel.INFO    (3)
                - ReportLevel.WARNING (2)
                - ReportLevel.ERROR   (1)
                - ReportLevel.NONE    (0)
        '''
        
        # If the level is a ReportLevel, just set it
        if isinstance(level, ReportLevel):
            cls.output_level = level
            return None
        elif isinstance(level, int):
            # If the level is an int, check if it is valid
            if level >= ReportLevel.NONE and level <= ReportLevel.DEBUG:
                cls.output_level = ReportLevel(level)
                return None
            else:
                raise ValueError(f"Invalid output level: {level}.")
        else:
            raise TypeError(f"Invalid type for output level: {type(level)}.")

    @classmethod
    def get_output_level(cls):
        return cls.output_level

    @staticmethod
    def get_time(level: ReportLevel = ReportLevel.NONE) -> str:
        ''' Get the current time.

        Parameters
        ----------
        level : ReportLevel, optional
            The level of the message to be printed, options are:
                - ReportLevel.DEBUG
                - ReportLevel.SUCCESS
                - ReportLevel.INFO
                - ReportLevel.WARNING
                - ReportLevel.ERROR
                - ReportLevel.NONE

        Returns
        -------
        string
            The current time in the format 'dd-mm-YYYY|HH:MM:SS'.
        '''

        # Get the current time
        today = datetime.datetime.now()

        # Return the current time
        return f"\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m"

    ## Private ##

    ## Public ##
    @staticmethod
    def print_message(message: str, level: ReportLevel) -> None:
        ''' Print a message with a specific level.

        Parameters
        ----------
        message : string
            The message to be printed.
        level : ReportLevel
            The level of the message to be printed, options are:
                - ReportLevel.DEBUG
                - ReportLevel.SUCCESS
                - ReportLevel.INFO
                - ReportLevel.WARNING
                - ReportLevel.ERROR
        '''

        # If there is no message, return
        if not message:
            return None

        # Get the color for the level
        setcolor = Error.color.get(level, '\033[1;0m')

        # Get the current time
        time_str = Error.get_time(level)
        base_message = f"[{time_str}] {setcolor}{level.name}\033[1;0m: {message}"

        if Error.output_level >= ReportLevel.DEBUG:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back.f_back.f_back # type: ignore
            detailed_message = (f"In function '{caller_frame.f_code.co_name}' " # type: ignore
                                f"line {caller_frame.f_lineno} " # type: ignore
                                f"from file '{caller_frame.f_code.co_filename}'.") # type: ignore
            print(f"{base_message} {detailed_message}")
        else:
            print(f"{base_message}")
        
        return None

    @staticmethod
    def report(code: ErrorCode, message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        '''Report an error based on the given code.

        Parameters
        ----------
        code : ErrorCode
            The error code.
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The integer value of the error code.
        '''

        Error.print_message(message, level)
        return code.value

    # Debug functions
    @staticmethod
    def print_attributes() -> None:
        ''' Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        
        # Mapping sections to their corresponding attributes and codes
        error_sections = {
            "GENERAL ERRORS": [
                ("No error", ErrorCode.OK),
                ("Abortion", ErrorCode.ABORT),
                ("Skip", ErrorCode.SKIP),
                ("Unknown error", ErrorCode.UNKNOWN),
            ],
            "FILE ERRORS": [
                ("File exists", ErrorCode.FILE_EXISTS),
                ("File does not exist", ErrorCode.FILE_NOT_EXIST),
                ("Read file error", ErrorCode.READ_FILE),
                ("Write file error", ErrorCode.WRITE_FILE),
                ("Untar error", ErrorCode.UNTAR_FILE),
                ("Unsupported extension", ErrorCode.UNSUPPORTED_EXTENSION),
                ("Broken PIPE", ErrorCode.BROKEN_PIPE),
                ("Empty file", ErrorCode.EMPTY_FILE),
                ("Corrupted file", ErrorCode.CORRUPTED_FILE),
            ],
            "DIRECTORY ERRORS": [
                ("Directory exists", ErrorCode.DIR_EXISTS),
                ("Directory creation error", ErrorCode.CREATE_DIR),
                ("Directory remotion error", ErrorCode.REMOVE_DIR),
                ("Directory does not exist", ErrorCode.DIR_NOT_EXIST),
                ("Directory access not allowed", ErrorCode.UNALLOWED_DIR),
            ],
            "VARIABLE ERRORS": [
                ("Wrong type", ErrorCode.WRONG_TYPE),
                ("Not set", ErrorCode.NOT_SET),
                ("Empty", ErrorCode.EMPTY),
                ("Value error", ErrorCode.VALUE_ERROR),
            ],
            "PROCESS ERRORS": [
                ("Subprocess error", ErrorCode.SUBPROCESS),
            ],
            "MOLECULE ERRORS": [
                ("Molecule parse error", ErrorCode.PARSE_MOLECULE),
                ("Malformed molecule", ErrorCode.MALFORMED_MOLECULE),
                ("Ligand not prepared", ErrorCode.LIGAND_NOT_PREPARED),
                ("Receptor not prepared", ErrorCode.RECEPTOR_NOT_PREPARED),
                ("Invalid molecule name", ErrorCode.INVALID_MOLECULE_NAME),
            ],
            "DOCKING ERRORS": [
                ("Docking Object Not Generated", ErrorCode.DOCKING_OBJECT_NOT_GENERATED),
                ("Receptor or Ligand Not Generated", ErrorCode.RECEPTOR_OR_LIGAND_NOT_GENERATED),
                ("Receptor or Ligand Descriptor Does Not Exist", ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST),
                ("Not Supported Docking Algorithm", ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM),
                ("Binding Site Not Found", ErrorCode.BINDING_SITE_NOT_FOUND),
                ("Docking Failed", ErrorCode.DOCKING_FAILED),
                ("Read Docking Log Error", ErrorCode.READ_DOCKING_LOG_ERROR),
            ],
            "ARCHIVE ERRORS": [
                ("Not Supported Archive", ErrorCode.NOT_SUPPORTED_ARCHIVE),
            ],
            "SCORING AND RESCORING ERRORS": [
                ("Unsupported Scoring Function", ErrorCode.UNSUPPORTED_SCORING_FUNCTION),
                ("Rescoring Failed", ErrorCode.RESCORING_FAILED),
                ("Missing ODDt Models", ErrorCode.MISSING_ODDT_MODELS),
            ],
            "CLUSTERING ERRORS": [
                ("Unsupported Clustering Algorithm", ErrorCode.UNSUPPORTED_CLUSTERING_ALGORITHM),
                ("Cluster Not Converged", ErrorCode.CLUSTER_NOT_CONVERGED),
            ],
            "DATABASE ERRORS": [
                ("Database Not Connected", ErrorCode.DATABASE_NOT_CONNECTED),
                ("Database Not Created", ErrorCode.DATABASE_NOT_CREATED),
                ("Engine Not Created", ErrorCode.ENGINE_NOT_CREATED),
                ("Session Not Created", ErrorCode.SESSION_NOT_CREATED),
                ("Data Not Found", ErrorCode.DATA_NOT_FOUND),
                ("Data Already Exists", ErrorCode.DATA_ALREADY_EXISTS),
                ("Malformed Payload", ErrorCode.MALFORMED_PAYLOAD),
            ],
        }

        # Print header
        print(f"\t+----------------------------------------------+")
        print(f"\t|            OCDocker Return codes             |")
        print(f"\t+----------------------------------------------+")

        # Iterate and print each section and its attributes
        for section_name, errors in error_sections.items():
            print(f"\n\t~~~~~~~~~~~~~~~~ {section_name} ~~~~~~~~~~~~~~~~")
            for error_description, error_code in errors:
                print(f"\t - {error_description}: {error_code}")

        return None
