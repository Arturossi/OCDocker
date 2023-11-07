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
from typing import Union

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
class ErrorCode(IntEnum):
    '''Class with all error codes used in OCDocker.'''

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

    # Directory errors
    DIR_EXISTS = 150
    CREATE_DIR = 151
    REMOVE_DIR = 152
    DIR_NOT_EXIST = 153
    UNALLOWED_DIR = 154

    # Variable errors
    WRONG_TYPE = 200
    NOT_SET = 201
    EMPTY = 202
    VALUE_ERROR = 203

    # Subprocess errors
    SUBPROCESS_ERROR = 300

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

class ReportLevel(IntEnum):
    DEBUG = 5
    SUCCESS = 4
    INFO = 3
    WARNING = 2
    ERROR = 1
    NONE = 0

class Error:
    '''Class to handle errors and standarize them across the whole code.'''

    output_level = ReportLevel.INFO

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
    def get_time():
        return datetime.datetime.now().strftime('%d-%m-%Y|%H:%M:%S')

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

        color = {
            ReportLevel.INFO: '1;96',
            ReportLevel.SUCCESS: '1;92',
            ReportLevel.WARNING: '1;93',
            ReportLevel.ERROR: '1;91',
            ReportLevel.DEBUG: '1;95',
        }.get(level, '1;0')

        time_str = Error.get_time()
        base_message = f"[{time_str}] {level.name}: {message}"

        if Error.output_level >= ReportLevel.DEBUG:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back.f_back.f_back # type: ignore
            detailed_message = (f"In function '{caller_frame.f_code.co_name}' " # type: ignore
                                f"line {caller_frame.f_lineno} " # type: ignore
                                f"from file '{caller_frame.f_code.co_filename}'.") # type: ignore
            print(f"\033[{color}m{base_message} {detailed_message}\033[1;0m")
        else:
            print(f"\033[{color}m{base_message}\033[1;0m")

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
                ("Subprocess error", ErrorCode.SUBPROCESS_ERROR),
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
    
    # Common errors
    @staticmethod
    def ok(message: str = "") -> int:
        ''' Return this when no error appears.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".

        Returns
        -------
        int
            The code for ok (0).
        '''

        return Error.report(ErrorCode.OK, message, ReportLevel.SUCCESS)

    @staticmethod
    def abort(message: str = "") -> int:
        ''' Return this when the process has been aborted.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".

        Returns
        -------
        int
            The code for abort (1).
        '''

        return Error.report(ErrorCode.ABORT, message, ReportLevel.WARNING)

    @staticmethod
    def skip(message: str = "", level: ReportLevel = ReportLevel.INFO) -> int:
        ''' Return this when the process has been skipped.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.INFO.

        Returns
        -------
        int
            The code for skip (2).
        '''

        return Error.report(ErrorCode.SKIP, message, level)

    @staticmethod
    def unknown(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the error is unknown.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for unknown error (-666).
        '''

        return Error.report(ErrorCode.UNKNOWN, message, level)

    # File errors
    @staticmethod
    def file_exists(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return when the file already exists.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for file exists error (100).
        '''

        return Error.report(ErrorCode.FILE_EXISTS, message, level)

    @staticmethod
    def file_do_not_exist(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the file do not exist.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for file do not exist error (101).
        '''

        return Error.report(ErrorCode.FILE_NOT_EXIST, message, level)

    @staticmethod
    def read_file(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a file could not be read.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for read file error (102).
        '''

        return Error.report(ErrorCode.READ_FILE, message, level)

    @staticmethod
    def write_file(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a file could not be written.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for write file error (103).
        '''

        return Error.report(ErrorCode.WRITE_FILE, message, level)

    @staticmethod
    def untar_file(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the untar action fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for untar file error (104).
        '''

        return Error.report(ErrorCode.UNTAR_FILE, message, level)

    @staticmethod
    def unsupported_extension(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the extension is not supported.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for unsupported extension error (105).
        '''

        return Error.report(ErrorCode.UNSUPPORTED_EXTENSION, message, level)

    @staticmethod
    def broken_pipe(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a broken pipe occurs.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for broken pipe error (106).
        '''

        return Error.report(ErrorCode.BROKEN_PIPE, message, level)

    @staticmethod
    def empty_file(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the file is empty.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for empty file error (107).
        '''

        return Error.report(ErrorCode.EMPTY_FILE, message, level)

    # Directory errors
    @staticmethod
    def dir_exists(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the directory already exists.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for directory already exists error (150).
        '''
        return Error.report(ErrorCode.DIR_EXISTS, message, level)

    @staticmethod
    def create_dir(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the directory creation fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for directory creation error (151).
        '''

        return Error.report(ErrorCode.CREATE_DIR, message, level)

    @staticmethod
    def remove_dir(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the directory removal fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for directory removal error (152).
        '''

        return Error.report(ErrorCode.REMOVE_DIR, message, level)

    @staticmethod
    def dir_does_not_exist(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the directory does not exist.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for directory does not exist error (153).
        '''

        return Error.report(ErrorCode.DIR_NOT_EXIST, message, level)

    @staticmethod
    def unnalowed_dir(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the accessed directory is not allowed for any reason.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for directory unallowed (154).
        '''

        return Error.report(ErrorCode.UNALLOWED_DIR, message, level)

    # Variable errors
    @staticmethod
    def wrong_type(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the variable has the wrong type.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for wrong type error (200).
        '''

        return Error.report(ErrorCode.WRONG_TYPE, message, level)

    @staticmethod
    def not_set(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the variable is not set.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for not set error (201).
        '''

        return Error.report(ErrorCode.NOT_SET, message, level)

    @staticmethod
    def empty(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the variable is empty.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for empty error (202).
        '''

        return Error.report(ErrorCode.EMPTY, message, level)

    @staticmethod
    def value_error(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when the variable has a value error.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for value error (203).
        '''

        return Error.report(ErrorCode.VALUE_ERROR, message, level)

    # Subprocess errors
    @staticmethod
    def subprocess(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when there is a problem running a subprocess.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for subprocess error (300).
        '''

        return Error.report(ErrorCode.SUBPROCESS_ERROR, message, level)

    # Molecules errors
    @staticmethod
    def parse_molecule(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a molecule could not be parsed.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.
        
        Returns
        -------
        int
            The code for parse molecule error (400).
        '''

        return Error.report(ErrorCode.PARSE_MOLECULE, message, level)

    @staticmethod
    def malformed_molecule(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a molecule is malformed.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for malformed molecule error (401).
        '''

        return Error.report(ErrorCode.MALFORMED_MOLECULE, message, level)

    @staticmethod
    def ligand_not_prepared(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a ligand could not be prepared.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for ligand not prepared error (402).
        '''

        return Error.report(ErrorCode.LIGAND_NOT_PREPARED, message, level)

    @staticmethod
    def receptor_not_prepared(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a receptor could not be prepared.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for receptor not prepared error (403).
        '''

        return Error.report(ErrorCode.RECEPTOR_NOT_PREPARED, message, level)

    @staticmethod
    def invalid_molecule_name(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when a molecule has an invalid name.

        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.

        Returns
        -------
        int
            The code for invalid molecule name error (404).
        '''

        return Error.report(ErrorCode.INVALID_MOLECULE_NAME, message, level)
    
    # Docking errors
    @staticmethod
    def docking_object_not_generated(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a docking object has not been generated.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for docking object not generated error (500).
        '''

        return Error.report(ErrorCode.DOCKING_OBJECT_NOT_GENERATED, message, level)

    @staticmethod
    def receptor_or_ligand_not_generated(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a receptor or ligand object has not been generated.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for receptor or ligand not generated error (501).
        '''

        return Error.report(ErrorCode.RECEPTOR_OR_LIGAND_NOT_GENERATED, message, level)

    @staticmethod
    def receptor_or_ligand_descriptor_does_not_exist(message: str = "", level: ReportLevel = ReportLevel.WARNING) -> int:
        ''' Return this when a receptor or ligand has no descriptor file.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.WARNING.

        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (502).
        '''

        return Error.report(ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST, message, level)

    @staticmethod
    def not_supported_docking_algorithm(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the docking algorithm is not supported.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.

        Returns
        -------
        int
            The code for not supported docking algorithm error (503).
        '''

        return Error.report(ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM, message, level)

    @staticmethod
    def binding_site_not_found(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the binding site has not been found.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.

        Returns
        -------
        int
            The code for binding site not found error (503).
        '''

        return Error.report(ErrorCode.BINDING_SITE_NOT_FOUND, message, level)

    @staticmethod
    def docking_failed(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the docking run has failed.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.

        Returns
        -------
        int
            The code for docking failed error (503).
        '''

        return Error.report(ErrorCode.DOCKING_FAILED, message, level)

    @staticmethod
    def read_docking_log_error(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the docking log had problems to be read.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.

        Returns
        -------
        int
            The code for read docking log error (503).
        '''

        return Error.report(ErrorCode.READ_DOCKING_LOG_ERROR, message, level)

    # Archive errors
    @staticmethod
    def not_supported_archive(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the archive format is not supported.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for not supported archive error.
        '''

        return Error.report(ErrorCode.NOT_SUPPORTED_ARCHIVE, message, level)

    # Scoring and rescoring errors
    @staticmethod
    def unsupported_scoring_function(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the scoring function is not supported.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for unsupported scoring function error.
        '''

        return Error.report(ErrorCode.UNSUPPORTED_SCORING_FUNCTION, message, level)

    @staticmethod
    def rescoring_failed(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the rescoring process has failed.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for rescoring failed error.
        '''

        return Error.report(ErrorCode.RESCORING_FAILED, message, level)

    @staticmethod
    def missing_oddt_models(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when no ODDt models are available.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for missing ODDt models error.
        '''

        return Error.report(ErrorCode.MISSING_ODDT_MODELS, message, level)

    @staticmethod
    def unsupported_clustering_algorithm(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when an unsupported clustering algorithm is specified.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for unsupported clustering algorithm error.
        '''

        return Error.report(ErrorCode.UNSUPPORTED_CLUSTERING_ALGORITHM, message, level)

    @staticmethod
    def cluster_not_converged(message: str = "", level: ReportLevel = ReportLevel.ERROR) -> int:
        ''' Return this when the clustering process has not converged.
        
        Parameters
        ----------
        message : str, optional
            Message to be printed. Default is "".
        level : ReportLevel, optional
            Level of message to be printed. Default is ReportLevel.ERROR.
        
        Returns
        -------
        int
            The code for cluster not converged error.
        '''

        return Error.report(ErrorCode.CLUSTER_NOT_CONVERGED, message, level)


# Functions
###############################################################################
## Private ##

## Public ##
