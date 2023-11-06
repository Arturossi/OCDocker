#!/usr/lib/python3

# Description
###############################################################################
'''
Sets of classes and functions that are used to make all return codes in OCDocker
standard. TODO: FINISH THIS

They are imported as:

import OCDocker.Error as ocerror
'''

# Imports
###############################################################################
import argparse
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

# Classes
###############################################################################
class Error:
    """Class to handle errors and standarize them across the whole code."""

    def __init__(self, output_level: int) -> None:
        '''Constructor for the Error class.
        
        Parameters
        ----------
        output_level : int
            The output level of the program. 
            - 0 is only errors
            - 1 is errors and warnings
            - 2 is errors, warnings and info
            - 3 is errors, warnings, info and success
            - 4 is errors, warnings, info, success and debug
        
        Returns
        -------
        None
    
        '''

        # OCDocker arguments
        self.output_level = output_level

        # Common errors
        self.okCode                             = 0
        self.abortCode                          = 1
        self.skipCode                           = 2
        self.unknownCode                        = -666

        # File errors
        self.fileExistsCode                     = 100
        self.fileDoNotExistCode                 = 101
        self.readFileCode                       = 102
        self.writeFileCode                      = 103
        self.untarFileCode                      = 104
        self.unsupportedExtensionCode           = 105
        self.brokenPipeCode                     = 106
        self.emptyFileCode                      = 107

        # Directory errors
        self.dirExistsCode                      = 150
        self.createDirCode                      = 151
        self.removeDirCode                      = 152
        self.dirDoesNotExistsCode               = 153
        self.dirUnallowedCode                   = 154

        # Variable errors
        self.wrongTypeCode                      = 200
        self.notSetCode                         = 201
        self.emptyCode                          = 202
        self.valueErrorCode                     = 203

        # Subprocess errors
        self.subprocessCode                     = 300

        # Molecule error
        self.parseMoleculeCode                  = 400
        self.malformedMoleculeCode              = 401
        self.ligandNotPreparedCode              = 402
        self.receptorNotPreparedCode            = 403
        self.invalidMoleculeName                = 404

        # Docking error
        self.dockingObjectNotGeneratedCode      = 500
        self.recLigObjectNotGeneratedCode       = 501
        self.recLigFileDoesNotExistCode         = 502
        self.notSupportedDockingAlgorithmCode   = 503
        self.bindingSiteNotFoundCode            = 504
        self.dockingFailedCode                  = 505
        self.readDockingLogError                = 506

        # Archive error
        self.notSupportedArchiveCode            = 600

        # Scoring and rescoring error
        self.unsupportedScoringFunctionCode     = 700
        self.rescoringFailedCode                = 701
        self.missingOddtModel                   = 702

        # Clustering error
        self.unsupportedClusteringAlgorithmCode = 750
        self.clusterNotConvergedCode            = 751

    ## Private ##
    def __print_info(self, message: str) -> None:
        '''Function to print info.

        Parameters
        ----------
        message : str
            Message to be printed.

        Returns
        -------
        None
        '''

        today = datetime.datetime.now()
        if self.output_level >= 4:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;96mINFO\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;96mINFO\033[1;0m: {message}")
        return None

    def __print_success(self, message: str) -> None:
        '''Print success.

        Parameters
        ----------
        message : str
            Message to be printed.

        Returns
        -------
        None
        '''

        today = datetime.datetime.now()
        if self.output_level >= 4:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;92mSUCCESS\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;92mSUCCESS\033[1;0m: {message}")

        return None

    def __print_warning(self, message: str) -> None:
        '''Function to print warning.

        Parameters
        ----------
        message : str
            Message to be printed.

        Returns
        -------
        None
        '''
        
        today = datetime.datetime.now()
        if self.output_level >= 3:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;93mWARNING\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;93mWARNING\033[1;0m: {message}")

        return None

    def __print_error(self, message: str) -> None:
        '''Print error.

        Parameters
        ----------
        message : string
            Message to be printed.

        Returns
        -------
        None
        '''

        today = datetime.datetime.now()
        if self.output_level >= 3:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;91mERROR\033[1;0m: {message} In function '{inspect.currentframe().f_back.f_back.f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_back.f_back.f_lineno} from file '{inspect.currentframe().f_back.f_back.f_back.f_code.co_filename}'.") # type: ignore
        else:
            print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] \033[1;91mERROR\033[1;0m: {message}")

        return None

    def __print_msg(self, message: str = "", level: str = "warn") -> None:
        '''Prints a message based on level.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        None
        '''

        if message:
            if level == "warn" and self.output_level >= 1:
                self.__print_warning(message)
            elif level == "error" and self.output_level >= 0:
                self.__print_error(message)
            elif level == "success" and self.output_level >= 3:
                self.__print_success(message)
            elif level == "info" and self.output_level >= 2:
                self.__print_info(message)

        return None

    ## Public ##
    # Common errors
    def ok(self, message: str = "") -> int:
        '''Return this when no error appears.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".

        Returns
        -------
        int
            The code for ok (0).
        '''

        self.__print_msg(message, "success")

        return self.okCode

    def abort(self, message: str = "") -> int:
        '''Return this when process has been aborted.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".

        Returns
        -------
        int
            The code for abort (1).
        '''

        self.__print_msg(message, "warn")

        return self.okCode
    
    def skip(self, message: str = "", level: str = "info") -> int:
        '''Return this when process has been skipped.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'info'.

        Returns
        -------
        int
            The code for skip (2).
        '''

        self.__print_msg(message, level)

        return self.skipCode

    def unknown(self, message: str = "", level: str = "warn") -> int:
        '''Return when the error is unknown.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for unkown error (-666).
        '''

        self.__print_msg(message, level)

        return self.unknownCode

    # File errors
    def file_exists(self, message: str = "", level: str = "warn") -> int:
        '''Return when the file already exists.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for file exists error (100).
        '''

        self.__print_msg(message, level)

        return self.fileExistsCode

    def file_do_not_exist(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the file do not exist.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for file do not exist error (101).
        '''
        
        self.__print_msg(message, level)

        return self.fileDoNotExistCode

    def read_file(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a file could not be read.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for read file error (102).
        '''

        self.__print_msg(message, level)

        return self.readFileCode

    def write_file(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a file could not be written.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for write file error (103).
        '''

        self.__print_msg(message, level)

        return self.writeFileCode

    def untar_file(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the untar action fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for untar file error (104).
        '''

        self.__print_msg(message, level)

        return self.untarFileCode

    def unsupported_extension(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the extension is not supported.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for unsupported extension error (105).
        '''

        self.__print_msg(message, level)

        return self.unsupportedExtensionCode
    
    def broken_pipe(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a broken pipe occurs.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for broken pipe error (106).
        '''

        self.__print_msg(message, level)

        return self.brokenPipeCode
    
    def empty_file(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the file is empty.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for empty file error (107).
        '''

        self.__print_msg(message, level)

        return self.emptyFileCode

    # Directory errors
    def dir_exists(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the directory already exists.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for create directory error (150).
    
        '''

        self.__print_msg(message, level)

        return self.dirExistsCode

    def create_dir(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the directory creation fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for create directory error (151).
    
        '''

        self.__print_msg(message, level)

        return self.createDirCode

    def remove_dir(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the directory remotion fails.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for remove directory error (152).
    
        '''

        self.__print_msg(message, level)

        return self.removeDirCode

    def dir_does_not_exist(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the directory does not exists.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for directory does not exists error (153).
        '''

        self.__print_msg(message, level)

        return self.dirDoesNotExistsCode

    def unnalowed_dir(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the accessed dir is not allowed for any reason.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for directory unallowed (154).
        '''

        self.__print_msg(message, level)

        return self.dirUnallowedCode

    # Variable errors
    def wrong_type(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the variable has wrong type.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for wrong type error (200).
        '''

        self.__print_msg(message, level)

        return self.wrongTypeCode

    def not_set(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the variable is not set.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for not set error (201).
        '''

        self.__print_msg(message, level)

        return self.notSetCode

    def empty(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the variable is empty.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for empty error (202).
        '''

        self.__print_msg(message, level)

        return self.emptyCode

    def value_error(self, message: str = "", level: str = "warn") -> int:
        '''Return this when the variable has a value error.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for value error (203).
        '''

        self.__print_msg(message, level)

        return self.valueErrorCode

    # Subprocess errors
    def subprocess(self, message: str = "", level: str = "warn") -> int:
        '''Return this when there is a problem runing a subprocess.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for subprocess error (300).
    
        '''

        self.__print_msg(message, level)

        return self.subprocessCode

    # Molecules errors
    def parse_molecule(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a molecule could not be parsed.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for parse molecule error (400).
        '''

        self.__print_msg(message, level)

        return self.parseMoleculeCode

    def malformed_molecule(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a molecule is malformed.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for malformed molecule error (401).
        '''

        self.__print_msg(message, level)

        return self.malformedMoleculeCode

    def ligand_not_prepared(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a ligand could not be prepared.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for ligand not prepared error (402).
        '''

        self.__print_msg(message, level)

        return self.ligandNotPreparedCode

    def receptor_not_prepared(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a receptor could not be prepared.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for receptor not prepared error (403).
        '''

        self.__print_msg(message, level)

        return self.receptorNotPreparedCode

    def invalid_molecule_name(self, message: str = "", level: str = "error") -> int:
        '''Return this when me molecule has an invalid name.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for invalid molecule name error (404).
        '''

        self.__print_msg(message, level)

        return self.invalidMoleculeName
    
    # Docking errors
    def docking_object_not_generated(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a docking object has not been generated.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for docking object not generated error (500).
        '''

        self.__print_msg(message, level)

        return self.dockingObjectNotGeneratedCode

    def receptor_or_ligand_not_generated(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a receptor or ligand object has not been generated.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.

        Returns
        -------
        int
            The code for receptor or ligand not generated error (501).
        '''

        self.__print_msg(message, level)

        return self.recLigObjectNotGeneratedCode

    def receptor_or_ligand_descriptor_does_not_exist(self, message: str = "", level: str = "warn") -> int:
        '''Return this when a receptor or ligand has no descriptor file.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (502).
        '''

        self.__print_msg(message, level)

        return self.recLigFileDoesNotExistCode
    
    def not_supported_docking_algorithm(self, message: str = "", level: str = "error") -> int:
        '''Return this when the docking algorithm is not supported.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (503).
        '''

        self.__print_msg(message, level)

        return self.notSupportedDockingAlgorithmCode
    
    def binding_site_not_found(self, message: str = "", level: str = "error") -> int:
        '''Return this when the binding site has not been found.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (503).
        '''

        self.__print_msg(message, level)

        return self.bindingSiteNotFoundCode
    
    def docking_failed(self, message: str = "", level: str = "error") -> int:
        '''Return this when the docking run has failed.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (503).
        '''

        self.__print_msg(message, level)

        return self.dockingFailedCode
    
    def read_docking_log_error(self, message: str = "", level: str = "error") -> int:
        '''Return this when the docking log had problems to be read.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (503).
        '''

        self.__print_msg(message, level)

        return self.readDockingLogError

    # Archive errors
    def not_supported_archive(self, message: str = "", level: str = "error") -> int:
        '''Return this when the archive is not supported. NOTE: SHOULD be removed in the future.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (600).
        '''

        self.__print_msg(message, level)

        return self.notSupportedArchiveCode
    
    # Scoring and rescoring errors
    def unsupported_scoring_function(self, message: str = "", level: str = "error") -> int:
        '''Return this when the scoring function is not supported.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for receptor or ligand descriptor does not exist error (700).
        '''

        self.__print_msg(message, level)

        return self.unsupportedScoringFunctionCode
    
    def rescoring_failed(self, message: str = "", level: str = "error") -> int:
        '''Return this when the rescoring has failed.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional.
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for rescoring failed error (701).
        '''

        self.__print_msg(message, level)

        return self.rescoringFailedCode
    
    def missing_oddt_models(self, message: str = "", level: str = "error") -> int:
        '''Return this when no ODDt model has been found.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional.
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for rescoring failed error (702).
        '''

        self.__print_msg(message, level)

        return self.missingOddtModel
    
    def unsupported_clustering_algorithm(self, message: str = "", level: str = "error") -> int:
        '''Return this when the user has provied an unsupported clustering algorithm.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional.
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for unsupported clustering algorithm error (750).
        '''

        self.__print_msg(message, level)

        return self.unsupportedClusteringAlgorithmCode

    def cluster_not_converged(self, message: str = "", level: str = "error") -> int:
        '''Return this when the cluster has not converged.

        Parameters
        ----------
        message : string, optional
            Message to be printed. Default is "".
        level : string, optional.
            Level of message to be printed. Default is 'warn'. Other options are 'info', 'success', 'error'.
        
        Returns
        -------
        int
            The code for cluster not converged error (751).
        '''

        self.__print_msg(message, level)

        return self.clusterNotConvergedCode

    # Debug functions
    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        
        print(f"\t+----------------------------------------------+")
        print(f"\t|            OCDocker Return codes             |")
        print(f"\t+----------------------------------------------+")

        print(f"\n\t~~~~~~~~~~~~~~~~ GENERAL ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - No error:                        {self.okCode}")
        print(f"\t - Abortion:                        {self.abortCode}")
        print(f"\t - Skip:                            {self.skipCode}")
        print(f"\t - Unknown error:                   {self.unknownCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~~~ FILE ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - File exists:                     {self.fileExistsCode}")
        print(f"\t - File does not exists:            {self.fileDoNotExistCode}")
        print(f"\t - Read file error:                 {self.readFileCode}")
        print(f"\t - Write file error:                {self.writeFileCode}")
        print(f"\t - Untar error:                     {self.untarFileCode}")
        print(f"\t - Unsupported extension error:     {self.unsupportedExtensionCode}")
        print(f"\t - Broken PIPE error:               {self.brokenPipeCode}")
        print(f"\t - Empty file:                      {self.emptyFileCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ DIRECTORY ERRORS ~~~~~~~~~~~~~~~")
        print(f"\t - Directory exists:                {self.dirExistsCode}")
        print(f"\t - Directory creation error:        {self.createDirCode}")
        print(f"\t - Directory remotion error:        {self.removeDirCode}")
        print(f"\t - Directory does not exist:        {self.dirDoesNotExistsCode}")
        print(f"\t - Directory access not allowed:    {self.dirUnallowedCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ VARIABLE ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Wrong type:                      {self.wrongTypeCode}")
        print(f"\t - Not set:                         {self.notSetCode}")
        print(f"\t - Empty:                           {self.emptyCode}")
        print(f"\t - Value error:                     {self.valueErrorCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~ PROCESS ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Subprocess error:                {self.subprocessCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ MOLECULE ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Molecule parse error:            {self.parseMoleculeCode}")
        print(f"\t - Malformed molecule error:        {self.malformedMoleculeCode}")
        print(f"\t - Ligand not prepared:             {self.ligandNotPreparedCode}")
        print(f"\t - Receptor not prepared:           {self.receptorNotPreparedCode}")
        print(f"\t - Invalid molecule name:           {self.invalidMoleculeName}")

        print(f"\n\t~~~~~~~~~~~~~~~~ DOCKING ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - Docking Object Generation")
        print(f"\t   error:                           {self.dockingObjectNotGeneratedCode}")
        print(f"\t - Receptor/Ligand Object")
        print(f"\t   Generation error:                {self.recLigObjectNotGeneratedCode}")
        print(f"\t - Receptor/Ligand File")
        print(f"\t   descriptor does not exist:       {self.recLigFileDoesNotExistCode}")
        print(f"\t - Not supported docking algoritm:  {self.notSupportedDockingAlgorithmCode}")
        print(f"\t - Binding site not found:          {self.bindingSiteNotFoundCode}")
        print(f"\t - Docking failed:                  {self.dockingFailedCode}")
        print(f"\t - Docking log failed to be read:   {self.readDockingLogError}")

        print(f"\n\t~~~~~~~~~~~~~~~~ ARCHIVE ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - Archive not supported:           {self.notSupportedArchiveCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ RESCORING ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Unsupported scoring function:    {self.unsupportedScoringFunctionCode}")
        print(f"\t - Rescoring failed:                {self.rescoringFailedCode}")
        print(f"\t - Missing ODDT model:              {self.missingOddtModel}")

        print(f"\n\t~~~~~~~~~~~~~~~ CLUSTERING ERRORS ~~~~~~~~~~~~~~~")
        print(f"\t - Unsupported clustering")
        print(f"\t   algorithm:                       {self.unsupportedClusteringAlgorithmCode}")
        print(f"\t   Cluster not converged:           {self.cluster_not_converged}")


        return None

# Functions
###############################################################################
## Private ##

## Public ##
