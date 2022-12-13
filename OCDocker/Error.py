#!/usr/lib/python3

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
    """Class to handle errors and standarize them across the whole code."""

    def __init__(self, args: argparse.Namespace) -> None:
        '''Constructor for the Error class.
        
        Parameters
        ----------
        args : argparse.Namespace
            Arguments from the command line.
        
        Returns
        -------
        None
        
        Raises
        ------
        None
        '''

        # OCDocker arguments
        self.args = args

        # Common errors
        self.okCode                           = 0
        self.abortCode                        = 1
        self.unknownCode                       = -666

        # File errors
        self.fileExistsCode                   = 100
        self.fileDoNotExistCode               = 101
        self.readFileCode                     = 102
        self.writeFileCode                    = 103
        self.untarFileCode                    = 104
        self.unsupportedExtensionCode         = 105
        self.brokenPipeCode                   = 106

        # Directory errors
        self.dirExistsCode                    = 150
        self.createDirCode                    = 151
        self.dirDoesNotExistsCode             = 152
        self.dirUnallowedCode                 = 153

        # Variable errors
        self.wrongTypeCode                    = 200
        self.notSetCode                       = 201
        self.emptyCode                        = 202
        self.emptyList                        = 203

        # Subprocess errors
        self.subprocessCode                   = 300

        # Molecule error
        self.parseMoleculeCode                = 400
        self.malformedMoleculeCode            = 401
        self.ligandNotPreparedCode            = 402
        self.receptorNotPreparedCode          = 403

        # Docking error
        self.dockingObjectNotGeneratedCode    = 500
        self.recLigObjectNotGeneratedCode     = 501
        self.recLigFileDoesNotExistCode       = 502
        self.notSupportedDockingAlgorithmCode = 503
        self.bindingSiteNotFoundCode          = 504
        self.dockingFailedCode                = 505

        # Archive error
        self.notSupportedArchiveCode          = 600

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

        Raises
        ------
        None
        '''

        today = datetime.datetime.now()
        if self.args.output_level >= 4:
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

        Raises
        ------
        None
        '''

        today = datetime.datetime.now()
        if self.args.output_level >= 4:
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

        Raises
        ------
        None
        '''
        
        today = datetime.datetime.now()
        if self.args.output_level >= 3:
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

        Raises
        ------
        None
        '''

        today = datetime.datetime.now()
        if self.args.output_level >= 3:
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

        Raises
        ------
        None
        '''

        if message:
            if level == "warn" and self.args.output_level >= 1:
                self.__print_warning(message)
            elif level == "error" and self.args.output_level >= 0:
                self.__print_error(message)
            elif level == "success" and self.args.output_level >= 3:
                self.__print_success(message)
            elif level == "info" and self.args.output_level >= 2:
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

        Raises
        ------
        None
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

        Raises
        ------
        None
        '''

        self.__print_msg(message, "warn")

        return self.okCode

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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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
            The code for unsupported extension error (105).

        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.brokenPipeCode

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
            The code for create directory error (151).
        
        Raises
        ------
        None
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
        
        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.createDirCode

    def dir_does_not_exists(self, message: str = "", level: str = "warn") -> int:
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
            The code for directory does not exists error (152).

        Raises
        ------
        None
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
            The code for directory unallowed (153).

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.emptyCode

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
        
        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.receptorNotPreparedCode

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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
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

        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.dockingFailedCode

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

        Raises
        ------
        None
        '''

        self.__print_msg(message, level)

        return self.notSupportedArchiveCode

    # Debug functions
    def print_attributes(self) -> None:
        '''Print the class attributes.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        '''
        
        print(f"\t+----------------------------------------------+")
        print(f"\t|            OCDocker Return codes             |")
        print(f"\t+----------------------------------------------+")

        print(f"\n\t~~~~~~~~~~~~~~~~ GENERAL ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - No error:                        {self.okCode}")
        print(f"\t - Abortion:                        {self.abortCode}")
        print(f"\t - Unknown error:                   {self.unknownCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~~~ FILE ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - File exists:                     {self.fileExistsCode}")
        print(f"\t - File does not exists:            {self.fileDoNotExistCode}")
        print(f"\t - Read file error:                 {self.readFileCode}")
        print(f"\t - Write file error:                {self.writeFileCode}")
        print(f"\t - Untar error:                     {self.untarFileCode}")
        print(f"\t - Unsupported extension error:     {self.unsupportedExtensionCode}")
        print(f"\t - Broken PIPE error:               {self.brokenPipeCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ DIRECTORY ERRORS ~~~~~~~~~~~~~~~")
        print(f"\t - Directory exists:                {self.dirExistsCode}")
        print(f"\t - Directory creation error:        {self.createDirCode}")
        print(f"\t - Directory does not exist:        {self.dirDoesNotExistsCode}")
        print(f"\t - Directory access not allowed:    {self.dirUnallowedCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ VARIABLE ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Wrong type:                      {self.wrongTypeCode}")
        print(f"\t - Not set:                         {self.notSetCode}")
        print(f"\t - Empty:                           {self.emptyCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~ PROCESS ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Subprocess error:                {self.subprocessCode}")

        print(f"\n\t~~~~~~~~~~~~~~~ MOLECULE ERRORS ~~~~~~~~~~~~~~~~")
        print(f"\t - Molecule parse error:            {self.parseMoleculeCode}")
        print(f"\t - Malformed molecule error:        {self.malformedMoleculeCode}")
        print(f"\t - Ligand not prepared:             {self.ligandNotPreparedCode}")
        print(f"\t - Receptor not prepared:           {self.receptorNotPreparedCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~ DOCKING ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - Docking Object Generation")
        print(f"\t   error:                           {self.dockingObjectNotGeneratedCode}")
        print(f"\t - Receptor/Ligand Object")
        print(f"\t   Generation error:                {self.recLigObjectNotGeneratedCode}")
        print(f"\t - Receptor/Ligand File")
        print(f"\t   descriptor does not exist:       {self.recLigFileDoesNotExistCode}")
        print(f"\t - Not supported docking algoritm:  {self.notSupportedDockingAlgorithmCode}")
        print(f"\t - Binding site not found:          {self.bindingSiteNotFoundCode}")
        print(f"\t - Docking Failed:                  {self.dockingFailedCode}")

        print(f"\n\t~~~~~~~~~~~~~~~~ ARCHIVE ERRORS ~~~~~~~~~~~~~~~~~")
        print(f"\t - Archive not supported:           {self.notSupportedArchiveCode}")

        return None

# Functions
###############################################################################
## Private ##

## Public ##
