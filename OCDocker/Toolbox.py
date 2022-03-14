#!/usr/lib/python3

# Imports
###############################################################################
import os
import sys
import shutil
import inspect
import tarfile
import datetime
import subprocess
import urllib.request

from tqdm import tqdm
from openbabel import pybel
from openbabel import openbabel

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
Sets of classes and functions that are used by the OCDocker in many sources.

They are imported as:

import OCDocker.Toolbox as octools
'''

# Classes
###############################################################################
class DownloadProgressBar(tqdm):
    """
    Deal with the progress bar to track download. Extends the tqdm class.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        '''
        b     [int] - Byte
        bsize [int] - Byte size
        tsize [int] - Current progress
        '''
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Functions
###############################################################################
## Private ##

## Public ##
def printv(message):
    '''
    Function to print if verbosity is invoked.
    Input:
      message   [string] - Message to be printed.
    Return:
      -
    '''
    if args.verbosity == 1:
        today = datetime.datetime.now()
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {message}")
    return

def print_info(message):
    '''
    Function to print warning.
    Input:
      message [string] - Message to be printed.
    Return:
      -
    '''
    today = datetime.datetime.now()
    if args.debug == 1 or args.verboity == 1:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['c']}INFO{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.")
    else:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['c']}INFO{clrs['n']}: {message}")
    return

def print_success(message):
    '''
    Print success.
    Input:
      message [string] - Message to be printed.
    Return:
      -
    '''
    today = datetime.datetime.now()
    if args.debug == 1:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['g']}SUCCSESS{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.")
    else:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['g']}SUCCSESS{clrs['n']}: {message}")
    return

def print_warning(message):
    '''
    Function to print warning.
    Input:
      message [string] - Message to be printed.
    Return:
      -
    '''
    today = datetime.datetime.now()
    if args.debug == 1 or args.verboity == 1:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['y']}WARNING{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.")
    else:
        print(f"[{clrs['c']}{today.strftime('%d-%m-%Y')}{clrs['n']}|{clrs['c']}{today.strftime('%H:%M:%S')}{clrs['n']}] {clrs['y']}WARNING{clrs['n']}: {message}")
    return

def print_error(message):
    '''
    Print error.
    Input:
      message [string] - Message to be printed.
    Return:
      -
    '''
    today = datetime.datetime.now()
    if args.debug == 1 or args.verboity == 1:
        print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] {clrs['r']}ERROR{clrs['n']}: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.")
    else:
        print(f"[\033[1;96m{today.strftime('%d-%m-%Y')}\033[1;0m|\033[1;96m{today.strftime('%H:%M:%S')}\033[1;0m] {clrs['r']}ERROR{clrs['n']}: {message}")
    return


def print_info_log(message, logfile, mode="a"):
    '''
    Function to print info into log.
    Input:
      message [string]            - Message to be printed.
      logfile [string]            - Log file path.
      mode    [string] DEFAULT: a - Open file mode.
    Return:
      -
    '''
    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] INFO: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_success_log(message, logfile, mode="a"):
    '''
    Function to print success into log.
    Input:
      message [string]            - Message to be printed.
      logfile [string]            - Log file path.
      mode    [string] DEFAULT: a - Open file mode.
    Return:
      -
    '''
    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] SUCCSESS: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_warning_log(message, logfile, mode="a"):
    '''
    Function to print warning into log.
    Input:
      message [string]            - Message to be printed.
      logfile [string]            - Log file path.
      mode    [string] DEFAULT: a - Open file mode.
    Return:
      -
    '''
    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] WARNING: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return

def print_error_log(message, logfile, mode="a"):
    '''
    Function to print error into log.
    Input:
      message [string]            - Message to be printed.
      logfile [string]            - Log file path.
      mode    [string] DEFAULT: a - Open file mode.
    Return:
      -
    '''
    today = datetime.datetime.now()
    with open(logfile, mode) as f:
        f.write(f"[{today.strftime('%d-%m-%Y')}|{today.strftime('%H:%M:%S')}] ERROR: {message} In function '{inspect.currentframe().f_back.f_code.co_name}' line {inspect.currentframe().f_back.f_lineno} from file '{inspect.currentframe().f_back.f_code.co_filename}'.\n")
    return


def print_section(n, name):
    '''
    Print the section header and write progress to the progress file.
    Input:
      n    [int]    - Number of the section.
      name [string] - Name of the section (Empty string for no log).
    Return:
      -
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
            with open("OCDocker_Progress.out", "w") as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')}: Starting new OCDocker run\n")
        else:
            with open("OCDocker_Progress.out", "a") as f:
                f.write(f"\n{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n")
    return

def section(n, name):
    '''
    Return the section header.
    Input:
      n    [int]    - Number of the section.
      name [string] - Name of the section.
    Return:
      [string] - The subsection composed string.
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

def print_subsection(n, name):
    '''
    Print the subsection header in progress file.
    Input:
      n    [int]    - Number of the subsection.
      name [string] - Name of the subsection (Empty string for no log).
    Return:
      -
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
        with open("OCDocker_Progress.out", "a") as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n")
    return

def subsection(n, name):
    '''
    Return the subsection header.
    Input:
      n    [int]    - Number of the subsection.
      name [string] - Name of the subsection.
    Return:
      [string] - The subsection composed string.
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

def print_sorry():
    '''
    Function to print sorry message.
    Input:
      -
    Return:
      -
    '''
    # Print a nice looking sorry message :/
    print(f"**We are {clrs['y']}t{clrs['r']}e"+
          f"{clrs['y']}r{clrs['r']}r{clrs['y']}i"+
          f"{clrs['r']}b{clrs['y']}l{clrs['r']}y"+
          f"{clrs['n']} sorry... =(\n")
    return

def untar(fname, out_path=".", delete=False):
    '''
    Untar a file.
    Input:
      fname    [string]  - File path to be untarred.
      out_path [string]  - Output path.
      delete   [boolean] - Flag to denote if the tar file should be deleted (True) or not (False).
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Print verboosity
    printv(f"Untarring file '{fname}' to the output '{out_path}'")
    # Check if the file has the right extensions
    if (fname.endswith("tar.gz") or fname.endswith(".tgz")):
        try:
            printv("Preparing to untar the file...")
            # open your tar.gz file
            with tarfile.open(name=fname) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(member=member, path=out_path)
            # Report success on untarring the file
            print_success(f"The file {fname} has been {clrs['g']}successfully{clrs['n']} untarred to the dir {out_path}!")
            # If delete flag is set, delete file
            if(delete):
                #shutil.rmtree(fname) # remove the files
                os.remove(fname) # remove the files
                print_success(f"The file {fname} has been {clrs['y']}deleted!{clrs['n']}") # Report success on deleting the file
            return errors.ok()
        except Exception as e:
            return errors.untar_file(message=f"{clrs['r']}Failed{clrs['n']} to untar the file {fname}.\n\n{clrs['r']}Error{clrs['n']}: {e}", level="error")
    else:
        # No supported extension has been provided
        return errors.unsupported_extension(message=f"The file {fname} is not a tar.gz file. {clrs['y']}Aborting execution{clrs['n']}", level="error")

def safe_create_dir(dirname):
    '''
    Create a dir if not exists.
    Input:
      dirname [string] - File path to be created.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Try to create
    try:
        # If file does not exists
        if not os.path.isdir(dirname):
            # Create it
            os.mkdir(dirname)
            # Print verbosity
            if args.verbosity:
                print_success(f"Successfully created the directory {dirname}")
            return errors.ok()
        else:
            # It exists
            return errors.file_exists(message="File 'dirname' already exists!", level="warn")
    except Exception as e:
        # Some error has occurred
        return errors.create_dir(message=f"Problem found while creating the dir {dirname}: {e}", level="error")
    # This should never appear since all the other paths ends in some kind of return
    return errors.unknown(message=f"What are you expecting for? This message should NEVER appear!!!!!!! Btw problems while creating a dir safetly.", level="error")

def download_url(url, out_path):
    '''
    Download a file from given url.
    Input:
      url      [string] - Url to be downloaded.
      out_path [string] - Output path.
    Return:
      -
    '''
    # Print verboosity
    printv(f"Downloading a file from '{url}' and saving to {out_path}.")
    # Create the progress bar object
    with DownloadProgressBar(unit="B",
                             unit_scale=True,
                             miniters=1,
                             desc=url.split(os.path.sep)[-1]) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)
    return

def run(cmd, logFile = ""):
    '''
    Run the given command (generic).
    Input:
      cmd     [list(string)]             - List containing the strings of the command.
      logfile [list(string)] DEFAULT: "" - Path to the logFile (empty string to suppress the output).
    Return:
      [int]
      See Error.py for all return codes.
    '''
    if not cmd:
        return errors.not_set(message=f"The variable cmd is not set or is an empty list!", type="error")

    if type(cmd) != list:
        return errors.wrong_type(message=f"The argument cmd has to be a list! Found '{type(cmd)}' instead...", type="error")

    # Print verboosity
    printv(f"Running the command '{' '.join(cmd)}'.")

    if logFile == "":
        printv(f"No log will be made")
        logFile = os.devnull
    else:
        printv(f"Logging into '{logFile}'")

    try:
        with open(logFile, "w") as outfile:
            subprocess.run(cmd, stdout=outfile)
    except Exception as e:
        return errors.subprocess(message=f"Found a problem while executing the command '{' '.join(cmd)}': {e}", level="error")

    return errors.ok()

def convert2mol2_legacy(input, output, logFile = ""):
    '''
    Convert a pdb/sdf/mol/smi file to '.mol2'. Uses external obabel software. [DEPRECATED]
    Input:
      input   [string]                   - Input path.
      output  [string]                   - Output path.
      logfile [list(string)] DEFAULT: "" - Path to the logFile (empty string to suppress the output).
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Print verboosity
    printv(f"Converting '{input}' to '.mol2'.")

    if os.path.isfile(output):
        return errors.file_exists(message=f"The file '{output}' already exists, aborting conversion.", level="warn")

    # Allowed extensions
    allowed = [".pdb", ".sdf", ".mol", ".smi"]

    # Get input and output extensions
    inputExtension = os.path.splitext(input)[1]
    outputExtension = os.path.splitext(output)[1]

    # Check if the input extension is supported
    if not inputExtension in allowed:
        return errors.wrong_type(type=f"The file '{input}' has not a supported extension. Found '{inputExtension}' and expected one of the following: {', '.join(allowed)}", level="warn")

    # If the output has no extension
    if outputExtension == "":
        # Add a mol2 extension to it
        output += ".mol2"

    # Execute the obabel command according to the extension
    if inputExtension == ".pdb":
        cmd = ["obabel", "-ipdb", str(input), "-omol2", "-O", str(output)]
    elif inputExtension == ".mol":
        cmd = ["obabel", str(input), "-omol2", "-O", str(output)]
    elif inputExtension == ".sdf":
        cmd = ["obabel", "-isdf", str(input), "-omol2", "-O", str(output)]
    elif inputExtension == ".smi":
        cmd = ["obabel", "-ismi", str(input), "-omol2", "-O", str(output)]
    else:
        return errors.unknown(message="What are you expecting to see here? This code should NEVER execute! (BTW, this is from unsupported file extension...)", level="error")

    # Return the execution code from the correct extension
    return run(cmd, logFile=logFile)

def convert2mol2(input, output):
    '''
    Convert a pdb/sdf/mol/smi file to '.mol2'. [DEPRECATED]
    Input:
      input  [string] - Input path.
      output [string] - Output path.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Print verboosity
    printv(f"Converting '{input}' to '.mol2'.")

    # Find the extension for input and output
    extension = validate_obabel_extension(input)
    outExtension = os.path.splitext(output)[1]

    # Check if the extension is valid
    if type(extension) != str:
        print_error(f"Problems while reading the molecule from file '{input}'.")
        return extension

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".mol2":
        print_warn(f"The output extension is not '.mol2', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.mol2'. Please pay attention, since this might be a problem in the future for you!")

    # Check if the output exists, if so, no need to convert
    if os.path.isfile(output):
        return errors.file_exists(message=f"The file '{output}' already exists, aborting conversion.", level="warn")
    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(extension, "mol2")
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, input)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, output)
    except Exception as e:
        return errors.subprocess(message=f"Error while running molecule conversion using obabel python lib. Error: {e}", level="error")
    return errors.ok()

def convert2pdb(input, output):
    '''
    Convert a mol2/sdf/mol/smi file to '.pdb'. [DEPRECATED]
    Input:
      input  [string] - Input path.
      output [string] - Output path.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Print verboosity
    printv(f"Converting '{input}' to '.pdb'.")

    # Find the extension for input and output
    extension = validate_obabel_extension(input)
    outExtension = os.path.splitext(output)[1]

    # Check if the extension is valid
    if type(extension) != str:
        print_error(f"Problems while reading the molecule from file '{input}'.")
        return extension

    # Discover if the output extension is pdbqt (to warn user if it is not)
    if outExtension != ".pdb":
        print_warn(f"The output extension is not '.pdb', is {outExtension}. This function converts {clrs['r']}ONLY{clrs['n']} to '.pdb'. Please pay attention, since this might be a problem in the future for you!")

    # Check if the output exists, if so, no need to convert
    if os.path.isfile(output):
        return errors.file_exists(message=f"The file '{output}' already exists, aborting conversion.", level="warn")

    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(extension, "pdb")
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, input)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, output)
    except Exception as e:
        return errors.subprocess(message=f"Error while running molecule conversion using obabel python lib. Error: {e}", level="error")

    return errors.ok()

def convertMols(input, output):
    '''
    Convert a molecule file between two extensions which obabel supports.
    Input:
      input  [string] - Input path.
      output [string] - Output path.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Find the extension for input and output
    inExtension = validate_obabel_extension(input)
    outExtension = validate_obabel_extension(output)

    # Print verboosity
    printv(f"Converting '{input}' to '.{outExtension}'.")

    # Check if the input extension is valid
    if type(inExtension) != str:
        print_error(f"Problems while reading the molecule from input file '{input}'.")
        return inExtension

    # Check if the output extension is valid
    if type(outExtension) != str:
        print_error(f"Problems while reading the molecule from output file '{output}'.")
        return outExtension

    # Check if the output exists, if so, no need to convert
    if os.path.isfile(output):
        return errors.file_exists(message=f"The file '{output}' already exists, aborting conversion.", level="warn")

    # Try to convert (if fails, throw exception for subprocess failing)
    try:
        # Create a conversor object
        obConversion = openbabel.OBConversion()
        # Set the conversion from the extension to pdbqt
        obConversion.SetInAndOutFormats(inExtension, outExtension)
        # Create an empty OBMol object
        mol = openbabel.OBMol()
        # Load the input file to the prebiusly loaded OBMol object
        obConversion.ReadFile(mol, input)
        # Write the mol object to the output performing the conversion
        obConversion.WriteFile(mol, output)
    except Exception as e:
        return errors.subprocess(message=f"Error while running molecule conversion from {inExtension} to {outExtension} using obabel python lib. Error: {e}", level="error")

    return errors.ok()

def split_and_convert(path, out_path, extension):
    '''
    Splits a multi-molecule file then save the output in multiple single-molecule file with the desired extension. (Supported by openbabel)
    Input:
      path      [string] - Path to the file which will be tested.
      extension [string] - Output desired extension.
    Return:
      [int]
      See Error.py for all return codes.
    '''
    # Finds the input extension
    extensionIn = validate_obabel_extension(path)

    # If input extension is not valid
    if type(extension) != str:
        # Return the unsupported_extension
        return errors.unsupported_extension(f"Unsupported extension provided while spliting '{path}' file. Supported extensions are the one supported by OpenBabel.", "error")

    # For each molecule in input file
    for mol in pybel.readfile(extensionIn, path):
        # Get its name and remove the "none string", strip blank spaces and then replace the remaining blank spaces for underscores
        molName = mol.title.replace("none", "").strip().replace(" ", "_")
        # Set the output file name
        outfile = f"{out_path}/{molName}.{extension}"
        # Try to convert
        try:
            # Write the file with the right extension
            mol.write(extension, outfile)
        # If fails
        except Exception as e:
            # Return write file error
            return errors.write_file(f"Problems while writing the file '{outfile}'.")
    # Since everything gone ok, return the ok code
    return errors.ok()

def validate_obabel_extension(path):
    '''
    Validate the input file extension to ensure the compability with obabel lib.
    Input:
      path [string] - Path to the file which will be tested.
    Return:
      [string/int] The extension if success, otherwise see Error.py for all return codes.
    '''
    supportedExtensions = [
                            'acesin', 'adf', 'alc', 'ascii', 'bgf', 'box', 'bs', 'c3d1', 'c3d2', 'cac',
                            'caccrt', 'cache', 'cacint', 'can', 'cdjson', 'cdxml', 'cht', 'cif', 'ck', 'cml',
                            'cmlr', 'cof', 'com', 'confabreport', 'CONFIG', 'CONTCAR', 'CONTFF', 'copy', 'crk2d', 'crk3d',
                            'csr', 'cssr', 'ct', 'cub', 'cube', 'dalmol', 'dmol', 'dx', 'ent', 'exyz',
                            'fa', 'fasta', 'feat', 'fh', 'fhiaims', 'fix', 'fps', 'fpt', 'fract', 'fs',
                            'fsa', 'gamin', 'gau', 'gjc', 'gjf', 'gpr', 'gr96', 'gro', 'gukin', 'gukout',
                            'gzmat', 'hin', 'inchi', 'inchikey', 'inp', 'jin', 'k', 'lmpdat', 'lpmd', 'mcdl',
                            'mcif', 'MDFF', 'mdl', 'ml2', 'mmcif', 'mmd', 'mmod', 'mna', 'mol', 'mol2',
                            'mold', 'molden', 'molf', 'molreport', 'mop', 'mopcrt', 'mopin', 'mp', 'mpc',
                            'mpd', 'mpqcin', 'mrv', 'msms', 'nul', 'nw', 'orcainp', 'outmol', 'paint',
                            'pcjson', 'pcm', 'pdb', 'pdbqt', 'png', 'pointcloud', 'POSCAR', 'POSFF', 'pov',
                            'pqr', 'pqs', 'qcin', 'report', 'rinchi', 'rsmi', 'rxn', 'sd', 'sdf',
                            'smi', 'smiles', 'stl', 'svg', 'sy2', 'tdd', 'text', 'therm', 'tmol',
                            'txt', 'txyz', 'unixyz', 'VASP', 'vmol', 'xed', 'xyz', 'yob', 'zin'
                          ]
    extension = os.path.splitext(path)[1][1:]
    if extension in supportedExtensions:
        return extension
    return errors.unsupported_extension(message=f"Unsupported extension for input molecule file! Supported extensions are '{' '.join(supportedExtensions)}' and got '{extension}'.")

def is_algorithm_allowed(path):
    '''
    Finds if the given dir is a folder from an allowed algorithm.
    Input:
      path [string] - Path to the directory which will be tested.
                      The algorithm list and their shortcodes:
                          AffinityPropagation: ap
                          AgglomerativeClustering: ac
                          Birch: bi
                          DBSCAN: db
                          KMeans:  km
                          MeanShift: ms
                          MiniBatchKMeans: mb
                          NoCluster: na
                          OPTICS: op
                          SpectralClustering: sc
    Return:
      [bool] True if is allowed / False if is not allowed
    '''
    # Allowed algorithms
    allowed = ["ap", "ac", "bi", "db", "km", "ms", "mb", "na", "op", "sc"]
    return path.split(os.path.sep).pop() in allowed
