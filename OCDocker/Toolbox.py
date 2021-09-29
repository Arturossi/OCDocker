#!/usr/lib/python3

# Imports
###############################################################################
import os
import sys
import shutil
import tarfile
import datetime
import subprocess
import urllib.request
from tqdm import tqdm
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
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# Functions
###############################################################################
def printv(message, verbosity):
    '''
    Function to print if verbosity is invoked.
    Input:
     message   [string] - Message to be printed
     verbosity [int]    - Flag for verbosity (0 - off; 1 - on)
    Return:
     -
    '''
    if verbosity == 1:
        print(message)

def print_warning(message):
    '''
    Function to print warning.
    Input:
     message [string] - Message to be printed
    Return:
     -
    '''
    print(f"{clrs['y']}WARNING{clrs['n']}: {message}")

def print_error(message):
    '''
    Print error.
    Input:
     message [string] - Message to be printed
    Return:
     -
    '''
    print(f"{clrs['r']}ERROR{clrs['n']}: {message}")

def print_section(n, name):
    '''
    Print the section header.
    Input:
     n [int] - Number of the section
     name [string] - Name of the
    Return:
     -
    '''
    print(f"\n{clrs['y']}+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n" +
          f"{clrs['r']}|" +
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

    if name == "Runtime Arguments":
        with open('OCDocker_Progress.out', 'w') as f:
            f.write(f"{datetime.now().strftime('%H:%M:%S')}: Starting new OCDocker run\n")
    else:
        with open('OCDocker_Progress.out', 'a') as f:
            f.write(f"\n{datetime.now().strftime('%H:%M:%S')}: {str(name)}...\n")

def section(n, name):
    '''
    Return the section header.
    Input:
     n    [int]    - Number of the section
     name [string] - Name of the section
    Return:
     section_string [string] - The subsection composed string
    '''
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
     n    [int]    - Number of the subsection
     name [string] - Name of the subsection
    Return:
     -
    '''
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

    with open('OCDocker_Progress.out', 'a') as f:
        f.write(datetime.now().strftime("%H:%M:%S")+": "+str(name)+"...\n")

def subsection(n, name):
    '''
    Return the subsection header.
    Input:
     n    [int]    - Number of the subsection
     name [string] - Name of the subsection
    Return:
     subsection_string [string] - The subsection composed string
    '''
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
    Function to print sorry message
    Input:
     -
    Return:
     -
    '''
    print(f"**We are {clrs['y']}t{clrs['r']}e"+
          f"{clrs['y']}r{clrs['r']}r{clrs['y']}i"+
          f"{clrs['r']}b{clrs['y']}l{clrs['r']}y"+
          f"{clrs['n']} sorry... =(\n")

def untar(fname, out_path=".", delete=False):
    '''
    Untar a file
    Input:
     fname     [string]  - File path to be untarred
     out_path  [string]  - Output path
     delete    [boolean] - Flag to denote if the tar file should be deleted (True) or not (False)
    Return:
     0 if success
     1 if problems while opening file
     2 if file is not tar.gz
    '''
    if (fname.endswith("tar.gz") or fname.endswith(".tgz")):
        try:
            print("Preparing to untar the file...")
            # open your tar.gz file
            with tarfile.open(name=fname) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(member=member, path=out_path)
            # Report success on untarring the file
            print(f"The file {fname} has been {clrs['g']}successfully{clrs['n']} untarred to the dir {out_path}!")
            # If delete flag is set, delete file
            if(delete):
                #shutil.rmtree(fname) # remove the files
                os.remove(fname) # remove the files
                print(f"The file {fname} has been {clrs['y']}deleted!{clrs['n']}") # Report success on deleting the file
            return 0
        except Exception as e:
            print(f"{clrs['r']}Failed{clrs['n']} to untar the file {fname}.\n\n{clrs['r']}Error{clrs['n']}: {e}")
            return 1
    else:
        print(f"The file {fname} is not a tar.gz file. {clrs['y']}Aborting execution{clrs['n']}")
        return 2

def safe_create_dir(dirname):
    '''
    Create a dir if not exists
    Input:
     dirname [string] - File path to be untarred
    Return:
      0 if success
      1 if folder exists
     -1 if any problem has occurred
     -2 should not appear
    '''
    try:
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
            return 0
        else:
            return 1
    except:
        return -1
    return -2

def download_url(url, out_path):
    '''
    Download a file from given url
    Input:
     url      [string] - Url to be downloaded
     out_path [string] - Output path
    Return:
      -
    '''
    with DownloadProgressBar(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=out_path, reporthook=t.update_to)

def run(cmd, logFile = ""):
    '''
    Run the command (generic)
    Input:
      cmd     [list(string)]             - List containing the strings of the command
      logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output
    Return:
      0 - No problems were found
      1 - The var cmd is not set or is empty list
      2 - The command list has wrong type
      3 - Problems while running the command
    '''
    if not cmd:
        print_error(f"The variable cmd is not set or is an empty list!")
        return 1

    if type(cmd) != list:
        print_error(f"The argument cmd has to be a list! Found {type(cmd)} instead...")
        return 2

    if logFile == "":
        logFile = os.devnull

    try:
        with open(logFile, "w") as outfile:
            subprocess.run(cmd, stdout=outfile)
    except Exception as e:
        print_error(f"Found a problem while executing the command '{' '.join(cmd)}': {e}")
        return 3
    return 0

def convert2mol2(input, output, logFile = ""):
    '''
    Convert a pdb/sdf/mol/smi file to .mol2
    Input:
      input   [string]                   - Input path
      output  [string]                   - Output path
      logfile [list(string)] DEFAULT: "" - Path to the logFile. If empty, suppress the output.
    Return:
      0  - No problem found in execution
      1  - The output file already exists
      2  - Not supported extension in input
      3  - Error while running command
      -1 - You should NEVER see this error, but its when no exception is thrown
    '''

    if os.path.isfile(output):
        print_warning(f"The file {output} already exists, aborting conversion.")
        return 1

    # Allowed extensions
    allowed = [".pdb", ".sdf", ".mol", ".smi"]

    # Get input and output extensions
    inputExtension = os.path.splitext(input)[1]
    outputExtension = os.path.splitext(output)[1]

    # Check if the input extension is supported
    if not inputExtension in allowed:
        print_warning(f"The file {input} has not a supported extension. Found {inputExtension} and expected one of the following: {', '.join(allowed)}")
        return 2

    # If the output has no extension
    if outputExtension == "":
        # Add a mol2 extension to it
        output += ".mol2"

    # Execute the obabel command
    if inputExtension == ".pdb":
        cmd = ['obabel', '-ipdb', str(input), '-omol2', '-O', str(output)]
    elif inputExtension == ".mol":
        cmd = ['obabel', str(input), '-omol2', '-O', str(output)]
    elif inputExtension == ".sdf":
        cmd = ['obabel', '-isdf', str(input), '-omol2', '-O', str(output)]
    elif inputExtension == ".smi":
        cmd = ['obabel', '-ismi', str(input), '-omol2', '-O', str(output)]
    else:
        print_error("What are you expecting to see here? This code should NEVER execute!")
        return -1

    run(cmd, logFile=logFile)
