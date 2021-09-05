#!/usr/lib/python3

# Imports
###############################################################################
import sys
import shutil
import tarfile
import datetime
from Initialise import *

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


# Functions
###############################################################################
def printv(text, verbosity):
    '''
    Function to print if verbosity is invoked.
    '''

    if verbosity == 1:
        print(text)

def print_section(n, name):
    '''
    Function to print section header.
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
            f.write(f"{datetime.now().strftime("%H:%M:%S")}: Starting new OCDocker run\n")
    else:
        with open('OCDocker_Progress.out', 'a') as f:
            f.write(f"\n{datetime.now().strftime("%H:%M:%S")}: {str(name)}...\n")

def section(n, name):
    '''
    Function to return section header.
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
    Function to print section header in progress file.
    Input:
     n    [int]    - Number of the subsection
     name [string] - Name of the subsection
    '''

    print(f"\n{clrs['r']}|" +
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

    with open('OCDocker_Progress.out', 'a') as f:
        f.write(datetime.now().strftime("%H:%M:%S")+": "+str(name)+"...\n")

def subsection(n, name):
    '''
    Function to return section header.
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
    '''

    print(f"**We are {clrs['y']}t{clrs['r']}e"+
          f"{clrs['y']}r{clrs['r']}r{clrs['y']}i"+
          f"{clrs['r']}b{clrs['y']}l{clrs['r']}y"+
          f"{clrs['n']} sorry... =(\n")

def untar(fname, out_path=".", delete=False):
    '''
    Function to untar a file
    Input:
     fname      [string]  - File path to be untarred
     out_path   [string]  - Output path
     delete     [boolean] - Flag to denote if the tar file should be deleted (True) or not (False)
    Return:
     0 if success
     1 if problems while opening file
     2 if file is not tar.gz
    '''

    if (fname.endswith("tar.gz") or fname.endswith(".tgz")):
        try:
            tar = tarfile.open(fname)
            tar.extractall(path=out_path)
            tar.close()
            if(delete):
                shutil.rmtree(fname)
                print(f"The file {fname} has been {clrs['y']}deleted!{clrs['n']}")
            print(f"The file {fname} has been {clrs['g']}successfully{clrs['n']} untarred to the dir {out_path}!")
            return 0
        except Exception as e:
            print(f"{clrs['r']}Failed{clrs['n']} to untar the file {fname}.\n\n{clrs['r']}Error{clrs['n']}: {e}")
            return 1
    else:
        print(f"The file {fname} is not a tar.gz file. {clrs['y']}Aborting execution{clrs['n']}")
        return 2

def safe_create_dir(dirname):
    '''
    Function to create a dir if not exists
    Input:
     dirname      [string]  - File path to be untarred
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
