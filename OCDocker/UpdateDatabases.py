#!/usr/lib/python3

# Imports
###############################################################################
import os
import shutil
import urllib.request
import textwrap as tw

from glob import glob
from tqdm import tqdm

import OCDocker.DUDEZ as ocdudez
import OCDocker.Toolbox as octools

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
Sets of classes and functions that are used to update the OCDocker database
They are imported as:
import OCDocker.Database as ocdb
'''

# Classes
###############################################################################


# Functions
###############################################################################
def create_directories():
    '''
    Create dirs
    '''
    _ = octools.safe_create_dir(ocdb)
    _ = octools.safe_create_dir(pdbbind_archive)
    _ = octools.safe_create_dir(dudez_archive)

def update_DUDEZ(args):
    '''
    Function to update the DUDEZ database
    Called by: update_databases()
    '''
    # Create tmp dir for download
    _ = octools.safe_create_dir("./tmp")

    print("Downloading the DUDEZ database")

    # Download file (with progress bar!!!)
    octools.download_url(dudez_download, "./tmp/DUDEZ.tgz")

    # Untar it (deleting the downloaded .tgz)
    octools.untar("./tmp/DUDEZ.tgz", out_path="./tmp", delete=True)

    # Move the folders (and subfolders) to right database folders
    shutil.move("./tmp/DOCKING_GRIDS_AND_POSES", dudez_archive)

    # Delete the temporary folder
    shutil.rmtree("./tmp")

    # THIS SECTION MIGHT CHANGE TO THE USE OF BLinDPyPr IN THE FUTURE

    # Run p2rank in the DUDEZ database
    ocdudez.runprank(args)

    return

def update_pdbbind(args):
    '''
    Function to update the pdbbind database
    Called by: update_databases()
    '''

    # Parameterizing the topics (this sounds strange but one large string concatenation was bugging the IDE)
    t1 = f"- Go to the PDBbind website ({clrs['c']}http://www.pdbbind.org.cn/download.php{clrs['n']})"

    t2 = f"- Download the {clrs['c']}Protein-ligand complexes: The general set minus refined set{clrs['n']}, untar it and put all the protein folders folder inside the{clrs['y']} {pdbbind_archive}/complexes{clrs['n']} folder and the{clrs['y']} index {clrs['n']} folder should be put in the{clrs['y']} {pdbbind_archive}{clrs['n']} folder."
    t2 += f" The{clrs['y']} readme{clrs['n']} folder should be {clrs['r']} deleted{clrs['n']}."

    t3 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']}, untar it and put all the protein folders folder inside the{clrs['y']} {pdbbind_archive}/complexes{clrs['n']} folder. The {clrs['y']} readme{clrs['n']} and {clrs['y']} index {clrs['n']} folders should be{clrs['r']} deleted{clrs['n']}."

    # Since no rsync option to update pdbbind database has been found you have to manually download/untar the files and put them inside the database folder
    print(tw.dedent("""
                 Unfortunately this step has not been able to be automatized... (yet) :(
    Please we kindly ask you to perform the following steps to update the PDBbind database

    """ + t1 + """

    """ + t2 + """

    """ + t3 + """

    """

    while(True):
        option = input('Once these steps are done, type "continue" (without the double quotes) and press enter to continue. To cancel just press enter without typing nothing.\n')
        if option.lower() == 'continue':
            print('Continuing the update proces...')
            break;
        elif option == "":
            print('User aborted the update.')
            quit();
        else:
            print('Unknown option!')

    # The following code is awaiting a better opportunity to show some work
    """pdbbind_files = glob.glob(f"{pdbbind_archive}/*.tar.gz")

    for pdbbind_file in pdbbind_files:
        f = os.path.join(pdbbind_archive, pdbbind_file)

        print(f'Trying to untar file {f}')
        octools.untar(f, out_path=pdbbind_archive)"""

def update_databases(args):
    '''
    Calls all the database update functions sequentially (PDBbind)
    Called by: RunOCDocker.py:main()
    '''
    print('\n\nUpdating ALL databases.\n')
    create_directories()
    #print('Updating PDBbind database...')
    #update_pdbbind(args)
    #print('\n\nDone updating PDBbind!\n')
    print('Updating DUDEZ database...')
    update_DUDEZ(args)
    print('\n\nDone updating DUDEZ!\n')
