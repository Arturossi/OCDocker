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
import OCDocker.Tools.runprank as runprank

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

def update_DUDEZ(verbosity):
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

    # Algorithms to be analyzed (Only Agglomerative Clustering)
    algorithms = {
        "AffinityPropagation": False,
        "AgglomerativeClustering": True,
        "Birch": False,
        "DBSCAN": False,
        "KMeans": False,
        "MeanShift": False,
        "MiniBatchKMeans": False,
        "NoCluster": False,
        "OPTICS": False,
        "SpectralClustering": False
    }

    # Generate boxes for all receptors
    print("Generating information regarding possible ligand site.")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{dudez_archive}/*")

    # For each directory in the database folder
    for d in tqdm(iterable=dirs, total=len(dirs)):
        # Set the input file name path
        fin = f"{d}/rec.crg.pdb"

        # Find the protein name
        ptn = d.split("/")[-1]

        # Set the output path
        fout = f"{d}/p2rank"

        # Create the p2rank output dir
        _ = octools.safe_create_dir(fout)

        # Run p2rank
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = cpu_cores, debug = False, boxMaxCutoff = 0.5, pocketCutoff = 0.1, verbose = verbosity)

        # Create the vina inputs from the boxes
        ocdudez.generate_vina_files(d)

    return

def update_pdbbind(verbosity):
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

def update_databases(verbosity):
    '''
    Calls all the database update functions sequentially (PDBbind)
    Called by: RunOCDocker.py:main()
    '''
    print('\n\nUpdating ALL databases.\n')
    create_directories()
    #print('Updating PDBbind database...')
    #update_pdbbind(verbosity)
    #print('\n\nDone updating PDBbind!\n')
    print('Updating DUDEZ database...')
    update_DUDEZ(verbosity)
    print('\n\nDone updating DUDEZ!\n')
