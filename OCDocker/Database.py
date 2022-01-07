#!/usr/lib/python3

# Imports
###############################################################################
import os
import shutil
import mimetypes
import urllib.request
import textwrap as tw

from glob import glob
from tqdm import tqdm

from OCDocker.Initialise import *
import OCDocker.DUDEZ as ocdudez
import OCDocker.Astex as ocastex
import OCDocker.PDBbind as ocpdbbind
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
## Private ##

## Public ##
def create_directories():
    '''
    Create necessary dirs.
    Input:
      -
    Return:
      -
    '''
    _ = octools.safe_create_dir(ocdb)
    _ = octools.safe_create_dir(pdbbind_archive)
    _ = octools.safe_create_dir(dudez_archive)
    _ = octools.safe_create_dir(astex_archive)

def update_DUDEZ():
    '''
    Updates the DUDE-Z database.
    Input:
      -
    Return:
      -
    '''
    # Create tmp dir for download
    _ = octools.safe_create_dir("./tmp")

    octools.printv("Downloading the DUDE-Z database")

    # Download file (with progress bar!!!)
    octools.download_url(dudez_download, "./tmp/DUDEZ.tgz")

    # Untar it (deleting the downloaded .tgz)
    octools.untar("./tmp/DUDEZ.tgz", out_path="./tmp", delete=True)

    # Move the folders (and subfolders) to right database folders
    shutil.move("./tmp/DOCKING_GRIDS_AND_POSES", dudez_archive)

    # Delete the temporary folder
    shutil.rmtree("./tmp")

    # Run p2rank in the DUDEZ database
    ocdudez.prepare()

    return

def update_pdbbind():
    '''
    Updates the PDBbind database from the Protein-ligand complexes: The refined set.
    Input:
      -
    Return:
      -
    '''
    # Parameterizing the topics (this sounds strange but one large string concatenation was bugging the IDE)
    t1 = f"- Go to the PDBbind website ({clrs['c']}http://www.pdbbind.org.cn/download.php{clrs['n']})."

    t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index), untar it and put all the protein folders folder inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder."
    t2 += f" The folders{clrs['y']} readme{clrs['n']} and{clrs['y']} index{clrs['n']} should be{clrs['r']} deleted{clrs['n']}."

    t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index), and provide the full path to it or put the file inside the {clrs['y']} {pdbbind_archive}{clrs['n']} folder and type continue (please, make sure that the downloaded file is the{clrs['c']} ONLY{clrs['n']} file inside the {clrs['y']} {pdbbind_archive}{clrs['n']} folder). If you want to skip the PDBbind update, type 'skip' (without quotes) and press enter."

    # Since no rsync option to update pdbbind database has been found you have to manually download/untar the files and put them inside the database folder
    print(tw.dedent("""
                 Unfortunately this step has not been able to be automatized... :(
    Please, we kindly ask you to perform the following steps to update the PDBbind database

    """ + t1 + """

    """ + t2 + """

    """))

    # Infinite loop (user can break it by sending an empty answer)
    while True:
        # Check the options
        option = input("Once these steps are done, type 'continue' and press enter to continue. To cancel just press enter without typing nothing.\n")

        if "'" in option or '"' in option:
            option = option.replace('"', "").replace("'", "")

        if option.lower() == "continue":
            octools.printv("Continuing the update proces...")
            pdbbindTar = glob(f"{pdbbind_archive}/*.tar.gz")[0]

            # Since everything is right, start to untar it and delete source .tar.gz file
            _ = octools.untar(pdbbindTar, out_path=pdbbind_archive, delete=True)

            # Check if there is a refined-set folder
            if os.path.isdir(f"{pdbbind_archive}/refined-set"):
                # For each file inside the refined-set folder
                for filename in os.listdir(os.path.join(pdbbind_archive, "refined-set")):
                    # Move it to the parent folder
                    shutil.move(f"{pdbbind_archive}/refined-set/{filename}", f"{pdbbind_archive}/{filename}")

                # Remove the refined-set folder
                os.rmdir(f"{pdbbind_archive}/refined-set")

            # Check if there is a readme file
            if os.path.isfile(f"{pdbbind_archive}/README.txt"):
                # Delete it
                os.remove(f"{pdbbind_archive}/README.txt")

            # Exit the loop
            break;

        elif option.lower() == "skip":
            octools.printv(f"The user decided to skip this update. Skipping!!!")
            return

        elif option == "":
            octools.print_warning("User aborted the update.")
            quit();

        else:
            octools.printv(f"Verifying if '{option}' is a valid path.")

            # Check if the .tar.gz file exists
            if os.path.isfile(option):
                octools.printv(f"The '{option}' is a valid file path. Checking if its MIME type and encoding are correct.")

                # Check its MIME type and encoding
                mime_type, enc = mimetypes.guess_type(option)
                if mime_type == "application/x-tar" and enc == "gzip":
                    octools.printv("The MIME type and encoding are correct! Following with the installation.")

                    # Since everything is right, start to untar it and delete source .tar.gz file
                    _ = octools.untar(option, out_path=pdbbind_archive, delete=True)

                    # Check if there is a refined-set folder
                    if os.path.isdir(f"{pdbbind_archive}/refined-set"):
                        # For each file inside the refined-set folder
                        for filename in os.listdir(os.path.join(pdbbind_archive, "refined-set")):
                            # Move it to the parent folder
                            shutil.move(f"{pdbbind_archive}/refined-set/{filename}", f"{pdbbind_archive}/{filename}")

                        # Remove the refined-set folder (which is empty)
                        os.rmdir(f"{pdbbind_archive}/refined-set")

                    # Check if there is a readme folder
                    if os.path.isdir(f"{pdbbind_archive}/readme"):
                        # Delete it
                        shutin.rmtree(f"{pdbbind_archive}/readme")

                    # Check if there is a index folder
                    if os.path.isdir(f"{pdbbind_archive}/index"):
                        # Delete it
                        shutin.rmtree(f"{pdbbind_archive}/index")

                    # Exit the loop
                    break

                else:
                    octools.print_warning(f"Wrong file type! The mime type must be 'application/x-tar' and its encoding must be 'gzip', however mime type '{mime_type}' and encoding '{enc}' have been found.")
            else:
                octools.print_warning(f"The string '{option}' is not a valid path!")

    # Run p2rank in the Astex database
    ocpdbbind.prepare()

    return

def update_astex():
    '''
    Updates the Astex database from the Astex Diverse Set.
    Input:
      -
    Return:
      -
    '''
    # Parameterizing the topics (this sounds strange but one large string concatenation was bugging the IDE)
    t1 = f"- Go to the CCDC website ({clrs['c']}https://www.ccdc.cam.ac.uk/support-and-resources/downloads{clrs['n']})."

    t2 = f"- Download the{clrs['c']} Astex Diverse Set{clrs['n']} located under the 'Validation Test Sets' section, untar it and put all the protein folders folder inside the{clrs['y']} {astex_archive}{clrs['n']} folder."
    t2 += f" The{clrs['y']} readme.txt{clrs['n']} file should be{clrs['r']} deleted{clrs['n']}."

    t2 = f"- Download the{clrs['c']} Astex Diverse Set{clrs['n']} located under the 'Validation Test Sets' section, and provide the full path to it or put the file inside the {clrs['y']} {astex_archive}{clrs['n']} folder and type continue (please, make sure that the downloaded file is the{clrs['c']} ONLY{clrs['n']} file inside the {clrs['y']} {astex_archive}{clrs['n']} folder). If you want to skip the Astex update, type 'skip' (without quotes) and press enter."

    # Since no rsync option to update pdbbind database has been found you have to manually download/untar the files and put them inside the database folder
    print(tw.dedent("""
                 Unfortunately this step has not been able to be automatized... :(
    Please, we kindly ask you to perform the following steps to update the PDBbind database

    """ + t1 + """

    """ + t2 + """

    """))

    # Infinite loop (user can break it by sending an empty answer)
    while True:
        # Check the options
        option = input("Once these steps are done, type 'continue' and press enter to continue. To cancel just press enter without typing nothing.\n")

        if "'" in option or '"' in option:
            option = option.replace('"', "").replace("'", "")

        if option.lower() == "continue":
            octools.printv("Continuing the update proces...")
            astexTar = glob(f"{astex_archive}/*.tar.gz")[0]

            # Since everything is right, start to untar it and delete source .tar.gz file
            _ = octools.untar(astexTar, out_path=astex_archive, delete=True)

            # Check if there is a refined-set folder
            if os.path.isdir(f"{astex_archive}/astex_diverse_set"):
                # For each file inside the refined-set folder
                for filename in os.listdir(os.path.join(astex_archive, "astex_diverse_set")):
                    # Move it to the parent folder
                    shutil.move(f"{astex_archive}/astex_diverse_set/{filename}", f"{astex_archive}/{filename}")

                # Remove the refined-set folder
                os.rmdir(f"{astex_archive}/astex_diverse_set")

            # Check if there is a readme file
            if os.path.isfile(f"{astex_archive}/README.txt"):
                # Delete it
                os.remove(f"{astex_archive}/README.txt")

            # Exit the loop
            break;

        elif option.lower() == "skip":
            octools.printv(f"The user decided to skip this update. Skipping!!!")
            return

        elif option == "":
            octools.print_warning("User aborted the update.")
            quit();

        else:
            octools.printv(f"Verifying if '{option}' is a valid path.")

            # Check if the .tar.gz file exists
            if os.path.isfile(option):
                octools.printv(f"The '{option}' is a valid file path. Checking if its MIME type and encoding are correct.")

                # Check its MIME type and encoding
                mime_type, enc = mimetypes.guess_type(option)
                if mime_type == "application/x-tar" and enc == "gzip":
                    octools.printv("The MIME type and encoding are correct! Following with the installation.")

                    # Since everything is right, start to untar it and delete source .tar.gz file
                    _ = octools.untar(option, out_path=astex_archive, delete=True)

                    # Check if there is a refined-set folder
                    if os.path.isdir(f"{astex_archive}/astex_diverse_set"):
                        # For each file inside the refined-set folder
                        for filename in os.listdir(os.path.join(astex_archive, "astex_diverse_set")):
                            # Move it to the parent folder
                            shutil.move(f"{astex_archive}/astex_diverse_set/{filename}", f"{astex_archive}/{filename}")

                        # Remove the refined-set folder (which is empty)
                        os.rmdir(f"{astex_archive}/astex_diverse_set")

                    # Check if there is a readme file
                    if os.path.isfile(f"{astex_archive}/README.txt"):
                        # Delete it
                        os.remove(f"{astex_archive}/README.txt")

                    # Exit the loop
                    break

                else:
                    octools.print_warning(f"Wrong file type! The mime type must be 'application/x-tar' and its encoding must be 'gzip', however mime type '{mime_type}' and encoding '{enc}' have been found.")
            else:
                octools.print_warning(f"The string '{option}' is not a valid path!")

    # Run p2rank in the Astex database
    ocastex.prepare()

    return

def update_databases():
    '''
    Calls all the database update functions sequentially.
    Input:
      -
    Return:
      -
    '''
    # Start the mimetypes
    mimetypes.init()

    print('\n\nUpdating ALL databases.\')
    create_directories()

    print('Updating PDBbind database...')
    update_pdbbind()
    print('\n\nDone updating PDBbind!\n')

    print('Updating Astex database...')
    update_astex()
    print('\n\nDone updating Astex!\n')

    print('Updating DUDEZ database...')
    update_DUDEZ()
    print('\n\nDone updating DUDEZ!\n')

    print('\n\nDone updating ALL databases.\')

    return
