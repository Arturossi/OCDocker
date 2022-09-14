#!/usr/lib/python3

# Imports
###############################################################################
import os
import shutil
import mimetypes
import urllib.request
import textwrap as tw

from glob import glob

from OCDocker.Initialise import *
import OCDocker.DUDEz as ocdudez
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
    # Create the base dir
    _ = octools.safe_create_dir(ocdb)
    # Create the pdbbind dir
    _ = octools.safe_create_dir(pdbbind_archive)
    # Create the dudez dir
    _ = octools.safe_create_dir(dudez_archive)
    # Create the Astex dir
    _ = octools.safe_create_dir(astex_archive)
    # Create the Parsed dir
    _ = octools.safe_create_dir(parsed_archive)

def update_DUDEz(overwrite = False):
    '''
    Updates the DUDE-Z database.
    Input:
      overwrite [bool] - If True, overwrites the existing database
    Return:
      -
    '''
    # Create tmp dir for download
    _ = octools.safe_create_dir("./tmp")

    octools.printv("Downloading the DUDE-Z database")

    # Download the benchmark grids indexes
    octools.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/DUDE-Z_targets", "./tmp/DUDE-Z_targets")

    # Initialize an empty list to store the targets
    targets = []
    # Read the targets into a list
    with open("./tmp/DUDE-Z_targets", "r") as f:
        targets = f.read().splitlines()
    
    # Check if the target list is empty
    if len(targets) == 0:
        return errors.file_do_not_exist("The target list is empty. Something went wrong with the download.", "error")
    
    # For each target (TODO: parallelize)
    for target in targets:
        # Trying to fix dudez lazy webmasters mistakes
        if target == "D4":
            target2 = "DRD4"
        else:
            target2 = target
        # Create a folder for the target in the archive
        _ = octools.safe_create_dir(f"{dudez_archive}/{target2}")
        # Download the target receptor
        octools.download_url(f"{dudez_download}/DOCKING_GRIDS_AND_POSES/{target2}/rec.crg.pdb", f"{dudez_archive}/{target2}/rec.crg.pdb")
        # Download the dudeZ ligands
        octools.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/{target}/ligands.smi", f"{dudez_archive}/{target2}/ligands.smi")
        # Download the dudeZ ligands
        octools.download_url(f"{dudez_download}/DUDE-Z-benchmark-grids/{target}/decoys.smi", f"{dudez_archive}/{target2}/decoys.smi")
        # Download the Extrema set
        octools.download_url(f"{dudez_download}/extrema/{target}/minus2/{target}_minus2.smi", f"{dudez_archive}/{target2}/extrema_minus2.smi")
        octools.download_url(f"{dudez_download}/extrema/{target}/minus1/{target}_minus1.smi", f"{dudez_archive}/{target2}/extrema_minus1.smi")
        octools.download_url(f"{dudez_download}/extrema/{target}/neutral/{target}_neutral.smi", f"{dudez_archive}/{target2}/extrema_neutral.smi")
        octools.download_url(f"{dudez_download}/extrema/{target}/plus1/{target}_plus1.smi", f"{dudez_archive}/{target2}/extrema_plus1.smi")
        octools.download_url(f"{dudez_download}/extrema/{target}/plus2/{target}_plus2.smi", f"{dudez_archive}/{target2}/extrema_plus2.smi")

    # Create the goldilocks folder
    _ = octools.safe_create_dir(f"{dudez_archive}/goldilocks")
    # Download the Goldilocks set (it is universal for all targets)
    octools.download_url(f"{dudez_download}/goldilocks/goldilocks_minus2.smi", f"{dudez_archive}/goldilocks/goldilocks_minus2.smi")
    octools.download_url(f"{dudez_download}/goldilocks/goldilocks_minus1.smi", f"{dudez_archive}/goldilocks/goldilocks_minus1.smi")
    octools.download_url(f"{dudez_download}/goldilocks/goldilocks_neutral.smi", f"{dudez_archive}/goldilocks/goldilocks_neutral.smi")
    octools.download_url(f"{dudez_download}/goldilocks/goldilocks_plus1.smi", f"{dudez_archive}/goldilocks/goldilocks_plus1.smi")
    octools.download_url(f"{dudez_download}/goldilocks/goldilocks_plus2.smi", f"{dudez_archive}/goldilocks/goldilocks_plus2.smi")

    # Process each target
    targets = [d for d in glob(f"{dudez_archive}/*") if os.path.basename(d.split(os.path.sep)[-1]) not in ['goldilocks']]

    # For each target (TODO: parallelize)
    for target in targets:
        # Get the target name
        target_name = os.path.basename(target)
        # Process the ligands
        octools.printv(f"Processing the ligands for {target_name}")
        # List to hold the tuples for each processing that will be made
        process_list = [("dudez_ligands", "ligands"), ("dudez_decoys", "decoys"), ("extrema/minus2", "extrema_minus2"), ("extrema/minus1", "extrema_minus1"), ("extrema/neutral", "extrema_neutral"), ("extrema/plus1", "extrema_plus1"), ("extrema/plus2", "extrema_plus2")]
        # Create the extrema folder inside the target folder
        _ = octools.safe_create_dir(f"{target}/extrema")
        # For each data
        for data in process_list:
            # Print which file is being processed
            octools.printv(f"Processing {target}/{data[1]}.smi")
            print(f"Processing {target}/{data[1]}.smi")
            # Create the ligands folder
            _ = octools.safe_create_dir(f"{target}/{data[0]}")
            # Process the ligands, splitting them into the multiple files
            with open(f"{target}/{data[1]}.smi", "r") as f:
                for line in f:
                    # Get the smiles and name of the ligand
                    smiles, name = line.split()
                    # Test if the file exists
                    if not os.path.isfile(f"{target}/{data[0]}/{name}.mol2") or overwrite:
                        # Convert it to mol2 (NOTE: There are many molecules with SAME name... currently I am not handling this. I am just accounting the first molecule and discarding the others. IMPORTANT: Error messages WILL pop while processing the data here! They may be safe to ignore, I guess...)
                        mol2 = octools.convertMolsFromString(smiles, f"{target}/{data[0]}/{name}.mol2")
                    else:
                        octools.print_warning(f"File {target}/{data[0]}/{name}.mol2 already exists. Skipping...")

    # Process the goldilocks set
    octools.printv("Processing the goldilocks set")
    # List to hold the tuples for each processing that will be made
    process_list = [("minus2", "goldilocks_minus2"), ("minus1", "goldilocks_minus1"), ("neutral", "goldilocks_neutral"), ("plus1", "goldilocks_plus1"), ("plus2", "goldilocks_plus2")]
    # For each data
    for data in process_list:
        # Create the ligands folder
        _ = octools.safe_create_dir(f"{dudez_download}/goldilocks/{data[0]}")
        # Process the ligands, splitting them into the multiple files
        with open(f"{dudez_download}/goldilocks/{data[1]}.smi", "r") as f:
            for line in f:
                # Get the smiles and name of the ligand
                smiles, name = line.split()
                # Test if the file exists
                if not os.path.isfile(f"{dudez_download}/goldilocks/{data[0]}/{name}.mol2") or overwrite:
                    # Convert it to mol2
                    mol2 = octools.convertMolsFromString(smiles, f"{dudez_download}/goldilocks/{data[0]}/{name}.mol2")

    # Delete the temporary folder
    shutil.rmtree("./tmp")

    # Run p2rank in the DUDEz database
    ocdudez.prepare()

    return errors.ok()

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

    #t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index), untar it and put all the protein folders folder inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder."
    #t2 += f" The folders{clrs['y']} readme{clrs['n']} and{clrs['y']} index{clrs['n']} should be{clrs['r']} deleted{clrs['n']}."

    t2 = f"- Download the{clrs['c']} Protein-ligand complexes: The refined set{clrs['n']} (it may have the number 3 as its index)."

    t3 = f"- Then provide the full path to it or put the file inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder and type continue (please, make sure that the downloaded file is the{clrs['c']} ONLY{clrs['n']} file inside the{clrs['y']} {pdbbind_archive}{clrs['n']} folder). If you want to skip the PDBbind update, type 'skip' (without quotes) and press enter. "

    # Since no rsync option to update pdbbind database has been found you have to manually download/untar the files and put them inside the database folder
    print(tw.dedent("""
            Unfortunately this step has not been able to be completely automatized... :(
    Please, we kindly ask you to perform the following steps to update the PDBbind database

    """ + t1 + """

    """ + t2 + """

    """ + t3 + """

    """))

    # Infinite loop (user can break it by sending an empty answer)
    while True:
        # Check the options
        option = input("Once these steps are done, type 'continue' and press enter to continue. To cancel just press enter without typing nothing.\n")

        # If there is quotes or double quotes in the path
        if "'" in option or '"' in option:
            # Remove them
            option = option.replace('"', "").replace("'", "")

        # If the option in lowercase is in the continue list (traductions may enter here)
        if option.lower() in ["continue", "continuar"]:
            octools.printv("Continuing the update proces...")
            # Find the pdbbindTar file
            pdbbindTar = glob(f"{pdbbind_archive}/*.tar.gz")[0]

            # Since everything is right, start to untar/ungz them and delete source .tar.gz file
            _ = octools.untar(pdbbindTar, out_path=f"{pdbbind_archive}/complex", delete=True)

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

        elif option.lower() in ["skip", "pular"]:
            octools.printv(f"The user decided to skip this update. Skipping!!!")
            return

        elif option == "":
            octools.print_warning("User aborted the update.")
            quit();

        else:
            octools.printv(f"Still not validated, please use the other way.")
            continue
            # Will not run the code below
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

    print("\n\nUpdating ALL databases.\n")
    create_directories()

    print("Updating PDBbind database...")
    update_pdbbind(overwrite = args.overwrite)
    print("\n\nDone updating PDBbind!\n")

    print("Updating Astex database...")
    update_astex(overwrite = args.overwrite)
    print("\n\nDone updating Astex!\n")

    print("Updating DUDEz database...")
    update_DUDEz(overwrite = args.overwrite)
    print("\n\nDone updating DUDEz!\n")

    print("\n\nDone updating ALL databases.\n")

    return
