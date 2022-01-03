#!/usr/lib/python3

# Imports
###############################################################################
from glob import glob
from tqdm import tqdm

from OCDocker.Initialise import *
import OCDocker.Vina as ocvina
import OCDocker.Toolbox as octools
import OCDocker.ExternalTools.runprank as runprank

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
Sets of classes and functions that are used to process the Astex Diverse
dataset.

They are imported as:

import OCDocker.Astex as ocastex
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def runprankAstex():
    '''
    Generate all vina required files for provided protein.
    Input:
     path [string] - Input path.
    Return:
      -
    '''
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
    octools.printv("Generating information regarding possible ligand site.")

    # Get all dirs paths in the DUDEZ database
    dirs = glob(f"{astex_archive}/*")

    # For each directory in the database folder
    for d in tqdm(iterable=dirs, total=len(dirs)):
        # Set the input file name path
        fin = f"{d}/protein"

        # Set the ligand input file name path
        lfin = f"{d}/ligand"

        # Find the protein name
        ptn = d.split("/")[-1]

        # Set the output path
        fout = f"{d}/p2rank"

        # Create the p2rank output dir
        _ = octools.safe_create_dir(fout)

        # Convert the protein file from mol2 to pdb
        _ = octools.convertMols(f"{fin}.mol2", f"{fin}.pdb")

        # Convert the ligand file from mol to mol2
        _ = octools.convertMols(f"{lfin}.mol", f"{lfin}.mol2")

        # Reset the input file variable
        fin = f"{fin}.pdb"

        # Run p2rank
        runprank.run_prank(fin, fout, algorithms, prank = prank, threads = args.cpu_cores,
                           debug = False, boxMaxCutoff = 0.5, pocketCutoff = 0.1, verbose = args.verbosity)

        # Create the vina inputs from the boxes
        ocvina.generate_vina_files_database(d, fin)

    return
