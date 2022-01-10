#!/usr/lib/python3

# Imports
###############################################################################
from OCDocker.Initialise import *
import OCDocker.baseDB as ocbdb

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
Sets of classes and functions that are used to process the DUDE-Z dataset.

They are imported as:

import OCDocker.DUDEZ as ocdudez
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##
def verify_integrity():
    '''
    Verifies the integrity of the DUDEZ database
    Input:
      -
    Return:
      -
    '''
    ocbdb.verify_integrity(dudez_archive)

def convert_debug_to_production(chosenAlgorithm = "ac", strict = False, removeDebug = False):
    '''
    Converts debug folders to production mode. It is required to choose an algorithm which will be used furtherly in the pipeline.
    Input:
     chosenAlgorithm [string] DEFAULT: ac  - The short code for the chosen algorithm. The choices are:
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
     strict          [bool] DEFAULT: False - If True does not convert the data even if there is only one dir, if False will convert the data if the protein has only one dir (this is good when you ran with only one algorithm, some proteins may have been run with "na")
     removeDebug     [bool] DEFAULT: False - If True removes debug folders (NO TURNING BACK), if False leave the dirs
    Return:
      -
    '''
    ocbdb.convert_debug_to_production(dudez_archive, chosenAlgorithm = chosenAlgorithm, strict = strict, removeDebug = removeDebug)

def prepare(overwrite = False):
    '''
    Prepares the DUDEZ database.
    Input:
     overwrite [bool] DEFAULT: False - If True, all files will be generated, otherwise will try to optimize file generation, skipping files with output already generated.
    Return:
      -
    '''
    ocbdb.prepare(dudez_archive, overwrite = overwrite)
