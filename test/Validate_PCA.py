#!/usr/bin/env python3

# Description
###############################################################################
""" Module to perform unit analysis to create the PCA."""

# Imports
###############################################################################

from OCDocker.Initialise import *

import OCDocker.OCScore.Dimensionality.PCA as ocpca

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

# Classes
###############################################################################

# Methods
###############################################################################

ocpca.run_pca('/data/hd4tb/OCDocker/data/ocdb/predictions/OCDocker_pre.csv.gz', 0.80, './')
