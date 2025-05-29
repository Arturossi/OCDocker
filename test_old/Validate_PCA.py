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
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Licensed under the Apache License, Version 2.0 (January 2004)
See: http://www.apache.org/licenses/LICENSE-2.0

Commercial use requires a separate license.  
Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

# Methods
###############################################################################

ocpca.run_pca('/data/hd4tb/OCDocker/data/ocdb/predictions/OCDocker_pre.csv.gz', 0.80, './')
