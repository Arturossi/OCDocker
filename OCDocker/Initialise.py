#!/usr/bin/env python3

# Description
###############################################################################
'''
Second layer of primordial variables and functions that are used to initialise
OCDocker library.\n
Almost all scripts that use OCDocker must import this file.

They are imported as:

from OCDocker.Initialise import *
'''

# Imports
###############################################################################
from OCDocker.PreInitialise import *
from OCDocker.DB.DB import setup_database

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

# Functions
###############################################################################

# Define Global Variables
###############################################################################

# Initialise
###############################################################################
setup_database()
