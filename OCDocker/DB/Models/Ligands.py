#!/usr/bin/env python3

# Description
###############################################################################
'''
SQLAlchemy model for ligand descriptors and complex relationships.

Usage:

from OCDocker.DB.Models.Ligands import Ligands
'''

# Imports
###############################################################################
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

import OCDocker.Ligand as ocl
from OCDocker.DB.Models.Base import base

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################

class Ligands(base):
    """ Define the Ligand table 
    
    Attributes
    ----------
    id : Integer
        Primary key of the table
    complexes : list
        Relationship to the Complexes table
    allDescriptors : list
        List of all descriptor column names
    """

    # Relationships
    complexes = relationship("Complexes", back_populates = "ligand", cascade = "all, delete-orphan", lazy = "joined")


# Add columns for each descriptor
Ligands.add_dynamic_columns(ocl.Ligand.allDescriptors)


# Functions
###############################################################################
## Private ##

## Public ##
