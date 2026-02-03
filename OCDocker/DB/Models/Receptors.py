#!/usr/bin/env python3

# Description
###############################################################################
'''
SQLAlchemy model for receptor descriptors and complex relationships.

Usage:

from OCDocker.DB.Models.Receptors import Receptors
'''

# Imports
###############################################################################
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

import OCDocker.Receptor as ocr
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

class Receptors(base):
    """ Define the Receptor table
    
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
    complexes = relationship("Complexes", back_populates = "receptor", cascade = "all, delete-orphan")

    # Declare the descriptors names as class attributes
    descriptors_names = {
        "count": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = ["TotalAALength", "AvgAALength", "countChain", "SASA", "DipoleMoment", "IsoelectricPoint", "GRAVY", "Aromaticity", "InstabilityIndex"]

    # Create all the descriptors to be class attributes
    allDescriptors = [f"{desc_prefix}{i}" for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors


# Add columns for each descriptor
Receptors.add_dynamic_columns(ocr.Receptor.allDescriptors)


# Functions
###############################################################################
## Private ##

## Public ##
