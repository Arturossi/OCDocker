#!/usr/bin/env python3

# Description
###############################################################################
"""
SQLAlchemy model for docking complexes with dynamic descriptor columns.

Usage:

from OCDocker.DB.Models.Complexes import Complexes
"""

# Imports
###############################################################################
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from OCDocker.DB.Models.Base import base
from OCDocker.DB.Models.Ligands import Ligands
from OCDocker.DB.Models.Receptors import Receptors

# License
###############################################################################
"""
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Classes
###############################################################################


class Complexes(base):
    """SQLAlchemy model linking ligand and receptor descriptor rows.

    Attributes
    ----------
    id : sqlalchemy.Integer
        Primary key.
    ligand_id : sqlalchemy.Integer
        Foreign key to :class:`Ligands`.
    receptor_id : sqlalchemy.Integer
        Foreign key to :class:`Receptors`.
    ligand : Ligands
        Parent ligand row.
    receptor : Receptors
        Parent receptor row.
    allDescriptors : list[str]
        Names of dynamically mapped scoring-function descriptor columns.
    """

    # Relationships
    ligand_id = Column(Integer, ForeignKey("ligands.id"))
    receptor_id = Column(Integer, ForeignKey("receptors.id"))

    ligand = relationship("Ligands", back_populates="complexes")
    receptor = relationship("Receptors", back_populates="complexes")

    # Complexes descriptors
    descriptors_names = {
        "smina_": [
            "vina",
            "scoring_dkoes",
            "vinardo",
            "old_scoring_dkoes",
            "fast_dkoes",
            "scoring_ad4",
        ],
        "vina_": ["vina", "vinardo"],
        # Keep Gnina hardcoded for stable DB schema, matching gnina_scoring_functions defaults.
        "gnina_": [
            "ad4_scoring",
            "default",
            "dkoes_fast",
            "dkoes_scoring",
            "dkoes_scoring_old",
            "vina",
            "vinardo",
        ],
        "plants_": ["chemplp", "plp", "plp95"],
        "oddt_": [f"rfscore_v{i}" for i in range(1, 4)]
        + ["PLECrf_p5_l1_s65536", "nnscore"],
    }

    allDescriptors = [
        f"{desc_prefix}{i}".upper()
        for desc_prefix, desc_indices in descriptors_names.items()
        for i in desc_indices
    ] + ["OCSCORE"]


# Add columns for each descriptor
Complexes.add_dynamic_columns(Complexes.allDescriptors)


# Functions
###############################################################################
## Private ##

## Public ##
