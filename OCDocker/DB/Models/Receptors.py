#!/usr/bin/env python3

# Description
###############################################################################
"""
SQLAlchemy model for receptor descriptors and complex relationships.

Usage:

from OCDocker.DB.Models.Receptors import Receptors
"""

# Imports
###############################################################################
from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from OCDocker.DB.Models.Base import base

try:
    import OCDocker.Receptor as ocr
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") not in {"Bio", "rdkit"}:
        raise
    ocr = None

# License
###############################################################################
"""OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

Contact: Artur Duque Rossi - arturossi10@gmail.com
"""

# Classes
###############################################################################


class Receptors(base):
    """SQLAlchemy model for receptor descriptor columns.

    Dynamic columns are added from :attr:`OCDocker.Receptor.Receptor.allDescriptors`.

    Attributes
    ----------
    id : sqlalchemy.Integer
        Primary key.
    complexes : list[Complexes]
        Related docking complexes for this receptor.
    allDescriptors : list[str]
        Names of dynamically mapped descriptor columns (class attribute).
    """

    # Relationships
    complexes = relationship(
        "Complexes", back_populates="receptor", cascade="all, delete-orphan"
    )

    # Declare the descriptors names as class attributes
    descriptors_names = {
        "count": [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = [
        "TotalAALength",
        "AvgAALength",
        "countChain",
        "SASA",
        "DipoleMoment",
        "IsoelectricPoint",
        "GRAVY",
        "Aromaticity",
        "InstabilityIndex",
    ]

    # Create all the descriptors to be class attributes
    allDescriptors = [
        f"{desc_prefix}{i}"
        for desc_prefix, desc_indices in descriptors_names.items()
        for i in desc_indices
    ] + single_descriptors


# Add columns for each descriptor
Receptors.add_dynamic_columns(
    ocr.Receptor.allDescriptors if ocr is not None else Receptors.allDescriptors
)


# Functions
###############################################################################
## Private ##

## Public ##
