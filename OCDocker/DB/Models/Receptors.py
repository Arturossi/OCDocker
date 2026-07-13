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
from typing import TYPE_CHECKING

from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from OCDocker.DB.Models.Base import base

if TYPE_CHECKING:
    import OCDocker.Receptor as ocr
else:
    try:
        import OCDocker.Receptor as ocr
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") not in {"Bio", "rdkit"}:
            raise
        ocr = None

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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
