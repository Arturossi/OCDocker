#!/usr/bin/env python3

# Description
###############################################################################
"""
SQLAlchemy model for ligand descriptors and complex relationships.

Usage:

from OCDocker.DB.Models.Ligands import Ligands
"""

# Imports
###############################################################################
from types import SimpleNamespace

from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship

from OCDocker.DB.Models.Base import base

try:
    import OCDocker.Ligand as ocl
except ModuleNotFoundError as exc:
    if getattr(exc, "name", "") != "rdkit":
        raise
    _fallback_ligand_descriptors = ["MolWt", "MolLogP", "NumHAcceptors", "NumHDonors"]
    ocl = SimpleNamespace(
        Ligand=SimpleNamespace(allDescriptors=_fallback_ligand_descriptors)
    )

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################


class Ligands(base):
    """SQLAlchemy model for ligand descriptor columns.

    Dynamic columns are added from :attr:`OCDocker.Ligand.Ligand.allDescriptors`.

    Attributes
    ----------
    id : sqlalchemy.Integer
        Primary key.
    complexes : list[Complexes]
        Related docking complexes for this ligand.
    allDescriptors : list[str]
        Names of dynamically mapped descriptor columns (class attribute).
    """

    allDescriptors = list(ocl.Ligand.allDescriptors)

    # Relationships
    complexes = relationship(
        "Complexes",
        back_populates="ligand",
        cascade="all, delete-orphan",
        lazy="joined",
    )


# Add columns for each descriptor
Ligands.add_dynamic_columns(Ligands.allDescriptors)


# Functions
###############################################################################
## Private ##

## Public ##
