#!/usr/bin/env python3

# Description
###############################################################################
"""
SQLAlchemy model for pocket descriptors and their receptor relationship.

A receptor may hold several pockets, so the pocket row references its receptor
and the receptor table never references a pocket.

Usage:

from OCDocker.DB.Models.Pockets import Pockets
"""

# Imports
###############################################################################
from typing import TYPE_CHECKING

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from OCDocker.DB.Models.Base import base

if TYPE_CHECKING:
    import OCDocker.Pocket as ocpocket
else:
    try:
        import OCDocker.Pocket as ocpocket
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "") not in {"Bio", "rdkit"}:
            raise
        ocpocket = None

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################


class Pockets(base):
    """SQLAlchemy model for pocket descriptor columns.

    Dynamic columns are added from :attr:`OCDocker.Pocket.Pocket.allDescriptors`.

    Attributes
    ----------
    id : sqlalchemy.Integer
        Primary key.
    receptor_id : sqlalchemy.Integer
        Foreign key to :class:`Receptors`.
    reference_ligand : sqlalchemy.String
        Reference ligand used to define the pocket.
    cutoff : sqlalchemy.Float
        Heavy-atom distance cutoff (angstroms) used to define the pocket.
    residues : sqlalchemy.Text
        Pocket residues as ``chain:number:insertion:resname`` joined by ``;``.
    receptor : Receptors
        Parent receptor row.
    allDescriptors : list[str]
        Names of dynamically mapped descriptor columns (class attribute).
    """

    # Relationships
    receptor_id = Column(Integer, ForeignKey("receptors.id"))

    receptor = relationship("Receptors", back_populates="pockets")

    # Pocket definition
    reference_ligand = Column(String(760), server_default=None)
    cutoff = Column(Float, server_default=None)
    residues = Column(Text, server_default=None)

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
        "countChain",
        "SASA",
        "GRAVY",
        "Aromaticity",
        "NetCharge",
        "countHBondDonors",
        "countHBondAcceptors",
    ]

    # Create all the descriptors to be class attributes
    allDescriptors = [
        f"{desc_prefix}{i}"
        for desc_prefix, desc_indices in descriptors_names.items()
        for i in desc_indices
    ] + single_descriptors


# Add columns for each descriptor
Pockets.add_dynamic_columns(
    ocpocket.Pocket.allDescriptors if ocpocket is not None else Pockets.allDescriptors
)


# Functions
###############################################################################
## Private ##

## Public ##
