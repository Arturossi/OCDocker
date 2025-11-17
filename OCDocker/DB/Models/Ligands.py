from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import base

import OCDocker.Ligand as ocl


class Ligands(base):
    """ Define the Ligand table """

    # Relationships
    complexes = relationship("Complexes", back_populates = "ligand", cascade = "all, delete-orphan", lazy = "joined")

# Add columns for each descriptor
Ligands.add_dynamic_columns(ocl.Ligand.allDescriptors)
