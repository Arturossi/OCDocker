from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

import OCDocker.Ligand as ocl

class Ligand(Base):
    """ Define the Ligand table """

    # Relationships
    complex = relationship("Complex", back_populates = "ligands")

    # Set foreign keys
    complex_id = Column(Integer, ForeignKey('Complex.id'))

# Add columns for each descriptor
Ligand.add_dynamic_columns(ocl.Ligand.allDescriptors)
