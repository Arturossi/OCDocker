from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

import OCDocker.Ligand as ocl

class Ligands(Base):
    """ Define the Ligand table """

    # Relationships
    complex_id = Column(Integer, ForeignKey('complexes.id'))
    complex = relationship("Complexes", back_populates = "ligand")

# Add columns for each descriptor
Ligands.add_dynamic_columns(ocl.Ligand.allDescriptors)
