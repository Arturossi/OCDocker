from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

import OCDocker.Ligand as ocl

class Ligand(Base):
    """ Define the Ligand table """

    # Table name
    __tablename__ = 'Ligand'

    # Relationships
    complex = relationship('Complex')

    # Add columns for each descriptor
    for descriptor in ocl.Ligand.allDescriptors:
        # Check is float or integer
        if descriptor.startswith("fr_") or descriptor.startswith("Num") or descriptor in ["HeavyAtomCount", "NHOHCount", "NOCount", "RingCount"]:
            # Create the column as an integer
            locals()[f"{descriptor}"] = Column(Integer, server_default = None)
        else:
            # Create the column as a float
            locals()[f"{descriptor}"] = Column(Float, server_default = None)
    