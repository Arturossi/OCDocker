from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import relationship
from OCDocker.DB.DB import Base
import OCDocker.Ligand as ocl

class Ligand(Base):
    """ Define the Ligand table """

    # Table name
    __tablename__ = 'Ligand'

    # Set the id column as the primary key
    id = Column(Integer, primary_key = True)

    # Add a column for the molecule name
    molecule_name = Column(String(2048))

    # Relationships
    complex = relationship('Complex')

    # Add a column for the creation and modification date
    creation_date = Column(String(2048))

    # Add columns for each descriptor
    for descriptor in ocl.Ligand.allDescriptors:
        # Check is float or integer
        if descriptor.startswith("fr_") or descriptor.startswith("Num") or descriptor in ["HeavyAtomCount", "NHOHCount", "NOCount", "RingCount"]:
            # Create the column as an integer
            locals()[f"{descriptor}"] = Column(Integer, server_default = None)
        else:
            # Create the column as a float
            locals()[f"{descriptor}"] = Column(Float, server_default = None)
    
    # Add created_at and modified_at columns (modified_at is updated automatically)
    created_at = Column(DateTime, server_default = func.now())
    modified_at = Column(DateTime, server_default = None, onupdate = func.now())
