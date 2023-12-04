from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

class Receptor(Base):
    """ Define the Receptor table """
    
    # Relationships
    complex = relationship("Complex", back_populates = "Receptor")

    # Set foreign keys
    complex_id = Column(Integer, ForeignKey('Complex.id'))
    
    # Declare the descriptors names as class attributes
    descriptors_names = {
        "count": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = ["TotalAALength", "AvgAALength", "countChain", "SASA", "DipoleMoment", "IsoelectricPoint", "GRAVY", "Aromaticity", "InstabilityIndex"]

    # Create all the descriptors to be class attributes
    allDescriptors = [f"{desc_prefix}{i}" for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

# Add columns for each descriptor
Receptor.add_dynamic_columns(Receptor.allDescriptors)
#Receptor.add_dynamic_columns(ocr.Receptor.allDescriptors)
