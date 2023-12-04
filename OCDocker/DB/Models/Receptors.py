from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

class Receptors(Base):
    """ Define the Receptor table """
    
    # Relationships
    complex_id = Column(Integer, ForeignKey('complexes.id'))
    complex = relationship("Complexes", back_populates = "receptor")

    # Declare the descriptors names as class attributes
    descriptors_names = {
        "count": ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = ["TotalAALength", "AvgAALength", "countChain", "SASA", "DipoleMoment", "IsoelectricPoint", "GRAVY", "Aromaticity", "InstabilityIndex"]

    # Create all the descriptors to be class attributes
    allDescriptors = [f"{desc_prefix}{i}" for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

# Add columns for each descriptor
Receptors.add_dynamic_columns(Receptors.allDescriptors)
#Receptor.add_dynamic_columns(ocr.Receptor.allDescriptors)
