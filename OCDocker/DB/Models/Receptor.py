from sqlalchemy import Column, Integer, String, Float, DateTime, func
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

class Receptor(Base):
    """ Define the Receptor table """
    
    # Table name
    __tablename__ = 'Receptor'

    # Relationships
    complex = relationship('Complex')

    # To Solve - Final implementation
    '''
    # Add columns for each descriptor - Flag: There is not yet 'allDescriptors' implemented in Receptor.py
    for descriptor in ocr.Receptor.allDescriptors:
        # Check is float or integer
        if descriptor.startswith("count") or descriptor in ['TotalAALength']:
            # Create the column as an integer
            locals()[f"{descriptor}"] = Column(Integer, server_default = None)
        else:
            # Create the column as a float
            locals()[f"{descriptor}"] = Column(Float, server_default = None)'''
    
    # Temporary - Receptors:
    # Declare the descriptors names as class attributes
    descriptors_names = {
        'count': ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    }

    # Declare the single descriptors names as class attributes
    single_descriptors = ['TotalAALength', 'AvgAALength', 'countChain', 'SASA', 'DipoleMoment', 'IsoelectricPoint', 'GRAVY', 'Aromaticity', 'InstabilityIndex']

    # Create all the descriptors to be class attributes
    allDescriptors = [f'{desc_prefix}{i}' for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

    for descriptor in allDescriptors:
        # Check is float or integer
        if descriptor.startswith("count") or descriptor in ['TotalAALength']:
            # Create the column as an integer
            locals()[f"{descriptor}"] = Column(Integer, server_default = None)
        else:
            # Create the column as a float
            locals()[f"{descriptor}"] = Column(Float, server_default = None)
