from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from OCDocker.DB.DB import Base

class Complex(Base):
    """ Define the Complex table """

    # Table name
    __tablename__ = 'Complex'

    # Set the id column as the primary key
    id = Column(Integer, primary_key = True)
    
    # Relationships
    ligand = relationship("Ligand")
    receptor = relationship("Receptor")

    # Set foreign keys
    ligand_id = Column(Integer, ForeignKey("Ligand.id"))
    receptor_id = Column(Integer, ForeignKey("Receptor.id"))

    # Add a column for the creation and modification date
    creation_date = Column(String(2048))
    
    # Complexes descriptors:
    descriptors_names = {
        'rescoring_': ['vina', 'scoring_dkoes', 'vinardo', 'old_scoring_dkoes', 'fast_dkoes', 'scoring_ad4'],
        'rfscore_v': range(1,4)
        }

    single_descriptors = ['chemplp', 'plp', 'plp95', 'PLECrf_p5_l1_s65536', 'nnscore']

    allDescriptors = [f'{desc_prefix}{i}' for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

    # Add columns for each descriptor
    for descriptor in allDescriptors:
        # Create the column as a float
        locals()[f"{descriptor}"] = Column(Float, server_default = None)

    # Add created_at and modified_at columns (modified_at is updated automatically)
    created_at = Column(DateTime, server_default = func.now())
    modified_at = Column(DateTime, server_default = None, onupdate = func.now())
