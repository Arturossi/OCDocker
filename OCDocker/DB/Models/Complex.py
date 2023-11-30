from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

class Complex(Base):
    """ Define the Complex table """

    # Table name
    __tablename__ = 'Complex'

    # Relationships
    ligand = relationship("Ligand")
    receptor = relationship("Receptor")

    # Set foreign keys
    ligand_id = Column(Integer, ForeignKey("Ligand.id"))
    receptor_id = Column(Integer, ForeignKey("Receptor.id"))

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

