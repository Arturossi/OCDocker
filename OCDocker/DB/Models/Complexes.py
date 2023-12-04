from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import Base

class Complexes(Base):
    """ Define the Complex table """

    # Relationships
    ligand = relationship("Ligands", back_populates = "complex")
    receptor = relationship("Receptors", back_populates = "complex")

    # Complexes descriptors
    descriptors_names = {
        "rescoring_": ["vina", "scoring_dkoes", "vinardo", "old_scoring_dkoes", "fast_dkoes", "scoring_ad4"],
        "rfscore_v": range(1, 4)
        }

    single_descriptors = ["chemplp", "plp", "plp95", "PLECrf_p5_l1_s65536", "nnscore"]

    allDescriptors = [f"{desc_prefix}{i}" for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices] + single_descriptors

# Add columns for each descriptor
Complexes.add_dynamic_columns(Complexes.allDescriptors)
