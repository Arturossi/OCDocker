from sqlalchemy import Column, ForeignKey, Integer
from sqlalchemy.orm import relationship
from OCDocker.DB.Models.Base import base

class Complexes(base):
    """ Define the Complex table """
    
    # Relationships
    ligand_id = Column(Integer, ForeignKey("ligands.id"))
    receptor_id = Column(Integer, ForeignKey("receptors.id"))

    ligand = relationship("Ligands", back_populates = "complexes")
    receptor = relationship("Receptors", back_populates = "complexes")

    # Complexes descriptors
    descriptors_names = {
        "smina_": ["vina", "scoring_dkoes", "vinardo", "old_scoring_dkoes", "fast_dkoes", "scoring_ad4"],
        "vina_": ["vina", "vinardo"],
        "plants_": ["chemplp", "plp", "plp95"],
        "oddt_": [f"rfscore_v{i}" for i in range(1, 4)] + ["PLECrf_p5_l1_s65536", "nnscore"]
    }

    allDescriptors = [f"{desc_prefix}{i}".upper() for desc_prefix, desc_indices in descriptors_names.items() for i in desc_indices]

# Add columns for each descriptor
Complexes.add_dynamic_columns(Complexes.allDescriptors)
