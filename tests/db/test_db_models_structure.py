#!/usr/bin/env python3

# Description
###############################################################################
'''
Coverage tests for DB model descriptor/relationship structure.
'''

# Imports
###############################################################################
import pytest

import OCDocker.DB.Models.Complexes as occomplexes
import OCDocker.DB.Models.Ligands as ocligands
import OCDocker.DB.Models.Receptors as ocreceptors
import OCDocker.DB.Models.PipelineRuns as ocpiperuns

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

@pytest.mark.order(411)
def test_ligands_model_has_dynamic_descriptor_columns():
    descs = ocligands.ocl.Ligand.allDescriptors
    assert isinstance(descs, list)
    assert len(descs) > 0
    # Validate a sample of dynamic attributes to keep the test stable.
    for desc in descs[:10]:
        assert hasattr(ocligands.Ligands, desc)


@pytest.mark.order(412)
def test_receptors_model_descriptor_catalog_and_columns():
    assert isinstance(ocreceptors.Receptors.single_descriptors, list)
    assert "TotalAALength" in ocreceptors.Receptors.single_descriptors
    assert "countA" in ocreceptors.Receptors.allDescriptors
    assert "countV" in ocreceptors.Receptors.allDescriptors
    assert "GRAVY" in ocreceptors.Receptors.allDescriptors
    assert hasattr(ocreceptors.Receptors, "countA")
    assert hasattr(ocreceptors.Receptors, "GRAVY")
    assert hasattr(ocreceptors.Receptors, "complexes")


@pytest.mark.order(413)
def test_complexes_model_descriptor_catalog_and_relationships():
    all_desc = occomplexes.Complexes.allDescriptors
    assert "SMINA_VINA" in all_desc
    assert "VINA_VINARDO" in all_desc
    assert "GNINA_DEFAULT" in all_desc
    assert "PLANTS_CHEMPLP" in all_desc
    assert "ODDT_RFSCORE_V1" in all_desc
    assert "OCSCORE" in all_desc

    for desc in ["SMINA_VINA", "VINA_VINARDO", "GNINA_DEFAULT", "PLANTS_CHEMPLP", "ODDT_RFSCORE_V1", "OCSCORE"]:
        assert hasattr(occomplexes.Complexes, desc)

    assert hasattr(occomplexes.Complexes, "ligand")
    assert hasattr(occomplexes.Complexes, "receptor")

@pytest.mark.order(414)
def test_pipeline_runs_model_metadata_columns():
    columns = ocpiperuns.PipelineRuns.__table__.columns

    assert "complex_id" in columns
    assert "representative_pose" in columns
    assert "representative_engine" in columns
    assert "rescoring_json" in columns
    assert "summary_json" in columns
    assert "payload_path" in columns
    assert "run_report_path" in columns
    assert columns["representative_engine"].type.length == 64
    assert columns["representative_pose"].type.length == 2048

    run = ocpiperuns.PipelineRuns(
        complex_id=7,
        representative_pose="pose.mol2",
        representative_engine="vina",
        rescoring_json="{}",
        summary_json="{}",
        payload_path="payload.pkl",
        run_report_path="run_report.json",
    )
    assert run.complex_id == 7
    assert run.representative_engine == "vina"

