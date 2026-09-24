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
import OCDocker.DB.Models.Pockets as ocpockets

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
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

@pytest.mark.order(415)
def test_pockets_model_descriptor_catalog_and_columns():
    assert ocpockets.Pockets.allDescriptors == ocpockets.ocpocket.Pocket.allDescriptors
    assert "countA" in ocpockets.Pockets.allDescriptors
    assert "NetCharge" in ocpockets.Pockets.allDescriptors
    assert "countHBondDonors" in ocpockets.Pockets.allDescriptors

    columns = ocpockets.Pockets.__table__.columns
    for desc in ocpockets.Pockets.allDescriptors:
        assert desc in columns
    for meta in ["receptor_id", "reference_ligand", "cutoff", "residues"]:
        assert meta in columns

    assert hasattr(ocpockets.Pockets, "receptor")

@pytest.mark.order(416)
def test_pockets_reference_receptor_but_receptors_do_not_reference_pockets():
    foreign_keys = {fk.target_fullname for fk in ocpockets.Pockets.__table__.foreign_keys}
    assert foreign_keys == {"receptors.id"}

    # A receptor may hold several pockets, so the receptor table has no pocket column
    assert not any("pocket" in column.name for column in ocreceptors.Receptors.__table__.columns)
    assert hasattr(ocreceptors.Receptors, "pockets")

@pytest.mark.order(417)
def test_receptor_holds_multiple_pockets_in_sqlite():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from OCDocker.DB.Models.Base import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as s:
        receptor = ocreceptors.Receptors(name="rec")
        receptor.pockets = [
            ocpockets.Pockets(name="rec_pocket0", reference_ligand="lig0.sdf", cutoff=8.0, residues="A:1::LYS", countK=1, NetCharge=1.0),
            ocpockets.Pockets(name="rec_pocket1", reference_ligand="lig1.sdf", cutoff=8.0, residues="A:2::ASP", countD=1, NetCharge=-1.0),
        ]
        s.add(receptor)
        s.commit()

        stored = s.query(ocreceptors.Receptors).filter_by(name="rec").one()
        assert sorted(p.name for p in stored.pockets) == ["rec_pocket0", "rec_pocket1"]
        assert all(p.receptor_id == stored.id for p in stored.pockets)
        assert s.query(ocpockets.Pockets).filter_by(name="rec_pocket1").one().countD == 1

        # Deleting the receptor removes its pockets
        s.delete(stored)
        s.commit()
        assert s.query(ocpockets.Pockets).count() == 0

