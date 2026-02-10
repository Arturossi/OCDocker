#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests for database CSV export helpers.
'''

# Imports
###############################################################################
from __future__ import annotations

import csv
import io
import pytest

from sqlalchemy.orm import sessionmaker

import OCDocker.DB.DB as ocdb
from OCDocker.DB.DBMinimal import create_engine
from OCDocker.DB.Models import Complexes, Ligands, Receptors

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


@pytest.mark.order(35)
def test_export_db_to_csv_invalid_format_raises():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        with pytest.raises(ValueError):
            _ = ocdb.export_db_to_csv(s, output_format='invalid')  # type: ignore[arg-type]


@pytest.mark.order(34)
def test_export_db_to_csv_returns_strings_on_empty_db():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)

    with Session() as s:
        js = ocdb.export_db_to_csv(s, output_format='json')
        assert isinstance(js, str)
        csv = ocdb.export_db_to_csv(s, output_format='csv')
        assert isinstance(csv, str)


@pytest.mark.order(33)
def test_export_db_to_csv_string_escapes_special_characters():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)

    complex_name = 'complex,"quoted"\nline'
    receptor_name = 'receptor,"alpha"\nline'
    ligand_name = 'ligand,"beta"\nline'

    with Session() as s:
        receptor = Receptors.Receptors(name=receptor_name)
        ligand = Ligands.Ligands(name=ligand_name)
        s.add_all([receptor, ligand])
        s.flush()

        complex_obj = Complexes.Complexes(
            name=complex_name,
            receptor_id=receptor.id,
            ligand_id=ligand.id,
        )
        s.add(complex_obj)
        s.commit()

        csv_text = ocdb.export_db_to_csv(s, output_format='csv', drop_na=False)
        rows = list(csv.DictReader(io.StringIO(csv_text)))

    assert len(rows) == 1
    assert rows[0]["name"] == complex_name
    assert rows[0]["receptor"] == receptor_name
    assert rows[0]["ligand"] == ligand_name
