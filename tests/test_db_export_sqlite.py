#!/usr/bin/env python3

# Description
###############################################################################
'''
Basic DB/DB.py coverage on a transient SQLite engine. Ensures tables create
cleanly and export helpers return expected types on empty DBs.
'''

# Imports
###############################################################################
import pytest

import pandas as pd
from sqlalchemy.orm import sessionmaker

import OCDocker.DB.DB as ocdb
from OCDocker.DB.DBMinimal import create_engine

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
All rights reserved. Use, reproduction, modification, and distribution are restricted and subject
to formal authorization from UFRJ. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##


@pytest.mark.order(36)
def test_create_tables_and_exports_on_sqlite_memory():
    # In-memory engine for isolation
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)

    # Open a session and exercise exports (DB is empty)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        df = ocdb.export_db_to_csv(s, output_format='dataframe')
        assert isinstance(df, pd.DataFrame)

        js = ocdb.export_db_to_csv(s, output_format='json')
        assert isinstance(js, str)

        csv = ocdb.export_db_to_csv(s, output_format='csv')
        assert isinstance(csv, str)
