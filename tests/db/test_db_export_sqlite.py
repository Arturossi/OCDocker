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
'''OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

Copyright (c) Federal University of Rio de Janeiro (UFRJ).

Licensed under the UFRJ License (see LICENSE). You may use, study, modify, and
redistribute this software for any purpose, including in publications and
derivative works, provided you preserve this notice and give appropriate credit
to UFRJ and the original developers listed above.

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
