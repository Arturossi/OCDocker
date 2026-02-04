#!/usr/bin/env python3

# Description
###############################################################################
'''
Exercise CRUD helpers in DB.Models.Base using an in-memory SQLite session and
the concrete Ligands model.
'''

# Imports
###############################################################################
import pytest

from sqlalchemy.orm import scoped_session, sessionmaker

import OCDocker.DB.Models.Base as base_mod
from OCDocker.DB.DBMinimal import create_engine
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models.Ligands import Ligands

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

@pytest.mark.order(49)
def test_base_crud_on_ligands_sqlite_memory():
    # Prepare transient engine + session and patch into Base module
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    Base.metadata.create_all(engine)
    Session = scoped_session(sessionmaker(bind=engine))
    base_mod.session = Session  # patch module-level session used by Base methods

    # Insert
    assert Ligands.insert({"name": "L1"}) is True

    # Insert-or-update (update path)
    assert Ligands.insert_or_update({"name": "L1"}) is True

    # Update by name
    assert Ligands.update("L1", {"name": "L1"}) is True

    # Delete by name
    assert Ligands.delete("L1") is True

    # Column helpers
    assert Ligands.determine_column_type("countAtoms").__class__.__name__ in ("Integer", "INTEGER")
    assert Ligands.determine_column_type("someFloat").__class__.__name__ in ("Float", "FLOAT")
