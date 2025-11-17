#!/usr/bin/env python3

'''
Additional DB coverage:
- setup_database lazy URL resolution via mocked Initialise
- export_db_to_csv branches that write to files for dataframe/json/csv
'''

from __future__ import annotations
import types
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import sessionmaker

import pytest

import OCDocker.DB.DB as ocdb
from OCDocker.DB.DBMinimal import create_engine


@pytest.mark.order(37)
def test_setup_database_uses_db_url(monkeypatch):
    # Provide a stub Initialise with db_url and no engine
    init = types.ModuleType('OCDocker.Initialise')
    init.db_url = "sqlite:///:memory:"
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init)

    eng = ocdb.setup_database()
    assert eng is not None


@pytest.mark.order(38)
def test_setup_database_derives_from_engine_url(monkeypatch):
    class _E:
        def __init__(self, url):
            self.url = url


    init = types.ModuleType('OCDocker.Initialise')
    init.engine = _E("sqlite:///:memory:")
    init.db_url = None
    monkeypatch.setitem(sys.modules, 'OCDocker.Initialise', init)

    eng = ocdb.setup_database()
    assert eng is not None


@pytest.mark.order(39)
def test_export_db_to_csv_writes_files(tmp_path):
    # Transient engine + session
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)

    df_out = tmp_path / "out.csv"
    json_out = tmp_path / "out.json"
    csv_out = tmp_path / "out2.csv"

    with Session() as s:
        # DataFrame to file path branch
        rc = ocdb.export_db_to_csv(s, output_format='dataframe', output_file=str(df_out))
        assert rc is None and df_out.exists()

        # JSON to file path branch
        rc2 = ocdb.export_db_to_csv(s, output_format='json', output_file=str(json_out))
        assert rc2 is None and json_out.exists()

        # CSV to file path branch
        rc3 = ocdb.export_db_to_csv(s, output_format='csv', output_file=str(csv_out))
        assert rc3 is None and csv_out.exists()

