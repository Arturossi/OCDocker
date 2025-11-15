#!/usr/bin/env python3

from __future__ import annotations
import pytest

import OCDocker.DB.DB as ocdb
from OCDocker.DB.DBMinimal import create_engine
from sqlalchemy.orm import sessionmaker


def test_export_db_to_csv_returns_strings_on_empty_db():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)

    with Session() as s:
        js = ocdb.export_db_to_csv(s, output_format='json')
        assert isinstance(js, str)
        csv = ocdb.export_db_to_csv(s, output_format='csv')
        assert isinstance(csv, str)


def test_export_db_to_csv_invalid_format_raises():
    engine = create_engine("sqlite:///:memory:")  # type: ignore[arg-type]
    ocdb.create_tables(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        with pytest.raises(ValueError):
            _ = ocdb.export_db_to_csv(s, output_format='invalid')  # type: ignore[arg-type]

