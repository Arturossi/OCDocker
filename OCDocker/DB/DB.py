#!/usr/bin/env python3

# Description
###############################################################################
"""
Sets of classes and functions that are used for creating everything required
for the database.

Usage:

import OCDocker.DB.DB as ocdb
"""

# Imports
###############################################################################
import csv
import json

from io import StringIO
from typing import Any, Dict, Iterator, Literal, Optional, Union

import pandas as pd
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session

import OCDocker.Error as ocerror

from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complexes, Ligands, Receptors

# License
###############################################################################
"""Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
"""

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##


def _iter_query_rows(query: Any, batch_size: int = 1000) -> Iterator[Any]:
    """Iterate query rows using streaming when supported.

    Parameters
    ----------
    query : Any
        SQLAlchemy query-like object.
    batch_size : int, optional
        Preferred streaming batch size for query backends that support it.

    Returns
    -------
    Iterator[Any]
        Iterator over query rows.
    """

    streamed = query
    if hasattr(streamed, "yield_per"):
        try:
            streamed = streamed.yield_per(batch_size)
        except Exception:
            streamed = query

    yielded_any = False
    try:
        for row in streamed:
            yielded_any = True
            yield row
        return
    except Exception:
        # Some query shapes (eager loaders) cannot be streamed with yield_per.
        if yielded_any:
            raise

    if streamed is not query:
        try:
            for row in query:
                yield row
            return
        except Exception:
            pass

    all_method = getattr(query, "all", None)
    if callable(all_method):
        for row in all_method():
            yield row


def _build_export_entry(
    complex_obj: Any,
    ligand: Any,
    receptor: Any,
    complex_columns: list[str],
    ligand_columns: list[str],
    receptor_columns: list[str],
    column_order: list[str],
) -> Dict[str, Any]:
    """Build one merged export row with stable column ordering."""

    ligand_name = getattr(ligand, "name", None)
    receptor_name = getattr(receptor, "name", None)
    merged_entry: Dict[str, Any] = {
        "name": getattr(complex_obj, "name", None),
        **{col: getattr(complex_obj, col, None) for col in complex_columns},
        **{col: getattr(receptor, col, None) for col in receptor_columns},
        **{col: getattr(ligand, col, None) for col in ligand_columns},
        "receptor": receptor_name,
        "ligand": (
            ligand_name.split("_")[-1] if isinstance(ligand_name, str) else ligand_name
        ),
    }

    return {col: merged_entry.get(col, None) for col in column_order}


def _iter_export_entries(
    session: Session,
    complex_columns: list[str],
    ligand_columns: list[str],
    receptor_columns: list[str],
    column_order: list[str],
    drop_na: bool,
    batch_size: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """Yield merged export rows from joined DB tables without materializing `.all()`."""

    query = (
        session.query(Complexes.Complexes, Ligands.Ligands, Receptors.Receptors)
        .join(Ligands.Ligands, Ligands.Ligands.id == Complexes.Complexes.ligand_id)
        .join(
            Receptors.Receptors,
            Receptors.Receptors.id == Complexes.Complexes.receptor_id,
        )
    )

    for complex_obj, ligand, receptor in _iter_query_rows(query, batch_size=batch_size):
        row = _build_export_entry(
            complex_obj,
            ligand,
            receptor,
            complex_columns,
            ligand_columns,
            receptor_columns,
            column_order,
        )
        if drop_na and any(value is None for value in row.values()):
            continue
        yield row


def _get_default_session_factory() -> Any:
    from OCDocker.DB.DBMinimal import DatabaseConfigurationError, get_default_session

    session_factory = get_default_session()
    if session_factory is None:
        raise DatabaseConfigurationError(
            "Default database session is not initialized. "
            "Pass a session explicitly or call OCDocker.Initialise.bootstrap(..., init_db=True)."
        )
    return session_factory


def _resolve_engine(engine: Optional[Engine] = None) -> Engine:
    if engine is not None:
        return engine

    from OCDocker.DB.DBMinimal import (
        DatabaseConfigurationError,
        create_engine,
        get_default_engine,
    )

    default_engine = get_default_engine()
    if default_engine is not None:
        return default_engine

    try:
        import OCDocker.Initialise as init
    except ImportError as exc:
        raise DatabaseConfigurationError(
            "Database engine is not initialized and OCDocker.Initialise is unavailable."
        ) from exc

    url = getattr(init, "db_url", None)
    if url is None:
        raise DatabaseConfigurationError(
            "Database URL is not configured. Pass an engine explicitly or bootstrap with init_db=True."
        )

    return create_engine(url)


## Public ##


def create_tables(engine: Optional[Engine] = None) -> None:
    '''Create all ORM tables bound to the provided or default engine.

    If no engine is provided, the default engine must have been initialized
    explicitly through application/CLI bootstrap.
    '''

    try:
        eng = _resolve_engine(engine)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"Could not resolve database engine to create tables: {exc}"
        ) from exc

    Base.metadata.create_all(eng)


def export_db_to_csv(
    session: Optional[Session] = None,
    output_format: Literal["dataframe", "json", "csv"] = "dataframe",
    output_file: Optional[str] = None,
    drop_na: bool = True,
    batch_size: int = 1000,
) -> Union[pd.DataFrame, str, None]:
    '''
    Merge data from Complexes, Ligands, and Receptors tables and export.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session, optional
        The session object to use for querying the database. If omitted, the
        explicitly initialized default session factory is used.
    output_format : {'dataframe','json','csv'}
        Output format. If 'dataframe', returns a DataFrame; for 'json'/'csv' returns a string
        unless `output_file` is provided (then returns None).
    output_file : str | None
        Optional path to write the result to disk.
    drop_na : bool
        If True, drops rows with missing values. Defaults to True.
    batch_size : int
        Streaming batch size for DB row iteration. Defaults to 1000.

    Returns
    -------
    pandas.DataFrame | str | None
        DataFrame or serialized string depending on `output_format`; None when writing to `output_file`.
    '''

    if session is None:
        session_factory = _get_default_session_factory()
        with session_factory() as db_session:
            return export_db_to_csv(
                db_session,
                output_format=output_format,
                output_file=output_file,
                drop_na=drop_na,
                batch_size=batch_size,
            )

    # Get the column order based on the table structure
    complex_columns = [
        c.name
        for c in Complexes.Complexes.__table__.columns
        if c.name
        not in ["created_at", "modified_at", "id", "name", "ligand_id", "receptor_id"]
    ]
    ligand_columns = [
        c.name
        for c in Ligands.Ligands.__table__.columns
        if c.name not in ["created_at", "modified_at", "id", "name"]
    ]
    receptor_columns = [
        c.name
        for c in Receptors.Receptors.__table__.columns
        if c.name not in ["created_at", "modified_at", "id", "name"]
    ]

    # Combine the column lists in the same order as the tables
    column_order = (
        ["name"]
        + complex_columns
        + receptor_columns
        + ligand_columns
        + ["receptor", "ligand"]
    )

    if output_format == "dataframe":
        if output_file:
            with open(output_file, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=column_order)
                writer.writeheader()
                for row in _iter_export_entries(
                    session,
                    complex_columns,
                    ligand_columns,
                    receptor_columns,
                    column_order,
                    drop_na=drop_na,
                    batch_size=batch_size,
                ):
                    writer.writerow(row)
            return None

        frames: list[pd.DataFrame] = []
        chunk: list[Dict[str, Any]] = []
        for row in _iter_export_entries(
            session,
            complex_columns,
            ligand_columns,
            receptor_columns,
            column_order,
            drop_na=drop_na,
            batch_size=batch_size,
        ):
            chunk.append(row)
            if len(chunk) >= batch_size:
                frames.append(pd.DataFrame(chunk, columns=column_order))
                chunk = []
        if chunk:
            frames.append(pd.DataFrame(chunk, columns=column_order))
        if not frames:
            return pd.DataFrame(columns=column_order)
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    if output_format == "json":
        output_buffer: Optional[StringIO] = None
        if output_file:
            handle: Any = open(output_file, "w", encoding="utf-8")
        else:
            output_buffer = StringIO()
            handle = output_buffer

        try:
            handle.write("[")
            wrote_any = False
            for row in _iter_export_entries(
                session,
                complex_columns,
                ligand_columns,
                receptor_columns,
                column_order,
                drop_na=drop_na,
                batch_size=batch_size,
            ):
                if wrote_any:
                    handle.write(",")
                handle.write(json.dumps(row))
                wrote_any = True
            handle.write("]")
        finally:
            if output_file:
                handle.close()

        if output_file:
            return None
        return output_buffer.getvalue() if output_buffer is not None else "[]"

    if output_format == "csv":
        csv_output_buffer: Optional[StringIO] = None
        if output_file:
            handle = open(output_file, "w", newline="")
        else:
            csv_output_buffer = StringIO()
            handle = csv_output_buffer

        try:
            writer = csv.DictWriter(handle, fieldnames=column_order)
            wrote_any = False
            for row in _iter_export_entries(
                session,
                complex_columns,
                ligand_columns,
                receptor_columns,
                column_order,
                drop_na=drop_na,
                batch_size=batch_size,
            ):
                if not wrote_any:
                    writer.writeheader()
                    wrote_any = True
                writer.writerow(row)
        finally:
            if output_file:
                handle.close()

        if output_file:
            return None
        return csv_output_buffer.getvalue() if csv_output_buffer is not None else ""

    ocerror.Error.value_error(
        f"Invalid output format: '{output_format}'. Please choose 'dataframe', 'json', or 'csv'."
    )
    raise ValueError(
        "Invalid output format. Please choose 'dataframe', 'json', or 'csv'."
    )


def export_table_to_csv(
    model: type[Base],
    filename: str,
    session: Optional[Session] = None,
    batch_size: int = 1000,
) -> None:
    '''
    Export a single ORM model's rows to CSV.

    Parameters
    ----------
    model : type[Base]
        ORM model class to export.
    filename : str
        Output CSV file path.
    session : sqlalchemy.orm.session.Session, optional
        SQLAlchemy session bound to the database engine. If omitted, the
        explicitly initialized default session factory is used.
    batch_size : int
        Streaming batch size for DB row iteration. Defaults to 1000.
    '''

    if session is None:
        session_factory = _get_default_session_factory()
        with session_factory() as db_session:
            return export_table_to_csv(
                model,
                filename,
                session=db_session,
                batch_size=batch_size,
            )

    columns = list(model.__table__.columns.keys())

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        query = session.query(model)
        for row in _iter_query_rows(query, batch_size=batch_size):
            writer.writerow([getattr(row, col) for col in columns])


# Explicit initialization only: call setup_database() from CLI or application bootstrap
def setup_database(
    url: Any = None,
    *,
    create_if_missing: bool = False,
    engine: Optional[Engine] = None,
) -> Engine:
    '''
    Ensure the database exists, create a new Engine, and create tables.

    Parameters
    ----------
    url : Any, optional
        Explicit SQLAlchemy URL/string. If omitted, resolves from explicitly
        initialized ``OCDocker.Initialise`` state or falls back to SQLite memory.
    create_if_missing : bool, optional
        Explicitly create a missing PostgreSQL/MySQL database.
    engine : Engine, optional
        Existing engine to bind tables to.

    Returns
    -------
    sqlalchemy.engine.base.Engine
        Live engine connected to the configured database URL.
    '''

    from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine

    if engine is not None:
        engine_obj = engine
        create_tables(engine_obj)
        return engine_obj

    # Resolve the configured DB URL lazily to avoid import-time side effects
    if url is None:
        try:
            import OCDocker.Initialise as init

            url = getattr(init, "db_url", None)
            if url is None:
                # Try deriving from an existing engine
                eng = getattr(init, "engine", None)
                if eng is not None:
                    url = eng.url
            # Final fallback suitable for tests/dev
            if url is None:
                url = "sqlite:///:memory:"
        except (ImportError, AttributeError):
            # Extremely defensive fallback for environments without Initialise
            url = "sqlite:///:memory:"

    # Create DB if it does not exist and the caller explicitly allowed creation.
    create_database_if_not_exists(url, create_if_missing=create_if_missing)

    # Create engine and tables
    engine_obj = create_engine(url)
    create_tables(engine_obj)

    return engine_obj
