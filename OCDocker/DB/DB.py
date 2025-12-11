#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used for creating everything required
for the database.

They are imported as:

import OCDocker.DB.DB as ocdb
'''

# Imports
###############################################################################

import csv
import json

import pandas as pd
from typing import Literal, Optional, Union

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session

from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complexes, Ligands, Receptors
import OCDocker.Error as ocerror

# May use session from Initialise - runtime global
try:
    from OCDocker.Initialise import session  # type: ignore
except ImportError:
    session = None  # type: ignore

import OCDocker.Toolbox.Printing as ocprint

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
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


def create_tables(engine: Optional[Engine] = None) -> None:
    '''Create all ORM tables bound to the provided engine.

    If no engine is provided, attempts to resolve the engine from
    OCDocker.Initialise (and creates one from db_url if necessary).
    '''

    eng = engine
    if eng is None:
        try:
            import OCDocker.Initialise as init  # type: ignore
            eng = getattr(init, 'engine', None)
            if eng is None:
                url = getattr(init, 'db_url', None)
                if url is None:
                    raise RuntimeError('Database URL is not configured')
                from OCDocker.DB.DBMinimal import create_engine as _ce  # local import to avoid cycles at import-time
                eng = _ce(url)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f'Could not resolve database engine to create tables: {e}')

    Base.metadata.create_all(eng)  # type: ignore[arg-type]


def setup_database() -> Engine:
    '''
    Ensure the database exists, create a new Engine, and create tables.

    Returns
    -------
    sqlalchemy.engine.base.Engine
        Live engine connected to the configured database URL.
    '''

    # Local import to avoid requiring optional deps at import-time
    from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine

    # Resolve the configured DB URL lazily to avoid import-time side effects
    try:
        import OCDocker.Initialise as init  # type: ignore
        url = getattr(init, 'db_url', None)
        if url is None:
            # Try deriving from an existing engine
            eng = getattr(init, 'engine', None)
            if eng is not None:
                url = eng.url
        # Final fallback suitable for tests/dev
        if url is None:
            url = "sqlite:///:memory:"
    except (ImportError, AttributeError):
        # Extremely defensive fallback for environments without Initialise
        url = "sqlite:///:memory:"

    # Create DB if it does not exist
    create_database_if_not_exists(url)  # type: ignore[arg-type]

    # Create engine and tables
    engine_obj = create_engine(url)  # type: ignore[arg-type]
    create_tables(engine_obj)

    return engine_obj


def export_table_to_csv(model: type[Base], filename: str, session: Session) -> None:
    '''
    Export a single ORM model's rows to CSV.

    Parameters
    ----------
    model : type[Base]
        ORM model class to export.
    filename : str
        Output CSV file path.
    session : sqlalchemy.orm.session.Session
        SQLAlchemy session bound to the database engine.
    '''

    data = session.query(model).all()
    columns = list(model.__table__.columns.keys())

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for row in data:
            writer.writerow([getattr(row, col) for col in columns])


def export_db_to_csv(
    session: Session,
    output_format: Literal['dataframe', 'json', 'csv'] = 'dataframe',
    output_file: Optional[str] = None,
    drop_na: bool = True
) -> Union[pd.DataFrame, str, None]:
    """
    Merge data from Complexes, Ligands, and Receptors tables and export.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        The session object to use for querying the database.
    output_format : {'dataframe','json','csv'}
        Output format. If 'dataframe', returns a DataFrame; for 'json'/'csv' returns a string
        unless `output_file` is provided (then returns None).
    output_file : str | None
        Optional path to write the result to disk.
    drop_na : bool
        If True, drops rows with missing values. Defaults to True.

    Returns
    -------
    pandas.DataFrame | str | None
        DataFrame or serialized string depending on `output_format`; None when writing to `output_file`.
    """
    
    # Query to fetch complexes with their ligands and receptors
    merged_data = session.query(Complexes.Complexes, Ligands.Ligands, Receptors.Receptors)\
        .join(Ligands.Ligands, Ligands.Ligands.id == Complexes.Complexes.ligand_id)\
        .join(Receptors.Receptors, Receptors.Receptors.id == Complexes.Complexes.receptor_id)\
        .all()

    # Prepare the merged result as a list of dictionaries
    result = []
    for complex_obj, ligand, receptor in merged_data:
        # Merge the data from the three tables into a single dictionary removing private attributes and IDs
        merged_entry = {
            'name': complex_obj.name,
            **{key: value for key, value in complex_obj.__dict__.items() if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name', 'ligand_id', 'receptor_id']},
            **{key: value for key, value in ligand.__dict__.items() if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name']},
            **{key: value for key, value in receptor.__dict__.items() if not key.startswith('_') and key not in ['created_at', 'modified_at', 'id', 'name']},
            'receptor': receptor.name,
            'ligand': ligand.name.split('_')[-1] # Extract the ligand name from the ligand filename
        }

        result.append(merged_entry)

    # Get the column order based on the table structure
    complex_columns = [c.name for c in Complexes.Complexes.__table__.columns if c.name not in ['created_at', 'modified_at', 'id', 'name', 'ligand_id', 'receptor_id']]
    ligand_columns = [c.name for c in Ligands.Ligands.__table__.columns if c.name not in ['created_at', 'modified_at', 'id', 'name']]
    receptor_columns = [c.name for c in Receptors.Receptors.__table__.columns if c.name not in ['created_at', 'modified_at', 'id', 'name']]

    # Combine the column lists in the same order as the tables
    column_order = ['name'] + complex_columns + receptor_columns + ligand_columns + ['receptor', 'ligand']

    # Reorder the result based on the column order
    result = [{col: entry.get(col, None) for col in column_order} for entry in result]

    # If drop_na is True, drop rows with any missing values
    if drop_na:
        result = [entry for entry in result if all(value is not None for value in entry.values())]
    
    # If complex_name ends with ligand, set the db column as pdbbind, otherwise set it as dudez
    #result['db'] = result['complex_name'].apply(lambda x: 'pdbbind' if x.endswith('ligand') else 'dudez') # type: ignore

    # Convert the result to a pandas DataFrame
    if output_format == 'dataframe':
        df = pd.DataFrame(result)
        if output_file:
            df.to_csv(output_file, index=False)
            return None
        return df

    # Return data in JSON format
    elif output_format == 'json':
        result_json = json.dumps(result, indent=4)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result_json)
            return None
        return result_json

    # Return data in CSV format
    elif output_format == 'csv':
        if output_file:
            # Extract fieldnames (keys from the first result entry)
            fieldnames = result[0].keys() if result else []
            
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(result)
            return None
        else:
            # Write to string (for return)
            if result:
                output = []
                fieldnames = result[0].keys()  # Use keys from the first dictionary
                output.append(','.join(fieldnames))  # header
                for row in result:
                    output.append(','.join(str(row[field]) for field in fieldnames))
                return '\n'.join(output)
            return ''

    else:
        # User-facing error: invalid output format
        ocerror.Error.value_error(f"Invalid output format: '{output_format}'. Please choose 'dataframe', 'json', or 'csv'.") # type: ignore
        raise ValueError("Invalid output format. Please choose 'dataframe', 'json', or 'csv'.")
    
# Explicit initialization only: call setup_database() from CLI or application bootstrap
    # Local import to avoid requiring optional deps at import-time
    from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine
