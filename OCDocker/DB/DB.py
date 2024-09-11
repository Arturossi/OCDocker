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

from sqlalchemy.engine.base import Engine
from sqlalchemy.orm.session import Session

from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complexes, Ligands, Receptors

from OCDocker.Initialise import db_url, engine

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Torres, P.H.M.;
[The Federal University of Rio de Janeiro]
Contact info:
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics
Av. Carlos Chagas Filho 373 - CCS - bloco G1-19,
Cidade Universitária - Rio de Janeiro, RJ, CEP: 21941-902
E-mail address: arturossi10@gmail.com
This project is licensed under Creative Commons license (CC-BY-4.0) (Ver qual)
'''

# Classes
###############################################################################

# Functions
###############################################################################
## Private ##

## Public ##

def create_tables() -> None:
    ''' Create the tables.

    Parameters
    ----------
    MockConnection : sqlalchemy.engine.mock.MockConnection
        The engine.
    '''

    # Create the tables
    Base.metadata.create_all(engine) # type: ignore

    return None

def setup_database() -> Engine:
    ''' Setup the database. 
    
    Parameters
    ----------
    db_url : str | sqlalchemy.engine.url.URL
        The database url.
        
    Returns
    -------
    Engine : sqlalchemy.engine.base.Engine
        The engine.
    '''

    # Create the database if it does not exist
    create_database_if_not_exists(db_url)

    # Create the engine
    engine = create_engine(db_url)

    # Create tables (nothing happens if table already exists) :)
    create_tables()
    
    return engine

def export_table_to_csv(model: Base, filename: str, session: Session) -> None:
    ''' Export a table to a CSV file.
    
    Parameters
    ----------
    model : Base
        The model.
    filename : str
        The filename.
    session : sqlalchemy.orm.session.Session
        The session.
    '''

    # Fetch all records
    data = session.query(model).all()
    
    # Get the column names
    columns = model.__table__.columns.keys()

    # Write to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header
        for row in data:
            writer.writerow([getattr(row, col) for col in columns])

def export_db_to_csv(session, output_format='dataframe', output_file=None, drop_na=True):
    """
    Merges data from Complexes, Ligands, and Receptors tables.
    
    Parameters:
    -----------
    session : sqlalchemy.orm.session.Session
        The session object to use for querying the database.
    output_format : str
        The format of the output, either 'dataframe', 'json', or 'csv'. Defaults to 'dataframe'.
    output_file : str | None
        If provided, writes the result to this file, otherwise returns the result as a string.
    drop_na : bool
        If True, drops rows with missing values. Defaults to True.
    
    Returns:
    --------
    str | None
        Merged data in JSON or CSV format. Returns None if writing to a file.
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

    column_order = [
        'complex_name', 'receptor_name', 'ligand_name',
        'SMINA_VINA', 'SMINA_SCORING_DKOES', 'SMINA_VINARDO', 'SMINA_OLD_SCORING_DKOES', 'SMINA_FAST_DKOES', 'SMINA_SCORING_AD4',
        'VINA_VINA', 'VINA_VINARDO', 'PLANTS_CHEMPLP', 'PLANTS_PLP', 'PLANTS_PLP95',
        'ODDT_RFSCORE_V1', 'ODDT_RFSCORE_V2', 'ODDT_RFSCORE_V3', 'ODDT_PLECRF_P5_L1_S65536', 'ODDT_NNSCORE'
    ]

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
        raise ValueError("Invalid output format. Please choose 'dataframe', 'json', or 'csv'.")
    
# Setup the database
setup_database()
