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

from sqlalchemy.engine.base import Engine

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

# Setup the database
setup_database()
