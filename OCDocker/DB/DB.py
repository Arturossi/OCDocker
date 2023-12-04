from sqlalchemy import create_engine
from sqlalchemy.engine.mock import MockConnection
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy_utils import create_database, database_exists
from typing import Union

from OCDocker.Initialise import session
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complexes, Ligands, Receptors

def create_database_if_not_exists(url: str) -> None:
    ''' Create the database if it does not exist.
    
    Parameters
    ----------
    url : str
        The database url.
    '''

    # If the database does not exist, create it
    if not database_exists(url):
        create_database(url)
    
    return None

def create_engine_and_session(url: str) -> MockConnection:
    ''' Create the engine and the session.

    Parameters
    ----------
    url : str
        The database url.

    Returns
    -------
    MockConnection : sqlalchemy.engine.mock.MockConnection
        The engine.
    '''

    # Make Session global
    global Session

    # Create the database if it does not exist
    create_database_if_not_exists(url)

    # Create the engine
    engine = create_engine(url, echo = True)

    # Create the session in a scoped session to avoid threading problems
    Session = scoped_session(sessionmaker(bind = engine))

    # Return the engine
    return engine

def create_tables(engine: MockConnection) -> None:
    ''' Create the tables.

    Parameters
    ----------
    MockConnection : sqlalchemy.engine.mock.MockConnection
        The engine.
    '''

    # Create the tables
    Base.metadata.create_all(engine)

    return None

def insert_data(table: DeclarativeMeta, payload: dict) -> None:
    ''' Insert data into the database.

    Parameters
    ----------
    table : sqlalchemy.ext.declarative.DeclarativeMeta
        The table.
    payload : dict
        The data to be inserted.
    '''

    # Open the session
    with session() as s:
        # Create the new data
        new_data = table(**payload)
        # Add the new data to the session
        s.add(new_data)
        # Commit the session
        s.commit()
 
    return None

def setup(url: str) -> None:
    ''' Setup the database. '''

    # Create the engine and the session
    engine = create_engine_and_session(url)

    # Create tables (nothing happens if table already exists) :)
    create_tables(engine)
    
    return None
