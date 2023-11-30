from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.session import Session
from sqlalchemy_utils import create_database, database_exists
from typing import Tuple

from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complex, Ligand, Receptor

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

def create_engine_and_session(url: str) -> Tuple[Engine, Session]:
    ''' Create the engine and the session.

    Parameters
    ----------
    url : str
        The database url.

    Returns
    -------
    engine : sqlalchemy.engine.base.Engine
        The engine.
    Session : sqlalchemy.orm.session.Session
        The session.
    '''

    # Create the database if it does not exist
    create_database_if_not_exists(url)

    # Create the engine
    engine = create_engine(url, echo = True)

    # Create the session in a scoped session to avoid threading problems
    Session = scoped_session(sessionmaker(bind = engine))

    # Return the engine and the session
    return engine, Session

def create_tables(engine: Engine) -> None:
    ''' Create the tables.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        The engine.
    '''

    # Create the tables
    Base.metadata.create_all(engine)

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
    with db_session() as session:
        # Create the new data
        new_data = table(**payload)
        # Add the new data to the session
        session.add(new_data)
        # Commit the session
        session.commit()

def setup(url: str) -> None:
    ''' Setup the database. '''

    # Create the engine and the session
    engine, Session = create_engine_and_session(url)

    # Create tables (nothing happens if table already exists) :)
    create_tables(engine)
