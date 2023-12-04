
from sqlalchemy.engine.base import Engine

from OCDocker.DB.DBMinimal import create_database_if_not_exists, create_engine
from OCDocker.DB.Models.Base import Base
from OCDocker.DB.Models import Complexes, Ligands, Receptors

from OCDocker.Initialise import db_url, engine

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
    create_database_if_not_exists(str(db_url))

    # Create the engine
    engine = create_engine(str(db_url))

    # Create tables (nothing happens if table already exists) :)
    create_tables()
    
    return engine

