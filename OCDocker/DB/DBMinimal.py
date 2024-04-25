from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import create_database, database_exists
from typing import Union

import OCDocker.Error as ocerror

def create_database_if_not_exists(url: URL) -> None:
    ''' Create the database if it does not exist.
    
    Parameters
    ----------
    url : sqlalchemy.engine.url.URL
        The database url.
    '''

    # If the database does not exist, create it
    if not database_exists(url):
        create_database(url)
    
    return None

def create_engine(url: URL, echo: bool = False) -> Engine:
    ''' Create the engine.

    Parameters
    ----------
    url : sqlalchemy.engine.url.URL
        The database url.
    echo : bool
        Echo the SQL commands.

    Returns
    -------
    Engine : sqlalchemy.engine.base.Engine
        The engine.
    '''

    # Create the engine
    engine = sqlalchemy_create_engine(url, echo = echo)

    # Return the engine (despite the lint flagging as a MockConnection, it is an Engine)
    return engine # type: ignore

def create_session(engine: Union[Engine, None]) -> Union[scoped_session, None]:
    ''' Create the session.

    Parameters
    ----------
    engine : from sqlalchemy.engine.base.Engine | None
        The engine.

    Returns
    -------
    scoped_session : sqlalchemy.orm.scoped_session
        The session.
    '''

    # Check if the engine is defined
    if engine is None:
        # The engine is not defined
        _ = ocerror.Error.engine_not_created("The engine is not defined. Please create the engine first.") # type: ignore
        print("The engine is not defined. Please create the engine first.")
        # Return None
        return None

    # Create the session in a scoped session to avoid threading problems
    session = scoped_session(sessionmaker(bind = engine))

    # Return the session
    return session
