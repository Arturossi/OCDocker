#!/usr/bin/env python3

# Description
###############################################################################
'''
Sets of classes and functions that are used for setting up the database.

They are imported as:

import OCDocker.DB.DBMinimal as ocdbmin
'''

# Imports
###############################################################################

from sqlalchemy import create_engine as sqlalchemy_create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import create_database, database_exists
from typing import Union, Optional

import OCDocker.Error as ocerror

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


def create_database_if_not_exists(url: Union[str, URL]) -> None:
    ''' Create the database if it does not exist.
    
    Parameters
    ----------
    url : str | sqlalchemy.engine.url.URL
        The database url (string or URL object).
    '''
    
    # Convert string URL to URL object if necessary
    if isinstance(url, str):
        url = make_url(url)

    # If the database does not exist, create it
    if not database_exists(url):
        create_database(url)
    
    return None


def create_engine(url: Union[str, URL], echo: bool = False, pool_size: int = 5, max_overflow: int = 10, pool_timeout: int = 30, pool_recycle: int = 3600) -> Engine:
    ''' Create the engine with connection pooling.

    Parameters
    ----------
    url : str | sqlalchemy.engine.url.URL
        The database url (string or URL object).
    echo : bool
        Echo the SQL commands.
    pool_size : int, optional
        Number of connections to maintain in the pool. Default is 5.
    max_overflow : int, optional
        Maximum number of connections to allow beyond pool_size. Default is 10.
    pool_timeout : int, optional
        Seconds to wait before giving up on getting a connection. Default is 30.
    pool_recycle : int, optional
        Seconds after which a connection is recreated. Default is 3600 (1 hour).

    Returns
    -------
    Engine : sqlalchemy.engine.base.Engine
        The engine with connection pooling configured.
    '''
    
    # Convert string URL to URL object if necessary
    if isinstance(url, str):
        url = make_url(url)

    # Create the engine with connection pooling configuration
    # For SQLite, pooling is handled differently, so we only apply pooling for non-SQLite databases
    if url.drivername == 'sqlite':
        # SQLite doesn't need connection pooling in the same way
        engine = sqlalchemy_create_engine(url, echo=echo, pool_pre_ping=True)
    else:
        # MySQL and other databases: configure connection pooling
        engine = sqlalchemy_create_engine(
            url,
            echo=echo,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True  # Verify connections before using them
        )

    # Return the engine (despite the lint flagging as a MockConnection, it is an Engine)
    return engine # type: ignore


def create_session(engine: Optional[Engine]) -> Optional[scoped_session]:
    ''' Create a scoped session for database operations.

    Parameters
    ----------
    engine : from sqlalchemy.engine.base.Engine | None
        The engine.

    Returns
    -------
    scoped_session : sqlalchemy.orm.scoped_session
        The scoped session factory. Use `with session() as s:` to get a session instance.
    
    Notes
    -----
    Session Lifecycle:
    - Always use context managers: `with session() as s: ...`
    - The context manager automatically handles commit/rollback and closing
    - The scoped_session registry is cleaned up automatically on application shutdown
    - For manual cleanup, call `cleanup_session(session)` or let atexit handlers run
    
    Example
    -------
    ::
    
        session = create_session(engine)
        with session() as s:
            result = s.query(Model).all()
            s.commit()  # Optional - context manager handles this
    '''

    # Check if the engine is defined
    if engine is None:
        # The engine is not defined
        _ = ocerror.Error.engine_not_created("The engine is not defined. Please create the engine first.") # type: ignore
        print("The engine is not defined. Please create the engine first.")
        # Return None
        return None

    # Create the session in a scoped session to avoid threading problems
    # scoped_session provides thread-local session instances
    session = scoped_session(sessionmaker(bind = engine))

    # Return the session
    return session


def cleanup_session(session: Optional[scoped_session]) -> None:
    ''' Clean up a scoped session by removing all sessions from the registry.

    This function removes all thread-local session instances from the scoped_session
    registry. It's automatically called on application shutdown via atexit handlers.
    
    Parameters
    ----------
    session : scoped_session | None
        The scoped session to clean up.
        
    Notes
    -----
    - This is safe to call multiple times (idempotent)
    - Errors during cleanup are silently ignored
    - Typically called automatically on application exit
    '''
    if session is not None:
        try:
            # Remove all thread-local sessions from the registry
            # This closes all active sessions and releases connections
            session.remove()
        except (AttributeError, RuntimeError):
            # Ignore errors during cleanup (session may already be closed or removed)
            pass


def cleanup_engine(engine: Optional[Engine]) -> None:
    ''' Clean up an engine by disposing of all connections in the pool.

    This function closes all connections in the connection pool and disposes of
    the engine. It's automatically called on application shutdown via atexit handlers.
    
    Parameters
    ----------
    engine : Engine | None
        The engine to clean up.
        
    Notes
    -----
    - This is safe to call multiple times (idempotent)
    - Errors during cleanup are silently ignored
    - Typically called automatically on application exit
    - Prevents connection leaks, especially important for MySQL
    '''
    if engine is not None:
        try:
            # Dispose of all connections in the pool
            # close=True ensures connections are properly closed, not just returned to pool
            engine.dispose(close=True)
        except (AttributeError, RuntimeError):
            # Ignore errors during cleanup (engine may already be disposed)
            pass
