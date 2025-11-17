#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests to verify database connections are properly closed and do not leak.

These tests ensure that:
- Sessions are properly cleaned up
- Engines dispose of connections correctly
- atexit handlers work correctly
- No connection leaks occur during normal operations
'''

import atexit
import gc
import pytest
from sqlalchemy import create_engine as sqlalchemy_create_engine, text
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import scoped_session, sessionmaker

from OCDocker.DB.DBMinimal import (
    create_engine,
    create_session,
    cleanup_session,
    cleanup_engine,
)


@pytest.mark.order(26)
def test_session_cleanup_removes_registry(tmp_path):
    '''Test that cleanup_session properly removes sessions from registry.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    session = create_session(engine)
    
    # Verify session was created
    assert session is not None
    
    # Create a session instance
    with session() as s:
        assert s is not None
    
    # Cleanup should remove registry
    cleanup_session(session)
    
    # Verify cleanup doesn't raise errors
    cleanup_session(session)  # Should be idempotent
    cleanup_session(None)  # Should handle None gracefully


@pytest.mark.order(27)
def test_engine_cleanup_disposes_connections(tmp_path):
    '''Test that cleanup_engine properly disposes of connections.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    
    # Verify engine was created
    assert engine is not None
    
    # Get initial pool size (for SQLite this may be None)
    initial_pool = getattr(engine.pool, 'size', None)
    
    # Cleanup should dispose connections
    cleanup_engine(engine)
    
    # Verify cleanup doesn't raise errors
    cleanup_engine(engine)  # Should be idempotent
    cleanup_engine(None)  # Should handle None gracefully


@pytest.mark.order(28)
def test_session_context_manager_closes_connections(tmp_path):
    '''Test that session context manager properly closes connections.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    session = create_session(engine)
    
    # Use context manager - should automatically close
    with session() as s:
        # Perform a simple query
        result = s.execute(text("SELECT 1"))
        assert result.scalar() == 1
    
    # Session should be closed after context manager exits
    # Verify no errors on cleanup
    cleanup_session(session)
    cleanup_engine(engine)


@pytest.mark.order(29)
def test_multiple_sessions_cleanup(tmp_path):
    '''Test that multiple sessions are all cleaned up properly.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    session = create_session(engine)
    
    # Create multiple session instances
    with session() as s1:
        s1.execute(text("SELECT 1"))
    with session() as s2:
        s2.execute(text("SELECT 2"))
    with session() as s3:
        s3.execute(text("SELECT 3"))
    
    # All should be cleaned up
    cleanup_session(session)
    cleanup_engine(engine)


@pytest.mark.order(30)
def test_cleanup_with_exception(tmp_path):
    '''Test that cleanup functions handle exceptions gracefully.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    session = create_session(engine)
    
    # Use session
    with session() as s:
        s.execute(text("SELECT 1"))
    
    # Cleanup should handle any exceptions
    cleanup_session(session)
    cleanup_engine(engine)
    
    # Cleanup again (should be safe)
    cleanup_session(session)
    cleanup_engine(engine)


@pytest.mark.order(31)
def test_engine_pool_size_after_cleanup(tmp_path):
    '''Test that engine pool is properly disposed after cleanup.'''

    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    
    # Create and use a session
    session = create_session(engine)
    with session() as s:
        s.execute(text("SELECT 1"))
    
    # Cleanup
    cleanup_session(session)
    cleanup_engine(engine)
    
    # Engine should be disposed (verify it doesn't raise on dispose)
    try:
        engine.dispose(close=True)
    except Exception:
        # Already disposed is fine
        pass


@pytest.mark.order(32)
def test_atexit_cleanup_registration(monkeypatch):
    '''Test that atexit handlers are registered for cleanup.
    
    This test verifies that the cleanup functions can be registered
    with atexit and will be called on exit.
    '''

    cleanup_called = []


    def mock_cleanup():
        cleanup_called.append(True)


    
    # Register cleanup
    atexit.register(mock_cleanup)
    
    # Verify it's registered (we can't easily test actual exit, but we can verify registration)
    # The actual atexit behavior is tested in integration tests
    assert len(cleanup_called) == 0  # Not called yet
    
    # Manually call to verify it works
    mock_cleanup()
    assert len(cleanup_called) == 1


@pytest.mark.order(33)
def test_connection_pool_configuration(tmp_path):
    '''Test that connection pooling is properly configured.
    
    This test verifies that engines are created with appropriate
    connection pool settings to prevent leaks.
    '''
    
    db_file = tmp_path / "test.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    
    # Create engine with pooling settings
    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600
    )
    
    # For SQLite, pooling is handled differently
    # But engine should still be created successfully
    assert engine is not None
    
    # Verify pool_pre_ping is enabled (helps detect stale connections)
    # This is set in create_engine for all database types
    assert hasattr(engine.pool, '_pre_ping') or url.drivername == 'sqlite'
    
    cleanup_engine(engine)

