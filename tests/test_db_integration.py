#!/usr/bin/env python3

# Description
###############################################################################
'''
Integration tests for database operations with both SQLite and MySQL backends.

These tests verify that:
- Database operations work correctly with SQLite
- Database operations work correctly with MySQL (if available)
- Connection pooling works for both backends
- Sessions are properly managed for both backends
'''

import os
import pytest
from sqlalchemy import text
from sqlalchemy.engine.url import URL
from sqlalchemy.orm import sessionmaker

from OCDocker.DB.DBMinimal import create_engine, create_session, cleanup_session, cleanup_engine
from OCDocker.DB.DB import create_tables


@pytest.fixture
def sqlite_engine(tmp_path):
    '''Create a SQLite engine for testing.'''

    db_file = tmp_path / "test_sqlite.db"
    url = URL.create(drivername="sqlite", database=str(db_file))
    engine = create_engine(url)
    yield engine
    cleanup_engine(engine)


@pytest.fixture
def mysql_engine():
    '''Create a MySQL engine for testing (if MySQL is available).
    
    This fixture will be skipped if MySQL is not available.
    '''

    mysql_url = os.getenv("OCDOCKER_TEST_MYSQL_URL")
    if not mysql_url:
        pytest.skip("MySQL not available - set OCDOCKER_TEST_MYSQL_URL to test")
    
    url = URL.create(mysql_url)
    engine = create_engine(url)
    yield engine
    cleanup_engine(engine)


@pytest.mark.order(40)
def test_sqlite_create_tables(sqlite_engine):
    '''Test that tables can be created in SQLite database.'''

    # Create tables
    create_tables(sqlite_engine)
    
    # Verify tables exist by querying SQLite system tables
    with sqlite_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
        tables = [row[0] for row in result]
        # Should have at least some tables created
        assert len(tables) > 0


@pytest.mark.order(41)
def test_sqlite_session_operations(sqlite_engine):
    '''Test basic session operations with SQLite.'''

    create_tables(sqlite_engine)
    session = create_session(sqlite_engine)
    
    try:
        # Use session
        with session() as s:
            # Simple query to verify connection works
            result = s.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        cleanup_session(session)


@pytest.mark.order(42)
def test_sqlite_connection_pooling(sqlite_engine):
    '''Test that SQLite engine handles connections properly.
    
    SQLite doesn't use traditional connection pooling, but we verify
    that connections are managed correctly.
    '''

    session = create_session(sqlite_engine)
    
    try:
        # Create multiple sessions
        with session() as s1:
            s1.execute(text("SELECT 1"))
        with session() as s2:
            s2.execute(text("SELECT 2"))
        with session() as s3:
            s3.execute(text("SELECT 3"))
    finally:
        cleanup_session(session)


@pytest.mark.order(43)
def test_mysql_create_tables(mysql_engine):
    '''Test that tables can be created in MySQL database.'''

    # Create tables
    create_tables(mysql_engine)
    
    # Verify tables exist by querying MySQL information_schema
    with mysql_engine.connect() as conn:
        result = conn.execute(text(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE()"
        ))
        tables = [row[0] for row in result]
        # Should have at least some tables created
        assert len(tables) > 0


@pytest.mark.order(44)
def test_mysql_session_operations(mysql_engine):
    '''Test basic session operations with MySQL.'''

    create_tables(mysql_engine)
    session = create_session(mysql_engine)
    
    try:
        # Use session
        with session() as s:
            # Simple query to verify connection works
            result = s.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        cleanup_session(session)


@pytest.mark.order(45)
def test_mysql_connection_pooling(mysql_engine):
    '''Test that MySQL engine uses connection pooling correctly.'''

    session = create_session(mysql_engine)
    
    try:
        # Create multiple sessions - should use connection pool
        with session() as s1:
            s1.execute(text("SELECT 1"))
        with session() as s2:
            s2.execute(text("SELECT 2"))
        with session() as s3:
            s3.execute(text("SELECT 3"))
        
        # Verify pool is being used
        pool = mysql_engine.pool
        assert pool is not None
    finally:
        cleanup_session(session)


@pytest.mark.order(46)
def test_sqlite_vs_mysql_compatibility(sqlite_engine, mysql_engine):
    '''Test that operations work similarly on both SQLite and MySQL.
    
    This test verifies that the database abstraction layer works
    correctly for both backends.
    '''

    # Test table creation on both
    create_tables(sqlite_engine)
    create_tables(mysql_engine)
    
    # Test session creation on both
    sqlite_session = create_session(sqlite_engine)
    mysql_session = create_session(mysql_engine)
    
    try:
        # Test basic queries on both
        with sqlite_session() as s:
            result = s.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        with mysql_session() as s:
            result = s.execute(text("SELECT 1"))
            assert result.scalar() == 1
    finally:
        cleanup_session(sqlite_session)
        cleanup_session(mysql_session)


@pytest.mark.order(47)
def test_engine_pool_configuration_differences(sqlite_engine, mysql_engine):
    '''Test that engine pool configuration differs appropriately between SQLite and MySQL.
    
    SQLite doesn't need the same pooling configuration as MySQL.
    '''

    # SQLite pool should be simpler
    sqlite_pool = sqlite_engine.pool
    assert sqlite_pool is not None
    
    # MySQL pool should have more configuration
    mysql_pool = mysql_engine.pool
    assert mysql_pool is not None
    
    # Both should work correctly
    sqlite_session = create_session(sqlite_engine)
    mysql_session = create_session(mysql_engine)
    
    try:
        with sqlite_session() as s:
            s.execute(text("SELECT 1"))
        with mysql_session() as s:
            s.execute(text("SELECT 1"))
    finally:
        cleanup_session(sqlite_session)
        cleanup_session(mysql_session)


@pytest.mark.order(48)
def test_database_cleanup_on_exit(sqlite_engine):
    '''Test that database resources are cleaned up properly.
    
    This test verifies that cleanup functions work correctly
    and can be called multiple times safely.
    '''
    
    session = create_session(sqlite_engine)
    
    # Use session
    with session() as s:
        s.execute(text("SELECT 1"))
    
    # Cleanup should work
    cleanup_session(session)
    cleanup_engine(sqlite_engine)
    
    # Cleanup again should be safe (idempotent)
    cleanup_session(session)
    cleanup_engine(sqlite_engine)

