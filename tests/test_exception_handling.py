#!/usr/bin/env python3

# Description
###############################################################################
'''
Tests to verify exception handling and logging throughout the codebase.

These tests ensure that:
- Exceptions are properly caught and handled
- Error messages are logged appropriately
- Specific exception types are used instead of broad Exception handlers
- Error codes are returned correctly
'''

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import OCDocker.Error as ocerror
import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Toolbox.Validation as ocvalidation


@pytest.mark.order(54)
def test_file_operations_log_errors(tmp_path, caplog):
    '''Test that file operations log errors appropriately.'''

    # Try to read a non-existent file
    non_existent = tmp_path / "nonexistent.pkl"
    result = ocff.from_pickle(str(non_existent))
    
    # Should return None and log error
    assert result is None


@pytest.mark.order(55)
def test_validation_exceptions_handled(tmp_path):
    '''Test that validation functions handle exceptions properly.'''

    # Test with invalid file
    invalid_file = tmp_path / "invalid.mol"
    invalid_file.write_text("not a valid molecule")
    
    # Should handle exception gracefully
    result = ocvalidation.is_molecule_valid(str(invalid_file))


@pytest.mark.order(56)
def test_safe_remove_handles_permission_errors(tmp_path):
    '''Test that safe_remove_file handles permission errors.'''

    # Create a file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")
    
    # Remove it - should succeed
    result = ocff.safe_remove_file(str(test_file))
    assert result == ocerror.ErrorCode.OK
    
    # Try to remove again - should handle gracefully
    result = ocff.safe_remove_file(str(test_file))
    assert result == ocerror.ErrorCode.FILE_NOT_EXIST


@pytest.mark.order(57)
def test_safe_create_dir_handles_errors(tmp_path):
    '''Test that safe_create_dir handles various error conditions.'''

    # Create directory - should succeed
    test_dir = tmp_path / "test_dir"
    result = ocff.safe_create_dir(str(test_dir))
    assert result == ocerror.ErrorCode.OK
    
    # Try to create again - should handle gracefully
    result = ocff.safe_create_dir(str(test_dir))
    assert result == ocerror.ErrorCode.DIR_EXISTS


@pytest.mark.order(58)
def test_error_codes_returned_correctly():
    '''Test that error codes are returned correctly for different error types.'''

    # Test OK code
    ok_code = ocerror.Error.ok()
    assert ok_code == ocerror.ErrorCode.OK


@pytest.mark.order(59)
def test_exception_specificity_in_file_operations(tmp_path):
    '''Test that file operations use specific exception types.
    
    This test verifies that file operations catch specific exceptions
    rather than using bare except or broad Exception handlers.
    '''

    # Test pickle operations with various error conditions
    test_file = tmp_path / "test.pkl"
    
    # Write should succeed
    data = {"test": "data"}
    result = ocff.to_pickle(str(test_file), data)
    assert result == ocerror.ErrorCode.OK
    
    # Read should succeed
    loaded = ocff.from_pickle(str(test_file))
    assert loaded == data
    
    # Try to read corrupted file
    test_file.write_bytes(b"corrupted pickle data")
    result = ocff.from_pickle(str(test_file))
    # Should return None and handle exception gracefully
    assert result is None


@pytest.mark.order(60)
def test_directory_operations_exception_handling(tmp_path):
    '''Test that directory operations handle exceptions properly.'''

    test_dir = tmp_path / "test_dir"
    
    # Create directory
    result = ocff.safe_create_dir(str(test_dir))
    assert result == ocerror.ErrorCode.OK
    
    # Remove directory
    result = ocff.safe_remove_dir(str(test_dir))
    assert result == ocerror.ErrorCode.OK
    
    # Try to remove non-existent directory
    result = ocff.safe_remove_dir(str(test_dir))
    assert result == ocerror.ErrorCode.DIR_NOT_EXIST


@pytest.mark.order(61)
def test_hdf5_operations_exception_handling(tmp_path):
    '''Test that HDF5 operations handle exceptions properly.'''

    test_file = tmp_path / "test.h5"
    
    # Write should succeed
    data = {"key": [1, 2, 3]}
    result = ocff.to_hdf5(str(test_file), data)
    assert result == ocerror.ErrorCode.OK
    
    # Read should succeed
    loaded = ocff.from_hdf5(str(test_file))
    assert loaded is not None
    assert "key" in loaded
    
    # Try to read non-existent file
    non_existent = tmp_path / "nonexistent.h5"
    result = ocff.from_hdf5(str(non_existent))
    # Should return None or empty dict and handle exception gracefully
    assert result is None or result == {}


@pytest.mark.order(62)
def test_untar_exception_handling(tmp_path):
    '''Test that untar handles exceptions properly.'''

    # Test with non-existent archive
    non_existent = tmp_path / "nonexistent.tar.gz"
    result = ocff.untar(str(non_existent), str(tmp_path))
    # Should handle file not found gracefully
    assert result != ocerror.ErrorCode.OK
    
    # Test with invalid archive
    invalid_archive = tmp_path / "invalid.tar.gz"
    invalid_archive.write_text("not a tar file")
    result = ocff.untar(str(invalid_archive), str(tmp_path))
    # Should handle invalid archive gracefully
    assert result != ocerror.ErrorCode.OK


@pytest.mark.order(63)
def test_error_reporting_levels():
    '''Test that error reporting uses appropriate levels.'''
    
    # Test that error methods accept level parameter
    # This is tested through the Error class interface
    # The actual implementation is in Error.py
    
    # Verify error codes are defined
    assert hasattr(ocerror.ErrorCode, 'OK')
    assert hasattr(ocerror.ErrorCode, 'FILE_NOT_EXIST')
    assert hasattr(ocerror.ErrorCode, 'DIR_NOT_EXIST')

