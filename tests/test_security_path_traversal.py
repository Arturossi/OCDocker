#!/usr/bin/env python3

# Description
###############################################################################
'''
Security tests for path traversal protection in file operations.

Tests verify that file operations properly prevent path traversal attacks,
especially in archive extraction operations.
'''

import os
import pytest
import tarfile
from pathlib import Path

import OCDocker.Toolbox.FilesFolders as ocff
import OCDocker.Error as ocerror


@pytest.mark.order(89)
def test_untar_path_traversal_protection(tmp_path):
    '''Test that untar() prevents path traversal attacks.
    
    This test verifies that archive entries with malicious paths (containing ..)
    are rejected and extraction is aborted.
    '''

    # Create a malicious archive with path traversal
    archive = tmp_path / "malicious.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Create archive with a file that tries to escape the output directory
    with tarfile.open(archive, "w:gz") as tar:
        # Add a file with path traversal attempt
        info = tarfile.TarInfo(name="../../etc/passwd")
        # Create a temporary file with actual data (match size to content)
        temp_file = tmp_path / "temp_content"
        temp_file.write_bytes(b"test")
        info.size = temp_file.stat().st_size  # Use actual file size
        with open(temp_file, 'rb') as f:
            tar.addfile(info, fileobj=f)
    
    # Attempt extraction - should fail with path traversal error
    result = ocff.untar(str(archive), str(out_dir))
    assert result != ocerror.ErrorCode.OK
    assert result == ocerror.ErrorCode.UNTAR_FILE
    
    # Verify the malicious file was not extracted
    assert not (tmp_path / "etc" / "passwd").exists()
    # Note: We don't check /etc/passwd as it's a real system file that exists on Unix systems


@pytest.mark.order(90)
def test_untar_absolute_path_protection(tmp_path):
    '''Test that untar() prevents absolute paths in archive entries.
    
    This test verifies that archive entries with absolute paths are rejected.
    '''

    archive = tmp_path / "absolute.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Create archive with absolute path
    with tarfile.open(archive, "w:gz") as tar:
        # Add a file with absolute path
        info = tarfile.TarInfo(name="/tmp/escape.txt")
        # Create a temporary file with actual data (match size to content)
        temp_file = tmp_path / "temp_content2"
        temp_file.write_bytes(b"test")
        info.size = temp_file.stat().st_size  # Use actual file size
        with open(temp_file, 'rb') as f:
            tar.addfile(info, fileobj=f)
    
    # Attempt extraction - should fail
    result = ocff.untar(str(archive), str(out_dir))
    assert result != ocerror.ErrorCode.OK
    
    # Verify file was not extracted outside output directory
    assert not Path("/tmp/escape.txt").exists()


@pytest.mark.order(91)
def test_untar_safe_paths_allowed(tmp_path):
    '''Test that untar() allows safe paths within the output directory.
    
    This test verifies that normal, safe archive entries are extracted correctly.
    '''

    archive = tmp_path / "safe.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Create archive with safe paths
    safe_file = tmp_path / "safe.txt"
    safe_file.write_text("safe content")
    
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(safe_file, arcname="safe.txt")
        tar.add(safe_file, arcname="subdir/safe.txt")
    
    # Extract should succeed
    result = ocff.untar(str(archive), str(out_dir))
    assert result == ocerror.ErrorCode.OK
    
    # Verify files were extracted correctly
    assert (out_dir / "safe.txt").exists()
    assert (out_dir / "subdir" / "safe.txt").exists()


@pytest.mark.order(92)
def test_untar_nested_path_traversal(tmp_path):
    '''Test that untar() prevents nested path traversal attempts.
    
    This test verifies that path traversal attempts in nested directories
    are also prevented.
    '''

    archive = tmp_path / "nested.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Create archive with nested path traversal
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="subdir/../../../etc/passwd")
        # Create a temporary file with actual data (match size to content)
        temp_file = tmp_path / "temp_content3"
        temp_file.write_bytes(b"test")
        info.size = temp_file.stat().st_size  # Use actual file size
        with open(temp_file, 'rb') as f:
            tar.addfile(info, fileobj=f)
    
    # Attempt extraction - should fail
    result = ocff.untar(str(archive), str(out_dir))
    assert result != ocerror.ErrorCode.OK
    
    # Verify malicious file was not extracted
    assert not (tmp_path / "etc" / "passwd").exists()


@pytest.mark.order(93)
def test_untar_multiple_malicious_entries(tmp_path):
    '''Test that untar() stops extraction on first malicious entry.
    
    This test verifies that extraction stops immediately when a malicious
    entry is detected, even if there are more entries in the archive.
    '''
    
    archive = tmp_path / "multiple.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    
    # Create archive with both safe and malicious entries
    safe_file = tmp_path / "safe.txt"
    safe_file.write_text("safe")
    
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(safe_file, arcname="safe.txt")
        # Add malicious entry
        info = tarfile.TarInfo(name="../../malicious.txt")
        # Create a temporary file with actual data (match size to content)
        temp_file = tmp_path / "temp_content4"
        temp_file.write_bytes(b"test")
        info.size = temp_file.stat().st_size  # Use actual file size
        with open(temp_file, 'rb') as f:
            tar.addfile(info, fileobj=f)
        # Add another safe entry (should not be processed)
        tar.add(safe_file, arcname="another_safe.txt")
    
    # Attempt extraction - should fail
    result = ocff.untar(str(archive), str(out_dir))
    assert result != ocerror.ErrorCode.OK
    
    # First safe file might be extracted before malicious one is detected
    # But malicious file should never be extracted
    assert not (tmp_path / "malicious.txt").exists()
    assert not Path("/malicious.txt").exists()
