#!/usr/bin/env python3

# Description
###############################################################################
'''
Security tests for path traversal protection in file operations.

Tests verify that file operations properly prevent path traversal attacks,
especially in archive extraction operations.
'''

# Imports
###############################################################################
import os
import pytest
import tarfile

from pathlib import Path

import OCDocker.Error as ocerror
import OCDocker.Toolbox.FilesFolders as ocff

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##

## Public ##

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


@pytest.mark.order(94)
def test_untar_rejects_symlink_member(tmp_path):
    '''Test that untar() rejects symlink entries in archives.'''

    archive = tmp_path / "symlink_attack.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with tarfile.open(archive, "w:gz") as tar:
        symlink_info = tarfile.TarInfo(name="safe_link")
        symlink_info.type = tarfile.SYMTYPE
        symlink_info.linkname = "/tmp/escape_target"
        tar.addfile(symlink_info)

        payload = tmp_path / "payload.txt"
        payload.write_bytes(b"evil")
        payload_info = tarfile.TarInfo(name="safe_link/payload.txt")
        payload_info.size = payload.stat().st_size
        with open(payload, "rb") as f:
            tar.addfile(payload_info, fileobj=f)

    result = ocff.untar(str(archive), str(out_dir))
    assert result == ocerror.ErrorCode.UNTAR_FILE
    assert not (out_dir / "safe_link").exists()


@pytest.mark.order(95)
def test_untar_rejects_hardlink_member(tmp_path):
    '''Test that untar() rejects hardlink entries in archives.'''

    archive = tmp_path / "hardlink_attack.tar.gz"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with tarfile.open(archive, "w:gz") as tar:
        regular_file = tmp_path / "base.txt"
        regular_file.write_bytes(b"base")
        tar.add(regular_file, arcname="base.txt")

        hardlink_info = tarfile.TarInfo(name="hardlink_payload")
        hardlink_info.type = tarfile.LNKTYPE
        hardlink_info.linkname = "../../etc/passwd"
        tar.addfile(hardlink_info)

    result = ocff.untar(str(archive), str(out_dir))
    assert result == ocerror.ErrorCode.UNTAR_FILE
    # The regular file added before the malicious entry may exist.
    assert not (out_dir / "hardlink_payload").exists()


@pytest.mark.order(96)
def test_untar_rejects_parent_symlink_escape(tmp_path):
    '''Test that untar() rejects writes through existing symlinks in the output tree.'''

    archive = tmp_path / "parent_symlink_escape.tar.gz"
    out_dir = tmp_path / "out"
    outside_dir = tmp_path / "outside"
    out_dir.mkdir()
    outside_dir.mkdir()

    link_dir = out_dir / "linkdir"
    try:
        os.symlink(outside_dir, link_dir)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported in this environment.")

    payload = tmp_path / "payload2.txt"
    payload.write_bytes(b"escape")

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="linkdir/escaped.txt")
        info.size = payload.stat().st_size
        with open(payload, "rb") as f:
            tar.addfile(info, fileobj=f)

    result = ocff.untar(str(archive), str(out_dir))
    assert result == ocerror.ErrorCode.UNTAR_FILE
    assert not (outside_dir / "escaped.txt").exists()
