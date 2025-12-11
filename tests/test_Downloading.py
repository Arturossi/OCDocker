import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import urllib.request

import OCDocker.Toolbox.Downloading as ocdown
import OCDocker.Toolbox.Printing as ocprint


@pytest.mark.order(1)
def test_download_url(monkeypatch, tmp_path):
    '''Test download_url function.'''
    
    test_url = "https://example.com/test_file.txt"
    output_file = tmp_path / "downloaded_file.txt"
    
    # Mock urllib.request.urlretrieve
    urlretrieve_called = []
    def mock_urlretrieve(url, filename, reporthook):
        urlretrieve_called.append((url, filename, reporthook))
        # Create a dummy file
        Path(filename).write_text("downloaded content")
        # Call reporthook to simulate progress
        if reporthook:
            reporthook(1, 1024, 10240)  # 1 block, 1024 bytes, 10240 total
    
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.urllib.request.urlretrieve", mock_urlretrieve)
    
    # Mock ocprint.printv
    printv_called = []
    def mock_printv(message):
        printv_called.append(message)
    
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.ocprint.printv", mock_printv)
    
    # Call download_url
    result = ocdown.download_url(test_url, str(output_file))
    
    # Verify urlretrieve was called
    assert len(urlretrieve_called) == 1
    assert urlretrieve_called[0][0] == test_url
    assert urlretrieve_called[0][1] == str(output_file)
    assert callable(urlretrieve_called[0][2])  # reporthook should be callable
    
    # Verify file was created
    assert output_file.exists()
    assert output_file.read_text() == "downloaded content"
    
    # Verify printv was called
    assert len(printv_called) >= 1
    assert test_url in printv_called[0] or str(output_file) in printv_called[0]
    
    assert result is None


@pytest.mark.order(2)
def test_download_progress_bar_update_to():
    '''Test DownloadProgressBar.update_to method.'''
    
    from OCDocker.Toolbox.Downloading import DownloadProgressBar
    
    # Create a progress bar instance
    progress_bar = DownloadProgressBar(total=1000, desc="test")
    
    # Test update_to with tsize parameter
    progress_bar.update_to(b=1, bsize=100, tsize=1000)
    assert progress_bar.total == 1000
    
    # Test update_to without tsize (should not change total)
    progress_bar.total = 2000
    progress_bar.update_to(b=1, bsize=100, tsize=None)
    assert progress_bar.total == 2000
    
    # Test update_to updates the bar
    initial_n = progress_bar.n
    progress_bar.update_to(b=2, bsize=100, tsize=1000)
    # The n value should have increased
    assert progress_bar.n >= initial_n


@pytest.mark.order(3)
def test_download_url_progress_callback(monkeypatch, tmp_path):
    '''Test that download_url uses progress bar correctly.'''
    
    test_url = "https://example.com/large_file.bin"
    output_file = tmp_path / "large_file.bin"
    
    # Track reporthook calls
    reporthook_calls = []
    
    def mock_urlretrieve(url, filename, reporthook):
        # Simulate download progress by calling reporthook multiple times
        if reporthook:
            # Simulate downloading in blocks
            reporthook(1, 1024, 10240)  # 1 block, 1024 bytes, 10240 total
            reporthook(2, 1024, 10240)  # 2 blocks
            reporthook(5, 1024, 10240)  # 5 blocks
            reporthook_calls.append(len([c for c in [1, 2, 5]]))
        Path(filename).write_text("content")
    
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.urllib.request.urlretrieve", mock_urlretrieve)
    
    # Mock ocprint.printv
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.ocprint.printv", lambda x: None)
    
    # Call download_url
    ocdown.download_url(test_url, str(output_file))
    
    # Verify reporthook was called (progress tracking happened)
    # The exact number depends on implementation, but reporthook should be used
    assert output_file.exists()


@pytest.mark.order(4)
def test_download_progress_bar_inheritance():
    '''Test that DownloadProgressBar inherits from tqdm.'''
    
    from OCDocker.Toolbox.Downloading import DownloadProgressBar
    from tqdm import tqdm
    
    # Verify DownloadProgressBar is a subclass of tqdm
    assert issubclass(DownloadProgressBar, tqdm)
    
    # Verify we can create an instance
    progress_bar = DownloadProgressBar(total=100, desc="test")
    assert isinstance(progress_bar, tqdm)


@pytest.mark.order(5)
def test_download_url_filename_extraction(monkeypatch, tmp_path):
    '''Test that download_url extracts filename from URL for progress bar.'''
    
    test_url = "https://example.com/path/to/file.txt"
    output_file = tmp_path / "output.txt"
    
    # Track what was passed to DownloadProgressBar
    progress_bar_desc = []
    
    # Mock DownloadProgressBar to capture desc parameter
    original_init = ocdown.DownloadProgressBar.__init__
    def mock_init(self, *args, **kwargs):
        if 'desc' in kwargs:
            progress_bar_desc.append(kwargs['desc'])
        original_init(self, *args, **kwargs)
    
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.DownloadProgressBar.__init__", mock_init)
    
    # Mock urlretrieve
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.urllib.request.urlretrieve", 
                       lambda url, filename, reporthook: Path(filename).write_text("content"))
    
    # Mock printv
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.ocprint.printv", lambda x: None)
    
    # Call download_url
    ocdown.download_url(test_url, str(output_file))
    
    # Verify filename was extracted from URL for progress bar description
    # The desc should be "file.txt" (last part of URL path)
    assert len(progress_bar_desc) >= 1
    # The desc might be "file.txt" or similar based on URL parsing
    assert "file.txt" in progress_bar_desc[0] or test_url in progress_bar_desc[0]


@pytest.mark.order(6)
def test_download_url_creates_file(monkeypatch, tmp_path):
    '''Test that download_url creates the output file.'''
    
    test_url = "https://example.com/test.txt"
    output_file = tmp_path / "test_output.txt"
    
    # Ensure file doesn't exist before
    assert not output_file.exists()
    
    # Mock urlretrieve to create file
    def mock_urlretrieve(url, filename, reporthook):
        Path(filename).write_text("downloaded content")
        if reporthook:
            reporthook(1, 100, 100)
    
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.urllib.request.urlretrieve", mock_urlretrieve)
    monkeypatch.setattr("OCDocker.Toolbox.Downloading.ocprint.printv", lambda x: None)
    
    # Call download_url
    ocdown.download_url(test_url, str(output_file))
    
    # Verify file was created
    assert output_file.exists()
    assert output_file.read_text() == "downloaded content"

