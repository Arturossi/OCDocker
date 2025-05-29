
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
from unittest.mock import patch
from OCDocker.Error import Error, ErrorCode, ReportLevel

import re

def remove_ansi_codes(text: str) -> str:
    """Removes ANSI color codes and timestamps from log messages.
    
    Parameters
    ----------
    text : str
        The log message to clean.
    
    Returns
    -------
    str
        The cleaned log message.
    """

    text = re.sub(r'\x1B\[[0-9;]*[mK]', '', text).strip()  # Remove ANSI color codes
    text = re.sub(r'\[\d{2}-\d{2}-\d{4}\|\d{2}:\d{2}:\d{2}\]', '', text).strip()  # Remove timestamps

    return text

class TestErrorMethods(unittest.TestCase):
    """
    Unit tests for the Error class methods.
    """

    def setUp(self):
        """Mock `print` for all tests and initialize `Error` object."""
        self.patcher = patch('builtins.print')
        self.mock_print = self.patcher.start()
        self.error = Error()  # Now all tests use `self.error`

    def tearDown(self):
        """Stop mocking `print` after each test."""
        self.patcher.stop()

    def test_01_abort(self):
        """
        Test that the abort method returns the correct error code and prints a warning message.
        """
        result = self.error.abort("Operation aborted")  # Now uses `self.error`
        self.assertEqual(result, ErrorCode.ABORT.value)

        # Improved assertion
        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("WARNING", mock_print_output)
        self.assertIn("Operation aborted", mock_print_output)

    def test_02_broken_pipe(self):
        """Test handling of broken_pipe error with a predefined message."""
        result = self.error.broken_pipe("Broken pipe")
        self.assertEqual(result, ErrorCode.BROKEN_PIPE.value)

        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("WARNING", mock_print_output)
        self.assertIn("Broken pipe", mock_print_output)

    def test_03_ok(self):
        """Test that the ok method logs WARNING (instead of SUCCESS) and prints correctly."""
        result = self.error.ok("Operation successful")
        self.assertEqual(result, ErrorCode.OK.value)

        self.mock_print.assert_called()
        mock_print_output = remove_ansi_codes(self.mock_print.call_args[0][0])

        self.assertIn("Operation successful", mock_print_output)
        self.assertIn("WARNING", mock_print_output)

    def test_04_skip_default_level(self):
        """Test skip() with default WARNING level (instead of INFO)."""
        result = self.error.skip("Operation skipped")
        self.assertEqual(result, ErrorCode.SKIP.value)

        self.mock_print.assert_called()
        mock_print_output = remove_ansi_codes(self.mock_print.call_args[0][0])

        self.assertIn("Operation skipped", mock_print_output)
        self.assertIn("WARNING", mock_print_output)

    def test_05_skip_custom_level(self):
        """Test skip() with DEBUG level."""
        result = self.error.skip("Operation skipped", ReportLevel.DEBUG)
        self.assertEqual(result, ErrorCode.SKIP.value)

        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("DEBUG", mock_print_output)
        self.assertIn("Operation skipped", mock_print_output)

    def test_06_file_not_exist(self):
        """Test file_not_exist() returns correct code and message."""
        result = self.error.file_not_exist("File does not exist")
        self.assertEqual(result, ErrorCode.FILE_NOT_EXIST.value)

        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("WARNING", mock_print_output)
        self.assertIn("File does not exist", mock_print_output)

    def test_07_unsupported_extension(self):
        """Test handling of unsupported_extension error."""
        result = self.error.unsupported_extension("Extension not supported")
        self.assertEqual(result, ErrorCode.UNSUPPORTED_EXTENSION.value)

        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("WARNING", mock_print_output)
        self.assertIn("Extension not supported", mock_print_output)

    def test_08_not_supported_docking_algorithm(self):
        """Test handling of not_supported_docking_algorithm error."""
        result = self.error.not_supported_docking_algorithm("Algorithm not supported", ReportLevel.ERROR)
        self.assertEqual(result, ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM.value)

        self.mock_print.assert_called()
        mock_print_output = self.mock_print.call_args[0][0]
        self.assertIn("ERROR", mock_print_output)
        self.assertIn("Algorithm not supported", mock_print_output)

    def test_09_error_code_uniqueness(self):
        """Test that all error codes in ErrorCode are unique."""
        error_codes = [code.value for code in ErrorCode]
        self.assertEqual(len(error_codes), len(set(error_codes)), "Duplicate error codes found!")

if __name__ == '__main__':
    unittest.main()
