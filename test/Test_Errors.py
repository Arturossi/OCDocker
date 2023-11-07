import unittest
from unittest.mock import patch
from OCDocker.Error import Error, ErrorCode, ReportLevel

class TestErrorMethods(unittest.TestCase):
    """
    This class contains unit tests for the Error class methods.
    It tests that error reporting is functioning correctly by asserting the printed messages and return values.
    """

    def setUp(self):
        """
        Set up the test case. This method is called before each test function is executed.
        """
        self.error = Error()

    def test_ok(self):
        """
        Test that the ok method returns the correct error code and prints the success message.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.ok("Operation successful")
            self.assertEqual(result, ErrorCode.OK.value)
            mock_print.assert_called_with("[SUCCESS] Operation successful")

    def test_abort(self):
        """
        Test that the abort method returns the correct error code and prints a warning message.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.abort("Operation aborted")
            self.assertEqual(result, ErrorCode.ABORT.value)
            mock_print.assert_called_with("[WARNING] Operation aborted")

    def test_skip_default_level(self):
        """
        Test that the skip method returns the correct error code and prints an informational message by default.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.skip("Operation skipped")
            self.assertEqual(result, ErrorCode.SKIP.value)
            mock_print.assert_called_with("[INFO] Operation skipped")

    def test_skip_custom_level(self):
        """
        Test that the skip method returns the correct error code and prints a debug message when a custom level is specified.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.skip("Operation skipped", ReportLevel.DEBUG)
            self.assertEqual(result, ErrorCode.SKIP.value)
            mock_print.assert_called_with("[DEBUG] Operation skipped")

    def test_unknown_default_level(self):
        """
        Test that the unknown method returns the correct error code and prints a warning message by default.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.unknown("Error is unknown")
            self.assertEqual(result, ErrorCode.UNKNOWN.value)
            mock_print.assert_called_with("[WARNING] Error is unknown")

    def test_unknown_custom_level(self):
        """
        Test that the unknown method returns the correct error code and prints an error message when a custom level is specified.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.unknown("Error is unknown", ReportLevel.ERROR)
            self.assertEqual(result, ErrorCode.UNKNOWN.value)
            mock_print.assert_called_with("[ERROR] Error is unknown")

    def test_file_exists(self):
        """
        Test the file_exists method to check if the proper warning and error code are returned when a file exists.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.file_exists("File exists")
            self.assertEqual(result, ErrorCode.FILE_EXISTS.value)
            mock_print.assert_called_with("[WARNING] File exists")

    def test_file_do_not_exist(self):
        """
        Test the file_do_not_exist method to check if the appropriate warning and error code are returned when a file does not exist.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.file_do_not_exist("File does not exist")
            self.assertEqual(result, ErrorCode.FILE_NOT_EXIST.value)
            mock_print.assert_called_with("[WARNING] File does not exist")

    def test_read_file(self):
        """
        Test the read_file method to verify the warning and error code returned when a file cannot be read.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.read_file("Cannot read file")
            self.assertEqual(result, ErrorCode.READ_FILE.value)
            mock_print.assert_called_with("[WARNING] Cannot read file")

    def test_write_file(self):
        """
        Test the write_file method to confirm the correct warning and error code are returned when a file cannot be written to.
        """
        with patch('builtins.print') as mock_print:
            result = self.error.write_file("Cannot write file")
            self.assertEqual(result, ErrorCode.WRITE_FILE.value)
            mock_print.assert_called_with("[WARNING] Cannot write file")
    
    def test_untar_file(self):
        """Test handling of untar_file error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.untar_file("Untar failed")
            self.assertEqual(result, ErrorCode.UNTAR_FILE.value)
            mock_print.assert_called_with("[WARNING] Untar failed")

    def test_unsupported_extension(self):
        """Test handling of unsupported_extension error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.unsupported_extension("Extension not supported")
            self.assertEqual(result, ErrorCode.UNSUPPORTED_EXTENSION.value)
            mock_print.assert_called_with("[WARNING] Extension not supported")

    def test_broken_pipe(self):
        """Test handling of broken_pipe error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.broken_pipe("Broken pipe")
            self.assertEqual(result, ErrorCode.BROKEN_PIPE.value)
            mock_print.assert_called_with("[WARNING] Broken pipe")

    def test_empty_file(self):
        """Test handling of empty_file error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.empty_file("File is empty")
            self.assertEqual(result, ErrorCode.EMPTY_FILE.value)
            mock_print.assert_called_with("[WARNING] File is empty")

    def test_dir_exists(self):
        """Test handling of dir_exists error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.dir_exists("Directory exists")
            self.assertEqual(result, ErrorCode.DIR_EXISTS.value)
            mock_print.assert_called_with("[WARNING] Directory exists")

    def test_create_dir(self):
        """Test the error handling when directory creation fails."""
        with patch('builtins.print') as mock_print:
            result = self.error.create_dir("Cannot create directory")
            self.assertEqual(result, ErrorCode.CREATE_DIR.value)
            mock_print.assert_called_with("[WARNING] Cannot create directory")

    def test_remove_dir(self):
        """Test the error handling when directory removal fails."""
        with patch('builtins.print') as mock_print:
            result = self.error.remove_dir("Cannot remove directory")
            self.assertEqual(result, ErrorCode.REMOVE_DIR.value)
            mock_print.assert_called_with("[WARNING] Cannot remove directory")

    def test_dir_does_not_exist(self):
        """Test the error handling when a non-existent directory is accessed."""
        with patch('builtins.print') as mock_print:
            result = self.error.dir_does_not_exist("Directory does not exist")
            self.assertEqual(result, ErrorCode.DIR_NOT_EXIST.value)
            mock_print.assert_called_with("[WARNING] Directory does not exist")

    def test_unnalowed_dir(self):
        """Test the error handling when access to a directory is unallowed."""
        with patch('builtins.print') as mock_print:
            result = self.error.unnalowed_dir("Directory access is unallowed")
            self.assertEqual(result, ErrorCode.UNALLOWED_DIR.value)
            mock_print.assert_called_with("[WARNING] Directory access is unallowed")

    def test_wrong_type(self):
        """Test the error handling when a variable has the wrong type."""
        with patch('builtins.print') as mock_print:
            result = self.error.wrong_type("Variable has wrong type")
            self.assertEqual(result, ErrorCode.WRONG_TYPE.value)
            mock_print.assert_called_with("[WARNING] Variable has wrong type")

    def test_not_set(self):
        """Test the error handling when a variable is not set."""
        with patch('builtins.print') as mock_print:
            result = self.error.not_set("Variable is not set")
            self.assertEqual(result, ErrorCode.NOT_SET.value)
            mock_print.assert_called_with("[WARNING] Variable is not set")

    def test_empty(self):
        """Test the error handling when a variable is empty."""
        with patch('builtins.print') as mock_print:
            result = self.error.empty("Variable is empty")
            self.assertEqual(result, ErrorCode.EMPTY.value)
            mock_print.assert_called_with("[WARNING] Variable is empty")

    def test_value_error_default(self):
        """Test the default error handling for a value error."""
        with patch('builtins.print') as mock_print:
            result = self.error.value_error()
            self.assertEqual(result, ErrorCode.VALUE_ERROR.value)
            mock_print.assert_called_with("[WARNING] ")

    def test_value_error_warning_with_message(self):
        """Test the error handling for a value error with a custom message."""
        with patch('builtins.print') as mock_print:
            result = self.error.value_error("Value error occurred")
            self.assertEqual(result, ErrorCode.VALUE_ERROR.value)
            mock_print.assert_called_with("[WARNING] Value error occurred")

    def test_subprocess_default(self):
        """Test the default error handling for a subprocess error."""
        with patch('builtins.print') as mock_print:
            result = self.error.subprocess()
            self.assertEqual(result, ErrorCode.SUBPROCESS_ERROR.value)
            mock_print.assert_called_with("[WARNING] ")

    def test_subprocess_warning_with_message(self):
        """Test the error handling for a subprocess error with a custom message."""
        with patch('builtins.print') as mock_print:
            result = self.error.subprocess("Subprocess error occurred")
            self.assertEqual(result, ErrorCode.SUBPROCESS_ERROR.value)
            mock_print.assert_called_with("[WARNING] Subprocess error occurred")

    def test_docking_object_not_generated(self):
        """Test the error handling when a docking object is not generated."""
        with patch('builtins.print') as mock_print:
            result = self.error.docking_object_not_generated("Docking object issue")
            self.assertEqual(result, ErrorCode.DOCKING_OBJECT_NOT_GENERATED.value)
            mock_print.assert_called_with("[WARNING] Docking object issue")

    def test_receptor_or_ligand_not_generated(self):
        """Test the error handling when a receptor or ligand is not generated."""
        with patch('builtins.print') as mock_print:
            result = self.error.receptor_or_ligand_not_generated("Receptor or ligand not generated")
            self.assertEqual(result, ErrorCode.RECEPTOR_OR_LIGAND_NOT_GENERATED.value)
            mock_print.assert_called_with("[WARNING] Receptor or ligand not generated")

    def test_receptor_or_ligand_descriptor_does_not_exist(self):
        """Test the error handling when a descriptor for receptor or ligand does not exist."""
        with patch('builtins.print') as mock_print:
            result = self.error.receptor_or_ligand_descriptor_does_not_exist("Descriptor missing")
            self.assertEqual(result, ErrorCode.RECEPTOR_OR_LIGAND_DESCRIPTOR_NOT_EXIST.value)
            mock_print.assert_called_with("[WARNING] Descriptor missing")

    def test_not_supported_docking_algorithm(self):
        """Test the error handling for an unsupported docking algorithm."""
        with patch('builtins.print') as mock_print:
            result = self.error.not_supported_docking_algorithm("Unsupported algorithm", level=ReportLevel.ERROR)
            self.assertEqual(result, ErrorCode.NOT_SUPPORTED_DOCKING_ALGORITHM.value)
            mock_print.assert_called_with("[ERROR] Unsupported algorithm")

    def test_unsupported_scoring_function(self):
        """Test the error handling for an unsupported scoring function."""
        with patch('builtins.print') as mock_print:
            result = self.error.unsupported_scoring_function("Unsupported scoring function")
            self.assertEqual(result, ErrorCode.UNSUPPORTED_SCORING_FUNCTION.value)
            mock_print.assert_called_with("[ERROR] Unsupported scoring function")

    def test_rescoring_failed(self):
        """Test the error handling when rescoring fails."""
        with patch('builtins.print') as mock_print:
            result = self.error.rescoring_failed("Rescoring failed")
            self.assertEqual(result, ErrorCode.RESCORING_FAILED.value)
            mock_print.assert_called_with("[ERROR] Rescoring failed")

    def test_missing_oddt_models(self):
        """Test the error handling when ODDt models are missing."""
        with patch('builtins.print') as mock_print:
            result = self.error.missing_oddt_models("Missing ODDt models")
            self.assertEqual(result, ErrorCode.MISSING_ODDT_MODELS.value)
            mock_print.assert_called_with("[ERROR] Missing ODDt models")

    def test_unsupported_clustering_algorithm(self):
        """Test handling of unsupported_clustering_algorithm error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.unsupported_clustering_algorithm("Unsupported clustering algorithm")
            self.assertEqual(result, ErrorCode.UNSUPPORTED_CLUSTERING_ALGORITHM.value)
            mock_print.assert_called_with("[ERROR] Unsupported clustering algorithm")

    def test_cluster_not_converged(self):
        """Test handling of cluster_not_converged error with a predefined message."""
        with patch('builtins.print') as mock_print:
            result = self.error.cluster_not_converged("Clustering not converged")
            self.assertEqual(result, ErrorCode.CLUSTER_NOT_CONVERGED.value)
            mock_print.assert_called_with("[ERROR] Clustering not converged")
        
        def test_error_with_no_message(self):
            """
            Test that the ok method handles being called without a message.
            """
            with patch('builtins.print') as mock_print:
                result = self.error.ok()
                self.assertEqual(result, ErrorCode.OK.value)
                mock_print.assert_called_with("[SUCCESS]")

    def test_error_level_is_not_changed(self):
        """
        Test that the default error level has not changed unexpectedly.
        """
        default_level = ReportLevel.WARNING  # Replace with actual default
        with patch('builtins.print') as mock_print:
            self.error.abort("Operation aborted")
            args, kwargs = mock_print.call_args
            self.assertTrue(args[0].startswith(f"[{default_level.name}]"))

    def test_error_handling_of_exception(self):
        """
        Test how the error class handles exceptions raised during the error reporting.
        """
        with patch('builtins.print', side_effect=Exception("Print failed")) as mock_print:
            with self.assertRaises(Exception):
                self.error.ok("This should fail")

    def test_error_code_uniqueness(self):
        """
        Test that all error codes in the ErrorCode enumeration are unique.
        """
        error_code_values = [code.value for code in ErrorCode]
        self.assertEqual(len(error_code_values), len(set(error_code_values)))

    def test_report_method_output(self):
        """
        Test that the report method is called correctly and returns the expected output.
        Assuming report is a method used internally to prepare the error message.
        """
        with patch.object(self.error, 'report', return_value=(ErrorCode.OK.value, "[SUCCESS] Operation successful")) as mock_report:
            result = self.error.ok("Operation successful")
            mock_report.assert_called_once()
            self.assertEqual(result, ErrorCode.OK.value)

if __name__ == '__main__':
    unittest.main()