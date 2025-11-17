import sys
import inspect
import ast
from pathlib import Path

import pytest


def load_is_doc_build():
    '''Load the is_doc_build function from Initialise.py for testing.
    
    SECURITY NOTE: This function uses exec(compile()) to extract a specific function
    from a known source file (Initialise.py) for testing purposes. The file path is
    constructed from the test file's location and never comes from user input.
    Only the 'is_doc_build' function is extracted and executed, making this safe.
    
    Returns
    -------
    callable
        The is_doc_build function from Initialise.py.
        
    Raises
    ------
    RuntimeError
        If the is_doc_build function is not found in Initialise.py.
    '''
    # Construct path to Initialise.py relative to this test file
    # This is a known, trusted path within the project structure
    path = Path(__file__).resolve().parents[1] / "OCDocker" / "Initialise.py"
    
    # Validate that the path is within the project directory (security check)
    project_root = Path(__file__).resolve().parents[1]
    if not str(path.resolve()).startswith(str(project_root.resolve())):
        raise RuntimeError(f"Security check failed: path {path} is outside project root")
    
    # Validate that the file exists and is a Python file
    if not path.exists() or path.suffix != '.py':
        raise RuntimeError(f"Expected Python file not found at {path}")
    
    source = path.read_text()
    tree = ast.parse(source)
    
    # Extract only the specific function we need (is_doc_build)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "is_doc_build":
            # Create a minimal module containing only this function
            mod = ast.Module([node], [])
            ast.fix_missing_locations(mod)
            # Execute in isolated namespace
            ns = {}
            # SECURITY: Only the extracted function is executed, not arbitrary code
            exec(compile(mod, filename=str(path), mode="exec"), ns)
            return ns["is_doc_build"]
    raise RuntimeError("is_doc_build not found")



is_doc_build = load_is_doc_build()


@pytest.mark.order(1)
def test_is_doc_build_pytest_and_after_clearing(monkeypatch):
    # Should detect the pytest environment
    assert is_doc_build() is True

    with monkeypatch.context() as mp:
        # Remove doc/test related modules
        for name in ["pytest", "unittest", "doctest", "sphinx", "sphinx.ext.autodoc"]:
            mp.delitem(sys.modules, name, raising=False)
        # Empty call stack
        mp.setattr(inspect, "stack", lambda: [], raising=False)
        assert is_doc_build() is False
    