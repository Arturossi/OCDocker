import sys
import inspect
import ast
from pathlib import Path

def load_is_doc_build():
    path = Path(__file__).resolve().parents[1] / "OCDocker" / "Initialise.py"
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "is_doc_build":
            mod = ast.Module([node], [])
            ast.fix_missing_locations(mod)
            ns = {}
            exec(compile(mod, filename=str(path), mode="exec"), ns)
            return ns["is_doc_build"]
    raise RuntimeError("is_doc_build not found")

is_doc_build = load_is_doc_build()

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
    