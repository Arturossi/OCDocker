import os
import sys
import logging
from pathlib import Path
import types
import importlib
import importlib.util
from importlib.machinery import PathFinder
from importlib.abc import MetaPathFinder, Loader
from unittest.mock import MagicMock
from enum import IntEnum

logging.basicConfig(level=logging.DEBUG)

# -----------------------------------------------------------------------------
# Locate repo root (directory that CONTAINS 'OCDocker/__init__.py')
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve()


def find_repo_root(start: Path) -> Path:
    for p in (start, *start.parents):
        pkg = p / "OCDocker"
        if pkg.is_dir() and (pkg / "__init__.py").exists():
            return p
    raise RuntimeError("Could not find repo root with 'OCDocker/__init__.py' above this file.")



REPO_ROOT = find_repo_root(HERE)
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("OC_BUILD_DOCS", "1")
os.environ.setdefault("MPLBACKEND", "agg")
logging.debug("Docs repo root: %s", REPO_ROOT)



# -----------------------------------------------------------------------------
# Levels & defaults FIRST (used by Error module and injections)
# -----------------------------------------------------------------------------
class ReportLevel(IntEnum):
    NONE  = 0
    INFO  = 1
    WARN  = 2
    ERROR = 3
    DEBUG = 4



DEFAULT_LEVEL = ReportLevel.INFO

# -----------------------------------------------------------------------------
# Minimal shims needed during autodoc import
# -----------------------------------------------------------------------------

# Provide OCDocker.Initialise with the attributes code expects
init_mod = types.ModuleType("OCDocker.Initialise")
init_mod.__all__ = ["session", "db_url", "get_db_url"]
init_mod.session = MagicMock(name="session")

# Many DB modules expect a connection URL at import time.
# Use an in-memory SQLite URL so imports don't fail.
init_mod.db_url = "sqlite:///:memory:"
init_mod.engine = MagicMock(name="engine")
init_mod.__all__.append("engine")


def _get_db_url():
    return init_mod.db_url



init_mod.get_db_url = _get_db_url
sys.modules["OCDocker.Initialise"] = init_mod

# Provide OCDocker.Error as a real module with the expected symbols
err_mod = types.ModuleType("OCDocker.Error")
err_mod.ReportLevel = ReportLevel
# Some code does "from OCDocker.Error import ocerror" or "Error"
# Make both aliases refer to the same module for .ReportLevel, .output_level, etc.
err_mod.Error = err_mod
err_mod.ocerror = err_mod
err_mod.output_level = DEFAULT_LEVEL
err_mod.__all__ = ["ReportLevel", "Error", "ocerror", "output_level"]
sys.modules["OCDocker.Error"] = err_mod

# If any stale non-package stubs exist that could block real packages, remove them.
for pkg in (
    "OCDocker",
    "OCDocker.Toolbox",
    "OCDocker.Docking",
    "OCDocker.OCScore",
    "OCDocker.DB",
    "OCDocker.Processing",
    "OCDocker.Rescoring",
):
    m = sys.modules.get(pkg)
    if m is not None and not hasattr(m, "__path__"):
        del sys.modules[pkg]

# Import or create the top-level package and attach common attributes
try:
    OCD = importlib.import_module("OCDocker")
except Exception:
    OCD = sys.modules.setdefault("OCDocker", types.ModuleType("OCDocker"))
    if not hasattr(OCD, "__path__"):
        OCD.__path__ = [str((REPO_ROOT / "OCDocker").resolve())]

# Attach attributes that many modules expect at the package level
setattr(OCD, "Error", err_mod)              # from OCDocker import Error
setattr(OCD, "ocerror", err_mod)            # from OCDocker import ocerror
setattr(OCD, "ReportLevel", ReportLevel)    # OCDocker.ReportLevel
setattr(OCD, "output_level", DEFAULT_LEVEL) # OCDocker.output_level
sys.modules["OCDocker"] = OCD



# -----------------------------------------------------------------------------
# Import-time global injection for ALL OCDocker submodules
# Ensures `output_level`, `ReportLevel`, `ocerror`, and `Error` are available
# unqualified in each module's globals BEFORE its code runs.
# -----------------------------------------------------------------------------
class _InjectingLoader(Loader):
    def __init__(self, base_loader, injected: dict):
        self._base = base_loader

        self._injected = injected









    def create_module(self, spec):
        if hasattr(self._base, "create_module"):
            return self._base.create_module(spec)

        return None








    def exec_module(self, module):
        for k, v in self._injected.items():
            if not hasattr(module, k):
                setattr(module, k, v)
        return self._base.exec_module(module)


class _InjectingFinder(MetaPathFinder):
    def __init__(self, pkg_prefix: str, injected: dict):
        self._pkg_prefix = pkg_prefix

        self._injected = injected








    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._pkg_prefix):
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _InjectingLoader(spec.loader, self._injected)
        return spec



_injected_globals = {
    "output_level": DEFAULT_LEVEL,
    "ReportLevel": ReportLevel,
    "ocerror": err_mod,   # module
    "Error": err_mod,     # alias to same module
}

if not any(isinstance(f, _InjectingFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _InjectingFinder("OCDocker", _injected_globals))



# -----------------------------------------------------------------------------
# Helper: load a specific module with extra injections (when needed)
# -----------------------------------------------------------------------------
def load_with_injected_globals(fullname: str, injected: dict):
    if fullname in sys.modules:
        mod = sys.modules[fullname]
        for k, v in injected.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)
        return mod

    spec = PathFinder.find_spec(fullname, None)
    if spec is None or spec.loader is None:
        return importlib.import_module(fullname)

    mod = types.ModuleType(fullname)
    mod.__file__ = spec.origin
    mod.__loader__ = spec.loader
    mod.__package__ = fullname.rpartition(".")[0]
    mod.__spec__ = spec

    if spec.submodule_search_locations is not None:
        mod.__path__ = list(spec.submodule_search_locations)

    for k, v in injected.items():
        setattr(mod, k, v)

    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod

# Pre-load modules that reference globals at import time
load_with_injected_globals(
    "OCDocker.Toolbox.Logging",
    {"output_level": DEFAULT_LEVEL, "ReportLevel": ReportLevel, "ocerror": err_mod, "Error": err_mod},
)
load_with_injected_globals(
    "OCDocker.Toolbox.Printing",
    {"output_level": DEFAULT_LEVEL, "ReportLevel": ReportLevel, "ocerror": err_mod, "Error": err_mod},
)

# Also attach Error/levels to common parent packages (best-effort)
for parent in (
    "OCDocker.Toolbox",
    "OCDocker.Docking",
    "OCDocker.DB",
    "OCDocker.Processing",
    "OCDocker.OCScore",
    "OCDocker.Rescoring",
):
    try:
        pkg = importlib.import_module(parent)
        for k, v in (
            ("Error", err_mod),
            ("ocerror", err_mod),
            ("ReportLevel", ReportLevel),
            ("output_level", DEFAULT_LEVEL),
        ):
            if not hasattr(pkg, k):
                setattr(pkg, k, v)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Sphinx configuration
# -----------------------------------------------------------------------------
project = "OCDocker"
copyright = "2025, Artur Duque Rossi"
author = "Artur Duque Rossi"
version = "0.11.1"
release = "0.11.1"

extensions = [
    "sphinxarg.ext",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}
html_static_path = ["_static"]

todo_include_todos = True

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Do NOT mock your own modules
autodoc_mock_imports = ["cupy", "torch", "oddt", "openbabel", "rdkit"]

# RST substitutions
rst_prolog = """
.. |NBS| replace:: Normalized Binding Score
.. |NBS_norm| replace:: Normalized Binding Score (scaled)
"""
