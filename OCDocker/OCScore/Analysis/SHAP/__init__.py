"""
Re-export SHAP public API for convenience.

This package collects the key functions/classes from its submodules so that
`from OCDocker.OCScore.Analysis import SHAP` exposes a simple, consistent API.
"""

# Import Plots first since Runner depends on it
from . import Plots as plots

# Import each module separately so one failure doesn't break all imports
from .Runner import run_shap_analysis, OutputPaths
from .Studies import StudyHandles, BestSelections, select_best_from_studies
from .Data import DataHandles, load_and_prepare_data
from .Model import build_neural_net
from .Explain import compute_shap_values

__all__ = [
    "run_shap_analysis",
    "OutputPaths",
    "StudyHandles",
    "BestSelections",
    "select_best_from_studies",
    "DataHandles",
    "load_and_prepare_data",
    "build_neural_net",
    "compute_shap_values",
    "plots",
]
