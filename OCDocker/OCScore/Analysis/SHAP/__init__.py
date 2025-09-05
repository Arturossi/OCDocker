"""
Re-export SHAP public API for convenience.

This package collects the key functions/classes from its submodules so that
`from OCDocker.OCScore.Analysis import SHAP` exposes a simple, consistent API.
"""

try:
    # High-level runner and output paths
    from .Runner import run_shap_analysis, OutputPaths
    # Study helpers
    from .Studies import StudyHandles, BestSelections, select_best_from_studies
    # Data loading/processing
    from .Data import DataHandles, load_and_prepare_data
    # Model and explainability
    from .Model import build_neural_net
    from .Explain import compute_shap_values
    # Plot helpers module
    from . import Plots as plots
except Exception:  # pragma: no cover
    # Keep import-time failures from breaking unrelated parts of the library
    pass

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
