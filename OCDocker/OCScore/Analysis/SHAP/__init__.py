
# Re-export SHAP public API for convenience
try:
    from .SHAP import (
        run_shap_analysis, OutputPaths,
        StudyHandles, BestSelections, select_best_from_studies,
        DataHandles, load_and_prepare_data,
        build_neural_net, compute_shap_values, plots
    )
except Exception:  # pragma: no cover
    # Keep import-time failures from breaking unrelated parts of the library
    pass
