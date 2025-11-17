
# Unified exports for Analysis package
try:
    from .SHAP import (
        run_shap_analysis, OutputPaths,
        StudyHandles, BestSelections, select_best_from_studies,
        DataHandles, load_and_prepare_data,
        build_neural_net, compute_shap_values, plots as shap_plots,
    )
except Exception:
    # Keep optional dependency failures from breaking the package
    run_shap_analysis = None  # type: ignore
    OutputPaths = None  # type: ignore
    StudyHandles = None  # type: ignore
    BestSelections = None  # type: ignore
    select_best_from_studies = None  # type: ignore
    DataHandles = None  # type: ignore
    load_and_prepare_data = None  # type: ignore
    build_neural_net = None  # type: ignore
    compute_shap_values = None  # type: ignore
    shap_plots = None  # type: ignore

from .Metrics import Ranking as RankingMetrics
from .Plotting import MetricsPlots as PlottingMetrics

try:
    from .Impact import (
        build_impact_overview,
        plot_impact_arrows_inline_labels,
        get_neutral_features,
    )
except Exception:
    build_impact_overview = None  # type: ignore
    plot_impact_arrows_inline_labels = None  # type: ignore
    get_neutral_features = None  # type: ignore

# Public API
__all__ = [
    # SHAP exports (may be None if dependencies missing)
    'run_shap_analysis',
    'OutputPaths',
    'StudyHandles',
    'BestSelections',
    'select_best_from_studies',
    'DataHandles',
    'load_and_prepare_data',
    'build_neural_net',
    'compute_shap_values',
    'shap_plots',
    # Metrics exports
    'RankingMetrics',
    # Plotting exports
    'PlottingMetrics',
    # Impact exports (may be None if dependencies missing)
    'build_impact_overview',
    'plot_impact_arrows_inline_labels',
    'get_neutral_features',
]
