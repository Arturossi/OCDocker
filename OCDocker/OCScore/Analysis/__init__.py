
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
    pass

from .Metrics import Ranking as RankingMetrics
from .Plotting import MetricsPlots as PlottingMetrics

try:
    from .Impact import (
        build_impact_overview,
        plot_impact_arrows_inline_labels,
        get_neutral_features,
    )
except Exception:
    pass
