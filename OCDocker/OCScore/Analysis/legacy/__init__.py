#!/usr/bin/env python3

# Description
###############################################################################
'''
Legacy OCScore analysis namespace.

Modules here implement the previous mixed-metric Optuna study workflow (RMSE +
AUC side by side, Test2 bootstrap tables). The staged pipeline uses
``Analysis.Metrics.Ranking`` for screening metrics and ``Optimization.StagedOptuna``
for orchestration.
'''

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
'''

LEGACY_ANALYSIS_MODULES = [
    "OCDocker.OCScore.Analysis.legacy.RankingMetrics",
    "OCDocker.OCScore.Analysis.legacy.StudyProcessing",
    "OCDocker.OCScore.Analysis.legacy.SHAP",
    "OCDocker.OCScore.Analysis.legacy.FeatureImportance",
    "OCDocker.OCScore.Analysis.legacy.PerformanceEvaluation",
    "OCDocker.OCScore.Analysis.legacy.Correlation",
    "OCDocker.OCScore.Analysis.legacy.NNUtils",
    "OCDocker.OCScore.Analysis.legacy.StatTests",
    "OCDocker.OCScore.Analysis.legacy.Impact",
    "OCDocker.OCScore.Analysis.legacy.ImpactPlots",
]

__all__ = ["LEGACY_ANALYSIS_MODULES"]
