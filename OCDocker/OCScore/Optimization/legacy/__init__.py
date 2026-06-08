#!/usr/bin/env python3

# Description
###############################################################################
'''
Legacy OCScore Optuna protocol namespace.

Pre-staged optimizers and helpers live here. Import explicitly from
``OCDocker.OCScore.Optimization.legacy`` — historical top-level paths
are not preserved.

Usage:

import OCDocker.OCScore.Optimization.legacy as oclegacy
'''

LEGACY_OPTUNA_MODULES = [
    "OCDocker.OCScore.Optimization.legacy.DNN",
    "OCDocker.OCScore.Optimization.legacy.XGBoost",
    "OCDocker.OCScore.Optimization.legacy.Transformer",
    "OCDocker.OCScore.Optimization.legacy.future.DNN",
    "OCDocker.OCScore.Optimization.legacy.models.dnn.DNNOptimizer",
    "OCDocker.OCScore.DNN.future.DNNOptimizer",
    "OCDocker.OCScore.Utils.legacy.StudyParser",
    "OCDocker.OCScore.Utils.legacy.Workers",
]

__all__ = ["LEGACY_OPTUNA_MODULES"]
