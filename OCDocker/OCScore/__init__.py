"""
The OCScore package is a collection of tools for scoring and ranking docking poses.

Packages
--------
- Analysis: Analysis functions for scoring outcomes of the OCScore package.
- Dimensionality: Dimensionality reduction methods.
- NN: Neural network training functions.
- Optimization: Optimization algorithms, it contains helpers for training functions and it is the module that probably you are looking for.
- Transformer: Transformer training functions.
- Utils: Utility functions.
- XGBoost: XGBoost training functions.

Modules
-------
- Scoring: Functions for loading models and applying scoring pipelines to get predictions.
- SimpleConsensus: Simple consensus scoring functions such as mean, median, max, min, std, variance, sum, range, 25th and 75th percentiles, kurtoisis, skewness.
"""

# Public API: Users should import OCScore modules directly
# No classes or functions exported at package level
__all__ = []
