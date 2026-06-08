#!/usr/bin/env python3

# Description
###############################################################################
'''
Regression tests to prevent mutable default arguments in selected OCScore APIs.
'''

# Imports
###############################################################################
import ast
import pytest

from pathlib import Path

# License
###############################################################################
'''
OCDocker
Authors: Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M.
Federal University of Rio de Janeiro
Carlos Chagas Filho Institute of Biophysics
Laboratory for Molecular Modeling and Dynamics

This program is proprietary software owned by the Federal University of Rio de Janeiro (UFRJ),
developed by Rossi, A.D.; Monachesi, M.C.E.; Spelta, G.I.; Torres, P.H.M., and protected under Brazilian Law No. 9,609/1998.
All rights reserved. Use, reproduction, modification, and distribution are allowed under this UFRJ license,
provided this copyright notice is preserved. See the LICENSE file for details.

Contact: Artur Duque Rossi - arturossi10@gmail.com
'''

# Classes
###############################################################################


# Functions
###############################################################################
## Private ##


def _get_default_expr(module_rel_path: str, function_name: str, parameter_name: str, class_name: str | None = None) -> ast.expr:
    project_root = Path(__file__).resolve().parents[3]
    module_path = project_root / module_rel_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    target_func = None
    if class_name is None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                target_func = node
                break
    else:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name == function_name:
                        target_func = child
                        break
                if target_func is not None:
                    break

    if target_func is None:
        raise AssertionError(f"Function not found: {module_rel_path}::{class_name + '.' if class_name else ''}{function_name}")

    arg_names = [arg.arg for arg in target_func.args.args]
    defaults = target_func.args.defaults
    default_start = len(arg_names) - len(defaults)
    default_map = {
        arg_names[default_start + i]: defaults[i]
        for i in range(len(defaults))
    }

    if parameter_name not in default_map:
        raise AssertionError(
            f"Default not found for parameter '{parameter_name}' in "
            f"{module_rel_path}::{class_name + '.' if class_name else ''}{function_name}"
        )

    return default_map[parameter_name]


## Public ##


@pytest.mark.order(647)
@pytest.mark.parametrize(
    "module_rel_path,class_name,function_name,parameter_name",
    [
        ("OCDocker/OCScore/Utils/legacy/Workers.py", None, "GAWorker", "best_params"),
        ("OCDocker/OCScore/Utils/legacy/Workers.py", None, "XGBworker", "params"),
        ("OCDocker/OCScore/Optimization/legacy/DNN.py", None, "optimize_NN", "data"),
        ("OCDocker/OCScore/Optimization/legacy/DNN.py", None, "perform_ablation_study_NN", "masks"),
        ("OCDocker/OCScore/Optimization/legacy/DNN.py", None, "perform_seed_ablation_study_NN", "seeds"),
        ("OCDocker/OCScore/Optimization/legacy/XGBoost.py", None, "optimize_XGB", "data"),
        ("OCDocker/OCScore/Optimization/legacy/Transformer.py", None, "optimize_Transformer", "data"),
        ("OCDocker/OCScore/Optimization/legacy/models/xgboost/OCxgboost.py", None, "run_xgboost", "params"),
        ("OCDocker/OCScore/Optimization/legacy/models/xgboost/XGBoostOptimizer.py", "XGBoostOptimizer", "__init__", "params"),
        ("OCDocker/OCScore/Optimization/legacy/models/transformer/TransOptimizer.py", "TransformerModel", "__init__", "init_params"),
    ],
)
def test_selected_defaults_are_none(module_rel_path, class_name, function_name, parameter_name):
    default_expr = _get_default_expr(module_rel_path, function_name, parameter_name, class_name=class_name)
    assert isinstance(default_expr, ast.Constant)
    assert default_expr.value is None
