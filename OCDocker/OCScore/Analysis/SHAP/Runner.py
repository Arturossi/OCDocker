
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os
import numpy as np
import pandas as pd

from .Studies import StudyHandles, select_best_from_studies
from .Data import load_and_prepare_data
from .Model import build_neural_net
from .Explain import compute_shap_values
from . import plots

@dataclass
class OutputPaths:
    out_dir: str
    feature_importance_png: str
    beeswarm_png: str
    shap_values_npy: str
    shap_values_csv: str | None = None

def run_shap_analysis(
    studies: StudyHandles,
    df_path: str,
    base_models_folder: str,
    study_number: int,
    out_dir: str,
    background_size: Optional[int] = None,
    eval_size: Optional[int] = None,
    explainer: str = "deep",
    stratify_by: Optional[List[str]] = None,
    seed: int = 0,
    save_csv: bool = True,
) -> OutputPaths:
    os.makedirs(out_dir, exist_ok=True)
    best = select_best_from_studies(studies)
    data = load_and_prepare_data(
        df_path=df_path,
        base_models_folder=base_models_folder,
        study_number=study_number,
        use_pca=False,
        use_pdb_train=True,
        random_seed=42,
    )
    neural = build_neural_net(
        input_dim=data.X_train.shape[1],
        autoencoder_params=best.autoencoder_params,
        nn_params=best.nn_params,
        seed=best.seed,
        mask=best.mask,
        use_gpu=None,
        verbose=False,
    )
    shap_2d = compute_shap_values(
        neural=neural,
        X_background=data.X_train,
        X_eval=data.X_test,
        explainer=explainer,
        background_size=background_size,
        eval_size=eval_size,
        stratify_by=stratify_by,
        rng_seed=seed,
    )
    shap_npy = os.path.join(out_dir, "shap_values.npy")
    np.save(shap_npy, shap_2d)
    shap_csv = None
    if save_csv:
        shap_csv = os.path.join(out_dir, "shap_values.csv")
        import pandas as pd
        pd.DataFrame(shap_2d, columns=data.feature_names).to_csv(shap_csv, index=False)
    imp_png = os.path.join(out_dir, "shap_feature_importance.png")
    bee_png = os.path.join(out_dir, "shap_beeswarm_plot.png")
    plots.feature_importance_barh(shap_2d, data.feature_names, out_png=imp_png, top_k=20)
    plots.beeswarm(shap_2d, data.X_test.iloc[:shap_2d.shape[0]], out_png=bee_png)
    return OutputPaths(
        out_dir=out_dir,
        feature_importance_png=imp_png,
        beeswarm_png=bee_png,
        shap_values_npy=shap_npy,
        shap_values_csv=shap_csv,
    )
