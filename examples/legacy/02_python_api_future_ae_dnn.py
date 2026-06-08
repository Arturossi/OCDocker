#!/usr/bin/env python3
"""
Legacy Example 02: Future AE -> DNN pipeline (embeddings)
This example shows how to train the future Autoencoder to generate embeddings,
then feed those embeddings into the future DNN optimizer.
"""

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime guard for missing torch
    raise SystemExit(f"PyTorch is required for this example: {exc}")

from OCDocker.OCScore.Dimensionality.legacy.AETrainer import AETrainer
from OCDocker.OCScore.Dimensionality.legacy.Autoencoder import Autoencoder
from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer


def main() -> None:
    # -----------------------
    # 1) Generate toy data
    # NOTE: Replace this block with your real feature matrix and labels.
    # -----------------------
    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 20)).astype(np.float32)
    y = rng.normal(size=(128,)).astype(np.float32)

    X_train, X_test = X[:96], X[96:]
    y_train, y_test = y[:96], y[96:]

    # -----------------------
    # 2) Train future AE
    # -----------------------
    ae = Autoencoder(
        input_size=X_train.shape[1],
        encoder_hidden_sizes=[32, 16],
        latent_dim=8,
        energy_head_sizes=[16]
    )

    ae_config = {
        "stage1": {
            "enabled": True,
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "recon_loss": "mse",
            "energy_loss": "huber",
            "lambda_recon": 1.0,
            "lambda_energy": 0.5,
            "noise_type": "none",
            "clip_grad": 1.0
        },
        "stage2": {"enabled": False},
        "optimization": {"loss_balancing": "fixed"},
        "checkpoint": {"save_best": False}
    }

    trainer = AETrainer(model=ae, config=ae_config, device=torch.device("cpu"), verbose=False)
    trainer.fit(X_train, X_val=X_test, y_train=y_train, y_val=y_test)

    # Encode embeddings from the trained AE
    with torch.no_grad():
        Z_train = ae.encode(torch.tensor(X_train, dtype=torch.float32)).cpu().numpy()
        Z_test = ae.encode(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

    # -----------------------
    # 3) Train future DNN on embeddings
    # -----------------------
    # Create a simple ranking dataset from the regression targets.
    # NOTE: In real projects, use a proper ranking dataset instead of reusing X_train.
    y_rank = (y_train > np.median(y_train)).astype(np.int64)

    dnn_config = {
        "model": {
            "shared_sizes": [16],
            "head_sizes": [8],
            "embedding_dim": 4,
            "dropout": 0.0,
            "batch_norm": False
        },
        "stage1": {
            "epochs": 1,
            "batch_size": 16,
            "noise_type": "none",
            "lambda_recon": 0.0  # reconstruction disabled when using embeddings
        },
        "stage2": {
            "enabled": True,
            "epochs": 1,
            "batch_size_per_target": None,
            "split_target_batches": False,
            "lambda_rank": 1.0,
            "lambda_cls": 1.0,
            "lambda_con": 0.0,
            "temperature": 0.1
        },
        "data": {
            "ranking_validation_fraction": 0.2,
            "ranking_split_by_target": False
        }
    }

    dnn = DNNOptimizer.from_embeddings(
        Z_train,
        y_train,
        Z_test,
        y_test,
        X_embeddings_validation=Z_train,
        y_validation=y_rank,
        storage="sqlite:///:memory:",
        random_seed=42,
        use_gpu=False,
        verbose=False,
        future_config=dnn_config
    )

    study = dnn.optimize(n_trials=25, n_jobs=1)
    print("DNN optimization completed. Best trial value:", study.best_trial.value)


if __name__ == "__main__":
    main()
