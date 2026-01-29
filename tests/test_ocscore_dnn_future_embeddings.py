import numpy as np
import pytest

try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover
    pytest.skip("torch not available", allow_module_level=True)

from OCDocker.OCScore.Dimensionality.future.Autoencoder import Autoencoder
from OCDocker.OCScore.DNN.future.DNNOptimizer import DNNOptimizer


def test_future_dnn_from_embeddings_minimal():
    rng = np.random.default_rng(0)

    X = rng.normal(size=(32, 20)).astype(np.float32)
    y = rng.normal(size=(32,)).astype(np.float32)

    ae = Autoencoder(
        input_size=20,
        encoder_hidden_sizes=[16],
        latent_dim=8,
        energy_head_sizes=None
    )

    with torch.no_grad():
        Z = ae.encode(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    X_train, X_test = Z[:24], Z[24:]
    y_train, y_test = y[:24], y[24:]

    future_config = {
        "model": {
            "shared_sizes": [16],
            "head_sizes": [8],
            "embedding_dim": 4,
            "dropout": 0.0,
            "batch_norm": False
        },
        "stage1": {
            "epochs": 1,
            "batch_size": 8,
            "noise_type": "none",
            "lambda_recon": 0.0
        },
        "stage2": {"enabled": False}
    }

    trainer = DNNOptimizer.from_embeddings(
        X_train,
        y_train,
        X_test,
        y_test,
        storage="sqlite:///:memory:",
        use_gpu=False,
        verbose=False,
        future_config=future_config
    )

    study = trainer.optimize(n_trials=1, n_jobs=1)
    assert study is not None
