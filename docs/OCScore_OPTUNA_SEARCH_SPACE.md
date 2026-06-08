# OCScore staged Optuna search space

## Where options are defined

Default Optuna search-space values for the staged PDBbind/DUDEz protocol live in:

`OCDocker/OCScore/Optimization/OptunaSearchSpace.py`

Stage configs reference these defaults through:

- `PDBbindOptunaConfig.search_space`
- `DUDEzOptunaConfig.search_space`

Sampling logic reads the config objects in `OCDocker/OCScore/Optimization/StagedOptuna.py`.

## How to tune later

Edit the dataclasses in `OptunaSearchSpace.py`, for example:

- `EncoderSearchSpace` — `encoder_hidden_size_options`, `encoder_depth_options`, `encoder_latent_dim_options`
- `ProjectionSearchSpace` — `projection_dim_options`
- `DecoderSearchSpace` — `decoder_lambda_rec_options`, `decoder_depth_options`, `decoder_hidden_size_options`
- `OptimizerSearchSpace` — learning-rate/weight-decay ranges and `batch_size_options`
- `PDBbindHeadSearchSpace` — regression loss and Huber delta
- `DUDEzHeadSearchSpace` — classifier head and transfer options
- `DEFAULT_ACTIVATION_OPTIONS` — activation candidates

Pass a customized config into the stage constructor when running programmatically:

```python
from OCDocker.OCScore.Optimization.OptunaSearchSpace import PDBbindSearchSpaceConfig, EncoderSearchSpace
from OCDocker.OCScore.Optimization.StagedOptuna import PDBbindOptunaConfig, PDBbindOptunaStage

pdbbind_config = PDBbindOptunaConfig(
    search_space=PDBbindSearchSpaceConfig(
        activation_options=("ReLU", "GELU", "SiLU", "Mish"),
        encoder=EncoderSearchSpace(hidden_size_options=(64, 128, 256)),
    )
)
```

## Activation functions

Activation names are centralized in `DEFAULT_ACTIVATION_OPTIONS`:

- ReLU
- LeakyReLU
- ELU
- GELU
- SiLU
- Mish

`build_activation_module()` constructs the module and raises a clear error when an activation is unavailable in the installed PyTorch build (for example `Mish` on very old PyTorch versions).

## Naming conventions

Optuna trial parameters use explicit prefixes:

| Prefix | Meaning |
| --- | --- |
| `encoder_*` | Reusable feature extractor (encoder) |
| `projection_*` | Optional projection block after the encoder |
| `decoder_*` | PDBbind-only reconstruction branch |
| `pdbbind_*` | PDBbind regression head / loss |
| `dudez_*` | DUDEz classifier head / transfer |
| `optimizer_*` | Learning rate, weight decay, batch size |

Model terminology in checkpoints and protocol logs:

- **encoder / feature_extractor** — shared representation
- **projection** — optional dense block after the encoder
- **regression_head** — PDBbind affinity output
- **classifier_head** — DUDEz active/decoy output
- **decoder** — optional PDBbind reconstruction regularizer

## Architecture constraints

### Encoder

The encoder is sampled to be monotonic (non-increasing):

`encoder_hidden_1 >= encoder_hidden_2 >= ... >= encoder_latent_dim`

Equal-width plateaus are allowed. Expansion inside the encoder is not sampled.

### Decoder

The decoder is optional and only used during PDBbind training when `decoder_lambda_rec > 0`.

Decoder hidden layers may expand toward the input dimension because the decoder maps from latent/projection space back toward the original feature dimension.

The decoder is **not** transferred to DUDEz. Only the encoder/feature extractor and optional projection block are reused for screening.

## Disabling reconstruction

Set `decoder_lambda_rec` to `0.0` in `DecoderSearchSpace.lambda_rec_options`, or sample `0.0` in a trial. This disables the decoder module and reconstruction loss while keeping regression training unchanged.
