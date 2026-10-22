# OCScore models

Pretrained OCScore models shipped with OCDocker, ready to score a docked library without any
training data.

The current generation is [`gen3b/`](gen3b/README.md). Run it with:

    ocdocker ocscore predict --csv <table>.csv --output-dir <dir>

It ships the frozen M1.1 release model, the eight-arm block ablation, a randomised-label control
and a set of reference baselines (mean and median consensus, ridge and elastic-net regression,
ECFP4 memorisation, random ranking). The control and the baselines are run by default, because a
screening number is only readable against them: `ocdocker ocscore predict` fails its
`control_check` when the randomised-label model scores above chance, which means the evaluation
set's labels are partly predictable without any activity supervision.

See [`gen3b/README.md`](gen3b/README.md) for the input table contract and the two properties of
the feature contract that constrain its use.

## Earlier generations

The 22 DUDEz ablation configurations (`01_*` to `22_*`) that shipped here previously were
export bundles in the older `ModelExport` format, read through `ocdocker ocscore score` and
`ocdocker ocscore external-blind`. They were superseded by `gen3b/` and removed; both of those
commands remain for scoring any bundle still in that format. Recover the old directories from
git history if they are needed.
