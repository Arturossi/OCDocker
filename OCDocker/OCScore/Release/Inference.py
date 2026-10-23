#!/usr/bin/env python3

# Description
###############################################################################
'''
Score an external table with every shipped OCScore model.

Takes a merged OCDocker results table, maps it onto the frozen feature contract, and scores it
with the release model, the ablation arms and the randomised-label control, reporting each seed
alongside its rank ensemble. Nothing is fitted here.

Two properties of the contract constrain the input. Scores and ligand descriptors are
standardised within receptor, so every receptor in the table needs its whole library present and
a lone compound cannot be scored. The 55 context features describe the receptor rather than any
compound, so a new target needs them computed once, with
``OCDocker.OCScore.Release.Inference.attach_context`` or the descriptor tooling, before the table
can be scored.

Usage:

import OCDocker.OCScore.Release.Inference as ocrelinference
'''

from __future__ import annotations

# Imports
###############################################################################
import json

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd
import torch

import OCDocker.OCScore.Analysis.Metrics.Ranking as ocranking
import OCDocker.OCScore.Release.Baselines as ocrelbaselines
import OCDocker.OCScore.Release.Bundle as ocrelbundle
import OCDocker.OCScore.Release.Context as ocrelcontext
import OCDocker.OCScore.Release.Catalog as ocrelcatalog

# License
###############################################################################
'''Copyright (c) Federal University of Rio de Janeiro (UFRJ), Artur Duque Rossi, and Pedro Henrique Monteiro Torres.

SPDX-License-Identifier: BSD-3-Clause

See the LICENSE file for full terms.
'''

# Constants
###############################################################################
ACTIVE_TOKENS = {"active", "actives", "ligand", "ligands", "1", "true"}
PREDICTIONS_CSV = "ocscore_predictions.csv"
METRICS_CSV = "ocscore_metrics.csv"
REPORT_JSON = "ocscore_prediction_report.json"
CONTROL_TOLERANCE = 0.10
BASELINE_FAMILY = "baselines"
CONTEXT_BLOCKS = {"C_receptor": "receptor", "D_pocket": "pocket"}
ROC_AUC_NAMES = ("ROC-AUC", "roc_auc", "ROC-AUC_group_mean")
BEDROC_NAMES = ("BEDROC", "bedroc", "BEDROC_global")

# Classes
###############################################################################


@dataclass
class PredictionConfig:
    '''Inputs for one scoring run.'''

    csv: Union[str, Path]
    output_dir: Union[str, Path]
    models_root: Optional[Union[str, Path]] = None
    families: Optional[tuple[str, ...]] = None
    receptor_descriptors: Optional[Union[str, Path]] = None
    group_column: str = "receptor"
    label_column: Optional[str] = None
    kind_column: str = "kind"
    smiles_column: Optional[str] = None
    keep_kinds: tuple[str, ...] = ("active", "inactive")
    device: str = "cpu"
    bedroc_alpha: float = 20.0
    command: list[str] = field(default_factory=list)


# Functions
###############################################################################
## Private ##


def _bare_protein_names(contract: ocrelbundle.FeatureContract) -> set[str]:
    '''Collect unprefixed names that belong to the protein blocks, not the ligand block.

    Parameters
    ----------
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.

    Returns
    -------
    set of str
        Bare descriptor names used by the receptor and pocket blocks.
    '''

    groups = contract.groups
    protein = groups["C_receptor"] + groups["D_pocket"]
    return {c.split("_", 1)[1] for c in protein}


def _first_present(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    '''Return the first candidate column present in a frame.

    Parameters
    ----------
    frame : pandas.DataFrame
        Table to inspect.
    candidates : sequence of str
        Column names in order of preference.

    Returns
    -------
    str or None
        The first present name, or None when none of them is.
    '''

    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _infer_context_prefix(columns: set, contract: ocrelbundle.FeatureContract) -> Optional[str]:
    '''Decide whether unprefixed protein descriptor columns are the receptor or the pocket set.

    The two sets share 24 of their names, so bare columns are ambiguous in general. Five names
    occur only in the receptor set and two only in the pocket set, and those markers settle it.

    Parameters
    ----------
    columns : set
        Column names present in the table.
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.

    Returns
    -------
    str or None
        ``"receptor"``, ``"pocket"``, or None when the table does not settle the question.
    '''

    groups = contract.groups
    receptor = {c.split("_", 1)[1] for c in groups["C_receptor"]}
    pocket = {c.split("_", 1)[1] for c in groups["D_pocket"]}
    saw_receptor = bool((receptor - pocket) & columns)
    saw_pocket = bool((pocket - receptor) & columns)
    if saw_receptor and not saw_pocket:
        return "receptor"
    if saw_pocket and not saw_receptor:
        return "pocket"
    return None


def _missing_by_block(dataframe: pd.DataFrame,
                      contract: ocrelbundle.FeatureContract) -> dict[str, list[str]]:
    '''Group the absent contract columns by the block they belong to.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Table to inspect.
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.

    Returns
    -------
    dict of str to list of str
        Block name to the columns it is missing, omitting complete blocks.
    '''

    present = set(dataframe.columns)
    out = {}
    for block, names in contract.groups.items():
        absent = [c for c in names if c not in present]
        if absent:
            out[block] = absent
    return out


def _labels_from(dataframe: pd.DataFrame, label_column: Optional[str],
                 kind_column: str) -> Optional[np.ndarray]:
    '''Derive binary labels from an explicit column or from the kind column.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Scored table.
    label_column : str, optional
        Numeric label column, when one exists.
    kind_column : str
        Column naming each compound active or inactive.

    Returns
    -------
    numpy.ndarray or None
        Binary labels, or None when the table carries none.
    '''

    if label_column and label_column in dataframe.columns:
        series = dataframe[label_column]
        if pd.api.types.is_numeric_dtype(series):
            return cast(np.ndarray, series.to_numpy(dtype=float).astype(int))
    if kind_column in dataframe.columns:
        normalised = dataframe[kind_column].astype(str).str.strip().str.lower()
        return cast(np.ndarray, normalised.isin(ACTIVE_TOKENS).to_numpy().astype(int))
    return None


## Public ##


def normalise_columns(dataframe: pd.DataFrame,
                      contract: ocrelbundle.FeatureContract) -> pd.DataFrame:
    '''Rename pipeline columns onto the names the feature contract expects.

    Score columns written in the pipeline's upper-case convention are mapped to their contract
    names, and bare RDKit descriptor columns are given the ``ligand_`` prefix. Columns already
    carrying a contract name are left alone.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Merged pipeline results table.
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.

    Returns
    -------
    pandas.DataFrame
        Table with contract column names.
    '''

    out = dataframe.copy()
    renames = {k: v for k, v in ocrelbundle.SCORE_RENAMES.items()
               if k in out.columns and v not in out.columns}
    out = out.rename(columns=renames)

    known = (set(contract.feature_order)
             | set(ocrelbundle.SCORE_RENAMES)
             | set(ocrelbundle.SCORE_RENAMES.values())
             | _bare_protein_names(contract))
    wanted = {c[len("ligand_"):] for c in contract.groups["B_ligand"]}
    prefix = {c: f"ligand_{c}" for c in out.columns
              if c in wanted and c not in known and f"ligand_{c}" not in out.columns}
    out = out.rename(columns=prefix)

    for block, tag in CONTEXT_BLOCKS.items():
        names = contract.groups[block]
        if any(c in out.columns for c in names):
            continue
        if _infer_context_prefix(set(out.columns), contract) != tag:
            continue
        out = out.rename(columns={c.split("_", 1)[1]: c for c in names
                                  if c.split("_", 1)[1] in out.columns})
    return out


def attach_context(dataframe: pd.DataFrame, descriptors: Union[str, Path, dict],
                   contract: ocrelbundle.FeatureContract,
                   groups: Optional[Any] = None) -> pd.DataFrame:
    '''Write each target's receptor and pocket descriptors onto its own rows.

    Context describes the target, so a table spanning several receptors needs one descriptor set
    per receptor. The mapping may therefore hold a single target's ``receptor`` and ``pocket``
    sub-mappings, applied to every row, or be keyed by receptor identifier.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Table to extend.
    descriptors : str, pathlib.Path or dict
        Descriptor mapping, or a path to its JSON.
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.
    groups : array-like, optional
        Receptor identifier per row, required for a mapping keyed by receptor.

    Returns
    -------
    pandas.DataFrame
        Table carrying every context column named in the contract.
    '''

    if not isinstance(descriptors, dict):
        descriptors = ocrelcontext.load_context(descriptors)

    out = dataframe.copy()
    if not ocrelcontext.is_multi_target(descriptors):
        for block, key in CONTEXT_BLOCKS.items():
            source = descriptors.get(key, {})
            for column in contract.groups[block]:
                if column in source:
                    out[column] = float(source[column])
        return out

    if groups is None:
        raise ValueError(
            "The descriptor file is keyed by receptor, so the receptor of each row is needed to "
            "apply it. Pass the group column, or supply a single-target descriptor file."
        )

    g = np.asarray(groups)
    unknown = sorted({str(t) for t in np.unique(g)} - set(descriptors))
    if unknown:
        raise ValueError(
            f"No descriptors supplied for receptor(s) {unknown}. Context describes the target, "
            "so every receptor in the table needs its own set."
        )

    for block, key in CONTEXT_BLOCKS.items():
        for column in contract.groups[block]:
            values = np.full(len(out), np.nan)
            for target, payload in descriptors.items():
                source = payload.get(key, {})
                if column in source:
                    values[g.astype(str) == target] = float(source[column])
            out[column] = values
    return out


def score_models(contract: ocrelbundle.FeatureContract,
                 specs: Sequence[ocrelcatalog.ModelSpec],
                 dataframe: pd.DataFrame,
                 groups: Any,
                 device: str = "cpu") -> pd.DataFrame:
    '''Score a prepared table with every requested model.

    Each ablation arm is scored through its own input mask, so a model only ever sees the blocks
    it was trained with.

    Parameters
    ----------
    contract : OCDocker.OCScore.Release.Bundle.FeatureContract
        Loaded feature contract.
    specs : sequence of OCDocker.OCScore.Release.Catalog.ModelSpec
        Models to run.
    dataframe : pandas.DataFrame
        Table carrying every contract column.
    groups : array-like
        Receptor identifier per row.
    device : str, optional
        Torch device.

    Returns
    -------
    pandas.DataFrame
        One column per model seed plus one per rank ensemble, aligned to the input rows.
    '''

    features = contract.featurize(dataframe, groups)
    out: dict[str, np.ndarray] = {}

    for spec in specs:
        mask = contract.mask(spec.live_blocks)
        tensor = torch.tensor(features * mask, device=device)
        cons = torch.zeros((len(features), 1), device=device)
        seed_scores: list[np.ndarray] = []
        for path, seed in zip(spec.checkpoints, spec.seeds):
            net, payload = ocrelbundle.load_network(path, device=device)
            recorded = payload.get("live_blocks")
            if recorded is not None and "".join(sorted(recorded)) != spec.live_blocks:
                raise ValueError(
                    f"{spec.name} seed {seed}: checkpoint records live blocks {recorded!r} but "
                    f"the catalogue derived {spec.live_blocks!r}."
                )
            with torch.no_grad():
                values = net(tensor, "scr", cons).cpu().numpy()
            seed_scores.append(values)
            out[f"{spec.name}|seed{seed}"] = values
        if len(seed_scores) > 1:
            out[f"{spec.name}|ensemble"] = ocrelbundle.rank_ensemble(seed_scores, groups)

    return pd.DataFrame(out, index=dataframe.index)


def evaluate_models(predictions: pd.DataFrame, labels: np.ndarray, groups: Any,
                    bedroc_alpha: float = 20.0) -> pd.DataFrame:
    '''Score every prediction column against the supplied labels.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Output of :func:`score_models`.
    labels : numpy.ndarray
        Binary labels aligned to the prediction rows.
    groups : array-like
        Receptor identifier per row; metrics are computed per receptor and averaged.
    bedroc_alpha : float, optional
        BEDROC weighting factor.

    Returns
    -------
    pandas.DataFrame
        One row per prediction column, sorted by BEDROC where available.
    '''

    rows = []
    for column in predictions.columns:
        metrics = ocranking.evaluate_screening_metrics(
            np.asarray(labels, dtype=int),
            predictions[column].to_numpy(dtype=float),
            groups=np.asarray(groups),
            higher_is_better=True,
            bedroc_alpha=bedroc_alpha,
        )
        model, _, variant = column.partition("|")
        rows.append({"model": model, "variant": variant, **metrics})

    frame = pd.DataFrame(rows)
    key = _first_present(frame, BEDROC_NAMES + ROC_AUC_NAMES)
    if key is None:
        return frame
    return frame.sort_values(key, ascending=False).reset_index(drop=True)


def control_verdict(metrics: pd.DataFrame, tolerance: float = CONTROL_TOLERANCE) -> dict[str, Any]:
    '''Judge whether the randomised-label control behaved as a control should.

    The control never saw a true label, so on an evaluation set whose labels reflect binding it
    must sit at chance. A control clearly above chance means the labels are partly predictable
    from the inputs without any activity supervision, which is a property of the set rather than
    of any model, and it makes the ranking of the other models uninterpretable.

    Parameters
    ----------
    metrics : pandas.DataFrame
        Output of :func:`evaluate_models`.
    tolerance : float, optional
        How far above 0.5 ROC-AUC the control may sit before the set is flagged.

    Returns
    -------
    dict
        Verdict with the control ROC-AUC and a human-readable message.
    '''

    key = _first_present(metrics, ROC_AUC_NAMES)
    if key is None:
        return {"status": "not_evaluated", "reason": "no ROC-AUC column in the metrics table"}

    control = metrics[metrics["model"].str.startswith("control_")]
    if control.empty:
        return {"status": "not_run", "reason": "no control family scored"}

    worst = float(control[key].max())
    passed = worst <= 0.5 + tolerance
    return {
        "status": "pass" if passed else "FAIL",
        "control_roc_auc_max": worst,
        "tolerance": tolerance,
        "message": (
            "Randomised-label control is at chance; model ranking can be read normally."
            if passed else
            f"Randomised-label control reaches ROC-AUC {worst:.4f}, above chance. The labels are "
            "partly predictable without activity supervision, so differences between the other "
            "models on this set are not attributable to the models."
        ),
    }


def run_prediction(config: PredictionConfig) -> dict[str, Any]:
    '''Score an external table with the shipped models and write the results.

    Parameters
    ----------
    config : PredictionConfig
        Run configuration.

    Returns
    -------
    dict
        Report describing the run, also written to disk as JSON.
    '''

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = ocrelbundle.FeatureContract(ocrelcatalog.contract_dir(config.models_root))
    checkpoint_families = (tuple(f for f in config.families if f != BASELINE_FAMILY)
                           if config.families else None)
    specs = ocrelcatalog.discover(config.models_root, checkpoint_families)
    if not specs:
        raise FileNotFoundError("No OCScore models found in the generation bundle.")

    table = pd.read_csv(config.csv, low_memory=False)
    if config.kind_column in table.columns and config.keep_kinds:
        table = table[table[config.kind_column].isin(config.keep_kinds)].reset_index(drop=True)

    table = normalise_columns(table, contract)

    if config.group_column not in table.columns:
        raise ValueError(
            f"Group column {config.group_column!r} is absent. Scores and ligand descriptors are "
            "standardised within receptor, so the table must say which receptor each row belongs "
            "to."
        )

    if config.receptor_descriptors is not None:
        table = attach_context(table, config.receptor_descriptors, contract,
                               groups=table[config.group_column].to_numpy())

    absent = _missing_by_block(table, contract)
    if absent:
        summary = "; ".join(f"{b}: {len(v)} of {len(contract.groups[b])} missing, "
                            f"e.g. {v[:3]}" for b, v in absent.items())
        raise ValueError(
            f"The table does not satisfy the feature contract. {summary}. Context blocks "
            "describe the receptor rather than any compound: compute them once for the target "
            "and supply them as prefixed columns or with --receptor-descriptors."
        )

    complete = table[contract.feature_order].notna().all(axis=1)
    dropped = int((~complete).sum())
    table = table[complete].reset_index(drop=True)
    if table.empty:
        raise ValueError("No rows carry a complete set of contract features.")

    groups = table[config.group_column].to_numpy()
    predictions = score_models(contract, specs, table, groups, device=config.device)

    if config.families is None or BASELINE_FAMILY in config.families:
        smiles = (table[config.smiles_column]
                  if config.smiles_column and config.smiles_column in table.columns else None)
        extra = ocrelbaselines.score_baselines(
            contract,
            ocrelcatalog.contract_dir(config.models_root).parent,
            contract.featurize(table, groups),
            smiles=smiles,
        )
        for name, values in extra.items():
            predictions[name] = values

    identity = [c for c in ("name", "ligand", config.group_column, config.kind_column)
                if c in table.columns]
    written = pd.concat([table[identity], predictions], axis=1)
    predictions_path = output_dir / PREDICTIONS_CSV
    written.to_csv(predictions_path, index=False)

    labels = _labels_from(table, config.label_column, config.kind_column)
    metrics_path: Optional[Path] = None
    verdict: dict[str, Any] = {"status": "not_run", "reason": "no labels in the table"}
    if labels is not None and 0 < int(labels.sum()) < len(labels):
        metrics = evaluate_models(predictions, labels, groups, config.bedroc_alpha)
        metrics_path = output_dir / METRICS_CSV
        metrics.to_csv(metrics_path, index=False)
        verdict = control_verdict(metrics)

    report = {
        "generation": ocrelcatalog.GENERATION,
        "models_root": str(ocrelcatalog.contract_dir(config.models_root).parent),
        "models_scored": [s.name for s in specs],
        "baselines_scored": sorted({c.split("|")[0] for c in predictions.columns
                                    if "|" in c and c.split("|")[1] in
                                    ("rule", "linear", "similarity", "control")}),
        "input_csv": str(Path(config.csv).resolve()),
        "n_rows_scored": int(len(table)),
        "n_rows_dropped_incomplete": dropped,
        "n_receptors": int(len(np.unique(groups))),
        "n_actives": int(labels.sum()) if labels is not None else None,
        "group_column": config.group_column,
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path) if metrics_path else None,
        "control_check": verdict,
        "command": list(config.command),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / REPORT_JSON).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "METRICS_CSV",
    "PREDICTIONS_CSV",
    "REPORT_JSON",
    "PredictionConfig",
    "attach_context",
    "control_verdict",
    "evaluate_models",
    "normalise_columns",
    "run_prediction",
    "score_models",
]
