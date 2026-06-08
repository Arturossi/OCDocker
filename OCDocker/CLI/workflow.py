#!/usr/bin/env python3
"""Shared docking/pipeline workflow helpers for VS and pipeline commands."""

from __future__ import annotations

import os
import math
import numbers
import time
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import OCDocker.Toolbox.Logging as oclogging

from OCDocker.CLI.common import _db_dependencies_available, _print_optional_dependency_hint

LOGGER = oclogging.get_logger("cli")

def _box_sort_key(path: Path) -> Tuple[int, object]:
    '''Sorting key for box files.

    Boxes named boxN.pdb (N=number) come first, sorted by N. Other names come later, sorted alphabetically.

    Parameters
    ----------
    path : Path
        The box file path.

    Returns
    -------
    Tuple[int, object]
        Sorting key.
    '''

    stem = path.stem.lower()
    if stem.startswith("box"):
        suffix = stem[3:]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, stem)

def _ensure_mol2_poses(pose_paths: List[str], dest_dir: Path, pose_engine_map: Optional[Dict[str, str]] = None) -> Tuple[List[str], Dict[str, str]]:
    '''Ensure a list of poses in MOL2 format, converting when needed.

    Returns a list of .mol2 paths and a mapping mol2->original path.
    Uses unique filenames based on engine source to avoid overwriting.

    Parameters
    ----------
    pose_paths : List[str]
        List of pose file paths to ensure are in MOL2 format.
    dest_dir : Path
        Destination directory for converted MOL2 files.
    pose_engine_map : Dict[str, str], optional
        Mapping from pose path to engine name (gnina, plants, smina, vina) to create unique filenames.

    Returns
    -------
    Tuple[List[str], Dict[str, str]]
        A tuple containing a list of .mol2 paths and a mapping from mol2 paths to original paths.
    '''

    dest_dir.mkdir(parents=True, exist_ok=True)
    mol2_paths: List[str] = []
    mapping: Dict[str, str] = {}

    import OCDocker.Toolbox.Conversion as occonversion
    for p in pose_paths:
        src = Path(p)
        if src.suffix.lower() == ".mol2":
            # Already MOL2 - use as-is but track mapping
            mol2_paths.append(str(src))
            mapping[str(src)] = str(src)
            continue

        # Create unique filename based on engine and original filename
        engine = pose_engine_map.get(str(src), "unknown") if pose_engine_map else "unknown"
        # Include engine in filename to avoid collisions
        unique_name = f"{engine}_{src.stem}.mol2"
        out = dest_dir / unique_name
        _ = occonversion.convert_mols(str(src), str(out), overwrite=True)
        mol2_paths.append(str(out))
        mapping[str(out)] = str(src)
    return mol2_paths, mapping


def _select_pipeline_representative_medoid(
    data: Any,
    clusters: Any,
    medoids: List[str],
) -> Tuple[str, Dict[str, Any]]:
    '''Choose the CLI pipeline representative medoid from the largest clusters.

    The shared clustering helper returns one medoid for each largest cluster.
    When more than one largest cluster exists, the CLI should select the
    medoid with the lowest intra-cluster distance score instead of relying on
    cluster label order alone.
    '''

    if not medoids:
        raise ValueError("medoids must not be empty")

    import numpy as np
    import pandas as pd
    from sklearn.metrics import pairwise_distances

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if isinstance(clusters, int) or getattr(clusters, "size", 0) == 0:
        return medoids[0], {"tie_breaker": "first_medoid_fallback"}

    cluster_sizes = np.bincount(clusters)
    largest_clusters = np.where(cluster_sizes == np.max(cluster_sizes))[0]
    medoid_order = {medoid: index for index, medoid in enumerate(medoids)}

    best_choice: Optional[Tuple[float, int, int]] = None
    best_info: Optional[Dict[str, Any]] = None

    for cluster_id in largest_clusters:
        cluster_data = data[clusters == cluster_id]
        if cluster_data.empty:
            continue

        distances = pairwise_distances(cluster_data, metric='euclidean')
        sum_distances = np.sum(distances, axis=1)
        medoid_index = int(np.argmin(sum_distances))
        medoid_label = cluster_data.index[medoid_index]
        medoid_score = float(sum_distances[medoid_index])
        mean_distance = medoid_score / max(len(cluster_data) - 1, 1)
        medoid_rank = medoid_order.get(medoid_label, len(medoids))

        candidate = (medoid_score, int(cluster_id), medoid_rank)
        if best_choice is None or candidate < best_choice:
            best_choice = candidate
            best_info = {
                "tie_breaker": (
                    "lowest_intra_cluster_medoid_distance"
                    if len(largest_clusters) > 1
                    else "single_largest_cluster"
                ),
                "largest_cluster_ids": [int(cid) for cid in largest_clusters],
                "selected_cluster_id": int(cluster_id),
                "selected_medoid_sum_distance": medoid_score,
                "selected_medoid_mean_distance": mean_distance,
            }
            best_medoid = medoid_label

    if best_info is None:
        return medoids[0], {"tie_breaker": "first_medoid_fallback"}

    return best_medoid, best_info


def _list_boxes(ligand_dir: Path, box_path: Path, all_boxes: bool) -> List[Path]:
    '''List box files to use.

    Parameters
    ----------
    ligand_dir : Path
        Directory containing the ligand file.
    box_path : Path
        Path to the primary box file.
    all_boxes : bool
        Whether to use all box*.pdb files in ligand_dir and box_path.parent.

    Returns
    -------
    List[Path]
        List of box file paths.
    '''

    if not all_boxes:
        return [box_path]

    candidates: List[Path] = []
    for d in {ligand_dir, box_path.parent}:
        candidates.extend(Path(p) for p in glob(str(d / "box*.pdb")))
    if box_path.is_file():
        candidates.append(box_path)

    unique: Dict[str, Path] = {}
    for p in candidates:
        try:
            unique[str(p.resolve())] = p
        except OSError:
            unique[str(p)] = p

    boxes = list(unique.values())
    boxes.sort(key=_box_sort_key)
    return boxes


def _is_integer_descriptor_name(descriptor: str) -> bool:
    '''Return True when a descriptor is expected to be integer-like.'''

    name = descriptor.strip()
    return (
        name.startswith("fr_")
        or name.startswith("Num")
        or name.startswith("count")
        or name in {"HeavyAtomCount", "NHOHCount", "NOCount", "RingCount", "TotalAALength"}
    )


def _to_numeric(value: Any) -> Optional[float]:
    '''Convert numeric-like values to finite float, otherwise return None.'''

    if isinstance(value, bool):
        return float(int(value))
    if not isinstance(value, numbers.Real):
        return None
    numeric_value = float(value)
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return numeric_value


def _rescoring_data_has_numeric_scores(data: Dict[str, Any]) -> bool:
    '''Check whether rescoring payload has at least one finite numeric score.

    Parameters
    ----------
    data : Dict[str, Any]
        Parsed rescoring payload keyed by rescoring label.

    Returns
    -------
    bool
        True when at least one score is numeric and finite, otherwise False.
    '''

    for raw_value in data.values():
        if _to_numeric(raw_value) is not None:
            return True
        if isinstance(raw_value, (list, tuple)):
            for item in raw_value:
                if _to_numeric(item) is not None:
                    return True
    return False


def _collect_log_file_signatures(paths: List[str]) -> Tuple[Dict[str, Tuple[int, int]], bool]:
    '''Collect file signatures for rescoring logs.

    Parameters
    ----------
    paths : List[str]
        Candidate rescoring log paths.

    Returns
    -------
    Tuple[Dict[str, Tuple[int, int]], bool]
        Mapping ``path -> (size_bytes, mtime_ns)`` for files that could be
        stat-ed, plus a flag that indicates at least one unstable/missing path.
    '''

    signatures: Dict[str, Tuple[int, int]] = {}
    has_unstable_paths = False

    for raw_path in sorted(set(paths)):
        path = str(raw_path)
        try:
            stat_result = os.stat(path)
            mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
            signatures[path] = (int(stat_result.st_size), mtime_ns)
        except (FileNotFoundError, PermissionError, OSError):
            has_unstable_paths = True

    return signatures, has_unstable_paths


def _wait_for_rescore_logs_ready(
    get_log_paths: Callable[[str], List[str]],
    read_log_data: Callable[..., Dict[str, Any]],
    out_dir: str,
    *,
    timeout_seconds: float = 2.0,
    poll_interval_seconds: float = 0.05,
) -> Tuple[List[str], Dict[str, Any], bool]:
    '''Wait for rescoring logs to become available and parse-ready.

    Polls log discovery and parsing helpers until either:
    1) at least one finite numeric score is parsed; or
    2) the timeout is reached.

    Parameters
    ----------
    get_log_paths : Callable[[str], List[str]]
        Function that returns rescoring log paths for an output directory.
    read_log_data : Callable[..., Dict[str, Any]]
        Function that parses rescoring logs and returns score data.
    out_dir : str
        Output directory where rescoring logs are expected.
    timeout_seconds : float, optional
        Maximum wait time before giving up. By default 2.0.
    poll_interval_seconds : float, optional
        Delay between polling attempts. By default 0.05.

    Returns
    -------
    Tuple[List[str], Dict[str, Any], bool]
        Tuple with the most recent log paths, parsed data snapshot, and a
        readiness flag indicating whether parse-ready numeric data was found.
    '''

    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    base_poll_interval = max(0.0, float(poll_interval_seconds))
    max_poll_interval = max(base_poll_interval, 0.5)
    last_paths: List[str] = []
    last_data: Dict[str, Any] = {}
    last_signatures: Dict[str, Tuple[int, int]] = {}
    stable_poll_count = 0

    while True:
        try:
            current_paths = sorted(str(path) for path in (get_log_paths(out_dir) or []))
        except Exception:
            current_paths = []

        if current_paths:
            current_signatures, has_unstable_paths = _collect_log_file_signatures(current_paths)
            paths_changed = current_paths != last_paths
            signatures_changed = current_signatures != last_signatures
            should_reparse = paths_changed or signatures_changed or has_unstable_paths or not last_data

            last_paths = list(current_paths)
            last_signatures = current_signatures

            if should_reparse:
                try:
                    current_data = read_log_data(current_paths, onlyBest=True) or {}
                except Exception:
                    current_data = {}

                if current_data:
                    last_data = current_data
                    if _rescoring_data_has_numeric_scores(current_data):
                        return last_paths, last_data, True
                stable_poll_count = 0
            else:
                stable_poll_count += 1
        else:
            stable_poll_count = 0
            last_signatures = {}

        if time.monotonic() >= deadline:
            return last_paths, last_data, False

        sleep_interval = base_poll_interval
        if base_poll_interval > 0.0 and stable_poll_count > 0:
            sleep_interval = min(max_poll_interval, base_poll_interval * (2 ** min(stable_poll_count, 4)))
        time.sleep(sleep_interval)


def _collect_numeric_descriptors(obj: Any, descriptor_names: List[str]) -> Dict[str, Union[int, float]]:
    '''Extract numeric descriptor values from an object into a payload dict.'''

    payload: Dict[str, Union[int, float]] = {}
    for descriptor in descriptor_names:
        if not hasattr(obj, descriptor):
            continue
        numeric_value = _to_numeric(getattr(obj, descriptor))
        if numeric_value is None:
            continue
        if _is_integer_descriptor_name(descriptor):
            payload[descriptor] = int(numeric_value)
        else:
            payload[descriptor] = numeric_value
    return payload


def _map_score_to_complex_column(raw_key: str) -> Optional[str]:
    '''Map raw rescoring keys to Complexes table descriptor columns.'''

    key = raw_key.strip().lower()
    key = key.replace("-", "_").replace(" ", "_").replace(".", "_")
    while "__" in key:
        key = key.replace("__", "_")

    direct_map = {
        "vina_vina": "VINA_VINA",
        "vina_vinardo": "VINA_VINARDO",
        "smina_vina": "SMINA_VINA",
        "smina_vinardo": "SMINA_VINARDO",
        "smina_scoring_dkoes": "SMINA_SCORING_DKOES",
        "smina_dkoes_scoring": "SMINA_SCORING_DKOES",
        "smina_old_scoring_dkoes": "SMINA_OLD_SCORING_DKOES",
        "smina_dkoes_scoring_old": "SMINA_OLD_SCORING_DKOES",
        "smina_fast_dkoes": "SMINA_FAST_DKOES",
        "smina_dkoes_fast": "SMINA_FAST_DKOES",
        "smina_scoring_ad4": "SMINA_SCORING_AD4",
        "smina_ad4_scoring": "SMINA_SCORING_AD4",
        # Keep Gnina mapping hardcoded like other engines, matching gnina_scoring_functions defaults.
        "gnina_ad4_scoring": "GNINA_AD4_SCORING",
        "gnina_scoring_ad4": "GNINA_AD4_SCORING",
        "gnina_ad4": "GNINA_AD4_SCORING",
        "gnina_default": "GNINA_DEFAULT",
        "gnina_dkoes_fast": "GNINA_DKOES_FAST",
        "gnina_fast_dkoes": "GNINA_DKOES_FAST",
        "gnina_dkoes_scoring": "GNINA_DKOES_SCORING",
        "gnina_scoring_dkoes": "GNINA_DKOES_SCORING",
        "gnina_dkoes": "GNINA_DKOES_SCORING",
        "gnina_dkoes_scoring_old": "GNINA_DKOES_SCORING_OLD",
        "gnina_old_scoring_dkoes": "GNINA_DKOES_SCORING_OLD",
        "gnina_vina": "GNINA_VINA",
        "gnina_vinardo": "GNINA_VINARDO",
        "plants_chemplp": "PLANTS_CHEMPLP",
        "plants_plp": "PLANTS_PLP",
        "plants_plp95": "PLANTS_PLP95",
        "oddt_plecrf_p5_l1_s65536": "ODDT_PLECRF_P5_L1_S65536",
        "oddt_nnscore": "ODDT_NNSCORE",
        "oddt_rfscore_v1": "ODDT_RFSCORE_V1",
        "oddt_rfscore_v2": "ODDT_RFSCORE_V2",
        "oddt_rfscore_v3": "ODDT_RFSCORE_V3",
    }
    if key in direct_map:
        return direct_map[key]

    # ODDT can emit several naming variants (e.g. rfscore_v1_pdbbind2016, plecrf_*).
    oddt_key = key[5:] if key.startswith("oddt_") else key
    if "rfscore_v1" in oddt_key or oddt_key.endswith("rfscore1"):
        return "ODDT_RFSCORE_V1"
    if "rfscore_v2" in oddt_key or oddt_key.endswith("rfscore2"):
        return "ODDT_RFSCORE_V2"
    if "rfscore_v3" in oddt_key or oddt_key.endswith("rfscore3"):
        return "ODDT_RFSCORE_V3"
    if "plec" in oddt_key:
        return "ODDT_PLECRF_P5_L1_S65536"
    if "nnscore" in oddt_key:
        return "ODDT_NNSCORE"

    return None


def _flatten_rescoring_to_complex_payload(rescoring: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], List[str]]:
    '''Flatten nested rescoring output into Complexes column payload.'''

    payload: Dict[str, float] = {}
    ignored_keys: List[str] = []

    for engine_scores in rescoring.values():
        if not isinstance(engine_scores, dict):
            continue
        for raw_key, raw_value in engine_scores.items():
            numeric_value = _to_numeric(raw_value)
            if numeric_value is None:
                continue
            column = _map_score_to_complex_column(str(raw_key))
            if not column:
                ignored_keys.append(str(raw_key))
                continue
            payload[column] = numeric_value

    # Keep ignored keys stable and deduplicated for user-facing warnings.
    ignored_keys = sorted(set(ignored_keys))
    return payload, ignored_keys


def _store_pipeline_results_in_db(
    job_name: str,
    receptor: Any,
    ligand: Any,
    rescoring: Dict[str, Dict[str, float]],
    box_label: Optional[str] = None,
) -> Tuple[bool, str, List[str]]:
    '''Store pipeline run data into DB tables (Receptors, Ligands, Complexes).'''

    from OCDocker.DB.DB import create_tables
    from OCDocker.DB.Models.Complexes import Complexes
    from OCDocker.DB.Models.Ligands import Ligands
    from OCDocker.DB.Models.Receptors import Receptors

    create_tables()

    receptor_name = str(getattr(receptor, "name", "") or f"{job_name}_receptor")
    ligand_name = str(getattr(ligand, "name", "") or f"{job_name}_ligand")

    receptor_payload: Dict[str, Union[str, int, float]] = {"name": receptor_name}
    receptor_payload.update(_collect_numeric_descriptors(receptor, list(getattr(Receptors, "allDescriptors", []))))

    ligand_payload: Dict[str, Union[str, int, float]] = {"name": ligand_name}
    ligand_payload.update(_collect_numeric_descriptors(ligand, list(getattr(Ligands, "allDescriptors", []))))

    receptor_ok = Receptors.insert_or_update(receptor_payload)
    ligand_ok = Ligands.insert_or_update(ligand_payload)
    if not receptor_ok or not ligand_ok:
        return False, "", []

    receptor_row = Receptors.find_first(receptor_name)
    ligand_row = Ligands.find_first(ligand_name)
    receptor_id = getattr(receptor_row, "id", None)
    ligand_id = getattr(ligand_row, "id", None)

    complex_name = f"{job_name}_{box_label}" if box_label else job_name
    complex_payload: Dict[str, Union[str, int, float]] = {"name": complex_name}
    if isinstance(receptor_id, int):
        complex_payload["receptor_id"] = receptor_id
    if isinstance(ligand_id, int):
        complex_payload["ligand_id"] = ligand_id

    score_payload, ignored_keys = _flatten_rescoring_to_complex_payload(rescoring)
    complex_payload.update(score_payload)

    complex_ok = Complexes.insert_or_update(complex_payload)
    return bool(complex_ok), complex_name, ignored_keys

