#!/usr/bin/env bash
# Full OCScore pipeline: raw prepare -> train -> export tools.
#
# Foreground:
#   ./examples/18_run_full_pipeline.sh
#
# Overnight (create the log directory first):
#   mkdir -p /data/hd4tb/OCDocker/data/ocdb2/OCScore/output/logs
#   nohup /data/hd4tb/OCDocker/OCDocker/examples/18_run_full_pipeline.sh \
#     > /data/hd4tb/OCDocker/data/ocdb2/OCScore/output/logs/full_pipeline.log 2>&1 &
#   tail -f /data/hd4tb/OCDocker/data/ocdb2/OCScore/output/logs/full_pipeline.log
#
# Stage 1 is skipped automatically when raw_prepare/merged_input_dataset.csv exists.
set -euo pipefail

# =============================================================================
# Configuration — edit these before running
# =============================================================================

# --- Paths ---
DATA="/data/hd4tb/OCDocker/data/ocdb2/OCScore"
OUT="${DATA}/output"

# --- Global ---
# Protocol and analysis values below are intentionally script literals.
# Edit this file or the generated protocol artifacts; do not override them via environment.
TRAIN_SEED="42"
REPLICAS="5"
# Runs independent replicas concurrently. Keep at 1 on memory-limited GPUs.
REPLICA_JOBS="2"
# Reuse completed replica directories at module level when rerunning same output.
RESUME_COMPLETED="true"
DEVICE="cuda"                    # cuda | cpu

# Runtime diagnostics/stability. These do not change the protocol search space.
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# --- Stage 2: Optuna train (PDBbind regression) ---
PDBBIND_TRIALS="100"
PDBBIND_EPOCHS="100"
# Parallel Optuna trials inside each PDBbind replica. Usually keep 1 on one GPU.
PDBBIND_N_JOBS="1"

# --- Stage 2: Optuna train (DUDEz screening) ---
DUDEZ_TRIALS="100"
DUDEZ_EPOCHS="100"
# Parallel Optuna trials inside each DUDEz replica. Usually keep 1 on one GPU.
DUDEZ_N_JOBS="1"

# --- Stage 3d: cross-validation ---
PDBBIND_CV_FOLDS="5"
PDBBIND_CV_EPOCHS="100"
DUDEZ_CV_FOLDS="5"
DUDEZ_CV_EPOCHS="100"

# --- Stage 3e: plots ---
PLOT_DPI="150"
ARCHITECTURE_PLOT_DPI="220"
ARCHITECTURE_PLOT_FORMATS="png"
ARCHITECTURE_PLOT_INCLUDE_DECODER="false"
PDBBIND_CV_PLOT_METRICS="RMSE,MAE,R2"
DUDEZ_CV_PLOT_METRICS="BEDROC,ROC-AUC,PR-AUC,EF1%,EF5%,NDCG@1%,NDCG@5%"

# --- Stage 3f: SHAP ---
SHAP_EXPLAINER="gradient"  # gradient | deep | kernel | permutation

# --- Strict validation + reporting (Stage 2) ---
GENERATE_FINAL_REPORT="true"
RUN_LEAKAGE_AUDIT="true"
RUN_BASELINES="true"
RUN_ABLATIONS="true"
# Run CV, plots, and SHAP for every completed replica in full and ablation protocols.
RUN_REPLICA_ANALYSIS="true"
# Train and analyze full, then each ablation one at a time.
INTERLEAVE_PROTOCOL_ANALYSIS="true"
# Skip analysis blocks that already wrote a completion marker.
RESUME_ANALYSIS="${RESUME_ANALYSIS:-true}"
# When true, fails at protocol load if replicas/trials are below production_claim mins
PRODUCTION_CLAIM_ENFORCE="true"

# =============================================================================
# Derived paths
# =============================================================================

LOG="${OUT}/logs"
RAW="${OUT}/raw_prepare"
TRAIN="${OUT}/train"
EXP="${OUT}/export"
PROTOCOL="${OUT}/protocol.generated.yml"
ANALYSIS_PROTOCOL="${OUT}/analysis_protocol.generated.yml"
OPTUNA_DB="${TRAIN}/optuna.db"
TRAIN_SUMMARY="${TRAIN}/staged_optuna_protocol.json"
MODELING_PDB="${TRAIN}/modeling_pdbbind.csv"
MODELING_DUDEZ="${TRAIN}/modeling_dudez.csv"

PDBBIND="${DATA}/PDBbind.csv"
DUDEZ="${DATA}/DUDEz.csv"

mkdir -p "$LOG" "$RAW" "$TRAIN" "$EXP"

OCDOCKER_OUTPUT_LEVEL="${OCDOCKER_OUTPUT_LEVEL:-2}"
OCDOCKER=(ocdocker --output-level "$OCDOCKER_OUTPUT_LEVEL" --no-splash)
ARCHITECTURE_DECODER_ARGS=()
case "${ARCHITECTURE_PLOT_INCLUDE_DECODER,,}" in
  true|1|yes) ARCHITECTURE_DECODER_ARGS=(--show-decoder) ;;
  *) ;;
esac

LOG_COLOR="${LOG_COLOR:-true}"
if [ -n "${NO_COLOR:-}" ]; then
  LOG_COLOR=false
fi

if [ "$LOG_COLOR" = true ]; then
  LOG_RESET=$'\033[0m'
  LOG_BOLD=$'\033[1m'
  LOG_DIM=$'\033[2m'
  LOG_CYAN=$'\033[36m'
  LOG_BLUE=$'\033[34m'
  LOG_YELLOW=$'\033[33m'
else
  LOG_RESET=""
  LOG_BOLD=""
  LOG_DIM=""
  LOG_CYAN=""
  LOG_BLUE=""
  LOG_YELLOW=""
fi

log_timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_detail() {
  local detail="$1"
  if [[ "$detail" == *=* ]]; then
    printf '  %s%s%s=%s\n' "$LOG_YELLOW" "${detail%%=*}" "$LOG_RESET" "${detail#*=}"
  else
    printf '  %s\n' "$detail"
  fi
}

log_banner() {
  local title="$1"
  shift || true
  printf '\n%s%s%s\n' "$LOG_DIM" "================================================================================" "$LOG_RESET"
  printf '%s[%s]%s %s%s%s\n' "$LOG_DIM" "$(log_timestamp)" "$LOG_RESET" "$LOG_BOLD$LOG_CYAN" "$title" "$LOG_RESET"
  local detail
  for detail in "$@"; do
    if [ -n "$detail" ]; then
      log_detail "$detail"
    fi
  done
  printf '%s%s%s\n' "$LOG_DIM" "================================================================================" "$LOG_RESET"
}

log_step() {
  local title="$1"
  shift || true
  printf '\n%s[%s]%s %s>>>%s %s%s%s\n' "$LOG_DIM" "$(log_timestamp)" "$LOG_RESET" "$LOG_BLUE" "$LOG_RESET" "$LOG_BOLD$LOG_BLUE" "$title" "$LOG_RESET"
  local detail
  for detail in "$@"; do
    if [ -n "$detail" ]; then
      printf '  '
      log_detail "$detail"
    fi
  done
}



if ! command -v "${OCDOCKER[0]}" >/dev/null 2>&1; then
  echo "ERROR: ocdocker is not on PATH. Activate the ocdocker conda env first." >&2
  exit 1
fi

for required_input in "$PDBBIND" "$DUDEZ"; do
  if [ ! -f "$required_input" ]; then
    echo "ERROR: missing input CSV: $required_input" >&2
    exit 1
  fi
done

# --- Build train protocol from parameters above ---
PROTOCOL_ABLATION_VARIANTS=(ligand_only sf_only ligand_sf receptor_sf)
FEATURE_POLICY_ABLATIONS=(
  no_pmi
  no_shape_core
  no_shape_core_no_receptor_length_pair
  no_shape_core_no_receptor_surface_counts
  no_shape_core_no_receptor_surface_size
  no_ligand_shape_size
  shape_only
  scoring_function_only
  ligand_plus_scoring_function
  ligand_plus_scoring_function_no_shape_core
  ligand_plus_scoring_function_no_shape_size
  ligand_plus_scoring_function_no_pmi
  ligand_plus_scoring_function_no_plants
  ligand_plus_scoring_function_no_shape_size_no_autocorr2d
  ligand_plus_scoring_function_clean_receptor
  no_scoring_function
  ligand_only
  receptor_plus_scoring_function
)
TOTAL_TRAIN_EXPERIMENTS=1
if [ "$RUN_ABLATIONS" = true ]; then
  TOTAL_TRAIN_EXPERIMENTS=$((TOTAL_TRAIN_EXPERIMENTS + ${#FEATURE_POLICY_ABLATIONS[@]}))
fi

generate_final_report_yaml=$([ "$GENERATE_FINAL_REPORT" = true ] && echo "true" || echo "false")
run_leakage_audit_yaml=$([ "$RUN_LEAKAGE_AUDIT" = true ] && echo "true" || echo "false")
run_baselines_yaml=$([ "$RUN_BASELINES" = true ] && echo "true" || echo "false")

EFFECTIVE_RESUME_COMPLETED="$RESUME_COMPLETED"
if [ "$INTERLEAVE_PROTOCOL_ANALYSIS" = true ] && [ "$RUN_ABLATIONS" = true ] && [ "$EFFECTIVE_RESUME_COMPLETED" != true ]; then
  log_step "CONFIG | interleaved ablations need resume" "resume_completed=true"
  EFFECTIVE_RESUME_COMPLETED=true
fi

write_train_protocol() {
  local protocol_path="$1"
  local ablation_enabled="$2"
  shift 2
  local variants=("$@")

  cat > "$protocol_path" <<EOF
name: full-pipeline-run
description: >
  Generated by examples/18_run_full_pipeline.sh.
  replicas=${REPLICAS}, replica_jobs=${REPLICA_JOBS},
  pdbbind trials=${PDBBIND_TRIALS} epochs=${PDBBIND_EPOCHS} n_jobs=${PDBBIND_N_JOBS},
  dudez trials=${DUDEZ_TRIALS} epochs=${DUDEZ_EPOCHS} n_jobs=${DUDEZ_N_JOBS},
  interleave_protocol_analysis=${INTERLEAVE_PROTOCOL_ANALYSIS}.

replicas: ${REPLICAS}
seed: ${TRAIN_SEED}

pdbbind:
  target_column: experimental
  trials: ${PDBBIND_TRIALS}
  epochs: ${PDBBIND_EPOCHS}
  n_jobs: ${PDBBIND_N_JOBS}
  search_phase: full
  enable_pruning: true
  split:
    strategy: receptor_heldout
    train_size: 0.6
    validation_size: 0.2
    test_size: 0.2

dudez:
  kind_column: kind
  positive_kind: ligands
  negative_kind: decoys
  trials: ${DUDEZ_TRIALS}
  epochs: ${DUDEZ_EPOCHS}
  n_jobs: ${DUDEZ_N_JOBS}
  primary_metric: BEDROC
  bedroc_alpha: 20.0
  scaling_strategy: pdbbind_scaler
  ignore_unknown_kind: false

ablation:
  enabled: ${ablation_enabled}
EOF

  if [ "${#variants[@]}" -eq 0 ]; then
    variants=("${PROTOCOL_ABLATION_VARIANTS[@]}")
  fi
  cat >> "$protocol_path" <<EOF
  variants:
EOF
  local variant
  for variant in "${variants[@]}"; do
    printf '    - %s\n' "$variant" >> "$protocol_path"
  done

  cat >> "$protocol_path" <<EOF

runtime:
  use_gpu: $([ "$DEVICE" = cuda ] && echo "true" || echo "false")
  pdbbind_only: false
  replica_jobs: ${REPLICA_JOBS}
  resume_completed: ${EFFECTIVE_RESUME_COMPLETED}

reporting:
  generate_final_report: ${generate_final_report_yaml}
  run_leakage_audit: ${run_leakage_audit_yaml}
  run_baselines: ${run_baselines_yaml}
  calibration_report_mode: ranking_only
EOF

  if [ "$PRODUCTION_CLAIM_ENFORCE" = true ]; then
    cat >> "$protocol_path" <<EOF

production_claim:
  enforce: true
  min_replicas: ${REPLICAS}
  min_pdbbind_trials: ${PDBBIND_TRIALS}
  min_dudez_trials: ${DUDEZ_TRIALS}
EOF
  fi
}

write_train_protocol "$PROTOCOL" false

write_analysis_protocol() {
  local protocol_path="$1"
  cat > "$protocol_path" <<EOF
name: full-pipeline-analysis
training_protocol: ${PROTOCOL}
description: >
  Post-training analysis settings generated by examples/18_run_full_pipeline.sh.
  These settings affect validation analyses, figures, SHAP, scoring exports,
  resume behavior, and analysis execution order.

seed: ${TRAIN_SEED}
device: ${DEVICE}

execution:
  interleave_protocol_analysis: ${INTERLEAVE_PROTOCOL_ANALYSIS}
  run_replica_analysis: ${RUN_REPLICA_ANALYSIS}
  resume_analysis: ${RESUME_ANALYSIS}
  run_ablations: ${RUN_ABLATIONS}
  score_full_dudez_table: true

cross_validation:
  pdbbind:
    folds: ${PDBBIND_CV_FOLDS}
    epochs: ${PDBBIND_CV_EPOCHS}
    seed: ${TRAIN_SEED}
  dudez:
    folds: ${DUDEZ_CV_FOLDS}
    epochs: ${DUDEZ_CV_EPOCHS}
    seed: ${TRAIN_SEED}

plots:
  dpi: ${PLOT_DPI}
  pdbbind_metrics: "${PDBBIND_CV_PLOT_METRICS}"
  dudez_metrics: "${DUDEZ_CV_PLOT_METRICS}"

architecture_plots:
  dpi: ${ARCHITECTURE_PLOT_DPI}
  formats: "${ARCHITECTURE_PLOT_FORMATS}"
  include_decoder: ${ARCHITECTURE_PLOT_INCLUDE_DECODER}

shap:
  explainer: ${SHAP_EXPLAINER}
  seed: ${TRAIN_SEED}
EOF
}

write_analysis_protocol "$ANALYSIS_PROTOCOL"

log_banner "CURRENT RUN | OCScore full pipeline" \
  "output_root=${OUT}" \
  "train_root=${TRAIN}" \
  "export_root=${EXP}" \
  "experiments=${TOTAL_TRAIN_EXPERIMENTS}" \
  "replicas=${REPLICAS}" \
  "replica_jobs=${REPLICA_JOBS}" \
  "pdbbind=${PDBBIND_TRIALS} trials x ${PDBBIND_EPOCHS} epochs" \
  "dudez=${DUDEZ_TRIALS} trials x ${DUDEZ_EPOCHS} epochs" \
  "device=${DEVICE}" \
  "resume_completed=${EFFECTIVE_RESUME_COMPLETED}"

log_banner "CURRENT STEP | PROTOCOL | generated" "path=${PROTOCOL}" "analysis_protocol=${ANALYSIS_PROTOCOL}" "interleave_protocol_analysis=${INTERLEAVE_PROTOCOL_ANALYSIS}" "run_ablations=${RUN_ABLATIONS}"

# --- Stage 1: merge raw pipeline tables (no global feature reduction) ---
if [ -f "${RAW}/merged_input_dataset.csv" ]; then
  log_banner "CURRENT STEP | STAGE 1 | raw input" "status=skipped" "found=${RAW}/merged_input_dataset.csv"
else
  log_banner "CURRENT STEP | STAGE 1 | raw input" "status=running" "pdbbind=${PDBBIND}" "dudez=${DUDEZ}" "output=${RAW}"
  "${OCDOCKER[@]}" ocscore reduce \
    --pdbbind-archive "$PDBBIND" \
    --dudez-archive "$DUDEZ" \
    --output-dir "$RAW"
fi

if [ ! -f "${RAW}/merged_input_dataset.csv" ]; then
  echo "ERROR: Stage 1 did not produce ${RAW}/merged_input_dataset.csv" >&2
  exit 1
fi

# --- Stage 2/3: staged training and analysis helpers ---
require_protocol_artifacts() {
  local protocol_label="$1"
  local protocol_dir="$2"
  local required_train_artifact
  for required_train_artifact in \
    "${protocol_dir}/staged_optuna_protocol.json" \
    "${protocol_dir}/modeling_pdbbind.csv" \
    "${protocol_dir}/modeling_dudez.csv"; do
    if [ ! -f "$required_train_artifact" ]; then
      echo "ERROR: ${protocol_label} did not produce ${required_train_artifact}" >&2
      exit 1
    fi
  done
}

run_train_stage() {
  local stage_label="$1"
  local protocol_path="$2"
  local train_output_dir="${3:-$TRAIN}"
  local feature_policy="${4:-full_ocscore}"
  local experiment_index="${5:-?}"
  local experiment_total="${6:-$TOTAL_TRAIN_EXPERIMENTS}"
  local experiment_kind="${7:-training}"
  local resume_hint="will_run_or_resume"
  if [ -f "${train_output_dir}/staged_optuna_protocol.json" ]; then
    resume_hint="summary_exists_resume_expected"
  elif [ -d "${train_output_dir}" ]; then
    resume_hint="partial_output_may_be_cleaned"
  fi

  log_banner "CURRENT EXPERIMENT | ${experiment_index}/${experiment_total} | ${stage_label}" \
    "kind=${experiment_kind}" \
    "feature_policy=${feature_policy}" \
    "output=${train_output_dir}" \
    "resume_hint=${resume_hint}" \
    "replicas=${REPLICAS}" \
    "replica_jobs=${REPLICA_JOBS}" \
    "pdbbind_trials=${PDBBIND_TRIALS}" \
    "dudez_trials=${DUDEZ_TRIALS}" \
    "device=${DEVICE}" \
    "protocol=${protocol_path}"
  local train_cmd=(
    "${OCDOCKER[@]}"
    ocscore train
    --protocol "$protocol_path"
    --raw-input-dir "$RAW"
    --output-dir "$train_output_dir"
    --feature-policy "$feature_policy"
  )
  "${train_cmd[@]}"
}

write_analysis_marker() {
  local marker="$1"
  local protocol_label="$2"
  local scope="$3"
  local extra_json="${4:-{}}"
  mkdir -p "$(dirname "$marker")"
  ANALYSIS_MARKER="$marker" \
  ANALYSIS_PROTOCOL_LABEL="$protocol_label" \
  ANALYSIS_SCOPE="$scope" \
  ANALYSIS_EXTRA_JSON="$extra_json" \
  SHAP_EXPLAINER_VALUE="$SHAP_EXPLAINER" \
  ANALYSIS_PROTOCOL_PATH="$ANALYSIS_PROTOCOL" \
  python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

marker = Path(os.environ["ANALYSIS_MARKER"])
raw_extra = os.environ.get("ANALYSIS_EXTRA_JSON") or "{}"
try:
    extra = json.loads(raw_extra)
except json.JSONDecodeError:
    extra, end = json.JSONDecoder().raw_decode(raw_extra)
    trailing = raw_extra[end:].strip()
    if trailing:
        print(
            f"WARNING: ignored trailing marker JSON payload text for {marker}: {trailing[:160]}",
            file=__import__("sys").stderr,
        )
analysis_protocol = Path(os.environ["ANALYSIS_PROTOCOL_PATH"])
payload = {
    "protocol_label": os.environ["ANALYSIS_PROTOCOL_LABEL"],
    "scope": os.environ["ANALYSIS_SCOPE"],
    "analysis_protocol": str(analysis_protocol),
    "analysis_protocol_sha256": __import__("hashlib").sha256(analysis_protocol.read_bytes()).hexdigest() if analysis_protocol.is_file() else None,
    "shap_explainer": os.environ["SHAP_EXPLAINER_VALUE"],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
}
payload.update(extra)
marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_protocol_best_analysis() {
  local protocol_label="$1"
  local protocol_dir="$2"
  local output_base="$3"
  local run_score="$4"
  local pdb_csv="${protocol_dir}/modeling_pdbbind.csv"
  local dudez_csv="${protocol_dir}/modeling_dudez.csv"
  local marker="${output_base}/best_model_analysis_complete.json"

  require_protocol_artifacts "$protocol_label" "$protocol_dir"

  log_banner "CURRENT STEP | ANALYSIS | ${protocol_label} | best model" "action=resolve best replicas" "protocol_dir=${protocol_dir}" "output=${output_base}"
  local best_pdb
  local best_dudez
  read -r best_pdb best_dudez < <(
    TRAIN_DIR="$protocol_dir" python - <<'PY'
import json
import os
from pathlib import Path

train = Path(os.environ["TRAIN_DIR"])
payload = json.loads((train / "staged_optuna_protocol.json").read_text(encoding="utf-8"))
agg = payload["aggregate_summary"]
print(agg["best_pdbbind_replica"]["replica_name"], agg["best_dudez_replica"]["replica_name"])
PY
  )

  local pdb_export="${protocol_dir}/${best_pdb}/pdbbind/best_model"
  local dudez_export="${protocol_dir}/${best_dudez}/dudez/best_model"
  local pdb_for_dudez="${protocol_dir}/${best_dudez}/pdbbind/best_model"
  local required_export
  for required_export in "$pdb_export" "$dudez_export" "$pdb_for_dudez"; do
    if [ ! -d "$required_export" ]; then
      echo "ERROR: missing export bundle directory: ${required_export}" >&2
      exit 1
    fi
  done

  echo "Best PDBbind replica (${protocol_label}): ${best_pdb} -> ${pdb_export}"
  echo "Best DUDEz replica (${protocol_label}):   ${best_dudez} -> ${dudez_export}"

  log_step "ANALYSIS | ${protocol_label} | architecture figures" "formats=${ARCHITECTURE_PLOT_FORMATS}" "dpi=${ARCHITECTURE_PLOT_DPI}"
  "${OCDOCKER[@]}" ocscore architecture-plot \
    --export-dir "$pdb_export" \
    --output-dir "${output_base}/pdbbind/figures/architecture" \
    --formats "$ARCHITECTURE_PLOT_FORMATS" \
    --dpi "$ARCHITECTURE_PLOT_DPI" \
    "${ARCHITECTURE_DECODER_ARGS[@]}"

  "${OCDOCKER[@]}" ocscore architecture-plot \
    --export-dir "$dudez_export" \
    --output-dir "${output_base}/dudez/figures/architecture" \
    --formats "$ARCHITECTURE_PLOT_FORMATS" \
    --dpi "$ARCHITECTURE_PLOT_DPI" \
    "${ARCHITECTURE_DECODER_ARGS[@]}"

  if [ "$RESUME_ANALYSIS" = true ] && [ -f "$marker" ]; then
    log_banner "CURRENT STEP | ANALYSIS | skipped" "protocol=${protocol_label}" "scope=best_model" "found=${marker}"
    return 0
  fi

  log_step "ANALYSIS | ${protocol_label} | validate export bundles" "pdbbind_export=${pdb_export}" "dudez_export=${dudez_export}"
  "${OCDOCKER[@]}" ocscore validate --export-dir "$pdb_export"
  "${OCDOCKER[@]}" ocscore validate --export-dir "$dudez_export"

  log_step "ANALYSIS | ${protocol_label} | load smoke test" "device=${DEVICE}"
  "${OCDOCKER[@]}" ocscore load --export-dir "$pdb_export" --device "$DEVICE"
  "${OCDOCKER[@]}" ocscore load --export-dir "$dudez_export" --device "$DEVICE"

  log_step "ANALYSIS | ${protocol_label} | retrain from exports" "device=${DEVICE}"
  "${OCDOCKER[@]}" ocscore retrain \
    --export-dir "$pdb_export" \
    --pdbbind-csv "$pdb_csv" \
    --dudez-csv "$dudez_csv" \
    --device "$DEVICE"

  "${OCDOCKER[@]}" ocscore retrain \
    --export-dir "$dudez_export" \
    --pdbbind-csv "$pdb_csv" \
    --dudez-csv "$dudez_csv" \
    --device "$DEVICE"

  log_step "ANALYSIS | ${protocol_label} | cross-validation" "pdbbind_folds=${PDBBIND_CV_FOLDS} epochs=${PDBBIND_CV_EPOCHS}" "dudez_folds=${DUDEZ_CV_FOLDS} epochs=${DUDEZ_CV_EPOCHS}"
  "${OCDOCKER[@]}" ocscore cross-validate \
    --export-dir "$pdb_export" \
    --pdbbind-csv "$pdb_csv" \
    --dudez-csv "$dudez_csv" \
    --n-folds "$PDBBIND_CV_FOLDS" \
    --seed "$TRAIN_SEED" \
    --epochs "$PDBBIND_CV_EPOCHS" \
    --output-dir "${output_base}/pdbbind/cross_validation" \
    --device "$DEVICE"

  "${OCDOCKER[@]}" ocscore cross-validate \
    --export-dir "$dudez_export" \
    --pdbbind-csv "$pdb_csv" \
    --dudez-csv "$dudez_csv" \
    --n-folds "$DUDEZ_CV_FOLDS" \
    --seed "$TRAIN_SEED" \
    --epochs "$DUDEZ_CV_EPOCHS" \
    --output-dir "${output_base}/dudez/cross_validation" \
    --device "$DEVICE"

  log_step "ANALYSIS | ${protocol_label} | CV plots" "dpi=${PLOT_DPI}" "pdbbind_metrics=${PDBBIND_CV_PLOT_METRICS}" "dudez_metrics=${DUDEZ_CV_PLOT_METRICS}"
  "${OCDOCKER[@]}" ocscore plot \
    --export-dir "$pdb_export" \
    --cv-dir "${output_base}/pdbbind/cross_validation" \
    --figures-dir "${output_base}/pdbbind/figures" \
    --metrics "$PDBBIND_CV_PLOT_METRICS" \
    --dpi "$PLOT_DPI"

  "${OCDOCKER[@]}" ocscore plot \
    --export-dir "$dudez_export" \
    --cv-dir "${output_base}/dudez/cross_validation" \
    --figures-dir "${output_base}/dudez/figures" \
    --metrics "$DUDEZ_CV_PLOT_METRICS" \
    --dpi "$PLOT_DPI"

  log_step "ANALYSIS | ${protocol_label} | SHAP" "explainer=${SHAP_EXPLAINER}" "device=${DEVICE}"
  "${OCDOCKER[@]}" ocscore shap \
    --export-dir "$pdb_export" \
    --pdbbind-csv "$pdb_csv" \
    --seed "$TRAIN_SEED" \
    --device "$DEVICE" \
    --explainer "$SHAP_EXPLAINER" \
    --output-dir "${output_base}/pdbbind/shap"

  "${OCDOCKER[@]}" ocscore shap \
    --export-dir "$dudez_export" \
    --dudez-csv "$dudez_csv" \
    --seed "$TRAIN_SEED" \
    --device "$DEVICE" \
    --explainer "$SHAP_EXPLAINER" \
    --output-dir "${output_base}/dudez/shap"

  if [ "$run_score" = true ]; then
    log_step "ANALYSIS | ${protocol_label} | score full DUDEz table" "output=${output_base}/dudez_predictions.csv"
    "${OCDOCKER[@]}" ocscore score \
      --export-dir "$dudez_export" \
      --pdbbind-export-dir "$pdb_for_dudez" \
      --raw-archive "$DUDEZ" \
      --output-csv "${output_base}/dudez_predictions.csv" \
      --device "$DEVICE"
  fi

  local best_marker_extra
  best_marker_extra=$(python - <<PY
import json
print(json.dumps({
    "best_pdbbind_replica": "${best_pdb}",
    "best_dudez_replica": "${best_dudez}",
    "best_model_paths": {
        "pdbbind": "${pdb_export}",
        "dudez": "${dudez_export}",
    },
}))
PY
  )
  write_analysis_marker "$marker" "$protocol_label" "best_model" "$best_marker_extra"
}

run_replica_task_analysis() {
  local protocol_label="$1"
  local replica_name="$2"
  local task="$3"
  local export_dir="$4"
  local pdb_csv="$5"
  local dudez_csv="$6"
  local output_base="$7"

  if [ ! -d "$export_dir" ]; then
    log_step "REPLICA ANALYSIS | skipped" "protocol=${protocol_label}" "replica=${replica_name}" "task=${task}" "reason=missing export" "path=${export_dir}"
    return 0
  fi

  local folds
  local epochs
  local cv_dir="${output_base}/${task}/cross_validation"
  local figures_dir="${output_base}/${task}/figures"
  local architecture_figures_dir="${figures_dir}/architecture"
  local shap_dir="${output_base}/${task}/shap"
  local marker="${output_base}/${task}/analysis_complete.json"

  if [ "$task" = "pdbbind" ]; then
    folds="$PDBBIND_CV_FOLDS"
    epochs="$PDBBIND_CV_EPOCHS"
  else
    folds="$DUDEZ_CV_FOLDS"
    epochs="$DUDEZ_CV_EPOCHS"
  fi

  log_banner "CURRENT STEP | REPLICA ANALYSIS | ${protocol_label} | ${replica_name} | ${task}" "export=${export_dir}" "output=${output_base}/${task}"
  log_step "REPLICA ANALYSIS | ${protocol_label} | ${replica_name} | ${task} | architecture figures" "formats=${ARCHITECTURE_PLOT_FORMATS}" "dpi=${ARCHITECTURE_PLOT_DPI}" "output=${architecture_figures_dir}"
  "${OCDOCKER[@]}" ocscore architecture-plot \
    --export-dir "$export_dir" \
    --output-dir "$architecture_figures_dir" \
    --formats "$ARCHITECTURE_PLOT_FORMATS" \
    --dpi "$ARCHITECTURE_PLOT_DPI" \
    "${ARCHITECTURE_DECODER_ARGS[@]}"

  if [ "$RESUME_ANALYSIS" = true ] && [ -f "$marker" ]; then
    log_step "REPLICA ANALYSIS | skipped" "protocol=${protocol_label}" "replica=${replica_name}" "task=${task}" "found=${marker}"
    return 0
  fi

  "${OCDOCKER[@]}" ocscore cross-validate \
    --export-dir "$export_dir" \
    --pdbbind-csv "$pdb_csv" \
    --dudez-csv "$dudez_csv" \
    --n-folds "$folds" \
    --seed "$TRAIN_SEED" \
    --epochs "$epochs" \
    --output-dir "$cv_dir" \
    --device "$DEVICE"

  "${OCDOCKER[@]}" ocscore plot \
    --export-dir "$export_dir" \
    --cv-dir "$cv_dir" \
    --figures-dir "$figures_dir" \
    --dpi "$PLOT_DPI"

  if [ "$task" = "pdbbind" ]; then
    "${OCDOCKER[@]}" ocscore shap \
      --export-dir "$export_dir" \
      --pdbbind-csv "$pdb_csv" \
      --seed "$TRAIN_SEED" \
      --device "$DEVICE" \
      --explainer "$SHAP_EXPLAINER" \
      --output-dir "$shap_dir"
  else
    "${OCDOCKER[@]}" ocscore shap \
      --export-dir "$export_dir" \
      --dudez-csv "$dudez_csv" \
      --seed "$TRAIN_SEED" \
      --device "$DEVICE" \
      --explainer "$SHAP_EXPLAINER" \
      --output-dir "$shap_dir"
  fi

  write_analysis_marker "$marker" "$protocol_label/${replica_name}" "$task"
}

run_protocol_replica_analysis() {
  local protocol_label="$1"
  local protocol_dir="$2"
  local pdb_csv="${protocol_dir}/modeling_pdbbind.csv"
  local dudez_csv="${protocol_dir}/modeling_dudez.csv"

  if [ ! -d "$protocol_dir" ]; then
    log_step "REPLICA ANALYSIS | skipped" "protocol=${protocol_label}" "reason=missing protocol directory" "path=${protocol_dir}"
    return 0
  fi
  if [ ! -f "$pdb_csv" ] || [ ! -f "$dudez_csv" ]; then
    log_step "REPLICA ANALYSIS | skipped" "protocol=${protocol_label}" "reason=missing modeling CSVs"
    return 0
  fi

  shopt -s nullglob
  local replica_dir
  for replica_dir in "${protocol_dir}"/replica_*; do
    local replica_name
    replica_name="$(basename "$replica_dir")"
    if [ ! -f "${replica_dir}/protocol_log.json" ]; then
      log_step "REPLICA ANALYSIS | skipped" "protocol=${protocol_label}" "replica=${replica_name}" "reason=incomplete replica"
      continue
    fi

    local output_base="${EXP}/replica_analysis/${protocol_label}/${replica_name}"
    run_replica_task_analysis "$protocol_label" "$replica_name" "pdbbind" \
      "${replica_dir}/pdbbind/best_model" "$pdb_csv" "$dudez_csv" "$output_base"
    run_replica_task_analysis "$protocol_label" "$replica_name" "dudez" \
      "${replica_dir}/dudez/best_model" "$pdb_csv" "$dudez_csv" "$output_base"
  done
  shopt -u nullglob
}

write_combined_ablation_summary() {
  if [ "$RUN_ABLATIONS" != true ]; then
    return 0
  fi

  TRAIN_DIR="$TRAIN" \
  FEATURE_POLICY_ABLATIONS_STR="${FEATURE_POLICY_ABLATIONS[*]}" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path
from typing import Any

train = Path(os.environ["TRAIN_DIR"])
ablation_root = train / "ablations"
variants = os.environ["FEATURE_POLICY_ABLATIONS_STR"].split()

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def build_row(variant: str, output_dir: Path) -> dict[str, Any] | None:
    summary_path = output_dir / "staged_optuna_protocol.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    aggregate = summary.get("aggregate_summary") or {}
    static_context = summary.get("static_context") or {}
    replicated_paths = summary.get("replicated_output_paths") or {}
    row: dict[str, Any] = {
        "feature_policy_name": variant,
        "output_dir": str(output_dir),
        "n_selected_features": static_context.get("n_selected_features"),
        "selected_features_hash": static_context.get("selected_features_hash"),
        "n_replicas": summary.get("n_replicas"),
        "n_successful_replicas": aggregate.get("n_successful_replicas"),
        "n_failed_replicas": aggregate.get("n_failed_replicas"),
        "staged_optuna_protocol_json": str(summary_path),
        "replicas_summary_csv": replicated_paths.get("replicas_summary_csv", str(output_dir / "replicas_summary.csv")),
        "aggregate_summary": aggregate,
        "output_paths": {
            **replicated_paths,
            "staged_optuna_protocol_json": str(summary_path),
            "staged_optuna_protocol_md": str(output_dir / "staged_optuna_protocol.md"),
        },
    }
    metrics = aggregate.get("metrics") or {}
    if isinstance(metrics, dict):
        for metric_name, metric_summary in metrics.items():
            if isinstance(metric_summary, dict):
                for stat_name in ("mean", "std", "median", "min", "max", "n"):
                    if stat_name in metric_summary:
                        row[f"{metric_name}_{stat_name}"] = metric_summary.get(stat_name)
    return row

rows = []
full_row = build_row("full_ocscore", train)
if full_row is not None:
    rows.append(full_row)
for variant in variants:
    row = build_row(variant, ablation_root / variant)
    if row is not None:
        rows.append(row)

if not rows:
    raise SystemExit("No completed full or ablation summaries found to consolidate.")

ablation_root.mkdir(parents=True, exist_ok=True)
json_path = ablation_root / "ablation_summary.json"
csv_path = ablation_root / "ablation_summary.csv"
payload = {
    "protocol": "full-pipeline-run",
    "full_protocol_output_dir": str(train),
    "variants": rows,
}
json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

fieldnames = []
for row in rows:
    for key in row:
        if key in {"aggregate_summary", "output_paths"}:
            continue
        if key not in fieldnames:
            fieldnames.append(key)
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        flat = {key: row.get(key) for key in fieldnames}
        if isinstance(flat.get("feature_blocks"), list):
            flat["feature_blocks"] = "+".join(flat["feature_blocks"])
        writer.writerow(flat)

final_report_path = train / "final_report.json"
if final_report_path.is_file():
    final_report = load_json(final_report_path)
    final_report["ablations"] = {
        "output_paths": {
            "ablation_summary_json": str(json_path),
            "ablation_summary_csv": str(csv_path),
        }
    }
    final_report_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"=== Combined ablation summaries: json={json_path} csv={csv_path} variants={len(rows)} ===")
PY
}

run_full_analysis_block() {
  run_protocol_best_analysis "full" "$TRAIN" "$EXP" true
  if [ "$RUN_REPLICA_ANALYSIS" = true ]; then
    run_protocol_replica_analysis "full" "$TRAIN"
  else
    log_step "REPLICA ANALYSIS | skipped" "protocol=full" "reason=RUN_REPLICA_ANALYSIS=false"
  fi
}

run_ablation_analysis_block() {
  local ablation_variant="$1"
  local protocol_dir="${TRAIN}/ablations/${ablation_variant}"
  run_protocol_best_analysis "ablations/${ablation_variant}" "$protocol_dir" "${EXP}/ablations/${ablation_variant}" false
  if [ "$RUN_REPLICA_ANALYSIS" = true ]; then
    run_protocol_replica_analysis "ablations/${ablation_variant}" "$protocol_dir"
  else
    log_step "REPLICA ANALYSIS | skipped" "protocol=ablations/${ablation_variant}" "reason=RUN_REPLICA_ANALYSIS=false"
  fi
}

if [ "$INTERLEAVE_PROTOCOL_ANALYSIS" = true ]; then
  experiment_index=1
  run_train_stage "full model" "$PROTOCOL" "$TRAIN" full_ocscore "$experiment_index" "$TOTAL_TRAIN_EXPERIMENTS" "baseline"
  require_protocol_artifacts "full" "$TRAIN"
  run_full_analysis_block

  if [ "$RUN_ABLATIONS" = true ]; then
    for ablation_variant in "${FEATURE_POLICY_ABLATIONS[@]}"; do
      experiment_index=$((experiment_index + 1))
      run_train_stage "ablation ${ablation_variant}" "$PROTOCOL" "${TRAIN}/ablations/${ablation_variant}" "$ablation_variant" "$experiment_index" "$TOTAL_TRAIN_EXPERIMENTS" "ablation"
      require_protocol_artifacts "ablation ${ablation_variant}" "${TRAIN}/ablations/${ablation_variant}"
      run_ablation_analysis_block "$ablation_variant"
    done

    run_train_stage "full report refresh" "$PROTOCOL" "$TRAIN" full_ocscore "refresh" "$TOTAL_TRAIN_EXPERIMENTS" "summary_refresh"
    require_protocol_artifacts "full" "$TRAIN"
    write_combined_ablation_summary
  else
    log_banner "CURRENT STEP | ABLATIONS | skipped" "reason=RUN_ABLATIONS=false"
  fi
else
  experiment_index=1
  run_train_stage "full model" "$PROTOCOL" "$TRAIN" full_ocscore "$experiment_index" "$TOTAL_TRAIN_EXPERIMENTS" "baseline"
  require_protocol_artifacts "full" "$TRAIN"
  run_full_analysis_block

  if [ "$RUN_ABLATIONS" = true ]; then
    for ablation_variant in "${FEATURE_POLICY_ABLATIONS[@]}"; do
      experiment_index=$((experiment_index + 1))
      run_train_stage "ablation ${ablation_variant}" "$PROTOCOL" "${TRAIN}/ablations/${ablation_variant}" "$ablation_variant" "$experiment_index" "$TOTAL_TRAIN_EXPERIMENTS" "ablation"
      require_protocol_artifacts "ablation ${ablation_variant}" "${TRAIN}/ablations/${ablation_variant}"
      run_ablation_analysis_block "$ablation_variant"
    done
    write_combined_ablation_summary
  fi
fi

log_banner "CURRENT STEP | DONE" "optuna_db=${OPTUNA_DB}" "train_outputs=${TRAIN}" "export_outputs=${EXP}"
