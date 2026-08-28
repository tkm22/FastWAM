#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/envs/fastwam/bin/python}"
JIT_ROOT="/raid/kaiming/FastWAM-jit"
RUN_PARENT="${ROOT_DIR}/runs/asym_libero_joint_2cam224_1e-4"
LATENT_RUN="${ROOT_DIR}/runs/libero_joint_2cam224_1e-4/2026-07-27_2017"
PIXEL_ASYM_RUN="${RUN_PARENT}/2026-08-19_190513_all_LN_10ep"
PIXEL_ASYM_VR_LPIPS_RUN="${RUN_PARENT}/2026-08-19_084450_all_LN_VR_LPIPS_10ep"
XPRED_NOVR_RUN="${RUN_PARENT}/2026-08-27_115123_s4_Aall_XPred_noVR_noLPIPS_seed42_10ep"
PROJECTION_ARTIFACT="${ROOT_DIR}/artifacts/2026-08-18_stride4_all222929.pt"
OUTPUT_ROOT="${ROOT_DIR}/evaluate_results/sampler_solver_400video_4models_seed42"
LOCK_PATH="${ROOT_DIR}/evaluate_results/.sampler_solver_400video_4models_seed42.lock"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
flock 9

run_latent() {
  "${PYTHON_BIN}" -u scripts/run_sampler_solver_400_video.py \
    --repository-root "${JIT_ROOT}" \
    --model-kind latent \
    --run-dir "${LATENT_RUN}" \
    --final-step 21700 \
    --model-name latent_fastwam_joint_step021700_raw \
    --output-dir "${OUTPUT_ROOT}/latent_fastwam_joint_step021700_raw" \
    --num-gpus 8 \
    --workers-per-gpu 1
}

run_pixel() {
  local model_name="$1"
  local run_dir="$2"
  local vr_enabled="$3"
  local lpips_enabled="$4"
  local prediction_type="$5"
  local final_step="$6"
  "${PYTHON_BIN}" -u scripts/run_sampler_solver_400_video.py \
    --repository-root "${ROOT_DIR}" \
    --model-kind pixel \
    --run-dir "${run_dir}" \
    --final-step "${final_step}" \
    --projection-artifact "${PROJECTION_ARTIFACT}" \
    --prediction-type "${prediction_type}" \
    --vr-enabled "${vr_enabled}" \
    --lpips-enabled "${lpips_enabled}" \
    --model-name "${model_name}" \
    --output-dir "${OUTPUT_ROOT}/${model_name}" \
    --num-gpus 8 \
    --workers-per-gpu 1
}

echo "[$(date '+%F %T')] starting four-model 400-episode video matrix"
echo "output_root=${OUTPUT_ROOT}"

run_latent
run_pixel \
  pixel_asym_aall_no_vr_step021700_raw \
  "${PIXEL_ASYM_RUN}" \
  false false asym_velocity 21700
run_pixel \
  pixel_asym_aall_vr_lpips_step021700_raw \
  "${PIXEL_ASYM_VR_LPIPS_RUN}" \
  true true asym_velocity 21700
run_pixel \
  pixel_xpred_aall_no_vr_step021700_raw \
  "${XPRED_NOVR_RUN}" \
  false false x0 21700

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summaries = sorted(root.glob("*/matrix_summary.json"))
if len(summaries) != 4:
    raise RuntimeError(f"Expected four model summaries, found {summaries}")
models = []
rows = []
for path in summaries:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("completed_cells") != payload.get("expected_cells") or payload.get("completed_cells") != 6:
        raise RuntimeError(f"Incomplete model matrix: {path}")
    if payload.get("completed_episodes") != 2400:
        raise RuntimeError(f"Wrong episode count: {path}")
    models.append(path.parent.name)
    rows.extend(payload["rows"])
combined = {
    "models": models,
    "models_count": len(models),
    "cells_per_model": 6,
    "episodes_per_cell": 400,
    "episodes_per_model": 2400,
    "completed_cells": len(rows),
    "completed_episodes": sum(row["episodes"] for row in rows),
    "rows": rows,
}
if combined["completed_cells"] != 24 or combined["completed_episodes"] != 9600:
    raise RuntimeError(f"Combined matrix mismatch: {combined}")
(root / "all_models_summary.json").write_text(
    json.dumps(combined, indent=2), encoding="utf-8"
)
print(f"completed four-model matrix: {root}")
PY

echo "[$(date '+%F %T')] four-model 400-episode video matrix completed"
