#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

USAGE="Usage: bash scripts/eval_asym_vr_lpips_raw.sh <run_id> <final_step> <projection_artifact> [vr_enabled] [lpips_enabled] [prediction_type]"
RUN_ID="${1:?${USAGE}}"
FINAL_STEP="${2:?${USAGE}}"
PROJECTION_ARTIFACT="${3:?${USAGE}}"
VR_ENABLED="${4:-true}"
LPIPS_ENABLED="${5:-true}"
PREDICTION_TYPE="${6:-asym_velocity}"
RUN_PARENT="asym_libero_joint_2cam224_1e-4"
RUN_DIR="${ROOT_DIR}/runs/${RUN_PARENT}/${RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/envs/fastwam/bin/python}"

if [[ ! "${FINAL_STEP}" =~ ^[0-9]+$ ]]; then
  echo "final_step must be an integer, got: ${FINAL_STEP}" >&2
  exit 1
fi
if [[ ! "${VR_ENABLED}" =~ ^(true|false)$ ]]; then
  echo "vr_enabled must be true or false, got: ${VR_ENABLED}" >&2
  exit 1
fi
if [[ ! "${LPIPS_ENABLED}" =~ ^(true|false)$ ]]; then
  echo "lpips_enabled must be true or false, got: ${LPIPS_ENABLED}" >&2
  exit 1
fi
if [[ "${VR_ENABLED}" == false && "${LPIPS_ENABLED}" == true ]]; then
  echo "lpips_enabled=true requires vr_enabled=true" >&2
  exit 1
fi
if [[ ! "${PREDICTION_TYPE}" =~ ^(asym_velocity|x0)$ ]]; then
  echo "prediction_type must be asym_velocity or x0, got: ${PREDICTION_TYPE}" >&2
  exit 1
fi
printf -v STEP_TAG 'step_%06d' "${FINAL_STEP}"
RAW_CKPT="${RUN_DIR}/checkpoints/weights/${STEP_TAG}.pt"

if [[ ! -s "${RAW_CKPT}" ]]; then
  echo "Final raw checkpoint is missing or empty: ${RAW_CKPT}" >&2
  exit 1
fi
if [[ ! -s "${RUN_DIR}/dataset_stats.json" ]]; then
  echo "Dataset statistics are missing: ${RUN_DIR}/dataset_stats.json" >&2
  exit 1
fi
if [[ ! -s "${PROJECTION_ARTIFACT}" ]]; then
  echo "Projection artifact is missing: ${PROJECTION_ARTIFACT}" >&2
  exit 1
fi

export PYTHON_BIN
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"${PYTHON_BIN}" tools/validate_asym_fastwam_artifact.py "${PROJECTION_ARTIFACT}"

echo "[$(date '+%F %T')] Starting four-model raw 400-episode video matrix"
echo "target_checkpoint=${RAW_CKPT}"
echo "projection_artifact=${PROJECTION_ARTIFACT}"
echo "target_prediction_type=${PREDICTION_TYPE} target_vr_enabled=${VR_ENABLED} target_lpips_enabled=${LPIPS_ENABLED}"

bash scripts/eval_sampler_solver_400video_4models.sh
