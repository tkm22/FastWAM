#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${1:-$(date +%Y-%m-%d_%H%M%S)_s4_Aall_XPred_VR_LPIPS_seed42_10ep}"
RUN_PARENT="asym_libero_joint_2cam224_1e-4"
RUN_DIR="${ROOT_DIR}/runs/${RUN_PARENT}/${RUN_ID}"
LOG_PATH="${ROOT_DIR}/logs/${RUN_ID}.log"
PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/envs/fastwam/bin/python}"
PROJECTION_ARTIFACT="${ROOT_DIR}/artifacts/2026-08-18_stride4_all222929.pt"
FINAL_STEP=21700

if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists: ${RUN_DIR}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -s "${PROJECTION_ARTIFACT}" ]]; then
  echo "Projection artifact is missing: ${PROJECTION_ARTIFACT}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_PATH}")"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHON_BIN
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"${PYTHON_BIN}" tools/validate_asym_fastwam_artifact.py "${PROJECTION_ARTIFACT}"

echo "[$(date '+%F %T')] Starting ${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "matched_change=prediction_type:x0"
echo "recipe=fresh 10ep seed42 A=all222929 logit_normal VR LPIPS train_eval=200"

RUN_TASK_BASENAME="${RUN_PARENT}" RUN_ID="${RUN_ID}" \
  bash scripts/train_zero1.sh 8 \
    task=libero_joint_2cam224_1e-4 \
    num_epochs=10 \
    eval_every=200 \
    eval_num_inference_steps=10 \
    seed=42 \
    "model.projection_artifact_path=${PROJECTION_ARTIFACT}" \
    model.asymflow.prediction_type=x0 \
    model.asymflow.timestep_sampling=logit_normal \
    model.asymflow.vr_enabled=true \
    model.asymflow.lpips_enabled=true \
    resume=null \
    wandb.enabled=true \
    wandb.workspace=minghao_workaholic \
    wandb.project=pixel-wam \
    wandb.name="${RUN_ID}"

bash scripts/eval_asym_vr_lpips_raw.sh \
  "${RUN_ID}" \
  "${FINAL_STEP}" \
  "${PROJECTION_ARTIFACT}" \
  true \
  true \
  x0
