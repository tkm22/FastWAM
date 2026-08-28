#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CURRENT_RUN_ID="2026-08-18_205228_100k_LN_VR_LPIPS_cont20ep"
CURRENT_FINAL_STEP=43400
CURRENT_TRAIN_PID="${CURRENT_TRAIN_PID:-1027225}"
CURRENT_A="${ROOT_DIR}/artifacts/2026-08-04_stride4_bal100k.pt"
CURRENT_CKPT="${ROOT_DIR}/runs/asym_libero_joint_2cam224_1e-4/${CURRENT_RUN_ID}/checkpoints/weights/step_043400.pt"
QUEUE_LOG="${ROOT_DIR}/logs/2026-08-19_eval_then_fullA_train_eval.log"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0

exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date '+%F %T')] Waiting for current training PID ${CURRENT_TRAIN_PID}"
while [[ -d "/proc/${CURRENT_TRAIN_PID}" ]]; do
  sleep 60
done

if [[ ! -s "${CURRENT_CKPT}" ]]; then
  echo "Current training exited without final raw checkpoint: ${CURRENT_CKPT}" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Current training completed; starting raw evaluation"
bash scripts/eval_asym_vr_lpips_raw.sh \
  "${CURRENT_RUN_ID}" \
  "${CURRENT_FINAL_STEP}" \
  "${CURRENT_A}"

echo "[$(date '+%F %T')] Current evaluation completed; starting fresh all-A training"
bash scripts/train_eval_asym_vr_lpips.sh

echo "[$(date '+%F %T')] Full queue completed"
