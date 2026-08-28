#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

UPSTREAM_QUEUE_PID="${UPSTREAM_QUEUE_PID:-2724159}"
UPSTREAM_RUN_ID="2026-08-19_084450_all_LN_VR_LPIPS_10ep"
UPSTREAM_STEP_TAG="step_021700"
UPSTREAM_CKPT="${ROOT_DIR}/runs/asym_libero_joint_2cam224_1e-4/${UPSTREAM_RUN_ID}/checkpoints/weights/${UPSTREAM_STEP_TAG}.pt"
QUEUE_LOG="${ROOT_DIR}/logs/2026-08-19_after_vr_eval_fm_Aall_train_eval.log"
PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/envs/fastwam/bin/python}"

export PYTHON_BIN
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0

exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date '+%F %T')] Waiting for upstream train+eval queue PID ${UPSTREAM_QUEUE_PID}"
while [[ -d "/proc/${UPSTREAM_QUEUE_PID}" ]]; do
  sleep 60
done

if [[ ! -s "${UPSTREAM_CKPT}" ]]; then
  echo "Upstream queue exited without its final raw checkpoint: ${UPSTREAM_CKPT}" >&2
  exit 1
fi

mapfile -t SUMMARIES < <(
  find "${ROOT_DIR}/evaluate_results/libero" -mindepth 2 -maxdepth 2 \
    -path "*/${UPSTREAM_RUN_ID}_${UPSTREAM_STEP_TAG}_raw_8gpu4worker_*/summary.json" \
    -printf '%T@ %p\n' | sort -nr | cut -d' ' -f2-
)
if (( ${#SUMMARIES[@]} == 0 )); then
  echo "Upstream queue exited without a completed raw evaluation summary" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${SUMMARIES[0]}" "${UPSTREAM_CKPT}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    summary = json.load(f)
if summary.get("ckpt") != sys.argv[2]:
    raise RuntimeError(f"summary checkpoint mismatch: {summary.get('ckpt')} != {sys.argv[2]}")
trials = sum(suite["total_trials"] for suite in summary["suite_stats"].values())
if trials != 2000:
    raise RuntimeError(f"upstream evaluation is incomplete: {trials} != 2000 trials")
print(f"verified_upstream_summary={sys.argv[1]}")
PY

echo "[$(date '+%F %T')] Upstream all-A VR+LPIPS train+eval completed; starting all-A FM train+eval"
bash scripts/train_eval_asym_fm.sh

echo "[$(date '+%F %T')] All-A FM train+eval queue completed"
