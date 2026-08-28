#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/envs/fastwam/bin/python}"
A_OUTPUT="${ROOT_DIR}/artifacts/2026-08-18_stride4_all222929.pt"
A_LOG="${ROOT_DIR}/logs/2026-08-18_stride4_all222929.log"
SOURCE_RUN="${ROOT_DIR}/runs/asym_libero_joint_2cam224_1e-4/2026-08-18_035500_100k_LN_VR_LPIPS_10ep"
SOURCE_STATE="${SOURCE_RUN}/checkpoints/state/step_021700"
SOURCE_A="artifacts/2026-08-04_stride4_bal100k.pt"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H%M%S)_vr_lpips_20ep}"
TRAIN_LOG="${ROOT_DIR}/logs/${RUN_ID}.log"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [[ -e "${A_OUTPUT}" ]]; then
  echo "A output already exists: ${A_OUTPUT}" >&2
  exit 1
fi
if [[ ! -s "${SOURCE_STATE}/trainer_state.json" ]]; then
  echo "Resume state is missing: ${SOURCE_STATE}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"

echo "[$(date '+%F %T')] Fitting A from all 222929 valid clips" | tee -a "${A_LOG}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/fit_asym_fastwam_procrustes.py \
  --dataset-roots \
    data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_object_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
    data/libero_mujoco3.3.2/libero_10_no_noops_lerobot \
  --output "${A_OUTPUT}" \
  --seed 0 \
  --vae-batch 8 \
  2>&1 | tee -a "${A_LOG}"

"${PYTHON_BIN}" tools/validate_asym_fastwam_artifact.py "${A_OUTPUT}"
"${PYTHON_BIN}" - "${A_OUTPUT}" <<'PY'
import sys

from asymflow.projection import ProjectionArtifact

artifact = ProjectionArtifact.load(sys.argv[1])
if artifact.metadata.get("total_clips") != 222_929:
    raise RuntimeError(f"Unexpected total_clips: {artifact.metadata}")
if "no endpoint replication" not in artifact.metadata.get("sampling", ""):
    raise RuntimeError(f"Unexpected sampling metadata: {artifact.metadata}")
PY

echo "[$(date '+%F %T')] A fit validated; resuming VR+LPIPS as ${RUN_ID}" | tee -a "${TRAIN_LOG}"
RUN_TASK_BASENAME=asym_libero_joint_2cam224_1e-4 RUN_ID="${RUN_ID}" \
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/train_zero1.sh 8 \
    task=libero_joint_2cam224_1e-4 \
    num_epochs=20 \
    "resume=${SOURCE_STATE}" \
    "model.projection_artifact_path=${SOURCE_A}" \
    model.asymflow.timestep_sampling=logit_normal \
    model.asymflow.vr_enabled=true \
    model.asymflow.lpips_enabled=true \
    wandb.enabled=true \
    wandb.workspace=minghao_workaholic \
    wandb.project=pixel-wam \
    "wandb.name=${RUN_ID}" \
    2>&1 | tee -a "${TRAIN_LOG}"
