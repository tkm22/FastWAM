#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

STAMP="${1:-$(date +%Y-%m-%d_%H%M%S)}"
FM_RUN_ID="${STAMP}_s4_Aall_XPred_noVR_noLPIPS_seed42_10ep"
VR_RUN_ID="${STAMP}_s4_Aall_XPred_VR_LPIPS_seed42_10ep"

echo "no_vr_run=${FM_RUN_ID}"
echo "vr_lpips_run=${VR_RUN_ID}"

bash scripts/train_eval_xpred_fm.sh "${FM_RUN_ID}"
bash scripts/train_eval_xpred_vr_lpips.sh "${VR_RUN_ID}"
