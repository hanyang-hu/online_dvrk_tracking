#!/bin/bash
set -euo pipefail

# Usage:
#   ./custom_live_tracking.sh bag1 PSM3 false
#   ./custom_live_tracking.sh bag1 PSM3 true

VIDEO_LABEL="${1:-bag1}"
MACHINE_LABEL="${2:-PSM3}"
USE_LUMPED_ERROR_INIT="${3:-true}"

find . -type f -name '*:Zone.Identifier' -delete

python scripts/custom_traj_tracking.py \
  --video_label "$VIDEO_LABEL" \
  --machine_label "$MACHINE_LABEL" \
  --use_nvdiffrast \
  --rotation_parameterization MixAngle \
  --searcher CMA-ES \
  --downscale_factor 2 \
  --use_low_res_mesh True \
  --use_pts_loss True \
  --use_contour_tip_net True \
  --online_iters 3 \
  --use_prev_joint_angles False \
  --interactive_prompts True \
  --use_lumped_error_init "$USE_LUMPED_ERROR_INIT"
