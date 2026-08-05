#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${DVRK_CONDA_ENV:-online_dvrk}"
ROS_SETUP="${ROS_HUMBLE_SETUP:-/opt/ros/humble/setup.bash}"

if [[ -n "${CONDA_EXE:-}" ]]; then
  CONDA_BASE="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/miniconda3"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "Could not find Conda. Install Conda or run this from a shell where conda is available." >&2
  exit 1
fi

CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Could not find Conda activation script: $CONDA_SH" >&2
  exit 1
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "Could not find ROS Humble setup file: $ROS_SETUP" >&2
  echo "Set ROS_HUMBLE_SETUP=/path/to/setup.bash if ROS Humble is installed elsewhere." >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"
source "$ROS_SETUP"

cd "$REPO_ROOT"
exec python scripts/custom_live_gui.py "$@"
