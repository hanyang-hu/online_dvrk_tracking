#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate online_dvrk
source /opt/ros/humble/setup.bash

cd "$REPO_ROOT"
exec python scripts/custom_live_gui.py "$@"
