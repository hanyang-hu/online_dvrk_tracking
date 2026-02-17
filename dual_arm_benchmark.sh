#!/bin/bash
set -e

find . -type f -name '*:Zone.Identifier' -delete

COMMON_ARGS="--rotation_parameterization MixAngle \
--tracking_visualization \
--downscale_factor 2 \
--use_low_res_mesh True \
--batch_size 50 \
--batch_iters 100 \
--use_bo_initializer \
--sample_number 2000 \
--final_iters 100"

REAL_BAGS=({0..7} {30..33})
SYN_BAGS=({0..15})
# REAL_BAGS=({0..0})
# SYN_BAGS=({0..0})

# ==========================================================
# FUNCTION TO RUN A SINGLE VARIANT (DUAL ARM)
# ==========================================================
run_variant () {

SEARCHER=$1
USE_JOINT=$2
ONLINE_ITERS=$3
DATA_DIR=$4
SEPARATE=$5
SOFT_SEP=$6
DIFF=$7

ARGS="--searcher $SEARCHER \
--online_iters $ONLINE_ITERS \
--data_dir $DATA_DIR \
--use_nvdiffrast"

# ---------------- Joint semantics ----------------
# ✓  -> use sensor  -> use_prev_joint_angles False
# ✗  -> no sensor   -> use_prev_joint_angles True
if [ "$USE_JOINT" == "True" ]; then
    ARGS="$ARGS --use_prev_joint_angles False"
else
    ARGS="$ARGS --use_prev_joint_angles True"
fi

# ---------------- Separation control ----------------
ARGS="$ARGS --separate_loss $SEPARATE --soft_separation $SOFT_SEP"

# ---------------- Gradient constraint ----------------
if [ "$SEARCHER" == "Gradient" ]; then
    ARGS="$ARGS --cos_reparams False"
fi

echo "--------------------------------------------------"
echo "DATA: $DATA_DIR | BAG: $DIFF"
echo "SEARCHER=$SEARCHER | joint=$USE_JOINT | sep=$SEPARATE | soft_sep=$SOFT_SEP"
echo "--------------------------------------------------"

python scripts/bimanual_tracking.py $COMMON_ARGS $ARGS --difficulty $DIFF

# rm -rf tracking
}

# ==========================================================
# ALL DUAL-ARM VARIANTS (FROM TABLE)
# ==========================================================
run_all_dual_arm () {

DATA_DIR=$1

# ===============================
# Gradient (×10)
# ===============================

# ---- Normal (joint loss) ----
run_variant Gradient False 10 $DATA_DIR False False $BAG
run_variant Gradient True  10 $DATA_DIR False False $BAG

# ---- Sep. (hard separation) ----
run_variant Gradient False 10 $DATA_DIR True False $BAG
run_variant Gradient True  10 $DATA_DIR True False $BAG

# ---- Sep. (soft separation) ----
run_variant Gradient False 10 $DATA_DIR True True $BAG
run_variant Gradient True  10 $DATA_DIR True True $BAG


# ===============================
# CMA-ES (×3)
# ===============================

# ---- Normal (joint loss) ----
run_variant CMA-ES False 3 $DATA_DIR False False $BAG
run_variant CMA-ES True  3 $DATA_DIR False False $BAG

# ---- Sep. (hard separation) ----
run_variant CMA-ES False 3 $DATA_DIR True False $BAG
run_variant CMA-ES True  3 $DATA_DIR True False $BAG

# ---- Sep. (soft separation) ----
run_variant CMA-ES False 3 $DATA_DIR True True $BAG
run_variant CMA-ES True  3 $DATA_DIR True True $BAG
}

# ==========================================================
# RUN ON SURGPOSE
# ==========================================================
for BAG_ID in ${REAL_BAGS[@]}; do
    BAG="$(printf '%06d' $BAG_ID)"
    run_all_dual_arm surgpose
done

# ==========================================================
# RUN ON SYNTHETIC
# ==========================================================
for BAG_ID in ${SYN_BAGS[@]}; do
    BAG="$(printf '%06d' $BAG_ID)"
    run_all_dual_arm synthetic
done

echo "Dual-arm benchmark complete."
