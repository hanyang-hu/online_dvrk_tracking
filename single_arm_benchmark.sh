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
ARMS=("PSM1" "PSM3")

# run_variant () {

# SEARCHER=$1
# RENDERER=$2
# USE_JOINT=$3
# USE_APP=$4
# KPTS_MODE=$5
# ONLINE_ITERS=$6
# DATA_DIR=$7
# DIFF=$8

# ARGS="--searcher $SEARCHER \
# --online_iters $ONLINE_ITERS \
# --data_dir $DATA_DIR"

# # ---------------- Renderer ----------------
# if [ "$RENDERER" == "nvdiffrast" ]; then
#     ARGS="$ARGS --use_nvdiffrast"
# fi

# # ---------------- Appearance ----------------
# if [ "$USE_APP" == "False" ]; then
#     ARGS="$ARGS --app_weight 0.0"
# fi

# # ---------------- Keypoints ----------------
# if [ "$KPTS_MODE" == "Learned" ]; then
#     ARGS="$ARGS --use_pts_loss True --use_contour_tip_net True"
# elif [ "$KPTS_MODE" == "OpenCV" ]; then
#     ARGS="$ARGS --use_pts_loss True --use_contour_tip_net False"
# elif [ "$KPTS_MODE" == "None" ]; then
#     ARGS="$ARGS --use_pts_loss False"
# fi

# # ---------------- Gradient constraint ----------------
# if [ "$SEARCHER" == "Gradient" ]; then
#     ARGS="$ARGS --cos_reparams False"
# fi

# # ---------------- Joint logic (TABLE SEMANTICS) ----------------
# # USE_JOINT=True  means ✓ in table (use sensor joints)
# # -> so use_prev_joint_angles must be False
# if [ "$USE_JOINT" == "True" ]; then
#     ARGS="$ARGS --use_prev_joint_angles False"
# else
#     ARGS="$ARGS --use_prev_joint_angles True"
# fi


# # ==========================================================
# # 🔍 DRY RUN (PRINT ONLY)
# # ==========================================================

# echo "--------------------------------------------------"
# echo "DATA: $DATA_DIR | BAG: $DIFF"
# echo "CMD:"
# echo "python scripts/sequential_tracking.py $COMMON_ARGS $ARGS --difficulty $DIFF"
# echo "--------------------------------------------------"
# echo ""
# }

# ==========================================================
# FUNCTION TO RUN A SINGLE VARIANT
# ==========================================================
run_variant () {

SEARCHER=$1
RENDERER=$2
USE_JOINT=$3
USE_APP=$4
KPTS_MODE=$5
ONLINE_ITERS=$6
DATA_DIR=$7
DIFF=$8

ARGS="--searcher $SEARCHER \
--online_iters $ONLINE_ITERS \
--data_dir $DATA_DIR"

# ---------------- Renderer ----------------
if [ "$RENDERER" == "nvdiffrast" ]; then
    ARGS="$ARGS --use_nvdiffrast"
fi

# ---------------- Appearance ----------------
if [ "$USE_APP" == "False" ]; then
    ARGS="$ARGS --app_weight 0.0"
fi

# ---------------- Keypoints ----------------
if [ "$KPTS_MODE" == "Learned" ]; then
    ARGS="$ARGS --use_pts_loss True --use_contour_tip_net True"
elif [ "$KPTS_MODE" == "OpenCV" ]; then
    ARGS="$ARGS --use_pts_loss True --use_contour_tip_net False"
elif [ "$KPTS_MODE" == "None" ]; then
    ARGS="$ARGS --use_pts_loss False"
fi

# ---------------- Gradient constraint ----------------
if [ "$SEARCHER" == "Gradient" ]; then
    ARGS="$ARGS --cos_reparams False"
fi

# ---------------- Joint logic (TABLE SEMANTICS) ----------------
# USE_JOINT=True  means ✓ in table (use sensor joints)
# -> so use_prev_joint_angles must be False
if [ "$USE_JOINT" == "True" ]; then
    ARGS="$ARGS --use_prev_joint_angles False"
else
    ARGS="$ARGS --use_prev_joint_angles True"
fi

echo "--------------------------------------------------"
echo "DATA: $DATA_DIR | BAG: $DIFF"
echo "CMD:"
echo "python scripts/sequential_tracking.py $COMMON_ARGS $ARGS --difficulty $DIFF"
echo "--------------------------------------------------"
echo ""

python scripts/sequential_tracking.py $COMMON_ARGS $ARGS --difficulty $DIFF
# rm -rf tracking
}

# ==========================================================
# SINGLE ARM VARIANTS FROM TABLE
# ==========================================================

run_all_single_arm () {

DATA_DIR=$1
ONLINE_GD_10=$2
ONLINE_GD_20=$3
ONLINE_ES_3=$4
ONLINE_ES_5=$5

# -------- GD (NvDiffRast) ----------
run_variant Gradient nvdiffrast False True Learned $ONLINE_GD_10 $DATA_DIR $BAG
run_variant Gradient nvdiffrast False True Learned $ONLINE_GD_20 $DATA_DIR $BAG

# -------- GD (PyTorch3D) ----------
run_variant Gradient pytorch3d False True Learned $ONLINE_GD_10 $DATA_DIR $BAG

# -------- XNES ----------
run_variant XNES nvdiffrast False True Learned $ONLINE_ES_3 $DATA_DIR $BAG

# -------- CMA-ES ----------
run_variant CMA-ES nvdiffrast False True Learned $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast True  True Learned 1 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast True  True Learned $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast False False Learned $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast False True None     $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast False True OpenCV   $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast False True Learned  $ONLINE_ES_3 $DATA_DIR $BAG
run_variant CMA-ES nvdiffrast False True Learned  $ONLINE_ES_5 $DATA_DIR $BAG
}

# ==========================================================
# LOOP DATASETS
# ==========================================================

for BAG_ID in ${REAL_BAGS[@]}; do
for ARM in ${ARMS[@]}; do
    BAG="$(printf '%06d' $BAG_ID)/$ARM"
    run_all_single_arm surgpose 10 20 3 5
done
done

for BAG_ID in ${SYN_BAGS[@]}; do
for ARM in ${ARMS[@]}; do
    BAG="$(printf '%06d' $BAG_ID)/$ARM"
    run_all_single_arm synthetic 10 20 3 5
done
done

echo "Single-arm benchmark complete."
