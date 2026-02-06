#!/bin/bash

# Base options for sequential_tracking.py
TRACKING_OPTS="--rotation_parameterization MixAngle \
--searcher CMA-ES \
--tracking_visualization \
--downscale_factor 2 \
--use_low_res_mesh True \
--use_pts_loss True \
--use_tip_emd_loss False \
--filter_option Kalman \
--use_nvdiffrast \
--use_bbox_optimizer \
--batch_size 50 \
--batch_iters 100 \
--use_bo_initializer \
--sample_number 1500 \
--online_iters 5 \
--final_iters 10 \
--use_prev_joint_angles True \
--data_dir synthetic_data \
--use_gt_kpts True"

# Loop over bag_id 1 to 10
for BAG_ID in {1..10}; do
    echo "Processing bag $BAG_ID ..."
    
    BAG_ID_str="${BAG_ID}"
    BAG_NAME="syn_new${BAG_ID}"

    # Run sequential tracking
    python scripts/sequential_tracking.py $TRACKING_OPTS --difficulty $BAG_NAME

    # Generate video for current bag
    python scripts/video_generator.py --bag_id $BAG_ID_str --iters_per_frame 5 --data_type synthetic

    # Remove tracking folder to save space
    rm -rf tracking
done

echo "All bags processed."
