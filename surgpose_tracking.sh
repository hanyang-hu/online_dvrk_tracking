#!/bin/bash
find . -type f -name '*:Zone.Identifier' -delete

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
--online_iters 3 \
--cos_reparams True \
--final_iters 50 \
--use_prev_joint_angles True \
--data_dir surgpose \
--use_gt_kpts True "

# Loop over bag_id 0 to 7 with both PSM1 and PSM3
for BAG_ID in {0..7}; do
    echo "Processing bag $BAG_ID ..."
    
    BAG_NAME="bag${BAG_ID}_PSM1"
    BAG_ID_str="${BAG_ID}_PSM1"

    # Run sequential tracking
    python scripts/sequential_tracking.py $TRACKING_OPTS --difficulty $BAG_NAME

    # Generate video for current bag
    python scripts/video_generator.py --bag_id $BAG_ID_str --iters_per_frame 3 --data_type surgpose

    # Remove tracking folder to save space
    rm -rf tracking

    BAG_NAME="bag${BAG_ID}_PSM3"
    BAG_ID_str="${BAG_ID}_PSM3"

    # Run sequential tracking
    python scripts/sequential_tracking.py $TRACKING_OPTS --difficulty $BAG_NAME

    # Generate video for current bag
    python scripts/video_generator.py --bag_id $BAG_ID_str --iters_per_frame 3 --data_type surgpose

    # Remove tracking folder to save space
    rm -rf tracking
done

echo "All bags processed."
