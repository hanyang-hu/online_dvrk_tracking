#!/bin/bash
find . -type f -name '*:Zone.Identifier' -delete

# Base options for sequential_tracking.py
TRACKING_OPTS="--rotation_parameterization MixAngle \
--searcher CMA-ES \
--tracking_visualization \
--downscale_factor 2 \
--use_low_res_mesh True \
--use_pts_loss False \
--filter_option Kalman \
--use_nvdiffrast \
--batch_size 50 \
--batch_iters 100 \
--use_bo_initializer \
--sample_number 2000 \
--online_iters 1 \
--cos_reparams True \
--use_contour_tip_net True \
--final_iters 100 \
--use_prev_joint_angles True \
--data_dir surgpose"

# Loop over bag_id from 0 to 7 as well as from 30 to 33 with both PSM1 and PSM3
for BAG_ID in {0..7} {30..33}; do
    echo "Processing bag $BAG_ID ..."
    
    BAG_NAME="$(printf '%06d' $BAG_ID)/PSM3"
    BAG_ID_str="${BAG_ID}_PSM3"

    # Run sequential tracking
    python scripts/sequential_tracking.py $TRACKING_OPTS --difficulty $BAG_NAME

    # Generate video for current bag
    python scripts/video_generator.py --bag_id $BAG_ID_str --iters_per_frame 1 --data_type surgpose

    # Remove tracking folder to save space
    rm -rf tracking

    BAG_NAME="$(printf '%06d' $BAG_ID)/PSM1"
    BAG_ID_str="${BAG_ID}_PSM1"

    # Run sequential tracking
    python scripts/sequential_tracking.py $TRACKING_OPTS --difficulty $BAG_NAME

    # Generate video for current bag
    python scripts/video_generator.py --bag_id $BAG_ID_str --iters_per_frame 1 --data_type surgpose

    # Remove tracking folder to save space
    rm -rf tracking
done

echo "All bags processed."
