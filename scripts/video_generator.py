import cv2
import os
import glob
import re
from PIL import Image


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate annotated GIF from overlay images.')
    parser.add_argument('--bag_id', type=str, default='14', help='Bag ID for the title text.')
    parser.add_argument('--iters_per_frame', type=int, default=10, help='Iterations per frame for the title text.')
    parser.add_argument('--data_type', type=str, default='synthetic', choices=['surgpose', 'synthetic'], help='Type of data: surgpose or synthetic.')
    args = parser.parse_args()

    # === Configuration ===
    image_folder = './tracking'           # Folder containing overlay_*.png
    output_gif = f'./GIFs/{args.data_type}_bag{args.bag_id}.gif'           # Output file
    title_lines = [
        # 'Separated CMA-ES',
        'CMA-ES',
        # 'Bi-manual Tracking',
        f'{args.iters_per_frame} iters/frame',
        f'{args.data_type.capitalize()}: Bag {args.bag_id}',
        # 'Less pts loss weight',
        # 'No distance loss',
        # 'Symmetric jaw angles',
        # 'Larger jaw variance',
        '2D Keypoint Detection',
        # 'No Filter',
        # 'Convolutional Distance Transform',
        # 'One Euro Filter',
        # 'Modified One Euro Filter',
        'Kalman Filter',
        # 'Cosine Joint Angle Transform',
        'with joint angle readings',
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    start_pos = (10, 30)  # x, y position of first line
    line_spacing = 35     # Pixels between lines
    fps = 20              # GIF frame rate

    # === Helper: Sort numerically ===
    def numerical_sort(path):
        match = re.search(r'(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1

    # === Load images ===
    image_paths = sorted(glob.glob(os.path.join(image_folder, 'overlay_*.png')), key=numerical_sort)
    if not image_paths:
        raise RuntimeError("No overlay_*.png files found in the folder.")

    gif_frames = []

    # === Process and annotate images ===
    for img_path in image_paths:
        frame = cv2.imread(img_path)

        # Draw each line of title text
        for i, line in enumerate(title_lines):
            pos = (start_pos[0], start_pos[1] + i * line_spacing)
            cv2.putText(frame, line, pos, font, font_scale, font_color, thickness, cv2.LINE_AA)

        # Convert BGR to RGB and store frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(rgb_frame))

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    # === Save as GIF ===
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

    print(f"GIF saved to: {output_gif}")


"""
python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw15"
python scripts/video_generator.py --bag_id 15 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw14"  
python scripts/video_generator.py --bag_id 14 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw9"  
python scripts/video_generator.py --bag_id 9 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw6"  
python scripts/video_generator.py --bag_id 6 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw5"  
python scripts/video_generator.py --bag_id 5 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw1"  
python scripts/video_generator.py --bag_id 1 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw8"  
python scripts/video_generator.py --bag_id 8 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw7"  
python scripts/video_generator.py --bag_id 7 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 300 --use_prev_joint_angles True --difficulty "rw11"  
python scripts/video_generator.py --bag_id 11 --iters_per_frame 5
rm -r tracking

"""

"""
Synthetic data

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw15"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 15 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw14"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 14 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw9"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 9 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw8"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 8 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw7"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 7 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw6"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 6 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw5"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 5 --iters_per_frame 5
rm -r tracking

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_rw1"   --data_dir synthetic_data
python scripts/video_generator.py --bag_id 1 --iters_per_frame 5
rm -r tracking
"""

"""

# Bi-manual SurgPose data
python scripts/bimanual_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --filter_option Kalman --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 3 --final_iters 100 --use_prev_joint_angles True --difficulty "000000" --data_dir surgpose --soft_separation False
python scripts/video_generator.py --bag_id 1 --iters_per_frame 3 --data_type surgpose
rm -r tracking

python scripts/bimanual_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --filter_option Kalman --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 3 --final_iters 100 --use_prev_joint_angles True --difficulty "000000" --data_dir synthetic --soft_separation False
python scripts/video_generator.py --bag_id 1 --iters_per_frame 3 --data_type surgpose
rm -r tracking
"""

"""
Pure synthetic data without real background

python scripts/sequential_tracking.py --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 2 --use_low_res_mesh True --use_pts_loss True --use_tip_emd_loss False --use_filter True --use_nvdiffrast --use_bbox_optimizer --batch_size 50 --batch_iters 100 --use_bo_initializer --sample_number 1500 --online_iters 5 --final_iters 10 --use_prev_joint_angles True --difficulty "syn_new1" --data_dir synthetic_data
python scripts/video_generator.py --bag_id 1 --iters_per_frame 5
rm -r tracking


"""
