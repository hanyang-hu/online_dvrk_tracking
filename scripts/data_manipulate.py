import os
from PIL import Image

# Base paths
rw_base = "./data/consecutive_prediction/"  # parent folder containing rw1, rw5, rw6, ...
real_base = "./data/consecutive_prediction/real_world_data"  # folder containing bag1, bag5, ...

# Loop over rw folders
for rw_folder in ["rw1", "rw5", "rw6", "rw7", "rw8", "rw9", "rw10", "rw11", "rw12", "rw14", "rw15"]:
    rw_path = os.path.join(rw_base, rw_folder)
    bag_id = rw_folder.replace("rw", "bag")
    bag_frames = os.path.join(real_base, bag_id, "frames")

    if not os.path.exists(bag_frames):
        print(f"⚠️ Skipping {rw_folder}: no {bag_frames}")
        continue

    # Get sorted list of frames (00000.jpg, 00001.jpg, ...)
    jpg_files = sorted([f for f in os.listdir(bag_frames) if f.endswith(".jpg")])
    subfolders = sorted([f for f in os.listdir(rw_path) if f.isdigit()])

    for sub in subfolders:
        sub_path = os.path.join(rw_path, sub)
        frame_idx = int(sub)  # assume folder name corresponds to frame index

        # Make sure frame exists
        if frame_idx >= len(jpg_files):
            print(f"⚠️ No frame {frame_idx:05d} for {rw_folder}/{sub}")
            continue

        frame_name = jpg_files[frame_idx]
        src_path = os.path.join(bag_frames, frame_name)
        dst_name = f"frame_{frame_name}"
        dst_path = os.path.join(sub_path, dst_name)

        # Get target size from corresponding PNG
        png_files = [f for f in os.listdir(sub_path) if f.endswith(".png")]
        if not png_files:
            print(f"⚠️ No PNG in {sub_path}")
            continue

        ref_img_path = os.path.join(sub_path, png_files[0])
        with Image.open(ref_img_path) as ref_img:
            target_size = ref_img.size  # (W, H)

        # Resize and save
        with Image.open(src_path) as img:
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(dst_path, "JPEG")

        print(f"✅ Saved {dst_name} to {rw_folder}/{sub}")

print("🎉 Done.")
