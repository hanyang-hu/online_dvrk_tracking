# Environment setup

This code has been tested on **Ubuntu 22.04** and **Ubuntu 24.04 (WSL 2)** with **CUDA 12.6**.

Please ensure that your CUDA version matches 12.6 before proceeding. If **CUDA 12.6** is installed in `/usr/local/cuda-12.6`, you can explicitly set it with:
```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

To setup the conda environment, run:
```bash
conda create --name online_dvrk python=3.10
conda activate online_dvrk

# Install PyTorch (CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install project dependencies
pip install -r requirements.txt
pip install FastGeodis --no-build-isolation
```

You will also need to install [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and [NvDiffRast](https://nvlabs.github.io/nvdiffrast/).  
If you encounter build issues, try adding the `--no-build-isolation` flag.

### Install PyTorch3D
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Install NvDiffRast
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

### Surgical SAM 2

Download the pretrained weights from [sam2.1_hiera_s_endo18](https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=drive_link), and place it under `./SurgicalSAM2/checkpoints`.

# Run benchmarking scripts

### Synthetic and Real-world (SurgPose) Datasets

Download the data from https://drive.google.com/file/d/1EKDdBhwoUJQ-o0qPJaMteezpD79vwvtJ/view?usp=sharing and place it under the `./data` folder.

Run the following commands to benchmark on the synthetic and real-world datasets:
```
bash ./single_arm_benchmark.sh
bash ./dual_arm_benchmark.sh

python ./scripts/single_arm_quantitative_results.py --evaluate_surgpose
python ./scripts/dual_arm_quantitative_results.py --evaluate_surgpose
```

### Our Dataset

**Note.** This dataset includes 6 trajectories of LND with painted markers to benchmark our method with the particle filters.

# Calibrate online videos

## Step 1: Prepare the input video

Place the video at:

```
data/online_videos/<video_id>/video.mp4
```

Example:

```
data/online_videos/000000/video.mp4
```

---

## Step 2: Annotate the first frame (interactive)

Run the video annotator to initialize keypoints and SAM prompts:

```bash
python scripts/video_annotator.py \
    --idx 000000 \
    --machine_label PSM3
```

**Annotation controls**

- Left click: tool keypoint  
- SHIFT + left click: foreground SAM prompt  
- CTRL + left click: background SAM prompt  
- ENTER: save and continue  
- `r`: reset annotations  
- `q` / `ESC`: quit  

---

## Step 3: Run video calibration

After annotation, start video calibration:

```bash
python scripts/video_calibration.py \
    --sample_number 1500 \
    --use_nvdiffrast \
    --use_bo_initializer \
    --video_label 000000 \
    --machine_label PSM3 \
    --searcher CMA-ES
```

**Note.** Make sure to modify `parseCtRNetArgs()` so that the image shape and camera intrinsics are consistent with your input data. 
