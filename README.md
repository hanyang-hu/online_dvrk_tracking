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

Download the data from https://drive.google.com/file/d/1PStrCA-Btru2URMU-hTThOaPMVJ679uw/view?usp=drive_link and put it under the `./data` folder.

**Note.** The benchmarking script is not yet ready. To verify that the environment is set up correctly, you can run:
```
bash surgpose_tracking.sh
```

### Grasp Dataset (SuPer)

To compare online tool tracking (with joint angle readings) with the particle filters proposed by [Richter et al.](https://arxiv.org/abs/2102.06235), download the dataset in ROSBag format and place under `./data/super`.

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

## Step 3: Run online tracking

After annotation, start online tracking:

```bash
python scripts/online_tracking.py \
    --sample_number 1500 \
    --use_nvdiffrast \
    --use_bo_initializer \
    --video_label 000000 \
    --machine_label PSM3 \
    --searcher CMA-ES
```

**Note.** Make sure to modify `parseCtRNetArgs()` so that the image shape and camera intrinsics are consistent with your input data. 
