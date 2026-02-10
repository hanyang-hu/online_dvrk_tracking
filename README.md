# Conda environment setup

```
conda create --name online_dvrk python=3.10
conda activate online_dvrk
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install FastGeodis --no-build-isolation
```

You also need to install [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and [NvDiffRast](https://nvlabs.github.io/nvdiffrast/).

For PyTorch3D, try the following commands:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

For NvDiffRast, try the following commands:
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
pip install ninja
```

# Run benchmarking scripts

Download the data from https://drive.google.com/file/d/1DBHTH_w-w-WuLFSiENn2cYPDvKj9-2gk/view?usp=sharing and put it under the `./data` folder.

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
