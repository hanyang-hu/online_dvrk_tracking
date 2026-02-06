# Conda environment setup

```
conda create --name bboxcali python=3.10
conda activate bboxcali
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install evotorch
pip install FastGeodis --no-build-isolation
pip install filterpy
```

You also need to install [NvDiffRast](https://nvlabs.github.io/nvdiffrast/) following the instructions below (for Linux):
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
pip install ninja
```

If necessary, to install [Deep Hough Transform](https://github.com/Hanqer/deep-hough-transform) and download the pretrained weights, run
```
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init
cd deep_hough_transform
wget http://data.kaizhao.net/projects/deep-hough-transform/dht_r50_nkl_d97b97138.pth
cd model/_cdht
python setup.py build 
python setup.py install --user
```

# Calibration with NvDiffRast + CMA-ES (instead of differentiable rendering)

For real-world data, run:
```
python scripts/sequential_tracing.py --sample_number 500 --final_iters 1000 --online_iters 10 --rotation_parameterization MixAngle --searcher CMA-ES --tracking_visualization --downscale_factor 1 --use_nvdiffrast --use_pts_loss True
```

For synthetic data (with ground-truth keypoints projections), run:
```
python scripts/synthetic_tracking.py --use_nvdiffrast --tracking_visualization --rotation_parameterization MixAngle --searcher CMA-ES --downscale_factor 1 --online_iters 5 --use_pts_loss True --use_opencv_kpts False
```

# Demo

<p float="left">
  <img src="./data/output_gd_iter30.gif" width="45%" />
  <img src="./data/output_iter10.gif" width="45%" />
</p>


