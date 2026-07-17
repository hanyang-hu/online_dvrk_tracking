# dVRK Live Tracking GUI

## Overview

Implementation of the paper [Real-time Rendering-based Surgical Instrument Tracking via
Evolutionary Optimization](https://arxiv.org/pdf/2603.11404).

This branch focuses on the live dVRK tracking GUI. The GUI runs Surgical SAM2 segmentation, dVRK rendering, and online tracking in a Conda Python 3.10 environment, and it can consume samples from offline files, mock-live replay, or a ROS 2 bridge.

## Architecture

Two Python environments are used:

```text
Conda Python 3.10:
    GUI, CUDA, SAM2, rendering, tracking

System Python 3.12:
    ROS 2 Jazzy bridge
```

The tracking process never imports `rclpy`. ROS 2 communication is handled by a separate bridge process:

```text
System Python 3.12 ROS 2 bridge
        <-> ZeroMQ over localhost
Conda Python 3.10 Qt/CUDA tracking GUI
```

## Tracking environment setup

The tracking GUI environment has been tested on **Ubuntu 22.04** and **Ubuntu 24.04 (WSL 2)** with **CUDA 12.6**.

Before installing Python packages, confirm that the CUDA toolkit used by the environment matches CUDA 12.6. If CUDA 12.6 is installed in `/usr/local/cuda-12.6`, set:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Create and activate the Conda environment:

```bash
conda create --name online_dvrk python=3.10
conda activate online_dvrk
```

Install PyTorch for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install the project dependencies. `requirements.txt` includes the GUI dependencies and the ZeroMQ transport packages `pyzmq` and `msgpack`:

```bash
pip install -r requirements.txt
pip install FastGeodis --no-build-isolation
```

Install PyTorch3D:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Install NvDiffRast:

```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
cd ..
```

If PyTorch3D, NvDiffRast, or FastGeodis build fails, retry with `--no-build-isolation`.

Download the Surgical SAM 2 pretrained weights from [sam2.1_hiera_s_endo18](https://drive.google.com/file/d/1DyrrLKst1ZQwkgKM7BWCCwLxSXAgOcMI/view?usp=drive_link), and place the checkpoint at:

```text
SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth
```

After setup, start the GUI from the repository root:

```bash
conda activate online_dvrk
python scripts/custom_live_gui.py
```

## Required models and calibration files

The GUI needs:

- A mp4 video and joint-angle YAML for offline or mock-live mode.
- Camera calibration YAML.
- Hand-eye calibration YAML.
- `LND.json` robot kinematic description.
- LND mesh files under `urdfs/dVRK/meshes`.
- Surgical SAM 2 checkpoint at `SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth`.
- Keypoint detection model weights when point loss is enabled.

## Launching the GUI

Run the GUI from the repository root:

```bash
conda activate online_dvrk
python scripts/custom_live_gui.py
```

Select an input source, configure the relevant paths or endpoints, then click **Load Initialization Frame** before adding prompts.

## Offline mode

Offline mode pairs frame `i` from the selected video with entry `"i"` from the joint-angle YAML. It stops at the shorter of the video and YAML sequences.

Use this mode for deterministic playback from files.

## Mock-live mode

Mock-live mode uses the same video and joint-angle YAML as offline mode, but replays samples according to wall-clock time. If tracking falls behind, stale samples are skipped so latency does not grow continuously.

Use **Replay rate** to choose the simulated stream rate. Enable **Loop** to replay the sequence repeatedly.

## ROS 2 bridge mode

ROS 2 bridge mode receives synchronized image and joint samples from `scripts/ros2_tracking_bridge.py` over ZeroMQ.

Default endpoints:

```text
Input endpoint:  tcp://127.0.0.1:5555
Result endpoint: tcp://127.0.0.1:5556
```

The GUI receives tracking samples from the input endpoint and sends overlay, pose, optimized joints, loss, and FPS results to the result endpoint.

## Installing ROS bridge dependencies

Install ROS-side dependencies in the system ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash

sudo apt install -y \
    python3-zmq \
    python3-msgpack \
    ros-jazzy-cv-bridge \
    ros-jazzy-message-filters
```

Do not install ROS packages into the Conda tracking environment.

## Testing without a dVRK

Use the fake ROS 2 publisher to replay a video and joint YAML as ROS topics, then run the bridge and GUI against those topics.

Verify ROS topics with:

```bash
ros2 topic list
ros2 topic hz --qos-profile sensor_data /stereo/left/image
ros2 topic echo --once /dvrk/PSM3/state_joint_current
ros2 topic echo --once /dvrk/PSM3/state_jaw_current
```

## Running the fake ROS 2 publisher

In a system Python 3.12 ROS shell:

```bash
source /opt/ros/jazzy/setup.bash

python3 scripts/mock_dvrk_ros2_publisher.py \
    --video data/custom/bag1/left.mp4 \
    --joint-angles data/custom/bag1/joint_angles.yaml \
    --rate 30 \
    --loop
```

## Running the ROS 2 bridge

In a second system Python 3.12 ROS shell:

```bash
source /opt/ros/jazzy/setup.bash

python3 scripts/ros2_tracking_bridge.py \
    --image-topic /stereo/left/image \
    --joint-topic /dvrk/PSM3/state_joint_current \
    --jaw-topic /dvrk/PSM3/state_jaw_current
```

## Starting the GUI with ROS input

In the Conda tracking environment:

```text
1. Activate the online_dvrk Conda environment.
2. Start scripts/custom_live_gui.py.
3. Select ROS 2 bridge.
4. Confirm the endpoints are:
   tcp://127.0.0.1:5555
   tcp://127.0.0.1:5556
5. Load the initialization frame.
6. Add foreground and background prompts.
7. Start tracking.
```

## GUI controls

- **Mode** selects Offline, Mock live, or ROS 2 bridge input.
- **Video** and **Joint angles yaml** are used by Offline and Mock live modes.
- **Camera calibration yaml**, **Handeye yaml**, and **LND json** are required in every mode.
- **Renderer** chooses `nvdiffrast` or `pytorch3d`.
- **Optimizer** chooses `CMA-ES`, `XNES`, or `Gradient`.
- **Downscale factor** trades resolution for speed.
- **Use low-res mesh** enables lightweight dVRK meshes.
- **Use point loss** enables ContourTipNet point loss.
- **Iterations/frame** and **Lumped error** can be adjusted while tracking is running.

Prompt controls:

- Left click: foreground instrument prompt.
- Right click: background prompt.
- **Clear Prompts** removes all current prompts.

## Pause, continue, and reinitialization

Click **Stop (Pause)** to pause tracking at the current sample.

Click **Continue** to resume with the current tracker state.

Click **Re-init** while paused to clear prompts, relabel the paused frame, and continue from that same paused image and joint state.

## ROS input topics

Default input topics consumed by the bridge:

```text
/stereo/left/image
/dvrk/PSM3/state_joint_current
/dvrk/PSM3/state_jaw_current
```

The bridge synchronizes these topics with `message_filters.ApproximateTimeSynchronizer`, converts the image to `bgr8`, concatenates arm and jaw joint positions, and sends raw samples to the GUI over ZeroMQ.

## ROS output topics

Default output topics published by the bridge:

```text
/dvrk_tracking/overlay
/dvrk_tracking/pose
/dvrk_tracking/joint_states
/dvrk_tracking/loss
/dvrk_tracking/fps
```

The output message headers use the original input sample timestamp.

## Troubleshooting

- If the GUI cannot load a ROS initialization frame, confirm the publisher and bridge are running, the endpoints match, and the image/joint topics are publishing.
- If `rclpy` cannot be imported, run the bridge with system Python after sourcing `/opt/ros/jazzy/setup.bash`.
- If CUDA, SAM2, renderer, or Qt imports fail, run the GUI from the `online_dvrk` Conda environment.
- If bridge latency grows, lower the GUI workload, reduce iterations per frame, or use mock-live/bridge modes that drop stale samples.
- If reinitialization looks inconsistent, pause first, click **Re-init**, add new prompts on the paused frame, then click **Continue**.
