# dVRK Live Tracking GUI

## Overview

Implementation of the paper [Real-time Rendering-based Surgical Instrument Tracking via
Evolutionary Optimization](https://arxiv.org/pdf/2603.11404).

This branch focuses on the live dVRK tracking GUI. The GUI runs Surgical SAM2 segmentation, dVRK rendering, and online tracking in a Conda Python 3.10 environment. It supports offline file playback, mock-live replay, and direct ROS 2 input/output.

## Supported environments

Laptop development:

```text
Ubuntu 24.04 WSL2
Conda Python 3.10
Offline mode
Mock-live mode
ROS 2 direct mode not required
```

Perception PC:

```text
Ubuntu 22.04
ROS 2 Humble
Conda environment: online_dvrk
Python 3.10
Direct rclpy integration
```

Offline and Mock-live modes do not require ROS to be installed or sourced.

## Architecture

Direct ROS mode runs in one Python process:

```text
Qt main thread:
    GUI and user interaction

ROS executor thread:
    rclpy subscriptions and publishers

Tracking QThread:
    SAM2, CUDA rendering, optimizer, and tracking
```

ROS callbacks only convert synchronized sensor messages into latest-sample buffers and publish latest tracking results. They do not run SAM2, CUDA inference, rendering, tracking optimization, or Qt widget operations.

## Conda tracking environment setup

The tracking GUI environment has been tested on **Ubuntu 22.04** and **Ubuntu 24.04 (WSL 2)** with **CUDA 12.6**.

If CUDA 12.6 is installed in `/usr/local/cuda-12.6`, set:

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
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
```

Install project dependencies:

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

## Ubuntu 22.04 ROS 2 Humble setup

Install ROS-side packages on the perception PC:

```bash
sudo apt update

sudo apt install -y \
    ros-humble-ros-base \
    ros-humble-rclpy \
    ros-humble-cv-bridge \
    ros-humble-message-filters \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    python3-rosdep
```

Initialize `rosdep` once per machine:

```bash
sudo rosdep init
rosdep update
```

This repository is launched as a normal Python application, so the explicit apt package list above is the primary installation path.

## rclpy and rosdep

Use `rclpy` at runtime. It is the ROS 2 Python client library that creates nodes, subscriptions, publishers, executors, QoS profiles, and timers.

`rosdep` is not a runtime communication library and does not replace `rclpy`. It is an installation tool that resolves package dependencies declared by ROS packages.

Do not run:

```bash
pip install rclpy
```

Use the ROS 2 Humble packages installed by apt, then expose them to the active Conda Python process by sourcing ROS after activating Conda.

## Direct ROS compatibility preflight

On the perception PC, activate in this order:

```bash
conda activate online_dvrk
source /opt/ros/humble/setup.bash
```

Run the import preflight:

```bash
python - <<'PY'
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version)

import rclpy
from cv_bridge import CvBridge
from message_filters import (
    ApproximateTimeSynchronizer,
    Subscriber,
)
from sensor_msgs.msg import Image, JointState

print("rclpy:", rclpy.__file__)
print("Direct ROS 2 imports succeeded.")
PY
```

The Python executable should be inside:

```text
.../envs/online_dvrk/bin/python
```

Run the runtime smoke test:

```bash
python scripts/check_direct_ros2_environment.py
```

This script prints the Python executable, Python version, ROS distribution, `rclpy` location, NumPy version, OpenCV location/version, `cv_bridge` availability, `message_filters` availability, and RMW implementation. It also creates a private ROS context and node, spins once, destroys the node, and shuts down cleanly.

## Required models and calibration files

The GUI needs:

- An MP4 video and joint-angle YAML for Offline or Mock-live mode.
- Camera calibration YAML.
- Hand-eye calibration YAML.
- `LND.json` robot kinematic description.
- LND mesh files under `urdfs/dVRK/meshes`.
- Surgical SAM 2 checkpoint at `SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth`.
- Keypoint detection model weights when point loss is enabled.

## Launching the GUI

Laptop Offline/Mock-live mode:

```bash
conda activate online_dvrk
python scripts/custom_live_gui.py
```

Perception PC direct ROS mode:

```bash
conda activate online_dvrk
source /opt/ros/humble/setup.bash
python scripts/custom_live_gui.py
```

The laptop command must work without sourcing ROS.

## Offline mode

Offline mode opens the selected MP4 directly and pairs video frame `i` with joint-angle YAML entry `"i"`. It stops at the shorter of the video and YAML sequences.

Use this mode for deterministic playback from files.

## Mock-live mode

Mock-live mode uses the same MP4 and joint-angle YAML as Offline mode, but replays samples according to wall-clock time. If tracking falls behind, stale samples are skipped so latency does not grow continuously.

Use **Replay rate** to choose the simulated stream rate. Enable **Loop** to replay the sequence repeatedly.

## ROS 2 mode

ROS 2 mode subscribes directly to synchronized image, arm-joint, and jaw topics inside the tracking application. Select **ROS 2** in the GUI, configure the topic names, click **Load Initialization Frame**, add prompts, then click **Start**.

ROS imports are lazy. Opening the GUI or using Offline/Mock-live mode does not require `rclpy`.

## Testing without a dVRK

Use the fake ROS 2 publisher to replay a video and joint YAML as ROS topics, then start the GUI in ROS 2 mode.

Verify topics with:

```bash
ros2 topic list
ros2 topic hz --qos-profile sensor_data /stereo/left/image
ros2 topic echo --once --qos-profile sensor_data /dvrk/PSM3/state_joint_current
ros2 topic echo --once --qos-profile sensor_data /dvrk/PSM3/state_jaw_current
```

To receive one synchronized sample without starting the GUI:

```bash
python scripts/test_direct_ros2_input.py
```

## Running the fake ROS 2 publisher

In a sourced Humble shell with `online_dvrk` active:

```bash
conda activate online_dvrk
source /opt/ros/humble/setup.bash

python scripts/mock_dvrk_ros2_publisher.py \
    --video data/custom/bag1/left.mp4 \
    --joint-angles data/custom/bag1/joint_angles.yaml \
    --rate 30 \
    --loop
```

The fake publisher sends image, arm-joint, and jaw messages with matching timestamps and sensor-data QoS.

## GUI controls

- **Mode** selects Offline, Mock live, or ROS 2 input.
- **Video** and **Joint angles yaml** are used by Offline and Mock-live modes.
- **Image topic**, **Arm joint topic**, **Jaw topic**, **Sync queue size**, **Sync slop**, and **Sample timeout** are used by ROS 2 mode.
- **Camera calibration yaml**, **Handeye yaml**, and **LND json** are required in every mode.
- **Renderer** chooses `nvdiffrast` or `pytorch3d`.
- **Optimizer** chooses `CMA-ES`, `XNES`, or `Gradient`.
- **Downscale factor** trades resolution for speed.
- **Use low-res mesh** enables lightweight dVRK meshes.
- **Use point loss** enables keypoint point loss.
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

Default input topics:

```text
/stereo/left/image
/dvrk/PSM3/state_joint_current
/dvrk/PSM3/state_jaw_current
```

The GUI synchronizes these topics with `message_filters.ApproximateTimeSynchronizer`, converts the image to `bgr8`, concatenates arm and jaw joint positions, and stores only the latest synchronized sample.

## ROS output topics

Default output topics:

```text
/dvrk_tracking/overlay
/dvrk_tracking/pose
/dvrk_tracking/joint_states
/dvrk_tracking/loss
/dvrk_tracking/fps
```

The output message headers use the original input sample timestamp.

## Troubleshooting

- If direct ROS imports fail, confirm `conda activate online_dvrk` ran before `source /opt/ros/humble/setup.bash`.
- If Python is `/usr/bin/python3`, restart the shell and activate Conda first.
- If `rclpy` imports but `cv_bridge` or OpenCV fails, run `scripts/check_direct_ros2_environment.py` and inspect possible Conda/ROS shared-library conflicts.
- If the GUI cannot load a ROS initialization frame, confirm the fake or real publisher is running, topic names match, QoS is compatible, and synchronization slop is large enough.
- If Offline or Mock-live mode fails on the laptop, do not source ROS; run from the `online_dvrk` Conda environment only.
- If tracking falls behind ROS input, stale samples are intentionally dropped to avoid growing latency.
