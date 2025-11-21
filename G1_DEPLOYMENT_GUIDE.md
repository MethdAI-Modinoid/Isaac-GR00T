# Isaac GR00T Training and Deployment Guide for Unitree G1

This guide provides step-by-step instructions for training and deploying NVIDIA Isaac GR00T N1.5 on the Unitree G1 humanoid robot.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Docker Environment Setup](#docker-environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Deployment on G1 Robot](#deployment-on-g1-robot)
- [Troubleshooting](#troubleshooting)

---

## Overview

Isaac GR00T N1.5 is a foundation model for humanoid robot control that can be fine-tuned on custom datasets. This guide covers:
1. Setting up the Docker environment with all dependencies
2. Using pre-formatted datasets or converting XR teleoperation data
3. Fine-tuning GR00T N1.5 on G1 demonstration data
4. Deploying to the physical G1 robot

---

## Prerequisites

### Hardware Requirements
- **Training**: NVIDIA GPU (H100, L40, RTX 4090, RTX 3090 Ti, or A6000)
- **Inference**: NVIDIA GPU (RTX 3090 or better recommended)
- **G1 Robot**: Unitree G1 humanoid robot with network connectivity

### Software Requirements
- Ubuntu 20.04 or 22.04
- Docker with NVIDIA Container Toolkit
- CUDA 12.8 (included in Docker image)
- Python 3.11 (included in Docker image)

### Network Requirements
- G1 robot accessible on network (default IP: 192.168.123.164)
- Machine running inference server accessible to robot control node

---

## Docker Environment Setup

### Prerequisites: Install NVIDIA Container Toolkit

If not already installed:

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Build and Run Docker Container

1. **Clone the repository**:
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

2. **Build the Docker image**:
```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -t gr00t:dev \
  --build-arg PYTHON_VERSION=3.11 \
  -f Dockerfile .
```

This may take 30-60 minutes on first build.

3. **Run the container**:
```bash
./run_docker.sh
```

The `run_docker.sh` script automatically mounts:
- `./checkpoints` → `/workspace/checkpoints`
- `./datasets` → `/workspace/datasets`
- `./g1_deploy` → `/workspace/g1_deploy`
- X11 display for GUI support

### Docker Container Environment

The Docker container includes:
- **CUDA 12.8** with cuDNN
- **Python 3.11** with virtual environment at `/workspace/.venv`
- **PyTorch** (nightly build with CUDA 12.8 support)
- **Flash Attention** compiled from source
- **CycloneDDS** for G1 robot communication (ROS2/DDS)
- **Unitree SDK2 Python** for G1 low-level control
- **All required dependencies** pre-installed

### Environment Variables (Already Set in Docker)

The following are pre-configured in the Docker image:

```bash
CUDA_HOME=/usr/local/cuda-12.8
CYCLONEDDS_HOME=/root/cyclonedds/install
CMAKE_PREFIX_PATH=/root/cyclonedds/install
PYTHONPATH=/workspace
VIRTUAL_ENV=/workspace/.venv
PATH=/workspace/.venv/bin:$PATH
```

### Initialize Inside Docker

After starting the container, activate the environment and install the package:

```bash
# As root user (default in container)
source .venv/bin/activate
pip install -e .
pip install logging_mp
```

---

## Dataset Preparation

### Using Existing PhysicalAI Datasets

The repository includes pre-formatted datasets for G1:

```bash
datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/
├── g1-pick-apple/
├── g1-pick-pear/
├── g1-pick-grapes/
└── g1-pick-starfruit/
```

### Converting Your Own XR Teleoperation Data

> ⚠️ **IN PROGRESS**: The XR teleoperation data conversion workflow is currently being developed. The conversion scripts are available in `xr_data_conversion/` but the full integration with the training pipeline is still being validated.

If you have XR teleoperation data, preliminary conversion scripts are available:

```bash
cd xr_data_conversion

# Install dependencies
pip install -r requirements.txt

# Convert XR data to LeRobot format
python convert_xr_to_lerobot.py \
    --input xr_data/pick_cube \
    --output ../datasets/xr-pick-cube \
    --fps 30

# Verify the converted dataset
python verify_dataset.py ../datasets/xr-pick-cube
```

**Note**: Please verify the converted data format matches the PhysicalAI dataset structure before using for training.

For details on the data format, see:
- `getting_started/LeRobot_compatible_data_schema.md`
- `xr_data_conversion/README.md`

### Dataset Requirements

Your dataset must include:
- `meta/modality.json` - Schema defining state/action dimensions
- `meta/episodes.jsonl` - Episode metadata
- `meta/tasks.jsonl` - Task descriptions
- `meta/info.json` - Dataset information
- `data/chunk-*/episode_*.parquet` - State/action data
- `videos/chunk-*/observation.images.*/episode_*.mp4` - Visual observations

---

## Training

### Quick Start Training

For detailed training instructions, refer to [`getting_started/3_0_new_embodiment_finetuning.md`](getting_started/3_0_new_embodiment_finetuning.md).

### Single Dataset Training

```bash
python scripts/gr00t_finetune.py \
    --dataset-path datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/ \
    --num-gpus 1 \
    --batch-size 64 \
    --output-dir ./checkpoints/g1-pick-apple/ \
    --data-config unitree_g1 \
    --max-steps 10000
```

### Multi-Dataset Training (Recommended)

Training on multiple related tasks improves generalization:

```bash
# Define dataset list
dataset_list=(
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-pear/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-grapes/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-starfruit/"
)

# Train on all fruit-picking datasets
python scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --num-gpus 1 \
    --batch-size 95 \
    --output-dir ./checkpoints/full-g1-mix-fruits/ \
    --data-config unitree_g1 \
    --max-steps 15000
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset-path` | Path(s) to dataset directories | Required |
| `--output-dir` | Checkpoint save directory | `/tmp/gr00t` |
| `--data-config` | Data config name (see below) | `fourier_gr1_arms_only` |
| `--batch-size` | Batch size per GPU | 32 |
| `--max-steps` | Maximum training steps | 10000 |
| `--num-gpus` | Number of GPUs | 1 |
| `--save-steps` | Steps between checkpoints | 1000 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--tune-diffusion-model` | Fine-tune diffusion head | True |
| `--tune-projector` | Fine-tune projector | True |
| `--tune-llm` | Fine-tune language model | False |
| `--lora-rank` | LoRA rank (0=no LoRA) | 0 |

### Available Data Configs for G1

- `unitree_g1` - Full body control (43 DOF)
- `unitree_g1_full_body` - Full body with waist
- `unitree_g1_arms_only` - Upper body only (arms + hands)

### Monitor Training

Checkpoints are saved to `<output-dir>/checkpoint-<step>/`:
```bash
ls -lh checkpoints/full-g1-mix-fruits/
# checkpoint-1000/
# checkpoint-2000/
# ...
# checkpoint-15000/
```

### Open-Loop Evaluation

After training, evaluate the policy on held-out trajectories to verify learning:

```bash
python scripts/eval_policy.py --plot \
    --embodiment_tag new_embodiment \
    --model_path ./checkpoints/full-g1-mix-fruits/checkpoint-15000 \
    --data_config unitree_g1 \
    --dataset_path datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/ \
    --video_backend decord \
    --modality_keys left_arm right_arm
```

This generates plots comparing:
- Ground truth actions from the dataset
- Predicted actions from the policy
- Visualization saved to `trajectory_<id>_mse_<value>.png`

**Note**: The `eval_policy.py` script uses a non-interactive matplotlib backend and will save plots to disk rather than displaying them.

For more details on evaluation, see [`getting_started/3_0_new_embodiment_finetuning.md`](getting_started/3_0_new_embodiment_finetuning.md).

---

## Deployment on G1 Robot

### Architecture Overview

The deployment uses a client-server architecture:

```
┌─────────────────┐        ┌──────────────────┐        ┌─────────────────┐
│  Policy Server  │◄──────►│  Control Node    │◄──────►│  G1 Robot       │
│  (GPU Machine)  │  ZMQ   │  (eval_g1.py)    │ DDS/ROS│  (Hardware)     │
└─────────────────┘        └──────────────────┘        └─────────────────┘
        │                           │
        │                           ├─ Image Client (camera feeds)
        │                           ├─ Low-level control (unitree_sdk2)
        │                           └─ Action execution
```

### Step 1: Start the Policy Inference Server

The inference server runs the GR00T model and provides action predictions via ZMQ.

**Inside Docker container** (or in a separate terminal):

```bash
# Activate the virtual environment
source .venv/bin/activate

# Start the inference server
python scripts/inference_service.py --server \
    --model_path ./checkpoints/full-g1-mix-fruits/checkpoint-15000 \
    --embodiment-tag new_embodiment \
    --data-config unitree_g1_full_body \
    --denoising-steps 4
```

**Parameters**:
- `--model_path`: Path to your trained checkpoint
- `--embodiment-tag`: Use `new_embodiment` for fine-tuned G1 models
- `--data-config`: Must match the config used during training
- `--denoising-steps`: 4 is recommended (faster inference, minimal quality loss vs 16)

**Note**: The server will listen on `localhost:5555` by default. If running the control node on a different machine, add `--host 0.0.0.0` to accept external connections.

### Step 2: Setup G1 Robot Environment

On the machine that will control the G1:

```bash
# Source ROS2 and CycloneDDS setup
source ~/drive2/humanoid_ws/install/setup.bash
source ~/drive2/humanoid_ws/src/unitree_ros2/cyclonedds_ws/install/setup.bash
source ~/drive2/humanoid_ws/src/unitree_ros2/setup.sh

# Set CycloneDDS configuration
export CYCLONEDDS_URI=file:///home/deepansh/drive2/humanoid_ws/src/unitree_ros2/cyclonedds_ws/src/cyclonedds.xml
export CYCLONEDDS_HOME="~/drive2/humanoid_ws/src/unitree_ros2/cyclonedds_ws/install/cyclonedds"

# Enable multicast on loopback (if needed)
sudo ip link set lo multicast on
```

### Step 3: Start Image Server on G1

SSH into the Unitree G1 robot and start the image server to stream camera feeds:

```bash
ssh unitree@192.168.123.X
cd image_server
python image.py
```

**Note**: Replace `192.168.123.X` with your G1's actual IP address.

### Step 4: Run the G1 Control Node

The control node connects to both the policy server (for inference) and the G1 robot (for control).

**In the Docker container**:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the G1 control script
python g1_deploy/eval_g1.py \
    --policy_host=0.0.0.0 \
    --lang_instruction="Pick the apple from the table and place it into the basket."
```

**Parameters**:
- `--policy_host`: IP address of the policy server (use `0.0.0.0` if running in same container, or specific IP if remote)
- `--lang_instruction`: Natural language task description for the robot

**Important**: 
- The script assumes the network interface is `eth0` (default for Docker macvlan)
- Edit `g1_deploy/eval_g1.py` to change the network interface if needed
- Ensure CycloneDDS multicast is enabled: `sudo ip link set lo multicast on`

### Network Configuration

**Important**: Ensure the control node can reach both:
1. The policy server (for inference)
2. The G1 robot (for control and camera feeds)

If using Docker, the container is configured with:
```bash
--network macvlan --ip 192.168.123.71
```

Adjust `run_docker.sh` if your network setup differs.

### Safety Considerations

⚠️ **Safety First**:
1. Always have emergency stop ready
2. Start with the robot in a safe, clear area
3. Test with low-risk tasks first
4. Monitor robot behavior closely
5. Use appropriate PD gains (Kp, Kd) - see `g1_deploy/eval_g1.py`

---

## Complete Docker Deployment Workflow

### Step-by-Step Deployment

**1. Build and Start Container**:
```bash
# Build the Docker image (first time only)
DOCKER_BUILDKIT=1 docker build -t gr00t:dev -f Dockerfile .

# Run the container with mounted volumes
./run_docker.sh
```

**2. Initialize Inside Docker** (first time only):
```bash
# As root user (default in container)
source .venv/bin/activate
pip install -e .
pip install logging_mp
```

**3. Start Inference Server** (in Docker terminal):
```bash
source .venv/bin/activate

python scripts/inference_service.py --server \
    --model_path ./checkpoints/full-g1-mix-fruits/checkpoint-15000 \
    --embodiment-tag new_embodiment \
    --data-config unitree_g1_full_body \
    --denoising-steps 4
```

**4. Start Image Server on G1** (SSH to robot):
```bash
ssh unitree@192.168.123.164
cd image_server
python image.py
```

**5. Run G1 Control** (in separate Docker terminal or new container):
```bash
source .venv/bin/activate

# Enable multicast on loopback
sudo ip link set lo multicast on

# Run control node
python g1_deploy/eval_g1.py \
    --policy_host=0.0.0.0 \
    --lang_instruction="Pick the apple from the table and place it into the basket."
```

**Note**: Steps 3 and 5 should run in separate terminals or tmux sessions within the same Docker container.

---

## Advanced Configuration

### Custom Data Config

For custom robot configurations, create your own data config by subclassing `BaseDataConfig`. See `gr00t/experiment/data_config.py` for examples and the base class definition.

Example usage with external config:
```bash
python scripts/gr00t_finetune.py \
    --data-config "my_module.my_configs:MyG1Config" \
    --dataset-path /path/to/dataset \
    ...
```

### PD Control Gains

Adjust in `g1_deploy/eval_g1.py`:

```python
Kp = [60, 60, 60, 100, 40, 40, ...]  # Proportional gains
Kd = [1, 1, 1, 2, 1, 1, ...]         # Derivative gains
```

Higher Kp = stiffer response
Higher Kd = more damping

### Action Frequency

The control loop runs at the G1's control frequency. Adjust `action_horizon` in your data config to control how often the policy is queried vs. executing the action chunk.

---

## Troubleshooting

### Training Issues

**CUDA Out of Memory**
```bash
# Reduce batch size
--batch-size 32  # or 16

# Use gradient checkpointing (if available in config)
--gradient-checkpointing
```

**Slow Training**
```bash
# Reduce denoising steps during training
# Edit data config or use fewer diffusion steps

# Use multiple GPUs
--num-gpus 4
```

### Deployment Issues

**Cannot connect to policy server**
- Check firewall rules: `sudo ufw allow 5555`
- Verify server is listening: `netstat -tuln | grep 5555`
- Test with client: `python scripts/inference_service.py --client --host <server_ip>`

**G1 robot not responding**
- Check robot network: `ping 192.168.123.164`
- Verify CycloneDDS is running: `echo $CYCLONEDDS_URI`
- Check robot state and mode (should be in debug mode by default, if not, switch to debug mode using controller)
- Ensure image server is running on G1

**Robot moves erratically**
- Lower PD gains (Kp, Kd)
- Reduce action scaling
- Check action limits in data config
- Verify action normalization

**Image not received**
- Check image_client connection
- Verify camera indices
- Test image server independently

### Docker Issues

**Permission denied for CycloneDDS**
- Already fixed in Dockerfile with `chmod -R a+rX /root/cyclonedds/install`

**Network issues in Docker**
- The container uses macvlan network by default (configured in `run_docker.sh`)
- For debugging, you can temporarily use host network: `docker run --network host ...`
- Verify the container IP matches your network: `ip addr show eth0`

**Module import errors**
- Ensure you've run `pip install -e .` after starting the container
- Check that virtual environment is activated: `source .venv/bin/activate`

**Matplotlib warnings (FigureCanvasAgg non-interactive)**
- The evaluation scripts use non-interactive backend to avoid display issues
- Plots are automatically saved to disk instead of being shown

**Cannot reach G1 robot from Docker**
- Verify macvlan network configuration in `run_docker.sh`
- Test connectivity: `ping 192.168.123.X` (replace X with robot IP)
- Ensure multicast is enabled: `sudo ip link set lo multicast on`

---

## Common Commands Reference

All commands should be run inside the Docker container with the virtual environment activated (`source .venv/bin/activate`).

### Training
```bash
# Multi-dataset training (recommended)
dataset_list=(
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-pear/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-grapes/"
    "datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-starfruit/"
)

python scripts/gr00t_finetune.py \
    --dataset-path ${dataset_list[@]} \
    --num-gpus 1 \
    --batch-size 95 \
    --output-dir ./checkpoints/full-g1-mix-fruits/ \
    --data-config unitree_g1 \
    --max-steps 15000
```

### Evaluation
```bash
# Open-loop evaluation with visualization
python scripts/eval_policy.py --plot \
    --embodiment_tag new_embodiment \
    --model_path ./checkpoints/full-g1-mix-fruits/checkpoint-15000 \
    --data_config unitree_g1 \
    --dataset_path datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/ \
    --video_backend decord \
    --modality_keys left_arm right_arm
```

### Deployment

**Inference Server**:
```bash
python scripts/inference_service.py --server \
    --model_path ./checkpoints/full-g1-mix-fruits/checkpoint-15000 \
    --embodiment-tag new_embodiment \
    --data-config unitree_g1_full_body \
    --denoising-steps 4
```

**G1 Control Node**:
```bash
python g1_deploy/eval_g1.py \
    --policy_host=0.0.0.0 \
    --lang_instruction="Pick the apple from the table and place it into the basket."
```

---

## Additional Resources

- **Main README**: `README.md` - Overview and quick start
- **Dataset Format**: `getting_started/LeRobot_compatible_data_schema.md`
- **XR Data Conversion**: `xr_data_conversion/README.md`
- **Tutorials**: `getting_started/` - Jupyter notebooks for learning
- **Examples**: `examples/` - Benchmark results and configurations
- **Developer Notes**: `devnotes.txt` - Quick reference commands

---

## Contact and Support

- **GitHub Issues**: https://github.com/NVIDIA/Isaac-GR00T/issues
- **Website**: https://developer.nvidia.com/isaac/gr00t
- **Paper**: https://arxiv.org/abs/2503.14734

---

## License

See `LICENSE` file for details.

SPDX-License-Identifier: Apache-2.0
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
