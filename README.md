# AIREC Robot Motion Mimicry Environment

A specialized Isaac Lab environment for training AIREC robots to mimic pre-recorded animations using reinforcement learning.

![AIREC Robot Mimic Training](https://github.com/user-attachments/assets/72036a2f-41ab-4317-ad30-8a165afa83a5)

## Overview

This project implements a motion mimicry task where an AIREC humanoid robot learns to reproduce joint motions from CSV animation files. The environment includes advanced features like ghost robot visualization, external force disturbances for robustness training, and comprehensive reward shaping.

### Key Features

- **Motion Mimicry**: Train robots to follow pre-recorded joint trajectories from CSV files
- **Ghost Robot Visualization**: Real-time target pose visualization for debugging and analysis
- **External Force Disturbances**: Configurable random forces for robustness training
- **Comprehensive Reward System**: Position tracking, velocity penalties, and action smoothness
- **Flexible Termination Conditions**: Joint limits, torso tilt, and animation completion
- **WandB Integration**: Automatic video recording and experiment tracking

## Installation

### Prerequisites

1. **Isaac Lab**: Follow the installation instructions at [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
2. **Python**: Python 3.8+ with PyTorch
3. **NVIDIA GPU**: CUDA-capable GPU for simulation

### Setup

1. **Clone and Setup Isaac Lab** (if not already done):
```bash
cd ~/
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

2. **Install the Isaac Lab RL package**:
```bash
cd scripts/AIREC_Packages/isaaclab_rl
pip install -e .
```

3. **Navigate to the mimic project**:
```bash
cd isaaclab_rl_project_mimic
```

4. **Setup WandB** (optional but recommended):
```bash
wandb login
```

## Quick Start

### Training

Train the AIREC robot to mimic the default walking animation:

```bash
# Basic training with visualization
python train.py --task Mimic --num_envs 256 --seed 42

# Headless training for maximum performance (no videos)
python train.py --task Mimic --num_envs 512 --headless

# Headless training with video recording enabled
python train.py --task Mimic --num_envs 512 --headless --enable_cameras

# Reproducible training with specific seed
python train.py --task Mimic --num_envs 256 --seed 42 --headless --enable_cameras

# High-performance training (large batch, no videos)
python train.py --task Mimic --num_envs 1024 --headless

# Debug training (few environments, with visualization)
python train.py --task Mimic --num_envs 64 --seed 42
```

### Evaluation

Test a trained model:

```bash
# Play with visualization (recommended)
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 64

# Play latest model with fewer environments for detailed observation
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 16

# Record evaluation videos
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 16 --video

# Single environment for detailed analysis
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 1

# Headless evaluation (for performance testing)
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 256 --headless

# Extended evaluation with many episodes
python play.py --task Mimic --checkpoint logs/*/prop_mimic/*/checkpoints/best_agent.pt --num_envs 64 --num_episodes 100
```

**Finding Your Latest Checkpoint:**
```bash
# List all training runs
ls -la logs/*/prop_mimic/

# Find the most recent checkpoint
find logs -name "best_agent.pt" -type f -exec ls -la {} \; | sort -k6,7
```

## Configuration

### Animation Files

The environment loads joint trajectories from CSV files. The default animation is `assets/animation/walkingsupport.csv`.

**CSV Format Requirements**:
- Columns: `H1, H2, H3, R1, R2, R3, R4, R5, R6, R7, L1, L2, L3, L4, L5, L6, L7, T1, T2, T3`
- Joint angles in radians
- 60 FPS sampling rate (configurable)

**Joint Mapping**:
- `H1-H3`: Head joints (1-3)
- `R1-R7`: Right arm joints (1-7) 
- `L1-L7`: Left arm joints (1-7)
- `T1-T3`: Torso joints (1-3)

### Key Configuration Files

#### Agent Configuration: `tasks/mimic/agents/prop_mimic.yaml`

```yaml
# Training parameters
max_global_timesteps_M: 500.0
num_eval_envs: 64

# Video recording
upload_videos: 1
video_upload_interval: 50  # Upload every 50 evaluation episodes

# WandB settings
wandb_kwargs:
  entity: "your_username"
  project: "airec_mimic"
  group: "mimic_training"
```

#### Environment Configuration: `tasks/mimic/mimic.py`

Key parameters in `MimicEnvCfg`:

```python
# Animation file and timing
animation_file: str = "assets/animation/walkingsupport.csv"
animation_dt_info: float = 1.0 / 60.0  # 60 FPS

# Episode settings
episode_length_s: float = 20.0
dynamic_episode_length_buffer_s: float = 2.0

# Control mode
control_mode: str = "position"  # or "velocity"

# External disturbances
external_disturbance: ExternalDisturbanceCfg = ExternalDisturbanceCfg()
```

## Reward System

The reward function balances multiple objectives:

### Position Tracking (`joint_pos_tracking_reward_scale: 4.0`)
- Exponential reward based on joint position errors
- Higher rewards for closer tracking of target poses

### Velocity Penalties (`current_joint_vel_penalty_scale: -0.001`)
- Penalizes excessive joint velocities
- Encourages smooth motion

### Action Smoothness (`action_smoothness_penalty_scale: -0.01`)
- Penalizes rapid changes in actions between timesteps
- Promotes stable control

### Staying Alive (`staying_alive_reward: 0.005`)
- Small constant reward for remaining active
- Encourages longer episodes

## External Force Disturbances

The environment can apply random external forces to improve robustness:

```python
external_disturbance = ExternalDisturbanceCfg(
    enable_disturbances=True,
    target_body_name="right_arm_link_5",
    force_magnitude_range=(300.0, 850.0),  # Newtons
    duration_range=(0.5, 2.5),  # seconds
    interval_range=(0.5, 3.0),  # seconds between disturbances
    disturbance_probability=0.01,  # 1% chance per step
    directional_bias=(0.0, 0.0, -1.0),  # Downward bias
    bias_strength=0.4,  # 40% bias, 60% random
    enable_force_visualization=True
)
```

## Termination Conditions

Episodes terminate when:

1. **Animation Complete**: The full animation sequence has been played
2. **Joint Limits**: Robot joints exceed 95% of their soft limits
3. **Torso Tilt**: Torso joints exceed configured tilt thresholds
4. **Episode Timeout**: Maximum episode length is reached

## Ghost Robot Visualization

The ghost robot shows the target pose in real-time:

- **Red colored** semi-transparent robot showing target positions
- **Base components hidden** to reduce visual clutter
- **Automatically disabled** in headless mode for performance

## Monitoring and Logging

### WandB Integration

The environment automatically logs:
- Reward components (position tracking, penalties, etc.)
- Animation frame progress
- Termination statistics
- External force magnitudes
- **Periodic videos** of training progress

### Console Output

Key metrics are printed during training:
- Animation loading status
- Force disturbance information
- Training progress and episode statistics

## Common Issues and Solutions

### Video Recording Error in Headless Mode
**Error:** `RuntimeError: Cannot render 'rgb_array' when the simulation render mode is 'NO_GUI_OR_RENDERING'`

**Solutions:**
1. **Enable cameras for headless video recording:**
   ```bash
   python train.py --task Mimic --headless --enable_cameras
   ```

2. **Disable video recording for pure headless training:**
   - Edit `tasks/mimic/agents/prop_mimic.yaml`
   - Change `upload_videos: 1` to `upload_videos: 0`

3. **Train with GUI for video recording:**
   ```bash
   python train.py --task Mimic --num_envs 256  # Remove --headless
   ```

### Low Framerates
- Use `--headless` flag for training
- Reduce `num_envs` parameter
- Disable force visualization: `enable_force_visualization=False`
- Disable video recording: set `upload_videos: 0` in config

### Animation Not Loading
- Check CSV file path in configuration
- Verify CSV column names match expected format
- Ensure joint angles are in radians, not degrees

### Poor Tracking Performance
- Increase `joint_pos_tracking_reward_scale`
- Adjust `pos_error_variance_scale` for reward sensitivity
- Check episode length is sufficient for full animation

### External Forces Not Working
- Verify robot body names with: `print(robot.body_names)`
- Check CUDA compatibility for force calculations
- Ensure Isaac Lab supports external forces

## Advanced Usage

### Custom Animations

1. Create a CSV file with the required column format
2. Update `animation_file` path in configuration
3. Adjust `animation_dt_info` to match your recording framerate

### Multi-Robot Training

Scale up environments for distributed training:

```bash
python train.py --task Mimic --num_envs 1024 --headless --device cuda:0
```

### Hyperparameter Sweeping

Use the included sweep functionality:

```bash
python sweep.py --config_path configs/sweep_config.yaml
```

## Repository Structure

```
isaaclab_rl_project_mimic/
├── assets/
│   ├── airec/              # Robot USD files and configs
│   └── animation/          # CSV animation files
├── tasks/
│   └── mimic/              # Environment implementation
│       ├── mimic.py        # Main environment class
│       ├── airec.py        # Robot-specific base class
│       └── agents/         # Agent configurations
├── configs/                # Training configurations  
├── train.py               # Training script
├── play.py               # Evaluation script
└── common_utils.py       # Shared utilities
```

## Contributing

1. Follow the existing code style and documentation standards
2. Add tests for new features
3. Update this README for any configuration changes
4. Use the provided formatters: `make format`

## License

This project is licensed under the BSD-3-Clause License - see the Isaac Lab project for details.

## Acknowledgments

- Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- Uses [SKRL](https://skrl.readthedocs.io/) for reinforcement learning
- Integration with [Weights & Biases](https://wandb.ai/) for experiment tracking