# Mimic Task Model Architecture Overview

## Task Description
The mimic task trains a robot (AIREC) to reproduce pre-recorded motion sequences from CSV animation files using reinforcement learning. The robot learns to track joint positions while maintaining balance and smooth movements.

## Environment Configuration

### Episode Settings
- **Episode Length**: 900 steps (15 seconds at 60Hz control frequency)
- **Control Frequency**: 60Hz (dt = 1/60s with physics_dt=1/120s and decimation=2)
- **Number of Environments**: 4096 parallel environments

### Robot Configuration
- **Robot Model**: AIREC humanoid robot
- **Controlled Joints**: 20 joints total
  - Head: 3 joints (H1, H2, H3)
  - Right Arm: 7 joints (R1-R7)
  - Left Arm: 7 joints (L1-L7)
  - Torso: 3 joints (T1, T2, T3)
- **Control Mode**: Position control with action smoothing (moving average α=0.1)
- **Base**: Fixed (no base actions)

## Observation Space

The agent receives two types of observations:

### 1. Proprioceptive Observations (`prop`)
**Dimension**: 61 values
- **Normalized Joint Positions** (20): Current positions of controlled joints, normalized to [-1, 1]
- **Normalized Joint Velocities** (20): Current velocities of controlled joints, normalized using hardware limits
- **Actions** (20): Previous actions sent to the robot
- **Left Hand Pose** (7): Position (3) + Quaternion orientation (4)
- **Right Hand Pose** (7): Position (3) + Quaternion orientation (4)

### 2. Ground Truth Observations (`gt`)
**Dimension**: 60 values
- **Current Joint Positions** (20): Raw joint positions
- **Current Joint Velocities** (20): Raw joint velocities
- **Target Joint Positions** (20): Desired positions from animation file

**Total Observation Size**: 121 values

## Action Space

**Dimension**: 20 continuous values
- **Range**: [-1, 1] for each joint
- **Mapping**: Actions are scaled to joint position limits
- **Smoothing**: Actions are smoothed using exponential moving average

## Neural Network Architecture

### Encoder Network
Processes concatenated observations through a shared encoder:
- **Input**: 121 (concatenated prop + gt observations)
- **Hidden Layers**: [1024, 512, 256]
- **Activations**: ELU for all layers
- **Layer Normalization**: Applied after each layer
- **Output**: 256-dimensional feature vector

### Policy Network (Actor)
Outputs action means for stochastic policy:
- **Input**: 256 (from encoder)
- **Hidden Layers**: [256, 128, 64]
- **Activations**: [ELU, ELU, ELU, Tanh]
- **Output**: 20 values (action means)
- **Log Std**: Separate learnable parameters (not network outputs)
- **Log Std Constraints**: Clipped to [-20.0, 2.0]
- **Initial Log Std**: 0

### Value Network (Critic)
Estimates state value for PPO:
- **Input**: 256 (from encoder)
- **Hidden Layers**: [256, 128]
- **Activations**: [ELU, ELU, Identity]
- **Output**: 1 value (state value estimate)

## PPO Algorithm Configuration

### Core Parameters
- **Rollout Length**: 32 steps
- **Learning Epochs**: 8 per update
- **Mini-batches**: 4 per epoch
- **Discount Factor (γ)**: 0.99
- **GAE Lambda (λ)**: 0.95
- **Learning Rate**: 1e-5
- **Gradient Clipping**: 0.5 (max norm)

### PPO-Specific Parameters
- **Ratio Clip (ε)**: 0.2
- **Value Clip**: 0.2
- **Entropy Loss Scale**: 0.01
- **Value Loss Scale**: 2.0
- **KL Divergence Threshold**: 0.01

### Value Function Preprocessing
- **Type**: Running Standard Scaler
- **Purpose**: Normalizes value targets for stable training

## Reward Function

See `mimic_reward_equation.md` for detailed reward formulation. Key components:
- **Position Tracking**: Weighted exponential reward for joint accuracy
- **Link Tracking**: End-effector position/orientation matching
- **Velocity Penalty**: Discourages excessive speeds
- **Smoothness Penalty**: Encourages smooth actions
- **Staying Alive**: Small constant reward

## Training Details

### Data Collection
- **Parallel Environments**: 4096
- **Steps per Rollout**: 32
- **Total Steps per Update**: 131,072

### Optimization
- **Encoder Optimization**: Enabled (shared between policy and value)
- **Batch Size**: 32,768 (rollout_length × num_envs ÷ mini_batches)
- **Updates per Epoch**: 4 (mini_batches)
- **Total Gradient Steps per Update**: 32 (learning_epochs × mini_batches)

### Checkpointing & Logging
- **Checkpoint Frequency**: Every evaluation
- **Video Recording**: Every 4 evaluation episodes
- **Video Length**: 900 timesteps (full episode)
- **Logging**: TensorBoard + Weights & Biases

## Key Design Choices

1. **Shared Encoder**: Both policy and value networks share the same encoder for efficiency
2. **Layer Normalization**: Improves training stability
3. **Separate Observation Types**: Proprioceptive and ground truth observations are concatenated
4. **Action Smoothing**: Hardware-level smoothing prevents jerky movements
5. **Normalized Observations**: Joint positions/velocities normalized for better learning
6. **Position Control**: More stable than velocity/torque control for mimicry tasks

## Model Capacity
- **Total Parameters**: ~1.5M
  - Encoder: ~1.1M
  - Policy: ~0.3M
  - Value: ~0.1M