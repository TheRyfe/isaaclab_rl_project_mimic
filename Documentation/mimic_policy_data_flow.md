# Mimic Policy Network Data Flow Documentation

## Overview
This document provides a comprehensive description of the data flow in and out of the mimic policy neural network, including all preprocessing, normalization, and data transformations.

## 1. Network Input Data Flow

### 1.1 Observation Components

The policy network receives two main observation types concatenated together:

#### **Proprioceptive Observations (`prop`)**
- **Source**: Robot's joint states and end-effector poses
- **Components** (total 54 dimensions):
  1. **Normalized Joint Positions** (20 dims)
     - Source: `robot.data.joint_pos` for first 20 joints
     - Normalization: `unscale()` function maps from joint limits to [-1, 1]
     - Formula: `(2.0 * x - upper - lower) / (upper - lower)`
     - Example: Joint at 0° with limits [-π, π] → 0.0
  
  2. **Normalized Joint Velocities** (20 dims)
     - Source: `robot.data.joint_vel` for first 20 joints  
     - Normalization: Using hardware velocity limits to [-1, 1]
     - Formula: Same `unscale()` as positions
     - Example: Velocity of 1 rad/s with limits [-10, 10] → 0.1
  
  3. **Actions** (20 dims)
     - Source: Previous step's actions sent to robot
     - Range: Already in [-1, 1] (raw from policy)
     - No additional normalization
  
  4. **Left Hand Pose** (7 dims)
     - Position (3): xyz in meters relative to base_link
     - Orientation (4): quaternion (w,x,y,z)
     - Source: `lhand_frame.data.target_pos_source` and `target_quat_source`
     - No normalization applied
  
  5. **Right Hand Pose** (7 dims)
     - Same format as left hand
     - Source: `rhand_frame.data.target_pos_source` and `target_quat_source`

#### **Ground Truth Observations (`gt`)**
- **Components** (total 60 dimensions):
  1. **Current Mimic Joint Positions** (20 dims)
     - Raw joint positions for the 20 controlled joints
     - Source: `robot.data.joint_pos[:, mimic_joint_indices]`
     - Units: radians
     - No normalization
  
  2. **Current Mimic Joint Velocities** (20 dims)
     - Raw joint velocities
     - Source: `robot.data.joint_vel[:, mimic_joint_indices]`
     - Units: rad/s
     - No normalization
  
  3. **Target Animation Joint Positions** (20 dims)
     - Target positions from CSV animation file
     - Source: `animation_pos_data[current_animation_step, :]`
     - Units: radians
     - Animation advances each environment step

### 1.2 Joint Mapping
The 20 controlled joints are mapped as follows:
```
CSV Column → Robot Joint Name
H1 → head_joint_1
H2 → head_joint_2  
H3 → head_joint_3
R1-R7 → right_arm_joint_1 through right_arm_joint_7
L1-L7 → left_arm_joint_1 through left_arm_joint_7
T1-T3 → torso_joint_1 through torso_joint_3
```

### 1.3 Complete Input Vector
Total input size: **114 dimensions**
- Proprioceptive: 54 dims
- Ground Truth: 60 dims

Example input tensor shape: `[num_envs, 114]`

## 2. Neural Network Architecture

### 2.1 Encoder Network
Processes the 114-dim input through:
1. **Layer 1**: Linear(114, 1024) + ELU + LayerNorm
2. **Layer 2**: Linear(1024, 512) + ELU + LayerNorm  
3. **Layer 3**: Linear(512, 256) + ELU + LayerNorm

Output: 256-dimensional feature vector

### 2.2 Policy Network
Takes 256-dim features and outputs actions:
1. **Layer 1**: Linear(256, 256) + ELU
2. **Layer 2**: Linear(256, 128) + ELU
3. **Layer 3**: Linear(128, 64) + ELU
4. **Layer 4**: Linear(64, 20) + Tanh

Output: 20 action means in [-1, 1]

Additionally outputs:
- **Log standard deviations**: 20 values
- Clipped to range [-20.0, 2.0]
- Used for stochastic action sampling

### 2.3 Value Network  
Estimates state value for PPO:
1. **Layer 1**: Linear(256, 256) + ELU
2. **Layer 2**: Linear(256, 128) + ELU
3. **Layer 3**: Linear(128, 1) + Identity

Output: Single scalar value estimate

## 3. Network Output Data Flow

### 3.1 Raw Policy Output
- **Action means**: 20 values in [-1, 1] from tanh activation
- **Log std**: 20 values for action distribution
- During training: Actions sampled from Gaussian(mean, exp(log_std))
- During evaluation: Deterministic (uses means directly)

### 3.2 Action Processing

1. **Clipping**: Actions passed through `torch.tanh()` for safety (already in [-1, 1])

2. **Position Control Mode Scaling**:
   ```python
   # Scale from [-1, 1] to joint position limits
   scaled_position = 0.5 * (action + 1.0) * (upper - lower) + lower
   ```
   Example: Action 0.5 for joint with limits [-π, π] → 0.5π radians

3. **Smoothing** (Moving Average):
   ```python
   smoothed_target = 0.1 * scaled_position + 0.9 * previous_target
   ```
   - Smoothing factor: 0.1 (10% new, 90% previous)
   - Reduces jerky movements

4. **Final Saturation**: Clamp to joint limits for safety

### 3.3 Applied to Robot
- Target positions sent to robot's position controller
- Joint indices: Only the 20 mimic joints are controlled
- Base actions: Currently disabled (num_base_actions = 0)

## 4. Data Flow Example

### Input Construction Example:
```python
# Environment has 4096 parallel simulations
num_envs = 4096

# 1. Proprioceptive observations
joint_pos_normalized = torch.tensor([0.1, -0.2, 0.3, ...])  # 20 values in [-1, 1]
joint_vel_normalized = torch.tensor([0.05, -0.1, 0.0, ...])  # 20 values in [-1, 1]
previous_actions = torch.tensor([0.2, -0.3, 0.1, ...])      # 20 values in [-1, 1]
lhand_pose = torch.tensor([0.5, 0.2, 1.1, 0.707, 0, 0, 0.707])  # pos + quat
rhand_pose = torch.tensor([0.5, -0.2, 1.1, 0.707, 0, 0, 0.707]) # pos + quat

prop_obs = torch.cat([
    joint_pos_normalized,  # 20
    joint_vel_normalized,  # 20
    previous_actions,      # 20
    lhand_pose[:3],       # 3
    lhand_pose[3:],       # 4
    rhand_pose[:3],       # 3
    rhand_pose[3:]        # 4
])  # Total: 54 dims

# 2. Ground truth observations
current_joint_pos = torch.tensor([0.523, -0.785, 1.047, ...])  # 20 values in radians
current_joint_vel = torch.tensor([0.5, -1.0, 0.0, ...])        # 20 values in rad/s
target_animation_pos = torch.tensor([0.6, -0.8, 1.1, ...])     # 20 values in radians

gt_obs = torch.cat([
    current_joint_pos,    # 20
    current_joint_vel,    # 20
    target_animation_pos  # 20
])  # Total: 60 dims

# 3. Complete observation
full_obs = torch.cat([prop_obs, gt_obs])  # 114 dims
```

### Output Processing Example:
```python
# Network outputs
raw_actions = torch.tensor([0.8, -0.6, 0.3, ...])  # 20 values

# Apply to robot (for joint with limits [-3.14, 3.14])
scaled_action = scale(0.8, -3.14, 3.14)  # = 2.513 radians
smoothed_target = 0.1 * 2.513 + 0.9 * previous_target
final_target = torch.clamp(smoothed_target, -3.14, 3.14)

# Send to robot
robot.set_joint_position_target(final_target, joint_ids=[...])
```

## 5. Key Characteristics

1. **Normalization Philosophy**: 
   - Joint states normalized to [-1, 1] for network stability
   - Raw values preserved in ground truth for accurate tracking
   - Actions use tanh to guarantee valid range

2. **Temporal Aspects**:
   - Animation advances 1 frame per environment step
   - Previous actions included for temporal context
   - Smoothing provides temporal consistency

3. **Safety Features**:
   - Multiple clipping/saturation stages
   - Velocity limits enforced
   - Smooth transitions via moving average

4. **Parallelization**:
   - All operations batched across environments
   - Typical batch size: 4096 environments
   - Efficient GPU utilization

## 6. Reward Signal (Additional Context)

The policy is trained to minimize the difference between current and target joint positions using:
- Exponential position tracking reward
- Joint-specific importance weights (head: 0.5, torso: 2.0, arms: 1.0)
- Additional link tracking for end-effectors
- Penalties for high velocities and jerky actions