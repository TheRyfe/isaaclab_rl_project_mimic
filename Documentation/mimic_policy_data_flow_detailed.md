# Detailed Mimic Policy Network Data Flow Documentation

## 1. EXACT INPUT DATA STRUCTURES

### 1.1 Proprioceptive Observations (54 dimensions)

#### 1.1.1 Normalized Joint Positions (20 dimensions)
**Source**: `robot.data.joint_pos[env_ids, prop_joint_indices]`

**Prop Joint Indices** (first 20 joints of the robot):
```
Index 0-19 correspond to the following joints IN THIS EXACT ORDER:
0: base_joint_trans_x
1: base_joint_trans_y  
2: base_joint_rot_yaw
3: torso_joint_1
4: torso_joint_2
5: torso_joint_3
6: left_arm_joint_1
7: left_arm_joint_2
8: left_arm_joint_3
9: left_arm_joint_4
10: left_arm_joint_5
11: left_arm_joint_6
12: left_arm_joint_7
13: right_arm_joint_1
14: right_arm_joint_2
15: right_arm_joint_3
16: right_arm_joint_4
17: right_arm_joint_5
18: right_arm_joint_6
19: right_arm_joint_7
```

**Normalization**: `unscale()` function: `(2.0 * x - upper - lower) / (upper - lower)`

**Joint Position Limits** (from soft_joint_pos_limits):
- The exact limits are loaded from the USD file at runtime
- For joints without defined limits, defaults to [-π, π] radians
- Positions are clamped to these limits before normalization

**Data Structure**:
```python
# Shape: [num_envs, 20]
normalised_joint_pos = torch.Tensor([
    [base_trans_x_norm, base_trans_y_norm, base_rot_yaw_norm,  # indices 0-2
     torso_j1_norm, torso_j2_norm, torso_j3_norm,              # indices 3-5
     left_arm_j1_norm, ..., left_arm_j7_norm,                  # indices 6-12
     right_arm_j1_norm, ..., right_arm_j7_norm]                # indices 13-19
])
# All values in range [-1, 1]
```

#### 1.1.2 Normalized Joint Velocities (20 dimensions)
**Source**: `robot.data.joint_vel[env_ids, prop_joint_indices]`

**Normalization**: Using hardware velocity limits
```python
normalised_joint_vel = unscale(
    joint_vel, 
    -hard_vel_limits[prop_joint_indices], 
    hard_vel_limits[prop_joint_indices]
)
```

**Velocity Limits** (from robot.data.joint_vel_limits):
- Loaded from USD file at runtime
- If not defined, defaults to ±10 rad/s

**Data Structure**:
```python
# Shape: [num_envs, 20]
# Same joint ordering as positions
# All values normalized to [-1, 1]
```

#### 1.1.3 Previous Actions (20 dimensions)
**Source**: `self.actions` (previous step's network output)

**Range**: Already in [-1, 1] (direct from policy network)

**Note**: These are the MIMICKED joints only, not the full 20 prop joints:
```python
# Shape: [num_envs, 20]
# Order corresponds to mimicked joints (see section 1.2.3)
actions = torch.Tensor([
    head_j1_action, head_j2_action, head_j3_action,      # H1, H2, H3
    right_arm_j1_action, ..., right_arm_j7_action,       # R1-R7
    left_arm_j1_action, ..., left_arm_j7_action,         # L1-L7
    torso_j1_action, torso_j2_action, torso_j3_action    # T1-T3
])
```

#### 1.1.4 Left Hand Pose (7 dimensions)
**Source**: `lhand_frame.data.target_pos_source[..., 0, :]` and `target_quat_source[..., 0, :]`

**Frame**: Relative to robot base_link with offset [0.0, 0.0, 0.02] from palm

**Data Structure**:
```python
# Position (3): [x, y, z] in meters
# Orientation (4): [w, x, y, z] quaternion
lhand_pose = torch.Tensor([x, y, z, qw, qx, qy, qz])
# No normalization applied
```

#### 1.1.5 Right Hand Pose (7 dimensions)
**Source**: `rhand_frame.data.target_pos_source[..., 0, :]` and `target_quat_source[..., 0, :]`

**Same structure as left hand**

### 1.2 Ground Truth Observations (60 dimensions)

#### 1.2.1 Current Mimicked Joint Positions (20 dimensions)
**Source**: `robot.data.joint_pos[:, mimic_joint_indices_in_robot]`

**Mimic Joint Indices Mapping**:
```python
# These are the ACTUAL joint indices in the robot for mimicked joints:
mimic_joint_indices_in_robot = [
    20,  # head_joint_1 (H1)
    21,  # head_joint_2 (H2)
    22,  # head_joint_3 (H3)
    13,  # right_arm_joint_1 (R1)
    14,  # right_arm_joint_2 (R2)
    15,  # right_arm_joint_3 (R3)
    16,  # right_arm_joint_4 (R4)
    17,  # right_arm_joint_5 (R5)
    18,  # right_arm_joint_6 (R6)
    19,  # right_arm_joint_7 (R7)
    6,   # left_arm_joint_1 (L1)
    7,   # left_arm_joint_2 (L2)
    8,   # left_arm_joint_3 (L3)
    9,   # left_arm_joint_4 (L4)
    10,  # left_arm_joint_5 (L5)
    11,  # left_arm_joint_6 (L6)
    12,  # left_arm_joint_7 (L7)
    3,   # torso_joint_1 (T1)
    4,   # torso_joint_2 (T2)
    5,   # torso_joint_3 (T3)
]
```

**Units**: Radians (raw, not normalized)

**Data Structure**:
```python
# Shape: [num_envs, 20]
# Order: H1,H2,H3,R1-R7,L1-L7,T1-T3
current_mimic_joints_pos = torch.Tensor([...])  # Raw radians
```

#### 1.2.2 Current Mimicked Joint Velocities (20 dimensions)
**Source**: `robot.data.joint_vel[:, mimic_joint_indices_in_robot]`

**Units**: rad/s (raw, not normalized)

**Same joint ordering as positions**

#### 1.2.3 Target Animation Joint Positions (20 dimensions)
**Source**: `animation_pos_data[current_animation_step, :]`

**Animation Data Loading**:
- Loaded from CSV file: `assets/animation/walkingsupport.csv`
- CSV columns: `H1,H2,H3,R1,R2,R3,R4,R5,R6,R7,L1,L2,L3,L4,L5,L6,L7,T1,T2,T3`
- Animation advances 1 frame per environment step
- When reaching max frames, stays at last frame (no looping)

**Units**: Radians (raw from CSV)

**Data Structure**:
```python
# Shape: [num_envs, 20]
# Direct from CSV in order: H1,H2,H3,R1-R7,L1-L7,T1-T3
target_animation_joint_pos = animation_pos_data[safe_anim_indices, :]
```

### 1.3 Complete Input Tensor

**Final Concatenation**:
```python
# Shape: [num_envs, 114]
full_observation = torch.cat([
    # Proprioceptive (54)
    normalised_joint_pos,      # 20 dims [0:20]
    normalised_joint_vel,      # 20 dims [20:40]
    actions,                   # 20 dims [40:60] (mimicked joints only, but still 20)
    lhand_pos,                # 3 dims  [60:63]
    lhand_quat,               # 4 dims  [63:67]
    rhand_pos,                # 3 dims  [67:70]
    rhand_quat,               # 4 dims  [70:74]
    # Ground Truth (60)
    current_mimic_joints_pos,  # 20 dims [74:94]
    current_mimic_joints_vel,  # 20 dims [94:114]
    target_animation_pos       # 20 dims [114:134]
], dim=-1)
```

## 2. NEURAL NETWORK ARCHITECTURE

### 2.1 Encoder Network
```python
# Input: 114 dimensions
Layer1: Linear(114, 1024) -> ELU -> LayerNorm
Layer2: Linear(1024, 512) -> ELU -> LayerNorm  
Layer3: Linear(512, 256) -> ELU -> LayerNorm
# Output: 256-dimensional feature vector
```

### 2.2 Policy Head
```python
# Input: 256 dimensions (from encoder)
Layer1: Linear(256, 256) -> ELU
Layer2: Linear(256, 128) -> ELU
Layer3: Linear(128, 64) -> ELU
Layer4: Linear(64, 20) -> Tanh
# Output: 20 action means in [-1, 1]

# Also outputs log_std for each action
log_std: Parameter(20) clipped to [-20.0, 2.0]
```

### 2.3 Value Head
```python
# Input: 256 dimensions (from encoder)
Layer1: Linear(256, 256) -> ELU
Layer2: Linear(256, 128) -> ELU
Layer3: Linear(128, 1) -> Identity
# Output: Single scalar value
```

## 3. OUTPUT DATA STRUCTURES

### 3.1 Raw Network Output
**Shape**: [num_envs, 20]

**Action Order** (matches CSV column order):
```
Index 0: head_joint_1 (H1)
Index 1: head_joint_2 (H2)
Index 2: head_joint_3 (H3)
Index 3: right_arm_joint_1 (R1)
Index 4: right_arm_joint_2 (R2)
Index 5: right_arm_joint_3 (R3)
Index 6: right_arm_joint_4 (R4)
Index 7: right_arm_joint_5 (R5)
Index 8: right_arm_joint_6 (R6)
Index 9: right_arm_joint_7 (R7)
Index 10: left_arm_joint_1 (L1)
Index 11: left_arm_joint_2 (L2)
Index 12: left_arm_joint_3 (L3)
Index 13: left_arm_joint_4 (L4)
Index 14: left_arm_joint_5 (L5)
Index 15: left_arm_joint_6 (L6)
Index 16: left_arm_joint_7 (L7)
Index 17: torso_joint_1 (T1)
Index 18: torso_joint_2 (T2)
Index 19: torso_joint_3 (T3)
```

### 3.2 Action Processing Pipeline

#### Step 1: Safety Clipping
```python
processed_actions = torch.tanh(actions)  # Ensures [-1, 1]
```

#### Step 2: Scaling to Joint Limits
**Joint Position Limits** (from deployment script):
```python
# Order: T1,T2,T3,H1,L1,R1,H2,L2,R2,H3,L3,R3,L4,R4,L5,R5,L6,R6,L7,R7
LOWER_LIMITS = np.array([
    -0.7854, -0.8727, -1.7453,  # T1, T2, T3
    -1.5708,                     # H1
    -1.2217, -1.2217,           # L1, R1
    -0.8727,                     # H2
    -2.1817, -2.1817,           # L2, R2
    -0.6981,                     # H3
    -2.0944, -2.0944,           # L3, R3
    -0.1745, -0.1745,           # L4, R4
    -2.9671, -2.9671,           # L5, R5
    -1.5708, -1.5708,           # L6, R6
    -0.1745, -0.1745            # L7, R7
])

UPPER_LIMITS = np.array([
    0.7854, 1.8326, 1.7453,     # T1, T2, T3
    1.5708,                      # H1
    4.1888, 4.1888,             # L1, R1
    0.8727,                      # H2
    0.3491, 0.3491,             # L2, R2
    0.6981,                      # H3
    2.0944, 2.0944,             # L3, R3
    2.4435, 2.4435,             # L4, R4
    2.9671, 2.9671,             # L5, R5
    1.5708, 1.5708,             # L6, R6
    1.5708, 1.5708              # L7, R7
])
```

**Scaling Formula**:
```python
scaled_position = 0.5 * (action + 1.0) * (upper - lower) + lower
```

#### Step 3: Moving Average Smoothing
```python
# Smoothing factor: 0.1 (10% new, 90% previous)
smoothed_target = 0.1 * scaled_position + 0.9 * previous_target
```

#### Step 4: Final Safety Saturation
```python
final_target = torch.clamp(smoothed_target, lower_limits, upper_limits)
```

### 3.3 Applied to Robot
```python
# Only the 20 mimicked joints are controlled
robot.set_joint_position_target(
    final_target, 
    joint_ids=mimic_joint_indices_in_robot
)
```

## 4. DATA FLOW EXAMPLE WITH REAL VALUES

### Input Example (Single Environment):
```python
# Proprioceptive observations
normalised_joint_pos = [
    0.0, 0.0, 0.0,           # base (not moving)
    -0.223, 0.402, 0.0,      # torso normalized
    0.6, -0.3, -0.1, 0.7, 0.2, 0.1, 0.05,  # left arm
    0.6, -0.3, -0.1, 0.7, 0.2, 0.1, 0.05   # right arm
]  # 20 values

normalised_joint_vel = [0.0] * 20  # Robot at rest

previous_actions = [
    0.0, 0.0, 0.0,           # head
    0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right arm
    0.0, -1.0, -0.03, 0.0, 0.0, 0.0, 0.0,  # left arm
    0.0, 0.0, 0.0            # torso
]  # 20 values

lhand_pose = [0.5, 0.2, 1.1, 1.0, 0.0, 0.0, 0.0]  # pos + quat
rhand_pose = [0.5, -0.2, 1.1, 1.0, 0.0, 0.0, 0.0]

# Ground truth observations
current_joint_pos = [
    0.0, 0.0, 0.0,           # H1, H2, H3
    0.0, -1.5708, 0.0, 0.0, 0.0, 0.0, 0.0,  # R1-R7
    0.0, -1.5708, -0.0262, 0.0, 0.0, 0.0, 0.0,  # L1-L7
    0.0, 0.0, 0.0            # T1, T2, T3
]  # 20 values in radians

current_joint_vel = [0.0] * 20  # rad/s

target_animation_pos = [
    0.0, 0.0, 0.0,           # H1, H2, H3 from CSV
    0.0, -1.5708, 0.0, 0.0, 0.0, 0.0, 0.0,  # R1-R7
    0.0, -1.5708, -0.0262, 0.0, 0.0, 0.0, 0.0,  # L1-L7
    0.0, 0.0, 0.0            # T1, T2, T3
]  # 20 values from animation frame
```

### Output Example:
```python
# Network outputs actions in [-1, 1]
raw_actions = [
    0.0, 0.0, 0.0,           # Head stays neutral
    0.0, -0.95, 0.1, 0.0, 0.0, 0.0, 0.0,  # Right arm tracks
    0.0, -0.95, -0.02, 0.0, 0.0, 0.0, 0.0,  # Left arm tracks
    0.0, 0.0, 0.0            # Torso stays neutral
]

# After scaling (example for R2: right_arm_joint_2)
# R2 limits: [-2.1817, 0.3491]
# raw_action = -0.95
scaled_R2 = 0.5 * (-0.95 + 1.0) * (0.3491 - (-2.1817)) + (-2.1817)
         = 0.5 * 0.05 * 2.5308 + (-2.1817)
         = -2.1184 radians

# After smoothing (assuming previous_target was -1.5708)
smoothed_R2 = 0.1 * (-2.1184) + 0.9 * (-1.5708)
            = -0.2118 + (-1.4137)
            = -1.6255 radians

# Final clamped value
final_R2 = clamp(-1.6255, -2.1817, 0.3491) = -1.6255 radians
```

## 5. IMPORTANT DETAILS

1. **Joint Ordering Mismatch**: The proprioceptive observations use the first 20 joints of the robot (including base joints), while the actions only control the 20 mimicked joints (no base control).

2. **Normalization Asymmetry**: Proprioceptive inputs are normalized, but ground truth inputs are raw radians to preserve accuracy for tracking.

3. **Velocity Limit Scaling**: The deployment script uses `VEL_SCALE = 0.5` to reduce maximum velocities for safety.

4. **Animation Timing**: At 60Hz control rate, each animation frame corresponds to 1/60 second.

5. **Zero Base Actions**: The configuration sets `num_base_actions = 0`, so the mobile base is not controlled by the policy.

6. **Joint Weights for Reward**: Different body parts have different importance (head: 0.5, torso: 2.0, arms: 1.0) in the tracking reward calculation.