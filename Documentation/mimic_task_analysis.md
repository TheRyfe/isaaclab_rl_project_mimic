# AIREC Mimic RL Task Analysis

## Overview
The AIREC Mimic RL task is a reinforcement learning environment designed to train the AIREC humanoid robot to mimic pre-recorded motion animations. The system is built on top of NVIDIA Isaac Lab and uses GPU-accelerated physics simulation.

## Architecture

### 1. Environment Hierarchy
- **AIRECEnv** (base class in `airec.py`): Provides core robot simulation functionality
- **MimicEnv** (child class in `mimic.py`): Extends AIRECEnv with motion mimicry capabilities

### 2. Key Components

#### Robot Configuration
- **AIREC Robot**: Humanoid robot with 47 joints loaded from USD file
- **Controlled Joints**: 20 joints are controlled by the RL agent:
  - Head joints (3): H1, H2, H3
  - Torso joints (3): T1, T2, T3
  - Right arm joints (7): R1-R7
  - Left arm joints (7): L1-L7
- **Control Mode**: Position control with action smoothing (moving average = 0.1)

#### Animation System
- **Animation Data**: CSV file containing joint positions over time
- **Format**: Columns for timestamp and 20 joint positions (H1-H3, R1-R7, L1-L7, T1-T3)
- **Timestep**: 60 Hz (0.0167s per frame)
- **Episode Length**: Fixed at 900 steps (15 seconds)

#### Ghost Robot Visualization
- A kinematic "ghost" robot shows the target pose from the animation
- Appears in red color to distinguish from the actual robot
- Helps visualize the desired motion during training

### 3. Observation Space

The observation space includes three types of data:

1. **Ground Truth (gt)**: 60 dimensions
   - Current joint positions (20)
   - Current joint velocities (20) 
   - Target joint positions from animation (20)

2. **Proprioceptive (prop)**: Variable dimensions including:
   - Normalized joint positions
   - Joint velocities
   - Previous actions
   - End-effector poses (position + quaternion for each hand)

### 4. Action Space
- **Dimensions**: 20 (one per controlled joint)
- **Range**: [-1, 1] normalized, scaled to joint limits
- **Processing**: Actions are processed with tanh for safety, then scaled to joint position limits

### 5. Reward Structure

The reward function includes multiple components:

#### Positive Rewards
1. **Joint Position Tracking** (scale: 3.0)
   - Exponential reward based on squared error between current and target positions
   - Optional joint-specific weights (head: 0.5, torso: 2.0, arms: 1.0)
   - Power scaling for increased sensitivity near perfect tracking
   - Optional normalization based on joint limits

2. **Link Tracking** (pos: 4.0, ori: 2.0)
   - Tracks specific body parts (default: both arm end-effectors)
   - Position and orientation matching with ghost robot

3. **Staying Alive** (scale: 0.005)
   - Small constant reward for maintaining the episode

#### Penalties
1. **Joint Velocity Penalty** (scale: -0.001)
   - Discourages excessive joint speeds

2. **Action Smoothness Penalty** (scale: -0.01)
   - Penalizes large changes between consecutive actions

3. **Joint Acceleration Penalty** (scale: -0.01)
   - Discourages jerky movements

### 6. External Disturbances

The environment can apply random external forces to test robustness:
- **Target**: Configurable body part (default: right arm link 5)
- **Force Range**: 30-150 N
- **Duration**: 0.5-1.5 seconds
- **Interval**: 0.5-3.0 seconds between disturbances
- **Directional Bias**: Can be configured (default: downward)
- **Visualization**: Red arrows show force direction and magnitude

### 7. Termination Conditions

Episodes can terminate based on:
1. **Time Limit**: Fixed episode length (900 steps)
2. **Joint Limits** (optional): If joints exceed 95% of soft limits
3. **Torso Tilt** (optional): If torso tilts beyond threshold (0.4 rad)

### 8. Deployment

The `deployment/run_RL.py` script shows how to deploy trained policies on the real robot:
- Uses ROS for robot communication
- Subscribes to joint state topics
- Publishes joint trajectory commands
- Runs at 10 Hz control frequency

## Training Objective

The primary goal is to train a policy that makes the AIREC robot accurately track the reference animation while:
1. Maintaining smooth, natural movements
2. Being robust to external disturbances
3. Respecting joint limits and physical constraints
4. Minimizing energy consumption

The combination of position tracking rewards, smoothness penalties, and external disturbances creates a challenging learning problem that requires the agent to balance accuracy with stability and efficiency.