#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
ROS deployment script for AIREC mimic policy.

This script loads a trained mimic policy and deploys it on the real AIREC robot
or in Gazebo simulation. The policy expects:
- 20 controlled joints: torso (3), head (3), left arm (7), right arm (7)
- Animation CSV data for target positions
- Both proprioceptive and ground truth observations

Usage:
1. Set checkpoint path:
   export MIMIC_CHECKPOINT_PATH=/path/to/your/checkpoint.pt
   
2. Launch robot/simulation:
   roslaunch your_robot_launch_file
   
3. Run this script:
   python run_RL_fixed.py
"""

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from std_msgs.msg import Float64MultiArray
from torobo_sensor_msgs.msg import ToroboState
import os
import tf
from threading import Lock
import torch
import pandas as pd
import dynamic_reconfigure.client
import gymnasium as gym

from isaaclab_rl.algorithms.policy_value import GaussianPolicy
from isaaclab_rl.models.encoder import AIRECEncoder

device = "cpu"
torch.set_default_dtype(torch.float32)

##### HPARAMS (from elle file)
RL_HZ = 10
ACTION_TAU = 0.1  # IMPORTANT: This should match training env's action_scale
OVERRIDE_VEL_SCALE = 0.1
EPISODE_TIMESTEPS = 900  # Match mimic training episode length

# Joint order and limits from elle file
LOWER_LIMITS = np.array([-0.7854, -0.8727, -1.7453, -1.5708, -1.2217, -1.2217, -0.8727, -2.1817,
        -2.1817, -0.6981, -2.0944, -2.0944, -0.1745, -0.1745, -2.9671, -2.9671,
        -1.5708, -1.5708, -0.1745, -0.1745])

UPPER_LIMITS = np.array([0.7854, 1.8326, 1.7453, 1.5708, 4.1888, 4.1888, 0.8727, 0.3491, 0.3491,
        0.6981, 2.0944, 2.0944, 2.4435, 2.4435, 2.9671, 2.9671, 1.5708, 1.5708,
        1.5708, 1.5708])

VEL_LIMITS = OVERRIDE_VEL_SCALE * np.array([0.8727, 1.5708, 1.5708, 5.5851, 2.6180, 2.6180, 4.7124, 2.6180, 2.6180,
         3.8397, 3.3161, 3.3161, 3.3161, 3.3161, 4.0143, 4.0143, 4.0143, 4.0143,
         4.0143, 4.0143])

# Subscriber message index params (from elle file)
HEAD_START_IDX = 7
LARM_START_IDX = 10
RARM_START_IDX = 27
TORSO_START_IDX = 44

# Reshuffle from subscriber message -> neural network input (from elle file)
index_reshuffle_map = {
    HEAD_START_IDX: 3,
    HEAD_START_IDX+1: 6,
    HEAD_START_IDX+2: 9,
    TORSO_START_IDX: 0,
    TORSO_START_IDX+1: 1,
    TORSO_START_IDX+2: 2,
    LARM_START_IDX: 4,
    LARM_START_IDX+1: 7,
    LARM_START_IDX+2: 10,
    LARM_START_IDX+3: 12,
    LARM_START_IDX+4: 14,
    LARM_START_IDX+5: 16,
    LARM_START_IDX+6: 18,
    RARM_START_IDX: 5,
    RARM_START_IDX+1: 8,
    RARM_START_IDX+2: 11,
    RARM_START_IDX+3: 13,
    RARM_START_IDX+4: 15,
    RARM_START_IDX+5: 17,
    RARM_START_IDX+6: 19
}

# Policy joint order (from elle file)
policy_joint_order = [
    "torso/joint_1",
    "torso/joint_2",
    "torso/joint_3",
    "head/joint_1",
    "left_arm/joint_1",
    "right_arm/joint_1",
    "head/joint_2",
    "left_arm/joint_2",
    "right_arm/joint_2",
    "head/joint_3",
    "left_arm/joint_3",
    "right_arm/joint_3",
    "left_arm/joint_4",
    "right_arm/joint_4",
    "left_arm/joint_5",
    "right_arm/joint_5",
    "left_arm/joint_6",
    "right_arm/joint_6",
    "left_arm/joint_7",
    "right_arm/joint_7"
]

# Animation data loading (mimic specific)
# First try the path from original run_RL.py
animation_file = os.path.join(os.path.dirname(__file__), "../../../assets/animation/walkingsupport.csv")
# If not found, try the absolute path
if not os.path.exists(animation_file):
    animation_file = "/home/d-airec/catkin_ws/isaaclab_rl/deployment/walkingsupport.csv"
    rospy.logwarn(f"Using fallback animation path: {animation_file}")
# Map CSV columns to policy joint order indices
csv_to_policy_idx = {
    "T1": 0, "T2": 1, "T3": 2,      # Torso
    "H1": 3, "H2": 6, "H3": 9,      # Head
    "L1": 4, "L2": 7, "L3": 10,     # Left arm
    "L4": 12, "L5": 14, "L6": 16, "L7": 18,
    "R1": 5, "R2": 8, "R3": 11,     # Right arm
    "R4": 13, "R5": 15, "R6": 17, "R7": 19
}
csv_columns = ["H1", "H2", "H3", "R1", "R2", "R3", "R4", "R5", "R6", "R7",
               "L1", "L2", "L3", "L4", "L5", "L6", "L7", "T1", "T2", "T3"]

animation_data = None
max_animation_steps = 0
current_animation_step = 1  # Start at frame 1 since frame 0 is all zeros

# Load animation data
try:
    if not os.path.exists(animation_file):
        raise FileNotFoundError(f"Animation file not found at: {animation_file}")
    
    animation_df = pd.read_csv(animation_file)
    missing_cols = [col for col in csv_columns if col not in animation_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in CSV: {missing_cols}")
    
    # Load data in CSV column order
    csv_data = animation_df[csv_columns].values
    max_animation_steps = len(csv_data)
    
    # Debug: Check raw CSV data
    print(f"[DEBUG] Raw CSV data shape: {csv_data.shape}")
    print(f"[DEBUG] Raw CSV data range: [{np.min(csv_data):.3f}, {np.max(csv_data):.3f}]")
    print(f"[DEBUG] First row of CSV data (in CSV column order):")
    for i, col in enumerate(csv_columns):
        if csv_data[0, i] != 0:
            print(f"  {col}: {csv_data[0, i]:.4f}")
    
    # Reorder to match policy joint order
    animation_data = np.zeros((max_animation_steps, 20))
    for csv_col, policy_idx in csv_to_policy_idx.items():
        csv_idx = csv_columns.index(csv_col)
        animation_data[:, policy_idx] = csv_data[:, csv_idx]
    
    # Convert to radians if needed
    if np.max(np.abs(animation_data)) > 10:
        print(f"[WARNING] Animation data appears to be in degrees, converting to radians")
        animation_data = np.deg2rad(animation_data)
    
    print(f"[INFO] Loaded animation with {max_animation_steps} steps")
    print(f"[INFO] Animation data range: [{np.min(animation_data):.3f}, {np.max(animation_data):.3f}] rad")
    
    # Debug: Check first few frames of animation
    print(f"[DEBUG] First 5 frames of animation data (after reordering to policy order):")
    for frame in range(min(5, max_animation_steps)):
        print(f"  Frame {frame}:")
        for i, name in enumerate(policy_joint_order):
            if animation_data[frame, i] != 0:
                print(f"    {i:2d} - {name}: {animation_data[frame, i]:.4f} rad ({np.rad2deg(animation_data[frame, i]):.2f} deg)")
    
    # Debug: Check if all frames are zero
    non_zero_frames = np.any(animation_data != 0, axis=1)
    print(f"[DEBUG] Number of non-zero frames: {np.sum(non_zero_frames)} out of {max_animation_steps}")
    if np.sum(non_zero_frames) > 0:
        first_non_zero = np.argmax(non_zero_frames)
        print(f"[DEBUG] First non-zero frame: {first_non_zero}")
    
except Exception as e:
    print(f"[ERROR] Failed to load animation file: {e}")
    print(f"[WARNING] Using zero positions as fallback")
    animation_data = np.zeros((1, 20))
    max_animation_steps = 1

# Global variables
latest_joint_pos = None
latest_joint_vel = None  # RAW velocities for GT observation
latest_joint_vel_norm = None
latest_joint_pos_norm = None
latest_lhand_pos = None
latest_rhand_pos = None

# Lock for thread safety
data_lock = Lock()

# TF listener
tf_listener = None

# Utility functions (from elle file)
def reshuffle_data(data_list, index_mapping_dict):
    """Reshuffles a list of data based on index mapping."""
    if data_list is None or len(data_list) == 0 or not index_mapping_dict:
        return data_list
    
    max_new_index = max(index_mapping_dict.values())
    reshuffled_list = [None] * (max_new_index + 1)
    
    for old_index, new_index in index_mapping_dict.items():
        if 0 <= old_index < len(data_list):
            reshuffled_list[new_index] = data_list[old_index]
    
    return reshuffled_list

def normalise(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def publish_joint_trajectory(publisher, joint_names, positions, time_from_start):
    """Publish joint trajectory command."""
    global debug_first_publish
    
    # Debug: Print what we're publishing on first call
    if 'debug_first_publish' not in globals():
        debug_first_publish = True
    
    if debug_first_publish:
        debug_first_publish = False
        rospy.loginfo("[DEBUG] Publishing joint trajectory with joints:")
        for i, (name, pos) in enumerate(zip(joint_names, positions)):
            rospy.loginfo(f"  {i:2d} - {name}: {pos:.4f} rad ({np.rad2deg(pos):.2f} deg)")
    
    trajectory = JointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * len(joint_names)
    point.accelerations = [0.0] * len(joint_names)
    point.effort = [0.0] * len(joint_names)
    point.time_from_start = rospy.Duration(time_from_start)
    trajectory.points.append(point)
    publisher.publish(trajectory)

# Global debug flags
debug_step_counter = 0

# Callbacks
def prop_callback(data):
    """Callback for joint states."""
    global latest_joint_pos, latest_joint_vel, latest_joint_vel_norm, latest_joint_pos_norm
    
    with data_lock:
        latest_joint_pos = np.array(reshuffle_data(data.link_position, index_reshuffle_map))
        latest_joint_vel = np.array(reshuffle_data(data.link_velocity, index_reshuffle_map))  # Store raw velocities
        latest_joint_pos_norm = normalise(latest_joint_pos, LOWER_LIMITS, UPPER_LIMITS)
        latest_joint_vel_norm = normalise(latest_joint_vel, -VEL_LIMITS, VEL_LIMITS)

def get_hand_transform(hand_link, base_link='/base_link'):
    """Get hand position and orientation relative to base_link.
    
    In simulation, positions are relative to robot base frame.
    """
    global tf_listener
    try:
        # Get transform from base to hand
        (trans, rot) = tf_listener.lookupTransform(base_link, hand_link, rospy.Time(0))
        # Note: In sim, there's a 2cm offset in Z: OffsetCfg(pos=[0.0, 0.0, 0.02])
        # But we'll keep raw transform for now
        # CRITICAL: TF returns quaternion as (x,y,z,w) but training expects (w,x,y,z)
        rot_wxyz = np.array([rot[3], rot[0], rot[1], rot[2]], dtype=np.float32)
        return np.array(trans, dtype=np.float32), rot_wxyz
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn(f"Failed to get transform from {base_link} to {hand_link}")
        # Return identity transform as fallback
        return np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w,x,y,z) identity

def get_proprioception(last_actions):
    """Construct proprioceptive observation for mimic policy.
    
    IMPORTANT: In simulation, this uses the last ACTIONS (normalized -1 to 1),
    not the current targets!
    """
    # Get hand transforms
    lhand_pos, lhand_rot = get_hand_transform('/left_hand/palm_link')
    rhand_pos, rhand_rot = get_hand_transform('/right_hand/palm_link')
    
    # DEBUG: Check if hand positions are reasonable
    if np.linalg.norm(lhand_pos) > 2.0 or np.linalg.norm(rhand_pos) > 2.0:
        rospy.logwarn(f"Hand positions seem too far: L={lhand_pos}, R={rhand_pos}")
    if abs(np.linalg.norm(lhand_rot) - 1.0) > 0.1 or abs(np.linalg.norm(rhand_rot) - 1.0) > 0.1:
        rospy.logwarn(f"Hand quaternions not normalized: L_norm={np.linalg.norm(lhand_rot)}, R_norm={np.linalg.norm(rhand_rot)}")
    
    prop = torch.cat((
        torch.tensor(latest_joint_pos_norm, dtype=torch.float32),
        torch.tensor(latest_joint_vel_norm, dtype=torch.float32),
        torch.tensor(last_actions, dtype=torch.float32),  # Use actions, not targets!
        torch.tensor(lhand_pos, dtype=torch.float32),
        torch.tensor(lhand_rot, dtype=torch.float32),
        torch.tensor(rhand_pos, dtype=torch.float32),
        torch.tensor(rhand_rot, dtype=torch.float32)
    ))
    return prop

def get_gt():
    """Construct ground truth observation with animation targets.
    
    GT observation structure (60D total):
    - indices 0-19: current joint positions (radians, NOT normalized)
    - indices 20-39: current joint velocities (rad/s, NOT normalized) 
    - indices 40-59: target animation positions (radians)
    
    This matches the training environment's _get_gt() method in mimic.py
    """
    global current_animation_step
    
    # Target positions from animation
    target_pos = animation_data[current_animation_step % max_animation_steps]
    
    # IMPORTANT: GT observation uses raw values, not normalized!
    # The training environment concatenates raw joint positions and velocities
    gt = torch.cat((
        torch.tensor(latest_joint_pos, dtype=torch.float32),           # 0-19: current positions (rad)
        torch.tensor(latest_joint_vel, dtype=torch.float32),           # 20-39: current velocities (rad/s) - RAW VALUES!
        torch.tensor(target_pos, dtype=torch.float32)                  # 40-59: animation targets (rad)
    ))
    return gt

def rl_policy_loop():
    """Main policy execution loop."""
    TOPIC_NAME = '/torobo/online_joint_trajectory_controller/command'
    
    torch.set_default_dtype(torch.float32)
    
    global current_animation_step, tf_listener
    
    # Observation space dimensions
    num_prop = 74  # 20 pos + 20 vel + 20 actions + 14 hand states
    num_gt = 60    # 20 current pos + 20 current vel + 20 target pos
    num_actions = 20
    
    observation_space = {
        "prop": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_prop,), dtype=np.float32),
        "gt": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_gt,), dtype=np.float32)
    }
    action_space = gym.spaces.Box(low=-1, high=1, shape=(num_actions,), dtype=np.float32)
    
    # Model configuration (from mimic training)
    encoder_cfg = {
        "layernorm": True,
        "state_preprocessor": "standard",  # Keep same as training
        "hiddens": [1024, 512, 256],
        "activations": ["elu", "elu", "elu"]
    }
    
    policy_cfg = {
        "clip_log_std": True,
        "initial_log_std": 0,
        "min_log_std": -20.0,
        "max_log_std": 2.0,
        "hiddens": [256, 128, 64],
        "activations": ["elu", "elu", "elu", "tanh"]
    }
    
    encoder = AIRECEncoder(observation_space, encoder_cfg, device=device)
    policy = GaussianPolicy(
        z_dim=encoder.num_outputs,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **policy_cfg,
    )
    
    print(encoder)
    print(policy)
    
    # Load checkpoint
    # Note: For sim2real, ensure this checkpoint is from Isaac Lab mimic training
    checkpoint_path = os.path.join(os.path.dirname(__file__), "/home/d-airec/catkin_ws/isaaclab_rl/deployment/best_agent.pt")
    
    if not os.path.exists(checkpoint_path):
        rospy.logerr(f"Checkpoint not found at: {checkpoint_path}")
        rospy.logerr("Please set MIMIC_CHECKPOINT_PATH environment variable")
        return
    
    rospy.loginfo(f"Loading checkpoint from: {checkpoint_path}")
    modules = torch.load(checkpoint_path, map_location=device)
    if type(modules) is dict:
        for name, _ in modules.items():
            print(name)
    encoder.load_state_dict(modules["encoder"])  # This includes state_preprocessor
    policy.load_state_dict(modules["policy"])
    
    # State preprocessor check
    if encoder.state_preprocessor is not None:
        rospy.loginfo("State preprocessor active")
    
    # Initialize ROS node
    rospy.init_node('rl_mimic_policy_node', anonymous=True)
    rate = rospy.Rate(hz=RL_HZ)
    
    # Initialize TF listener
    tf_listener = tf.TransformListener()
    
    # Set velocity overrides (from elle file)
    client = dynamic_reconfigure.client.Client('torobo/online_joint_trajectory_controller/override_params')
    params = {}
    for joint_name in policy_joint_order:
        param_name = joint_name + '_speed_override'
        params[param_name] = OVERRIDE_VEL_SCALE
    client.update_configuration(params)
    print("Updated joint speed overrides")
    
    # Create publisher
    publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
    
    while publisher.get_num_connections() == 0:
        rospy.sleep(1)
    
    # Subscribe to joint states
    rospy.Subscriber("/torobo/torobo_states", ToroboState, prop_callback)
    
    # Wait for initial data
    rospy.loginfo("Waiting for initial joint state...")
    while not rospy.is_shutdown():
        with data_lock:
            if latest_joint_pos_norm is not None:
                break
        rate.sleep()
    
    rospy.loginfo("Initial joint state received. Starting policy loop.")
    
    # Verify TF frames
    rospy.loginfo("Checking for required TF frames...")
    try:
        tf_listener.waitForTransform('/base_link', '/left_hand/palm_link', rospy.Time(), rospy.Duration(5.0))
        tf_listener.waitForTransform('/base_link', '/right_hand/palm_link', rospy.Time(), rospy.Duration(5.0))
        rospy.loginfo("TF frames verified!")
    except tf.Exception as e:
        rospy.logerr(f"Required TF frames not available: {e}")
        return
    
    # Initialize targets from current robot position
    with data_lock:
        if latest_joint_pos is not None:
            # Start from current robot position to avoid jumps
            cur_targets = latest_joint_pos.copy()
            prev_targets = cur_targets.copy()
        else:
            # Fallback to animation frame 1 if no joint data
            initial_animation_targets = animation_data[1]  # Use frame 1 since frame 0 is zeros
            cur_targets = initial_animation_targets.copy()
            prev_targets = cur_targets.copy()
    
    # First, move robot to a reasonable starting pose (not T-pose)
    # Use animation frame 1 which has arms at -90 degrees
    initial_pose = animation_data[1].copy()
    rospy.loginfo("Moving to initial pose (arms down)...")
    rospy.loginfo(f"Initial pose targets: {initial_pose[:5]}... (first 5 joints)")
    rospy.loginfo(f"Initial pose - Left arm: {initial_pose[4]:.3f}, {initial_pose[7]:.3f}, {initial_pose[10]:.3f} rad")
    rospy.loginfo(f"Initial pose - Right arm: {initial_pose[5]:.3f}, {initial_pose[8]:.3f}, {initial_pose[11]:.3f} rad")
    publish_joint_trajectory(
        publisher=publisher,
        joint_names=policy_joint_order,
        positions=list(initial_pose),
        time_from_start=3.0  # Give 3 seconds to reach initial position
    )
    rospy.sleep(3.5)
    
    # Update targets to match where we moved
    with data_lock:
        if latest_joint_pos is not None:
            cur_targets = latest_joint_pos.copy()
            prev_targets = cur_targets.copy()
    
    # Episode management
    episode_steps = 0
    actions_history = []
    targets_history = []
    animation_targets_history = []
    
    # Initialize last actions to zeros (as in simulation reset)
    last_actions = np.zeros(20, dtype=np.float32)
    
    # Debug mode: bypass policy and follow animation directly
    BYPASS_POLICY = True  # Set to True to test animation following without policy
    
    while not rospy.is_shutdown() and episode_steps < EPISODE_TIMESTEPS:
        with data_lock:
            if latest_joint_pos_norm is None:
                rospy.logwarn("Lost joint state data")
                rate.sleep()
                continue
            
            # Get observations
            obs = {
                "prop": get_proprioception(last_actions),
                "gt": get_gt()
            }
        
        # Debug: Log observations before policy
        if episode_steps < 5:
            rospy.loginfo(f"\n[DEBUG] Step {episode_steps} - Before Policy:")
            rospy.loginfo(f"  Current joint pos (normalized): {latest_joint_pos_norm[:5]}...")  # First 5 joints
            rospy.loginfo(f"  Current targets (rad): {cur_targets[:5]}...")
            anim_targets = animation_data[current_animation_step % max_animation_steps]
            rospy.loginfo(f"  Animation targets (rad): {anim_targets[:5]}...")
            rospy.loginfo(f"  GT observation shape: {obs['gt'].shape}")
            rospy.loginfo(f"  GT obs structure:")
            rospy.loginfo(f"    - Current pos (0-19): {obs['gt'][:5].numpy()}...")
            rospy.loginfo(f"    - Current vel (20-39): {obs['gt'][20:25].numpy()}...")
            rospy.loginfo(f"    - Animation targets (40-59): {obs['gt'][40:45].numpy()}...")
            rospy.loginfo(f"    - Animation targets mean: {obs['gt'][40:60].mean().item():.3f}")
        
        # CRITICAL DEBUG: Log observations and outputs periodically
        if episode_steps % 100 == 0 or episode_steps < 3:
            prop_raw = obs["prop"].numpy()
            gt_raw = obs["gt"].numpy()
            rospy.loginfo(f"\n=== Step {episode_steps} DEBUG ===")
            rospy.loginfo(f"Prop obs: min={prop_raw.min():.3f}, max={prop_raw.max():.3f}, mean={prop_raw.mean():.3f}")
            rospy.loginfo(f"GT obs: min={gt_raw.min():.3f}, max={gt_raw.max():.3f}, mean={gt_raw.mean():.3f}")
            # Key components
            rospy.loginfo(f"  Joint pos norm (0-19): [{prop_raw[:20].min():.2f}, {prop_raw[:20].max():.2f}]")
            rospy.loginfo(f"  Joint vel norm (20-39): [{prop_raw[20:40].min():.2f}, {prop_raw[20:40].max():.2f}]")
            rospy.loginfo(f"  Actions (40-59): [{prop_raw[40:60].min():.2f}, {prop_raw[40:60].max():.2f}]")
            rospy.loginfo(f"  Hand poses (60-73): [{prop_raw[60:].min():.2f}, {prop_raw[60:].max():.2f}]")
            rospy.loginfo(f"  GT current vel (20-39): [{gt_raw[20:40].min():.2f}, {gt_raw[20:40].max():.2f}] rad/s")
            rospy.loginfo(f"  GT target pos (40-59): [{gt_raw[40:60].min():.2f}, {gt_raw[40:60].max():.2f}] rad")
        
        # Run policy
        z = encoder(obs)
        
        # Fix encoder output shape if needed
        if z.shape == torch.Size([256, 1]):
            z = z.squeeze(-1).unsqueeze(0)  # Convert [256, 1] to [1, 256]
        
        if BYPASS_POLICY:
            # Bypass policy - create actions to track animation
            anim_targets = animation_data[current_animation_step % max_animation_steps]
            # Simple P-controller: action proportional to error
            errors = anim_targets - latest_joint_pos
            # Need to normalize errors to action space [-1, 1]
            # First normalize errors by joint ranges
            normalized_errors = errors / (UPPER_LIMITS - LOWER_LIMITS)
            # Then apply proportional gain
            actions = np.clip(2.0 * normalized_errors, -1.0, 1.0)  # Higher gain for better tracking
            
            # Enhanced logging for bypass mode
            max_error_idx = np.argmax(np.abs(errors))
            rospy.loginfo(f"[BYPASS] Step {episode_steps}: max error {np.abs(errors).max():.3f} rad ({np.rad2deg(np.abs(errors).max()):.1f} deg) on {policy_joint_order[max_error_idx]}")
            if episode_steps % 50 == 0:
                rospy.loginfo(f"  Current pos: {latest_joint_pos[:5]}...")
                rospy.loginfo(f"  Target pos:  {anim_targets[:5]}...")
                rospy.loginfo(f"  Actions:     {actions[:5]}...")
        else:
            # Normal policy execution
            actions, _, _ = policy.act(z, deterministic=True)
            actions = actions[0].detach().cpu().numpy()  # Get first (and only) environment
        
        # Store actions for next observation
        last_actions = actions.copy()
        
        # Log actions periodically
        if episode_steps % 100 == 0 or episode_steps < 3:
            rospy.loginfo(f"  Actions: min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
            saturated = np.sum(np.abs(actions) > 0.9)
            if saturated > 0:
                rospy.loginfo(f"  WARNING: {saturated} actions saturated at ±0.9")
        
        # Scale actions
        scaled_targets = scale(actions, LOWER_LIMITS, UPPER_LIMITS)
        
        # Apply smoothing (matching training env's act_moving_average = 0.1)
        
        # Apply smoothing (matching elle file's ACTION_TAU)
        cur_targets = ACTION_TAU * scaled_targets + (1 - ACTION_TAU) * prev_targets
        cur_targets = np.clip(cur_targets, LOWER_LIMITS, UPPER_LIMITS)
        
        # Debug smoothing effect
        if first_policy_output or episode_steps < 5:
            rospy.loginfo(f"\n[DEBUG] Step {episode_steps} - After smoothing (tau={ACTION_TAU}):")
            for i in range(min(6, len(cur_targets))):
                rospy.loginfo(f"  {policy_joint_order[i]}: {prev_targets[i]:.3f} -> {cur_targets[i]:.3f} rad (diff: {cur_targets[i]-prev_targets[i]:.4f})")
        
        # Safety checks
        if np.any(np.isnan(cur_targets)):
            rospy.logerr("NaN detected in actions! Using previous targets.")
            cur_targets = prev_targets.copy()
        else:
            # Velocity limiting
            max_joint_vel_change = 0.1
            vel_change = cur_targets - prev_targets
            vel_change = np.clip(vel_change, -max_joint_vel_change, max_joint_vel_change)
            cur_targets = prev_targets + vel_change
            prev_targets = cur_targets.copy()
        
        # Store history for first few steps
        if episode_steps < 10:
            actions_history.append(actions.copy())
            targets_history.append(cur_targets.copy())
            animation_targets_history.append(animation_data[current_animation_step % max_animation_steps].copy())
        
        # Debug: Compare target vs animation
        if first_policy_output or episode_steps < 5:
            rospy.loginfo(f"\n[DEBUG] Step {episode_steps} - Final targets vs Animation:")
            target_pos = animation_data[current_animation_step % max_animation_steps]
            for i in range(min(6, len(cur_targets))):
                diff = cur_targets[i] - target_pos[i]
                rospy.loginfo(f"  {policy_joint_order[i]}: target={cur_targets[i]:.3f}, anim={target_pos[i]:.3f}, diff={diff:.3f} rad ({np.rad2deg(diff):.1f} deg)")
        
        first_policy_output = False
        
        # Publish command
        publish_joint_trajectory(
            publisher=publisher,
            joint_names=policy_joint_order,
            positions=list(cur_targets),
            time_from_start=1/RL_HZ
        )
        
        # Advance animation
        current_animation_step += 1
        if current_animation_step >= max_animation_steps:
            current_animation_step = 1  # Loop back to frame 1, not 0
            rospy.loginfo("Animation looped")
        
        # Progress logging
        episode_steps += 1
        if episode_steps % 100 == 0:
            rospy.loginfo(f"Progress: {episode_steps}/{EPISODE_TIMESTEPS} (animation frame: {current_animation_step})")
        
        rate.sleep()
    
    # Episode complete
    rospy.loginfo(f"Episode complete! Executed {episode_steps} steps.")
    
    # Final summary
    if len(actions_history) > 0:
        actions_array = np.array(actions_history)
        rospy.loginfo(f"\nSummary: Actions range [{actions_array.min():.3f}, {actions_array.max():.3f}]")
        saturated_count = np.sum(np.abs(actions_array) > 0.9)
        if saturated_count > 0:
            rospy.loginfo(f"WARNING: {saturated_count}/{actions_array.size} actions were saturated")
    
    # Return to safe position
    safe_targets = np.zeros(20)
    publish_joint_trajectory(
        publisher=publisher,
        joint_names=policy_joint_order,
        positions=list(safe_targets),
        time_from_start=2.0
    )
    
    rospy.loginfo("RL Mimic Policy Node shutting down.")

if __name__ == '__main__':
    try:
        rl_policy_loop()
    except rospy.ROSInterruptException:
        pass