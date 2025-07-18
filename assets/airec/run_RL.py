#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
import numpy as np
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
import rospy
from std_msgs.msg import Float64MultiArray
from torobo_sensor_msgs.msg import ToroboState
from sensor_msgs.msg import Image
from gpio_msgs.msg import GpioStates
import datetime
import numpy as np
import os
import tf
from threading import Lock # To protect shared data
import torch

from isaaclab_rl.algorithms.policy_value import GaussianPolicy
from isaaclab_rl.models.encoder import AIRECEncoder

device = "cpu"
torch.set_default_dtype(torch.float32)



torso_joints = [
    "torso/joint_1",
    "torso/joint_2",
    "torso/joint_3"
]

larm_joints = [
    "left_arm/joint_1",
    "left_arm/joint_2",
    "left_arm/joint_3",
    "left_arm/joint_4",
    "left_arm/joint_5",
    "left_arm/joint_6",
    "left_arm/joint_7",
]

rarm_joints=[
    "right_arm/joint_1",
    "right_arm/joint_2",
    "right_arm/joint_3",
    "right_arm/joint_4",
    "right_arm/joint_5",
    "right_arm/joint_6",
    "right_arm/joint_7",
]
torso_lower_limits = np.array([-0.7854, -0.8727, -1.7453])
torso_upper_limits = np.array([0.7854, 1.8326, 1.7453])

arm_lower_limits = np.array([-1.2217, -2.1817, -2.0944, -0.1745, -2.9671, -1.5708, -0.1745])
arm_upper_limits = np.array([4.1888, 0.3491, 2.0944, 2.4435, 2.9671, 1.5708, 1.5708])

actuated_joints = torso_joints + rarm_joints
actuated_upper = np.concatenate((torso_upper_limits, arm_upper_limits))
actuated_lower = np.concatenate((torso_lower_limits, arm_lower_limits))

# Global variable to store the latest joint positions
# This needs to be accessible by both the callback and the main loop.
latest_joint_positions = None
latest_larm_joint_pos = None
latest_rarm_joint_pos = None
latest_torso_joint_pos = None
latest_larm_joint_vel = None
latest_rarm_joint_vel = None
latest_torso_joint_vel = None
latest_torso_joint_pos_norm = None
latest_larm_joint_pos_norm = None
latest_rarm_joint_pos_norm = None

cur_targets = np.zeros((len(actuated_joints),))
prev_targets = np.zeros((len(actuated_joints),))

# A lock to ensure thread-safe access to latest_joint_positions
# The callback runs in a separate thread from the main loop.
data_lock = Lock()

larm_start_idx = 10
rarm_start_idx = 27
torso_start_idx = 44


def format(data):
    return [f"{pos:.4f}" for pos in data]

def normalise(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def touch_sensor_callback2(data):
    global tactile
    tac = data.data
    tactile = tuple(tac[10:86])
    # print(tactile[48:55])

def AngleSensorCallback2(data):
    """
    Callback function for the /torobo/torobo_states topic.
    This function is called every time a new message is received.
    It updates the global latest_joint_positions variable.
    """
    global latest_torso_joint_pos
    global latest_larm_joint_pos
    global latest_rarm_joint_pos
    global latest_torso_joint_vel
    global latest_larm_joint_vel
    global latest_rarm_joint_vel

    global latest_torso_joint_pos_norm
    global latest_larm_joint_pos_norm
    global latest_rarm_joint_pos_norm

    with data_lock:
        joint_pos = data.link_position
        joint_vel = data.link_velocity

        latest_torso_joint_pos = np.array(joint_pos[torso_start_idx:torso_start_idx+3])
        latest_larm_joint_pos = np.array(joint_pos[larm_start_idx:larm_start_idx+7])
        latest_rarm_joint_pos = np.array(joint_pos[rarm_start_idx:rarm_start_idx+7])

        latest_torso_joint_pos_norm = normalise(latest_torso_joint_pos, torso_lower_limits, torso_upper_limits)
        latest_larm_joint_pos_norm = normalise(latest_larm_joint_pos, arm_lower_limits, arm_upper_limits)
        latest_rarm_joint_pos_norm = normalise(latest_rarm_joint_pos, arm_lower_limits, arm_upper_limits)

        latest_torso_joint_vel = np.array(joint_vel[torso_start_idx:torso_start_idx+3])
        latest_larm_joint_vel = np.array(joint_vel[larm_start_idx:larm_start_idx+7])
        latest_rarm_joint_vel = np.array(joint_vel[rarm_start_idx:rarm_start_idx+7])

        # latest_torso_joint_pos_d = np.rad2deg(latest_torso_joint_pos)
        # latest_larm_joint_pos_d = np.rad2deg(latest_larm_joint_pos)
        # latest_rarm_joint_pos_d = np.rad2deg(latest_rarm_joint_pos)

        # latest_torso_joint_vel_d = np.rad2deg(latest_torso_joint_vel)
        # latest_larm_joint_vel_d = np.rad2deg(latest_larm_joint_vel)
        # latest_rarm_joint_vel_d = np.rad2deg(latest_rarm_joint_vel)

        # rospy.loginfo(f"Callback: latest_torso_joint_pos_d: {format(latest_torso_joint_pos_d)}")
        # rospy.loginfo(f"Callback: latest_larm_joint_pos_d: {format(latest_larm_joint_pos_d)}")
        # rospy.loginfo(f"Callback: latest_rarm_joint_pos_d: {format(latest_rarm_joint_pos_d)}")
        # rospy.loginfo(f"Callback: latest_torso_joint_vel_d: {format(latest_torso_joint_vel_d)}")
        # rospy.loginfo(f"Callback: latest_larm_joint_vel_d: {format(latest_larm_joint_vel_d)}")
        # rospy.loginfo(f"Callback: latest_rarm_joint_vel_d: {format(latest_rarm_joint_vel_d)}")
        # print("*******")

def get_proprioception():
    normalised_joint_pos = np.concatenate((latest_torso_joint_pos_norm, latest_rarm_joint_pos_norm))
    joint_vel = np.concatenate((latest_torso_joint_vel, latest_rarm_joint_vel))

    print(np.shape(normalised_joint_pos), np.shape(cur_targets))

    prop = torch.cat((torch.tensor(normalised_joint_pos), torch.tensor(joint_vel), torch.tensor(cur_targets), torch.tensor(prev_targets)))
    return prop

def rl_policy_loop():

    TOPIC_NAME = '/torobo/joint_trajectory_controller/command'

    torch.set_default_dtype(torch.float32)

    cur_targets = np.zeros((len(actuated_joints),))
    prev_targets = np.zeros((len(actuated_joints),))

    num_prop = 40
    num_gt = 0
    num_tactile = 0
    num_actions = 10

    # load policy
    observation_space = {
        "prop": np.zeros(num_prop),
        # "gt": np.zeros(num_gt),
        # "tactile": np.zeros(num_tactile)
    }
    action_space = np.zeros(num_actions)

    encoder_cfg = {
        "layernorm": True,
        "state_preprocessor": None,
        "hiddens": [512, 256],
        "activations": ["elu", "elu", "elu"]
    }

    policy_cfg = {
        "clip_log_std": True,
        "initial_log_std": 0,
        "min_log_std": -20.0,
        "max_log_std": 2.0,
        "hiddens": [128, 64],
        "activations": ["elu", "elu", "identity"]
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

    path = "/home/elle/catkin_ws/IsaacLab-main/logs/airec/test/2025-07-18_12-33-41/checkpoints/best_agent.pt"

    modules = torch.load(path, map_location=device)
    if type(modules) is dict:
        for name, data in modules.items():
            print(name)

    # Load pre-trained weights (uncomment and modify paths as needed)
    encoder.load_state_dict(modules["encoder"])
    # encoder.state_preprocessor.load_state_dict(modules["state_preprocessor"])
    encoder = encoder.to(device)
    # encoder.state_preprocessor = encoder.state_preprocessor.to(device)
    policy.load_state_dict(modules["policy"])


    # Initializes a rospy node.
    rospy.init_node('rl_policy_node', anonymous=True)

    rl_hz = 10

    rate = rospy.Rate(rl_hz) # 10 Hz

    # Create a publisher.
    publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)

    while publisher.get_num_connections() == 0:
        rospy.sleep(1)

    # rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/torobo/torobo_states",ToroboState, AngleSensorCallback2)
    rospy.Subscriber("/torobo/gpio_state_controller/analog_io",Float64MultiArray,touch_sensor_callback2)

    rospy.loginfo("RL Policy Node started. Waiting for first joint state message...")

    # Wait until the first message has been received and data is available
    # This prevents the policy from trying to use None as an observation.
    while not rospy.is_shutdown():
        with data_lock:
            if latest_torso_joint_pos_norm is not None:
                break
        rospy.loginfo("Waiting for initial joint state data...")
        rate.sleep()

    rospy.loginfo("Initial joint state received. Starting policy loop.")

    while not rospy.is_shutdown():
        # 1. Get the current observation (joint positions)
        with data_lock:
            if latest_torso_joint_pos_norm is not None:
                
                obs = {
                    "prop": get_proprioception().to(dtype=torch.float32)
                }
            else:
                # This should ideally not happen after the initial wait, but good for robustness
                rospy.logwarn("No joint position data available for policy. Skipping this iteration.")
                rate.sleep()
                continue

        # 2. Use the observation in your RL policy
        # This is where your actual RL algorithm would go.
        # For demonstration, we'll just print the observation and simulate an action.
        rospy.loginfo(f"Policy Loop: obs: {obs['prop'].size()}")

        # --- Placeholder for your RL Policy Logic ---
        z = encoder(obs).T
        actions = policy.act(z, deterministic=True)[0][0]
        actions = actions.detach().cpu().numpy()
        cur_targets = scale(actions, actuated_lower, actuated_upper)

        act_moving_average = 0.1
        cur_targets = act_moving_average * cur_targets + (1-act_moving_average) * prev_targets
        cur_targets = np.clip(cur_targets, actuated_lower, actuated_upper)
        prev_targets = cur_targets

        rospy.loginfo(f"Policy Loop: Simulated Action: {np.rad2deg(cur_targets)}")

        # 3. Apply the action (e.g., publish to a robot command topic)
        # print(actuated_joints)
        publish_joint_trajectory(
                publisher = publisher,
                joint_names = actuated_joints,
                positions = list(cur_targets),
                time_from_start = 1 / rl_hz 
        )
        # 4. Sleep to maintain the desired loop rate
        rate.sleep()
    
    rospy.loginfo("RL Policy Node shutting down.")


def publish_joint_trajectory(publisher, joint_names, positions, time_from_start):
    """
    Function for publishing message to move joints

    Parameters
    ----------
    publisher : rospy.Publisher
        publisher
    joint_names : list of string
        list of joint names
    positions : list of float
        list of joint's goal positions(radian)
    time_from_start : float
        transition time from start

    Returns
    -------
    None

    Throws
    ------
    None
    """

    # Creates a message.
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

    # Publish the message.
    publisher.publish(trajectory)


if __name__ == '__main__':
    try:
        rl_policy_loop()
    except rospy.ROSInterruptException:
        pass
