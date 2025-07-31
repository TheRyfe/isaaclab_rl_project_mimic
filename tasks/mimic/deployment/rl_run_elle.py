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
import dynamic_reconfigure.client

from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image, CameraInfo
from apriltag_ros.msg import AprilTagDetectionArray
from cv_bridge import CvBridge

# import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf.transformations import quaternion_multiply, quaternion_matrix
import tf.transformations as tf_trans

import argparse

import warnings

# 特定の警告のみを無視
warnings.filterwarnings("ignore")

from isaaclab_rl.algorithms.policy_value import GaussianPolicy
from isaaclab_rl.models.encoder import AIRECEncoder

# from rl_utils import reshuffle_data, normalise, scale, publish_joint_trajectory
# from rl_utils import index_reshuffle_map, policy_joint_order, LOWER_LIMITS, UPPER_LIMITS

device = "cpu"
torch.set_default_dtype(torch.float32)

##### HPARAMS
RL_HZ = 10
ACTION_TAU = 0.01
OVERRIDE_VEL_SCALE = 0.1
EPISODE_TIMESTEPS = 100
RESET_GOAL_SCALE = 0.1

# configure end effector frames
BASE_LINK = '/base_link'
LHAND_EE = "/left_hand_first_finger/link_2"
RHAND_EE = "/right_hand_first_finger/link_2"
EE_OFFSET = [0, -0.1, 0]

default_x = 0.6
default_y = 0.3
default_z = 1.4
default_lhand_goal = np.array([default_x, default_y, default_z])
default_rhand_goal = np.array([default_x, -default_y, default_z])


smooth_both = "logs/airec/test/2025-07-21_16-30-53/checkpoints/best_agent.pt"
vel_lim_1 = "logs/airec/test/2025-07-22_13-19-44/checkpoints/best_agent.pt"
finger = "logs/airec/test/2025-07-23_10-54-45/checkpoints/best_agent.pt"
correct_limits =  "logs/airec/test/2025-07-23_12-34-34/checkpoints/best_agent.pt"
norm_vel = "logs/airec/test/2025-07-23_14-37-02/checkpoints/best_agent.pt"
new = "logs/airec/test/2025-07-23_15-01-01/checkpoints/best_agent.pt"
latest
simon="/home/d-airec/Downloads/best_agent_simon.pt"

CHECKPOINT_PATH = os.path(simon)

# t1,t2,t3,h1,l1,r1,h2...
default_joint_pos_policy_order = np.array([-0.3491,  0.6981,0,
                                           0, 1.0472,  1.0472,
                                           0, -0.8727, -0.8727, 
                                           0, -0.1745, -0.1745, 
                                           1.2217,  1.2217, 
                                           0.6981,  0.6981, 
                                           0.3491,  0.3491, 
                                           0.0873,  0.0873,])

actuated_dof = range(20)

# UTILS
# policy order - t1,t2,t3,h1,l1,r1
LOWER_LIMITS = np.array([-0.7854, -0.8727, -1.7453, -1.5708, -1.2217, -1.2217, -0.8727, -2.1817,
        -2.1817, -0.6981, -2.0944, -2.0944, -0.1745, -0.1745, -2.9671, -2.9671,
        -1.5708, -1.5708, -0.1745, -0.1745])

UPPER_LIMITS = np.array([0.7854, 1.8326, 1.7453, 1.5708, 4.1888, 4.1888, 0.8727, 0.3491, 0.3491,
        0.6981, 2.0944, 2.0944, 2.4435, 2.4435, 2.9671, 2.9671, 1.5708, 1.5708,
        1.5708, 1.5708])

VEL_LIMITS = OVERRIDE_VEL_SCALE * np.array([0.8727, 1.5708, 1.5708, 5.5851, 2.6180, 2.6180, 4.7124, 2.6180, 2.6180,
         3.8397, 3.3161, 3.3161, 3.3161, 3.3161, 4.0143, 4.0143, 4.0143, 4.0143,
         4.0143, 4.0143])

# subscriber message index params
HEAD_START_IDX = 7
LARM_START_IDX = 10
RARM_START_IDX = 27
TORSO_START_IDX = 44
LHAND_START_IDX = 17
RHAND_START_IDX = 34

# reshuffle from subscriber message -> neural network input
index_reshuffle_map = {
    HEAD_START_IDX: 3,   # head_joint_1 (old index 3) moves to new index 0
    HEAD_START_IDX+1: 6,   # head_joint_2 (old index 6) moves to new index 1
    HEAD_START_IDX+2: 9,   # head_joint_3 (old index 9) moves to new index 2
    TORSO_START_IDX: 0,   # torso_joint_1 (old index 0) moves to new index 3
    TORSO_START_IDX+1: 1,   # torso_joint_2 (old index 1) moves to new index 4
    TORSO_START_IDX+2: 2,   # torso_joint_3 (old index 2) moves to new index 5
    LARM_START_IDX: 4,   # left_arm_joint_1 (old index 4) moves to new index 6
    LARM_START_IDX+1: 7,   # left_arm_joint_2 (old index 7) stays at new index 7
    LARM_START_IDX+2: 10,  # left_arm_joint_3 (old index 10) moves to new index 8
    LARM_START_IDX+3: 12,  # left_arm_joint_4 (old index 12) moves to new index 9
    LARM_START_IDX+4: 14, # left_arm_joint_5 (old index 14) moves to new index 10
    LARM_START_IDX+5: 16, # left_arm_joint_6 (old index 16) moves to new index 11
    LARM_START_IDX+6: 18, # left_arm_joint_7 (old index 18) moves to new index 12
    RARM_START_IDX: 5,  # right_arm_joint_1 (old index  ``5) moves to new index 13
    RARM_START_IDX+1: 8,  # right_arm_joint_2 (old index 8) moves to new index 14
    RARM_START_IDX+2: 11, # right_arm_joint_3 (old index 11) moves to new index 15
    RARM_START_IDX+3: 13, # right_arm_joint_4 (old index 13) moves to new index 16
    RARM_START_IDX+4: 15, # right_arm_joint_5 (old index 15) moves to new index 17
    RARM_START_IDX+5: 17, # right_arm_joint_6 (old index 17) moves to new index 18
    RARM_START_IDX+6: 19   # right_arm_joint_7 (old index 19) stays at new index 19
}

# Let's say your original data (e.g., joint names) is ordered by the original indexes 0-19
# We'll use a simplified example for clarity, but it could be any data
subscriber_joint_order = [

    "e1", "e2", "e3", "e4", "e5", "e6", "e7",
   
    "head_joint_1_data",   # Index 3
    "head_joint_2_data",   # Index 6
    "head_joint_3_data",   # Index 9

    "left_arm_joint_1_data", # Index 4
    "left_arm_joint_2_data", # Index 7
    "left_arm_joint_3_data", # Index 10
    "left_arm_joint_4_data", # Index 12
    "left_arm_joint_5_data", # Index 14
    "left_arm_joint_6_data", # Index 16
    "left_arm_joint_7_data", # Index 18

    "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10",

    "right_arm_joint_1_data",# Index 5
    "right_arm_joint_2_data",# Index 8
    "right_arm_joint_3_data",# Index 11
    "right_arm_joint_4_data",# Index 13
    "right_arm_joint_5_data",# Index 15
    "right_arm_joint_6_data",# Index 17
    "right_arm_joint_7_data", # Index 19

    "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10",

    "torso_joint_1_data", 
    "torso_joint_2_data",  
    "torso_joint_3_data", 
]

policy_joint_order = [
    "torso/joint_1",  # Index 0
    "torso/joint_2",  # Index 1
    "torso/joint_3",  # Index 2
    "head/joint_1",   # Index 3
    "left_arm/joint_1", # Index 4
    "right_arm/joint_1",# Index 5
    "head/joint_2",   # Index 6
    "left_arm/joint_2", # Index 7
    "right_arm/joint_2",# Index 8
    "head/joint_3",   # Index 9
    "left_arm/joint_3", # Index 10
    "right_arm/joint_3",# Index 11
    "left_arm/joint_4", # Index 12
    "right_arm/joint_4",# Index 13
    "left_arm/joint_5", # Index 14
    "right_arm/joint_5",# Index 15
    "left_arm/joint_6", # Index 16
    "right_arm/joint_6",# Index 17
    "left_arm/joint_7", # Index 18
    "right_arm/joint_7" # Index 19
]




class AprilDetector:
    def __init__(self):
                
        # airec realsense
        camera_info_topic_name = "/torobo/head/sr300/camera/color/camera_info"

        # local realsense
        # camera_info_topic_name = "/camera/color/camera_info"


        self.image_inf_sub = rospy.Subscriber(camera_info_topic_name, CameraInfo, self.image_info_callback)
        self.camera_matrix = None
        self.dist_coeffs = None
        
        self.pose_pub = rospy.Publisher("/world_pose", PoseStamped, queue_size=10)
        
        self.tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_callback)
        self.bridge = CvBridge()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # camera frame
        self.cam_link = "head/sr300/camera_color_optical_frame"

        self.latest_lhand_goal = None
        self.latest_rhand_goal = None

    def transform_cam2world(self, pos, quat, cam_link):
        """
        カメラ座標系の位置と回転をワールド座標系に変換する
        :param pos: [x, y, z] カメラ座標系の位置
        :param quat: [qx, qy, qz, qw] カメラ座標系のクォータニオン
        :return: ワールド座標 (pos, quat)
        """
        
        # ワールド座標系 ("world") からカメラ座標系 ("camera_link") への変換を取得
        self.trans = self.tf_buffer.lookup_transform("world", cam_link, rospy.Time(0), rospy.Duration(0))

        # 平行移動 (カメラ → ワールド)
        self.T = np.array([
            [1, 0, 0, self.trans.transform.translation.x],
            [0, 1, 0, self.trans.transform.translation.y],
            [0, 0, 1, self.trans.transform.translation.z],
            [0, 0, 0, 1]
        ])

        # 回転行列を取得
        self.q_trans = [
            self.trans.transform.rotation.x,
            self.trans.transform.rotation.y,
            self.trans.transform.rotation.z,
            self.trans.transform.rotation.w
        ]
        self.R = quaternion_matrix(self.q_trans)[:3, :3]  # 3x3の回転行列
        
        
        try:
            # カメラ座標をワールド座標に変換
            pos_camera = np.array([[pos[0]], [pos[1]], [pos[2]], [1]])
            pos_world = self.T @ np.vstack((self.R @ pos_camera[:3], [1]))  # ワールド座標系の位置

            # 回転（クォータニオン）を変換
            q_camera = np.array(quat)
            q_world = quaternion_multiply(self.q_trans, q_camera)

            world_pos = pos_world[:3].flatten().tolist()
            world_quat = q_world.tolist()

            return world_pos, world_quat

        except tf2_ros.LookupException:
            rospy.logerr("TF transform lookup failed")
        except tf2_ros.ConnectivityException:
            rospy.logerr("TF connectivity error")
        except tf2_ros.ExtrapolationException:
            rospy.logerr("TF extrapolation error")

        return None, None


    def tag_callback(self, msg):
        if msg.detections:
            for detection in msg.detections:
                # print(detection.id)
                pose = detection.pose.pose.pose
                position = pose.position
                orientation = pose.orientation
                tag_center_pos = np.array([position.x, position.y, position.z])
                tag_center_quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
            
                world_pos, world_quat = self.transform_cam2world(tag_center_pos, tag_center_quat, self.cam_link)
                r = R.from_quat(world_quat)
                # world_euler = r.as_euler('xyz', degrees=True)

                if detection.id[0] == 0: 
                    self.latest_lhand_goal = world_pos
                    # print("Left hand goal detected")
                elif detection.id[0] == 1:
                    self.latest_rhand_goal = world_pos
                    # print("Right hand goal detected")

                if world_pos and world_quat:
                    # `PoseStamped` メッセージを作成して publish
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = rospy.Time.now()
                    pose_msg.header.frame_id = "world"
                    pose_msg.pose.position.x = world_pos[0]
                    pose_msg.pose.position.y = world_pos[1]
                    pose_msg.pose.position.z = world_pos[2]
                    pose_msg.pose.orientation.x = world_quat[0]
                    pose_msg.pose.orientation.y = world_quat[1]
                    pose_msg.pose.orientation.z = world_quat[2]
                    pose_msg.pose.orientation.w = world_quat[3]

                    self.pose_pub.publish(pose_msg)
                    # rospy.loginfo(f"Published Pose: Position {world_pos}, Orientation {world_quat}")

    
    def image_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)


### FUNCTIONS

def reshuffle_data(data_list, index_mapping_dict):
    """
    Reshuffles a list of data based on a dictionary that maps old indexes to new indexes.

    Args:
        data_list (list): The list of data (e.g., joint names, joint positions)
                          currently ordered by the 'old' indexes.
        index_mapping_dict (dict): A dictionary where keys are the 'old' indexes (int)
                                   and values are the 'new' indexes (int).

    Returns:
        list: A new list with the data reshuffled according to the 'new' indexes.
              The length of the returned list will be based on the maximum 'new' index.
              If an 'old' index is in the mapping but not present in data_list
              (e.g., index out of bounds for data_list), it will be skipped.
              If a 'new' index does not have a corresponding 'old' index in the
              mapping, that position in the output list will be filled with None.
    """
    if not data_list:
        return []
    if not index_mapping_dict:
        return list(data_list) # No mapping, return original

    # Determine the maximum 'new' index to set the size of the new list
    max_new_index = -1
    for new_idx in index_mapping_dict.values():
        if new_idx > max_new_index:
            max_new_index = new_idx

    # Initialize the new list with None values up to the maximum new index
    # We add 1 because indexes are 0-based
    reshuffled_list = [None] * (max_new_index + 1)

    # Populate the new list
    for old_index, new_index in index_mapping_dict.items():
        if 0 <= old_index < len(data_list):
            # Place the item from the old_index position into the new_index position
            reshuffled_list[new_index] = data_list[old_index]
        else:
            print(f"Warning: Old index {old_index} from mapping is out of bounds for data_list.")

    return reshuffled_list



def publish_joint_trajectory(publisher, joint_names, positions, time_from_start):
    """
    Function for publishing message to move joints
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


def format(data):
    return [f"{pos:.1f}" for pos in data]

def normalise(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


# Global variable to store the latest joint positions
# This needs to be accessible by both the callback and the main loop.
latest_joint_pos = None
latest_joint_vel_norm = None
latest_joint_pos_norm = None
latest_lhand_pos = None
latest_rhand_pos = None
lhand_lgoal_distance = None
rhand_rgoal_distance = None
lhand_lgoal_euclidean_distance = None
rhand_rgoal_euclidean_distance = None



# A lock to ensure thread-safe access to latest_joint_positions
# The callback runs in a separate thread from the main loop.
data_lock = Lock()



def prop_callback(data):
    """
    Callback function for the /torobo/torobo_states topic.
    """
    global latest_joint_pos
    global latest_joint_vel_norm
    global latest_joint_pos_norm

    with data_lock:
        latest_joint_pos = np.array(reshuffle_data(data.link_position, index_reshuffle_map))
        joint_vel = np.array(reshuffle_data(data.link_velocity, index_reshuffle_map))
        latest_joint_pos_norm = normalise(latest_joint_pos, LOWER_LIMITS, UPPER_LIMITS)
        latest_joint_vel_norm = normalise(joint_vel, -VEL_LIMITS, VEL_LIMITS)


def tactile_callback(data):
    global tactile
    tac = data.data
    tactile = tuple(tac[10:86])

def get_proprioception(cur_targets, prev_targets):

    prop = torch.cat((
        torch.tensor(latest_joint_pos_norm), 
        torch.tensor(latest_joint_vel_norm), 
        torch.tensor(cur_targets), 
        torch.tensor(prev_targets),
        # torch.tensor(latest_lhand_pos),
        # torch.tensor(latest_rhand_pos)
        )
    )
    return prop

def get_gt(lhand_pos, rhand_pos, lhand_goal, rhand_goal):

    # goal_offset_left = [-0.1, 0, 0.1]
    # goal_offset_right =[0, -0.1, 0.1]

    lhand_lgoal_distance = lhand_pos - lhand_goal 
    rhand_rgoal_distance = rhand_pos - rhand_goal 
    lhand_lgoal_euclidean_distance = np.linalg.norm(lhand_lgoal_distance)
    rhand_rgoal_euclidean_distance = np.linalg.norm(rhand_rgoal_distance)
    euclidean_distances = np.array([lhand_lgoal_euclidean_distance, rhand_rgoal_euclidean_distance])
    print(lhand_lgoal_distance)
    print(rhand_rgoal_distance)
    print(euclidean_distances)

    gt = torch.cat(
        (
            # xyz diffs (3,)
            torch.tensor(lhand_lgoal_distance),
            torch.tensor(rhand_rgoal_distance),
            # euclidean
            torch.tensor(euclidean_distances),
        )
    )
    return gt

def rl_policy_loop():

    TOPIC_NAME = '/torobo/online_joint_trajectory_controller/command'

    torch.set_default_dtype(torch.float32)


    # 20 actuated joints x 4 (pos, vel, target, last target) + 2 x xyz hand pos
    num_prop = 80 #+ 6

    # xyz and euclidean dist for each hand
    num_gt = 8

    # unused at the moment
    num_tactile = 0

    # head, torso, arms
    num_actions = 20

    # load policy
    observation_space = {
        "prop": np.zeros(num_prop),
        "gt": np.zeros(num_gt),
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

    # Load pre-trained weights (uncomment and modify paths as needed)
    modules = torch.load(CHECKPOINT_PATH, map_location=device)
    if type(modules) is dict:
        for name, data in modules.items():
            print(name)
    encoder.load_state_dict(modules["encoder"])
    encoder = encoder.to(device)
    policy.load_state_dict(modules["policy"])


    # Initializes a rospy node.
    rospy.init_node('rl_policy_node', anonymous=True)
    rate = rospy.Rate(hz=RL_HZ)

    client = dynamic_reconfigure.client.Client('torobo/online_joint_trajectory_controller/override_params')
    h1 = 'head/joint_1_speed_override'
    h2 = 'head/joint_2_speed_override'
    h3 = 'head/joint_3_speed_override'
    r1 = 'right_arm/joint_1_speed_override'
    r2 = 'right_arm/joint_2_speed_override'
    r3 = 'right_arm/joint_3_speed_override'
    r4 = 'right_arm/joint_4_speed_override'
    r5 = 'right_arm/joint_5_speed_override'
    r6 = 'right_arm/joint_6_speed_override'
    r7 = 'right_arm/joint_7_speed_override'
    l1 = 'left_arm/joint_1_speed_override'
    l2 = 'left_arm/joint_2_speed_override'
    l3 = 'left_arm/joint_3_speed_override'
    l4 = 'left_arm/joint_4_speed_override'
    l5 = 'left_arm/joint_5_speed_override'
    l6 = 'left_arm/joint_6_speed_override'
    l7 = 'left_arm/joint_7_speed_override'
    t1 = 'torso/joint_1_speed_override'
    t2 = 'torso/joint_2_speed_override'
    t3 = 'torso/joint_3_speed_override'
    
    max_speed = OVERRIDE_VEL_SCALE
    params = {
        h1 : max_speed, h2 : max_speed, h3 : max_speed,
        r1 : max_speed, r2 : max_speed, r3 : max_speed, r4 : max_speed, r5 : max_speed, r6 : max_speed, r7 : max_speed,
        l1 : max_speed, l2 : max_speed, l3 : max_speed, l4 : max_speed, l5 : max_speed, l6 : max_speed, l7 : max_speed,
        t1 : max_speed, t2 : max_speed, t3 : max_speed
    }
    client.update_configuration(params)
    print("Updated joint speed overrides to:", params)

    # Create a publisher.
    publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)

    while publisher.get_num_connections() == 0:
        rospy.sleep(1)

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()
    parent_frame = 'base_link'
    lhand_goal_frame = "aa_lhand_goal"
    rhand_goal_frame = "aa_rhand_goal"

    wake_time = 5
    print("Moving to default position")
    publish_joint_trajectory(
        publisher = publisher,
        joint_names = policy_joint_order,
        positions = list(default_joint_pos_policy_order),
        time_from_start = wake_time
    )
    rospy.sleep(wake_time)


    rospy.loginfo("Creating april tag detector")
    april_detector = AprilDetector()
    # Wait until the first message has been received and data is available
    # This prevents the policy from trying to use None as an observation.
    while not rospy.is_shutdown():
        with data_lock:
            if april_detector.latest_lhand_goal is not None and april_detector.latest_rhand_goal is not None:
                break
        # rospy.loginfo("Waiting for initial joint state data...")
        rate.sleep() 

    x = input("Ready for RL policy? (y/n): ")
    if x != "y":
        exit()

    # rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/torobo/torobo_states",ToroboState, prop_callback)
    rospy.Subscriber("/torobo/gpio_state_controller/analog_io",Float64MultiArray, tactile_callback)

    rospy.loginfo("RL Policy Node started. Waiting for first joint state message...")

    # Wait until the first message has been received and data is available
    # This prevents the policy from trying to use None as an observation.
    while not rospy.is_shutdown():
        with data_lock:
            if latest_joint_pos_norm is not None:
                break
        rospy.loginfo("Waiting for initial joint state data...")
        rate.sleep()

    rospy.loginfo("Initial joint state received. Starting policy loop.")

    cur_targets = default_joint_pos_policy_order
    prev_targets = default_joint_pos_policy_order


    while not rospy.is_shutdown():

        # run 1 episode
        for t in range(EPISODE_TIMESTEPS):

            left_pos_noise = np.random.uniform(-1.0, 1.0, 3) * RESET_GOAL_SCALE
            right_pos_noise = np.random.uniform(-1.0, 1.0, 3) * RESET_GOAL_SCALE

            # read this from the april tag detector
            april_tag_goal=True 
            if april_tag_goal:
                lhand_goal = np.array(april_detector.latest_lhand_goal)
                rhand_goal = np.array(april_detector.latest_rhand_goal)
                goal_q = tf.transformations.quaternion_from_euler(0, 0, 0)

            else:
                lhand_goal = default_lhand_goal + left_pos_noise
                rhand_goal = default_rhand_goal + right_pos_noise
                goal_q = tf.transformations.quaternion_from_euler(0, 0, 0)

            # 1. Get the current observation (joint positions)
            with data_lock:
                if latest_joint_pos_norm is not None:
                
                    (latest_lhand_pos, rot) = listener.lookupTransform(BASE_LINK, LHAND_EE, rospy.Time(0))
                    (latest_rhand_pos, rot) = listener.lookupTransform(BASE_LINK, RHAND_EE, rospy.Time(0))     
                    # (lhand_goal, rot) = listener.lookupTransform('/base_link', '/lhand_goal_frame', rospy.Time(0))
                    # (rhand_goal, rot) = listener.lookupTransform('/base_link', '/rhand_goal_frame', rospy.Time(0))        
                    # 
                    # latest_lhand_pos -= np.array(EE_OFFSET)
                    # latest_rhand_pos -= np.array(EE_OFFSET)

             
                    obs = {
                        "prop": get_proprioception(cur_targets, prev_targets).to(dtype=torch.float32),
                        "gt": get_gt(latest_lhand_pos, latest_rhand_pos, lhand_goal, rhand_goal).to(dtype=torch.float32)
                    }
                else:
                    # This should ideally not happen after the initial wait, but good for robustness
                    rospy.logwarn("No joint position data available for policy. Skipping this iteration.")
                    rate.sleep()
                    continue

            # Publish the transform
            br.sendTransform(lhand_goal, goal_q, rospy.Time.now(), lhand_goal_frame, parent_frame)
            br.sendTransform(rhand_goal, goal_q, rospy.Time.now(), rhand_goal_frame, parent_frame)

            # # 2. Use the observation in your RL policy
            # This is where your actual RL algorithm would go.
            # For demonstration, we'll just print the observation and simulate an action.
            rospy.loginfo(f"Policy obs: {obs['prop'].size()} {obs['gt'].size()}")

            # --- Placeholder for your RL Policy Logic ---
            z = encoder(obs).T

            # the policy activation is tanh, so outputs are between -1 and 1
            actions = policy.act(z, deterministic=True)[0][0]
            actions = actions.detach().cpu().numpy()

            # scale the actions from [-1, 1] to [min, max] joint limits
            cur_targets = scale(actions, LOWER_LIMITS, UPPER_LIMITS)

            # only contribute ACTION_TAU % of the new action to the current action
            # this is really important to make airec motion slow and smooth 
            cur_targets = ACTION_TAU * cur_targets + (1-ACTION_TAU) * prev_targets

            # in case the moving average computation violates limits, perform clipping
            cur_targets = np.clip(cur_targets, LOWER_LIMITS, UPPER_LIMITS)

            # 3. Apply the action (e.g., publish to a robot command topic)
            rospy.loginfo(f"Policy Action: {format(np.rad2deg(cur_targets))}")
            publish_joint_trajectory(
                publisher = publisher,
                joint_names = policy_joint_order,
                positions = list(cur_targets),
                time_from_start = 1 / RL_HZ 
            )

            # 4. Sleep to maintain the desired loop rate
            rate.sleep()
            prev_targets = cur_targets

        # stop the robot
        publish_joint_trajectory(
                publisher = publisher,
                joint_names = policy_joint_order,
                positions = list(latest_joint_pos),
                time_from_start = 1 / RL_HZ 
            )

        x = input("Reset with new goal? (y/n)")
        if x != "y":
            break
    
    rospy.loginfo("RL Policy Node shutting down.")




if __name__ == '__main__':
    try:
        rl_policy_loop()
    except rospy.ROSInterruptException:
        pass


