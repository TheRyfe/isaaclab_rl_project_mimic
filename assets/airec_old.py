import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Helper for converting degrees/sec to radians/sec
DEG_TO_RAD = 1.0
#math.pi / 180.0

"""

['root_2', 'base_link_trans_x', 'root', 'base_link_trans_y', 'base_link_rot_yaw', 'base_footprint', 
'base_link', 'base_front_left_wheel_link', 'base_front_right_wheel_link', 'base_link_tip', 'base_rear_left_wheel_link', 
'base_rear_right_wheel_link', 'base_front_left_wheel_link_inner_barrel_center', '
base_front_left_wheel_link_outer_barrel_center', 'base_front_right_wheel_link_inner_barrel_center', 
'base_front_right_wheel_link_outer_barrel_center', 'torso_link_0', 'base_rear_left_wheel_link_inner_barrel_center',
 'base_rear_left_wheel_link_outer_barrel_center', 'base_rear_right_wheel_link_inner_barrel_center', 
'base_rear_right_wheel_link_outer_barrel_center', 'torso_link_1', 'torso_link_2', 'torso_link_3', 'torso_link_tip', 
'torso_left_shoulder_link', 'torso_right_shoulder_link', 'torso_waist_top_link', 'head_link_0', 'left_arm_link_0', 
'right_arm_link_0', 'torso_a_fsr_gpai_port_link', 'torso_b_fsr_gpai_port_link', 'torso_c_fsr_gpai_port_link', 
'torso_d_fsr_gpai_port_link', 'torso_e_fsr_gpai_port_link', 'torso_f_fsr_gpai_port_link', 'torso_g_fsr_gpai_port_link',
'head_link_1', 'left_arm_link_1', 'right_arm_link_1', 'head_link_2', 'left_arm_link_2', 
'left_arm_link_1_a_fsr_gpai_port_link', 'left_arm_link_1_b_fsr_gpai_port_link', 'right_arm_link_2', 
'right_arm_link_1_a_fsr_gpai_port_link', 'right_arm_link_1_b_fsr_gpai_port_link', 'head_link_3', 'left_arm_link_3', 
'left_arm_link_2_a_fsr_gpai_port_link', 'left_arm_link_2_b_fsr_gpai_port_link', 'left_arm_link_2_c_fsr_gpai_port_link',
'right_arm_link_3', 'right_arm_link_2_a_fsr_gpai_port_link', 'right_arm_link_2_b_fsr_gpai_port_link', 
'right_arm_link_2_c_fsr_gpai_port_link', 'head_link_tip', 'left_arm_link_4', 'left_arm_link_3_a_fsr_gpai_port_link', 
'left_arm_link_3_b_fsr_gpai_port_link', 'left_arm_link_3_c_fsr_gpai_port_link', 'left_arm_link_3_d_fsr_gpai_port_link',
'left_arm_link_3_e_fsr_gpai_port_link', 'left_arm_link_3_f_fsr_gpai_port_link', 'right_arm_link_4', 
'right_arm_link_3_a_fsr_gpai_port_link', 'right_arm_link_3_b_fsr_gpai_port_link', 'right_arm_link_3_c_fsr_gpai_port_link', 
'right_arm_link_3_d_fsr_gpai_port_link', 'right_arm_link_3_e_fsr_gpai_port_link', 'right_arm_link_3_f_fsr_gpai_port_link', 
'head_insta360_camera_link', 'head_link_face_center', 'head_see3cam_left_camera_link', 'head_see3cam_right_camera_link',
'head_sr300_camera_link', 'left_arm_link_5', 'left_arm_link_4_a_fsr_gpai_port_link', 'left_arm_link_4_b_fsr_gpai_port_link',
'left_arm_link_4_c_fsr_gpai_port_link', 'left_arm_link_4_d_fsr_gpai_port_link', 'left_arm_link_4_e_fsr_gpai_port_link', 
'left_arm_link_4_f_fsr_gpai_port_link', 'right_arm_link_5', 'right_arm_link_4_a_fsr_gpai_port_link', 
'right_arm_link_4_b_fsr_gpai_port_link', 'right_arm_link_4_c_fsr_gpai_port_link', 'right_arm_link_4_d_fsr_gpai_port_link',
'right_arm_link_4_e_fsr_gpai_port_link', 'right_arm_link_4_f_fsr_gpai_port_link', 'head_insta360_camera_color_frame', 
'head_see3cam_left_camera_color_frame', 'head_see3cam_right_camera_color_frame', 'head_sr300_camera_color_frame', 
'left_arm_link_6', 'left_arm_link_5_a_fsr_gpai_port_link', 'right_arm_link_6', 'right_arm_link_5_a_fsr_gpai_port_link',
'head_insta360_camera_color_optical_frame', 'head_see3cam_left_camera_color_optical_frame', 
'head_see3cam_right_camera_color_optical_frame', 'head_sr300_camera_color_optical_frame', 'left_arm_link_7', 
'left_arm_link_6_a_fsr_gpai_port_link', 'right_arm_link_7', 'right_arm_link_6_a_fsr_gpai_port_link', 'left_arm_link_tip',
'right_arm_link_tip', 'left_arm_link_load', 'left_hand_base_link', 'right_arm_link_load', 'right_hand_base_link',
'left_hand_first_finger_link_0', 'left_hand_palm_fsr_gpai_port_link', 'left_hand_palm_link', 
'left_hand_second_finger_link_0', 'left_hand_third_finger_link_0', 'left_hand_thumb_link_0', 
'right_hand_first_finger_link_0', 'right_hand_palm_fsr_gpai_port_link', 'right_hand_palm_link', 
'right_hand_second_finger_link_0', 'right_hand_third_finger_link_0', 'right_hand_thumb_link_0',
'left_hand_first_finger_link_1', 'left_hand_second_finger_link_1', 'left_hand_third_finger_link_1',
'left_hand_thumb_link_1', 'right_hand_first_finger_link_1', 'right_hand_second_finger_link_1',
'right_hand_third_finger_link_1', 'right_hand_thumb_link_1', 'left_hand_first_finger_link_2', 
'left_hand_second_finger_link_2', 'left_hand_third_finger_link_2', 'left_hand_thumb_link_2', 
'right_hand_first_finger_link_2', 'right_hand_second_finger_link_2', 'right_hand_third_finger_link_2', 
'right_hand_thumb_link_2', 'left_hand_first_finger_link_tip', 'left_hand_second_finger_link_tip', 
'left_hand_third_finger_link_tip', 'left_hand_thumb_link_2_3', 'right_hand_first_finger_link_tip', '
right_hand_second_finger_link_tip', 'right_hand_third_finger_link_tip', 'right_hand_thumb_link_2_3', 
'left_hand_first_finger_link_tip_fsr_gpai_port_link', 'left_hand_second_finger_link_tip_fsr_gpai_port_link',
'left_hand_third_finger_link_tip_fsr_gpai_port_link', 'left_hand_thumb_link_3', 
'right_hand_first_finger_link_tip_fsr_gpai_port_link', 'right_hand_second_finger_link_tip_fsr_gpai_port_link', 
'right_hand_third_finger_link_tip_fsr_gpai_port_link', 'right_hand_thumb_link_3', 'left_hand_thumb_link_4',
'right_hand_thumb_link_4', 'left_hand_thumb_link_tip', 'right_hand_thumb_link_tip', 
'left_hand_thumb_link_tip_fsr_gpai_port_link', 'right_hand_thumb_link_tip_fsr_gpai_port_link']



"""

import os
parent_dir = os.getcwd()

AIREC_CFG = ArticulationCfg(
    ###########################################################################
    # Where and how to load the AIREC USD
    ###########################################################################
    spawn=sim_utils.UsdFileCfg(
        # usd_path=os.path.join(parent_dir, "airec_gym/assets/airec/dry-airec.usd"),
        usd_path=os.path.join(parent_dir, "airec_gym/assets/torobo2/dry-airec_planar_move_floating_base.usd"),

        activate_contact_sensors=True,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        #if you want self-collision enabled, uncomment and adjust:
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            fix_root_link=True, # base physics are broken if this is False
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.01,
            stabilization_threshold=0.01,
        ),
    ),
    soft_joint_pos_limit_factor=1.0,


    ###########################################################################
    # Initial placement of the robot in the world
    ###########################################################################
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 1),
    #     joint_pos={
    #         "base_joint_trans_x": 0.0,
    #         "base_joint_trans_y": -0.569,
    #         "base_joint_rot_yaw": 0.0,
    #         "panda_joint4": -2.810,
    #         "panda_joint5": 0.0,
    #         "panda_joint6": 3.037,
    #         "panda_joint7": 0.741,
    #         "panda_finger_joint.*": 0.04,
    #     },
    # ),

    ###########################################################################
    # Actuators
    ###########################################################################
    actuators={ # type: ignore[attr-defined]
        # ---------------------------------------------------------------------
        # Base joints (translation & yaw) - no data in your tables, so unchanged
        # ---------------------------------------------------------------------
        # "base_joints": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "base_joint_trans_x",
        #         "base_joint_trans_y",
        #         "base_joint_rot_yaw",
        #     ],
        #     stiffness=1000,
        #     damping=1000,
        #     velocity_limit=0.1,
        #     effort_limit=1, 
        # ),

        # ---------------------------------------------------------------------
        # Base wheels (台車駆動関節1–4): 700 deg/s => ~12.217 rad/s, 50 Nm
        # ---------------------------------------------------------------------
        # "base_wheels": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         "base_front_left_wheel_joint",
        #         "base_front_right_wheel_joint",
        #         "base_rear_left_wheel_joint",
        #         "base_rear_right_wheel_joint",
        #     ],
        #     stiffness=10000.0,
        #     damping=1000.0,
        #     velocity_limit=0.001 * DEG_TO_RAD,  # ~12.217
        #     effort_limit=10.0,                 # Nm
        # ),

        # ---------------------------------------------------------------------
        # Torso joints - no data in the tables, so unchanged
        # ---------------------------------------------------------------------
        "torso_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_joint_1",
                "torso_joint_2",
                "torso_joint_3",
            ],
            stiffness=10000.0,
            damping=1000.0,
        ),

        # ---------------------------------------------------------------------
        # Head joints (頭部関節1–3) from your second table:
        #   Joint 1 => 320 deg/s => ~5.585 rad/s, 8 Nm
        #   Joint 2 => 180 deg/s => ~3.142 rad/s, 6 Nm
        #   Joint 3 => 100 deg/s => ~1.745 rad/s, 4 Nm
        # ---------------------------------------------------------------------
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_joint_1",
                "head_joint_2",
                "head_joint_3",
            ],
            stiffness=100.0,
            damping=10.0,
            velocity_limit={
                "head_joint_1": 320.0 * DEG_TO_RAD,
                "head_joint_2": 180.0 * DEG_TO_RAD,
                "head_joint_3": 100.0 * DEG_TO_RAD,
            },
            effort_limit={
                "head_joint_1": 8.0,
                "head_joint_2": 6.0,
                "head_joint_3": 4.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Left arm joints (腕関節1–4 + 手関節1–3 => 7 DOFs):
        #
        # Example values from your first table:
        #   1) 150 deg/s => ~2.617 rad/s, 70 Nm
        #   2) 150 deg/s => ~2.617 rad/s, 150 Nm
        #   3) 120 deg/s => ~2.094 rad/s, 100 Nm
        #   4) 190 deg/s => ~3.316 rad/s, 190 Nm
        #   5) 210 deg/s => ~3.665 rad/s, 80 Nm
        #   6) 210 deg/s => ~3.665 rad/s, 60 Nm
        #   7) 210 deg/s => ~3.665 rad/s, 50 Nm
        # ---------------------------------------------------------------------
        "left_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_joint_1",
                "left_arm_joint_2",
                "left_arm_joint_3",
                "left_arm_joint_4",
                "left_arm_joint_5",
                "left_arm_joint_6",
                "left_arm_joint_7",
            ],
            stiffness=10000.0,
            damping=1000.0,
            velocity_limit={
                "left_arm_joint_1": 150.0 * DEG_TO_RAD,
                "left_arm_joint_2": 150.0 * DEG_TO_RAD,
                "left_arm_joint_3": 120.0 * DEG_TO_RAD,
                "left_arm_joint_4": 190.0 * DEG_TO_RAD,
                "left_arm_joint_5": 210.0 * DEG_TO_RAD,
                "left_arm_joint_6": 210.0 * DEG_TO_RAD,
                "left_arm_joint_7": 210.0 * DEG_TO_RAD,
            },
            effort_limit={
                "left_arm_joint_1": 70.0,
                "left_arm_joint_2": 150.0,
                "left_arm_joint_3": 100.0,
                "left_arm_joint_4": 190.0,
                "left_arm_joint_5": 80.0,
                "left_arm_joint_6": 60.0,
                "left_arm_joint_7": 50.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Right arm joints (same data as left arm, unless your specs differ)
        # ---------------------------------------------------------------------
        "right_arm_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_arm_joint_1",
                "right_arm_joint_2",
                "right_arm_joint_3",
                "right_arm_joint_4",
                "right_arm_joint_5",
                "right_arm_joint_6",
                "right_arm_joint_7",
            ],
            stiffness=10000.0,
            damping=1000.0,
            velocity_limit={
                "right_arm_joint_1": 150.0 * DEG_TO_RAD,
                "right_arm_joint_2": 150.0 * DEG_TO_RAD,
                "right_arm_joint_3": 120.0 * DEG_TO_RAD,
                "right_arm_joint_4": 190.0 * DEG_TO_RAD,
                "right_arm_joint_5": 210.0 * DEG_TO_RAD,
                "right_arm_joint_6": 210.0 * DEG_TO_RAD,
                "right_arm_joint_7": 210.0 * DEG_TO_RAD,
            },
            effort_limit={
                "right_arm_joint_1": 70.0,
                "right_arm_joint_2": 150.0,
                "right_arm_joint_3": 100.0,
                "right_arm_joint_4": 190.0,
                "right_arm_joint_5": 80.0,
                "right_arm_joint_6": 60.0,
                "right_arm_joint_7": 50.0,
            },
        ),        #         # This was in your original config, though it looks suspicious:
        #         "right_hand_third_finger_joint_1",

        # ---------------------------------------------------------------------
        # Left hand joints (指関節), from your third table: 75 deg/s, 1 Nm
        # ---------------------------------------------------------------------
        "left_hand_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_first_finger_joint_1",
                "left_hand_second_finger_joint_1",
                "left_hand_third_finger_joint_1",
                "left_hand_thumb_joint_1",
                "left_hand_thumb_joint_2",
                "left_hand_thumb_joint_3",
                "left_hand_first_finger_joint_2",
                "left_hand_second_finger_joint_2",
                "left_hand_third_finger_joint_2",
                "left_hand_thumb_joint_4",
            ],
            stiffness=0.3,
            damping=3.0,
            velocity_limit=75.0 * DEG_TO_RAD,  # ~1.309
            effort_limit=1.0,                 # Nm
        ),

        # ---------------------------------------------------------------------
        # Right hand joints (same as left hand)
        # ---------------------------------------------------------------------
        "right_hand_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_hand_first_finger_joint_1",
                "right_hand_second_finger_joint_1",
                "right_hand_third_finger_joint_1",
                "right_hand_thumb_joint_1",
                "right_hand_thumb_joint_2",
                "right_hand_thumb_joint_3",
                "right_hand_first_finger_joint_2",
                "right_hand_second_finger_joint_2",
                "right_hand_third_finger_joint_2",
                "right_hand_thumb_joint_4",
            ],
            stiffness=0.3,
            damping=3.0,
            velocity_limit=75.0 * DEG_TO_RAD,  # ~1.309
            effort_limit=1.0,                 # Nm
        ),
    },
)