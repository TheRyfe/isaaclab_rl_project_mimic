"""Configuration for the AIREC Humanoid Robot.

This configuration loads the AIREC robot from a local USD file and explicitly
defines actuator groups for all 47 joints. The USD file’s default joint drive
properties (e.g. stiffness, damping, and other drive parameters) are maintained.

Reference: /home/simon/IsaacLab/Models/AIREC/dry-airec.usd
"""
import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

VEL_SCALE =  0.1

AIREC_CFG = ArticulationCfg(
    ###########################################################################
    # Where and how to load the AIREC USD
    ###########################################################################

    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/simon/IsaacLab/scripts/AIREC_Packages/isaaclab_rl/isaaclab_rl_project_mimic/assets/airec/dry-airec.usd",  # <-- AIREC USD location
        activate_contact_sensors=True,
        #to fix self collision, look up collision filtering in isaac sim docs
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,  # Fix the root link to prevent it from moving
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.00001,
            stabilization_threshold=0.00001,
        ),
    ),

    ###########################################################################
    # Initial state: Place the robot in the world.
    # Do not override the USD-specified joint positions so that the original
    # drive (and other joint) properties remain intact.
    ###########################################################################
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Robot base position in world (x, y, z). Adjust if needed.
    ),

    ###########################################################################
    # Actuators: Each group explicitly lists certain joint names.
    # Empty stiffness and damping dicts are passed so that the USD default
    # drive properties are not overridden.
    ###########################################################################

    actuators={ #type: ignore[attr-defined]

        # ---------------------------------------------------------------------
        # Torso joints - no data in the tables, so unchanged
        # ---------------------------------------------------------------------
        "torso_joints": ImplicitActuatorCfg(
            joint_names_expr=[
                "torso_joint_1",
                "torso_joint_2",
                "torso_joint_3",
            ],
            stiffness={"torso_joint_1": 3000.0,
                       "torso_joint_2": 1000.0,
                       "torso_joint_3": 500.0,
                    },
            damping={"torso_joint_1": 150.0,
                    "torso_joint_2": 100.0,
                    "torso_joint_3": 50.0
                    },
            velocity_limit_sim={
                "torso_joint_1": 0.872 * VEL_SCALE,
                "torso_joint_2": 1.571 * VEL_SCALE,
                "torso_joint_3": 1.571 * VEL_SCALE
            },
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
            stiffness={".*": 1000.0},
            damping={".*": 100.0},
            velocity_limit_sim={
                "head_joint_1": 5.585 * VEL_SCALE,
                "head_joint_2": 4.712 * VEL_SCALE,
                "head_joint_3": 3.839 * VEL_SCALE
            },
            effort_limit={
                "head_joint_1": 8.0,
                "head_joint_2": 6.0,
                "head_joint_3": 4.0,
            },
        ),

        # ---------------------------------------------------------------------
        # Left arm joints (腕関節1–4 + 手関節1–3 => 7 DOFs):
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
            stiffness={"left_arm_joint_1": 300,
                    "left_arm_joint_2": 300,
                    "left_arm_joint_3": 200,
                    "left_arm_joint_4": 200,
                    "left_arm_joint_5": 20,
                    "left_arm_joint_6": 20,
                    "left_arm_joint_7": 20,
                    },
            damping={"left_arm_joint_1": 30,
                    "left_arm_joint_2": 30,
                    "left_arm_joint_3": 20,
                    "left_arm_joint_4": 20,
                    "left_arm_joint_5": 1,
                    "left_arm_joint_6": 1,
                    "left_arm_joint_7": 1
            },
            velocity_limit_sim={
                "left_arm_joint_1": 2.617 * VEL_SCALE,
                "left_arm_joint_2": 2.617 * VEL_SCALE,
                "left_arm_joint_3": 3.316 * VEL_SCALE,
                "left_arm_joint_4": 3.316 * VEL_SCALE,
                "left_arm_joint_5": 4.014 * VEL_SCALE,
                "left_arm_joint_6": 4.014 * VEL_SCALE,
                "left_arm_joint_7": 4.014 * VEL_SCALE
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
            stiffness={"right_arm_joint_1": 300,
                    "right_arm_joint_2": 300,
                    "right_arm_joint_3": 200,
                    "right_arm_joint_4": 200,
                    "right_arm_joint_5": 20,
                    "right_arm_joint_6": 20,
                    "right_arm_joint_7": 20,
                    },
            damping={"right_arm_joint_1": 30,
                    "right_arm_joint_2": 30,
                    "right_arm_joint_3": 20,
                    "right_arm_joint_4": 20,
                    "right_arm_joint_5": 1,
                    "right_arm_joint_6": 1,
                    "right_arm_joint_7": 1
            },
            velocity_limit_sim={
                "right_arm_joint_1": 2.617 * VEL_SCALE,
                "right_arm_joint_2": 2.617 * VEL_SCALE,
                "right_arm_joint_3": 3.316 * VEL_SCALE,
                "right_arm_joint_4": 3.316 * VEL_SCALE,
                "right_arm_joint_5": 4.014 * VEL_SCALE,
                "right_arm_joint_6": 4.014 * VEL_SCALE,
                "right_arm_joint_7": 4.014 * VEL_SCALE
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
        ),
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
            stiffness={".*": 3.0},
            damping={".*": 0.3},
            velocity_limit=75.0,  # ~1.309
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
            stiffness={".*": 3.0},
            damping={".*": 0.3},
            velocity_limit=75.0,  # ~1.309
            effort_limit=1.0,                 # Nm
        ),
    },
)

