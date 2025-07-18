# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import field

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

# AIREC specific imports
from assets.airec import AIREC_CFG
from tasks.mimic.airec import AIRECEnv, AIRECEnvCfg, scale

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import VecEnvObs
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz, quat_apply_inverse
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# =============================================================================
# Configuration
# =============================================================================


@configclass
class RewardsCfg:
    """Configuration for reward components and scales for the mimicry task."""

    # Position tracking rewards
    joint_pos_tracking_reward_scale: float = 4.0
    pos_error_variance_scale: float = 0.25
    use_weighted_pos_tracking: bool = True
    # Joint importance weights for different body parts
    head_joint_weight: float = 0.5
    torso_joint_weight: float = 2.0
    arm_joint_weight: float = 1.0

    # Velocity tracking rewards
    joint_vel_tracking_reward_scale: float = 2.0
    vel_error_variance_scale: float = 0.5

    # Orientation rewards
    orientation_reward_scale: float = 0.5
    orientation_error_threshold: float = 0.30  # in radians

    # Penalties
    current_joint_vel_penalty_scale: float = -0.001
    action_smoothness_penalty_scale: float = -0.01
    joint_acceleration_penalty_scale: float = -0.01
    energy_penalty_scale: float = -0.001

    # Other rewards
    staying_alive_reward: float = 0.005


@configclass
class TerminationCfg:
    """Configuration for episode termination conditions."""

    # Joint limit termination
    enable_joint_limit_termination: bool = True
    joint_limit_buffer: float = 0.95  # Terminate at 95% of soft limits

    # Torso tilt termination
    enable_torso_tilt_termination: bool = True
    torso_tilt_limit: float = 0.4  # in radians
    torso_joints_to_check: list[str] = field(default_factory=lambda: ["torso_joint_1", "torso_joint_2"])


@configclass
class ExternalDisturbanceCfg:
    """Configuration for external force disturbances."""

    enable_disturbances: bool = True

    # Target body configuration
    target_body_name: str = "right_arm_link_5"

    # Force parameters (in Newtons, applied in global/world coordinate frame)
    force_magnitude_range: tuple[float, float] = (300.0, 850.0)

    # Duration parameters (in seconds)
    duration_range: tuple[float, float] = (0.5, 2.5)  # Shorter duration for more frequent changes

    # Interval between disturbances (in seconds)
    interval_range: tuple[float, float] = (0.5, 3.0)  # Shorter cooldown

    # Probability of applying disturbance per environment per step
    disturbance_probability: float = 0.01  # 1% chance per step per env

    # Episode start timeout (in seconds) - no forces applied during this period
    episode_start_timeout: float = 4.0

    # Directional bias parameters (for simulating leaning)
    # Bias values are normalized direction weights (will be normalized to unit vector)
    directional_bias: tuple[float, float, float] = (0.0, 0.0, -1.0)  # Default: downward bias
    bias_strength: float = 0.4  # 0.0 = pure random, 1.0 = pure bias direction

    # Visualization parameters
    enable_force_visualization: bool = True
    force_arrow_scale: float = 1.0  # Scale factor: arrow length = force magnitude * scale

    # Arrow shape parameters
    arrow_length_scale: float = 1.0  # Length scaling factor for arrow
    arrow_thickness: float = 0.3  # Cross-sectional thickness (Y and Z dimensions)
    arrow_color: tuple[float, float, float] = (1.0, 0.0, 0.0)  # RGB color (red by default)


# -- Ghost Robot Configuration
# Defines the USD asset and initial state for the ghost robot used for visualization.
KINEMATIC_GHOST_USD_FILE_PATH = "assets/airec/airec_ghost.usd"
DEFAULT_KINEMATIC_GHOST_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/GhostKinematicRobot",
    spawn=UsdFileCfg(
        usd_path=KINEMATIC_GHOST_USD_FILE_PATH,
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={},
)


@configclass
class MimicEnvCfg(AIRECEnvCfg):
    """Configuration for the Motion Mimicry environment."""

    # -- Task-specific parameters
    animation_file: str = "assets/animation/walkingsupport.csv"
    animation_dt_info: float = 1.0 / 60.0
    dynamic_episode_length_buffer_s: float = 2.0

    # -- Robot control parameters
    num_base_actions: int = 0
    num_prop_joints: int = 20
    csv_column_joint_names: list[str] = [
        "H1",
        "H2",
        "H3",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "L1",
        "L2",
        "L3",
        "L4",
        "L5",
        "L6",
        "L7",
        "T1",
        "T2",
        "T3",
    ]
    obs_list: list[str] = ["gt", "prop"]
    control_mode: str = "position"

    # -- Sub-configurations
    rewards: RewardsCfg = RewardsCfg()
    termination: TerminationCfg = TerminationCfg()
    external_disturbance: ExternalDisturbanceCfg = ExternalDisturbanceCfg()

    # -- Ghost Visualizer Configuration
    enable_ghost_visualizer: bool = True
    ghost_robot_cfg: ArticulationCfg = DEFAULT_KINEMATIC_GHOST_CFG.replace(
        spawn=DEFAULT_KINEMATIC_GHOST_CFG.spawn.replace(
            # Set kinematic properties for the ghost to save performance.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=0,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.0,
                stabilization_threshold=0.0,
            ),
            activate_contact_sensors=False,
            # Set the ghost robot's color
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3), roughness=0.4, metallic=0.1),  # Red color
        )
    )


# =============================================================================
# Environment
# =============================================================================


class MimicEnv(AIRECEnv):
    """
    An environment for training a robot to mimic a pre-recorded animation.

    This environment rewards an RL agent for tracking the joint positions from a CSV animation
    file. It includes a "ghost" robot visualization to show the target pose at each frame.
    The core logic for advancing the animation and updating states is handled in the
    overridden `_compute_intermediate_values` method in the parent `AIRECEnv`.
    """

    cfg: MimicEnvCfg
    ghost_robot: Articulation | None
    ghost_mimic_joint_indices: torch.Tensor | None

    def __init__(self, cfg: MimicEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initializes the Motion Mimicry Environment.

        This environment trains a robot to mimic pre-recorded animations from CSV files.
        It includes features like ghost robot visualization, external force disturbances,
        and comprehensive reward shaping for motion tracking.
        """
        # Initialize the Motion Mimicry Environment
        self.cfg = cfg
        self.ghost_robot = None
        self.ghost_mimic_joint_indices = None

        # -- Pre-initialization checks and setup
        provisional_physics_dt = cfg.sim.dt
        provisional_decimation = cfg.decimation
        provisional_control_dt = provisional_physics_dt * provisional_decimation

        # Check if control frequency matches animation frequency
        if not math.isclose(provisional_control_dt, cfg.animation_dt_info, rel_tol=1e-5):
            print(
                f"[MimicEnv __init__] CONFIG WARNING: Provisional control_dt ({provisional_control_dt:.6f}s) "
                f"from cfg (sim.dt={cfg.sim.dt}, decimation={cfg.decimation}) "
                f"does NOT match cfg.animation_dt_info ({cfg.animation_dt_info:.6f}s). "
                f"The environment will proceed with control_dt={provisional_control_dt:.6f}s."
            )

        # Load animation data statically to determine the required episode length
        self.max_animation_steps = 0
        self._load_animation_data_static(cfg.animation_file, cfg.csv_column_joint_names)
        self._mimic_env_determined_max_animation_steps = self.max_animation_steps
        # Animation steps determined for episode length calculation

        # Dynamically adjust episode length to ensure the full animation can be played
        required_episode_length_s = cfg.episode_length_s
        if self._mimic_env_determined_max_animation_steps > 0 and provisional_control_dt > 0:
            calculated_animation_duration_s = self._mimic_env_determined_max_animation_steps * provisional_control_dt
            required_episode_length_s_for_anim = calculated_animation_duration_s + cfg.dynamic_episode_length_buffer_s
            if required_episode_length_s_for_anim > required_episode_length_s:
                print(
                    f"[MimicEnv __init__ PRE-SUPER] INFO: Original cfg.episode_length_s: {cfg.episode_length_s:.2f}s."
                )
                print(
                    f"[MimicEnv __init__ PRE-SUPER] INFO: Animation requires {self._mimic_env_determined_max_animation_steps} control steps ({calculated_animation_duration_s:.2f}s). "
                    f"With buffer, desired episode length is {required_episode_length_s_for_anim:.2f}s."
                )
                required_episode_length_s = required_episode_length_s_for_anim

        # -- Prepare configuration for the parent AIRECEnv
        modified_parent_cfg = copy.deepcopy(cfg)
        modified_parent_cfg.episode_length_s = required_episode_length_s
        self.num_mimic_joints = len(cfg.csv_column_joint_names)
        if self.num_mimic_joints == 0:
            raise ValueError("'MimicEnvCfg.csv_column_joint_names' cannot be empty.")

        # Map joint names from the CSV file to the robot's actual joint names
        self.robot_mimicked_joint_names_ordered = []
        self.csv_to_robot_joint_map = {
            "H1": "head_joint_1",
            "H2": "head_joint_2",
            "H3": "head_joint_3",
            "R1": "right_arm_joint_1",
            "R2": "right_arm_joint_2",
            "R3": "right_arm_joint_3",
            "R4": "right_arm_joint_4",
            "R5": "right_arm_joint_5",
            "R6": "right_arm_joint_6",
            "R7": "right_arm_joint_7",
            "L1": "left_arm_joint_1",
            "L2": "left_arm_joint_2",
            "L3": "left_arm_joint_3",
            "L4": "left_arm_joint_4",
            "L5": "left_arm_joint_5",
            "L6": "left_arm_joint_6",
            "L7": "left_arm_joint_7",
            "T1": "torso_joint_1",
            "T2": "torso_joint_2",
            "T3": "torso_joint_3",
        }
        for csv_name in cfg.csv_column_joint_names:
            robot_name = self.csv_to_robot_joint_map.get(csv_name)
            if robot_name is None:
                raise ValueError(f"CSV column '{csv_name}' not found in csv_to_robot_joint_map.")
            self.robot_mimicked_joint_names_ordered.append(robot_name)

        # Override parent config with mimic-specific dimensions and settings
        modified_parent_cfg.actuated_joint_names = self.robot_mimicked_joint_names_ordered
        modified_parent_cfg.num_actions = self.num_mimic_joints
        modified_parent_cfg.num_gt_observations = self.num_mimic_joints * 3
        if "prop" in modified_parent_cfg.obs_list:
            cfg_num_prop_joints = cfg.num_prop_joints
            modified_parent_cfg.num_prop_observations = cfg_num_prop_joints * 2 + 7 * 2 + self.num_mimic_joints

        # Initialize the parent AIRECEnv with modified configuration
        super().__init__(cfg=modified_parent_cfg, render_mode=render_mode, **kwargs)

        # -- Post-initialization setup
        _physics_dt_final = self.sim.get_physics_dt()
        self.control_dt = _physics_dt_final * self.cfg.decimation
        if self.control_dt <= 0:
            raise ValueError(
                f"[MimicEnv __init__ POST-SUPER] CRITICAL: Final control_dt ({self.control_dt}) must be positive."
            )

        if hasattr(self.cfg, "animation_dt_info") and not math.isclose(
            self.control_dt, self.cfg.animation_dt_info, rel_tol=1e-5
        ):
            print(
                f"[MimicEnv __init__] POST-SUPER WARNING: Final control_dt ({self.control_dt:.6f}s) "
                f"does NOT match cfg.animation_dt_info ({self.cfg.animation_dt_info:.6f}s). "
            )
        else:
            print(
                f"[MimicEnv __init__] POST-SUPER INFO: Final control_dt ({self.control_dt:.6f}s) matches animation_dt_info."
            )

        # Load animation data into a tensor for runtime access
        self._load_animation_data()
        # print(f"DEBUG: Animation loaded - steps: {self.max_animation_steps}, data exists: {hasattr(self, 'animation_pos_data')}")

        # Final check on episode length vs animation length
        if hasattr(self, "max_animation_steps") and self.max_animation_steps > 0 and self.control_dt > 0:
            required_total_control_steps_for_animation = self.max_animation_steps
            if self.max_episode_length < required_total_control_steps_for_animation:
                print(
                    f"[MimicEnv __init__] POST-SUPER CRITICAL WARNING: Final max_episode_length ({self.max_episode_length}) "
                    f"is STILL SHORTER than animation steps ({required_total_control_steps_for_animation})."
                )

        # Create joint index mappings for the main controllable robot
        try:
            self.mimic_joint_indices_in_robot = torch.tensor(
                [self.robot.joint_names.index(name) for name in self.robot_mimicked_joint_names_ordered],
                device=self.device,
                dtype=torch.long,
            )
        except ValueError as e:
            print(f"ERROR: A mapped robot joint name was not found in MAIN robot's `joint_names` list: {e}")
            raise
        except AttributeError:
            print(
                "ERROR: self.robot or self.robot.joint_names not available for mimic_joint_indices_in_robot. This might happen if parent __init__ failed."
            )
            raise

        # Create joint index mappings for the ghost robot after it has been spawned in _setup_scene
        if self.cfg.enable_ghost_visualizer and self.ghost_robot is not None:
            if hasattr(self.ghost_robot, "joint_names"):
                try:
                    self.ghost_mimic_joint_indices = torch.tensor(
                        [self.ghost_robot.joint_names.index(name) for name in self.robot_mimicked_joint_names_ordered],
                        device=self.device,
                        dtype=torch.long,
                    )
                    # print("INFO: Successfully created ghost_mimic_joint_indices.")
                except ValueError as e:
                    print(
                        f"[ERROR] MimicEnv.__init__: A mimicked joint name was not found in GHOST_ROBOT's `joint_names` list: {e}"
                    )
                    print(f"    GHOST_ROBOT available joints (first 25): {self.ghost_robot.joint_names[:25]}")
                    print(f"    Trying to map these names: {self.robot_mimicked_joint_names_ordered}")
                    self.ghost_mimic_joint_indices = None
                except AttributeError as e:
                    print(
                        f"[ERROR] MimicEnv.__init__: self.ghost_robot or its attributes not fully available for ghost_mimic_joint_indices: {e}"
                    )
                    self.ghost_mimic_joint_indices = None
            else:
                print(
                    "[WARNING] MimicEnv.__init__: self.ghost_robot does not have 'joint_names' attribute. Cannot create ghost_mimic_joint_indices."
                )
                self.ghost_mimic_joint_indices = None
        elif self.cfg.enable_ghost_visualizer:
            print(
                "[WARNING] MimicEnv.__init__: Ghost visualizer enabled in Cfg but self.ghost_robot is None after super init. Cannot create ghost_mimic_joint_indices."
            )
            self.ghost_mimic_joint_indices = None

        # Setup velocity limits for mimicked joints
        if self.robot.data.joint_vel_limits is None or self.robot.data.joint_vel_limits.numel() == 0:
            print(
                "[MimicEnv __init__] WARNING: robot.data.joint_vel_limits not populated. Using default large limits (+/-10 rad/s)."
            )
            self.mimic_joint_vel_limits_lower = torch.full(
                (self.num_mimic_joints,), -10.0, device=self.device, dtype=torch.float32
            )
            self.mimic_joint_vel_limits_upper = torch.full(
                (self.num_mimic_joints,), 10.0, device=self.device, dtype=torch.float32
            )
        else:
            raw_mimic_joint_vel_limits = self.robot.data.joint_vel_limits[self.mimic_joint_indices_in_robot, :].clone()
            self.mimic_joint_vel_limits_lower = raw_mimic_joint_vel_limits[:, 0] * 0.8
            self.mimic_joint_vel_limits_upper = raw_mimic_joint_vel_limits[:, 1] * 0.8
        problematic_limits_mask = torch.isclose(
            self.mimic_joint_vel_limits_lower, torch.tensor(0.0, device=self.device)
        ) & torch.isclose(self.mimic_joint_vel_limits_upper, torch.tensor(0.0, device=self.device))
        if torch.any(problematic_limits_mask):
            num_problematic = torch.sum(problematic_limits_mask).item()
            print(
                f"[MimicEnv __init__] WARNING: {num_problematic} mimicked joints have zero scaled velocity limits. Overriding to +/- 0.1 rad/s."
            )
            self.mimic_joint_vel_limits_lower[problematic_limits_mask] = -0.1
            self.mimic_joint_vel_limits_upper[problematic_limits_mask] = 0.1

        # -- Initialize runtime variables for the mimicry task
        self.current_animation_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.global_env_steps_counter = 0
        self.extras["log"] = {
            "mimic_pos_tracking_reward": torch.zeros(self.num_envs, device=self.device),
            "mimic_staying_alive_reward": torch.zeros(self.num_envs, device=self.device),
            "mimic_current_vel_penalty": torch.zeros(self.num_envs, device=self.device),
            "mimic_action_smoothness_penalty": torch.zeros(self.num_envs, device=self.device),
            "mimic_total_reward": torch.zeros(self.num_envs, device=self.device),
            "current_animation_frame": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "joint_limit_violations": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "torso_tilt_violations": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "disturbance_active": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "disturbance_force_magnitude": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
        }

        # -- Initialize external disturbance system
        if self.cfg.external_disturbance.enable_disturbances:
            # Find the target body index
            body_ids, body_names = self.robot.find_bodies(self.cfg.external_disturbance.target_body_name)
            if body_ids:
                self.disturbance_body_id = body_ids[0]
                print(f"[INFO] External disturbance target: {body_names[0]} (body_id: {self.disturbance_body_id})")
                print(f"[DEBUG] Robot type: {type(self.robot)}")
                print(f"[DEBUG] Robot prim path: {self.robot.cfg.prim_path}")
                print(f"[DEBUG] Total robot bodies: {len(self.robot.body_names)}")

                # Check robot configuration for force support
                print(f"[DEBUG] Robot spawn config: {type(self.robot.cfg.spawn)}")
                if hasattr(self.robot.cfg.spawn, "rigid_props"):
                    print(f"[DEBUG] Rigid body properties: {self.robot.cfg.spawn.rigid_props}")
                if hasattr(self.robot.cfg.spawn, "articulation_props"):
                    print(f"[DEBUG] Articulation properties: {self.robot.cfg.spawn.articulation_props}")

                # Check if external forces are enabled
                print(
                    f"[DEBUG] Robot methods related to force: {[m for m in dir(self.robot) if 'force' in m.lower() or 'external' in m.lower()]}"
                )

            else:
                print(
                    f"[WARNING] Could not find body '{self.cfg.external_disturbance.target_body_name}' for disturbances"
                )
                print(f"[DEBUG] Available bodies: {self.robot.body_names[:20]}...")  # Show first 20 body names
                self.cfg.external_disturbance.enable_disturbances = False
                self.disturbance_body_id = None

        # Initialize disturbance tracking tensors
        self.disturbance_forces = torch.zeros((self.num_envs, 3), device=self.device)
        self.disturbance_torques = torch.zeros((self.num_envs, 3), device=self.device)
        self.disturbance_remaining_time = torch.zeros(self.num_envs, device=self.device)
        self.disturbance_cooldown_time = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_time = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.external_disturbance.enable_disturbances:
            print(f"[INFO] External disturbance system initialized:")
            print(f"  - Force range: {self.cfg.external_disturbance.force_magnitude_range} N")
            print(f"  - Duration range: {self.cfg.external_disturbance.duration_range} s")
            print(f"  - Interval range: {self.cfg.external_disturbance.interval_range} s")
            print(f"  - Probability: {self.cfg.external_disturbance.disturbance_probability}")
            print(f"  - Visualization: {self.cfg.external_disturbance.enable_force_visualization}")
        if hasattr(self, "animation_pos_data") and self.max_animation_steps <= 0:
            print("[WARNING] Animation data appears empty after full loading. Mimicry may not function.")
        # MimicEnv initialization complete

    def _setup_scene(self):
        """Sets up the simulation scene, including the ghost robot."""
        # Setup simulation scene with ghost robot and visualization markers
        # Call parent setup to spawn the main robot and other base assets
        super()._setup_scene()
        # Parent scene setup complete

        # Spawn the ghost robot if enabled in the configuration and GUI is available
        if self.cfg.enable_ghost_visualizer and self.sim.has_gui():
            self.ghost_robot = Articulation(self.cfg.ghost_robot_cfg)
            self.scene.articulations["ghost_robot"] = self.ghost_robot
            print("[INFO] Ghost visualizer robot added to the scene.")

            # Hide base-related visuals for the ghost robot
            self._hide_ghost_base_visuals()
        elif self.cfg.enable_ghost_visualizer and not self.sim.has_gui():
            print("[INFO] Ghost visualizer disabled in headless mode for performance.")

        # Initialize force visualization markers if enabled and GUI is available
        if (
            self.cfg.external_disturbance.enable_disturbances
            and self.cfg.external_disturbance.enable_force_visualization
            and self.sim.has_gui()
        ):
            try:
                # Create arrow marker configuration with configurable parameters
                base_arrow_scale = (
                    self.cfg.external_disturbance.arrow_length_scale,
                    self.cfg.external_disturbance.arrow_thickness,
                    self.cfg.external_disturbance.arrow_thickness,
                )
                force_arrow_cfg = VisualizationMarkersCfg(
                    prim_path="/World/Visuals/force_arrows",
                    markers={
                        "arrow": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                            scale=base_arrow_scale,  # Use configurable scale
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=self.cfg.external_disturbance.arrow_color,  # Use configurable color
                                roughness=0.4,
                                metallic=0.0,
                            ),
                        )
                    },
                )

                print(f"[DEBUG] Creating visualization markers with custom config")
                print(f"[DEBUG] Arrow marker config details: {force_arrow_cfg.markers}")

                # Initialize the visualization markers
                self.force_visualization_markers = VisualizationMarkers(force_arrow_cfg)
                print("[INFO] Force visualization markers initialized successfully.")
                print(f"[DEBUG] Marker prototypes: {self.force_visualization_markers.num_prototypes}")

            except Exception as e:
                print(f"[ERROR] Failed to initialize force visualization markers: {e}")
                print(f"[DEBUG] Trying fallback with original config...")
                try:
                    # Fallback to original config
                    force_arrow_cfg = RED_ARROW_X_MARKER_CFG.copy()
                    force_arrow_cfg.prim_path = "/World/Visuals/force_arrows"
                    self.force_visualization_markers = VisualizationMarkers(force_arrow_cfg)
                    print("[INFO] Fallback force visualization markers initialized.")
                except Exception as e2:
                    print(f"[ERROR] Fallback also failed: {e2}")
                    self.force_visualization_markers = None
        else:
            self.force_visualization_markers = None
            if (
                self.cfg.external_disturbance.enable_disturbances
                and self.cfg.external_disturbance.enable_force_visualization
                and not self.sim.has_gui()
            ):
                print("[INFO] Force visualization disabled in headless mode for performance.")
            else:
                print("[INFO] Force visualization disabled in config")

        # Scene setup complete

    def _load_animation_data_static(self, animation_file_path: str, csv_columns: list[str]):
        """Loads animation data from a CSV file just to determine its length for pre-init calculations."""
        # print(f"DEBUG: Loading animation file: {animation_file_path}")
        try:
            df = pd.read_csv(animation_file_path)
            missing_cols = [name for name in csv_columns if name not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"Joints {missing_cols} missing from CSV '{animation_file_path}'. Available columns: {df.columns.tolist()}."
                )
            animation_relevant_df = df[csv_columns]
            animation_np = animation_relevant_df.values
            self.max_animation_steps = animation_np.shape[0]
            # print(f"DEBUG: Loaded {self.max_animation_steps} frames for setup.")
            if self.max_animation_steps == 0:
                print(f"[MimicEnv _load_animation_data_static] CRITICAL: CSV '{animation_file_path}' loaded 0 frames.")
        except FileNotFoundError:
            print(f"ERROR: Animation file not found at '{animation_file_path}'.")
            self.max_animation_steps = 0
        except Exception as e:
            print(f"ERROR loading animation data statically from '{animation_file_path}': {e}")
            self.max_animation_steps = 0
        if self.max_animation_steps == 0:
            print("[MimicEnv _load_animation_data_static] CRITICAL: Failed to load animation for length check.")

    def _load_animation_data(self):
        """Loads the full animation data from a CSV file into a runtime tensor."""
        # print(f"DEBUG: Loading animation file: {self.cfg.animation_file}")
        try:
            df = pd.read_csv(self.cfg.animation_file)
            missing_cols = [name for name in self.cfg.csv_column_joint_names if name not in df.columns]
            if missing_cols:
                # print(f"DEBUG: Missing CSV columns: {missing_cols}. Available: {df.columns.tolist()}")
                raise KeyError(f"Joints {missing_cols} missing from CSV. Available: {df.columns.tolist()}.")
            animation_relevant_df = df[self.cfg.csv_column_joint_names]
            animation_np = animation_relevant_df.values
            self.animation_pos_data = torch.tensor(animation_np, dtype=torch.float32, device=self.device)
            self.max_animation_steps = self.animation_pos_data.shape[0]

            # print(f"DEBUG: Animation data shape: {self.animation_pos_data.shape}, steps: {self.max_animation_steps}")
            # if self.max_animation_steps > 5:
            #     print(f"DEBUG: First 3 rows (first 5 joints):\n{self.animation_pos_data[:3, :5]}")

            if self.max_animation_steps == 0:
                print("[MimicEnv _load_animation_data] CRITICAL: CSV loaded 0 frames.")
            elif torch.all(torch.isclose(self.animation_pos_data, torch.zeros_like(self.animation_pos_data))):
                print(
                    "[MimicEnv _load_animation_data] WARNING: Loaded animation data is all zeros. Check CSV (units should be radians)."
                )
            else:
                print(f"[INFO] Successfully loaded {self.max_animation_steps} animation frames for playback.")
        except FileNotFoundError:
            print(f"ERROR: Animation file not found: '{self.cfg.animation_file}'.")
            self.animation_pos_data = torch.empty((0, self.num_mimic_joints), device=self.device, dtype=torch.float32)
            self.max_animation_steps = 0
        except Exception as e:
            print(f"ERROR loading animation data from '{self.cfg.animation_file}': {e}")
            self.animation_pos_data = torch.empty((0, self.num_mimic_joints), device=self.device, dtype=torch.float32)
            self.max_animation_steps = 0
        if self.max_animation_steps == 0:
            print("[MimicEnv _load_animation_data] CRITICAL: Failed to load animation for playback.")

    def _reset_idx(self, env_ids: torch.Tensor):
        """Resets the state for specified environments."""
        # if self.num_envs > 0 and env_ids.numel() > 0:
        #     print(
        #         f"DEBUG: MimicEnv._reset_idx - START - env_ids: {env_ids.tolist()}, current_animation_step BEFORE any action: {self.current_animation_step[env_ids].tolist()}"
        #     )
        # Reset animation-specific states first
        if env_ids.numel() > 0:
            self.current_animation_step[env_ids] = 0
            self.previous_actions[env_ids] = 0.0
            # print(f"DEBUG: MimicEnv._reset_idx - Set current_animation_step[{env_ids.tolist()}] to 0.")

            # Reset disturbance states
            if self.cfg.external_disturbance.enable_disturbances:
                self.disturbance_forces[env_ids] = 0.0
                self.disturbance_torques[env_ids] = 0.0
                self.disturbance_remaining_time[env_ids] = 0.0
                self.disturbance_cooldown_time[env_ids] = 0.0
                self.episode_start_time[env_ids] = 0.0

                # Clear any active forces on reset
                try:
                    # Create force/torque tensors only for environments being reset
                    if isinstance(env_ids, torch.Tensor):
                        num_reset_envs = len(env_ids)
                    else:
                        num_reset_envs = self.num_envs

                    zero_forces = torch.zeros((num_reset_envs, len(self.robot.body_names), 3), device=self.device)
                    zero_torques = torch.zeros((num_reset_envs, len(self.robot.body_names), 3), device=self.device)

                    self.robot.set_external_force_and_torque(forces=zero_forces, torques=zero_torques, env_ids=env_ids)
                except Exception as e:
                    print(f"[WARNING] Could not clear forces on reset: {e}")

            # Clear force visualizations on reset
            if self.force_visualization_markers is not None:
                # If all environments are being reset, hide all arrows
                if len(env_ids) == self.num_envs:
                    self.force_visualization_markers.set_visibility(False)
                else:
                    # Otherwise, update visualization to reflect cleared forces
                    self._visualize_forces()

        # Call parent reset logic
        super()._reset_idx(env_ids)

        # if self.num_envs > 0 and env_ids.numel() > 0:
        # print(
        #     f"DEBUG: MimicEnv._reset_idx - END - env_ids: {env_ids.tolist()}, current_animation_step AFTER super()._reset_idx(): {self.current_animation_step[env_ids].tolist()}"
        # )

    def _apply_action(self) -> None:
        """Processes and applies the actions from the RL agent to the robot."""
        # Normalize actions to [-1, 1] range using tanh for safety
        processed_actions = torch.tanh(self.actions)

        # Apply actions based on the configured control mode
        if self.cfg.control_mode == "velocity":
            scaled_target_velocities = scale(
                processed_actions, self.mimic_joint_vel_limits_lower, self.mimic_joint_vel_limits_upper
            )
            self.robot.set_joint_velocity_target(scaled_target_velocities, joint_ids=self.mimic_joint_indices_in_robot)
        elif self.cfg.control_mode == "position":
            scaled_target_positions = self.scale_action(processed_actions)
            self.robot.set_joint_position_target(scaled_target_positions, joint_ids=self.mimic_joint_indices_in_robot)
        else:
            raise ValueError(f"Unsupported control_mode: '{self.cfg.control_mode}'.")

    def _get_gt(self) -> torch.Tensor:
        """Constructs the ground-truth observation for the mimicry task."""
        # Get current state of the mimicked joints
        current_mimic_joints_pos = self.robot.data.joint_pos[:, self.mimic_joint_indices_in_robot]
        current_mimic_joints_vel = self.robot.data.joint_vel[:, self.mimic_joint_indices_in_robot]

        # Get the target pose from the animation data for the current animation step
        target_animation_joint_pos = torch.zeros_like(current_mimic_joints_pos)
        if self.max_animation_steps > 0:
            safe_anim_indices = torch.clamp(self.current_animation_step, 0, self.max_animation_steps - 1)
            target_animation_joint_pos = self.animation_pos_data[safe_anim_indices, :]

        # Concatenate into the final ground-truth observation tensor: (current_pos, current_vel, target_pos)
        return torch.cat((current_mimic_joints_pos, current_mimic_joints_vel, target_animation_joint_pos), dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        """Calculates rewards based on the robot's mimicry performance."""
        # Get current robot state and target animation pose
        current_mimic_joints_pos = self.robot.data.joint_pos[:, self.mimic_joint_indices_in_robot]
        current_mimic_joints_vel = self.robot.data.joint_vel[:, self.mimic_joint_indices_in_robot]
        target_animation_joint_pos = torch.zeros_like(current_mimic_joints_pos)
        if self.max_animation_steps > 0:
            safe_anim_indices = torch.clamp(self.current_animation_step, 0, self.max_animation_steps - 1)
            target_animation_joint_pos = self.animation_pos_data[safe_anim_indices, :]

        # Compute reward components by calling the reward function
        total_reward, pos_track_rew, staying_alive_rew, current_vel_pen, action_smooth_pen = (
            compute_mimic_rewards_simplified(
                current_mimic_joints_pos,
                target_animation_joint_pos,
                current_mimic_joints_vel,
                self.actions,
                self.previous_actions,
                self.cfg.rewards,
                self.num_mimic_joints,
            )
        )

        # Update previous actions for the next step's smoothness penalty
        self.previous_actions = self.actions.clone()

        # Log reward components for debugging and analysis
        log = self.extras["log"]
        log["mimic_pos_tracking_reward"] = pos_track_rew
        log["mimic_staying_alive_reward"] = staying_alive_rew
        log["mimic_current_vel_penalty"] = current_vel_pen
        log["mimic_action_smoothness_penalty"] = action_smooth_pen
        log["mimic_total_reward"] = total_reward
        log["current_animation_frame"] = self.current_animation_step.float()

        # Update termination tracking
        if hasattr(self, "joint_limit_violations"):
            log["joint_limit_violations"] = self.joint_limit_violations
        else:
            log["joint_limit_violations"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        if hasattr(self, "torso_tilt_violations"):
            log["torso_tilt_violations"] = self.torso_tilt_violations
        else:
            log["torso_tilt_violations"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        # Update disturbance tracking
        log["disturbance_active"] = (self.disturbance_remaining_time > 0).float()
        log["disturbance_force_magnitude"] = torch.norm(self.disturbance_forces, dim=1)

        return total_reward

    def _apply_external_disturbances(self):
        """
        Apply random external disturbances to the robot for robustness training.

        This system applies random forces to a target body on the robot to simulate
        unexpected external disturbances. Forces can be biased in certain directions
        (e.g., downward for gravity-like effects) and include configurable timing,
        magnitude, and visualization.
        """
        if not self.cfg.external_disturbance.enable_disturbances:
            return

        if not hasattr(self, "disturbance_body_id") or self.disturbance_body_id is None:
            return

        dt = self.control_dt

        # Update timers
        self.disturbance_remaining_time = torch.clamp(self.disturbance_remaining_time - dt, min=0.0)
        self.disturbance_cooldown_time = torch.clamp(self.disturbance_cooldown_time - dt, min=0.0)
        self.episode_start_time = self.episode_start_time + dt

        # Check if we're still in the episode start timeout period
        in_timeout = self.episode_start_time < self.cfg.external_disturbance.episode_start_timeout

        # Find environments that can receive new disturbances
        can_disturb = (self.disturbance_remaining_time == 0) & (self.disturbance_cooldown_time == 0) & (~in_timeout)

        # Randomly select which environments to disturb
        disturb_mask = can_disturb & (
            torch.rand(self.num_envs, device=self.device) < self.cfg.external_disturbance.disturbance_probability
        )

        if disturb_mask.any():
            num_disturbed = disturb_mask.sum()
            cfg = self.cfg.external_disturbance
            # print(f"[DEBUG] Applying disturbances to {num_disturbed} environments: {torch.where(disturb_mask)[0].tolist()}")

            # Generate random force magnitudes
            force_magnitudes = torch.zeros(self.num_envs, device=self.device)
            force_magnitudes[disturb_mask] = sample_uniform(
                cfg.force_magnitude_range[0], cfg.force_magnitude_range[1], (num_disturbed,), device=self.device
            )

            # Generate force directions with bias
            if cfg.bias_strength > 0.0:
                # Normalize the bias direction
                bias_dir = torch.tensor(cfg.directional_bias, device=self.device, dtype=torch.float32)
                bias_dir = bias_dir / torch.norm(bias_dir)

                # Generate random directions
                random_directions = torch.zeros(self.num_envs, 3, device=self.device)
                random_directions[disturb_mask] = torch.randn(num_disturbed, 3, device=self.device)
                random_directions[disturb_mask] = random_directions[disturb_mask] / torch.norm(
                    random_directions[disturb_mask], dim=1, keepdim=True
                )

                # Blend random and bias directions
                final_directions = torch.zeros(self.num_envs, 3, device=self.device)
                final_directions[disturb_mask] = (
                    cfg.bias_strength * bias_dir.unsqueeze(0)
                    + (1.0 - cfg.bias_strength) * random_directions[disturb_mask]
                )
                # Normalize the blended direction
                final_directions[disturb_mask] = final_directions[disturb_mask] / torch.norm(
                    final_directions[disturb_mask], dim=1, keepdim=True
                )
            else:
                # Pure random directions (no bias)
                final_directions = torch.zeros(self.num_envs, 3, device=self.device)
                final_directions[disturb_mask] = torch.randn(num_disturbed, 3, device=self.device)
                final_directions[disturb_mask] = final_directions[disturb_mask] / torch.norm(
                    final_directions[disturb_mask], dim=1, keepdim=True
                )

            # Apply force only to disturbed environments
            self.disturbance_forces[disturb_mask] = (
                force_magnitudes[disturb_mask].unsqueeze(1) * final_directions[disturb_mask]
            )

            # Set durations for new disturbances
            self.disturbance_remaining_time[disturb_mask] = sample_uniform(
                cfg.duration_range[0], cfg.duration_range[1], (num_disturbed,), device=self.device
            )

            # Set cooldown for next disturbance
            self.disturbance_cooldown_time[disturb_mask] = sample_uniform(
                cfg.interval_range[0], cfg.interval_range[1], (num_disturbed,), device=self.device
            )

        # Clear forces for environments where disturbance has ended
        force_ended = self.disturbance_remaining_time == 0
        self.disturbance_forces[force_ended] = 0.0

        # Apply external forces to robot bodies
        if self.disturbance_forces.any():
            active_envs = torch.where(torch.norm(self.disturbance_forces, dim=1) > 0)[0]
            if len(active_envs) > 0:
                # # Print active forces occasionally for debugging
                # if not hasattr(self, "_last_active_print_time") or (self.global_env_steps_counter % 60 == 0):
                #     print(f"[DEBUG] Active forces in {len(active_envs)} envs: {active_envs.tolist()[:5]}")
                #     print(f"[DEBUG] Force magnitudes: {torch.norm(self.disturbance_forces[active_envs[:3]], dim=1).tolist()}")
                #     self._last_active_print_time = self.global_env_steps_counter

                # Apply external forces using Isaac Lab's standard API only
                try:
                    # Get body orientations (quaternions) for the disturbance body
                    # body_state_w is (num_envs, num_bodies, 13) where indices 3:7 are quaternion (w,x,y,z)
                    body_orientations = self.robot.data.body_state_w[:, self.disturbance_body_id, 3:7]

                    # Transform global forces to local body frame
                    # quat_apply_inverse rotates vectors from world to body frame
                    local_forces = quat_apply_inverse(body_orientations, self.disturbance_forces)

                    # Create force tensor in proper format: (num_envs, num_bodies, 3)
                    external_forces = torch.zeros((self.num_envs, len(self.robot.body_names), 3), device=self.device)
                    external_forces[:, self.disturbance_body_id, :] = local_forces

                    external_torques = torch.zeros_like(external_forces)

                    # Apply using Isaac Lab API
                    self.robot.set_external_force_and_torque(forces=external_forces, torques=external_torques)

                    # print(f"[DEBUG] Applied forces using Isaac Lab standard API with global-to-local transformation")

                except Exception as e:
                    print(f"[ERROR] Force application failed: {e}")

    def _visualize_forces(self):
        """Visualize the external forces using arrow markers."""
        if self.force_visualization_markers is None or not self.sim.has_gui():
            return

        # Find which environments have active forces
        active_forces = torch.norm(self.disturbance_forces, dim=1) > 0
        num_active = active_forces.sum().item()

        if num_active == 0:
            # Hide all markers when no forces are active
            self.force_visualization_markers.set_visibility(False)
            return

        try:
            # Get the positions where forces are applied
            body_positions = self.robot.data.body_state_w[:, self.disturbance_body_id, :3]

            # Prepare data for all environments (we'll use marker indices to show only active ones)
            all_positions = body_positions.clone()

            # Convert force vectors to arrow orientations
            # Default orientation: arrow points along +X axis
            # We need to rotate it to align with force direction
            orientations = torch.zeros((self.num_envs, 4), device=self.device)
            orientations[:, 0] = 1.0  # Default quaternion (w=1, x=y=z=0)

            # Calculate arrow scales based on force magnitude
            force_magnitudes = torch.norm(self.disturbance_forces, dim=1)
            max_expected_force = self.cfg.external_disturbance.force_magnitude_range[1]

            # Scale arrows: length proportional to force magnitude
            # Default arrow scale from config is (1.0, 0.1, 0.1) for X, Y, Z
            base_scale = torch.tensor([1.0, 0.5, 0.5], device=self.device)
            scales = torch.zeros((self.num_envs, 3), device=self.device)

            # For active forces, calculate proper orientation and scale
            if num_active > 0:
                active_indices = torch.where(active_forces)[0]

                for idx in active_indices:
                    force_vec = self.disturbance_forces[idx]
                    force_mag = force_magnitudes[idx]

                    if force_mag > 0:
                        # Normalize force vector to get direction
                        force_dir = force_vec / force_mag

                        # Calculate quaternion to rotate from +X to force direction
                        # This is a simplified version - for more accuracy, use proper quaternion calculations
                        orientations[idx] = self._vector_to_quaternion(force_dir)

                        # Scale arrow based on force magnitude
                        scale_factor = (force_mag / max_expected_force) * 2.0  # Scale factor for visibility
                        scales[idx] = base_scale * scale_factor

            # Create marker indices for only active forces
            if num_active > 0:
                # Extract only the active force data
                active_positions = all_positions[active_forces]
                active_orientations = orientations[active_forces]
                active_scales = scales[active_forces]

                # All arrows use the same prototype (index 0)
                active_marker_indices = torch.zeros(num_active, dtype=torch.long, device=self.device)

                # Visualize only active arrows
                self.force_visualization_markers.visualize(
                    translations=active_positions,
                    orientations=active_orientations,
                    scales=active_scales,
                    marker_indices=active_marker_indices,
                )
                self.force_visualization_markers.set_visibility(True)

                # # Debug print (occasional)
                # if self.global_env_steps_counter % 60 == 0:
                #     print(f"[DEBUG] Visualizing {num_active} force arrows")
                #     for i in range(min(3, num_active)):
                #         idx = torch.where(active_forces)[0][i]
                #         print(f"  Arrow {i}: pos={all_positions[idx].cpu().numpy()}, force_mag={force_magnitudes[idx]:.1f}N")

        except Exception as e:
            print(f"[ERROR] Force visualization failed: {e}")
            import traceback

            traceback.print_exc()

    def _hide_ghost_base_visuals(self):
        """
        Hide the base-related visual meshes for the ghost robot.

        This method selectively hides visual components of the ghost robot's base,
        wheels, and lower torso to reduce visual clutter while maintaining visibility
        of the important articulated parts (arms, hands, upper torso).
        """
        try:
            from pxr import UsdGeom, Usd

            # Specific base-related prim names to hide (exact matches)
            base_exact_names = {
                "base_link",
                "base_footprint",
                "base_front_left_wheel_link",
                "base_front_right_wheel_link",
                "base_rear_left_wheel_link",
                "base_rear_right_wheel_link",
                "base_link_tip",
                "root",
                "root_2",
                "base_link_trans_x",
                "base_link_trans_y",
                "base_link_rot_yaw",
                "torso_link_0",
            }

            # Get the stage
            stage = self.sim.stage

            # For each environment
            for env_idx in range(self.num_envs):
                ghost_prim_path = f"/World/envs/env_{env_idx}/GhostKinematicRobot"

                # First, try to traverse all prims under the ghost robot to find base-related ones
                ghost_prim = stage.GetPrimAtPath(ghost_prim_path)
                if ghost_prim.IsValid():
                    # Traverse all descendants
                    for prim in Usd.PrimRange(ghost_prim):
                        prim_name = prim.GetName()
                        prim_path = str(prim.GetPath())

                        # Check if this prim is exactly one of our base components
                        # or if it contains wheel/base_link but NOT hand_base_link
                        should_hide = False

                        # Exact match check
                        if prim_name in base_exact_names:
                            should_hide = True
                        # Check for wheel links
                        elif "wheel" in prim_name.lower():
                            should_hide = True
                        # Check for base links but exclude hand_base_link
                        elif "base" in prim_name.lower() and "hand" not in prim_path.lower():
                            # Additional check: only hide if it's really a base component
                            if any(x in prim_name.lower() for x in ["base_link", "base_foot", "root"]):
                                should_hide = True

                        if should_hide:
                            # Try to hide the prim and all its visual children
                            self._hide_prim_and_visuals(prim)
                            # print(f"[DEBUG] Hiding prim: {prim.GetPath()}")

            print("[INFO] Hidden base visuals for ghost robot")

        except Exception as e:
            print(f"[WARNING] Could not hide ghost base visuals: {e}")
            import traceback

            traceback.print_exc()

    def _hide_prim_and_visuals(self, prim):
        """Hide a prim and all its visual representations."""
        from pxr import UsdGeom, Usd

        # Make the prim itself invisible
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.MakeInvisible()

        # Hide all child prims that might contain visuals
        for child in Usd.PrimRange(prim):
            child_imageable = UsdGeom.Imageable(child)
            if child_imageable:
                child_imageable.MakeInvisible()

    def _vector_to_quaternion(self, vec: torch.Tensor) -> torch.Tensor:
        """Convert a direction vector to a quaternion that rotates +X axis to that direction."""
        # Normalize the vector
        vec = vec / torch.norm(vec)

        # Default forward direction is +X
        forward = torch.tensor([1.0, 0.0, 0.0], device=vec.device)

        # Calculate rotation axis (cross product)
        axis = torch.cross(forward, vec)
        axis_length = torch.norm(axis)

        # Handle the case where vectors are parallel
        if axis_length < 1e-6:
            if torch.dot(forward, vec) > 0:
                # Same direction, no rotation needed
                return torch.tensor([1.0, 0.0, 0.0, 0.0], device=vec.device)
            else:
                # Opposite direction, rotate 180 degrees around Y or Z
                return torch.tensor([0.0, 0.0, 1.0, 0.0], device=vec.device)

        # Normalize axis
        axis = axis / axis_length

        # Calculate angle
        angle = torch.acos(torch.clamp(torch.dot(forward, vec), -1.0, 1.0))

        # Convert to quaternion using angle-axis representation
        half_angle = angle / 2.0
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)

        return torch.cat([w.unsqueeze(0), xyz])

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Override step to include external disturbances."""
        # Store action
        action = action.to(self.device)
        if hasattr(self.cfg, "action_noise_model") and self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        self._pre_physics_step(action)

        # Increment global step counter
        self.global_env_steps_counter += 1

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Perform physics stepping for the configured number of decimation steps
        for _ in range(self.cfg.decimation):
            if hasattr(self, "_sim_step_counter"):
                self._sim_step_counter += 1
            self._apply_action()

            # Apply external disturbances before writing to sim
            self._apply_external_disturbances()

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if (
                hasattr(self, "_sim_step_counter")
                and self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # Update force visualization
        self._visualize_forces()

        # Continue with rest of step logic
        self._compute_intermediate_values()

        # Update episode counters
        self.episode_length_buf += 1
        if hasattr(self, "common_step_counter"):
            self.common_step_counter += 1

        # Get rewards and dones
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # Reset environments that have finished
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            if hasattr(self.sim, "forward"):
                self.sim.forward()
            if (
                hasattr(self.sim, "has_rtx_sensors")
                and self.sim.has_rtx_sensors()
                and hasattr(self.cfg, "rerender_on_reset")
                and self.cfg.rerender_on_reset
            ):
                self.sim.render()

        # Apply interval-based events if configured
        if hasattr(self.cfg, "events") and self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Get final observations for the agent
        self.obs_buf = self._get_observations()

        # Apply observation noise if configured
        if hasattr(self.cfg, "observation_noise_model") and self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        Computes intermediate values for the environment.

        The core logic for updating animation steps and the ghost robot is centralized
        in the parent AIRECEnv._compute_intermediate_values method. This method just
        delegates the call to its parent.
        """
        super()._compute_intermediate_values(env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determines if episodes have terminated or been truncated."""
        # Termination condition: The animation sequence has completed (natural end)
        if self.max_animation_steps <= 0:
            terminated = torch.ones_like(self.current_animation_step, dtype=torch.bool, device=self.device)
            if not hasattr(self, "_no_anim_data_critical_warned_dones"):
                print("[MimicEnv _get_dones] CRITICAL: max_animation_steps <= 0. Forcing animation_completed=True.")
                self._no_anim_data_critical_warned_dones = True
        else:
            terminated = self.current_animation_step >= (self.max_animation_steps - 1)

        # Joint limit termination
        if self.cfg.termination.enable_joint_limit_termination:
            current_pos = self.robot.data.joint_pos[:, self.mimic_joint_indices_in_robot]
            lower_limits = self.robot.data.soft_joint_pos_limits[:, self.mimic_joint_indices_in_robot, 0]
            upper_limits = self.robot.data.soft_joint_pos_limits[:, self.mimic_joint_indices_in_robot, 1]

            buffer = self.cfg.termination.joint_limit_buffer
            joint_limit_violated = torch.any(
                (current_pos < lower_limits * buffer) | (current_pos > upper_limits * buffer), dim=1
            )
            terminated = terminated | joint_limit_violated

            # Store for logging (as float for mean calculation)
            if not hasattr(self, "joint_limit_violations"):
                self.joint_limit_violations = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            self.joint_limit_violations = joint_limit_violated.float()

        # Torso tilt termination
        if self.cfg.termination.enable_torso_tilt_termination:
            # Get torso joint indices
            torso_indices = []
            for joint_name in self.cfg.termination.torso_joints_to_check:
                if joint_name in self.robot.joint_names:
                    torso_indices.append(self.robot.joint_names.index(joint_name))

            if torso_indices:
                torso_positions = self.robot.data.joint_pos[:, torso_indices]
                torso_tilt_violated = torch.any(
                    torch.abs(torso_positions) > self.cfg.termination.torso_tilt_limit, dim=1
                )
                terminated = terminated | torso_tilt_violated

                # Store for logging (as float for mean calculation)
                if not hasattr(self, "torso_tilt_violations"):
                    self.torso_tilt_violations = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
                self.torso_tilt_violations = torso_tilt_violated.float()

        # Truncation condition: Episode length exceeds the maximum allowed time
        time_out = self.episode_length_buf >= self.max_episode_length
        truncated = time_out & (~terminated)

        return terminated, truncated


# =============================================================================
# Reward Function
# =============================================================================


def compute_mimic_rewards_simplified(
    current_positions: torch.Tensor,
    target_positions: torch.Tensor,
    current_velocities: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    rewards_cfg: RewardsCfg,
    num_tracked_joints: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the different components of the mimicry reward.

    Args:
        current_positions: The current joint positions of the robot.
        target_positions: The target joint positions from the animation.
        current_velocities: The current joint velocities of the robot.
        actions: The actions taken in the current step.
        previous_actions: The actions taken in the previous step.
        rewards_cfg: The configuration object for reward scaling factors.
        num_tracked_joints: The number of joints being tracked for mimicry.

    Returns:
        A tuple containing the total reward and its individual components.
    """
    batch_size = current_positions.shape[0]
    device = current_positions.device

    # -- Constant "staying alive" reward
    staying_alive_rew_component = torch.full(
        (batch_size,), rewards_cfg.staying_alive_reward, device=device, dtype=torch.float32
    )

    # -- Position tracking reward and velocity penalty (if joints are being tracked)
    if num_tracked_joints == 0:
        pos_tracking_reward = torch.zeros(batch_size, device=device)
        current_joint_vel_penalty = torch.zeros(batch_size, device=device)
    else:
        # Position tracking reward based on squared error (exponentially shaped)
        pos_error_sq_sum = torch.sum(torch.square(target_positions - current_positions), dim=-1)
        pos_variance_term = rewards_cfg.pos_error_variance_scale * float(num_tracked_joints)
        pos_variance_term = max(pos_variance_term, 1e-6)  # avoid division by zero
        pos_tracking_reward = (
            torch.exp(-pos_error_sq_sum / pos_variance_term) * rewards_cfg.joint_pos_tracking_reward_scale
        )
        # Velocity penalty to discourage excessive speed
        current_vel_sq_sum = torch.sum(torch.square(current_velocities), dim=-1)
        current_joint_vel_penalty = current_vel_sq_sum * rewards_cfg.current_joint_vel_penalty_scale

    # -- Action smoothness penalty to discourage jerky movements
    action_diff_sq_sum = torch.sum(torch.square(actions - previous_actions), dim=-1)
    action_smoothness_penalty = action_diff_sq_sum * rewards_cfg.action_smoothness_penalty_scale

    # -- Total reward is the sum of all components
    total_rewards = (
        pos_tracking_reward + staying_alive_rew_component + current_joint_vel_penalty + action_smoothness_penalty
    )

    return (
        total_rewards,
        pos_tracking_reward,
        staying_alive_rew_component,
        current_joint_vel_penalty,
        action_smoothness_penalty,
    )
