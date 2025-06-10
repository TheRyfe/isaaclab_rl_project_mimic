# =============================================================================
# To Do List
# =============================================================================
# - Change animation to walk in circle
# - Use old model with the base for training
# - Tweak the reward function, network and reset logic
# - Fix wrist bug
# - Add randomized external forces
# - Make ghost only applicable to the test envs only
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
from isaaclab.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaaclab.utils.configclass import MISSING

# =============================================================================
# Configuration
# =============================================================================


@configclass
class RewardsCfg:
    """Configuration for reward components and scales for the mimicry task."""

    joint_pos_tracking_reward_scale: float = 2.5
    staying_alive_reward: float = 0.01
    current_joint_vel_penalty_scale: float = -0.0001
    action_smoothness_penalty_scale: float = -0.005
    pos_error_variance_scale: float = 0.25


@configclass
class TerminationCfg:
    """Configuration for episode termination conditions."""

    terminate_on_high_error: bool = False
    max_avg_pos_error_threshold: float = 0.8
    max_avg_vel_error_threshold: float = 1.5
    pos_error_running_avg_alpha: float = 0.05
    vel_error_running_avg_alpha: float = 0.05
    error_termination_grace_period_steps: int = 10


# -- Ghost Robot Configuration
# Defines the USD asset and initial state for the ghost robot used for visualization.
KINEMATIC_GHOST_USD_FILE_PATH = "/home/simon/IsaacLab/SimScripts/AIREC_DeepMimic/isaaclab_rl/isaaclab_rl_project_mimic/assets/airec/airec_ghost.usd"

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
    animation_file: str = "/home/simon/IsaacLab/SimScripts/AIREC_DeepMimic/isaaclab_rl/isaaclab_rl_project_mimic/assets/animation/recording4RL.csv"
    animation_dt_info: float = 1.0 / 60.0
    dynamic_episode_length_buffer_s: float = 2.0

    # -- Robot control parameters
    num_base_actions: int = 0
    num_prop_joints: int = 20
    csv_column_joint_names: list[str] = [
        "H1", "H2", "H3",
        "R1", "R2", "R3", "R4", "R5", "R6", "R7",
        "L1", "L2", "L3", "L4", "L5", "L6", "L7",
        "T1", "T2", "T3",
    ]
    obs_list: list[str] = ["gt", "prop"]
    control_mode: str = "position"

    # -- Sub-configurations
    rewards: RewardsCfg = RewardsCfg()
    termination: TerminationCfg = TerminationCfg()

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
            # To set the ghost's opacity and color, uncomment the following lines.
            # This requires importing PreviewSurfaceCfg from the correct module.
            # visual_material=PreviewSurfaceCfg(
            #     opacity=0.4,
            #     diffuse_color=(0.3, 0.3, 0.8)
            # )
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
        """Initializes the mimicry environment."""
        #print("DEBUG: MimicEnv.__init__ - START")
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
        #print(
        #    f"DEBUG: MimicEnv.__init__ (after static load) - _mimic_env_determined_max_animation_steps: {self._mimic_env_determined_max_animation_steps}"
        #)

        # Dynamically adjust episode length to ensure the full animation can be played
        required_episode_length_s = cfg.episode_length_s
        if self._mimic_env_determined_max_animation_steps > 0 and provisional_control_dt > 0:
            calculated_animation_duration_s = self._mimic_env_determined_max_animation_steps * provisional_control_dt
            required_episode_length_s_for_anim = calculated_animation_duration_s + cfg.dynamic_episode_length_buffer_s
            if required_episode_length_s_for_anim > required_episode_length_s:
                print(f"[MimicEnv __init__ PRE-SUPER] INFO: Original cfg.episode_length_s: {cfg.episode_length_s:.2f}s.")
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
            "H1": "head_joint_1", "H2": "head_joint_2", "H3": "head_joint_3",
            "R1": "right_arm_joint_1", "R2": "right_arm_joint_2", "R3": "right_arm_joint_3",
            "R4": "right_arm_joint_4", "R5": "right_arm_joint_5", "R6": "right_arm_joint_6", "R7": "right_arm_joint_7",
            "L1": "left_arm_joint_1", "L2": "left_arm_joint_2", "L3": "left_arm_joint_3",
            "L4": "left_arm_joint_4", "L5": "left_arm_joint_5", "L6": "left_arm_joint_6", "L7": "left_arm_joint_7",
            "T1": "torso_joint_1", "T2": "torso_joint_2", "T3": "torso_joint_3",
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

        # Initialize the parent environment
        #print("DEBUG: MimicEnv.__init__ - Calling super().__init__()")
        super().__init__(cfg=modified_parent_cfg, render_mode=render_mode, **kwargs)
        #print("DEBUG: MimicEnv.__init__ - Returned from super().__init__()")

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
        print(
            f"DEBUG: MimicEnv.__init__ (after full load) - self.max_animation_steps: {self.max_animation_steps}, self.animation_pos_data exists: {hasattr(self, 'animation_pos_data')}"
        )

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
                    print("INFO: MimicEnv.__init__ - Successfully created ghost_mimic_joint_indices.")
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
        }
        if hasattr(self, "animation_pos_data") and self.max_animation_steps <= 0:
            print("[MimicEnv __init__] WARNING: Animation data appears empty after full loading. Mimicry may not function.")
        print("DEBUG: MimicEnv.__init__ - END")

    def _setup_scene(self):
        """Sets up the simulation scene, including the ghost robot."""
        print("DEBUG: MimicEnv._setup_scene - START")
        # Call parent setup to spawn the main robot and other base assets
        super()._setup_scene()
        print("DEBUG: MimicEnv._setup_scene - After super()._setup_scene()")

        # Spawn the ghost robot if enabled in the configuration
        if self.cfg.enable_ghost_visualizer:
            self.ghost_robot = Articulation(self.cfg.ghost_robot_cfg)
            self.scene.articulations["ghost_robot"] = self.ghost_robot
            print("[INFO] Ghost visualizer robot added to the scene.")
        print("DEBUG: MimicEnv._setup_scene - END")

    def _load_animation_data_static(self, animation_file_path: str, csv_columns: list[str]):
        """Loads animation data from a CSV file just to determine its length for pre-init calculations."""
        print(f"DEBUG: _load_animation_data_static - Loading: {animation_file_path}")
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
            print(f"DEBUG: _load_animation_data_static - Loaded {self.max_animation_steps} frames for setup.")
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
        print(f"DEBUG: _load_animation_data - Loading: {self.cfg.animation_file}")
        try:
            df = pd.read_csv(self.cfg.animation_file)
            missing_cols = [name for name in self.cfg.csv_column_joint_names if name not in df.columns]
            if missing_cols:
                print(
                    f"DEBUG: _load_animation_data - Missing CSV columns: {missing_cols}. Available columns in CSV: {df.columns.tolist()}"
                )
                raise KeyError(f"Joints {missing_cols} missing from CSV. Available: {df.columns.tolist()}.")
            animation_relevant_df = df[self.cfg.csv_column_joint_names]
            animation_np = animation_relevant_df.values
            self.animation_pos_data = torch.tensor(animation_np, dtype=torch.float32, device=self.device)
            self.max_animation_steps = self.animation_pos_data.shape[0]

            print(
                f"DEBUG: _load_animation_data - self.animation_pos_data.shape: {self.animation_pos_data.shape}, self.max_animation_steps set to: {self.max_animation_steps}"
            )
            if self.max_animation_steps > 5:
                print(
                    f"DEBUG: _load_animation_data - First 3 rows of self.animation_pos_data[:, 0:5] (first 5 joints):\n{self.animation_pos_data[:3, :5]}"
                )

            if self.max_animation_steps == 0:
                print("[MimicEnv _load_animation_data] CRITICAL: CSV loaded 0 frames.")
            elif torch.all(torch.isclose(self.animation_pos_data, torch.zeros_like(self.animation_pos_data))):
                print("[MimicEnv _load_animation_data] WARNING: Loaded animation data is all zeros. Check CSV (units should be radians).")
            else:
                print(f"[MimicEnv _load_animation_data] Successfully loaded {self.max_animation_steps} frames for playback.")
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
        if self.num_envs > 0 and env_ids.numel() > 0:
            print(
                f"DEBUG: MimicEnv._reset_idx - START - env_ids: {env_ids.tolist()}, current_animation_step BEFORE any action: {self.current_animation_step[env_ids].tolist()}"
            )
        # Reset animation-specific states first
        if env_ids.numel() > 0:
            self.current_animation_step[env_ids] = 0
            self.previous_actions[env_ids] = 0.0
            print(f"DEBUG: MimicEnv._reset_idx - Set current_animation_step[{env_ids.tolist()}] to 0.")

        # Call parent reset logic
        super()._reset_idx(env_ids)

        if self.num_envs > 0 and env_ids.numel() > 0:
            print(
                f"DEBUG: MimicEnv._reset_idx - END - env_ids: {env_ids.tolist()}, current_animation_step AFTER super()._reset_idx(): {self.current_animation_step[env_ids].tolist()}"
            )

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
        total_reward, pos_track_rew, staying_alive_rew, current_vel_pen, action_smooth_pen = compute_mimic_rewards_simplified(
            current_mimic_joints_pos,
            target_animation_joint_pos,
            current_mimic_joints_vel,
            self.actions,
            self.previous_actions,
            self.cfg.rewards,
            self.num_mimic_joints,
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
        
        #WanDB
        self.extras["counters"] = {}

        return total_reward

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
        # Termination condition 1: The animation sequence has completed
        if self.max_animation_steps <= 0:
            animation_completed = torch.ones_like(self.current_animation_step, dtype=torch.bool, device=self.device)
            if not hasattr(self, "_no_anim_data_critical_warned_dones"):
                print("[MimicEnv _get_dones] CRITICAL: max_animation_steps <= 0. Forcing animation_completed=True.")
                self._no_anim_data_critical_warned_dones = True
        else:
            animation_completed = self.current_animation_step >= (self.max_animation_steps - 1)

        # Truncation condition: Episode length exceeds the maximum allowed time
        time_out = self.episode_length_buf >= self.max_episode_length

        # Optional termination condition: High tracking error (currently disabled)
        terminated_by_error = torch.zeros_like(animation_completed, dtype=torch.bool)
        if self.cfg.termination.terminate_on_high_error:
            pass
        
        # Combine termination conditions
        terminated = animation_completed | terminated_by_error
        # Truncation occurs on timeout if not already terminated
        truncated = time_out & (~terminated)

        # Log episode ending information for debugging
        if hasattr(self, "global_env_steps_counter") and (torch.any(terminated) or torch.any(truncated)):
            done_envs = torch.where(terminated | truncated)[0].tolist()
            terminated_envs = torch.where(terminated)[0].tolist()
            truncated_envs = torch.where(truncated)[0].tolist()
            print(f"DEBUG: _get_dones - GlobalStep: {self.global_env_steps_counter} - Episodes ENDING for envs: {done_envs}")
            if terminated_envs:
                print(f"DEBUG: _get_dones - TERMINATED envs: {terminated_envs}")
            if truncated_envs:
                print(f"DEBUG: _get_dones - TRUNCATED envs: {truncated_envs}")

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