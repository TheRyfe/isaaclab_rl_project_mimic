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
    num_eval_envs: int = 8  # Number of envs at the START of the batch to use for evaluation (will show ghost).
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
        )
    )


# =============================================================================
# Environment
# =============================================================================


class MimicEnv(AIRECEnv):
    """
    An environment for training a robot to mimic a pre-recorded animation.
    """

    cfg: MimicEnvCfg
    ghost_robot: Articulation | None
    ghost_mimic_joint_indices: torch.Tensor | None

    def __init__(self, cfg: MimicEnvCfg, render_mode: str | None = None, **kwargs):
        """Initializes the mimicry environment."""
        self.cfg = cfg
        self.ghost_robot = None
        self.ghost_mimic_joint_indices = None

        # -- FIX: Initialize attributes used in _setup_scene BEFORE calling the parent constructor.
        # The parent constructor calls `_setup_scene`, which depends on these attributes.
        # We initialize them on the CPU first and move them to the correct device after super().__init__().
        self.eval_env_ids = torch.tensor([], dtype=torch.long)
        self.is_eval_env_mask = torch.zeros(cfg.scene.num_envs, dtype=torch.bool)
        if self.cfg.enable_ghost_visualizer and self.cfg.num_eval_envs > 0:
            num_envs = cfg.scene.num_envs
            if self.cfg.num_eval_envs > num_envs:
                print(f"[WARNING] num_eval_envs ({self.cfg.num_eval_envs}) > num_envs ({num_envs}). Clamping.")
                self.cfg.num_eval_envs = num_envs
            self.eval_env_ids = torch.arange(0, self.cfg.num_eval_envs, dtype=torch.long)
            if self.eval_env_ids.numel() > 0:
                self.is_eval_env_mask[self.eval_env_ids] = True
                print(f"[INFO] MimicEnv: Initialized eval envs (on CPU): {self.eval_env_ids.tolist()}")

        # -- Pre-initialization checks and dynamic config adjustments
        provisional_physics_dt = cfg.sim.dt
        provisional_decimation = cfg.decimation
        provisional_control_dt = provisional_physics_dt * provisional_decimation

        if not math.isclose(provisional_control_dt, cfg.animation_dt_info, rel_tol=1e-5):
            print(
                f"[MimicEnv __init__] CONFIG WARNING: Provisional control_dt ({provisional_control_dt:.6f}s) "
                f"does NOT match cfg.animation_dt_info ({cfg.animation_dt_info:.6f}s)."
            )

        self.max_animation_steps = 0
        self._load_animation_data_static(cfg.animation_file, cfg.csv_column_joint_names)
        self._mimic_env_determined_max_animation_steps = self.max_animation_steps

        required_episode_length_s = cfg.episode_length_s
        if self._mimic_env_determined_max_animation_steps > 0 and provisional_control_dt > 0:
            calculated_animation_duration_s = self._mimic_env_determined_max_animation_steps * provisional_control_dt
            required_episode_length_s_for_anim = calculated_animation_duration_s + cfg.dynamic_episode_length_buffer_s
            if required_episode_length_s_for_anim > required_episode_length_s:
                print(f"[MimicEnv __init__ PRE-SUPER] INFO: Original cfg.episode_length_s: {cfg.episode_length_s:.2f}s.")
                print(
                    f"[MimicEnv __init__ PRE-SUPER] INFO: Animation requires {self._mimic_env_determined_max_animation_steps} steps. "
                    f"Desired episode length is {required_episode_length_s_for_anim:.2f}s."
                )
                required_episode_length_s = required_episode_length_s_for_anim

        modified_parent_cfg = copy.deepcopy(cfg)
        modified_parent_cfg.episode_length_s = required_episode_length_s
        self.num_mimic_joints = len(cfg.csv_column_joint_names)
        if self.num_mimic_joints == 0:
            raise ValueError("'MimicEnvCfg.csv_column_joint_names' cannot be empty.")

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

        modified_parent_cfg.actuated_joint_names = self.robot_mimicked_joint_names_ordered
        modified_parent_cfg.num_actions = self.num_mimic_joints
        modified_parent_cfg.num_gt_observations = self.num_mimic_joints * 3
        if "prop" in modified_parent_cfg.obs_list:
            cfg_num_prop_joints = cfg.num_prop_joints
            modified_parent_cfg.num_prop_observations = cfg_num_prop_joints * 2 + 7 * 2 + self.num_mimic_joints

        # Call the parent constructor, which will trigger _setup_scene
        super().__init__(cfg=modified_parent_cfg, render_mode=render_mode, **kwargs)

        # -- Post-super initializations
        # Now that the parent is initialized, self.device is available.
        self.eval_env_ids = self.eval_env_ids.to(self.device)
        self.is_eval_env_mask = self.is_eval_env_mask.to(self.device)

        _physics_dt_final = self.sim.get_physics_dt()
        self.control_dt = _physics_dt_final * self.cfg.decimation
        if self.control_dt <= 0:
            raise ValueError("[MimicEnv __init__ POST-SUPER] CRITICAL: Final control_dt must be positive.")

        if hasattr(self.cfg, "animation_dt_info") and not math.isclose(
            self.control_dt, self.cfg.animation_dt_info, rel_tol=1e-5
        ):
            print("[MimicEnv __init__] POST-SUPER WARNING: Final control_dt does NOT match cfg.animation_dt_info.")
        else:
            print("[MimicEnv __init__] POST-SUPER INFO: Final control_dt matches animation_dt_info.")

        self._load_animation_data()

        if hasattr(self, "max_animation_steps") and self.max_animation_steps > 0 and self.control_dt > 0:
            if self.max_episode_length < self.max_animation_steps:
                print(
                    f"[MimicEnv __init__] POST-SUPER CRITICAL WARNING: Final max_episode_length ({self.max_episode_length}) "
                    f"is STILL SHORTER than animation steps ({self.max_animation_steps})."
                )

        try:
            self.mimic_joint_indices_in_robot = torch.tensor(
                [self.robot.joint_names.index(name) for name in self.robot_mimicked_joint_names_ordered],
                device=self.device,
                dtype=torch.long,
            )
        except (ValueError, AttributeError) as e:
            print(f"ERROR creating mimic_joint_indices_in_robot: {e}")
            raise

        if self.robot.data.joint_vel_limits is None or self.robot.data.joint_vel_limits.numel() == 0:
            self.mimic_joint_vel_limits_lower = torch.full((self.num_mimic_joints,), -10.0, device=self.device, dtype=torch.float32)
            self.mimic_joint_vel_limits_upper = torch.full((self.num_mimic_joints,), 10.0, device=self.device, dtype=torch.float32)
        else:
            raw_mimic_joint_vel_limits = self.robot.data.joint_vel_limits[self.mimic_joint_indices_in_robot, :].clone()
            self.mimic_joint_vel_limits_lower = raw_mimic_joint_vel_limits[:, 0] * 0.8
            self.mimic_joint_vel_limits_upper = raw_mimic_joint_vel_limits[:, 1] * 0.8
        
        problematic_limits_mask = torch.isclose(self.mimic_joint_vel_limits_lower, torch.tensor(0.0, device=self.device)) & torch.isclose(self.mimic_joint_vel_limits_upper, torch.tensor(0.0, device=self.device))
        if torch.any(problematic_limits_mask):
            self.mimic_joint_vel_limits_lower[problematic_limits_mask] = -0.1
            self.mimic_joint_vel_limits_upper[problematic_limits_mask] = 0.1

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
            print("[MimicEnv __init__] WARNING: Animation data appears empty.")

    def _setup_scene(self):
        """Sets up the simulation scene, including the ghost robot."""
        super()._setup_scene()

        if self.cfg.enable_ghost_visualizer:
            self.ghost_robot = Articulation(self.cfg.ghost_robot_cfg)
            self.scene.articulations["ghost_robot"] = self.ghost_robot
            print("[INFO] Ghost visualizer robot added to the scene.")

            # Find the corresponding joint indices on the ghost robot
            if hasattr(self.ghost_robot, "joint_names"):
                try:
                    self.ghost_mimic_joint_indices = torch.tensor(
                        [self.ghost_robot.joint_names.index(name) for name in self.robot_mimicked_joint_names_ordered],
                        device=self.device,
                        dtype=torch.long,
                    )
                except (ValueError, AttributeError) as e:
                    print(f"ERROR creating ghost_mimic_joint_indices: {e}")
                    self.ghost_mimic_joint_indices = None
            
            # Hide ghost in non-evaluation (training) environments to improve performance
            if self.eval_env_ids.numel() < self.num_envs:
                # Get IDs of training envs by inverting the evaluation env mask
                train_env_ids = torch.where(~self.is_eval_env_mask)[0]

                if train_env_ids.numel() > 0:
                    self.ghost_robot.set_visibility(False, env_ids=train_env_ids)
                    print(f"[INFO] Made ghost robot invisible for {train_env_ids.numel()} training environments.")

    def _load_animation_data_static(self, animation_file_path: str, csv_columns: list[str]):
        """Loads animation data from a CSV file just to determine its length."""
        try:
            df = pd.read_csv(animation_file_path)
            missing_cols = [name for name in csv_columns if name not in df.columns]
            if missing_cols:
                raise KeyError(f"Joints {missing_cols} missing from CSV '{animation_file_path}'.")
            self.max_animation_steps = len(df)
            if self.max_animation_steps == 0:
                print(f"[MimicEnv] CRITICAL: CSV '{animation_file_path}' loaded 0 frames.")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR loading animation data statically from '{animation_file_path}': {e}")
            self.max_animation_steps = 0

    def _load_animation_data(self):
        """Loads the full animation data from a CSV file into a runtime tensor."""
        try:
            df = pd.read_csv(self.cfg.animation_file)
            animation_relevant_df = df[self.cfg.csv_column_joint_names]
            animation_np = animation_relevant_df.values
            self.animation_pos_data = torch.tensor(animation_np, dtype=torch.float32, device=self.device)
            self.max_animation_steps = self.animation_pos_data.shape[0]

            if self.max_animation_steps == 0:
                print("[MimicEnv] CRITICAL: CSV loaded 0 frames for playback.")
            else:
                print(f"[MimicEnv] Successfully loaded {self.max_animation_steps} frames for playback.")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR loading animation data from '{self.cfg.animation_file}': {e}")
            self.animation_pos_data = torch.empty((0, self.num_mimic_joints), device=self.device, dtype=torch.float32)
            self.max_animation_steps = 0

    def _reset_idx(self, env_ids: torch.Tensor):
        """Resets the state for specified environments."""
        if env_ids.numel() > 0:
            self.current_animation_step[env_ids] = 0
            self.previous_actions[env_ids].zero_()
        super()._reset_idx(env_ids)

    def _apply_action(self) -> None:
        """Processes and applies the actions from the RL agent to the robot."""
        processed_actions = torch.tanh(self.actions)
        if self.cfg.control_mode == "position":
            scaled_target_positions = self.scale_action(processed_actions)
            self.robot.set_joint_position_target(scaled_target_positions, joint_ids=self.mimic_joint_indices_in_robot)
        else:  # velocity
            scaled_target_velocities = scale(processed_actions, self.mimic_joint_vel_limits_lower, self.mimic_joint_vel_limits_upper)
            self.robot.set_joint_velocity_target(scaled_target_velocities, joint_ids=self.mimic_joint_indices_in_robot)

    def _get_gt(self) -> torch.Tensor:
        """Constructs the ground-truth observation for the mimicry task."""
        current_mimic_joints_pos = self.robot.data.joint_pos[:, self.mimic_joint_indices_in_robot]
        current_mimic_joints_vel = self.robot.data.joint_vel[:, self.mimic_joint_indices_in_robot]
        target_animation_joint_pos = torch.zeros_like(current_mimic_joints_pos)
        if self.max_animation_steps > 0:
            safe_anim_indices = torch.clamp(self.current_animation_step, 0, self.max_animation_steps - 1)
            target_animation_joint_pos = self.animation_pos_data[safe_anim_indices, :]
        return torch.cat((current_mimic_joints_pos, current_mimic_joints_vel, target_animation_joint_pos), dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        """Calculates rewards based on the robot's mimicry performance."""
        current_mimic_joints_pos = self.robot.data.joint_pos[:, self.mimic_joint_indices_in_robot]
        current_mimic_joints_vel = self.robot.data.joint_vel[:, self.mimic_joint_indices_in_robot]
        target_animation_joint_pos = torch.zeros_like(current_mimic_joints_pos)
        if self.max_animation_steps > 0:
            safe_anim_indices = torch.clamp(self.current_animation_step, 0, self.max_animation_steps - 1)
            target_animation_joint_pos = self.animation_pos_data[safe_anim_indices, :]

        total_reward, pos_track_rew, staying_alive_rew, current_vel_pen, action_smooth_pen = compute_mimic_rewards_simplified(
            current_mimic_joints_pos, target_animation_joint_pos, current_mimic_joints_vel, self.actions,
            self.previous_actions, self.cfg.rewards, self.num_mimic_joints)
        
        self.previous_actions = self.actions.clone()
        
        log = self.extras["log"]
        log["mimic_pos_tracking_reward"] = pos_track_rew
        log["mimic_staying_alive_reward"] = staying_alive_rew
        log["mimic_current_vel_penalty"] = current_vel_pen
        log["mimic_action_smoothness_penalty"] = action_smooth_pen
        log["mimic_total_reward"] = total_reward
        log["current_animation_frame"] = self.current_animation_step.float()
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determines if episodes have terminated or been truncated."""
        if self.max_animation_steps <= 0:
            animation_completed = torch.ones_like(self.current_animation_step, dtype=torch.bool)
        else:
            animation_completed = self.current_animation_step >= (self.max_animation_steps - 1)
        
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = animation_completed
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
    """Calculates the different components of the mimicry reward."""
    batch_size = current_positions.shape[0]
    device = current_positions.device

    # -- Liveness Reward: Constant reward for not terminating
    staying_alive_rew_component = torch.full(
        (batch_size,), rewards_cfg.staying_alive_reward, device=device, dtype=torch.float32
    )

    # -- Reward/Penalties that depend on joint tracking
    if num_tracked_joints == 0:
        pos_tracking_reward = torch.zeros(batch_size, device=device)
        current_joint_vel_penalty = torch.zeros(batch_size, device=device)
    else:
        # -- Position Tracking Reward: Exponentially scaled reward for matching target joint positions
        pos_error_sq_sum = torch.sum(torch.square(target_positions - current_positions), dim=-1)
        pos_variance_term = rewards_cfg.pos_error_variance_scale * float(num_tracked_joints)
        pos_variance_term = max(pos_variance_term, 1e-6)  # Avoid division by zero
        pos_tracking_reward = (
            torch.exp(-pos_error_sq_sum / pos_variance_term) * rewards_cfg.joint_pos_tracking_reward_scale
        )

        # -- Velocity Penalty: Penalize high joint velocities to encourage smoother movements
        current_vel_sq_sum = torch.sum(torch.square(current_velocities), dim=-1)
        current_joint_vel_penalty = current_vel_sq_sum * rewards_cfg.current_joint_vel_penalty_scale

    # -- Action Smoothness Penalty: Penalize large changes in actions between steps
    action_diff_sq_sum = torch.sum(torch.square(actions - previous_actions), dim=-1)
    action_smoothness_penalty = action_diff_sq_sum * rewards_cfg.action_smoothness_penalty_scale

    # -- Total Reward
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