# =============================================================================
# Imports
# =============================================================================
from __future__ import annotations

import math
import time
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

# AIREC specific imports
from assets.airec import AIREC_CFG

# Isaac Lab imports
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import (
    ContactSensor,
    ContactSensorCfg,
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
)
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
)

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)


# =============================================================================
# Configuration
# =============================================================================


@configclass
class AIRECEnvCfg(DirectRLEnvCfg):
    """Configuration for the base AIREC robot environment."""

    # -- Physics and Simulation settings
    physics_dt = 1 / 120
    decimation = 2
    render_interval = 2
    episode_length_s = 5.0

    # -- RL settings
    obs_stack = 1
    control_mode: str = "position"
    num_base_actions: int = 2

    # -- Simulation Configuration
    sim: SimulationCfg = SimulationCfg(
        dt=physics_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(bounce_threshold_velocity=0.2),
    )

    # -- Scene and Asset Configuration
    robot_cfg: ArticulationCfg = AIREC_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)

    # -- Default Joint Sets (can be overridden by child configs)
    actuated_base_joints = [
        "base_joint_trans_x",
        "base_joint_trans_y",
        "base_joint_rot_yaw",
    ]
    actuated_torso_joints = ["torso_joint_1", "torso_joint_2", "torso_joint_3"]
    actuated_head_joints = ["head_joint_1", "head_joint_2", "head_joint_3"]
    actuated_larm_joints = [
        "left_arm_joint_1", "left_arm_joint_2", "left_arm_joint_3", "left_arm_joint_4",
        "left_arm_joint_5", "left_arm_joint_6", "left_arm_joint_7",
    ]
    actuated_rarm_joints = [
        "right_arm_joint_1", "right_arm_joint_2", "right_arm_joint_3", "right_arm_joint_4",
        "right_arm_joint_5", "right_arm_joint_6", "right_arm_joint_7",
    ]
    actuated_lhand_joints = [
        "left_hand_first_finger_joint_1", "left_hand_second_finger_joint_1", "left_hand_third_finger_joint_1",
        "left_hand_thumb_joint_1", "left_hand_thumb_joint_2", "left_hand_thumb_joint_3",
        "left_hand_first_finger_joint_2", "left_hand_second_finger_joint_2", "left_hand_third_finger_joint_2",
        "left_hand_thumb_joint_4",
    ]
    actuated_rhand_joints = [
        "right_hand_first_finger_joint_1", "right_hand_second_finger_joint_1", "right_hand_third_finger_joint_1",
        "right_hand_thumb_joint_1", "right_hand_thumb_joint_2", "right_hand_thumb_joint_3",
        "right_hand_first_finger_joint_2", "right_hand_second_finger_joint_2", "right_hand_third_finger_joint_2",
        "right_hand_thumb_joint_4",
    ]
    base_wheels = [
        "base_front_left_wheel_joint", "base_front_right_wheel_joint", "base_rear_left_wheel_joint",
        "base_rear_right_wheel_joint",
    ]
    actuated_joint_names = actuated_head_joints + actuated_torso_joints + actuated_larm_joints + actuated_rarm_joints

    # -- Default RL Space Dimensions (will be re-calculated or overridden by child configs)
    num_actions = len(actuated_joint_names)
    num_actions += 2
    obs_list = ["prop"]
    num_gt_observations: int = 0
    num_states = 0
    num_prop_joints = 40
    num_prop_observations = num_prop_joints * 2 + num_actions + 7 * 2
    num_observations = 0
    action_space = num_actions
    observation_space = num_observations
    state_space = num_states

    # -- Hand Frame Sensor Configuration
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/EndEffectorFrameTransformer"
    lhand_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_hand_palm_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.02]),
            )
        ],
    )
    rhand_config: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_hand_palm_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.02]),
            )
        ],
    )

    # -- Other settings
    write_image_to_file = False
    aux_list = []
    normalise_prop = True
    normalise_pixels = True
    frame_stack = 1
    random_crop = False
    num_cameras = 1
    object_type = "rigid"
    img_dim = 84


# =============================================================================
# Environment
# =============================================================================


class AIRECEnv(DirectRLEnv):
    """
    A base reinforcement learning environment for the AIREC robot.

    This class provides the core functionalities for simulating the AIREC robot,
    including setting up the scene, applying actions, and computing observations.
    It is designed to be inherited by task-specific environments like MimicEnv.
    """

    cfg: AIRECEnvCfg

    def __init__(self, cfg: AIRECEnvCfg, render_mode: str | None = None, **kwargs):
        """Initializes the base AIREC environment."""
        self.obs_stack = 1
        # The configuration object `cfg` will be an instance of a child class's config (e.g., MimicEnvCfg) at runtime.
        self.cfg = cfg
        super().__init__(cfg, render_mode, **kwargs)

        # -- Initialize robot and environment properties
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        print(f"[INFO] Available joint names: {self.robot.data.joint_names}")
        print(f"[INFO] Number of joints: {len(self.robot.data.joint_names)}")
        print("[INFO] AIREC setup complete...")

        num_total_robot_joints = len(self.robot.joint_names)
        print("NUM JOINTS:", num_total_robot_joints)

        # Determine the indices of the joints that will be controlled by the agent's actions
        self.actuated_dof_indices = []
        for joint_name in self.cfg.actuated_joint_names:
            try:
                self.actuated_dof_indices.append(self.robot.joint_names.index(joint_name))
            except ValueError:
                print(f"[AIRECEnv __init__ WARNING] Actuated joint '{joint_name}' not found in robot.joint_names.")
        self.actuated_dof_indices.sort()

        num_actuated_joints = len(self.actuated_dof_indices)
        print("NUM ACTUATED JOINTS", num_actuated_joints)

        # Determine the indices for joints used in proprioceptive observations
        if self.cfg.num_prop_joints <= num_total_robot_joints and self.cfg.num_prop_joints > 0:
            self.prop_joint_indices = torch.arange(self.cfg.num_prop_joints, device=self.device, dtype=torch.long)
        else:
            print(
                f"[AIRECEnv __init__ WARNING] Invalid cfg.num_prop_joints ({self.cfg.num_prop_joints}). "
                f"Defaulting to all {num_total_robot_joints} joints for self.joint_pos/vel state."
            )
            self.prop_joint_indices = torch.arange(num_total_robot_joints, device=self.device, dtype=torch.long)

        # -- Initialize runtime tensors
        self.actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self.joint_pos = torch.zeros((self.num_envs, len(self.prop_joint_indices)), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, len(self.prop_joint_indices)), device=self.device)

        self.target_base_pos = torch.zeros((self.num_envs, 7), device=self.device)
        self.target_base_vel = torch.zeros((self.num_envs, 7), device=self.device)

        self.external_force_b = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)
        self.external_torque_b = torch.zeros((self.num_envs, self.robot.num_bodies, 3), device=self.device)

        self.lhand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.lhand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.rhand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.rhand_rot = torch.zeros((self.num_envs, 4), device=self.device)

        self._episode_timestep_counter = torch.zeros((self.num_envs,), dtype=torch.int16, device=self.device)

        # Initialize logging dictionary (can be extended by child classes)
        self.extras["log"] = {
            "l_dist_reward": None,
            "r_dist_reward": None,
            "joint_vel_penalty": None,
        }

    def _setup_scene(self):
        """Spawns and configures all assets in the simulation scene."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.lhand_frame = FrameTransformer(self.cfg.lhand_config)
        self.rhand_frame = FrameTransformer(self.cfg.rhand_config)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(1000, 1000)))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["lhand_frame"] = self.lhand_frame
        self.scene.sensors["rhand_frame"] = self.rhand_frame

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def configure_gym_env_spaces(self, obs_stack=1):
        """Wrapper to configure the Gym action and observation spaces."""
        self.obs_stack = self.cfg.obs_stack
        self._configure_gym_env_spaces()

    def _configure_gym_env_spaces(self):
        """Defines the structure and dimensions of the observation and action spaces."""
        print("CONFIGURING GYM SPACES (AIRECEnv)", self.obs_stack)
        # Calculate observation dimensions based on the config (provided by child env)
        self.num_gt_observations = self.cfg.num_gt_observations * self.obs_stack
        self.num_tactile_observations = 2 * self.obs_stack
        self.num_prop_observations = self.cfg.num_prop_observations * self.obs_stack

        self.num_actions = self.cfg.num_actions
        self.num_states = self.cfg.num_states
        self.state_space = None

        # Create observation space dictionary based on the obs_list from the config
        gym_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_prop_observations,))
            if k == "gt":
                gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_gt_observations,))
            if k == "tactile":
                gym_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_tactile_observations,))

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Dict(gym_dict)

        # Create auxiliary observation space if specified
        if self.cfg.aux_list is not None and len(self.cfg.aux_list) > 0:
            state_dict = {}
            for k in self.cfg.aux_list:
                if k == "prop":
                    state_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_prop_observations,))
                elif k == "gt":
                    state_dict[k] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_gt_observations,))
            if state_dict:
                self.single_observation_space["aux"] = gym.spaces.Dict(state_dict)
                self.aux_space = gym.vector.utils.batch_space(self.single_observation_space["aux"], self.num_envs)
            else:
                self.single_observation_space["aux"] = gym.spaces.Dict({})
                self.aux_space = None
        else:
            self.single_observation_space["aux"] = gym.spaces.Dict({})
            self.aux_space = None

        # Batch the spaces for the vectorized environment
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        print("Single observation space (AIRECEnv):", self.single_observation_space)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Stores the actions from the agent before the physics simulation step."""
        self.actions = actions.clone()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """
        Executes one time-step of the environment's dynamics.

        This method is overridden from DirectRLEnv to ensure that `_compute_intermediate_values`
        is called at every step, which is necessary for the mimicry task logic to run correctly.
        """
        action = action.to(self.device)
        if hasattr(self.cfg, "action_noise_model") and self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # Perform physics stepping for the configured number of decimation steps
        for _ in range(self.cfg.decimation):
            if hasattr(self, "_sim_step_counter"):
                self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if hasattr(self, "_sim_step_counter") and self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        # Compute intermediate values after physics and scene updates.
        # This is crucial for updating animation steps, ghost visuals, and derived states.
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
            if hasattr(self.sim, "has_rtx_sensors") and self.sim.has_rtx_sensors() and hasattr(self.cfg, "rerender_on_reset") and self.cfg.rerender_on_reset:
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

    def _apply_action(self) -> None:
        """Applies the agent's actions to the robot."""
        base_actions_raw = self.actions[:, : self.cfg.num_base_actions]
        robot_joint_actions_raw = self.actions[:, self.cfg.num_base_actions :]

        if robot_joint_actions_raw.shape[1] > 0 and self.actuated_dof_indices:
            if robot_joint_actions_raw.shape[1] != len(self.actuated_dof_indices):
                pass  # Mismatch warning could be added here if necessary

            if self.cfg.control_mode == "position":
                scaled_position_targets = self.scale_action(robot_joint_actions_raw)
                self.robot.set_joint_position_target(scaled_position_targets, joint_ids=self.actuated_dof_indices)
            elif self.cfg.control_mode == "velocity":
                self.robot.set_joint_velocity_target(robot_joint_actions_raw, joint_ids=self.actuated_dof_indices)
            else:
                raise ValueError(f"Unsupported control_mode: '{self.cfg.control_mode}' in AIRECEnv.")

        if self.cfg.num_base_actions > 0:
            # Apply forces to the robot's base for translation
            current_external_force_b = torch.zeros_like(self.external_force_b)
            if self.cfg.num_base_actions >= 1:
                current_external_force_b[:, 0, 0] = base_actions_raw[:, 0] * 100.0
            if self.cfg.num_base_actions >= 2:
                current_external_force_b[:, 0, 1] = base_actions_raw[:, 1] * 100.0

            if hasattr(self.robot, "root_physx_view"):
                self.robot.root_physx_view.apply_forces_and_torques_at_position(
                    force_data=current_external_force_b,
                    torque_data=torch.zeros_like(self.external_torque_b),
                    position_data=None,
                    indices=self.robot._ALL_INDICES,
                    is_global=False,
                )

    def scale_action(self, action):
        """Scales normalized actions [-1, 1] to the robot's joint position limits."""
        lower_limits_actuated = self.robot_dof_lower_limits[self.actuated_dof_indices]
        upper_limits_actuated = self.robot_dof_upper_limits[self.actuated_dof_indices]

        scaled_actions = scale(action, lower_limits_actuated, upper_limits_actuated)
        saturated_scaled_actions = saturate(scaled_actions, lower_limits_actuated, upper_limits_actuated)
        return saturated_scaled_actions

    def get_observations(self):
        """Wrapper for _get_observations."""
        return self._get_observations()

    def _get_observations(self) -> dict:
        """Constructs the observation dictionary for the agent."""
        obs_dict = {}
        for k in self.cfg.obs_list:
            if k == "prop":
                obs_dict[k] = self._get_proprioception()
            elif k == "pixels":
                raise NotImplementedError("Pixel observations not fully implemented in this snippet.")
            elif k == "gt":
                # This will call the _get_gt method from the child class (e.g., MimicEnv)
                obs_dict[k] = self._get_gt()
            elif k == "tactile":
                raise NotImplementedError("Tactile observations not fully implemented in this snippet.")
            else:
                print(f"Unknown observations type: {k}")
        return {"policy": obs_dict}

    def _get_proprioception(self):
        """Constructs the proprioceptive observation vector."""
        prop = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.lhand_pos,
                self.lhand_rot,
                self.rhand_pos,
                self.rhand_rot,
                self.actions,
            ),
            dim=-1,
        )
        return prop

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Resets the state for specified environments."""
        if env_ids is None:
            env_ids_tensor = self.robot._ALL_INDICES_TORCH
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = env_ids

        # Call parent reset logic (from DirectRLEnv)
        super()._reset_idx(env_ids_tensor)
        # Reset the robot's physical state
        self._reset_robot(env_ids_tensor)
        # Reset episode-specific counters
        self._episode_timestep_counter[env_ids_tensor] *= 0
        # Compute initial intermediate values for the reset environments
        self._compute_intermediate_values(env_ids_tensor)

    def _reset_robot(self, env_ids: torch.Tensor):
        """Resets the robot's physical state to its default configuration."""
        # Reset joint states
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.set_joint_position_target(joint_pos, joint_ids=None, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset root state (base position and orientation)
        robot_default_state = self.robot.data.default_root_state[env_ids].clone()
        robot_default_state[:, 0:3] = robot_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        robot_default_state[:, 7:] = torch.zeros_like(robot_default_state[:, 7:])
        self.robot.write_root_state_to_sim(robot_default_state, env_ids=env_ids)

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        """
        Computes intermediate values. This merged method contains both base AIREC logic
        and mimicry-specific logic from MimicEnv when it is the active child environment.
        """
        # -- Mimic-specific debug print (only active when running MimicEnv)
        if hasattr(self, "current_animation_step") and hasattr(self, "global_env_steps_counter"):
            if self.num_envs > 0:
                anim_step_env0_str = "N/A"
                if self.current_animation_step.numel() > 0:
                    anim_step_env0_str = str(self.current_animation_step[0].item())

        # -- Core AIREC logic: update proprioceptive and sensor states
        _env_ids_to_use = env_ids if env_ids is not None else self.robot._ALL_INDICES
        full_joint_pos = self.robot.data.joint_pos[_env_ids_to_use]
        full_joint_vel = self.robot.data.joint_vel[_env_ids_to_use]

        self.joint_pos[_env_ids_to_use] = full_joint_pos[:, self.prop_joint_indices]
        self.joint_vel[_env_ids_to_use] = full_joint_vel[:, self.prop_joint_indices]

        self.lhand_pos[_env_ids_to_use] = self.lhand_frame.data.target_pos_source[..., 0, :][_env_ids_to_use]
        self.lhand_rot[_env_ids_to_use] = self.lhand_frame.data.target_quat_source[..., 0, :][_env_ids_to_use]
        self.rhand_pos[_env_ids_to_use] = self.rhand_frame.data.target_pos_source[..., 0, :][_env_ids_to_use]
        self.rhand_rot[_env_ids_to_use] = self.rhand_frame.data.target_quat_source[..., 0, :][_env_ids_to_use]

        # -- Mimic-specific logic (advancing animation, updating ghost)
        if hasattr(self, "global_env_steps_counter"):
            self.global_env_steps_counter += 1

            if hasattr(self, "num_mimic_joints") and hasattr(self, "current_animation_step"):
                if self.global_env_steps_counter % 200 == 0 and self.num_envs > 0 and self.num_mimic_joints > 0:
                    pass  # Original mimic debug print logic was here

                # Advance animation step
                update_slice = slice(None) if env_ids is None else env_ids
                if self.max_animation_steps > 0 and self.current_animation_step.numel() > 0:
                    current_steps_for_slice = self.current_animation_step[update_slice]
                    not_at_anim_end_mask = current_steps_for_slice < (self.max_animation_steps - 1)
                    if isinstance(update_slice, slice):
                        indices_to_advance = torch.where(not_at_anim_end_mask)[0]
                    else:
                        indices_to_advance = update_slice[not_at_anim_end_mask]
                    if indices_to_advance.numel() > 0:
                        self.current_animation_step[indices_to_advance] += 1

                # Update ghost robot visualization
                if self.cfg.enable_ghost_visualizer and hasattr(self, "ghost_robot") and self.ghost_robot is not None:
                    indices_to_use_for_ghost = None
                    if hasattr(self, "ghost_mimic_joint_indices") and self.ghost_mimic_joint_indices is not None:
                        indices_to_use_for_ghost = self.ghost_mimic_joint_indices
                    elif hasattr(self, "mimic_joint_indices_in_robot"):
                        if not hasattr(self, "_warned_ghost_indices_fallback"):
                            print("WARNING: AIRECEnv - 'ghost_mimic_joint_indices' not found. Falling back to main robot's mimic indices.")
                            self._warned_ghost_indices_fallback = True
                        indices_to_use_for_ghost = self.mimic_joint_indices_in_robot

                    if indices_to_use_for_ghost is not None and indices_to_use_for_ghost.numel() > 0:
                        safe_anim_indices = torch.clamp(self.current_animation_step, 0, self.max_animation_steps - 1)
                        target_pos_ghost = self.animation_pos_data[safe_anim_indices, :]
                        # if self.num_envs > 0:
                        #     print(f"[GOAL_STATE_DEBUG Step: {self.global_env_steps_counter}] Env 0: AnimFrame={self.current_animation_step[0].item()}, TargetPos={target_pos_ghost[0, :3].tolist()}")
                        
                        full_ghost_pos = self.ghost_robot.data.default_joint_pos.clone()
                        if env_ids is None:
                            full_ghost_pos[:, indices_to_use_for_ghost] = target_pos_ghost
                        else:
                            full_ghost_pos.index_put_((env_ids[:, None], indices_to_use_for_ghost), target_pos_ghost[env_ids, :])
                        
                        self.ghost_robot.write_joint_state_to_sim(full_ghost_pos, torch.zeros_like(full_ghost_pos), env_ids=env_ids)
                    else:
                        if not hasattr(self, "_warned_no_ghost_indices"):
                            print("WARNING: AIRECEnv - No valid ghost indices. Skipping ghost update.")
                            self._warned_no_ghost_indices = True


# =============================================================================
# Utility Functions
# =============================================================================


@torch.jit.script
def scale(x, lower, upper):
    """Scale Tensors from [-1., 1.] to [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    """Scale Tensors from [lower, upper] to [-1., 1.]."""
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Creates a random rotation quaternion from two random numbers."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """Calculates the angular distance between two rotation quaternions."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def distance_reward(object_ee_distance, std: float = 0.1):
    """Calculates a reward based on distance, shaped by a standard deviation."""
    r_reach = 1 - torch.tanh(object_ee_distance / std)
    return r_reach


@torch.jit.script
def contact_reward(normalised_forces, aperture, binary: bool = False):
    """Calculates a reward based on contact forces and gripper aperture."""
    min_aperture = 0.3
    aperture_mask = (aperture > min_aperture).float()
    f_left, f_right = normalised_forces[:, 0], normalised_forces[:, 1]
    if binary:
        min_force = 0.0
        r_contact_left = torch.where(f_left > min_force, 0.5, 0.0)
        r_contact_right = torch.where(f_right > min_force, 0.5, 0.0)
    else:
        r_contact_left = 2 * f_left * (1 - f_left)
        r_contact_right = 2 * f_right * (1 - f_right)
    r_contact_left *= aperture_mask
    r_contact_right *= aperture_mask
    r_contact = r_contact_left + r_contact_right
    return r_contact


@torch.jit.script
def lift_reward(object_pos, minimal_height: float, episode_timestep_counter):
    """Calculates a reward for lifting an object above a certain height."""
    object_height = object_pos[:, 2]
    is_lifted = torch.where(object_height > minimal_height, 1.0, 0.0)
    is_lifted *= (episode_timestep_counter > 15).float()
    return is_lifted


@torch.jit.script
def object_goal_reward(object_goal_distance, r_lift, std: float = 0.1):
    """Calculates a reward for an object reaching a goal."""
    std = 0.3  # Note: This overrides the input `std` argument.
    object_goal_tracking = 1 - torch.tanh(object_goal_distance / std)
    return object_goal_tracking


@torch.jit.script
def joint_vel_penalty(robot_joint_vel):
    """Calculates a penalty for high joint velocities."""
    r_joint_vel = torch.sum(torch.square(robot_joint_vel), dim=1)
    return r_joint_vel