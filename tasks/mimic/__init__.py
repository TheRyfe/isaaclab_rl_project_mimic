# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import agents
from .mimic import MimicEnv, MimicEnvCfg

# Get the path to the agent configuration file
agents_dir = os.path.dirname(agents.__file__)
agent_cfg_path = os.path.join(agents_dir, "prop_mimic.yaml")

# Register the environment with gymnasium
gym.register(
    id="Mimic",
    entry_point="tasks.mimic.mimic:MimicEnv",  # Corrected entry point
    kwargs={
        "env_cfg_entry_point": mimic.MimicEnvCfg,
        "skrl_cfg_entry_point": agent_cfg_path,
    },
    disable_env_checker=True,
)