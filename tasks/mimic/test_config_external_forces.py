"""
Test configuration for external force features in mimic task.
This shows how to use the new timeout, directional bias, and arrow visualization features.
"""

from tasks.mimic.mimic import MimicEnvCfg, ExternalDisturbanceCfg

# Example 1: No forces during first 3 seconds, strong downward bias
downward_bias_cfg = ExternalDisturbanceCfg(
    enable_disturbances=True,
    target_body_name="right_arm_link_5",
    force_magnitude_range=(1000.0, 3000.0),
    duration_range=(0.5, 2.0),
    interval_range=(1.0, 3.0),
    disturbance_probability=0.02,
    episode_start_timeout=3.0,  # No forces for first 3 seconds
    directional_bias=(0.0, 0.0, -1.0),  # Pure downward
    bias_strength=0.9,  # 90% downward, 10% random
    enable_force_visualization=True,
    force_arrow_scale=1.0,
    arrow_shape_scale=(2.0, 0.5, 0.5),  # Longer arrows
    arrow_color=(1.0, 0.0, 0.0)  # Red
)

# Example 2: Forward leaning bias with blue arrows
forward_lean_cfg = ExternalDisturbanceCfg(
    enable_disturbances=True,
    target_body_name="torso_link",
    force_magnitude_range=(500.0, 1500.0),
    duration_range=(1.0, 3.0),
    interval_range=(2.0, 5.0),
    disturbance_probability=0.01,
    episode_start_timeout=5.0,  # Longer timeout
    directional_bias=(1.0, 0.0, -0.5),  # Forward and slightly down
    bias_strength=0.7,  # 70% biased direction
    enable_force_visualization=True,
    force_arrow_scale=1.5,
    arrow_shape_scale=(1.5, 0.8, 0.8),  # Thicker arrows
    arrow_color=(0.0, 0.0, 1.0)  # Blue
)

# Example 3: Random forces (no bias) with green arrows
random_forces_cfg = ExternalDisturbanceCfg(
    enable_disturbances=True,
    target_body_name="head_link",
    force_magnitude_range=(200.0, 800.0),
    duration_range=(0.2, 1.0),
    interval_range=(0.5, 2.0),
    disturbance_probability=0.05,
    episode_start_timeout=1.0,  # Short timeout
    directional_bias=(0.0, 0.0, -1.0),  # Doesn't matter
    bias_strength=0.0,  # Pure random (no bias)
    enable_force_visualization=True,
    force_arrow_scale=2.0,
    arrow_shape_scale=(0.8, 0.3, 0.3),  # Thin arrows
    arrow_color=(0.0, 1.0, 0.0)  # Green
)

# Usage in environment configuration:
def create_test_env_cfg():
    """Create a test environment configuration with custom external forces."""
    cfg = MimicEnvCfg()
    
    # Choose one of the example configurations
    cfg.external_disturbance = downward_bias_cfg
    
    # Or disable forces entirely
    # cfg.external_disturbance.enable_disturbances = False
    
    return cfg