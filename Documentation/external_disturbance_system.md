# External Disturbance System Implementation

## Overview

The external disturbance system is designed to improve the robustness of the learned mimicry policy by applying random external forces during training. This helps the robot learn to maintain balance and recover from unexpected perturbations, making the policy more robust for real-world deployment.

## System Architecture

### 1. Configuration (ExternalDisturbanceCfg)

The system is highly configurable through the `ExternalDisturbanceCfg` class:

```python
@configclass
class ExternalDisturbanceCfg:
    # Core settings
    enable_disturbances: bool = True
    target_body_name: str = "right_arm_link_5"  # Body to apply forces to
    
    # Force parameters
    force_magnitude_range: tuple[float, float] = (30.0, 150.0)  # Newtons
    
    # Timing parameters
    duration_range: tuple[float, float] = (0.5, 1.5)  # seconds
    interval_range: tuple[float, float] = (0.5, 3.0)  # cooldown between forces
    disturbance_probability: float = 0.01  # 1% per timestep per env
    episode_start_grace_steps: int = 130  # ~2.17s grace period at 60Hz
    
    # Direction control
    directional_bias: tuple[float, float, float] = (0.0, 0.0, -1.0)  # downward
    bias_strength: float = 0.6  # 60% bias, 40% random
    
    # Visualization
    enable_force_visualization: bool = True
    arrow_color: tuple[float, float, float] = (1.0, 0.0, 0.0)  # red
```

### 2. State Tracking

The system maintains several state tensors per environment:

- `disturbance_forces`: Current force vector being applied (N×3)
- `disturbance_remaining_time`: Time left for current disturbance (N×1)
- `disturbance_cooldown_time`: Time before next disturbance can occur (N×1)
- `episode_step_counter`: Steps since episode start for grace period (N×1)

## Implementation Details

### Force Generation Process

1. **Timing Control**
   - Forces are applied stochastically based on `disturbance_probability`
   - Each environment independently decides when to apply forces
   - Grace period prevents forces during initial episode steps
   - Cooldown period ensures forces aren't constantly applied

2. **Force Direction Calculation**
   ```python
   # Blend random and biased directions
   final_direction = bias_strength * bias_direction + 
                     (1 - bias_strength) * random_direction
   # Normalize to unit vector
   final_direction = final_direction / ||final_direction||
   ```

3. **Magnitude Selection**
   - Uniformly sampled from `force_magnitude_range`
   - Typical range: 30-150 N (suitable for humanoid robot)

### Force Application Pipeline

The force application follows Isaac Lab's standard API:

1. **Global to Local Transformation**
   ```python
   # Get body orientation quaternion
   body_orientation = robot.data.body_state_w[:, body_id, 3:7]
   
   # Transform global force to local body frame
   local_force = quat_apply_inverse(body_orientation, global_force)
   ```

2. **Force Tensor Construction**
   ```python
   # Create force tensor for all bodies
   external_forces = torch.zeros((num_envs, num_bodies, 3))
   external_forces[:, target_body_id, :] = local_forces
   
   # Apply via Isaac Lab API
   robot.set_external_force_and_torque(forces=external_forces, 
                                      torques=zero_torques)
   ```

### Probability Scaling

The system automatically adjusts probability based on control frequency:

```python
# Scale probability to maintain consistent behavior across frequencies
standard_dt = 1/60  # Reference: 60Hz
time_scale_factor = control_dt / standard_dt
adjusted_probability = base_probability * time_scale_factor
```

## Visualization System

### Arrow Markers

Forces are visualized using 3D arrow markers:

1. **Position**: Placed at the target body location
2. **Orientation**: Aligned with force direction using quaternion rotation
3. **Scale**: Proportional to force magnitude
4. **Color**: Configurable (default: red)

### Quaternion Calculation

The system converts force directions to quaternions for arrow orientation:

```python
def _vector_to_quaternion(vec):
    # Rotate from +X axis (arrow default) to force direction
    axis = cross(+X, force_direction)
    angle = acos(dot(+X, force_direction))
    # Convert to quaternion
    return quaternion_from_axis_angle(axis, angle)
```

## Integration Points

### 1. Environment Step Loop

```python
def step(self, action):
    # ... pre-physics
    
    # Apply disturbances once per environment step
    self._apply_external_disturbances()
    
    # Physics stepping
    for _ in range(decimation):
        self._apply_action()
        self.sim.step()
    
    # Update visualization
    self._visualize_forces()
    
    # ... post-physics
```

### 2. Reset Handling

On environment reset:
- Clear all active forces
- Reset timers and counters
- Hide visualization markers

### 3. Logging Integration

The system tracks:
- `disturbance_active`: Binary indicator per environment
- `disturbance_force_magnitude`: Current force magnitude

## Design Rationale

1. **Target Body Selection**: Right arm link chosen as it's far from center of mass, creating significant perturbations

2. **Force Ranges**: 30-150N provides meaningful disturbance without being unrealistic for a humanoid robot

3. **Directional Bias**: Default downward bias simulates gravity-like effects and leaning scenarios

4. **Grace Period**: Allows robot to stabilize at episode start before applying forces

5. **Stochastic Application**: Random timing prevents the policy from memorizing disturbance patterns

## Performance Considerations

- Force calculations are vectorized for GPU efficiency
- Visualization only updates when forces are active
- Probability scaling ensures consistent behavior across different control frequencies
- Local force transformation ensures physically correct application regardless of body orientation

## Usage Example

```yaml
# In environment config
external_disturbance:
  enable_disturbances: true
  target_body_name: "right_arm_link_5"
  force_magnitude_range: [50.0, 200.0]  # Stronger forces
  bias_strength: 0.8  # More directional
  directional_bias: [1.0, 0.0, 0.0]  # Forward push
```

This system significantly improves policy robustness by exposing the robot to varied disturbances during training, leading to more stable real-world performance.