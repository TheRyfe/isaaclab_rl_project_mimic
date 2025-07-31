# Observation Normalization Status

## Proprioceptive Observations (74 dimensions)

| Component | Dimensions | Normalized | Range/Units | Method |
|-----------|------------|------------|-------------|---------|
| Joint Positions | 20 | ✅ Yes | [-1, 1] | `unscale()` from joint limits |
| Joint Velocities | 20 | ✅ Yes | [-1, 1] | `unscale()` from velocity limits |
| Actions | 20 | ✅ Yes | [-1, 1] | Policy output (already normalized) |
| Left Arm Link 5 Position | 3 | ✅ Yes | [-1, 1] | `unscale()` from workspace bounds |
| Left Arm Link 5 Rotation | 4 | ⚠️ Quaternion | unit quaternion | Inherently normalized |
| Right Arm Link 5 Position | 3 | ✅ Yes | [-1, 1] | `unscale()` from workspace bounds |
| Right Arm Link 5 Rotation | 4 | ⚠️ Quaternion | unit quaternion | Inherently normalized |

## Ground Truth Observations (34 dimensions)

| Component | Dimensions | Normalized | Range/Units | Method |
|-----------|------------|------------|-------------|---------|
| Target Animation Joint Positions | 20 | ✅ Yes | [-1, 1] | `unscale()` from joint limits |
| Ghost Left Arm Link 5 Position | 3 | ✅ Yes | [-1, 1] | `unscale()` from workspace bounds |
| Ghost Left Arm Link 5 Rotation | 4 | ⚠️ Quaternion | unit quaternion | Inherently normalized |
| Ghost Right Arm Link 5 Position | 3 | ✅ Yes | [-1, 1] | `unscale()` from workspace bounds |
| Ghost Right Arm Link 5 Rotation | 4 | ⚠️ Quaternion | unit quaternion | Inherently normalized |

## Notes

1. **Normalization Formula**: The `unscale()` function uses: `(2.0 * x - upper - lower) / (upper - lower)` to map from [lower, upper] to [-1, 1]

2. **Workspace Bounds**: Arm link 5 positions are normalized using the following bounds:
   - Lower bounds: [-1.5, -1.5, -0.5] meters (x, y, z)
   - Upper bounds: [1.5, 1.5, 2.5] meters (x, y, z)
   - These bounds represent reasonable arm reach for a humanoid robot

3. **Quaternions**: While quaternions are unit normalized (magnitude = 1), they are not scaled to a specific range like [-1, 1]. Each component can range from -1 to 1, but the constraint is that w² + x² + y² + z² = 1.

4. **Complete Normalization**: All non-quaternion observations are now normalized to [-1, 1], providing consistent input scales for the neural network.

5. **Joint Limits**: Joint positions are normalized using soft joint limits from the robot model, ensuring the normalization respects physical constraints.