# Mimic Task Reward Function

## Complete Reward Function

$$R_{\text{total}} = R_{\text{pos}} + R_{\text{link}} + R_{\text{alive}} + P_{\text{vel}} + P_{\text{smooth}}$$

**LaTeX:**
```latex
R_{\text{total}} = R_{\text{pos}} + R_{\text{link}} + R_{\text{alive}} + P_{\text{vel}} + P_{\text{smooth}}
```

## Component Definitions

### 1. Joint Position Tracking Reward ($R_{\text{pos}}$)

$$R_{\text{pos}} = 3.0 \cdot \left( \exp\left(-\frac{\sum_{i=1}^{20} w_i \cdot e_{i,\text{norm}}^2}{0.25 \cdot 20 \cdot \bar{w}}\right) \right)^{2.0}$$

**LaTeX:**
```latex
R_{\text{pos}} = 3.0 \cdot \left( \exp\left(-\frac{\sum_{i=1}^{20} w_i \cdot e_{i,\text{norm}}^2}{0.25 \cdot 20 \cdot \bar{w}}\right) \right)^{2.0}
```

Where:
- **Normalized joint error**: $e_{i,\text{norm}} = \frac{(\theta_{i,\text{target}} - \theta_{i,\text{current}}) - \text{mid}_i}{0.5 \cdot \text{range}_i}$
  
  **LaTeX:**
  ```latex
  e_{i,\text{norm}} = \frac{(\theta_{i,\text{target}} - \theta_{i,\text{current}}) - \text{mid}_i}{0.5 \cdot \text{range}_i}
  ```
  
  - $\text{mid}_i = \frac{\theta_{i,\text{upper}} + \theta_{i,\text{lower}}}{2}$
  
    **LaTeX:**
    ```latex
    \text{mid}_i = \frac{\theta_{i,\text{upper}} + \theta_{i,\text{lower}}}{2}
    ```
  
  - $\text{range}_i = \theta_{i,\text{upper}} - \theta_{i,\text{lower}}$
  
    **LaTeX:**
    ```latex
    \text{range}_i = \theta_{i,\text{upper}} - \theta_{i,\text{lower}}
    ```

- **Joint weights** ($w_i$):
  - Head joints (H1, H2, H3): $w_i = 0.5$
  - Torso joints (T1, T2, T3): $w_i = 2.0$
  - Arm joints (R1-R7, L1-L7): $w_i = 1.0$

- **Mean weight**: $\bar{w} = \frac{1}{20}\sum_{i=1}^{20} w_i$

  **LaTeX:**
  ```latex
  \bar{w} = \frac{1}{20}\sum_{i=1}^{20} w_i
  ```

### 2. Link Tracking Reward ($R_{\text{link}}$)

$$R_{\text{link}} = R_{\text{link,pos}} + R_{\text{link,ori}}$$

**LaTeX:**
```latex
R_{\text{link}} = R_{\text{link,pos}} + R_{\text{link,ori}}
```

#### Position Component:
$$R_{\text{link,pos}} = 4.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} ||\mathbf{p}_{j,\text{real}} - \mathbf{p}_{j,\text{ghost}}||^2}{0.1 \cdot 2}\right)$$

**LaTeX:**
```latex
R_{\text{link,pos}} = 4.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} ||\mathbf{p}_{j,\text{real}} - \mathbf{p}_{j,\text{ghost}}||^2}{0.1 \cdot 2}\right)
```

#### Orientation Component:
$$R_{\text{link,ori}} = 2.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} (1 - |q_{j,\text{real}} \cdot q_{j,\text{ghost}}|)}{0.2 \cdot 2}\right)$$

**LaTeX:**
```latex
R_{\text{link,ori}} = 2.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} (1 - |q_{j,\text{real}} \cdot q_{j,\text{ghost}}|)}{0.2 \cdot 2}\right)
```

Where:
- Tracked links: "right_arm_link_5" and "left_arm_link_5" ($j = 1, 2$)
- $\mathbf{p}_j$ = 3D position vector
- $q_j$ = quaternion orientation (w, x, y, z)

### 3. Staying Alive Reward ($R_{\text{alive}}$)

$$R_{\text{alive}} = 0.005$$

**LaTeX:**
```latex
R_{\text{alive}} = 0.005
```

### 4. Joint Velocity Penalty ($P_{\text{vel}}$)

$$P_{\text{vel}} = -0.001 \cdot \sum_{i=1}^{20} \dot{\theta}_i^2$$

**LaTeX:**
```latex
P_{\text{vel}} = -0.001 \cdot \sum_{i=1}^{20} \dot{\theta}_i^2
```

Where $\dot{\theta}_i$ is the velocity of joint $i$

### 5. Action Smoothness Penalty ($P_{\text{smooth}}$)

$$P_{\text{smooth}} = -0.01 \cdot \sum_{k=1}^{20} (a_k^{(t)} - a_k^{(t-1)})^2$$

**LaTeX:**
```latex
P_{\text{smooth}} = -0.01 \cdot \sum_{k=1}^{20} (a_k^{(t)} - a_k^{(t-1)})^2
```

Where:
- $a_k^{(t)}$ = action for joint $k$ at time step $t$
- $a_k^{(t-1)}$ = action for joint $k$ at previous time step

## Summary of Reward Scales

| Component | Scale Factor |
|-----------|-------------|
| Joint Position Tracking | 3.0 |
| Link Position Tracking | 4.0 |
| Link Orientation Tracking | 2.0 |
| Staying Alive | 0.005 |
| Joint Velocity Penalty | -0.001 |
| Action Smoothness Penalty | -0.01 |

## Additional Parameters

- **Position error variance scale**: 0.25
- **Link position error variance**: 0.1
- **Link orientation error variance**: 0.2
- **Position tracking power scale**: 2.0
- **Number of tracked joints**: 20
- **Number of tracked links**: 2

## Complete LaTeX Code for Document

```latex
% Complete reward function
R_{\text{total}} = R_{\text{pos}} + R_{\text{link}} + R_{\text{alive}} + P_{\text{vel}} + P_{\text{smooth}}

% Joint position tracking reward
R_{\text{pos}} = 3.0 \cdot \left( \exp\left(-\frac{\sum_{i=1}^{20} w_i \cdot e_{i,\text{norm}}^2}{0.25 \cdot 20 \cdot \bar{w}}\right) \right)^{2.0}

% Normalized joint error
e_{i,\text{norm}} = \frac{(\theta_{i,\text{target}} - \theta_{i,\text{current}}) - \text{mid}_i}{0.5 \cdot \text{range}_i}

% Joint midpoint and range
\text{mid}_i = \frac{\theta_{i,\text{upper}} + \theta_{i,\text{lower}}}{2}
\text{range}_i = \theta_{i,\text{upper}} - \theta_{i,\text{lower}}

% Mean weight
\bar{w} = \frac{1}{20}\sum_{i=1}^{20} w_i

% Link tracking reward
R_{\text{link}} = R_{\text{link,pos}} + R_{\text{link,ori}}

% Link position component
R_{\text{link,pos}} = 4.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} ||\mathbf{p}_{j,\text{real}} - \mathbf{p}_{j,\text{ghost}}||^2}{0.1 \cdot 2}\right)

% Link orientation component
R_{\text{link,ori}} = 2.0 \cdot \exp\left(-\frac{\sum_{j=1}^{2} (1 - |q_{j,\text{real}} \cdot q_{j,\text{ghost}}|)}{0.2 \cdot 2}\right)

% Staying alive reward
R_{\text{alive}} = 0.005

% Joint velocity penalty
P_{\text{vel}} = -0.001 \cdot \sum_{i=1}^{20} \dot{\theta}_i^2

% Action smoothness penalty
P_{\text{smooth}} = -0.01 \cdot \sum_{k=1}^{20} (a_k^{(t)} - a_k^{(t-1)})^2
```