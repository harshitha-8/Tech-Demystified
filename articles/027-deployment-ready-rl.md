# Deployment-Ready Reinforcement Learning: Bridging the Sim-to-Real Gap

### Engineering robust RL policies for real-world robotics and production systems

Reinforcement learning has produced spectacular results in simulation: agents mastering complex games, robots performing acrobatic maneuvers, and systems optimizing intricate control tasks. Yet when these same policies are deployed to real hardware or production environments, they often fail catastrophically. **The sim-to-real gap** — the chasm between simulated perfection and messy reality — remains one of the hardest challenges in applied RL.

Recent breakthroughs in legged locomotion, particularly quadrupedal robots navigating complex terrain using purely simulation-trained policies, have demonstrated that **zero-shot sim-to-real transfer is possible**. But extending these successes to other domains (humanoid robotics, manipulation, industrial control) and production environments requires understanding not just RL algorithms, but the **engineering decisions that make or break deployment**.

This article explores the practical engineering of deployment-ready RL systems:

- **Action space design**: How control formulations affect exploration, stability, and transfer
- **Observation space engineering**: What sensors to use, what information to provide, and when to break Markov assumptions
- **Domain randomization**: Injecting the right kinds of variation to bridge sim-to-real
- **Asymmetric actor-critic**: Exploiting privileged simulation information for better policies
- **Student-teacher distillation**: Learning from oracle policies with perfect information
- **Reward shaping**: Designing objectives that produce desired behaviors without over-specification
- **System identification**: Matching simulation to reality through careful measurement
- **Safety and robustness**: Handling edge cases, failures, and distribution shift
- **Deployment infrastructure**: Monitoring, debugging, and iterating on hardware

We'll focus on **robotics applications** as the canonical sim-to-real challenge, but the principles apply broadly to deploying RL in production: trading systems, recommendation engines, resource allocation, and any domain where simulation training must transfer to real-world deployment.

## The Sim-to-Real Problem: Why RL Policies Fail in Reality

Before diving into solutions, let's understand what makes sim-to-real transfer so challenging.

#### Mismatch #1: Dynamics

**Simulation**: Perfect physics, deterministic dynamics, no latency
```python
# Simulated dynamics (instant, perfect)
next_state = physics_engine.step(state, action)
# Exactly the response you'd expect from equations
```

**Reality**: Sensor noise, actuator delays, unmodeled dynamics
```python
# Real hardware
send_command(action)  # Latency: 5-50ms
time.sleep(0.02)  # Control loop delay
state = read_sensors()  # Noisy, filtered, delayed
# Actual response differs from expected
```

**Gap**: 
- **Latency**: 20-50ms delays between observation and action effect
- **Actuator dynamics**: Motors don't instantly achieve commanded torque/position
- **Unmodeled effects**: Friction, backlash, cable stiffness, temperature effects
- **State estimation errors**: Sensors give noisy, biased measurements

#### Mismatch #2: Perception

**Simulation**: Ground truth state (robot joint angles, object positions, contact forces)

**Reality**: Must estimate state from noisy sensors (IMU, cameras, encoders)

**Example**:
```
Simulation: robot.get_linear_velocity() → [1.23, 0.45, 0.0] (exact)
Reality: IMU integration → [1.18, 0.52, -0.03] (drift, bias, noise)
```

#### Mismatch #3: Task Distribution

**Simulation**: Controlled environment (flat ground, known objects, predictable scenarios)

**Reality**: Infinite variability (uneven terrain, lighting changes, unexpected obstacles)

**Policy trained on**: Perfectly flat floor with friction = 0.9
**Deployed to**: Carpet (friction 1.5), tile (friction 0.6), wet surfaces (friction 0.3)

**Result**: Policy has never seen these conditions → fails

#### Mismatch #4: Safety Constraints

**Simulation**: Can reset infinitely, breaking robot is free

**Reality**: Hardware is expensive, dangerous, and slow to iterate

**Consequence**: Must be conservative, can't explore aggressively, limited training data

#### Why This Matters

A policy that achieves 95% success rate in simulation might achieve:
- 30% success rate on hardware (with no adaptation)
- 70% success rate (with domain randomization)
- 90% success rate (with proper system identification + domain rand + careful engineering)

**The difference between research demos and production systems is engineering that last 20%.**

## Action Space Design: The Foundation of Stable Control

How you formulate the action space has profound effects on exploration, stability, and sim-to-real transfer.

#### Option 1: Direct Torque Control

**Formulation**: Policy outputs joint torques directly

$$\tau = \pi(s; \theta)$$

**Advantages**:
- Maximum flexibility: policy can learn any control strategy
- No assumptions about task structure

**Disadvantages**:
- ❌ **Unstable**: Small mistakes → large torques → hardware damage
- ❌ **Hard to explore**: Random torques often cause falls
- ❌ **Poor initialization**: Untrained policy flails wildly

**When to use**: Never for initial deployment (too dangerous)

#### Option 2: Position Control (PD Controller)

**Formulation**: Policy outputs desired joint positions, low-level PD controller tracks them

$$q_{\text{desired}} = \pi(s; \theta)$$
$$\tau = K_p (q_{\text{desired}} - q) + K_d (\dot{q}_{\text{desired}} - \dot{q})$$

**Advantages**:
- ✅ **Safe**: PD controller provides damping, limits velocities
- ✅ **Stable exploration**: Random positions less dangerous than random torques

**Disadvantages**:
- ⚠️ **Reduced bandwidth**: PD controller filters policy output
- ⚠️ **Can't learn dynamic behaviors**: PD controller fights against rapid movements

#### Option 3: Position Residuals (Hybrid) — **Recommended**

**Formulation**: Policy outputs small adjustments to a reference position

$$q_{\text{desired}} = q_{\text{ref}} + K_a \cdot \pi(s; \theta)$$
$$\tau = K_p (q_{\text{desired}} - q) + K_d (0 - \dot{q})$$

Where:
- $q_{\text{ref}}$: Reference pose (e.g., standing configuration)
- $K_a$: Action scale (limits deviation from reference)
- $\pi(s; \theta)$: Policy output (typically normalized to [-1, 1])

**Advantages**:
- ✅ **Safe initialization**: Starts near stable configuration
- ✅ **Smooth exploration**: Residuals keep robot near valid states
- ✅ **Natural regularization**: Large deviations from reference penalized implicitly
- ✅ **Tunable flexibility**: Adjust $K_a$ and $K_p$ to control policy authority

**Disadvantages**:
- ⚠️ Requires choosing appropriate reference pose
- ⚠️ May struggle with highly dynamic tasks requiring large deviations

**Implementation**:
```python
class ResidualPositionController:
    def __init__(self, reference_pose, Kp, Kd, action_scale):
        self.q_ref = reference_pose  # Standing pose
        self.Kp = Kp  # Position gains (per joint)
        self.Kd = Kd  # Velocity gains (per joint)
        self.action_scale = action_scale  # Limits residual magnitude
    
    def compute_torque(self, policy_output, q_current, q_dot_current):
        # Policy outputs normalized action [-1, 1]
        action = np.clip(policy_output, -1, 1)
        
        # Compute desired position as reference + residual
        q_desired = self.q_ref + self.action_scale * action
        
        # PD control law
        tau = self.Kp * (q_desired - q_current) + self.Kd * (0 - q_dot_current)
        
        return tau
```

#### PD Gain Tuning for Sim-to-Real

**Critical insight**: Gain choice affects both exploration and transfer

**High gains** (stiff control):
```python
Kp = 100.0  # Very stiff
Kd = 10.0
action_scale = 0.1  # Small residuals
```

**Effect**:
- ✅ Tracks desired positions precisely
- ✅ Transfers well initially (matches real hardware stiffness)
- ❌ **Limited exploration**: Policy can't deviate much from reference
- ❌ **Reduced learning**: PD controller does most of the work

**Low gains** (compliant control):
```python
Kp = 20.0  # Compliant
Kd = 2.0
action_scale = 0.5  # Larger residuals allowed
```

**Effect**:
- ✅ **Better exploration**: Policy has more authority
- ✅ **Faster learning**: Policy learns richer behaviors
- ❌ Less precise tracking
- ❌ May be less stable initially

**Recommended approach**:

**Phase 1 (Early training)**: Lower gains for exploration
```python
# Simulation training
Kp = 30.0
Kd = 3.0
action_scale = 0.3
```

**Phase 2 (Fine-tuning)**: Gradually increase gains
```python
# Late training / sim-to-real
Kp = 50.0
Kd = 5.0
action_scale = 0.2
```

**Hardware deployment**: Match simulation gains exactly (system ID critical!)

#### Alternative: Feedforward Torque + PD

Mathematically equivalent to residuals but conceptually different:

$$\tau = \tau_{\text{ff}} + K_p (q_{\text{ref}} - q) + K_d (0 - \dot{q})$$

Where $\tau_{\text{ff}} = K_p \cdot K_a \cdot \pi(s; \theta)$

**Interpretation**: Policy provides feedforward torque, PD stabilizes around reference

**Use when**: You want to think about forces rather than positions

## Observation Space Engineering: What Information to Provide

Choosing what the policy observes is as important as choosing actions.

#### Minimal Observation Space (Recommended Starting Point)

```python
observation = np.concatenate([
    # Proprioception (robot's own state)
    q,                    # Joint positions [n_joints]
    q_dot,                # Joint velocities [n_joints]
    imu_orientation,      # Orientation (quaternion or euler) [3-4]
    imu_angular_velocity, # Angular velocity [3]
    
    # Command (what we're asking robot to do)
    command_vel,          # Desired linear velocity [2-3]
    command_yaw_rate,     # Desired angular velocity [1]
    
    # Previous action (helps with smoothness)
    action_prev,          # Last action taken [n_joints]
])
```

**Rationale**: This is **usually sufficient** for locomotion tasks if:
- Simulation is well-identified
- Actuation dynamics are modeled correctly
- State estimation is accurate

#### Common Mistakes: Over-Augmenting Observations

**❌ BAD: Adding observations as a crutch for poor system identification**

```python
# Don't do this!
observation = np.concatenate([
    q, q_dot, imu_orientation, imu_angular_velocity,
    
    # Bad additions:
    estimated_ground_friction,  # Should be handled via domain rand
    estimated_foot_height,      # Should be implicit in joint state
    estimated_external_forces,  # Usually not needed
    observation_history_50_steps,  # 1 second of history?! Why?
])
```

**Why this fails**:
- Masks underlying problems (bad system ID, poor actuator model)
- Creates dependency on information not available on hardware
- Increases sample complexity
- Doesn't actually help in most cases

**✅ GOOD: Fix the root cause instead**

```python
# If linear velocity estimate is poor, don't add 50 timesteps of history
# Instead: Train a learned velocity estimator

class VelocityEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_proprioception * 10, 128),  # 10 timesteps
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 3)  # vx, vy, vz
        )
    
    def forward(self, proprio_history):
        return self.encoder(proprio_history)

# Train alongside policy using ground truth supervision in simulation
```

#### When to Augment Observations

**Legitimate reasons to add information**:

**1. Task requires temporal reasoning**
```python
# Navigation in a maze (need memory of where you've been)
observation = np.concatenate([
    base_observation,
    visited_locations_map,  # Explicit spatial memory
])
```

**2. Motion imitation from reference trajectory**
```python
# Following motion capture data
observation = np.concatenate([
    base_observation,
    reference_joint_positions[t:t+10],  # Future reference
    phase_variable,  # Where in motion cycle
])
```

**3. Dealing with non-Markovian dynamics**
```python
# Contact with deformable objects (trampoline, soft terrain)
# One timestep insufficient to predict response
observation = np.concatenate([
    base_observation,
    contact_history_5_steps,  # Recent contact forces
])
```

**4. Limited sensor quality**
```python
# When sensors are fundamentally noisy and you can't improve them
observation = np.concatenate([
    base_observation,
    observation_history_3_steps,  # Short history for filtering
])
# But: Try explicit filtering first (Kalman, learned estimator)
```

#### State Estimation: Explicit vs. Implicit

**Problem**: On hardware, you don't have ground truth linear velocity

**Bad solution**: Add observation history and hope policy learns to estimate
```python
# ❌ Implicit estimation (inefficient)
obs = np.concatenate([q, q_dot, imu_ang_vel] * 10)  # 10 timesteps
# Policy must learn to differentiate/integrate to get linear velocity
```

**Good solution**: Train explicit estimator
```python
# ✅ Explicit estimation
velocity_estimator = VelocityEstimator()
estimated_velocity = velocity_estimator(proprioception_history)
obs = np.concatenate([q, q_dot, imu_ang_vel, estimated_velocity])

# Estimator trained with supervised learning in simulation
# Then frozen and deployed with policy
```

**Best solution**: Use model-based state estimation (when possible)
```python
# ✅✅ Model-based (EKF, contact-aided estimation)
# Uses kinematics + contact detection + IMU fusion
estimated_velocity = kalman_filter.update(imu_data, joint_data, contact_state)
```

**Hierarchy of state estimation quality**:
1. Model-based estimation (Kalman, factor graphs) — **Best**
2. Learned estimator (trained alongside policy) — **Good**
3. Implicit estimation (history + recurrence) — **Last resort**

## Domain Randomization: Preparing for Reality's Variability

**Core idea**: If the policy experiences wide variation in simulation, real-world variation becomes just another test case.

#### What to Randomize

**1. Dynamics parameters**
```python
class DomainRandomization:
    def randomize_dynamics(self, robot):
        # Mass distribution
        for link in robot.links:
            link.mass *= np.random.uniform(0.8, 1.2)
            link.inertia *= np.random.uniform(0.7, 1.3)
        
        # Friction
        robot.ground_friction = np.random.uniform(0.4, 1.5)
        
        # Actuator dynamics
        for joint in robot.joints:
            # Torque delay
            joint.actuation_delay = np.random.uniform(0.01, 0.05)  # 10-50ms
            
            # Torque noise
            joint.torque_noise_std = np.random.uniform(0.0, 0.05)
            
            # Position/velocity sensor noise
            joint.position_noise_std = np.random.uniform(0.0, 0.01)
            joint.velocity_noise_std = np.random.uniform(0.0, 0.05)
        
        # IMU noise
        robot.imu.angular_vel_noise_std = np.random.uniform(0.0, 0.1)
        robot.imu.orientation_drift = np.random.uniform(0.0, 0.02)
```

**2. Environmental variation**
```python
def randomize_environment(self, env):
    # Terrain
    terrain_types = ['flat', 'slopes', 'stairs', 'rough', 'mixed']
    env.terrain = np.random.choice(terrain_types)
    
    if env.terrain == 'rough':
        # Height map perturbations
        env.ground_height_map = generate_perlin_noise(
            amplitude=np.random.uniform(0.0, 0.1)
        )
    
    # External disturbances
    if np.random.rand() < 0.3:  # 30% of episodes
        # Apply random push
        push_timing = np.random.uniform(2.0, 8.0)  # seconds
        push_force = np.random.uniform(50, 200)  # Newtons
        push_direction = np.random.uniform(0, 2*np.pi)
        env.schedule_push(push_timing, push_force, push_direction)
```

**3. Control latency**
```python
class LatencySimulator:
    def __init__(self):
        self.action_buffer = deque(maxlen=10)
        
    def add_action(self, action, timestamp):
        # Randomize latency each step
        latency = np.random.uniform(0.01, 0.05)  # 10-50ms
        self.action_buffer.append((action, timestamp + latency))
    
    def get_action_to_apply(self, current_time):
        # Return action whose timestamp has passed
        for action, timestamp in self.action_buffer:
            if timestamp <= current_time:
                return action
        return None  # No action ready yet
```

#### How Much to Randomize

**Trade-off**: More randomization → more robust, but slower learning

**Curriculum approach** (recommended):

**Phase 1: Narrow distribution (fast learning)**
```python
# First 500M timesteps
friction = np.random.uniform(0.8, 1.1)
mass_scale = np.random.uniform(0.95, 1.05)
latency = 0.02  # Fixed 20ms
```

**Phase 2: Medium distribution**
```python
# 500M - 1B timesteps
friction = np.random.uniform(0.6, 1.3)
mass_scale = np.random.uniform(0.85, 1.15)
latency = np.random.uniform(0.015, 0.035)
```

**Phase 3: Wide distribution (final robustness)**
```python
# 1B+ timesteps
friction = np.random.uniform(0.4, 1.5)
mass_scale = np.random.uniform(0.7, 1.3)
latency = np.random.uniform(0.01, 0.05)
# Add external pushes, terrain variation, etc.
```

**Automatic Domain Randomization (ADR)**:

Learn randomization ranges automatically:
```python
class ADR:
    def __init__(self):
        self.friction_range = [0.9, 1.0]  # Start narrow
        
    def update_ranges(self, success_rate):
        if success_rate > 0.95:  # Too easy
            # Expand range
            self.friction_range[0] -= 0.05
            self.friction_range[1] += 0.05
        elif success_rate < 0.7:  # Too hard
            # Narrow range
            self.friction_range[0] += 0.02
            self.friction_range[1] -= 0.02
```

#### Domain Randomization Anti-Patterns

**❌ Over-randomization without justification**
```python
# Don't randomize things that don't vary in reality
robot.num_legs = np.random.randint(2, 6)  # Nonsense
gravity = np.random.uniform(5, 15)  # Unless on moon
```

**❌ Randomizing correlated parameters independently**
```python
# Mass and inertia are related!
link.mass *= np.random.uniform(0.8, 1.2)
link.inertia *= np.random.uniform(0.8, 1.2)  # Should scale with mass^2
```

**✅ Physically-motivated randomization**
```python
mass_scale = np.random.uniform(0.8, 1.2)
link.mass *= mass_scale
link.inertia *= mass_scale ** 2  # Inertia scales with mass squared
```

## Asymmetric Actor-Critic: Exploiting Privileged Information

**Key insight**: Use ground truth information in simulation to train better policies, but deploy without it.

#### The Problem

**In simulation**: Perfect state information (contact forces, friction, object poses)
**On hardware**: Limited sensors (no force sensors, noisy perception)

**Naive approach**: Train policy using only hardware-available sensors
**Result**: Slow learning, suboptimal policies

#### Asymmetric Actor-Critic Solution

**Architecture**:

```python
class AsymmetricActorCritic(nn.Module):
    def __init__(self, obs_dim, privileged_dim, action_dim):
        super().__init__()
        
        # Actor: Only uses observations available on hardware
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic: Uses ALL available information in simulation
        self.critic = nn.Sequential(
            nn.Linear(obs_dim + privileged_dim, 512),  # Concatenate
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1)  # Value estimate
        )
    
    def forward(self, obs, privileged_obs=None):
        # Actor output (deploys to hardware)
        action = self.actor(obs)
        
        # Critic output (only used in simulation training)
        if privileged_obs is not None:
            critic_input = torch.cat([obs, privileged_obs], dim=-1)
            value = self.critic(critic_input)
        else:
            value = None  # Not needed on hardware
        
        return action, value
```

**Privileged observations** (only in simulation):
```python
privileged_obs = np.concatenate([
    ground_truth_linear_velocity,  # Perfect velocity
    contact_forces,                # Force sensors (not on hardware)
    friction_coefficient,          # Ground truth friction
    external_forces,               # Disturbances
    foot_height_above_ground,      # Perfect clearance
    center_of_mass_position,       # CoM state
])
```

**Actor observations** (available on hardware):
```python
actor_obs = np.concatenate([
    joint_positions,
    joint_velocities,
    imu_orientation,
    imu_angular_velocity,
    command_velocity,
])
# No privileged information!
```

**Training loop**:
```python
def train_step(self, obs, privileged_obs, reward, next_obs, next_priv_obs, done):
    # Critic uses privileged info for better value estimates
    value = self.critic(torch.cat([obs, privileged_obs], dim=-1))
    next_value = self.critic(torch.cat([next_obs, next_priv_obs], dim=-1))
    
    # Compute advantage using privileged information
    advantage = reward + gamma * next_value * (1 - done) - value
    
    # Actor learns from advantage, but doesn't see privileged info
    action_logprobs = self.actor.log_prob(obs, action)
    actor_loss = -(action_logprobs * advantage.detach()).mean()
    
    # Critic learns to predict value with privileged information
    critic_loss = (value - target_value).pow(2).mean()
    
    return actor_loss, critic_loss
```

**Deployment** (hardware):
```python
# Only actor network deployed
# No privileged observations needed
action = actor(observation)
```

**Why this works**:

- **Better credit assignment**: Critic knows true state (friction, contacts) → more accurate advantages
- **Faster learning**: Policy gets cleaner learning signal
- **Richer behaviors**: Actor learns to implicitly infer privileged information from history
- **Hardware-deployable**: Actor never depends on privileged info

## Student-Teacher Distillation: Learning from Oracle Policies

**Scenario**: Task requires information available in simulation but not on hardware

**Example**: Navigating rough terrain with elevation map from top-down sensor (unavailable on robot)

#### Two-Stage Training

**Stage 1: Train Teacher with Perfect Information**

```python
class TeacherPolicy(nn.Module):
    def __init__(self, obs_dim, privileged_dim, action_dim):
        super().__init__()
        
        # Teacher sees EVERYTHING
        self.network = nn.Sequential(
            nn.Linear(obs_dim + privileged_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, obs, privileged_obs):
        x = torch.cat([obs, privileged_obs], dim=-1)
        return self.network(x)

# Train teacher with RL
teacher_obs = np.concatenate([
    base_observations,
    elevation_map,  # Perfect terrain knowledge
    object_positions,  # Perfect localization
])

# Teacher achieves near-optimal performance
```

**Stage 2: Distill into Student with Limited Information**

```python
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Student only sees hardware-available info
        # May include history or learned representations
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim * 10, 256),  # 10 timesteps history
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        
        self.policy_head = nn.Linear(128, action_dim)
    
    def forward(self, obs_history):
        features = self.encoder(obs_history)
        return self.policy_head(features)

# Train student via behavior cloning
def distill_student(teacher, student, dataset):
    for obs_history, privileged_obs in dataset:
        # Teacher action (with privileged info)
        teacher_action = teacher(obs_history[..., -1, :], privileged_obs)
        
        # Student tries to match
        student_action = student(obs_history)
        
        # Behavior cloning loss
        loss = F.mse_loss(student_action, teacher_action.detach())
        loss.backward()
        optimizer.step()
```

**DAgger (Dataset Aggregation)** — improves distillation:

```python
# Iterative distillation
for iteration in range(10):
    # 1. Collect data using student policy
    student_trajectories = collect_trajectories(student, env)
    
    # 2. Label with teacher actions
    for obs_history, privileged_obs in student_trajectories:
        teacher_action = teacher(obs_history[..., -1, :], privileged_obs)
        dataset.add(obs_history, teacher_action)
    
    # 3. Update student on aggregated dataset
    student.train(dataset)
    
    print(f"Iteration {iteration}: Student performance = {evaluate(student)}")
```

**Result**: Student learns to implicitly infer privileged information from observation history

## Reward Shaping: Specifying What You Want Without Over-Constraining

Reward design determines what behavior emerges. Bad rewards → bad behavior, even with perfect algorithms.

#### Reward Design Principles

**1. Sparse vs. Dense Rewards**

**Sparse** (reward only at goal):
```python
reward = 1.0 if reached_goal else 0.0
```
- ✅ Doesn't over-specify how to achieve goal
- ❌ Very hard to learn (credit assignment problem)

**Dense** (reward at every step based on progress):
```python
reward = -distance_to_goal  # Negative distance
```
- ✅ Easier to learn (clear gradient)
- ⚠️ May learn unintended shortcuts

**Curriculum rewards** (recommended):
```python
# Start dense, gradually make sparse
alpha = min(1.0, timestep / 1e9)  # Gradually increase to 1
reward = (1 - alpha) * dense_reward + alpha * sparse_reward
```

**2. Multi-Objective Reward**

Most tasks require balancing multiple objectives:

```python
class LocomotionReward:
    def compute(self, state, action):
        # Primary objective: forward velocity
        reward_velocity = state.velocity_x * dt
        
        # Secondary objectives (regularization)
        reward_energy = -0.001 * torch.sum(action ** 2)  # Energy efficiency
        reward_smoothness = -0.005 * torch.sum((action - action_prev) ** 2)
        reward_upright = 0.1 * (1.0 - abs(state.pitch))  # Stay upright
        reward_contact = -0.01 * torch.sum(state.foot_forces_z > 0)  # Minimize contact
        
        # Penalties
        penalty_joint_limits = -1.0 if any(state.q > q_max) else 0.0
        penalty_fall = -10.0 if state.height < 0.3 else 0.0
        
        # Combine
        total_reward = (
            reward_velocity +
            reward_energy +
            reward_smoothness +
            reward_upright +
            reward_contact +
            penalty_joint_limits +
            penalty_fall
        )
        
        return total_reward
```

**Weight tuning**: Iterate to get desired behavior, often requires dozens of attempts

**3. Termination Conditions as Implicit Rewards**

```python
# Termination = penalty = -remaining_reward
# If episode can last 1000 steps with reward +1 per step:
# Early termination at step 100 → lose 900 reward
# This creates implicit penalty for failure

def check_termination(state):
    if state.height < 0.3:  # Fell
        return True, -10.0  # Explicit penalty
    
    if abs(state.roll) > 1.0:  # Tipped over
        return True, -10.0
    
    if timestep >= max_timesteps:
        return True, 0.0
    
    return False, 0.0
```

**4. Shaping Rewards Must Be Potential-Based**

To avoid reward hacking, use **potential-based shaping**:

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

Where $\Phi(s)$ is a potential function.

**Example**:
```python
# ❌ BAD: Not potential-based (can be hacked)
reward = -current_distance_to_goal

# ✅ GOOD: Potential-based (progress reward)
potential = -distance_to_goal
reward = gamma * potential_next - potential_current
# Rewards only for making progress, not for being close
```

#### Common Reward Hacking Examples

**Problem**: Policy finds unintended way to maximize reward

**Example 1: Knee-jerking**
```python
# Reward: High forward velocity
reward = state.velocity_x

# Policy learns: Oscillate knees rapidly → high velocity measurement → high reward
# But robot doesn't actually move forward (velocity sensor noise)
```

**Fix**: Measure velocity over longer horizon, penalize joint accelerations
```python
reward = np.mean(state.velocity_x_history[-10:])  # Average over 10 steps
reward += -0.01 * torch.sum(joint_accelerations ** 2)
```

**Example 2: Spinning**
```python
# Reward: Reach goal position
reward = -distance_to_goal

# Policy learns: Spin in circles around goal (distance always small)
```

**Fix**: Require forward progress toward goal
```python
# Vector from robot to goal
goal_direction = (goal_position - robot_position) / distance_to_goal
# Dot product with velocity → reward for moving toward goal
reward = torch.dot(goal_direction, robot_velocity)
```

**Example 3: Falling forward**
```python
# Reward: Forward velocity
reward = state.velocity_x

# Policy learns: Fall forward (briefly high velocity before hitting ground)
```

**Fix**: Penalize termination, require staying upright
```python
reward = state.velocity_x if state.is_alive else -10.0
reward += 0.1 * (1.0 - abs(state.pitch))  # Upright bonus
```

## System Identification: Making Simulation Match Reality

**Domain randomization helps, but proper system identification is critical.**

#### What to Measure

**1. Mass and Inertia**
```python
# Measure real robot mass distribution
# CAD models often inaccurate (cables, sensors, batteries add mass)

real_mass = measure_on_scale()  # kg
sim_mass = robot_model.get_total_mass()

mass_error = abs(real_mass - sim_mass) / real_mass
if mass_error > 0.05:  # >5% error
    print(f"WARNING: Mass mismatch {mass_error*100:.1f}%")
    # Update simulation model
```

**2. Actuator Bandwidth**
```python
# Measure actuator step response
def measure_actuator_response(joint_id):
    # Command step input
    set_position(joint_id, 0.0)
    time.sleep(1.0)
    
    # Record response to step
    t0 = time.time()
    set_position(joint_id, 1.0)  # 1 radian step
    
    times = []
    positions = []
    for _ in range(1000):  # 1 second at 1kHz
        times.append(time.time() - t0)
        positions.append(get_position(joint_id))
        time.sleep(0.001)
    
    # Fit second-order system
    # Estimate natural frequency, damping ratio
    omega_n, zeta = fit_second_order(times, positions)
    
    return omega_n, zeta

# Update simulation actuator model
sim_actuator.set_bandwidth(omega_n, zeta)
```

**3. Sensor Noise Characteristics**
```python
# Measure IMU noise when robot is stationary
def characterize_imu_noise():
    samples = []
    for _ in range(10000):  # 10 seconds at 1kHz
        samples.append(imu.read_angular_velocity())
        time.sleep(0.001)
    
    samples = np.array(samples)
    
    # Compute noise statistics
    noise_std = np.std(samples, axis=0)
    noise_bias = np.mean(samples, axis=0)
    
    print(f"Angular velocity noise std: {noise_std}")
    print(f"Angular velocity bias: {noise_bias}")
    
    # Update simulation
    sim_imu.set_noise(std=noise_std, bias=noise_bias)
```

**4. Friction Coefficient**
```python
# Measure friction on different surfaces
def measure_friction(surface_type):
    # Pull robot sideways with force gauge
    # Measure force required to initiate sliding
    
    normal_force = robot_weight
    lateral_force = measure_pulling_force()
    
    mu_static = lateral_force / normal_force
    
    print(f"{surface_type} friction: {mu_static:.2f}")
    
    # Add to domain randomization range
    domain_rand.add_surface(surface_type, mu_static)
```

#### Iterative Sim-to-Real Calibration

```python
def calibration_loop():
    for iteration in range(10):
        print(f"\n=== Calibration Iteration {iteration} ===")
        
        # 1. Deploy policy to hardware
        hardware_success_rate = evaluate_on_hardware(policy, n_trials=50)
        
        # 2. Measure hardware behavior
        hardware_trajectories = collect_hardware_data(policy, n_episodes=10)
        
        # 3. Compare with simulation predictions
        sim_trajectories = simulate_same_commands(hardware_trajectories)
        
        # 4. Compute error metrics
        trajectory_error = compute_trajectory_error(hardware_trajectories, sim_trajectories)
        
        print(f"Hardware success rate: {hardware_success_rate:.1%}")
        print(f"Trajectory RMSE: {trajectory_error:.3f}")
        
        if hardware_success_rate > 0.9 and trajectory_error < 0.1:
            print("Calibration successful!")
            break
        
        # 5. Update simulation parameters to minimize error
        # Use system identification techniques
        updated_params = optimize_sim_params(
            hardware_trajectories,
            sim_trajectories,
            current_params
        )
        
        update_simulation(updated_params)
        
        # 6. Fine-tune policy in updated simulation
        policy = fine_tune_policy(policy, n_steps=10e6)
```

## Deployment Infrastructure: Monitoring and Iteration

**Production RL systems require robust deployment infrastructure.**

#### Real-Time Monitoring

```python
class RobotMonitor:
    def __init__(self):
        self.metrics = {
            'joint_positions': [],
            'joint_velocities': [],
            'joint_torques': [],
            'imu_orientation': [],
            'policy_inference_time': [],
            'command_execution_rate': [],
        }
        
    def log_state(self, robot_state, policy_output, dt):
        self.metrics['joint_positions'].append(robot_state.q)
        self.metrics['joint_torques'].append(robot_state.tau)
        self.metrics['policy_inference_time'].append(dt)
        
        # Safety checks
        self.check_joint_limits(robot_state.q)
        self.check_torque_limits(robot_state.tau)
        self.check_orientation(robot_state.orientation)
        self.check_latency(dt)
    
    def check_joint_limits(self, q):
        if np.any(q < self.q_min) or np.any(q > self.q_max):
            self.trigger_alert("Joint limit exceeded!")
            self.emergency_stop()
    
    def check_latency(self, dt):
        if dt > 0.05:  # 50ms threshold
            self.trigger_warning(f"High latency: {dt*1000:.1f}ms")
```

#### Safety Layer

```python
class SafetyLayer:
    """Wraps policy to enforce safety constraints"""
    
    def __init__(self, policy):
        self.policy = policy
        self.last_action = None
        
    def get_action(self, observation):
        # Get policy action
        raw_action = self.policy(observation)
        
        # Clip to safe ranges
        safe_action = self.enforce_constraints(raw_action, observation)
        
        self.last_action = safe_action
        return safe_action
    
    def enforce_constraints(self, action, observation):
        # 1. Joint position limits
        q_current = observation['joint_positions']
        q_desired = self.compute_desired_position(action, q_current)
        q_safe = np.clip(q_desired, self.q_min, self.q_max)
        
        # 2. Velocity limits
        if self.last_action is not None:
            action_change = action - self.last_action
            max_change = 0.1  # Limit acceleration
            action_change = np.clip(action_change, -max_change, max_change)
            action = self.last_action + action_change
        
        # 3. Stability check
        if self.is_unstable(observation):
            # Override with safe default action (stand still)
            action = self.get_safe_default_action()
        
        return action
    
    def is_unstable(self, observation):
        # Check if robot is tipping
        roll = observation['orientation_roll']
        pitch = observation['orientation_pitch']
        
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # 30 degrees
            return True
        
        return False
```

#### Hardware-in-the-Loop Testing

```python
class HILTester:
    """Test policies on hardware systematically"""
    
    def run_test_suite(self, policy):
        results = {}
        
        # Test 1: Standing stability
        results['standing'] = self.test_standing(policy, duration=30)
        
        # Test 2: Walking on flat ground
        results['flat_walk'] = self.test_walking(
            policy, surface='flat', distance=10
        )
        
        # Test 3: Walking on carpet
        results['carpet_walk'] = self.test_walking(
            policy, surface='carpet', distance=10
        )
        
        # Test 4: External disturbances
        results['push_recovery'] = self.test_push_recovery(
            policy, n_pushes=10
        )
        
        # Test 5: Command tracking
        results['command_tracking'] = self.test_command_following(
            policy, commands=generate_test_commands()
        )
        
        # Test 6: Battery depletion
        results['low_battery'] = self.test_low_voltage(
            policy, voltage_range=(20, 24)
        )
        
        return results
```

## Summary: Engineering Robust RL Deployments

Deploying reinforcement learning from simulation to reality is fundamentally an **engineering challenge**, not an algorithmic one. The difference between research demos that work once and production systems that work reliably is **systematic engineering across multiple layers**.

**Key Engineering Principles**:

1. **Action Space Design**: Use position residuals with PD control for safe, stable exploration and deployment

2. **Observation Space**: Start minimal. Add information only when task genuinely requires it. Fix system identification before augmenting observations.

3. **Domain Randomization**: Systematically vary simulation parameters to prepare for real-world variation. Use curriculum to balance learning speed and robustness.

4. **Asymmetric Actor-Critic**: Exploit privileged simulation information for better training while keeping deployment practical.

5. **System Identification**: Measure real hardware carefully. Make simulation match reality through iterative calibration.

6. **Reward Shaping**: Design rewards that specify goals without over-constraining methods. Use potential-based shaping to avoid reward hacking.

7. **Safety and Monitoring**: Deploy with safety layers, real-time monitoring, and systematic testing.

**Common Failure Modes to Avoid**:

- Over-augmenting observations as a crutch for poor system ID
- Using high PD gains that limit exploration
- Domain randomization without physical motivation
- Reward hacking through poorly designed objectives
- Deploying without systematic hardware testing
- Ignoring actuator dynamics and latency

**The Path to Production**:

1. Start with well-identified simulation (system ID)
2. Design action space for safety (residuals + PD)
3. Keep observations minimal (only hardware-available)
4. Use asymmetric actor-critic with privileged info
5. Apply domain randomization with curriculum
6. Iteratively calibrate simulation to hardware
7. Deploy with safety layer and monitoring
8. Test systematically across conditions
9. Iterate based on hardware failures

**Sim-to-real transfer is possible**, as evidenced by recent successes in quadrupedal locomotion and emerging results in manipulation. But success requires treating deployment as an **engineering discipline** that combines machine learning with controls, robotics, and systems engineering.

The future of RL deployment lies not in better algorithms (though they help), but in **better engineering practices**: simulation fidelity, systematic testing, safety infrastructure, and iterative development cycles that tighten the sim-to-real loop.

---

*This article is part of the Tech Demystified series. For more articles on AI, robotics, and production ML systems, see the [Tech Demystified repository](https://github.com/harshitha-8/Tech-Demystified).*

**References and Further Reading:**
- Hwangbo et al. (2019). "Learning agile and dynamic motor skills for legged robots"
- Miki et al. (2022). "Learning robust perceptive locomotion for quadrupedal robots in the wild"
- Lee et al. (2020). "Learning quadrupedal locomotion over challenging terrain"
- Rudin et al. (2022). "Learning to walk in minutes using massively parallel deep reinforcement learning"
- Peng et al. (2018). "Sim-to-real transfer of robotic control with dynamics randomization"
- OpenAI et al. (2019). "Learning dexterous in-hand manipulation"
- Andrychowicz et al. (2020). "Learning dexterous in-hand manipulation"
