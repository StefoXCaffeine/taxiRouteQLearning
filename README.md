# Taxi Route Q-Learning

A reinforcement learning project that implements Q-Learning to train an autonomous taxi agent to navigate a grid world, pick up passengers, and drop them off at their destinations.

## Objective

The goal of this project is to implement a Q-Learning algorithm that enables a taxi agent to learn optimal navigation policies in the Gymnasium Taxi-v3 environment. The agent must learn to:
- Navigate through a 5x5 grid world
- Pick up passengers from designated locations
- Drop them off at their requested destinations
- Maximize cumulative rewards while minimizing penalties

## Architecture

### Environment
- **Gymnasium Taxi-v3**: A discrete environment with:
  - **500 possible states** (combinations of taxi location, passenger location, and destination)
  - **6 possible actions**: South, North, East, West, Pickup, Dropoff

### Q-Table
- **Dimensions**: 500 x 6 (states x actions)
- **Initialization**: Zero-initialized matrix
- **Purpose**: Stores Q-values representing the expected future rewards for each state-action pair

### Technology Stack
- Python
- NumPy (numerical computations)
- Gymnasium (reinforcement learning environment)
- Matplotlib & Seaborn (visualization)

## Methodology

### Q-Learning Algorithm

The project implements the standard Q-Learning update rule:

```
Q(s,a) ← Q(s,a) + α [R(s,a) + γ max Q(s',a') - Q(s,a)]
```

Where:
- **α (learning_rate)**: 0.01 - Controls how much new information overrides old information
- **γ (gamma)**: 0.99 - Discount factor for future rewards
- **R(s,a)**: Immediate reward from taking action a in state s
- **max Q(s',a')**: Maximum expected future reward from the new state

### Epsilon-Greedy Policy

Balances exploration vs exploitation:
- **Exploration**: Random action selection (probability ε)
- **Exploitation**: Best known action selection (probability 1-ε)

**Epsilon Decay**:
```
ε = ε_min + (ε_max - ε_min) * e^(-decay_rate * episode)
```

Parameters:
- ε_max: 1.0 (initial exploration rate)
- ε_min: 0.001 (minimum exploration rate)
- decay_rate: 0.01 (exponential decay)

## Procedure

1. **Environment Setup**: Initialize the Taxi-v3 environment with rendering capabilities
2. **Q-Table Initialization**: Create a 500x6 matrix initialized to zero
3. **Training Loop**:
   - For each episode:
     - Reset environment to initial state
     - Decay epsilon for exploration-exploitation balance
     - For each step:
       - Select action using epsilon-greedy policy
       - Execute action and observe reward and new state
       - Update Q-value using Bellman equation
       - Terminate if goal reached or max steps exceeded
4. **Evaluation**: Test the learned policy over multiple episodes

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| total_episodes | 10 | Training episodes |
| total_test_episodes | 10 | Testing episodes |
| max_steps | 100 | Maximum steps per episode |
| learning_rate | 0.01 | Step size for Q-value updates |
| gamma | 0.99 | Discount factor for future rewards |
| epsilon | 1.0 → 0.001 | Exploration rate (decaying) |
| decay_rate | 0.01 | Epsilon decay rate |

## Implementation Details

### Key Components

1. **Epsilon-Greedy Policy Function**: Dynamically selects between exploration and exploitation based on current epsilon value
2. **Q-Value Update**: Temporal Difference learning that bootstraps from current estimates
3. **Logging**: Detailed console output showing Q-value evolution for each state-action pair

### Reward Structure (Taxi-v3)
- **-1** per step (encourages efficiency)
- **+20** for successful dropoff
- **-10** for invalid pickup/dropoff actions

## Future Implementations

### Short-term Improvements
- **Increase Training Episodes**: Current 10 episodes are insufficient for convergence; recommend 1000+ episodes
- **Performance Metrics**: Add reward tracking and success rate visualization over episodes
- **Heatmap Visualization**: Display final Q-table as heatmap for policy interpretation

### Medium-term Enhancements
- **Deep Q-Learning (DQN)**: Replace Q-table with neural network for function approximation
- **Experience Replay**: Store and sample past experiences to break correlation between consecutive samples
- **Target Network**: Implement separate target network for stable training

### Long-term Extensions
- **Double Q-Learning**: Reduce maximization bias in Q-value estimation
- **Dueling DQN**: Separate value and advantage streams for better state evaluation
- **Custom Environments**: Create more complex taxi scenarios with multiple passengers, traffic, or dynamic obstacles
- **Multi-agent Systems**: Train multiple taxis to cooperate or compete in the same environment

## Usage

```bash
# Install dependencies
pip install numpy gymnasium matplotlib seaborn

# Run the notebook
jupyter notebook QLearning.ipynb
```

## License

This project is licensed under the terms specified in the LICENSE file.
