
# 🧠 Deep Q-Network (DQN) in Custom Environment

This project implements a Deep Q-Network (DQN) agent trained in a custom Gymnasium environment called `HomeFoodEnvStatic`. The agent learns to navigate a grid-based world to reach a desired food goal while avoiding distractions ("hell states") using reinforcement learning.

---

## 📘 What is DQN?

Deep Q-Network (DQN) combines Q-learning with deep neural networks to approximate the action-value function in environments with large or continuous state spaces.

### 🔑 Core Concepts:
- **Q-Learning**: An off-policy algorithm that learns the value of taking a given action in a given state.
- **Neural Network**: Used to estimate `Q(s, a)` values instead of a lookup table.
- **Experience Replay**: Past experiences are stored and sampled randomly to break correlation in training data.
- **Target Network**: A copy of the main network that updates slowly to provide stable learning targets.
- **Epsilon-Greedy Strategy**: Encourages exploration by choosing random actions with probability `ε`.

---

## 🔁 DQN Training Pipeline

1. Initialize Q-network and target network
2. Store experiences `(state, action, reward, next_state)` in a replay buffer
3. Sample mini-batches for training
4. Compute target Q-values:
   ```
   target = reward + gamma * max(Q_target(next_state))
   ```
5. Minimize the loss between predicted and target Q-values
6. Periodically update the target network

---

## 🖼️ Q-Value Heatmap

The following heatmap represents the Q-values learned by the agent:

![Q-Value Heatmaps](action_value_heatmap.png)

---

## 🚗 Application of Duelling DQN in Autonomous Vehicle Project

In a later project, we extended the DQN approach using **Duelling DQN (DDQN)** to train an **autonomous vehicle** on a simulated driving track.

### 🔍 Why DDQN?
Standard DQN tends to **overestimate action values**. DDQN reduces this bias by decoupling:
- **Action selection** using the **main network**
- **Action evaluation** using the **target network**

```python
a_max = argmax(Q_main(s'))
target = r + gamma * Q_target(s', a_max)
```

### 🚘 Project Highlights:
- State space: camera feed processed via CNN
- Action space: throttle, brake, steering
- Reward design: promotes lane-following, penalizes collisions and sudden turns
- DDQN helped stabilize training and improved overall performance

> ✅ **Outcome**: The DDQN-based agent demonstrated improved consistency and fewer crashes in complex driving scenarios compared to DQN.

---

## 📦 Dependencies

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Training
```bash
python main.py
```

---

## 🧠 Summary: DQN vs DDQN

| Feature        | DQN                      | DDQN                         |
|----------------|---------------------------|-------------------------------|
| Q-value Est.   | Can overestimate          | More accurate                 |
| Training       | Less stable               | More stable & consistent      |
| Use Case       | Simple tasks              | Complex environments like AVs |

---