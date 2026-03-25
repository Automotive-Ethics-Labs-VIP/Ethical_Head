# Ethical Head - Autonomous Vehicle Decision System

Welcome to the Ethical Head project! This system acts as the "brain" for self-driving cars in the CARLA simulator, helping them make safe and ethical decisions when faced with difficult situations on the road.

## Core Components
Our model relies on several key components that work together to drive the car:

1. **Policy Network (The Decision Maker):** This is the part of the brain that looks at the current situation and decides what action to take (e.g., brake, swerve, accelerate).
2. **Value Network (The Critic):** This part looks at the road and predicts how "good" or "safe" the current situation is.
3. **Reward Model (The Judge):** This is an AI that has learned from human feedback. It watches the decisions the car makes and gives a score based on how ethical the action was.
4. **Replay Buffer (The Memory):** A temporary storage area where the car remembers its recent actions and what happened next, so it can learn from them later.
5. **Ethical Agent (`ethical_agent.py`):** The primary interface that connects our AI brain to the car exactly how the CARLA simulator expects.

## How the Components Interact
- When the car is driving, the **CARLA simulator** sends a snapshot of the world (a 40-number list describing speed, pedestrians, obstacles, etc.) to the **Ethical Agent**.
- The **Ethical Agent** passes this snapshot to the **Policy Network**, which chooses the best action.
- During training, the **Reward Model** and the **Value Network** step in to evaluate these actions. They give the car "rewards" (points) or penalties, which the car stores in its **Replay Buffer**.
- Finally, the system uses those stored memories to update the **Policy Network** so it makes better decisions next time!

## The Training Process
Training the car is like training a dog—we use a process called Reinforcement Learning:
1. **Rollout (Practice):** The car drives around in the simulator, making sequence of decisions and remembering what happened.
2. **Evaluation (Scoring):** After a while, the car stops and looks at its memories. The **Reward Model** scores how well the car did. 
3. **Update (Learning):** The car uses a method called PPO (Proximal Policy Optimization) to adjust its brain. It tries to figure out how to get higher scores next time without forgetting how to drive properly.

## Ethical Theories & Human Feedback
What makes a decision "ethical"? We mix explicit rules with human intuition!

### Explicit Ethical Theories
We program the car with classic philosophy rules:
- **Utilitarianism:** "Maximize the good." The car tries to minimize the total number of injuries, choosing the action that harms the absolute fewest people.
- **Kantian Ethics:** "Intention matters." The car refuses to intentionally use people as tools. For example, it might aggressively brake to avoid a crowd, but it will *never* intentionally swerve *into* a bystander just to save others.
- **Virtue Ethics:** "What would a good driver do?" The car is rewarded for showing courage, wisdom, and avoiding reckless acceleration.

### Human Feedback (RLHF)
Sometimes, philosophical rules are too rigid. That's where humans come in! 
Through a process called Reinforcement Learning from Human Feedback (RLHF), humans watch pairs of situations the car drove through and simply vote on which one was "better" or "more ethical." 
The **Reward Model** learns to predict these human preferences and scores the car accordingly, blending strict ethical theories with real human values.

## Usage Guide
Using the model is straightforward and requires no complex AI knowledge!

### 1. Driving the Car (Predicting Actions)
To use the model to drive the car in CARLA, we use the `ethical_agent.py` file. The CARLA simulator will send a **40-dimensional state vector** (a list of 40 numbers containing the car's speed, the number of pedestrians, positions, etc.). 

Here's how to feed that input into the system to get an action:

```python
from ethical_agent import EthicalAgent

# 1. Load the trained brain
agent = EthicalAgent('checkpoints/trained_model.pt')

# 2. Get the 40-number input from CARLA (example snapshot)
# This includes velocity, pedestrians, obstacles, etc.
carla_state_vector = [20.5, 2, 0.0, 1.2, 0, 1, 0] + [0.0]*33 

# 3. Ask the agent for the best decision
action = agent.get_action(carla_state_vector)

# The action will be a number from 0 to 4:
# 0: maintain_course, 1: brake_hard, 2: swerve_left, 3: swerve_right, 4: accelerate
print(f"The car decided to: {agent.get_action_name(action)}")
```

### 2. Adding Human Feedback
If you are watching the car drive and want to provide feedback on a specific decision, you can do so directly through the agent:

```python
# Enable feedback tracking when loading the agent
agent = EthicalAgent('checkpoints/trained_model.pt', enable_feedback=True)

# The car makes a decision
action = agent.get_action(carla_state_vector)

# You observe the car and give it a rating (1-5) and some text feedback
agent.add_human_feedback(
    state=carla_state_vector,
    action=action,
    feedback="The car braked perfectly in time to avoid the pedestrian.",
    rating=5
)

# Save the feedback to a file so the reward model can learn from it later!
agent.save_feedback('data/human_feedback.json')
```

### 3. Collecting Preference Data for Training 
To collect pairs of trajectories for human annotators to compare:
```bash
python scripts/collect_trajectories.py
```
This will save driving clips to `data/unannotated_pairs.json`. 

Next, humans can vote on these clips using the annotation tool:
```bash
python scripts/annotate_trajectories.py
```
This saves the human votes to `data/human_preferences.json`, which the Reward Model will use to learn human ethics!