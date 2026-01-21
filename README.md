# AEL Team A - RLHF Implementation

## Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
source venv/bin/activate
```

Requires **Python 3.10**.

---

## Dependency Compatibility

All packages tested compatible with Python 3.10 and each other:

| Package | Version | Depends On |
|---------|---------|------------|
| torch | 2.2.2 | numpy 1.26.4 ✓ |
| stable-baselines3 | 2.4.0 | torch ≥1.13 ✓, gymnasium ✓ |
| gymnasium | 0.29.1 | numpy ✓ |
| numpy | 1.26.4 | - |
| pandas | 2.2.2 | numpy ✓ |
| scipy | 1.13.0 | numpy ✓ |
| tensorboard | 2.16.2 | numpy ✓ |

**Note:** NumPy pinned to 1.26.4 (NumPy 2.x breaks torch/scipy compatibility).

---

## Current Task Assignment

### Pair 1: Policy Network
**Team Members:** Khai + Nicolo

**Task:** Implement Policy Network
- Input: 40D state vector
- Output: Probability distribution over 5 actions
- Architecture: Input(40) → Dense(64, ReLU) → Dense(32, ReLU) → Softmax(5)
- File: `src/networks/policy.py`

**Requirements:**
- Use PyTorch `nn.Module`
- Implement forward pass
- Add helper method to sample actions from distribution
- Total parameters: 4,869

---

### Pair 2: Value Network
**Team Members:** Sami + Ammar

**Task:** Implement Value Network
- Input: 40D state vector
- Output: Single scalar value (expected future reward)
- Architecture: Input(40) → Dense(64, ReLU) → Dense(32, ReLU) → Linear(1)
- File: `src/networks/value.py`

**Requirements:**
- Use PyTorch `nn.Module`
- Implement forward pass
- Output single value prediction
- Total parameters: ~3,500

---

**Deliverable:** Both networks implemented and tested with dummy 40D input by end of week.

---

## Architecture Reference

**State Vector (40D):**
```
[velocity_ego, num_passengers, lane_position, velocity_delta,
 num_ped_straight, num_ped_left, num_ped_right, obstacle_types(33)]
```

**Actions (5):** maintain_course, brake_hard, swerve_left, swerve_right, accelerate

**Networks:**
- Policy: 4,869 parameters
- Value: ~3,500 parameters  
- Reward: ~4,000 parameters


**Anticpated Structure**
project-root/
│
├── networks/                 # Neural networks only (pure modules)
│   ├── policy.py
│   ├── value.py
│   ├── reward_model.py
│   └── __init__.py
│
├── envs/                     # Environment + state construction
│   ├── carla_env.py
│   ├── state_encoder.py
│   └── __init__.py
│
├── data/                     # Offline data (not code)
│   ├── rollouts/             # Saved trajectories
│   ├── preferences/          # Human comparison labels
│   └── checkpoints/
│
├── buffers/                  # Experience storage
│   ├── rollout_buffer.py
│   └── preference_buffer.py
│
├── algorithms/               # Learning algorithms (NO env logic)
│   ├── ppo.py
│   ├── gae.py
│   ├── reward_training.py
│   └── __init__.py
│
├── training/                 # Orchestration scripts
│   ├── collect_rollouts.py
│   ├── train_reward_model.py
│   ├── train_policy.py
│   └── main_loop.py
│
├── evaluation/
│   ├── metrics.py
│   ├── eval_policy.py
│   └── human_eval.py
│
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── seeding.py
│   └── kl_divergence.py
│
├── configs/
│   ├── ppo.yaml
│   ├── reward_model.yaml
│   └── env.yaml
│
├── scripts/                  # CLI entry points
│   ├── train_sft.py
│   ├── train_rlhf.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
