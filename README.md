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


## Ethical Configuration

The system supports three reward components:

1. **R_base (α=0.3)**: Safety heuristics
2. **R_learned (β=0.7)**: Learned from human preferences
3. **R_explicit (optional)**: Explicit ethical theories

### Default (Learned Only):
- Ethics implicitly encoded in human preference labels
- Reflects whatever ethical framework humans used when labeling

### With Explicit Ethics:
- Adds formal utilitarian, Kantian, and virtue ethics
- Provides theoretical grounding and consistency
- Can override learned biases

### Example:
```python
# Utilitarian-leaning
config = Config()
config.use_explicit_ethics = False

# With ethical theory enforcement
config = Config()
config.use_explicit_ethics = True
```