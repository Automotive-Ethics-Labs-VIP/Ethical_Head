# Utilitarian 100% Accuracy — Strategy Options

## Background

The utilitarian label is a **deterministic function** of three state features:
- `state[4]` — pedestrians killed if maintain course
- `state[5]` — pedestrians killed if swerve left
- `state[6]` — pedestrians killed if swerve right

Rule: `argmin(state[4], state[5], state[6])`, ties broken by preference order
`maintain > swerve_left > swerve_right`. Verified 200/200 labels match this rule.

100% accuracy is achievable. The barrier is the training approach, not the data.

---

## Option A — Behavioral Cloning (Supervised Learning)

**Core idea:** Replace RL entirely with cross-entropy training directly on labeled scenarios.

**Why it works:** Labels are ground truth with no noise. 160 training scenarios with a
deterministic rule → a small MLP will fit them perfectly and generalize to the 40 test
scenarios because they follow the same rule.

**Files to add:**
- `scripts/train_supervised.py` — new training script
- `training/supervised_trainer.py` — trainer using cross-entropy loss

**Training loop (pseudocode):**
```python
for epoch in range(epochs):
    for sid, state in train_scenarios.items():
        label = decisions[sid]['utilitarian']       # ground-truth 0/1/2
        logits = policy_net(state_tensor)[[0,2,3]]  # mask invalid actions 1, 4
        loss = cross_entropy(logits, remap_label(label))
        loss.backward()
    optimizer.step()
```

**Expected result:** ~100% train accuracy, ~100% test accuracy.

**Pros:**
- Guaranteed convergence (deterministic labels, sufficient capacity)
- Fast — no rollout collection, no reward model, no GAE
- No class imbalance problem (cross-entropy weights each sample equally)

**Cons:**
- Replaces the RLHF pipeline entirely for utilitarian
- Does not produce a reward model (only a policy)
- Less "interesting" academically — it's memorisation of 160 scenarios

---

## Option B — Death-count Shaped Reward (Improved RL)

**Core idea:** Keep the RL pipeline but replace the binary ±1 reward with a
continuous death-count penalty and remove inverse-frequency weighting.

**Changes to `environments/ethical_env.py`:**

```python
# Remove CLASS_REWARD_WEIGHTS entirely.
# In EthicalScenarioEnv.step():

peds = [state[4], state[5], state[6]]  # straight / left / right
deaths_chosen  = peds[predicted_label]
deaths_optimal = min(peds)

if action in INVALID_POLICY_ACTIONS:
    reward = -5.0                              # hard penalty, unchanged
elif predicted_label == true_label:
    reward = +1.0                              # correct
else:
    reward = -(deaths_chosen - deaths_optimal) # graded: -1 to -N
```

**Also apply preference oversampling fix in `training/trainer_patched.py`:**
Remove the inverse-frequency weighting from `_balance_preferences` so that
swerve_left (50% of data) is not underweighted in the reward model.

**Expected result:** Stronger gradient signal → model learns to compare death counts
rather than memorise a pattern. Should reach 80–95%+ given enough timesteps.

**Pros:**
- Stays within the existing RLHF framework
- Produces both a policy and a reward model
- More robust to label noise (graded signal, not binary)

**Cons:**
- No guarantee of 100% — RL convergence depends on hyperparameters
- Still requires long training runs (500k–2M timesteps)
- Reward model and PPO add variance

---

## Decision

**Option A was chosen for implementation** (see `scripts/train_supervised.py` and
`training/supervised_trainer.py`). Option B is available as a fallback if the
supervised approach needs to be replaced with one that keeps the RLHF structure.

To switch to Option B: revert to the `trainer_patched.py` training path and apply
the reward shaping changes described above.

---

## Bugs Fixed (prior to this session — previously in CHANGELOG.md)

### Bug 1 — Combined rewards bypassed in PPO advantage computation
**Files:** `training/trainer.py`, `data/replay_buffer.py`

`update_policy()` correctly computed `combined_rewards` (base + learned ethical
reward) but never passed it anywhere. The subsequent call to
`self.buffer.compute_advantages_and_returns()` ignored `combined_rewards` entirely
and read raw rewards directly from `self.buffer.rewards`, cutting the reward model
out of every training update.

**Fix:** Added an optional `rewards` parameter to
`ReplayBuffer.compute_advantages_and_returns()`. When provided, GAE uses this
tensor instead of `self.rewards`. `update_policy()` now passes `combined_rewards`
to that call so the reward model's signal propagates into PPO advantages as intended.

---

### Bug 2 — Mock environment used pure Gaussian noise as base reward
**File:** `training/trainer.py` (`_step_environment`)

`_step_environment()` set `reward = np.random.randn()` when no real environment
was attached. This made `R_base` semantically meaningless, providing no gradient
signal to guide the policy.

**Fix:** Replaced the Gaussian sample with a structured mock reward:
```python
reward = float(np.clip(1.0 - 0.5 * abs(state[0]), -1.0, 1.0))
```
This rewards the agent for keeping `state[0]` (a speed/position proxy) near
neutral, giving the policy a learnable signal without requiring a real environment.
