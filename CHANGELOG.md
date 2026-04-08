# Changelog

## [Unreleased] - 2026-04-08

### Fixed

#### Bug 1 — Combined rewards bypassed in PPO advantage computation (`training/trainer.py`, `data/replay_buffer.py`)

`update_policy()` correctly computed `combined_rewards` (base + learned ethical reward) but
never used it. The subsequent call to `self.buffer.compute_advantages_and_returns()` ignored
`combined_rewards` entirely and read raw rewards directly from `self.buffer.rewards`, cutting the
reward model out of every training update.

**Fix:** Added an optional `rewards` parameter to `ReplayBuffer.compute_advantages_and_returns()`
(`data/replay_buffer.py`). When provided, GAE uses this tensor instead of `self.rewards`.
`update_policy()` now passes `combined_rewards` to that call, so the reward model's signal
propagates into PPO advantages as intended.

---

#### Bug 2 — Mock environment used pure Gaussian noise as base reward (`training/trainer.py:550`)

`_step_environment()` set `reward = np.random.randn()` when no real environment was attached.
This made the base reward (`R_base`) semantically meaningless, providing no gradient signal to
guide the policy.

**Fix:** Replaced the Gaussian sample with a structured mock reward:

```python
reward = float(np.clip(1.0 - 0.5 * abs(state[0]), -1.0, 1.0))
```

This rewards the agent for keeping `state[0]` (a speed/position proxy) near neutral, giving
the policy a learnable signal without requiring a real environment.

---

### Note on `beta = 0.7` (`training/config.py:63`)

`beta = 0.7` was not changed. The value itself is correct design (70% learned reward, 30% base).
It only appeared harmful because Bugs 1 and 2 caused the entire reward signal to be Gaussian
noise — the reward model was bypassed (Bug 1) and the base reward was pure noise (Bug 2).
With both fixed, `beta = 0.7` now correctly weights the trained reward model's output.
