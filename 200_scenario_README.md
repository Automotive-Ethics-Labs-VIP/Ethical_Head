# CATA 200-Scenario RLHF Pipeline

Extends the original 30-scenario run to the full 200-scenario dataset.

---

## Data files generated

| File | Contents |
|------|----------|
| `data/scenarios_200.json` | 200 × 40-D state vectors |
| `data/decisions_200.json` | 200 × {utilitarian, kantian} labels |
| `data/preferences/edge_cases_utilitarian.json` | 320 train preference pairs |
| `data/preferences/edge_cases_kantian.json` | 320 train preference pairs |
| `data/splits/train_test_split.json` | 160 train / 40 test (stratified 80/20) |

Label distribution (utilitarian, 200 total):
- maintain     60  (30 %)
- swerve_left 100  (50 %)
- swerve_right 40  (20 %)

Kantian: always "maintain" (200/200) — model should reach ~100% easily.

---

## Quick-start workflow

```bash
# ── 0. Clean old checkpoints ──────────────────────────────────────────
rm -rf checkpoints/utilitarian checkpoints/kantian

# ── 1. Parse PDFs into JSON data (already done — skip if files exist) ─
python scripts/parse_200_scenarios.py

# ── 2. Create stratified 80/20 train/test split ───────────────────────
python scripts/split_scenarios_200.py

# ── 3a. Quick smoke-test (100k steps, ~1 min) ─────────────────────────
python scripts/train_edge_cases_200.py --theory utilitarian --timesteps 100000

# ── 3b. Full training run ─────────────────────────────────────────────
python scripts/train_edge_cases_200.py --theory utilitarian --timesteps 500000
python scripts/train_edge_cases_200.py --theory kantian     --timesteps 200000

# ── 4. Evaluate on held-out test set (40 scenarios) ───────────────────
python scripts/test_edge_cases_200.py \
    --checkpoint checkpoints/utilitarian/final_checkpoint.pt \
    --theory utilitarian

python scripts/test_edge_cases_200.py \
    --checkpoint checkpoints/kantian/final_checkpoint.pt \
    --theory kantian

# ── 5. Optional: evaluate on ALL 200 to see training fit ──────────────
python scripts/test_edge_cases_200.py \
    --checkpoint checkpoints/utilitarian/final_checkpoint.pt \
    --theory utilitarian \
    --split all
```

---

## Key improvements over the 30-scenario run

### 1. Stratified split (new)
`split_scenarios_200.py` groups scenarios by utilitarian label before
splitting, so the 40-scenario test set gets ~12 maintain / ~20 left /
~8 right instead of a random draw that might be all one class.

### 2. Richer preference pairs
Each of the 160 training scenarios generates **2 preference pairs**
(preferred vs each of the two non-preferred actions), giving 320 pairs
total vs. the original 48. This gives the reward model ~6× more signal.

### 3. Better test reporting (`test_edge_cases_200.py`)
- Per-class accuracy with visual bar
- Full 3×3 confusion matrix (rows = true, cols = predicted)
- List of every misclassified scenario
- `--split all` mode to check training fit separately from generalisation

---

## Expected results (calibrated from 30-scenario run)

| Theory | Train acc | Test acc |
|--------|-----------|----------|
| Kantian | ~100% | ~100% |
| Utilitarian | 55–75% | 45–65% |

Utilitarian is the hard problem: the model must learn that the correct
action is whichever minimises total deaths, which requires reading
dims [5,6,7] (num_ped_if_straight/left/right) and comparing them.

If utilitarian test accuracy is stuck below 40%:
- Try `--timesteps 1000000`
- Check whether your reward model is trained on preference pairs or
  directly supervised — direct supervision often converges faster here
- Consider adding a shaped reward: `reward = -num_killed` rather than
  pure preference learning

---

## Scenario encoding reminder (40-D vector)

```
[0]  velocity_ego
[1]  num_passengers
[2]  lane_position
[3]  velocity_delta
[4]  num_ped_if_straight    ← key for utilitarian
[5]  num_ped_if_left        ← key for utilitarian
[6]  num_ped_if_right       ← key for utilitarian
[7–17]  left-path features  (pedestrian/cyclist/vehicle/…/child/elderly/pregnant/disabled)
[18–28] straight-path features
[29–39] right-path features
```

Action space:  0 = maintain  |  1 = swerve_left  |  2 = swerve_right
