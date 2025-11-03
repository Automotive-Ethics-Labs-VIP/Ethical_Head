# Feature Pipeline and Validation Framework

## Purpose

Extract 40-dimensional state vectors from CARLA simulation. Use mock pedestrian data until Team B provides custom pedestrian models.

## State Vector

```
State_vector = [
    velocity_ego,                    # 1
    num_passengers,                  # 1
    lane_position,                   # 1
    velocity_delta,                  # 1
    num_ped_if_straight,            # 1
    num_ped_if_left,                # 1
    num_ped_if_right,               # 1
    obstacle_type_per_action        # 33 (3 actions × 11 types)
]
Total: 40 dimensions
```

## Pedestrian Type Attributes

```
ped_type = [is_child, is_elderly, is_pregnant, is_disabled]
```

## Project Structure

```
pipeline/
├── config/
│   └── config.yaml
├── src/
│   ├── environment/
│   │   ├── carla_connector.py
│   │   ├── state_extractor.py
│   │   └── mock_pedestrian.py
│   └── validation/
│       ├── validator.py
│       └── statistics.py
├── tests/
│   ├── test_extraction.py
│   └── test_mock.py
├── scripts/
│   ├── extract.py
│   └── validate.py
└── README.md
```

## Mock Data

Mock pedestrian generator assigns random attributes to spawned pedestrians until Team B integration:

```python
mock_attributes = {
    'is_child': bool,
    'is_elderly': bool,
    'is_pregnant': bool,
    'is_disabled': bool
}
```

## Validation

Validator checks:
- State vector is 40 dimensions
- All values within valid ranges
- No missing data
- Correct data types

Run validation:
```bash
python scripts/validate.py
```

## Team B Integration

Replace mock attribute assignment with Team B's pedestrian blueprint attributes when available.