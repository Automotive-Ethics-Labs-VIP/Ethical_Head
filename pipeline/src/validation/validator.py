import numpy as np
import yaml
import math

class Validator:
    """Validate extracted state vectors."""
    def __init__(self, config_path: str = 'config/config.yaml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # expected dimension for full state vector
        self.expected_dim = 40
        # flatten nested validation config into simple ranges dict
        self.ranges = config.get('validation', {})
        # names of features in order; adjust when full extractor is implemented
        self.feature_order = [
            'velocity_ego',
            'num_passengers',
            'lane_position',
            'velocity_delta',
            'num_ped_if_straight',
            'num_ped_if_left',
            'num_ped_if_right'
        ] + [f'obstacle_{i}' for i in range(33)]

    def validate(self, state: np.ndarray) -> bool:
        """
        Validate a single state vector.
        :param state: numpy array or list of length 40
        :return: True if valid; raises ValueError on problems
        """
        # convert to numpy array
        vector = np.asarray(state, dtype=float)
        # dimensionality check
        if vector.shape[0] != self.expected_dim:
            raise ValueError(f"State vector must have {self.expected_dim} elements, "
                             f"got {vector.shape[0]}")
        # NaN / missing
        if np.isnan(vector).any():
            raise ValueError("State vector contains NaN values")
        # range checks
        for idx, value in enumerate(vector):
            # look up range by name when possible
            feature_name = self.feature_order[idx] if idx < len(self.feature_order) else None
            range_key = f"{feature_name}_range" if feature_name else None
            if range_key and range_key in self.ranges:
                low, high = self.ranges[range_key]
                if not (low <= value <= high):
                    raise ValueError(
                        f"Feature '{feature_name}' out of range "
                        f"({value} not in [{low}, {high}])"
                    )
        # data type check (must be float or int)
        if not all(isinstance(x, (int, float, np.floating, np.integer)) for x in vector):
            raise TypeError("State vector must contain numeric types")
        return True
