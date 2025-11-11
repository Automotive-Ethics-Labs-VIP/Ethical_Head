import json
import numpy as np
from typing import List, Dict, Any

class Statistics:
    """
    Track feature distributions over multiple state vectors, and generate summaries.
    """
    def __init__(self):
        self.samples: List[np.ndarray] = []

    def update(self, state_vector: List[float]) -> None:
        """Add a state vector to the statistics buffer."""
        self.samples.append(np.asarray(state_vector, dtype=float))

    def summary(self) -> Dict[str, Any]:
        """Compute mean, standard deviation, min and max for each feature."""
        if not self.samples:
            return {}
        data = np.vstack(self.samples)
        stats = {
            'mean': data.mean(axis=0).tolist(),
            'std': data.std(axis=0).tolist(),
            'min': data.min(axis=0).tolist(),
            'max': data.max(axis=0).tolist(),
            'count': int(data.shape[0])
        }
        return stats

    def save(self, filepath: str) -> None:
        """Save summary statistics to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.summary(), f, indent=2)

    def load(self, filepath: str) -> Dict[str, Any]:
        """Load previously saved summary statistics."""
        with open(filepath, 'r') as f:
            return json.load(f)
