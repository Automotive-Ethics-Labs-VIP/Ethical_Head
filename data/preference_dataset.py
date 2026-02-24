"""
Human Preference Dataset Manager
================================
This module defines the `PreferenceDataset` class, which is responsible for storing, 
formatting, and managing human feedback data for the RLHF pipeline.

In the Reinforcement Learning from Human Feedback (RLHF) process, a human annotator
is presented with two trajectories (A and B) and must choose which one represents 
better/safer/more ethical driving behavior.

This module stores those choices in a structured JSON format and provides a critical
`get_training_format()` method that converts the saved preference data back into 
PyTorch tensors required for training the Bradley-Terry Reward Model.

Key components:
- `add_preference`: Adds a new (traj_a, traj_b, preference) tuple to memory.
- `save` / `load`: Persists the dataset to disk as a JSON file.
- `get_training_format`: Translates binary choices (0 or 1) into reward model targets (1.0 or 0.0).
"""

import json
import uuid
import datetime
from pathlib import Path
import torch

class PreferenceDataset:
    def __init__(self, filepath=None):
        """
        Initializes the PreferenceDataset.
        
        Args:
            filepath (str or Path, optional): Path to load/save preferneces.
        """
        self.filepath = Path(filepath) if filepath else None
        self.data = []
        
        if self.filepath and self.filepath.exists():
            self.load(self.filepath)

    def add_preference(self, traj_a, traj_b, preference, annotator_id="system"):
        """
        Adds a new preference record to the dataset.
        
        Args:
            traj_a (tuple): (states, actions) for trajectory A. Note states/actions should be lists or tensors.
            traj_b (tuple): (states, actions) for trajectory B.
            preference (int): 0 if A is better, 1 if B is better.
            annotator_id (str): ID of the human annotator.
        """
        
        # Helper to convert tensors to lists for JSON serialization
        def _to_list(item):
            if isinstance(item, torch.Tensor):
                return item.tolist()
            if isinstance(item, (list, tuple)):
                 # handle nested tensors just in case
                 return [_to_list(x) for x in item]
            return item
            
        states_a, actions_a = traj_a
        states_b, actions_b = traj_b
        
        record = {
            "id": f"pref_{uuid.uuid4().hex[:8]}",
            "trajectory_a": {
                "states": _to_list(states_a),
                "actions": _to_list(actions_a)
            },
            "trajectory_b": {
                "states": _to_list(states_b),
                "actions": _to_list(actions_b)
            },
            "preference": int(preference),
            "annotator_id": annotator_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.data.append(record)

    def save(self, filepath=None):
        """Saves the dataset to a JSON file."""
        path = Path(filepath) if filepath else self.filepath
        if not path:
            raise ValueError("No filepath provided for saving.")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def load(self, filepath=None):
        """Loads the dataset from a JSON file."""
        path = Path(filepath) if filepath else self.filepath
        if not path or not path.exists():
             raise FileNotFoundError(f"Cannot load dataset. File not found: {path}")
             
        with open(path, 'r') as f:
            self.data = json.load(f)

    def get_training_format(self):
        """
        Returns the data in a format suitable for Bradley-Terry model training.
        
        Returns:
            list of tuples: [(traj_A, traj_B, preference_tensor), ...]
            where traj_A = (states_tensor, actions_tensor)
        """
        training_data = []
        for record in self.data:
            # Convert A back to tensors
            states_a = torch.tensor(record["trajectory_a"]["states"], dtype=torch.float32)
            actions_a = torch.tensor(record["trajectory_a"]["actions"], dtype=torch.long)
            traj_a = (states_a, actions_a)
            
            # Convert B back to tensors
            states_b = torch.tensor(record["trajectory_b"]["states"], dtype=torch.float32)
            actions_b = torch.tensor(record["trajectory_b"]["actions"], dtype=torch.long)
            traj_b = (states_b, actions_b)
            
            # Formatting preference: the algorithms/BT.py expects preference=1.0 for A, 0.0 for B.
            # The JSON format uses 0=A better, 1=B better.
            # So, if record["preference"] == 0 (A better), BT wants 1.0.
            # If record["preference"] == 1 (B better), BT wants 0.0.
            perf_val = 1.0 if record["preference"] == 0 else 0.0
            
            training_data.append((traj_a, traj_b, perf_val))
            
        return training_data

    def get_statistics(self):
        """Returns statistics about the dataset."""
        if not self.data:
            return {"total_pairs": 0}
            
        total = len(self.data)
        a_preferred = sum(1 for r in self.data if r["preference"] == 0)
        b_preferred = sum(1 for r in self.data if r["preference"] == 1)
        
        annotators = set(r.get("annotator_id", "unknown") for r in self.data)
        
        return {
            "total_pairs": total,
            "prefer_A_count": a_preferred,
            "prefer_B_count": b_preferred,
            "prefer_A_pct": (a_preferred / total) * 100,
            "prefer_B_pct": (b_preferred / total) * 100,
            "unique_annotators": len(annotators)
        }
        
    def __len__(self):
        return len(self.data)