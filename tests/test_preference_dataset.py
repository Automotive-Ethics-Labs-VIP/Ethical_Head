import sys
from pathlib import Path
import os
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from data.preference_dataset import PreferenceDataset

def test_preference_dataset():
    print("\n=== Testing PreferenceDataset ===")
    
    # Use a temporary file
    test_file = "test_prefs.json"
    
    try:
        dataset = PreferenceDataset(filepath=test_file)
        
        # 1. Test adding data (with tensors that need converting to lists)
        traj_a = (torch.randn(5, 40), torch.randint(0, 5, (5,))) 
        traj_b = (torch.randn(5, 40), torch.randint(0, 5, (5,)))
        
        dataset.add_preference(traj_a, traj_b, preference=0, annotator_id="test_user") # prefer A
        dataset.add_preference(traj_b, traj_a, preference=1, annotator_id="test_user") # prefer B
        
        assert len(dataset) == 2
        print("✓ Adding data passed")
        
        # 2. Test saving and finding the file
        dataset.save()
        assert os.path.exists(test_file)
        print("✓ Saving JSON passed")
        
        # 3. Test loading 
        new_dataset = PreferenceDataset(filepath=test_file)
        assert len(new_dataset) == 2
        assert new_dataset.data[0]["preference"] == 0
        assert new_dataset.data[1]["preference"] == 1
        print("✓ Loading JSON passed")
        
        # 4. Test training format conversion
        training_data = new_dataset.get_training_format()
        assert len(training_data) == 2
        
        t_a, t_b, pref_val = training_data[0]
        assert isinstance(t_a[0], torch.Tensor)
        assert t_a[0].shape == (5, 40) # verify state tensor shape restored
        assert pref_val == 1.0 # 0=A better -> 1.0 logic
        
        _, _, pref_val_2 = training_data[1]
        assert pref_val_2 == 0.0 # 1=B better -> 0.0 logic
        print("✓ BT Training formatting passed")
        
        # 5. Test stats
        stats = new_dataset.get_statistics()
        assert stats["total_pairs"] == 2
        assert stats["prefer_A_count"] == 1 
        assert stats["unique_annotators"] == 1
        print("✓ Statistics passed")
        
        print("\n✓✓✓ ALL PREFERENCE DATASET TESTS PASSED ✓✓✓")
        
    finally:
         # cleanup
         if os.path.exists(test_file):
             os.remove(test_file)

if __name__ == "__main__":
    test_preference_dataset()
