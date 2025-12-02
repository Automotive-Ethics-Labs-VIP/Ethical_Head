import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    """
    A 3-layer feedforward policy network.

    Architecture:
        Input: 40-dim state
        Hidden1: Linear(40 → 64) + ReLU
        Hidden2: Linear(64 → 32) + ReLU
        Output: Linear(32 → 5) + Softmax over action dimension

    Output:
        A probability distribution over 5 actions (shape: [batch, 5]).
    """

    def __init__(self):
        super().__init__()

        #### Define layers
        
        # hidden layers
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # output logits for actions
        self.fc3 = nn.Linear(32, 5)

        # softmax for cleaner forward pass
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape [batch, 40] or [40]

        Returns:
            Tensor of shape [batch, 5] representing action probabilities.
        """
        # validate input
        if x.shape[-1] != 40:
            raise ValueError(f"Input to PolicyNetwork must have 40 features, got shape {tuple(x.shape)}")

        # check batch dimension exists
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        # softamx -> probabilities
        probs = self.softmax(logits)

        # validate probalities
        sums = probs.sum(dim=-1, keepdim=True)
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-6):
            probs = probs / sums

        return probs
