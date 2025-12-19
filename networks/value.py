import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    """
    A 3-layer feed-forward value network (critic).

    Architecture:
        Input:  40-dim state vector
        Hidden: Linear(40 → 64) + ReLU
        Hidden: Linear(64 → 32) + ReLU
        Output: Linear(32 → 1)

    Output:
        A single scalar V(s) representing the expected return from the given state.
    """
    def __init__(self):
        super().__init__()

        #### Define layers
        
        # hidden layers
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 32)
        
        # output value
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape [batch, 40] or [40]

        Returns:
            Tensor of shape [batch, 1] representing expected value of reward.
        """

        # validate input type
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"ValueNetwork input must be torch.Tensor, got {type(x)}")

        # validate input
        if x.shape[-1] != 40:
            raise ValueError(f"Input to ValueNetwork must have 40 features, got shape {tuple(x.shape)}")

        # check batch dimension exists
        is_single_input = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            is_single_input = True

        # forward pass with relu activations for hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # final layer is single scalar
        value = self.fc3(x)

        # check for non-finite output
        if not torch.isfinite(value).all():
            raise RuntimeError("ValueNetwork produced non-finite output")
        
        # if single input, squeeze batch dimension
        if is_single_input:
            return value.squeeze(0)

        # return value as a single dimension tensor
        return value
    
