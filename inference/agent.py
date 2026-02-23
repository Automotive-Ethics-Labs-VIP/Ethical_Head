import torch
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
import logging

from networks.policy import PolicyNetwork
from networks.value import ValueNetwork
from networks.reward import RewardModel


class EthicalAgent:
    """
    Production inference agent for ethical decision-making in autonomous vehicles.
    
    This is the main interface between the RL model and the CARLA simulation.
    
    Interface contract with CARLA team:
        Input:  40-dim state vector (numpy array, list, or tensor)
        Output: Action index (0-4)
    
    Action Space:
        0: maintain_course
        1: brake_hard
        2: swerve_left
        3: swerve_right
        4: accelerate
    
    State Vector Format (40 dimensions):
        [velocity_ego, num_passengers, lane_position, velocity_delta,
         num_ped_if_straight, num_ped_if_left, num_ped_if_right,
         obstacle_features[33]]
    """
    
    ACTION_NAMES = {
        0: 'maintain_course',
        1: 'brake_hard',
        2: 'swerve_left',
        3: 'swerve_right',
        4: 'accelerate'
    }
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = 'cpu',
        load_value_network: bool = False,
        load_reward_model: bool = False
    ):
        """
        Initialize the ethical agent with trained models.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (.pt file)
            device: 'cpu' or 'cuda' for inference
            load_value_network: Whether to load value network (for analysis)
            load_reward_model: Whether to load reward model (for analysis)
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize policy network (required)
        self.policy = PolicyNetwork().to(self.device)
        
        # Optional: load value and reward networks for analysis
        self.value_net = None
        self.reward_model = None
        
        if load_value_network:
            self.value_net = ValueNetwork().to(self.device)
            
        if load_reward_model:
            self.reward_model = RewardModel().to(self.device)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set to evaluation mode
        self.policy.eval()
        if self.value_net is not None:
            self.value_net.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        
        self.logger.info(f"EthicalAgent initialized on {self.device}")
        self.logger.info(f"Policy network loaded from {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load model weights from checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load policy (required)
            if 'policy_state_dict' in checkpoint:
                self.policy.load_state_dict(checkpoint['policy_state_dict'])
            else:
                raise KeyError("Checkpoint missing 'policy_state_dict'")
            
            # Load optional networks
            if self.value_net is not None and 'value_state_dict' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_state_dict'])
                self.logger.info("Value network loaded")
            
            if self.reward_model is not None and 'reward_state_dict' in checkpoint:
                self.reward_model.load_state_dict(checkpoint['reward_state_dict'])
                self.logger.info("Reward model loaded")
            
            # Log training metadata if available
            if 'epoch' in checkpoint:
                self.logger.info(f"Model trained for {checkpoint['epoch']} epochs")
            if 'metrics' in checkpoint:
                self.logger.info(f"Training metrics: {checkpoint['metrics']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    def _validate_state(self, state_vector) -> torch.Tensor:
        """
        Validate and convert state vector to tensor.
        
        Args:
            state_vector: Input state (list, numpy array, or tensor)
        
        Returns:
            Validated tensor of shape [40] or [batch, 40]
        
        Raises:
            ValueError: If state has incorrect shape or invalid values
        """
        # Convert to tensor
        if isinstance(state_vector, torch.Tensor):
            state = state_vector.to(self.device)
        elif isinstance(state_vector, np.ndarray):
            state = torch.from_numpy(state_vector).float().to(self.device)
        elif isinstance(state_vector, list):
            state = torch.FloatTensor(state_vector).to(self.device)
        else:
            raise TypeError(
                f"State must be tensor, numpy array, or list. Got {type(state_vector)}"
            )
        
        # Validate shape
        if state.dim() == 1:
            if state.shape[0] != 40:
                raise ValueError(f"Expected 40-dim state, got {state.shape[0]}")
        elif state.dim() == 2:
            if state.shape[1] != 40:
                raise ValueError(f"Expected shape [batch, 40], got {state.shape}")
        else:
            raise ValueError(f"State must be 1D or 2D, got shape {state.shape}")
        
        # Check for invalid values
        if not torch.isfinite(state).all():
            raise ValueError("State contains NaN or Inf values")
        
        return state
    
    @torch.no_grad()
    def get_action(
        self,
        state_vector,
        deterministic: bool = True
    ) -> int:
        """
        Get action from state vector.
        
        This is the main method the CARLA team will call.
        
        Args:
            state_vector: 40-dim state vector (list, numpy array, or tensor)
            deterministic: If True, return argmax action (recommended for deployment).
                          If False, sample from distribution (for exploration).
        
        Returns:
            action: Integer in {0, 1, 2, 3, 4}
        
        Example:
            >>> agent = EthicalAgent('checkpoints/best_model.pt')
            >>> state = [0.5, 2, 0.0, ...] # 40 values from CARLA
            >>> action = agent.get_action(state)
            >>> print(f"Action: {agent.ACTION_NAMES[action]}")
        """
        state = self._validate_state(state_vector)
        
        # Handle both single state [40] and batch [N, 40]
        is_single = state.dim() == 1
        if is_single:
            state = state.unsqueeze(0)  # Add batch dimension
        
        if deterministic:
            # Greedy action selection (argmax)
            probs = self.policy(state)
            action = probs.argmax(dim=-1)
        else:
            # Stochastic action selection (sample from distribution)
            if is_single:
                action, _ = self.policy.sample_action(state.squeeze(0))
                action = torch.tensor([action]) if isinstance(action, int) else action.unsqueeze(0)
            else:
                # For batch, sample each individually
                actions = []
                for i in range(state.shape[0]):
                    a, _ = self.policy.sample_action(state[i])
                    actions.append(a if isinstance(a, int) else a.item())
                action = torch.tensor(actions)
        
        # Convert to Python int/list
        if is_single:
            return int(action[0].item())
        else:
            return [int(a.item()) for a in action]
    
    @torch.no_grad()
    def get_action_probabilities(self, state_vector) -> Dict[str, float]:
        """
        Get full probability distribution over actions.
        
        Useful for debugging, visualization, or explanation systems.
        
        Args:
            state_vector: 40-dim state vector
        
        Returns:
            Dictionary mapping action names to probabilities
        
        Example:
            >>> probs = agent.get_action_probabilities(state)
            >>> print(probs)
            {'maintain_course': 0.15, 'brake_hard': 0.65, ...}
        """
        state = self._validate_state(state_vector)
        probs = self.policy(state).squeeze().cpu().numpy()
        
        return {
            self.ACTION_NAMES[i]: float(probs[i])
            for i in range(5)
        }
    
    @torch.no_grad()
    def get_action_with_confidence(
        self,
        state_vector,
        deterministic: bool = True
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Get action along with confidence score and full distribution.
        
        Args:
            state_vector: 40-dim state vector
            deterministic: Use greedy or stochastic action selection
        
        Returns:
            Tuple of (action, confidence, probability_dict)
            - action: Selected action index
            - confidence: Probability of selected action
            - probability_dict: Full distribution over all actions
        
        Example:
            >>> action, confidence, probs = agent.get_action_with_confidence(state)
            >>> if confidence < 0.5:
            >>>     print("Warning: Low confidence decision")
        """
        state = self._validate_state(state_vector)
        prob_dict = self.get_action_probabilities(state)
        action = self.get_action(state, deterministic=deterministic)
        confidence = prob_dict[self.ACTION_NAMES[action]]
        
        return action, confidence, prob_dict
    
    @torch.no_grad()
    def get_value_estimate(self, state_vector) -> Optional[float]:
        """
        Get value network's estimate of expected future return.
        
        Only available if value network was loaded.
        
        Args:
            state_vector: 40-dim state vector
        
        Returns:
            Expected future reward (float) or None if value net not loaded
        """
        if self.value_net is None:
            self.logger.warning("Value network not loaded")
            return None
        
        state = self._validate_state(state_vector)
        value = self.value_net(state).squeeze()
        
        return float(value.cpu().item())
    
    @torch.no_grad()
    def get_ethical_reward(self, state_vector, action: int) -> Optional[float]:
        """
        Get learned ethical reward for a (state, action) pair.
        
        Only available if reward model was loaded.
        
        Args:
            state_vector: 40-dim state vector
            action: Action index (0-4)
        
        Returns:
            Ethical reward score (float) or None if reward model not loaded
        """
        if self.reward_model is None:
            self.logger.warning("Reward model not loaded")
            return None
        
        state = self._validate_state(state_vector)
        
        # Handle single state
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action_tensor = torch.tensor([action], device=self.device)
        else:
            action_tensor = torch.tensor([action] * state.shape[0], device=self.device)
        
        reward = self.reward_model(state, action_tensor).squeeze()
        
        return float(reward.cpu().item())
    
    @torch.no_grad()
    def analyze_decision(self, state_vector) -> Dict:
        """
        Comprehensive analysis of decision-making for a given state.
        
        Returns all available information: action, probabilities, value, rewards.
        Useful for debugging and understanding model behavior.
        
        Args:
            state_vector: 40-dim state vector
        
        Returns:
            Dictionary containing full decision analysis
        """
        state = self._validate_state(state_vector)
        
        # Get action and probabilities
        action, confidence, probs = self.get_action_with_confidence(state)
        
        analysis = {
            'action': action,
            'action_name': self.ACTION_NAMES[action],
            'confidence': confidence,
            'action_probabilities': probs,
        }
        
        # Add value estimate if available
        value = self.get_value_estimate(state)
        if value is not None:
            analysis['value_estimate'] = value
        
        # Add ethical rewards for all actions if available
        if self.reward_model is not None:
            analysis['ethical_rewards'] = {
                self.ACTION_NAMES[a]: self.get_ethical_reward(state, a)
                for a in range(5)
            }
        
        return analysis
    
    def batch_predict(
        self,
        state_batch: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Batch prediction for multiple states (more efficient than looping).
        
        Args:
            state_batch: Array of shape [batch_size, 40]
            deterministic: Use greedy action selection
        
        Returns:
            Array of actions [batch_size]
        """
        states = self._validate_state(state_batch)
        
        if deterministic:
            probs = self.policy(states)
            actions = probs.argmax(dim=-1)
        else:
            actions = []
            for i in range(states.shape[0]):
                action, _ = self.policy.sample_action(states[i])
                actions.append(action)
            actions = torch.tensor(actions, device=self.device)
        
        return actions.cpu().numpy()
    
    def save_to_onnx(self, output_path: Union[str, Path]):
        """
        Export policy network to ONNX format for deployment.
        
        Useful for integration with other systems or embedded deployment.
        
        Args:
            output_path: Where to save the .onnx file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(1, 40, device=self.device)
        
        # Export
        torch.onnx.export(
            self.policy,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['state'],
            output_names=['action_probs'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action_probs': {0: 'batch_size'}
            }
        )
        
        self.logger.info(f"Model exported to ONNX: {output_path}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            'device': str(self.device),
            'policy_parameters': sum(p.numel() for p in self.policy.parameters()),
            'value_network_loaded': self.value_net is not None,
            'reward_model_loaded': self.reward_model is not None,
        }
        
        if self.value_net is not None:
            info['value_parameters'] = sum(p.numel() for p in self.value_net.parameters())
        
        if self.reward_model is not None:
            info['reward_parameters'] = sum(p.numel() for p in self.reward_model.parameters())
        
        return info


# Convenience function for quick inference
def quick_inference(checkpoint_path: str, state_vector, device: str = 'cpu') -> int:
    """
    Quick one-off inference without creating agent instance.
    
    Args:
        checkpoint_path: Path to model checkpoint
        state_vector: 40-dim state
        device: 'cpu' or 'cuda'
    
    Returns:
        Action index (0-4)
    """
    agent = EthicalAgent(checkpoint_path, device=device)
    return agent.get_action(state_vector)