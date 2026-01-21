import torch
from algorithms.bradley_terry import bradley_terry_loss

def train_reward_model(
    reward_model,
    preference_dataset,
    optimizer,
    epochs=5,
    device='cpu',
    log_interval=10
):
    """
    Train reward model on human preference data.
    
    Args:
        reward_model: RewardModel instance
        preference_dataset: Iterable yielding (traj_A, traj_B, preference)
        optimizer: PyTorch optimizer
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        log_interval: Log every N batches
    
    Returns:
        List of average losses per epoch
    """
    reward_model.to(device)
    reward_model.train()
    
    epoch_losses = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (traj_A, traj_B, pref) in enumerate(preference_dataset):
            # Move data to device
            states_A, actions_A = traj_A
            states_B, actions_B = traj_B
            
            states_A = states_A.to(device)
            actions_A = actions_A.to(device)
            states_B = states_B.to(device)
            actions_B = actions_B.to(device)
            
            traj_A = (states_A, actions_A)
            traj_B = (states_B, actions_B)
            
            # Compute loss
            optimizer.zero_grad()
            loss = bradley_terry_loss(reward_model, traj_A, traj_B, pref)
            loss.backward()
            
            # Optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_loss)
        print(f"[Reward Model] Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
    
    return epoch_losses