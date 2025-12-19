import torch
import torch.nn.functional as F

from policy import PolicyNetwork
from value import ValueNetwork


def main():
    print("=== Initializing networks ===")
    policy = PolicyNetwork()
    value_net = ValueNetwork()

    print("Policy params:",
          sum(p.numel() for p in policy.parameters()))
    print("Value params:",
          sum(p.numel() for p in value_net.parameters()))

    # Single test state
    state = torch.randn(40)

    print("\n=== Forward pass test ===")

    # Policy forward
    probs = policy(state)
    action, logprob = policy.sample_action(state)

    print("Policy probs:", probs)
    print("Sum probs:", probs.sum().item())
    print("Sampled action:", action)
    print("Log prob:", logprob.item())

    # Value forward
    value = value_net(state)
    print("Value estimate:", value.item())

    print("\n=== Backward pass (smoke test) ===")

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=1e-3
    )

    # Fake reward (always positive)
    reward = torch.tensor(1.0)

    advantage = reward - value.detach()

    policy_loss = -logprob * advantage
    value_loss = F.mse_loss(value, reward)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
    print("\n✓ Smoke test passed (forward + backward)")


if __name__ == "__main__":
    main()
