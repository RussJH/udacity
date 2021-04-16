import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """ QNetwork model"""

    def __init__(self, state_size, action_size, seed=0):
        """Constructor for QNetwork model to initialize states, actions and random seed
        Args:
            state_size:  number of states
            action_size: number of actions
            seed: rng seed value
        """

        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)  # First Layer
        self.fc2 = nn.Linear(64, 64)          # Second Layer
        self.fc3 = nn.Linear(64, action_size) # Third Layer

    def forward(self, state):
        """Network of state to action values
        Args:
            state: state to map to an action
        Returns:
            mapped state to action values
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)