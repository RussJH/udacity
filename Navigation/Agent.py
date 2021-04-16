
from Model import QNetwork
from ReplayBuffer import ReplayBuffer

import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Deep Q-learning agent """

    def __init__(self, state_size, action_size, seed=0):
        """
        Initialize Agent
        TODO: Add init code here
        """
        self.state_size = state_size
        self.action_size = action_size

        # QNetwork model
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4) # Learning rate TODO: Comment this more

        # Memory buffer
        self.memory = ReplayBuffer(action_size, seed=seed)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save the experience in the buffer
        Args:
            state: current state
            action: selected action
            reward: reward given
            next_state: next state to advance to
            done: has the memory completed
        """
        # save experience 
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn after every 4 time steps
        self.t_step = (self.t_step + 1 ) % 4
        if self.t_step == 0:
            if(len(self.memory) > self.memory.batch_size):
                experiences = self.memory.get_random_sample()
                if(not done):
                    self.learn(experiences, 0.99)

    def learn(self, experiences, gamma):
        """
        TODO: implement learning
        """
        states, actions, rewards, next_states, dones = experiences

         # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update(self.qnetwork_local, self.qnetwork_target, 1e-3)

    def update(self, local_model, target_model, tau):
        """Update the model params
        target = t*local+(1-t)*target

        Args:
            local_model: pytorch model source
            target_model: pytorch model destination
            tau: interpolation param
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def get_action(self, state, eps=0.):
        """Gets the action from the given state
        Args:
            state: current state
            eps: epsilon value for a epsilon-greedy policy
        Returns:
            action selection based on the epsilon-greedy policy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon greedy selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))