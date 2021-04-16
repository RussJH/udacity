from collections import deque, namedtuple
import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Buffer to store the experinces """

    def __init__(self, action_size, seed=0):
        """Initialize the buffer

        Args:
            action_size: number of actions that can be selected
            seed: initialization seed for random number generator
        """
        self.action_size = action_size
        self.batch_size  = 64                   # set batch size
        self.memory = deque(maxlen=int(1e5))    # memory length
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"] )
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add the experience to the buffer

        Args:
            state: Current state of the agent.
            action: The action of the agent.
            reward: The reward for the action taken.
            next_state: The next state of the agent.
            done: Has the episode ended.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def get_random_sample(self):
        """Get a random sample batch from the buffer 
        
        Returns:
            a batch of state, action, reward, next state and done tuples from the buffer 
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)


    def __len__(self):
        """Get size of buffer"""
        return len(self.memory)
    
