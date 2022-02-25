import numpy as np
from typing import Dict
from typing import Optional
import torch
from icecream import ic

class Memory():
    def __init__(self, gamma, n_states) -> None:
        self.buffer = []
        self.nentries = 0
        self.gamma = gamma
        self.n_states = n_states
    

    def store(self, transition):
        self.buffer.append(transition)
        self.nentries += 1

    def edit_last_transition(self, **kwargs):
        self.buffer[-1]["reward"] = kwargs["reward"]
        self.buffer[-1]["new_ob"] = kwargs["new_ob"]
        self.buffer[-1]["done"] = kwargs["done"]
        self.buffer[-1]["step"] = kwargs["step"]

    def compute_returns(self):
        R = 0
        for step in reversed(range(len(self.buffer))):
            R = self.buffer[step]["reward"] + self.gamma * R * (1 - self.buffer[step]["done"])
            self.buffer[step]["returns"] = R
    
    def compute_gae(self, tau, returns: bool = False):
        """
            Compute generalized advantage estimate which is basically an exponential average
            of the TD(n) advantage function
                if tau set to 1:  It is a regular TD(n) advantage function
                if tau set to 0: It is a TD(0) advantage function
        """
        if returns:
            key = 'returns'
        else:
            key = 'advantage'
        self.buffer[-1][key] = torch.tensor([0])
        for step in reversed(range(len(self.buffer) - 1)):
            delta = self.buffer[step]['reward'] + self.gamma * (self.buffer[step+1]['value'] * \
                (1 - int(self.buffer[step]['done'])) - self.buffer[step]['value'])
            self.buffer[step][key] = delta + self.gamma * tau * \
                 self.buffer[step + 1][key] *  (1 - int(self.buffer[step]['done']))
                
    def generate_batches(self):
        obs = torch.vstack([x["ob"] for x in self.buffer])
        old_probs = torch.vstack([x["log_prob"] for x in self.buffer])
        actions = torch.vstack([x["action"] for x in self.buffer])
        advantage = torch.vstack([x["advantage"] for x in self.buffer])
        returns = torch.vstack([torch.tensor(x["returns"]) for x in self.buffer])
        value = torch.vstack([x["value"] for x in self.buffer])
        return obs, old_probs, actions, advantage, returns, value

    def clear(self):
        self.buffer = []
        self.nentries = 0

    def __len__(self):
        return len(self.buffer)

    
    