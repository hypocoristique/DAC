import numpy as np
from typing import Dict
import torch.nn.functional as F
import torch
from core import AverageMeter
from icecream import ic

class Memory():
    def __init__(self, mem_size: int, gamma, tau, batch_size: int) -> None:
        self.buffer = []
        self.nentries = 0
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mean = AverageMeter()

    def store(self, transition):
        self.buffer.append(transition)
        self.nentries += 1

    def edit_last_transition(self, **kwargs):
        self.buffer[-1]["reward"] = kwargs["reward"]
        self.buffer[-1]["new_ob"] = kwargs["new_ob"]
        self.buffer[-1]["done"] = kwargs["done"]
        self.buffer[-1]["step"] = kwargs["step"]

    def compute_cumR(self, discriminator):
        """
            Compute generalized advantage estimate which is basically an exponential average
            of the TD(n) advantage function
                if tau set to 1:  It is a regular TD(n) advantage function
                if tau set to 0: It is a TD(0) advantage function
        """
        for it in reversed(range(len(self.buffer))):
            with torch.no_grad():
                a = F.one_hot(self.buffer[it]['action'].view(-1), 4)
                reward_it = torch.clamp(F.logsigmoid(discriminator(torch.cat((self.buffer[it]['ob'],a), dim=-1))), min=-100, max=0)
                self.buffer[it]['reward'] = reward_it
                self.mean.update(reward_it)
                if self.buffer[it]['done']:
                    self.buffer[it]['advantage'] = reward_it
                    self.mean.reset()
                else:
                    self.buffer[it]['advantage'] = self.mean.avg
            

    def generate_batches(self, mode: str = 'discriminator'):
        obs = torch.vstack([x["ob"] for x in self.buffer])
        old_probs = torch.vstack([x["log_prob"] for x in self.buffer])
        actions = torch.vstack([x["action"] for x in self.buffer])
        value = torch.vstack([x["value"] for x in self.buffer])
        if mode == 'discriminator':
            advantage = torch.vstack([x["value"] for x in self.buffer])
        else:
            advantage = torch.vstack([x["advantage"] for x in self.buffer]).detach()
            
        for _ in range(len(self.buffer) // self.batch_size):
            id_batch = np.random.randint(0, len(self.buffer), self.batch_size)
            yield obs[id_batch,:], old_probs[id_batch,:], actions[id_batch,:], advantage[id_batch,:], value[id_batch,:]

    def clear(self):
        self.buffer = []
        self.nentries = 0

    