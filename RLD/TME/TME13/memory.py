from unicodedata import name
import numpy as np
from typing import Dict
import torch.nn.functional as F
import torch
from icecream import ic
from copy import deepcopy


def get_transition(ob, action, reward, new_ob, done, it, episode, goal):
    ob = torch.tensor(ob, dtype = torch.float)
    action = torch.tensor(action, dtype=torch.int64)
    reward = torch.tensor(reward, dtype= torch.float)
    new_ob = torch.tensor(new_ob, dtype= torch.float)
    done = torch.tensor(done, dtype= torch.int)
    it = torch.tensor(it, dtype=torch.int)
    episode = torch.tensor(episode, dtype=torch.int)
    goal = torch.tensor(goal, dtype=torch.float)
    return dict(ob=ob, action=action,
            reward=reward, new_ob=new_ob, done=done, it=it, episode=episode, goal=goal)


class Memory():
    def __init__(self, mem_size: int, batch_size: int) -> None:
        self.buffer = []
        self.nentities = 0
        self.mem_size = mem_size
        self.batch_size = batch_size

    def store(self, transition):
        self.buffer.append(transition)
        self.nentities += 1
        if self.nentities > self.mem_size:
            del self.buffer[0]

    def create_hindsight_goal(self, max_episodes):
        supplementary_episode = []
        new_goal = self.buffer[-1]['new_ob']
        if (new_goal != self.buffer[-1]['goal']).all():
            tr = deepcopy(self.buffer[-1])
            tr['goal'] = new_goal
            tr['reward'] = tr['reward'] + 1.1
            tr['done'] += 1
            ic(new_goal)
            supplementary_episode.append(tr)
            for it in reversed(range(len(self.buffer) - max_episodes,len(self.buffer)-1)):
                tr = deepcopy(self.buffer[it])
                tr['goal'] = new_goal
                supplementary_episode.append(tr)
            supplementary_episode.reverse()
            self.buffer = self.buffer + supplementary_episode

    def sample(self):
        id_batch = np.random.randint(0, len(self.buffer), self.batch_size)
        batch = np.array(self.buffer)[id_batch]
        obs = torch.vstack([x["ob"] for x in batch])
        action = torch.vstack([x["action"] for x in batch])
        reward = torch.vstack([x["reward"] for x in batch])
        new_obs = torch.vstack([x["new_ob"] for x in batch])
        done = torch.vstack([x["done"] for x in batch])
        goal = torch.vstack([x["goal"] for x in batch])
        return obs, action, reward, new_obs, done, goal

    def clear(self):
        self.buffer = []
        self.nentities = 0

    