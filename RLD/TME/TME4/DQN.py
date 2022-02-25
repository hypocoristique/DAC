import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from torch import nn
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from memory import Memory
from datetime import datetime
import random
from icecream import ic

class DQN(object):
    def __init__(self, env, opt, memory=None):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents=0
        self.explo = opt.explo
        self.Qnetwork = NN(inSize=self.featureExtractor.outSize, outSize=env.action_space.n, layers=[opt.hiddenSize])
        self.Qtarget = NN(inSize=self.featureExtractor.outSize, outSize=env.action_space.n, layers=[opt.hiddenSize])
        # self.Qtarget.load_state_dict(self.Qnewtork.state_dict())
        self.criterion = nn.SmoothL1Loss()
        self.gamma = 0.999
        self.optim_network = torch.optim.Adam(self.Qnetwork.parameters())
        self.memory = memory
        self.batch_size = opt.batchSize

    def act(self, obs):
        if random.random() < self.explo:
            a = random.randint(0,self.action_space.n-1)
        else:
            a = torch.argmax(self.Qnetwork(torch.tensor(obs, dtype=torch.float))).item()
        return a

    # sauvegarde du modèle
    def save(self, outputDir):
        pass

    # chargement du modèle.
    def load(self, inputDir):
        pass

    # apprentissage de l'agent
    def learn(self):
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        else:
            if self.memory is not None:
                _, _, batch = self.memory.sample(self.batch_size)
                self.last_source = [tr[0] for tr in batch]
                self.last_action =  [tr[1] for tr in batch]
                self.last_reward = [tr[2] for tr in batch]
                self.last_dest = [tr[3] for tr in batch]
                self.done   = torch.tensor([x[4] for x in batch], dtype=torch.int)

            Q_target = self.Qtarget(torch.tensor(self.last_dest, dtype=torch.float))
            target = torch.tensor(self.last_reward, dtype=torch.float) + (self.gamma * torch.squeeze(Q_target.max(dim = -1).values, dim=-1))*(1-self.done)

            if self.memory == None:
                Q = self.Qnetwork(torch.tensor(self.last_source, dtype=torch.float))[0][self.last_action]
                loss = self.criterion(Q, target)
            else:
                Q = self.Qnetwork(torch.tensor(self.last_source, dtype=torch.float))
                index = torch.tensor(self.last_action, dtype=torch.int64).unsqueeze(-1).unsqueeze(-1)
                loss = self.criterion(torch.gather(Q, dim=-1, index=index), target.detach())

            # target = torch.tensor(self.last_reward, dtype=torch.float) + (self.gamma*torch.squeeze(torch.max(Q_target, dim=-1).values, dim=-1))*(1-self.done)
            # if self.memory is not None:
            #     # Q = self.Qnetwork(torch.tensor(self.last_source, dtype=torch.float))
            #     Q = []
            #     for action in range(self.batch_size):
            #         Q.append(self.Qnetwork(torch.tensor(self.last_source[action], dtype=torch.float))[0][self.last_action[action]]*(1-self.done[action]))
            #     Q = torch.tensor(Q, dtype=torch.float)
            # else:
            #     Q = self.Qnetwork(torch.tensor(self.last_source, dtype=torch.float))[0][self.last_action]
            # loss = self.criterion(Q, target)

            loss.backward()
            self.optim_network.step()
            self.optim_network.zero_grad()

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            self.last_source = ob
            self.last_action = action
            self.last_dest = new_ob
            self.last_reward = reward
            self.done = done
            # si on atteint la taille max d'episodes en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            self.last_done=done
            tr = (ob, action, reward, new_ob, done)
            if self.memory == None:
                self.lastTransition = tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)
            else:
                self.memory.store(tr)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

    def update_target(self):
        if self.memory is not None:
            # self.Qtarget.parameters() = self.Qnetwork.parameters()
            self.Qtarget.load_state_dict(self.Qnetwork.state_dict())