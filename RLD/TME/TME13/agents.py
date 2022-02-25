from core import NN
import random
import torch
from torch import nn
import numpy as np
from icecream import ic
from typing import Optional
from memory import get_transition, Memory

class DQN(object):
    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test = False
        self.nbEvents=0
        self.ep_length = 0
        self.explo = opt.explo
        self.Qnetwork = NN(inSize=self.featureExtractor.outSize*2, outSize=env.action_space.n, layers=[32,64])
        self.Qtarget = NN(inSize=self.featureExtractor.outSize*2, outSize=env.action_space.n, layers=[32,64])
        self.Qtarget.load_state_dict(self.Qnetwork.state_dict())
        self.criterion = nn.SmoothL1Loss()
        self.gamma = 0.999
        self.optim_network = torch.optim.Adam(self.Qnetwork.parameters())
        self.memory = Memory(mem_size = opt.mem_size, batch_size=opt.batch_size)
        self.batch_size = opt.batch_size
        self.her = opt.her
        self.epochs = opt.epochs

    def act(self, obs, goal):
        obs = np.concatenate([obs, goal], axis=1)
        if random.random() < self.explo:
            a = random.randint(0,self.action_space.n-1)
        else:
            with torch.no_grad():
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
        if self.test or self.batch_size >= self.memory.nentities:
            pass
        else:
            for _ in range(self.epochs):
                obs, action, reward, new_obs, done, goal = self.memory.sample()
                obs_goal = torch.cat([obs, goal], dim=-1)
                next_obs_goal = torch.cat([new_obs, goal], dim=-1)

                with torch.no_grad():
                    Q_target = self.Qtarget(next_obs_goal)
                target = reward + (self.gamma * Q_target.max(dim = -1).values.unsqueeze(-1))*(1-done)
                Q = self.Qnetwork(obs_goal)
                loss = self.criterion(torch.gather(Q, dim=-1, index=action), target.detach())

                loss.backward()
                self.optim_network.step()
                self.optim_network.zero_grad()

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self, ob, action, new_ob, reward, done, it, episode, goal):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:
            # si on atteint la taille max d'episodes en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            fake_done = done
            self.ep_length+=1
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = get_transition(ob, action, reward, new_ob, done, it, episode, goal)
            self.memory.store(tr)
            if self.her:
                if self.ep_length % self.opt.her_frequency==0:
                    self.memory.create_hindsight_goal(self.ep_length)
                elif fake_done:
                    self.memory.create_hindsight_goal(self.ep_length)
                    self.ep_length = 0

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        if self.nbEvents % self.opt.freqTarget == 0:
            self.update_target()
        return self.nbEvents%self.opt.freqOptim == 0

    def update_target(self):
        self.Qtarget.load_state_dict(self.Qnetwork.state_dict())