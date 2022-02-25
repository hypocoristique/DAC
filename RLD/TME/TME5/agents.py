import argparse
import sys
import matplotlib
#matplotlib.use("Qt5agg")
#matplotlib.use("TkAgg")
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from core import NN
import random
from memory import Memory
from icecream import ic
from torch.distributions import Categorical
from icecream import ic
from networks import ActorCritic

class Agent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0


    def act(self, obs):
        a=self.action_space.sample()
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass

    # apprentissage de l'agent. Dans cette version rien à faire
    def learn(self):
        # sampler les mini batch et faire la descente de gradient ici.
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        pass

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        if not self.test:

            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done):
        if self.test:
            return False
        self.nbEvents+=1
        if done:
            return True
        return self.nbEvents%self.opt.freqOptim == 0

class ActorCriticAgent(Agent):
    def __init__(self, env, opt, logger):
        super(ActorCriticAgent, self).__init__(env, opt)
        """
        Doing basic Policy Gradient requires to sample an episode to the end before
        computing score function grads. => high variance.
        In this version, we use a critic which updates action-value fn parameters
        And an actor which updates policy parameter in direction suggested by the critic.
            Critic is updated by either TD, Markov, TD-lambda.
            Actor is updated by policy gradient.
        TD error of value function is an unbiased estimator of advantage function.
        Therefore we will use an approximator of the value function instead of the advantage function.
        """
        self.explo = opt.explo
        self.decay = opt.decay
        self.discount = opt.gamma
        self.epochs_value = opt.epochs_value
        self.logger = logger
        # ActorCritic
        self.actor_critic = ActorCritic(inSize=self.featureExtractor.outSize, outSize=self.action_space.n)
        self.optimizer = torch.optim.Adam(params= self.actor_critic.parameters(), lr = opt.lr)

        # target
        self.actor_critic_target = ActorCritic(inSize=self.featureExtractor.outSize, outSize=self.action_space.n)
        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
        self.actor_critic_target.eval()

        # Buffer
        self.buffer = []

        self.criterion = torch.nn.SmoothL1Loss(reduction='none')

    def get_policy(self, obs):   
        """
        This function is based on recommandations of pytorch/distributions
        We need to sample an action from the policy, and count on pytorch autograd to take that into account.
        """    
        obs = torch.tensor(obs)
        # shape batch_size, n_actions
        a_probs, _ = self.actor_critic(obs)
        return Categorical(a_probs)

    def act(self, obs):
        obs = torch.tensor(obs)
        return self.get_policy(obs).sample().item()
    
    def _compute_returns(self):
        R = self.buffer[-1]["value"]
        self.buffer[-1]["target"] = R
        for step in reversed(range(len(self.buffer)-1)):
            R = self.buffer[step]["reward"] + self.discount * R #* self.buffer[step]["done"]
            self.buffer[step]["target"] = R

    def _train_actor_critic_network(self, obs_batch, a_batch, r_batch, next_obs_batch, done_batch, v_batch, target_batch):
        loss = self.criterion(v_batch, target_batch)
        logprob = Categorical(self.actor_critic(obs_batch)[0]).log_prob(a_batch)
        loss_policy = - (logprob * loss).sum()
        loss = loss.sum()
        self.logger.direct_write('Loss/critic', loss, self.episode_count)
        self.logger.direct_write('Loss/actor', loss_policy, self.episode_count)
        return loss + loss_policy
        

    def store(self,ob, action, new_ob, reward, done, it, episode_count):
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        self.episode_count = episode_count
        if not self.test:
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            ob = torch.tensor(ob)
            v = self.actor_critic(ob)[1].squeeze(-1)
            tr = {"ob": ob,
                  "action": action,
                  "reward": reward,
                  "new_ob": new_ob,
                  "done": done,
                  "value": v}
            self.buffer.append(tr)


    def learn(self):
        if self.test:
            return
        else:
            self._compute_returns()
            obs_batch, a_batch, r_batch, next_obs_batch, done_batch, v_batch, target_batch = \
                                        torch.vstack([x["ob"] for x in self.buffer]), \
                                        torch.tensor([x["action"] for x in self.buffer]).unsqueeze(-1), \
                                        torch.tensor([x["reward"] for x in self.buffer]).unsqueeze(-1), \
                                        torch.tensor([x["new_ob"] for x in self.buffer]), \
                                        torch.tensor([x["done"] for x in self.buffer], dtype=torch.int).unsqueeze(-1), \
                                        torch.vstack([x["value"] for x in self.buffer]), \
                                        torch.vstack([x["target"] for x in self.buffer])
                                        #torch.vstack([x["target"] for x in self.buffer])
            loss_critic = self._train_actor_critic_network(obs_batch, a_batch, r_batch,
                                                           next_obs_batch, done_batch, v_batch, target_batch)
            self.optimizer.zero_grad()
            loss_critic.backward()
            print(loss_critic.item(), len(self.buffer))
            self.optimizer.step()
            
            self.buffer = []

    def update_target(self):
        self.actor_critic_target.load_state_dict(self.actor_critic.state_dict())
    

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = ActorCriticAgent(env,config, logger)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j, i)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            
            if i % config["freqTarget"] == 0:
                agent.update_target()

            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()