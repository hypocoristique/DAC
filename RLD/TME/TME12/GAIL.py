import pickle
import torch
from core import NN, Agent, ActorCriticNetwork
from memory import Memory
from torch.distributions import Categorical
import torch.nn.functional as F
from core import AverageMeter

class GailAgent(Agent):
    def __init__(self, env, opt, logger):
        super(GailAgent, self).__init__(env, opt)
        self.featureExtractor = opt.featExtractor(env)
        self.nSteps = 0
        self.batch_size = opt.batch_size
        self.clip = opt.clip
        self.c1 = opt.c1
        self.c2 = opt.c2
        self.gamma = opt.gamma
        self.epochs = opt.epochs
        self.logger = logger
        self.memory = Memory(mem_size=opt.mem_size, gamma=opt.gamma, tau = opt.tau, batch_size=opt.batch_size)
        self.expert_states, self.expert_actions = self.load_expert_transition(opt.expert_data_path)

        self.discriminator = NN(self.featureExtractor.outSize + self.action_space.n, 1, layers=[64, 32], finalActivation=None)
        self.actor_critic = ActorCriticNetwork(inSize=self.featureExtractor.outSize, outSize=self.action_space.n)
        self.optimizer = torch.optim.Adam(params= self.actor_critic.parameters(), lr = opt.lr)
        self.discriminator_optim = torch.optim.Adam(params= self.discriminator.parameters(), lr = opt.lr)


    def _get_policy(self, obs):
        value, a_probs = self.actor_critic(obs)
        return value, Categorical(a_probs)

    def act(self, obs):
        obs = torch.tensor(obs)
        with torch.no_grad():
            state_value, probs = self._get_policy(obs)
        action = probs.sample()
        log_prob = probs.log_prob(action)

        tr = {"ob" : obs,
              "action": action,
              "log_prob": log_prob,
              "value": state_value}

        if not self.test:
            self.memory.store(tr)
        return action.item()

    def store(self,ob, action, new_ob, reward, done, it, episode_count):
        self.episode_count = episode_count
        if not self.test:
            edit_tr = {"reward": reward,
                       "new_ob": new_ob,
                       "done": done,
                       "step": it}
            self.memory.edit_last_transition(**edit_tr)

    def timeToLearn(self,done):
        """
            Collect set of trajectories Dk by running current policy for freqOptim steps
        """
        if self.test:
            return False
        self.nSteps+=1
        if self.nSteps%self.opt.freqOptim == 0:
            return True

    def learn(self):
        # Train Discriminator
        d_expert_meter = AverageMeter()
        d_data_meter = AverageMeter()
        actor_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        entropy_meter = AverageMeter()
        for epoch in range(self.epochs):
            for i, dic in zip(range(self.expert_actions.shape[0]//self.batch_size),
                                            self.memory.generate_batches(mode='discriminator')):
                obs, _ , actions, *_ = dic
                actions = F.one_hot(actions.view(-1), self.action_space.n)
                ob_a = torch.cat([obs, actions], dim=-1)
                noise_agent = torch.normal(0, 0.01, size=ob_a.shape)
                d_data = torch.sigmoid(self.discriminator(ob_a) + noise_agent)
                data_loss = F.binary_cross_entropy_with_logits(d_data, torch.zeros_like(d_data))
                s = self.expert_states[self.batch_size*i:self.batch_size*(i+1), :]
                a = self.expert_actions[self.batch_size*i:self.batch_size*(i+1), :]
                ob_a = torch.cat([s,a], dim=-1)
                noise_expert = torch.normal(0, 0.01, size=ob_a.shape)
                d_expert = torch.sigmoid(self.discriminator(ob_a) + noise_expert)
                expert_loss = F.binary_cross_entropy_with_logits(d_expert, torch.ones_like(d_expert))
                discriminator_loss = data_loss + expert_loss
                self.discriminator_optim.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optim.step()

                d_expert_meter.update(d_expert.mean(), self.batch_size)
                d_data_meter.update(d_data.mean(), self.batch_size)
                d_loss_meter.update(discriminator_loss, self.batch_size)
            self.logger.direct_write('d_data', d_data_meter.avg, epoch)
            self.logger.direct_write('d_expert', d_expert_meter.avg, epoch)
            self.logger.direct_write('d_loss', d_loss_meter.avg, epoch)
            d_data_meter.reset()
            d_expert_meter.reset()
            d_loss_meter.reset()

        # Train actor critic
        self.memory.compute_cumR(self.discriminator)
        for epoch in range(self.epochs):
            for obs, old_probs, actions, advantage, value in self.memory.generate_batches(mode='actorcritic'):
                # need obs, old_probs, actions, advantage,
                new_critic_value, distrib = self._get_policy(obs)
                entropy = distrib.entropy().mean()
                new_probs = distrib.log_prob(actions)

                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.clip, 1+self.clip) * advantage
                actor_loss = - torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Fit value function un cumulative return
                critic_loss =  F.smooth_l1_loss(new_critic_value, advantage)

                # Total loss
                total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                entropy_meter.update(entropy)
                actor_loss_meter.update(actor_loss)
            self.logger.direct_write('entropy', entropy_meter.avg, epoch)
            self.logger.direct_write('actor', actor_loss_meter.avg, epoch)
            entropy_meter.reset()
            actor_loss_meter.reset()
            
        self.memory.clear()
        return {'loss': total_loss}

    def load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions