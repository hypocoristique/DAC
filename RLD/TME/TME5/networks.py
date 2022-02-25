import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, inSize, outSize):
        super(ActorCritic, self).__init__()
        self .fc1 = nn.Linear(inSize, 128)
        self.actor = nn.Linear(128,outSize)
        self.critic = nn.Linear(128,1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x = x.type(torch.float)
        x = self.relu(self.fc1(x))
        action_prob = self.softmax(self.actor(x))
        state_values = self.critic(x)
        return action_prob, state_values

# class ActorCriticAgent(Agent):
#     def __init__(self, env, opt, logger):
#         super(ActorCriticAgent, self).__init__(env, opt)
#         """
#         Doing basic Policy Gradient requires to sample an episode to the end before
#         computing score function grads. => high variance.
#         In this version, we use a critic which updates action-value fn parameters
#         And an actor which updates policy parameter in direction suggested by the critic.
#             Critic is updated by either TD, Markov, TD-lambda.
#             Actor is updated by policy gradient.
#         TD error of value function is an unbiased estimator of advantage function.
#         Therefore we will use an approximator of the value function instead of the advantage function.
#         """
#         self.explo = opt.explo
#         self.decay = opt.decay
#         self.discount = opt.gamma
#         self.epochs_value = opt.epochs_value
#         self.logger = logger
#         # Critic
#         self.value_target_network = torch.nn.Sequential(
#             torch.nn.Linear(self.featureExtractor.outSize, 1)
#         )  
#         self.value_network = torch.nn.Sequential(
#             torch.nn.Linear(self.featureExtractor.outSize, 1)
#         )   
#         self.value_target_network.load_state_dict(self.value_network.state_dict())
#         self.value_target_network.eval()
#         self.optim_value = torch.optim.Adam(params= self.value_network.parameters(), lr = opt.lr_value)

#         # Actor
#         self.policy_network = NN(inSize =self.featureExtractor.outSize, outSize=self.action_space.n,
#                      layers=[32, 32], finalActivation=torch.nn.Softmax(dim=-1))
#         self.optim_policy = torch.optim.Adam(params= self.policy_network.parameters(), lr = opt.lr_policy)

#         # Buffer
#         self.buffer = []

#         self.batch_size = opt.batch_size
#         self.target_network = opt.target_network

#         self.criterion_policy = torch.nn.SmoothL1Loss()
#         self.criterion_v = torch.nn.SmoothL1Loss()

#     def get_policy(self, obs):   
#         """
#         This function is based on recommandations of pytorch/distributions
#         We need to sample an action from the policy, and count on pytorch autograd to take that into account.
#         """    
#         obs = torch.tensor(obs)
#         with torch.no_grad():
#             # shape batch_size, n_actions
#             a_probs = self.policy_network(obs)
#         return Categorical(a_probs)

#     def act(self, obs):
#         obs = torch.tensor(obs)
#         return self.get_policy(obs).sample().item()

#     def _train_value_network(self, obs_batch, r_batch, next_obs_batch, done_batch):
#         v_hat = self.value_network(obs_batch).squeeze(-1)
#         v_tilde = self.value_target_network(next_obs_batch).squeeze(-1)
#         target = r_batch + self.discount * v_tilde * (1 - done_batch)
#         loss = self.criterion_v(v_hat, target)
#         self.logger.direct_write('Loss/value', loss, self.episode_count)
#         self.optim_value.zero_grad()
#         loss.backward()
#         self.optim_value.step()
    
#     def _train_policy_network(self, obs_batch, a_batch,r_batch, next_obs_batch, done_batch):
#         advantage_target = r_batch + self.discount * self.value_network(next_obs_batch).squeeze(-1) \
#              * (1 - done_batch) - self.value_network(obs_batch).squeeze(-1)
#         logprob = self.get_policy(obs_batch).log_prob(a_batch)
#         loss_policy = (- logprob * advantage_target).mean()
#         self.optim_value.zero_grad()
#         loss_policy.backward()
#         self.optim_policy.step()
        

#     def store(self,ob, action, new_ob, reward, done, it, episode_count):
#         # Si l'agent est en mode de test, on n'enregistre pas la transition
#         self.episode_count = episode_count
#         if not self.test:
#             # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
#             if it == self.opt.maxLengthTrain:
#                 print("undone")
#                 done=False
#             tr = (ob, action, reward, new_ob, done)
#             self.buffer.append(tr)

#     def learn(self):
#         # sampler les mini batch et faire la descente de gradient ici.
#         # Si l'agent est en mode de test, on n'entraîne pas
#         # decay at each episode
#         # self.explo *= self.decay
#         if self.test:
#             return
#         else:
#             obs_batch, a_batch, r_batch, next_obs_batch, done_batch = \
#                                         torch.tensor([x[0] for x in self.buffer]), \
#                                         torch.tensor([x[1] for x in self.buffer]).unsqueeze(-1), \
#                                         torch.tensor([x[2] for x in self.buffer]).unsqueeze(-1), \
#                                         torch.tensor([x[3] for x in self.buffer]), \
#                                         torch.tensor([x[4] for x in self.buffer], dtype=torch.int).unsqueeze(-1)   
#             self._train_value_network(obs_batch, r_batch, next_obs_batch, done_batch)
#             self._train_policy_network(obs_batch, a_batch, r_batch, next_obs_batch, done_batch)
#             self.buffer = []

#     def update_target(self):
#         self.value_target_network.load_state_dict(self.value_network.state_dict())