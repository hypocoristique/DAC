import pickle
import torch
from core import NN, Agent


class BehavioralCloning(Agent):
    def __init__(self, env, opt, logger):
        super(BehavioralCloning, self).__init__(env, opt)
        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.lr = opt.lr
        self.batch_size = opt.batch_size
        self.logger = logger

        self.expert_states, self.expert_actions = self.load_expert_transition(opt.expert_data_path)

        self.model = NN(self.featureExtractor.outSize, self.action_space.n, layers=[64, 32], finalActivation=torch.tanh)
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def act(self, obs):
        obs = torch.tensor(obs)
        with torch.no_grad():
            action = self.model.forward(obs).reshape(-1)
        return action.argmax().int().item()

    def learn(self):
        running_loss = 0
        for i in range(self.expert_actions.shape[0]//self.batch_size):
            s = self.expert_states[self.batch_size*i:self.batch_size*(i+1), :]
            a = self.expert_actions[self.batch_size*i:self.batch_size*(i+1), :]

            output = self.model(s)
            loss = self.loss(output, a)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            running_loss += loss.item()

        return {'loss': running_loss / (self.expert_actions.shape[0]//self.batch_size)}

    def time_to_learn(self, done):
        if self.test:
            return False
        else:
            return True

    def load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions
