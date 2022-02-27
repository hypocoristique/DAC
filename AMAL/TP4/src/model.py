import torch
import torch.nn as nn
from icecream import ic
from train import TrainingPytorch
from torch.nn.functional import softmax

class RNN(TrainingPytorch):
    def __init__(self, input_size, hidden_size, output_size, criterion, opt='Adam', mode=None, model=None, ckpt_save_path=None, seq_length=None):
        super(RNN, self).__init__(criterion=criterion, opt=opt, mode=mode, ckpt_save_path=ckpt_save_path)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mode = mode
        self.seq_length = seq_length
        if self.mode == 'generation':
            self.linear_one_step = nn.Linear(2*self.hidden_size, self.hidden_size, bias=True, dtype=torch.float)
        else:
            self.linear_one_step = nn.Linear(self.input_size+self.hidden_size, self.hidden_size, bias=True, dtype=torch.float)
        self.linear_decode = nn.Linear(self.hidden_size,self.output_size, dtype=torch.float)
        self.activation = nn.Tanh()
        self.embedding  = nn.Embedding(num_embeddings=self.output_size, embedding_dim = self.hidden_size)

    def _one_step(self,x,h):
        h = torch.cat((x,h), dim=1)
        h = self.linear_one_step(h)
        h = self.activation(h)
        return h
    
    #one step : x est BxR^d et h BxH
    #forward : x est BxTxd et h est BxH. Doit renvoyer h1, h2, ..., hT

    def forward(self,x,h=None):
        if self.mode == 'generation':
            x = self.embedding(x)
        h0 = torch.zeros(x.shape[0], self.hidden_size)
        H = [h0]
        for t in range(x.shape[1]-1):
            x_t = x[:,t,:].view(x.shape[0], -1)
            h = self._one_step(x_t, H[-1])
            H.append(h)
        H = torch.stack(H, dim=0)
        return H

    def decode(self,h):
        #Pas besoin d'un softmax si on utilise comme loss la cross-entropy car elle le fait directement
        yhat = self.linear_decode(h)
        return yhat

    def predict(self, x, h=None):
        x = torch.unsqueeze(self.embedding(x), dim=0)
        h0 = torch.unsqueeze(torch.zeros(self.hidden_size), dim=0)
        H = [h0]
        for t in range(x.shape[1]-1):
            x_t = x[:,t,:].view(x.shape[0], -1)
            h = self._one_step(x_t, H[-1])
            H.append(h)
        H = torch.stack(H, dim=0)
        yhat = torch.multinomial(softmax(self.decode(H)[0,0,:]), num_samples=1).item()
        # yhat = torch.argmax(self.decode(H)[0,0,:]).item()
        return yhat