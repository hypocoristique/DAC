import torch
from torch import nn
import torch.nn.functional as F
from icecream import ic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear_one_step = nn.Linear(input_size+hidden_size, hidden_size, bias=True, dtype=torch.float)
        self.linear_decode = nn.Linear(hidden_size, output_size, dtype=torch.float)
        self.activation = nn.Tanh()

    def one_step(self, x, h):
        h = h.to(device)
        h = torch.cat((x,h), dim=1)
        h = self.linear_one_step(h)
        h = self.activation(h)
        return h

    def forward(self, x):
        #x est de shape length*batch_size*embedding_dim
        h0 = torch.zeros(x.shape[1], self.hidden_size).to(device)
        H = [h0]
        for l in range(x.shape[0]):
            H.append(self.one_step(x[l], H[l]))
        # for t in range(x.shape[1]-1):
        #     x_t = x[:,t,:].view(x.shape[0], -1)
        #     h = self.one_step(x_t, H[-1])
        #     H.append(h)
        H = torch.stack(H, dim=0)
        return H

    def decode(self, h):
        yhat = self.linear_decode(h)
        return yhat

class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_i = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_c = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_o = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_decode = nn.Linear(hidden_size, output_size, dtype=torch.float)

    def one_step(self, x, h, C):
        h = h.to(device)
        C = C.to(device)
        input = torch.cat((h,x), dim=1)
        f = F.sigmoid(self.linear_f(input))
        i = F.sigmoid(self.linear_f(input))
        C = f * C + i * F.tanh(self.linear_c(input))
        o = F.sigmoid(self.linear_o(input))
        h = o * F.tanh(C)
        return h, C, f, i, o

    def forward(self, x):
        length = x.shape[0]
        batch_size = x.shape[1]
        h0 = torch.zeros(batch_size, self.hidden_size).to(device)
        C0 = torch.zeros(batch_size, self.hidden_size).to(device)
        H, C = [h0], [C0]
        F, I, O = [], [], []
        for l in range(length):
            h, c, f, i, o = self.one_step(x[l], H[l], C[l])
            H.append(h), C.append(c), F.append(f), I.append(i), O.append(o)
        H = torch.stack(H, dim=0)
        C = torch.stack(C, dim=0)
        F = torch.stack(F, dim=0)
        I = torch.stack(I, dim=0)
        O = torch.stack(O, dim=0)
        return H, C, F, I, O
    
    def decode(self, h):
        yhat = self.linear_decode(h)
        return yhat


class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.linear_z = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_r = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_h = nn.Linear(input_size+hidden_size, hidden_size, dtype=torch.float)
        self.linear_decode = nn.Linear(hidden_size, output_size, dtype=torch.float)

    def one_step(self, x, h):
        h = h.to(device)
        input = torch.cat((h,x), dim=1)
        z = torch.sigmoid(self.linear_z(input))
        r = torch.sigmoid(self.linear_z(input))
        h = (1-z) * h + z * torch.tanh(self.linear_h(torch.cat((r * h, x), dim=1)))
        return h, z, r

    def forward(self, x):
        length = x.shape[0]
        batch_size = x.shape[1]
        h0 = torch.zeros(batch_size, self.hidden_size)
        H = [h0]
        Z, R = [], []
        for l in range(length):
            h, z, r = self.one_step(x[l], H[l])
            H.append(h), Z.append(z), R.append(r)
        H = torch.stack(H, dim=0)
        Z = torch.stack(Z, dim=0)
        R = torch.stack(R, dim=0)
        return H, Z, R
    
    def decode(self, h):
        yhat = self.linear_decode(h)
        return yhat
