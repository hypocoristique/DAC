import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

writer = SummaryWriter()

class MLP(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim, dropout=0., batchnorm=False, layernorm=False):
        super().__init__()
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dim//2, input_dim//4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_dim//4, 100)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(latent_dim, output_dim)
        self.grads = {}
        if dropout != 0.:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(input_dim//2)
            self.bn2 = nn.BatchNorm1d(input_dim//4)
            self.bn3 = nn.BatchNorm1d(100)
        if layernorm:
            self.ln1 = nn.LayerNorm(input_dim//2)
            self.ln2 = nn.LayerNorm(input_dim//4)
            self.ln3 = nn.LayerNorm(100)
    
    def forward(self, x, writer, epoch, save_grad=False):
        if self.dropout != 0.:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            d1 = self.dropout1(a1)
            x2 = self.linear2(d1)
            a2 = self.relu2(x2)
            d2 = self.dropout2(a2)
            x3 = self.linear3(d2)
            a3 = self.relu3(x3)
            d3 = self.dropout3(a3)
            out = self.linear4(d3)
        elif self.batchnorm:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            b1 = self.bn1(a1)
            x2 = self.linear2(b1)
            a2 = self.relu2(x2)
            b2 = self.bn2(a2)
            x3 = self.linear3(b2)
            a3 = self.relu3(x3)
            b3 = self.bn3(a3)
            out = self.linear4(b3)
        elif self.layernorm:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            l1 = self.ln1(a1)
            x2 = self.linear2(l1)
            a2 = self.relu2(x2)
            l2 = self.ln2(a2)
            x3 = self.linear3(l2)
            a3 = self.relu3(x3)
            l3 = self.ln3(a3)
            out = self.linear4(l3)
        else:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            x2 = self.linear2(a1)
            a2 = self.relu2(x2)
            x3 = self.linear3(a2)
            a3 = self.relu3(x3)
            out = self.linear4(a3)
        if save_grad:
            a1.register_hook(self.store_grad("a1"))
            a2.register_hook(self.store_grad("a2"))
            a3.register_hook(self.store_grad("a3"))
            out.register_hook(self.store_grad("out"))
            for key, value in self.grads.items():
                writer.add_histogram(f'{key}_grad', value, epoch)
        return out
    
    def store_grad(self, var):
        def hook(grad):
            self.grads[var] = grad
        return hook

