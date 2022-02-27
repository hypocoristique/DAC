import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

writer = SummaryWriter()

class BasicNet(nn.Module):
    def __init__(self, embeddings, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.linear1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(embedding_dim//2, output_dim)
    
    def forward(self, x):
        x_emb = self.embedding(x)
        x_mean = torch.mean(x_emb, dim=1)
        x1 = self.linear1(x_mean)
        a1 = self.relu1(x1)
        x2 = self.linear2(a1)
        return x2, None

class SimpleAttention(nn.Module):
    def __init__(self, embeddings, embedding_dim, output_dim, pad_idx):
        super(SimpleAttention, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, output_dim)
        )
        self.q = nn.Linear(embedding_dim, 1, bias=False)
    
    def forward(self, x):
        x_emb = self.embedding(x)
        alpha = torch.where((x == self.pad_idx).unsqueeze(dim=2), torch.tensor(-float("Inf"), dtype=torch.float),
                                    self.q(x_emb))
        alpha = F.softmax(alpha, dim=1)
        alpha_expanded = alpha.expand(x_emb.shape)
        z = (alpha_expanded * x_emb).sum(dim=1)
        logits = self.MLP(z)
        return logits, alpha

class ComplexAttention(nn.Module):
    def __init__(self, embeddings, embedding_dim, output_dim, pad_idx):
        super(ComplexAttention, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.value = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim)
        )
        self.query = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim)
        )
        self.key = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim*2),
            nn.ReLU(),
            nn.Linear(embedding_dim*2, embedding_dim)
        )
        self.MLP = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim//2),
            nn.ReLU(),
            nn.Linear(embedding_dim//2, output_dim)
        )
    
    def forward(self, x):
        x_emb = self.embedding(x)
        k = self.key(x_emb)
        v = self.value(x_emb)
        q = self.query(k.mean(dim=1))
        k = k.unsqueeze(dim=2)
        q = q.unsqueeze(dim=1).unsqueeze(dim=-1).repeat(1, k.shape[1], 1, 1)
        qx = torch.matmul(k, q).squeeze()
        alpha = torch.where((x == self.pad_idx), torch.tensor(-float("Inf"), dtype=torch.float),
                                    qx)
        alpha = F.softmax(alpha, dim=1)
        alpha_expanded = alpha.unsqueeze(dim=-1).expand(v.shape)
        z = (alpha_expanded * v).sum(dim=1)
        logits = self.MLP(z)
        return logits, alpha
