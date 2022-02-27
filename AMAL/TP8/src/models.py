import torch
import torch.nn as nn
from icecream import ic

class CNN1(nn.Module):
    """1 conv layers, 16 out_channels"""
    def __init__(self, vocab_size, embedding_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=1)
        self.linear1 = nn.Linear(in_features=16, out_features=10)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=10, out_features=output_size)
    
    def conv_block(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], x.shape[2], -1)
        x = self.conv_block(x)
        x = self.pool1(x)
        x = torch.max(x, dim=2).values
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x


class CNN2(nn.Module):
    """2 conv layers, from 32 to 64 out_channels"""
    def __init__(self, vocab_size, embedding_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.linear1 = nn.Linear(in_features=64, out_features=10)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=10, out_features=output_size)
    
    def conv_block(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], x.shape[2], -1)
        x = self.conv_block(x)
        x = self.pool2(x)
        x = torch.max(x, dim=2).values
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x


class CNN3(nn.Module):
    """4 conv layers from 16 to 128 out channels"""
    def __init__(self, vocab_size, embedding_dim, output_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=5, stride=1)
        self.linear1 = nn.Linear(in_features=128, out_features=10)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=10, out_features=output_size)
    
    def conv_block(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.shape[0], x.shape[2], -1)
        x = self.conv_block(x)
        x = self.pool4(x)
        x = torch.max(x, dim=2).values
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.linear2(x)
        return x