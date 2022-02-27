import logging

from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
from icecream import ic
from model import MLP
from utils import State, train_fn, test_fn, MNIST_Dataset
from pathlib import Path
import time

#Config
TRAIN_RATIO = 0.05 # Ratio du jeu de train à utiliser
BATCH_SIZE = 300
LATENT_DIM = 100
NUM_CLASSES = 10
LEARNING_RATE = 0.0033
NB_EPOCHS = 100
DROPOUT = 0.528 #Turn to 0. for BatchNorm or LayerNorm
REGULARIZATION = 'L2' # 'L1' or 'L2'
REG_LAMBDA = 0.00093 #Regularization parameter
BATCHNORM = False #Turn to False for LayerNorm
LAYERNORM = True
savepath = Path('../models/'+time.strftime("%Y%M%d_%H-%M-%S")+'.pch')
# savepath = Path('../models/model.pch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = MNIST_Dataset(batch_size=BATCH_SIZE, train_ratio=TRAIN_RATIO)
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
test_dataloader = data.test_dataloader()

input_dim = data.dim_in
output_dim = data.dim_out

model = MLP(input_dim=input_dim, latent_dim=LATENT_DIM, output_dim=NUM_CLASSES, dropout=DROPOUT, batchnorm=BATCHNORM, layernorm=LAYERNORM)

if REGULARIZATION == 'L2':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=REG_LAMBDA)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter()
if savepath.is_file():
    print('Loading model...')
    with savepath.open('rb') as fp:
        state = torch.load(fp, map_location=device)
        model = state.model
        # model.load_state_dict(state.model.state_dict())
        print('Done')
else:
    state = State(model, optimizer, criterion)

train_fn(train_dataloader, val_dataloader, test_dataloader, state, regularization=REGULARIZATION, reg_lambda=REG_LAMBDA,
            savepath=savepath, nb_epochs=NB_EPOCHS, writer=writer, device=device)

# test_fn(test_dataloader, state, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))