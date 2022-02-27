from textloader import *
from generate import *
from utils import *
from models import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from icecream import ic
from torch import nn
from torch.nn import Embedding
from pathlib import Path

#Parameters
BATCH_SIZE = 128
EMBEDDING_DIM = 100
LATENT_DIM = 100
OUTPUT_DIM = len(lettre2id)
NB_EPOCHS = 10
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
savepath = Path('../models/model_GRU.pch')

#Load data
txt = open("../data/trump_full_speech.txt", "r").read()
ds = TextDataset(txt)
train_data = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=BATCH_SIZE)

#Création du modèle
embedder = nn.Embedding(num_embeddings=OUTPUT_DIM, embedding_dim=EMBEDDING_DIM).to(device)
model = GRU(input_size=EMBEDDING_DIM, hidden_size=LATENT_DIM, output_size=OUTPUT_DIM).to(device)
parameters = list(embedder.parameters())+list(model.parameters())
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)

#Training
writer = SummaryWriter()
if savepath.is_file():
    print('Loading model...')
    with savepath.open('rb') as fp:
        state = torch.load(fp, map_location=device)
        embedder = state.embedder
        model = state.model
        # embedder.load_state_dict(state.embedder.state_dict())
        # model.load_state_dict(state.model.state_dict())
        print('Done')
else:
    state = State(embedder, model, optimizer)

train(train_data, state, LSTM=False, GRU=True, savepath=savepath, nb_epochs=NB_EPOCHS, writer=writer, device=device)

ic(generate_beam(model, embedder, model.decode, eos=1, k=5, start='American people want ', maxlen=200, LSTM=False, GRU=True, nucleus=True))