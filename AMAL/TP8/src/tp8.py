import logging

import torch.optim as optim

from torch.nn.modules.pooling import MaxPool1d
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sentencepiece as spm

from tp8_preprocess import TextDataset
from models import CNN1, CNN2, CNN3
from utils import train_fn, State
import time
from icecream import ic

# Utiliser tp8_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=1000
TEST_BATCHSIZE=500
EMBEDDING_DIM=200
OUTPUT_SIZE=2
LEARNING_RATE=0.001
NB_EPOCHS = 2
savepath = Path('../models/'+time.strftime("%Y%M%d_%H-%M-%S")+'.pch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate, shuffle=True)
val_iter = DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate, shuffle=True)
test_iter = DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate, shuffle=True)

ic(len(train_iter), len(val_iter), len(test_iter))

#  TODO: 
model = CNN3(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, output_size=OUTPUT_SIZE).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if savepath.is_file():
    print('Loading model...')
    with savepath.open('rb') as fp:
        state = torch.load(fp, map_location=device)
        model = state.model
        # model.load_state_dict(state.model.state_dict())
        print('Done')
else:
    state = State(model, optimizer, criterion)


# labels = torch.zeros(2, dtype=torch.float)
# ic(labels)
# for num, (_, label_batch) in enumerate(train_iter):
#     for i in range(label_batch.shape[0]):
#         labels[label_batch[i]] += 1
# ic(labels)
#La classe majoritaire est donc 0 (799501 contre 799499)


train_fn(train_iter, val_iter, test_iter, state, savepath=savepath, nb_epochs=NB_EPOCHS, writer=SummaryWriter(), device=device)

#Formule de rec :
# S(i+1) = s(i+1) * s(i)
# W(i+1) = W(i) +k(i-1)