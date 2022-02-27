import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np

from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from model import BasicNet, SimpleAttention, ComplexAttention
from utils import State, train_fn
from tp9_preprocessing import get_imdb_data, collate_fn
from icecream import ic

word2id, embeddings, train_data, test_data = get_imdb_data()

#Faire un softmax de <q,t_i> pour obtenir alpha
#Mais attention, faut pas faire le softmax sur tout le vecteur (à cause du padding),
# faut le faire uniquement sur les valeurs non nulles. Possibilité : mettre tous les caractères en trop à float('inf')

embedding_dim = embeddings.shape[1]
ic(embedding_dim)

output_dim = 2
BATCH_SIZE = 8
NB_EPOCHS = 10
PAD_IDX = 400001
# savepath = Path('../models/'+time.strftime("%Y%M%d_%H-%M-%S")+'.pch')
savepath = Path('../models/ComplexAttention.pch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)


# model = BasicNet(torch.Tensor(embeddings), embedding_dim, output_dim)
# model = SimpleAttention(torch.Tensor(embeddings), embedding_dim, output_dim, pad_idx=PAD_IDX)
model = ComplexAttention(torch.Tensor(embeddings), embedding_dim, output_dim, pad_idx=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

if savepath.is_file():
    print('Loading model...')
    with savepath.open('rb') as fp:
        state = torch.load(fp, map_location=device)
        model = state.model
        # model.load_state_dict(state.model.state_dict())
        print('Done')
else:
    state = State(model, optimizer, criterion)

train_fn(train_dataloader, test_dataloader, state, savepath, nb_epochs=NB_EPOCHS)

if savepath == Path('../models/SimpleAttention.pch'):
    iterator = iter(test_dataloader)
    x, _, _ = next(iterator)
    top = 15
    id2word = {ix:word for word, ix in word2id.items()}
    model.eval()
    with torch.no_grad():
        ic(x.shape)
        _ , alpha = model(x)
        alpha = alpha.view(-1)
        top_values, top_idx = torch.topk(alpha, top)
        phrase = []
        x = x.view(-1).tolist()
        for idx in x:
            phrase.append(id2word[idx])
        print(f'input phrase: {phrase}')
        for word_idx, couple in enumerate(zip(top_values.tolist(), top_idx.tolist())):
            print(f'word: {id2word[x[couple[1]]]}, attention value: {couple[0]}')