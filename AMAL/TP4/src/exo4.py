import string
import unicodedata
import torch
import sys
import random
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from utils import device
from model import RNN
from icecream import ic

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]


txt = open('../data/trump_full_speech.txt', 'r').read()
batch_size = 64
nb_epochs = 30
hidden_size = 30
seq_length = 20

train = TrumpDataset(txt, maxsent=300)
train_data = DataLoader(train, shuffle=True, batch_size=batch_size)
criterion = nn.CrossEntropyLoss()
train_features,train_label=next(iter(train_data))
ic(train_features.shape, train_label.shape)

model = RNN(input_size=1, hidden_size=hidden_size, output_size=len(id2lettre), criterion=criterion, mode='generation', opt='Adam', ckpt_save_path='../models', seq_length=seq_length)
model.fit(train_data, n_epochs=nb_epochs, lr=0.001)

Yhat_seq = []
# x = [random.choice(list(lettre2id.values()))]
x = string2code('The world is ')
# x = train_data.dataset[2][0]
ic(code2string(x))
x = torch.unsqueeze(torch.tensor(x), dim=0)
ic('x.shape', x.shape)
for i in range(seq_length): 
    yhat = model.predict(x)
    # yhat_str = code2string(yhat)
    yhat_str = id2lettre[yhat]
    Yhat_seq.append(yhat_str)
    # x = torch.unsqueeze(yhat, dim=0)
ic('Yhat_seq', len(Yhat_seq), Yhat_seq)