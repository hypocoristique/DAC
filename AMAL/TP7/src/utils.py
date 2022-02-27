import os
import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import torch.nn.functional as F
from torch.distributions import Categorical

class State:
    def __init__(self, model, optimizer, criterion) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch, self.iteration = 0, 0


def train_fn(train_dataloader, val_dataloader, test_dataloader, state, regularization=None, reg_lambda=0.1, savepath='../model/model.pch',
                nb_epochs=10, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    for epoch in range(state.epoch, nb_epochs):
        cumloss = 0
        cum_sum_correct = 0
        val_cumloss = 0
        val_cum_sum_correct = 0
        test_cumloss = 0
        test_cum_sum_correct = 0

        #train_set
        state.model.train()
        for batch_img, batch_labels in train_dataloader:
            state.optimizer.zero_grad()
            if epoch % (nb_epochs // 20) == 0:
                logits = state.model(batch_img, writer, epoch, save_grad=True)
            else:
                logits = state.model(batch_img, writer, epoch)
            loss = state.criterion(logits, batch_labels)
            if regularization == 'L1':
                l1_penalty = sum(p.abs().sum() for p in state.model.parameters())
                loss = loss + reg_lambda * l1_penalty
            cumloss += loss.item()
            loss.backward()
            state.optimizer.step()
            pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
            sum_correct = torch.sum(pred_labels==batch_labels)
            cum_sum_correct += sum_correct
            state.iteration += 1
            if epoch % (nb_epochs // 20) == 0:
                entropy_histogram(writer, logits, epoch)

        #val_set
        with torch.no_grad():
            for val_batch_img, val_batch_labels in val_dataloader:
                val_logits = state.model(val_batch_img, writer, epoch)
                val_loss = state.criterion(val_logits, val_batch_labels)
                val_cumloss += val_loss.item()
                val_pred_labels = torch.argmax(F.softmax(val_logits, dim=1), dim=1)
                val_sum_correct = torch.sum(val_pred_labels==val_batch_labels)
                val_cum_sum_correct += val_sum_correct
        
        #test_set
            state.model.eval()
            for test_batch_img, test_batch_labels in test_dataloader:
                test_logits = state.model(test_batch_img, writer, epoch)
                test_loss = state.criterion(test_logits, test_batch_labels)
                test_cumloss += test_loss.item()
                test_pred_labels = torch.argmax(F.softmax(test_logits, dim=1), dim=1)
                test_sum_correct = torch.sum(test_pred_labels==test_batch_labels)
                test_cum_sum_correct += test_sum_correct
        
        writer.add_scalar("Train loss", cumloss, epoch)
        total_acc = (cum_sum_correct / len(train_dataloader.dataset)).item()
        writer.add_scalar("Train accuracy", total_acc, epoch)
        writer.add_scalar("Validation loss", val_cumloss, epoch)
        val_total_acc = (val_cum_sum_correct / len(val_dataloader.dataset)).item()
        writer.add_scalar("Validation accuracy", val_total_acc, epoch)
        writer.add_scalar("Test loss", test_cumloss, epoch)
        test_total_acc = (test_cum_sum_correct / len(test_dataloader.dataset)).item()
        writer.add_scalar("Test accuracy", test_total_acc, epoch)
        if epoch % (nb_epochs // 10) == 0:
            ic(state.epoch, cumloss, total_acc)
            ic(val_cumloss, val_total_acc)
            ic(test_cumloss, test_total_acc)
        if epoch % (nb_epochs // 20) == 0:
            for name, param in state.model.named_parameters():
                writer.add_histogram(name, param, epoch)

        with savepath.open("wb") as fp:
                state.epoch = epoch
                state.model.load_state_dict(state.model.state_dict())
                # state.optimizer.load_state_dict(state.optimizer.state_dict())
                torch.save(state, fp)
        
# def test_fn(test_dataloader, state, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     state.model.eval()
#     epoch = 0
#     test_cum_sum_correct = 0
#     for test_batch_img, test_batch_labels in test_dataloader:
#         test_logits = state.model(test_batch_img, writer, epoch)
#         test_pred_labels = torch.argmax(F.softmax(test_logits, dim=1), dim=1)
#         test_sum_correct = torch.sum(test_pred_labels==test_batch_labels)
#         test_cum_sum_correct += test_sum_correct
#     test_total_acc = (test_cum_sum_correct / len(test_dataloader.dataset)).item()
#     ic(test_total_acc)


def entropy_histogram(writer, logits, epoch):
    random_logits = torch.randn_like(logits)
    writer.add_histogram('entropy_output', Categorical(logits=logits).entropy(), global_step=epoch)
    writer.add_histogram('entropy_random_output', Categorical(logits=random_logits).entropy(), global_step=epoch)


class MNIST_Dataset(Dataset):
    def __init__(self, batch_size, train_ratio) -> None:
        super().__init__()
        self.batch_size = batch_size
        ds = prepare_dataset("com.lecun.mnist")
        #En train :
        shape = ds.train.images.data().shape
        self.dim_in = shape[1]*shape[2]
        self.dim_out = len(set(ds.train.labels.data())) #set() supprime les doublons d'une séquence
        train_img = torch.tensor(ds.train.images.data(), dtype=torch.float).view(-1, self.dim_in)/255.
        train_labels = torch.tensor(ds.train.labels.data(), dtype=torch.long)
        ds_train = TensorDataset(train_img, train_labels)
        self.train_length = int(shape[0]*train_ratio)
        self.mnist_train, self.mnist_val = random_split(ds_train, [self.train_length, shape[0]-self.train_length])
        #En test :
        test_img = torch.tensor(ds.test.images.data(), dtype=torch.float).view(-1, self.dim_in)/255.
        test_labels = torch.tensor(ds.test.labels.data(), dtype=torch.long)
        self.mnist_test = TensorDataset(test_img, test_labels)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def input_dim(self):
        return self.dim_in
    
    def output_dim(self):
        return self.dim_out