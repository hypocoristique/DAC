import os
import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
import torch.nn.functional as F


class State:
    def __init__(self, model, optimizer, criterion) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch, self.iteration = 0, 0


def train_fn(train_dataloader, val_dataloader, state, savepath='../model/model.pch',
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
        for batch_img, batch_lens, batch_labels in train_dataloader:
            state.optimizer.zero_grad()
            logits, _ = state.model(batch_img)
            loss = state.criterion(logits, batch_labels)
            cumloss += (loss.item()/batch_img.shape[0])
            loss.backward()
            state.optimizer.step()
            pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
            sum_correct = torch.sum(pred_labels==batch_labels)
            cum_sum_correct += sum_correct
            state.iteration += 1

        #val_set
        with torch.no_grad():
            for val_batch_img, val_batch_lens, val_batch_labels in val_dataloader:
                val_logits, _ = state.model(val_batch_img)
                val_loss = state.criterion(val_logits, val_batch_labels)
                val_cumloss += (val_loss.item()/val_batch_img.shape[0])
                val_pred_labels = torch.argmax(F.softmax(val_logits, dim=1), dim=1)
                val_sum_correct = torch.sum(val_pred_labels==val_batch_labels)
                val_cum_sum_correct += val_sum_correct
        
        writer.add_scalar("Train loss", cumloss, epoch)
        total_acc = (cum_sum_correct / len(train_dataloader.dataset)).item()
        writer.add_scalar("Train accuracy", total_acc, epoch)
        writer.add_scalar("Validation loss", val_cumloss, epoch)
        val_total_acc = (val_cum_sum_correct / len(val_dataloader.dataset)).item()
        writer.add_scalar("Validation accuracy", val_total_acc, epoch)
        ic(epoch, cumloss, total_acc)
        ic(val_cumloss, val_total_acc)

        with savepath.open("wb") as fp:
                state.epoch = epoch
                state.model.load_state_dict(state.model.state_dict())
                # state.optimizer.load_state_dict(state.optimizer.state_dict())
                torch.save(state, fp)


                