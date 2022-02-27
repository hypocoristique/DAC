import os
import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from icecream import ic

class State:
    def __init__(self, model, optimizer, criterion) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epoch, self.iteration = 0, 0

writer = SummaryWriter()

# def train_fn(train_data, state, savepath='../model/model.pch', nb_epochs=10, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     cumloss = 0
#     for epoch in range(state.epoch, nb_epochs):
#         for batch_id, (x, y) in enumerate(train_data):
#             state.iteration += 1
#             state.optimizer.zero_grad()
#             #x_emb est de taille length*batch_size*embedding_dim
#             yhat = state.model(x)
#             loss = state.criterion(yhat, y)
#             cumloss += loss.item()
#             loss.backward()
#             state.optimizer.step()
#             if batch_id % 50 == 0:
#                 cumloss_relative = cumloss/state.iteration
#                 writer.add_scalar("Train loss", loss.item(), state.iteration)
#                 ic(epoch+1, batch_id)
#                 ic('loss for this batch', loss.item())
#                 ic('train loss', cumloss_relative)
#                 # print(
#                 #     f'Epoch n°{epoch+1}, Batch n°{batch_id}',
#                 #     f'\n Batch loss for this batch={loss.item():.4f}',
#                 #     f'\n Train loss={cumloss_relative:.4f}',
#                 #     f'\n',
#                 #     )
#             with savepath.open("wb") as fp:
#                 state.epoch = epoch + 1
#                 state.embedder.load_state_dict(state.embedder.state_dict())
#                 state.model.load_state_dict(state.model.state_dict())
#                 state.optimizer.load_state_dict(state.optimizer.state_dict())
#                 torch.save(state, fp)

def train_fn(train_dataloader, val_dataloader, test_dataloader, state, savepath='../model/model.pch',
                nb_epochs=10, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    val_iterations = 0
    test_iterations = 0
    for epoch in range(state.epoch, nb_epochs):
        cumloss = 0
        cum_sum_correct = 0
        val_cumloss = 0
        val_cum_sum_correct = 0
        test_cumloss = 0
        test_cum_sum_correct = 0

        #train_set
        state.model.train()
        for batch_id, (batch_img, batch_labels) in enumerate(train_dataloader):
            state.iteration += 1
            batch_size = batch_labels.shape[0]
            batch_img = batch_img.to(device)
            batch_labels = batch_labels.to(device)
            state.optimizer.zero_grad()
            logits = state.model(batch_img)
            loss = state.criterion(logits, batch_labels)
            cumloss += loss.item()
            loss.backward()
            state.optimizer.step()
            pred_labels = torch.argmax(F.softmax(logits, dim=1), dim=1)
            sum_correct = torch.sum(pred_labels==batch_labels)
            batch_accuracy = sum_correct/batch_size
            trivial_accuracy = torch.sum(torch.zeros_like(batch_labels)==batch_labels)/batch_size #0 est le label majoritaire
            cum_sum_correct += sum_correct
            if batch_id % (len(train_dataloader) // 100) == 0:
                writer.add_scalar(f"Train batch loss", loss, state.iteration)
                writer.add_scalar(f"Train batch accuracy", batch_accuracy, state.iteration)
                writer.add_scalar(f"Train relative batch accuracy", batch_accuracy/trivial_accuracy, state.iteration)
                ic(batch_id, loss.item())

        #val_set
        with torch.no_grad():
            for val_batch_id, (val_batch_img, val_batch_labels) in enumerate(val_dataloader):
                val_iterations += 1
                batch_size = val_batch_labels.shape[0]
                val_batch_img = val_batch_img.to(device)
                val_batch_labels = val_batch_labels.to(device)
                val_logits = state.model(val_batch_img)
                val_loss = state.criterion(val_logits, val_batch_labels)
                val_cumloss += val_loss.item()
                val_pred_labels = torch.argmax(F.softmax(val_logits, dim=1), dim=1)
                val_sum_correct = torch.sum(val_pred_labels==val_batch_labels)
                val_batch_accuracy = val_sum_correct/batch_size
                val_trivial_accuracy = torch.sum(torch.zeros_like(val_batch_labels)==val_batch_labels)/batch_size #0 est le label majoritaire
                val_cum_sum_correct += val_sum_correct
                writer.add_scalar("Val batch loss", val_loss, val_iterations)
                writer.add_scalar(f"Val batch accuracy", val_batch_accuracy, state.iteration)
                writer.add_scalar(f"Val relative batch accuracy", val_batch_accuracy/val_trivial_accuracy, state.iteration)
                ic(val_batch_id, val_loss.item())
        
        #test_set
            state.model.eval()
            for test_batch_id, (test_batch_img, test_batch_labels) in enumerate(test_dataloader):
                test_iterations += 1
                batch_size = test_batch_labels.shape[0]
                test_batch_img = test_batch_img.to(device)
                test_batch_labels = test_batch_labels.to(device)
                test_logits = state.model(test_batch_img)
                test_loss = state.criterion(test_logits, test_batch_labels)
                test_cumloss += test_loss.item()
                test_pred_labels = torch.argmax(F.softmax(test_logits, dim=1), dim=1)
                test_sum_correct = torch.sum(test_pred_labels==test_batch_labels)
                test_batch_accuracy = test_sum_correct/batch_size
                test_trivial_accuracy = torch.sum(torch.zeros_like(test_batch_labels)==test_batch_labels)/batch_size #0 est le label majoritaire
                test_cum_sum_correct += test_sum_correct
                writer.add_scalar("Test batch loss", test_loss, test_iterations)
                writer.add_scalar(f"Test batch accuracy", test_batch_accuracy, state.iteration)
                writer.add_scalar(f"Test relative batch accuracy", test_batch_accuracy/test_trivial_accuracy, state.iteration)
                ic(test_batch_id, test_loss.item())
        
        
        writer.add_scalar("Train loss", cumloss, epoch)
        total_acc = (cum_sum_correct / len(train_dataloader.dataset)).item()
        writer.add_scalar("Train accuracy", total_acc, epoch)
        writer.add_scalar("Validation loss", val_cumloss, epoch)
        val_total_acc = (val_cum_sum_correct / len(val_dataloader.dataset)).item()
        writer.add_scalar("Validation accuracy", val_total_acc, epoch)
        writer.add_scalar("Test loss", test_cumloss, epoch)
        test_total_acc = (test_cum_sum_correct / len(test_dataloader.dataset)).item()
        writer.add_scalar("Test accuracy", test_total_acc, epoch)
        # if epoch % (nb_epochs // 10) == 0:
        ic(state.epoch, cumloss, total_acc)
        ic(val_cumloss, val_total_acc)
        ic(test_cumloss, test_total_acc)

        with savepath.open("wb") as fp:
                state.epoch = epoch
                state.model.load_state_dict(state.model.state_dict())
                # state.optimizer.load_state_dict(state.optimizer.state_dict())
                torch.save(state, fp)