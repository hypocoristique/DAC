import os
import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
from icecream import ic

writer = SummaryWriter()

class TrainingPytorch(nn.Module):
    def __init__(self, criterion, opt, mode=None, ckpt_save_path=None):
        super().__init__()
        self.optimizer = opt
        self.state = {}
        self.ckpt_save_path = ckpt_save_path
        self.criterion = criterion
        self.mode = mode
        ic(mode)

    def __train_epoch(self, train_data):
        epoch_loss = 0
        epoch_acc  = 0
        for x, y in train_data:
            H = self.forward(x)
            if self.mode == 'forecast':
                yhat = self.decode(H)
                yhat = yhat.view(yhat.shape[1], yhat.shape[0], -1)
                y = y.view(y.shape[0], y.shape[1], -1)
                n_correct = 0
            elif self.mode == 'generation':
                yhat = self.decode(H)
                y = y.view(-1)
                x = x.reshape(-1, x.shape[-1])
                yhat = yhat.reshape(-1, yhat.shape[-1])
                n_correct = 0
            else:
                yhat = self.decode(H[-1])
                n_correct = (torch.argmax(yhat, dim=1) == y).sum().item()
            epoch_acc += n_correct/(x.shape[0])
            loss = self.criterion(yhat, y)
            loss.backward()
            epoch_loss += loss.item()
            self.opt.step()
            self.opt.zero_grad()
        return epoch_loss/len(train_data), epoch_acc/len(train_data)
    
    def __validate(self, val_data):
        epoch_loss = 0
        epoch_acc  = 0
        for x, y in val_data:
            H = self.forward(x)
            if self.mode == 'forecast':
                yhat = self.decode(H)
                yhat = yhat.view(yhat.shape[1], yhat.shape[0], -1)
                y = y.view(y.shape[0], y.shape[1], -1)
                n_correct = 0
            elif self.mode == 'generation':
                yhat = self.decode(H)
                y = y.view(-1)
                x = x.reshape(-1, x.shape[-1])
                yhat = yhat.reshape(-1, yhat.shape[-1])
                n_correct = 0
            else:
                yhat = self.decode(H[-1])
                n_correct = (torch.argmax(yhat, dim=1) == y).sum().item()
            loss = self.criterion(yhat, y)
            epoch_acc += n_correct / x.shape[0]
            epoch_loss += loss.item()
        return epoch_loss/len(val_data), epoch_acc/len(val_data)

    def fit(self, train_data, val_data=None, n_epochs=100, lr=0.001, ckpt=False):
        if self.optimizer == "SGD":
            self.opt = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif self.optimizer == "Adam":
            self.opt = optim.Adam(self.parameters(), lr=lr)
        start_epoch = 0
        start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        if ckpt:
            state = torch.load(ckpt)
            start_epoch = state['epoch']
            self.load_state_dict(state['state_dict'])
            for g in self.opt.param_groups:
                g['lr'] = state['lr']
        
        for epoch in range(start_epoch, n_epochs):
            train_loss, train_acc = self.__train_epoch(train_data)
            print('Epoch {:2d} loss: {:1.4f}  Train acc: {:1.4f}'.format(epoch, train_loss, train_acc))
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Acc/train', train_acc, epoch)
            if val_data is not None:
                with torch.no_grad():
                    val_loss, val_acc = self.__validate(val_data)
                print('Epoch {:2d} loss_val: {:1.4f}  val_acc: {:1.4f}'.format(epoch, val_loss, val_acc))
                writer.add_scalar('Loss/test', val_loss, epoch)
                writer.add_scalar('Acc/test', val_acc, epoch)
        
            if self.ckpt_save_path:
                if epoch%20==0 or epoch==n_epochs:
                    self.state['lr'] = lr
                    self.state['epoch'] = epoch
                    self.state['state_dict'] = self.state_dict()
                    if not os.path.exists(self.ckpt_save_path):
                        os.mkdir(self.ckpt_save_path)
                    torch.save(self.state, os.path.join(self.ckpt_save_path, f'ckpt_{start_time}_epoch{epoch}.ckpt'))
