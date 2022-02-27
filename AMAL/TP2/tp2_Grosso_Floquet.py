import torch
from torch import nn
from torch.autograd import gradcheck
from torch.nn.modules.loss import MSELoss
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
from icecream import ic
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax,dtype=torch.float)
datay = torch.tensor(datay,dtype=torch.float).reshape(-1,1)

#Normalization
datax = (datax-torch.mean(datax))/torch.std(datax)
datay = (datay-torch.mean(datay))/torch.std(datay)

x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size = 0.1)

def linear(x,w,b):
    return x@w.T+b

def mse(yhat, y):
    return (1/y.shape[0])*torch.norm(yhat-y)**2

#Question 1
#Changer l'argument batch_size=y_train.shape[0]) pour obtenir le mini-batch et le SGD (batch_size=1)

def train(x, y, batch_size=y_train.shape[0], epochs=100, lr=0.003):
    w = torch.randn(1, x.shape[1], requires_grad=True).to(torch.float)
    b = torch.randn(1,1, requires_grad=True).to(torch.float)

    for epoch in range(epochs):
        for batch in range(int(x.shape[0]/batch_size)):
            if batch_size*batch+batch <= x.shape[0]:
                x_batch = x[batch*batch_size:batch*batch_size+batch_size,]
                y_batch = y[batch*batch_size:batch*batch_size+batch_size,]
            else:
                x_batch = x[batch*batch_size:x.shape[0],]
                y_batch = y[batch*batch_size:y.shape[0],]
            loss = mse(linear(x_batch,w,b),y_batch)
            loss.backward()
            with torch.no_grad():
                w -= lr*w.grad
                b -= lr*b.grad
                w.grad.zero_()
                b.grad.zero_()
        if epoch%10==0:
            ic(epoch, loss.item())
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/test', mse(linear(x_test,w,b),y_test), epoch)
    return loss.item()

ic(train(x_train, y_train))

#Question 2

#With_optim
class LinReg(nn.Module):
    def __init__(self):
        super(LinReg,self).__init__()
        self.linear_1=nn.Linear(x_train.shape[-1],x_train.shape[-1]//2)
        self.tan1=nn.Tanh()
        self.linear_2=nn.Linear(x_train.shape[-1]//2, y_train.shape[-1])
        self.Loss=nn.MSELoss()

    def forward(self,x):
        out1=self.linear_1(x)
        out2=self.tan1(out1)
        out3=self.linear_2(out2)
        return out3

def fit(x, y, model, opt, batch_size=y_train.shape[0], epochs=100,lr=0.01):
    for epoch in range(epochs):
        model.train() #for training
        for i in range(int(x.shape[0]/batch_size)):
            start_i = i*batch_size
            end_i = start_i + batch_size
            xb    = x[start_i:end_i]
            yb    = y[start_i:end_i]
            yhat  = model(x)
            loss  = nn.MSELoss()(yhat,y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if epoch%10==0:
                ic(epoch, loss)
            writer.add_scalar('Loss/train_optim', loss, epoch) 
        model.eval() #for evaluation
        with torch.no_grad():
            yhat=model(x)
            test_loss=nn.MSELoss()(yhat,y)
            writer.add_scalar('Loss/train_optim', loss, epoch) 
    return loss,test_loss

def get_model(model):
    if model=='LinReg':
        model=LinReg()
    return model, torch.optim.SGD(model.parameters(),lr=0.01)

model, opt = get_model('LinReg')
ic(fit(x_train,y_train,model,opt))


#Sequential
def train_sequential(x, y, epochs=100,lr=0.01):
    w = torch.nn.Parameter(torch.randn(1, x.shape[1]).to(torch.float))
    b = torch.nn.Parameter(torch.randn(1).to(torch.float))
    model = nn.Sequential(
        nn.Linear(x_train.shape[-1], x_train.shape[-1]//2),
        nn.Tanh(),
        nn.Linear(x_train.shape[-1]//2, y_train.shape[-1]),
    )
    for epoch in range(epochs):
        yhat = model(x)
        loss = nn.MSELoss()(yhat,y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= lr*param.grad
        writer.add_scalar('Loss/train_sequential', loss, epoch) 
        if epoch%10==0:
            ic(epoch, loss.item())
    return loss.item()

ic(train_sequential(x_train, y_train))