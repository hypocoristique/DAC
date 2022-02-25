from utils import AffineLayer, FlowModel, AverageMeter
from networks import Glow
from sklearn import datasets
import torch
from torch.distributions import normal, independent
import datetime
from utils import plot_to_tensorboard
from torch.utils.tensorboard import SummaryWriter
from icecream import ic
import numpy as np
import torchvision

epochs = 10000
lr = 0.001
batch_size = 10000
dim = 2
n_samples = 50000

z_distrib = independent.Independent(
    normal.Normal(torch.zeros(dim), torch.ones(dim)), 1)

data, _ = datasets.make_moons(
    n_samples=n_samples, shuffle=True, noise=0.05, random_state=0
) 


model = Glow(z_distrib, dim, 8)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter(
    "runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

plot_to_tensorboard(data, path='./origin.png')
for k in range(epochs):
    epoch_loss = AverageMeter()
    for _ in range(len(data)//batch_size):
        id_batch = np.random.randint(0, len(data), batch_size)
        x = torch.tensor(data[id_batch,:], dtype=torch.float)
        z = z_distrib.sample_n(batch_size)
        zhat_log_prob, z_hat, log_det = model.f(x)
        loss = zhat_log_prob.mean() + log_det.mean()
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()
        epoch_loss.update(-loss.item(), batch_size)
    #tot_loss.update(-loss.item(), 1)
    # Plot time
    with torch.no_grad():
        z = z_distrib.sample_n(batch_size)
        x_hat = model.invf(z)[0][-1]
        img = plot_to_tensorboard(x_hat)
    writer.add_image('Generated distrib', img, k, dataformats="HWC")
    writer.add_scalar('loss', -loss, k)
    ic(epoch_loss.avg, k)
    epoch_loss.reset()