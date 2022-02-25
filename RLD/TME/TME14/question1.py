from utils import AffineLayer, FlowModel, AffineCouplingLayer
import torch
from torch.distributions import normal, independent
import datetime
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

epochs = 1000
lr = 0.01
batch_size = 32
dim = 2
mu = - torch.ones(dim)
sigma = 5 * torch.ones(dim)

x_distrib = independent.Independent(normal.Normal(mu, sigma), 1)
z_distrib = independent.Independent(
    normal.Normal(torch.zeros(dim), torch.ones(dim)), 1)


model = FlowModel(z_distrib, AffineLayer(dim=dim), AffineLayer(dim=dim))#,AffineCouplingLayer(dim=dim))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

writer = SummaryWriter(
    "runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

for k in range(epochs):
    x = x_distrib.sample_n(batch_size)
    z = z_distrib.sample_n(batch_size)
    zhat_log_prob, z_hat, log_det = model.f(x)
    loss = zhat_log_prob.mean() + log_det.mean()
    optimizer.zero_grad()
    (-loss).backward()
    optimizer.step()
    writer.add_scalar('loss', -loss, k)
    with torch.no_grad():
        x_hat, _  = model.invf(z)
    writer.add_histogram('x_hat', x_hat[-1], k)
    writer.add_histogram('x', x, k)
    ic(-loss, k)