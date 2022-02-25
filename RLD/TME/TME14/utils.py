from torch.utils.data import DataLoader
import torch.nn as nn
import enum
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")
import seaborn as sns
from icecream import ic
import cv2
sns.set_theme()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def toDataFrame(t: torch.Tensor, origin: str):
    t = t.cpu().detach().numpy()
    df = pd.DataFrame(data=t, columns=(f"x{ix}" for ix in range(t.shape[1])))
    df['ix'] = df.index * 1.
    df["origin"] = origin
    return df

def plot_to_tensorboard(data, path='./temp.png'):
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.axis('off')
    ax.scatter(data[:,0], data[:,1])
    canvas.draw()       # draw the canvas, cache the renderer

    s, (width, height) = canvas.print_to_buffer()
    X = np.fromstring(s, np.uint8).reshape((height, width, 4))
    cv2.imwrite(path, X)
    #X = torch.from_numpy(X).view(-1, height, width)
    return X

def scatterplots(samples: List[Tuple[str, torch.Tensor]], col_wrap=4):
    """Draw the 

    Args:
        samples (List[Tuple[str, torch.Tensor]]): The list of samples with their types
        col_wrap (int, optional): Number of columns in the graph. Defaults to 4.

    Raises:
        NotImplementedError: If the dimension of the data is not supported
    """
    # Convert data into pandas dataframes
    _, dim = samples[0][1].shape
    samples = [toDataFrame(sample, name) for name, sample in samples]
    data = pd.concat(samples, ignore_index=True)

    g = sns.FacetGrid(data, height=2, col_wrap=col_wrap, col="origin", sharex=False, sharey=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    if dim == 1:
        g.map(sns.kdeplot, "distribution")
        plt.show()
    elif dim == 2:
        g.map(sns.scatterplot, "x0", "x1", alpha=0.6)
        plt.show()
    else:
        raise NotImplementedError()


def iter_data(dataset, bs):
    """Infinite iterator"""
    while True:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        yield from iter(loader)


class MLP(nn.Module):
    """Réseau simple 4 couches"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)


# --- Modules de base

class FlowModule(nn.Module):
    """
    On a f: z -> x
    Et invf x -> z
    """
    def __init__(self):
        super().__init__()

    def invf(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f^-1(x) and log |det J_f^-1(x)|"""
        ...

    def f(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns f(x) and log |det J_f(x)|"""
        ...

# --- Affine Flow Layer

class AffineLayer(FlowModule):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(dim))
        self.t = nn.Parameter(torch.zeros(dim))
    
    def invf(self, y):
        return (y * torch.exp(-self.s) - self.t), - torch.sum(torch.log(torch.abs(self.s.exp())))

    def f(self, x):
       return (x + self.t)* torch.exp(self.s), torch.sum(torch.log(torch.abs(self.s.exp())))

# --- Glow Module

class AffineCouplingLayer(FlowModule):
    def __init__(self, dim):
        super().__init__()
        self.mlp = MLP(dim//2, dim, 64*dim)
    
    def invf(self, y):
        y_1, y_2 = y.chunk(2, dim=-1)
        log_s, t = self.mlp(y_1).chunk(2,-1)
        s = torch.sigmoid(log_s + 2)
        x_2 = y_2 / s - t 
        x = torch.cat([y_1,x_2], dim=-1)
        log_det = - torch.sum(torch.log(torch.abs(s)))
        return x, log_det

    def f(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        log_s, t = self.mlp(x_1).chunk(2,-1)
        s = torch.sigmoid(log_s + 2)
        y_2 = (x_2 + t) * s
        y = torch.cat([x_1 ,y_2], dim=-1)
        log_det = torch.sum(torch.log(torch.abs(s)))
        return y, log_det


class Conv1DLayer(FlowModule):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim,dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            nn.init.orthogonal_(self.W)

    def invf(self, y):
        W_inv = torch.inverse(self.W)
        x = torch.matmul(y, W_inv)
        #P, L, U = torch.lu_unpack(*self.W.lu())
        #S = torch.diag(U)
        log_det = - torch.slogdet(self.W)[1]
        #log_det = - 1 * 1 * torch.sum(torch.log(torch.abs(S)))
        return x, log_det

    def f(self, x):
        y = torch.matmul(x, self.W)
        log_det = torch.slogdet(self.W)[1]
        # P, L, U = torch.lu_unpack(*self.W.lu())
        # S = torch.diag(U)
        # h = 1, w = 1
        #log_det = 1 * 1 * torch.sum(torch.log(torch.abs(S)))
        return y, log_det

# --- utilities
class FlowModules(FlowModule):
    """A container for a succession of flow modules"""
    def __init__(self, *flows: FlowModule):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def apply(self, modules_iter, caller, x):
        m, _ = x.shape
        logdet = torch.zeros(m, device=x.device)
        zs = [x]
        for module in modules_iter:
            x, _logdet = caller(module, x)
            zs.append(x)
            logdet += _logdet
        return zs, logdet            

    def modulenames(self, backward=False):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if backward else self.flows)]

    def f(self, x):
        zs, logdet = self.apply(self.flows, lambda m, x: m.f(x), x)
        return zs, logdet

    def invf(self, y):
        zs, logdet = self.apply(reversed(self.flows), lambda m, y: m.invf(y), y)
        return zs, logdet


class FlowModel(FlowModules):
    """Flow model = prior + flow modules"""
    def __init__(self, prior, *flows: FlowModule):
        super().__init__(*flows)
        self.prior = prior

    # def invf(self, x):
    #     # Just computes the prior
    #     zs, logdet = super().invf(x)
    #     logprob = self.prior.log_prob(zs[-1])
    #     return logprob, zs, logdet

    def f(self, x):
        # Just computes the prior
        zs, logdet = super().f(x)
        logprob = self.prior.log_prob(zs[-1])
        return logprob, zs, logdet

