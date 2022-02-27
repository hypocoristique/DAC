import torch
from tp1 import mse, linear

# Test du gradient de MSE

yhat = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
# grad_output = torch.transpose(torch.ones(yhat.shape), 0, 1)
torch.autograd.gradcheck(mse, (yhat, y))

#  TODO:  Test du gradient de Linear

X = torch.randn(10,5, requires_grad=True, dtype=torch.float64)
W = torch.randn(5,3, requires_grad=True, dtype=torch.float64)
b = torch.randn(3, requires_grad=True, dtype=torch.float64)
# grad_output = [torch.transpose(torch.ones(a.shape), 0, 1) for a in [X, W, b]]
torch.autograd.gradcheck(linear, (X, W, b))