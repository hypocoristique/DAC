import torch
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context, mse, linear
from icecream import ic

# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    ctxl = Context()
    ctxm = Context()
    yhat = Linear.forward(ctxl,x,w,b)
    loss = MSE.forward(ctxm,yhat,y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    grad_output = torch.tensor(1, dtype=torch.float64)
    grad_output = MSE.backward(ctxm,grad_output)
    grad_output = grad_output[0]
    # grad_output = torch.squeeze(torch.stack(list(grad_output), dim=0), dim=0)
    grad_x, grad_w, grad_b = Linear.backward(ctxl,grad_output)

    ##  TODO:  Mise à jour des paramètres du modèle

    w -= epsilon*grad_w
    b -= epsilon*grad_b
