import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from icecream import ic

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    # output = output.view(output.shape[1], output.shape[0], -1)
    flatten_output = torch.flatten(output[1:-1], start_dim=0, end_dim=1) #on ne considère plus le premier caractère (car dur pour le modele de le deviner avec h = torch.zeros)
    flatten_target = torch.flatten(target[1:])
    mask = torch.where(flatten_target==padcar, 0, 1).type(torch.float)
    loss = F.cross_entropy(flatten_output, flatten_target, reduction='none')
    return torch.sum(torch.matmul(mask, loss))/torch.sum(mask)
    #faut faire un mask en L*B*A où on fixe tout à 1 sauf quand on a du padding on fixe à 0. On fait un .view pour avoir target en (L*B, A) puis un flatten sur output
    #Puis ensuite on calcul la CE puis on fait CE*Mask


class State:
    def __init__(self, embedder, model, optimizer) -> None:
        self.embedder = embedder
        self.model = model
        self.optimizer = optimizer
        # self.criterion = criterion
        self.epoch, self.iteration = 0, 0

def train(train_data, state, LSTM=False, GRU=False, savepath='../model/model.pch', nb_epochs=10, writer=SummaryWriter(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    cumloss = 0
    for epoch in range(state.epoch, nb_epochs):
        for batch_id, x in enumerate(train_data):
            state.iteration += 1
            state.optimizer.zero_grad()
            x = x.to(device)
            x_emb = state.embedder(x).to(device)
            #x_emb est de taille length*batch_size*embedding_dim
            if LSTM:
                H, _, F, I, O = state.model.forward(x_emb).to(device)
            elif GRU:
                H, Z, R = state.model.forward(x_emb).to(device)
            else:
                H = state.model.forward(x_emb).to(device)
            yhat = state.model.decode(H)
            loss = maskedCrossEntropy(output=yhat, target=x, padcar=0)
            cumloss += loss.item()
            loss.backward()
            state.optimizer.step()
            if batch_id % 50 == 0:
                cumloss_relative = cumloss/state.iteration
                writer.add_scalar("Train loss", loss.item(), state.iteration)
                if LSTM:
                    writer.add_histogram("Forget gate", F.flatten(), state.iteration)
                    writer.add_histogram("Input gate", I.flatten(), state.iteration)
                    writer.add_histogram("Output gate", O.flatten(), state.iteration)
                if GRU:
                    writer.add_histogram("Update gate", Z.flatten(), state.iteration)
                    writer.add_histogram("Reset gate", R.flatten(), state.iteration)
                ic(epoch+1, batch_id)
                ic('loss for this batch', loss.item())
                ic('train loss', cumloss_relative)
                # print(
                #     f'Epoch n°{epoch+1}, Batch n°{batch_id}',
                #     f'\n Batch loss for this batch={loss.item():.4f}',
                #     f'\n Train loss={cumloss_relative:.4f}',
                #     f'\n',
                #     )
            with savepath.open("wb") as fp:
                state.epoch = epoch + 1
                state.embedder.load_state_dict(state.embedder.state_dict())
                state.model.load_state_dict(state.model.state_dict())
                state.optimizer.load_state_dict(state.optimizer.state_dict())
                torch.save(state, fp)

