import torch
from utils import device, ForecastMetroDataset
from model import RNN
from torch.utils.data import DataLoader
from torch import nn
from icecream import ic

#  TODO:  Question 3 : Prédiction de séries temporelles

if __name__ == '__main__':
    batch_size = 64
    num_stations = 80
    DIM_INPUT = 2
    LENGTH = 50

    train, test = torch.load('../data/hzdataset.pch')
    train = train[:, :, :num_stations, :]
    test = test[:, :, :num_stations, :]

    train_dataset = ForecastMetroDataset(train[:,:,:num_stations, :DIM_INPUT], length=LENGTH)
    test_dataset = ForecastMetroDataset(test[:,:,:num_stations, :DIM_INPUT], length=LENGTH, stations_max=train_dataset.stations_max)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_features, train_labels = next(iter(train_data))
    ic(train_features.shape, train_labels.shape)

    input_size = num_stations*train_data.dataset[0][0].shape[-1]
    
    criterion = nn.MSELoss()
    model = RNN(input_size=input_size, hidden_size=50, output_size=input_size, criterion=criterion, opt='Adam', mode='forecast')
    model.fit(train_data, val_data=test_data, n_epochs=50, lr=0.005)