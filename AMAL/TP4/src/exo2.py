import torch
from utils import device, SampleMetroDataset
from model import RNN
from torch.utils.data import DataLoader
from torch import nn
from icecream import ic

#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

if __name__ == '__main__':
    batch_size = 64
    num_stations = 10

    train, test = torch.load('../data/hzdataset.pch')
    train = train[:, :, :num_stations, :]
    test = test[:, :, :num_stations, :]

    train_dataset = SampleMetroDataset(train)
    test_dataset = SampleMetroDataset(test, stations_max=train_dataset.stations_max)

    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    input_size = train_data.dataset[0][0].shape[-1]

    criterion = nn.CrossEntropyLoss()
    model = RNN(input_size=input_size, hidden_size=30, output_size=num_stations, criterion=criterion, opt='Adam')
    model.fit(train_data, val_data=test_data, n_epochs=50, lr=0.001)