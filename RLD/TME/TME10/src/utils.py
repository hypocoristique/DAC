import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data/", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
        # Setting default dims here because we know them. Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform, download=True)

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, self.batch_size)
    
    # def total_batches(self):
    #     self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
    #     return len(self.train_dataloader())
