from random import seed
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from utils import MNISTDataModule
from models import VAE
from icecream import ic

def show_img(img):
    img = img.clamp(0,1)
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    
def reconstruction(dataloader, vae, model_type, input_dim):
    with torch.no_grad():
        plt.ion()
        images, labels = next(iter(dataloader))
        print('Original images')
        show_img(make_grid(images[1:50], 10, 5))
        plt.show()
        print('Reconstructed images')
        if model_type == 'MLP':
            images = images.view(-1, input_dim)
        _, x_hat, _, _, _ = vae(images)
        x_hat = torch.unsqueeze(x_hat, dim=1)
        x_hat = x_hat.view(x_hat.shape[0], x_hat.shape[1], 28, 28).clamp(0, 1)
        np_imagegrid = make_grid(x_hat[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

def generate(vae):
    with torch.no_grad():
        latent = torch.randn(128, vae.latent_dim)
        img_recon = vae.model.decoder(latent)
        img_recon = img_recon.view(-1, 1, 28, 28)
        img_recon = img_recon.cpu()
        fig, ax = plt.subplots(figsize=(5, 5))
        show_img(make_grid(img_recon.data[:100],10,5))
        plt.show()


if __name__ == '__main__':
    PATH = '../models/CNN_latent2.ckpt'

    BATCH_SIZE = 128
    INPUT_DIM = 28*28
    LATENT_DIM = 2
    LEARNING_RATE = 1e-4 #1e-3 pour MLP
    MAX_EPOCHS = 20
    ENC_OUT_DIM = 16 #Par exemple 64 pour LinearModel, 8 pour CNN
    MODEL_TYPE = 'CNN'

    vae = VAE.load_from_checkpoint(PATH, input_dim=INPUT_DIM, enc_out_dim=ENC_OUT_DIM, latent_dim=LATENT_DIM, learning_rate=LEARNING_RATE, model_type=MODEL_TYPE)
    datamodule = MNISTDataModule()
    datamodule.prepare_data()
    datamodule.setup(stage='test')
    test_dataloader = datamodule.test_dataloader()
    vae.eval()
    reconstruction(test_dataloader, vae, model_type=MODEL_TYPE, input_dim=INPUT_DIM)
    generate(vae)

    