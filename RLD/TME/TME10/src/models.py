import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid
from icecream import ic

class MLP(nn.Module):
    def __init__(self, input_dim, enc_out_dim=64, latent_dim=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.fc_enc = nn.Linear(input_dim, enc_out_dim)
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)
        self.fc_dec_1 = nn.Linear(latent_dim, enc_out_dim)
        self.fc_dec_2 = nn.Linear(enc_out_dim, input_dim)
        self.relu_enc = nn.ReLU()
        self.relu_dec = nn.ReLU()
    
    def encoder(self, x):
        x_enc = self.relu_enc(self.fc_enc(x))
        x_mu = self.fc_mu(x_enc)
        x_logvar = self.fc_logvar(x_enc)
        return x_mu, x_logvar
    
    def decoder(self, x):
        x_fc1 = self.relu_dec(self.fc_dec_1(x))
        x_fc2 = self.fc_dec_2(x_fc1)
        return x_fc2

class CNN(nn.Module):
    def __init__(self, enc_out_dim=8, latent_dim=2):
        super(CNN, self).__init__()
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=enc_out_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=enc_out_dim, out_channels=enc_out_dim*2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(in_features=enc_out_dim*2*7*7, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=enc_out_dim*2*7*7, out_features=latent_dim)
        self.relu_enc1 = nn.ReLU()
        self.relu_enc2 = nn.ReLU()
        self.fc_dec = nn.Linear(latent_dim, enc_out_dim*2*7*7)
        self.convtr1 = nn.ConvTranspose2d(in_channels=enc_out_dim*2, out_channels=enc_out_dim, kernel_size=4, stride=2, padding=1)
        self.convtr2 = nn.ConvTranspose2d(in_channels=enc_out_dim, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.relu_dec = nn.ReLU()
    
    def encoder(self, x):
        x1 = self.relu_enc1(self.conv1(x))
        x2 = self.relu_enc2(self.conv2(x1))
        x2 = x2.view(x2.shape[0], -1)
        x_mu = self.fc_mu(x2)
        x_logvar = self.fc_logvar(x2)
        return x_mu, x_logvar

    def decoder(self, x):
        x_fc = self.fc_dec(x)
        x_fc = x_fc.view(x.shape[0], self.enc_out_dim*2, 7, 7)
        x_convtr1 = self.relu_dec(self.convtr1(x_fc))
        x_convtr2 = self.convtr2(x_convtr1)
        return x_convtr2

class VAE(pl.LightningModule):
    def __init__(self, input_dim, enc_out_dim=64, latent_dim=2, learning_rate=1e-3, model_type='MLP') -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.name = "lt-VAE"
        if model_type == 'MLP':
            self.model = MLP(input_dim=input_dim, enc_out_dim=enc_out_dim, latent_dim=latent_dim)
        elif model_type == 'CNN':
            self.model = CNN(enc_out_dim=enc_out_dim, latent_dim=latent_dim)
            # self.encoder = CNNEncoder(enc_out_dim=enc_out_dim, latent_dim=latent_dim)
            # self.decoder = CNNDecoder(enc_out_dim=enc_out_dim, latent_dim=latent_dim)
        else:
            raise ValueError("model_type must be MLP or CNN")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def kl_divergence(self, mu, std):
        p_z = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q_z = torch.distributions.Normal(mu, std)
        kl_div = torch.distributions.kl_divergence(q_z, p_z).sum(dim=-1).mean()
        # kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_div
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        if self.model_type == 'MLP':
            x = x.view(-1, self.input_dim)
        x_logits, x_hat, mu, std, z = self.forward(x)
        if self.model_type == 'CNN':
            x_logits = x_logits.view(x_logits.shape[0], -1)
            x = x.view(x.shape[0], -1)
        recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').sum(dim=-1).mean()
        # recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim))
        # recon_loss = F.mse_loss(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='none').sum(dim=1).mean()
        kl = self.kl_divergence(mu, std)
        elbo = kl + recon_loss
        self.log_dict({'elbo': elbo, 'kl': kl, 'recon_loss': recon_loss})
        if (self.current_epoch % 5) == 0 and (batch_idx == 0):
            ic(elbo, kl, recon_loss)
            self.log_images(x, x_hat, z, stage='train')
        return elbo
    
    def test_step(self, batch, batch_idx):
        x, _ = batch
        if self.model_type == 'MLP':
            x = x.view(-1, self.input_dim)
        x_logits, x_hat, mu, std, z = self.forward(x)
        if self.model_type == 'CNN':
            x_logits = x_logits.view(x_logits.shape[0], -1)
            x = x.view(x.shape[0], -1)
        recon_loss = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').sum(dim=-1).mean()
        # recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim))
        # recon_loss = F.mse_loss(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim))
        kl = self.kl_divergence(mu, std)
        elbo = kl + recon_loss
        self.log_dict({'test_elbo': elbo, 'test_kl': kl, 'test_recon_loss': recon_loss})
        if (self.current_epoch % 5) == 0 and (batch_idx == 0):
            ic(elbo, kl, recon_loss)
            self.log_images(x, x_hat, z, stage='test')
        return elbo

    def training_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        self.log_dict({"loss/train": total_loss})

    def test_epoch_end(self, outputs):
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        self.log_dict({"loss/test": total_loss})
    
    def forward(self, x):
        mu, logvar = self.model.encoder(x)
        std = torch.sqrt(torch.exp(logvar))
        z = mu + std * torch.empty_like(std).normal_()
        x_logits = self.model.decoder(z)
        x_hat = torch.sigmoid(x_logits)
        return x_logits, x_hat, mu, std, z
    
    def log_images(self, x, x_hat, z, stage):
        z_img = make_grid(z[:50].view(-1, 1, 1, self.latent_dim))
        self.logger.experiment.add_image(f'{stage}_latent_epoch{self.current_epoch}', z_img, self.global_step)
        x_img = make_grid(x[:50].reshape(-1, 1, 28, 28))
        x_hat_img = make_grid(x_hat[:50].reshape(-1, 1, 28, 28))
        self.logger.experiment.add_images(f'{stage}_images_epoch{self.current_epoch}', torch.stack((x_img, x_hat_img)), self.global_step)
