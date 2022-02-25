import torch
import torch.nn
from utils import MNISTDataModule
from models import VAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import ProgressBar
import optuna
from torch.utils.tensorboard import SummaryWriter
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
import time
from icecream import ic





# LOG_PATH = '../models/'
# def objective(trial: optuna.trial.Trial) -> float:
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 0.1)
#     # latent_dim = trial.suggest_int('latent_dim', 2, 100)
#     enc_out_dim = trial.suggest_categorical('enc_out_dim', [1, 2, 4, 8, 16, 32, 64, 128])
#     # batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024])

#     # datamodule = MNISTDataModule(batch_size=batch_size)
#     datamodule = MNISTDataModule(batch_size=BATCH_SIZE)
#     datamodule.setup(stage="fit")
#     model = VAE(input_dim=INPUT_DIM, enc_out_dim=enc_out_dim, latent_dim=LATENT_DIM, learning_rate = learning_rate)
#     logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
#     trainer = pl.Trainer(default_root_dir=LOG_PATH, logger=logger, max_epochs=MAX_EPOCHS,
#                 callbacks=[PyTorchLightningPruningCallback(trial, monitor="elbo"), ProgressBar(refresh_rate=100)], gpus=1 if torch.cuda.is_available() else None)
#     hyperparameters = dict(learning_rate=learning_rate, enc_out_dim=enc_out_dim)
#     trainer.logger.log_hyperparams(hyperparameters)
#     trainer.fit(model, datamodule=datamodule)
#     return trainer.callback_metrics["elbo"].item()

# pruner = optuna.pruners.MedianPruner()
# study = optuna.create_study(direction="minimize", pruner=pruner)
# study.optimize(objective, n_trials=20, timeout=60000)
# print("Number of finished trials: {}".format(len(study.trials)))
# best = study.best_trial
# print(f"Best trial: {best}")
# print("Value: {}".format(best.value))
# print("Params: ")
# for key, value in best.params.items():
#     print("{}: {}".format(key, value))


if __name__ =='__main__':
    BATCH_SIZE = 128
    INPUT_DIM = 28*28
    LATENT_DIM = 2
    LEARNING_RATE = 1e-4
    MAX_EPOCHS = 30
    ENC_OUT_DIM = 16 #Par exemple 64 pour LinearModel, 16 pour CNN
    MODEL_TYPE = 'CNN'
    SAVE_PATH = "../models/"

    datamodule = MNISTDataModule(batch_size=BATCH_SIZE)
    LATENT_DIMS = [2, 16]
    MODEL_TYPES = ['MLP', 'CNN']
    for latent_dim in LATENT_DIMS:
        for model_type in MODEL_TYPES:
            ic(model_type, latent_dim)
            model = VAE(input_dim=INPUT_DIM, enc_out_dim=ENC_OUT_DIM, latent_dim=latent_dim, learning_rate = LEARNING_RATE, model_type=model_type)
            trainer = pl.Trainer(default_root_dir=SAVE_PATH, max_epochs=MAX_EPOCHS, callbacks=[ProgressBar(refresh_rate=500)],
                                    gpus=1 if torch.cuda.is_available() else None)
            trainer.fit(model, datamodule=datamodule)