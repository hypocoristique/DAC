import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split,TensorDataset
from pathlib import Path
from datamaestro import prepare_dataset
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torch.distributions import Categorical
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from icecream import ic

BATCH_SIZE = 300
TRAIN_RATIO = 0.05
LEARNING_RATE = 0.0033
LOG_PATH = "./runs/lightning_logs"

LATENT_DIM = 100
NUM_CLASSES = 10
MAX_EPOCHS = 100
DROPOUT = 0.528 #Turn to 0. for BatchNorm or LayerNorm
REGULARIZATION = 'L2' # 'L1' or 'L2'
REG_LAMBDA = 0.00093 #Regularization parameter
BATCHNORM = False #Turn to False for LayerNorm
LAYERNORM = True


class LitMLP(pl.LightningModule):
    def __init__(self, input_dim, latent_dim, output_dim, regularization=None, reg_lambda=0.01, dropout=0., batchnorm=False, layernorm=False, learning_rate=1e-3, max_epochs=1000):
        super().__init__()
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.max_epochs = max_epochs
        self.name = "MLP-lightning"
        self.acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.linear1 = nn.Linear(input_dim, input_dim//2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(input_dim//2, input_dim//4)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(input_dim//4, 100)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(latent_dim, output_dim)
        self.grads = {}
        if dropout != 0.:
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        if batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(input_dim//2)
            self.batchnorm2 = nn.BatchNorm1d(input_dim//4)
            self.batchnorm3 = nn.BatchNorm1d(100)
        if layernorm:
            self.layernorm1 = nn.LayerNorm(input_dim//2)
            self.layernorm2 = nn.LayerNorm(input_dim//4)
            self.layernorm3 = nn.LayerNorm(100)
    
    def forward(self, x):
        if self.dropout != 0.:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            d1 = self.dropout1(a1)
            x2 = self.linear2(d1)
            a2 = self.relu2(x2)
            d2 = self.dropout2(a2)
            x3 = self.linear3(d2)
            a3 = self.relu3(x3)
            d3 = self.dropout3(a3)
            out = self.linear4(d3)
        elif self.batchnorm:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            b1 = self.batchnorm1(a1)
            x2 = self.linear2(b1)
            a2 = self.relu2(x2)
            b2 = self.batchnorm2(a2)
            x3 = self.linear3(b2)
            a3 = self.relu3(x3)
            b3 = self.batchnorm3(a3)
            out = self.linear4(b3)
        elif self.layernorm:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            l1 = self.layernorm1(a1)
            x2 = self.linear2(l1)
            a2 = self.relu2(x2)
            l2 = self.layernorm2(a2)
            x3 = self.linear3(l2)
            a3 = self.relu3(x3)
            l3 = self.layernorm3(a3)
            out = self.linear4(l3)
        else:
            x1 = self.linear1(x)
            a1 = self.relu1(x1)
            x2 = self.linear2(a1)
            a2 = self.relu2(x2)
            x3 = self.linear3(a2)
            a3 = self.relu3(x3)
            out = self.linear4(a3)
        return out

    def configure_optimizers(self):
        """ Définit l'optimiseur """
        if self.regularization == 'L2':
            optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate, weight_decay=self.reg_lambda)
        else:
            optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        """ une étape d'apprentissage
        doit retourner soit un scalaire (la loss),
        soit un dictionnaire qui contient au moins la clé 'loss'"""
        x, y = batch
        logits = self(x) ## equivalent à self.model(x)
        loss = self.loss(logits, y)
        if self.regularization == 'L1':
            l1_penalty = sum(p.abs().sum() for p in self.parameters())
            loss = loss + self.reg_lambda * l1_penalty
        acc = self.acc(logits, y)
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        if self.current_epoch % (self.max_epochs // 20) == 0:
            self.entropy_histogram(logits)
        return logs

    def validation_step(self, batch, batch_idx):
        """ une étape de validation
        doit retourner un dictionnaire"""
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.val_acc(logits, y)
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        self.log("val_accuracy", acc, on_step=False,on_epoch=True)
        return logs

    def test_step(self,batch, batch_idx):
        """ une étape de test """
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        acc = self.test_acc(logits, y)
        logs = {"loss": loss, "accuracy": acc, "nb": len(x)}
        return logs

    def training_epoch_end(self, outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque d'apprentissage.
        Par exemple ici calculer des valeurs à logger"""
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_acc = total_acc / total_nb
        self.log_dict({"loss/train": total_loss, "acc/train": total_acc})
        # Le logger de tensorboard est accessible directement avec self.logger.experiment.add_XXX

    def validation_epoch_end(self, outputs):
        """ hook optionel, si on a besoin de faire quelque chose apres une époque de validation."""
        total_acc = sum([o['accuracy'] for o in outputs])
        total_nb = sum([o['nb'] for o in outputs])
        total_loss = sum([o['loss'] for o in outputs]) / len(outputs)
        total_acc = total_acc / total_nb
        self.log_dict({"loss/val": total_loss, "acc/val": total_acc})

    def test_epoch_end(self, outputs):
        pass

    def entropy_histogram(self, logits):
        random_logits = torch.randn_like(logits)
        self.logger.experiment.add_histogram('entropy_output', Categorical(logits=logits).entropy(), global_step=self.current_epoch)
        self.logger.experiment.add_histogram('entropy_random_output', Categorical(logits=random_logits).entropy(), global_step=self.current_epoch)
    
    def on_after_backward(self):
        if self.current_epoch % (self.max_epochs // 20) == 0:
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, self.current_epoch)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f'{name}_grad', param.grad, self.current_epoch)


class LitMnistData(pl.LightningDataModule):

    def __init__(self,batch_size=BATCH_SIZE,train_ratio=TRAIN_RATIO):
        super().__init__()
        self.dim_in = None
        self.dim_out = None
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def prepare_data(self):
        ### Do not use "self" here.
        prepare_dataset("com.lecun.mnist")

    def setup(self,stage=None):
        ds = prepare_dataset("com.lecun.mnist")
        if stage =="fit" or stage is None:
            # Si on est en phase d'apprentissage
            shape = ds.train.images.data().shape
            self.dim_in = shape[1]*shape[2]
            self.dim_out = len(set(ds.train.labels.data()))
            ds_train = TensorDataset(torch.tensor(ds.train.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.train.labels.data()).long())
            train_length = int(shape[0]*self.train_ratio)
            self.mnist_train, self.mnist_val, = random_split(ds_train,[train_length,shape[0]-train_length])
        if stage == "test" or stage is None:
            # en phase de test
            self.mnist_test= TensorDataset(torch.tensor(ds.test.images.data()).view(-1,self.dim_in).float()/255., torch.tensor(ds.test.labels.data()).long())

    def train_dataloader(self):
        return DataLoader(self.mnist_train,batch_size=self.batch_size)
    def val_dataloader(self):
        return DataLoader(self.mnist_val,batch_size=self.batch_size)
    def test_dataloader(self):
        return DataLoader(self.mnist_test,batch_size=self.batch_size)

    def input_dim(self):
        return self.dim_in
    def output_dim(self):
        return self.dim_out


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    dropout = trial.suggest_float('dropout', low=0.2, high=0.6)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0001, 0.1)
    
    datamodule = LitMnistData()
    datamodule.setup(stage="fit")
    model = LitMLP(input_dim=datamodule.dim_in, latent_dim=LATENT_DIM, output_dim=datamodule.dim_out, regularization=REGULARIZATION,
                    reg_lambda=reg_lambda, dropout=dropout, batchnorm=False, layernorm=False, learning_rate=learning_rate, max_epochs=MAX_EPOCHS)
    logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)
    trainer = pl.Trainer(default_root_dir=LOG_PATH, logger=logger, max_epochs=MAX_EPOCHS,
                callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy")], gpus=1 if torch.cuda.is_available() else None)
    hyperparameters = dict(learning_rate=learning_rate, dropout=dropout, reg_lambda=reg_lambda)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)
    return trainer.callback_metrics["val_accuracy"].item()

pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=10, timeout=6000)
print("Number of finished trials: {}".format(len(study.trials)))
best = study.best_trial
print(f"Best trial: {best}")
print("Value: {}".format(best.value))
print("Params: ")
for key, value in best.params.items():
    print("{}: {}".format(key, value))




# data = LitMnistData()

# data.prepare_data()
# data.setup(stage="fit")
# input_dim = data.input_dim()
# output_dim = data.output_dim()

# model = LitMLP(input_dim=input_dim, latent_dim=LATENT_DIM, output_dim=NUM_CLASSES, regularization=REGULARIZATION, reg_lambda=REG_LAMBDA, dropout=DROPOUT, batchnorm=BATCHNORM, layernorm=LAYERNORM, learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS)

# logger = TensorBoardLogger(save_dir=LOG_PATH, name=model.name, version=time.asctime(), default_hp_metric=False)

# trainer = pl.Trainer(default_root_dir=LOG_PATH, logger=logger, max_epochs=MAX_EPOCHS)
# trainer.fit(model, data)
# trainer.test(model, data)
