from typing import Union
from lightning import LightningModule
import numpy as np
from omegaconf import DictConfig
import torch
from torchmetrics import AUROC, Accuracy
import torch.nn.functional as F
from scripts.model import DynamicMLP, SimpleAutoEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clearml import Logger


class MLPModel(LightningModule):
    def __init__(self, num_features, model_cfg, clearml_logger: Logger, output_dim: int, val_fold_id=None, test_fold_id=None):
        super().__init__()
        self.model_cfg = model_cfg
        
        # the clearml logger
        self.clearml_logger = clearml_logger
        
        # keep track of the kfold
        self.val_fold_id = val_fold_id
        self.is_val_kfold = self.val_fold_id is not None
        
        # keep track of the kfold
        self.test_fold_id = test_fold_id
        
        # Initialize your MLP model here...
        self.model = DynamicMLP(num_features, self.model_cfg.first_layer_dim, self.model_cfg.num_layers, output_dim, self.model_cfg.batch_norm, self.model_cfg.dropout_prob)
        
        # For accumulating loss
        self.train_epoch_loss = []
        self.val_epoch_loss = []
        self.test_epoch_loss = []
        
        # Loss
        self.loss_function = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        
        self.update_loss('train', loss.item(), y, y_hat)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss)
        
        self.update_loss('val', loss.item(), y, y_hat)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        
        self.log('test_loss', loss)
        
        self.update_loss('test', loss.item(), y, y_hat)
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch  # This unpacks the tuple and ignores y
        return self(x), y
    
    def log_metrics(self, phase, loss):
        self.clearml_logger.report_scalar(
            title=f"Loss Test fold {self.test_fold_id:02d}" + (f" - Validation fold {self.val_fold_id:02d}" if self.is_val_kfold else ""),
            series=phase,
            value=loss,
            iteration=self.current_epoch
        )
    
    def update_loss(self, phase, loss, y=None, y_hat=None):
        if phase == "train":
            self.train_epoch_loss.append(loss)
        elif phase == "val":
            self.val_epoch_loss.append(loss)
        elif phase == "test":
            self.test_epoch_loss.append(loss)
        else:
            raise Exception("Phase not defined")
            
    def reset_loss(self, phase):
        if phase == "train":
            self.train_epoch_loss = []
        elif phase == "val":
            self.val_epoch_loss = []
        elif phase == "test":
            self.test_epoch_loss = []
        else:
            raise Exception("Phase not defined")

    def configure_optimizers(self):
        if self.model_cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.model_cfg.learning_rate, weight_decay=self.model_cfg.weight_decay)
        elif self.model_cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.model_cfg.learning_rate, weight_decay=self.model_cfg.weight_decay)
        else:
            raise Exception("Optimizer not defined")
        
        if "lr_scheduler" not in self.model_cfg.keys() or self.model_cfg.lr_scheduler == "none":
            return optimizer
        elif self.model_cfg.lr_scheduler == "plateau":
            lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, eps=0.0005, verbose=True)
        else:
            raise Exception("LR Scheduler not defined")
        
        return {"optimizer": optimizer,
                "lr_scheduler": lr_scheduler,
                "monitor": "val_loss"}
        
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        
        avg_train_loss = np.mean(self.train_epoch_loss)

        self.log_metrics("Train", avg_train_loss) # self.train_mse_per_feature)
        self.reset_loss('train')

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
        avg_val_loss = np.mean(self.val_epoch_loss)

        self.log_metrics("Validation" if not self.is_val_kfold else "Development", avg_val_loss) # self.val_mse_per_feature)
        self.reset_loss('val')
        
    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        
        avg_test_loss = np.mean(self.test_epoch_loss)

        self.log_metrics("Test" if not self.is_val_kfold else "Validation", avg_test_loss)
        self.reset_loss('test')


# old modules
class LitAutoEncoder(LightningModule):
    def __init__(self, model_cfg: DictConfig, input_shape: int):
        super().__init__()

        self.save_hyperparameters()

        self.model_cfg = model_cfg

        self.auto_encoder = SimpleAutoEncoder(self.model_cfg, input_shape=input_shape)

        if self.model_cfg.loss == "mse":
            self.metric = torch.nn.MSELoss()
        elif self.model_cfg.loss == "ce":
            self.metric = torch.nn.BCEWithLogitsLoss()
        else:   
            raise Exception("other losses to be defined yet")

    def forward(self, x):
        return self.auto_encoder(x)
    
    def encode(self, x):
        return self.auto_encoder.encoder(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.model_cfg.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        self.log("val_loss", loss)

        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch

        x_hat = self(x)

        loss = self.metric(x_hat, x)
        self.log("test_loss", loss)

        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        x_hat = self(x)

        return x_hat

class LitEncoderOnly(LitAutoEncoder):
    def forward(self, x):
        return self.encode(x)
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch

        x_hat = self(x)

        return x_hat, y
