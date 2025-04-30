from lightning.pytorch.core.datamodule import LightningDataModule
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scripts.dataset import TorchDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class LightningDataset(LightningDataModule):
    def __init__(self, x_train, y_train, x_test, y_test, cfg):
        super().__init__()
        
        self.cfg = cfg
        
        # Split and preprocess data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=self.cfg.seed)
        
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self._scale_and_standardize(x_train, x_val, x_test, y_train, y_val, y_test)
    
    def setup(self, stage=None):
        self.train_dataset = TorchDataset(self.x_train, self.y_train, **self.cfg.model.augmentations)
        self.val_dataset = TorchDataset(self.x_val, self.y_val)
        self.test_dataset = TorchDataset(self.x_test, self.y_test)
    
    def _scale_and_standardize(self, x_train, x_val, x_test, y_train, y_val, y_test):
        # Ensure inputs are as expected
        if not isinstance(x_train, pd.DataFrame) or not isinstance(x_val, pd.DataFrame) or not isinstance(x_test, pd.DataFrame):
            raise ValueError("x_train, x_val, and x_test must be pandas DataFrame objects.")
        if not isinstance(y_train, pd.Series) or not isinstance(y_val, pd.Series) or not isinstance(y_test, pd.Series):
            raise ValueError("y_train, y_val, and y_test must be pandas Series objects.")
        
        if self.cfg.dataset.standardization == "no_x_normalization_y":
            
            scaler = StandardScaler()
            y_train_scaled = pd.Series(scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
            self.y_scaler = scaler
            y_val_scaled = pd.Series(scaler.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
            y_test_scaled = pd.Series(scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
            
            return x_train, x_val, x_test, y_train_scaled, y_val_scaled, y_test_scaled
        
        elif self.cfg.dataset.standardization == "normalization":
            scaler = MinMaxScaler()
            
            x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
            x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
            
            
            scaler = StandardScaler()
            y_train_scaled = pd.Series(scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
            self.y_scaler = scaler
            y_val_scaled = pd.Series(scaler.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
            y_test_scaled = pd.Series(scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
            
            return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
            
        elif self.cfg.dataset.standardization == "min_max":
            scaler = MinMaxScaler()
            
            x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
            x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
            
            
            scaler = MinMaxScaler()
            y_train_scaled = pd.Series(scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
            self.y_scaler = scaler
            y_val_scaled = pd.Series(scaler.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
            y_test_scaled = pd.Series(scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
            
            return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
            
        elif self.cfg.dataset.standardization == "min_max_authors_bad":
            scaler = MinMaxScaler()
            
            scaler.fit(pd.concat([x_train, x_val, x_test]))
            x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns, index=x_train.index)
            x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns, index=x_val.index)
            x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
            
            
            scaler = MinMaxScaler()
            scaler.fit(np.concatenate([y_train.values.reshape(-1, 1), y_val.values.reshape(-1, 1), y_test.values.reshape(-1, 1)], axis=0))
            self.y_scaler = scaler
            y_train_scaled = pd.Series(scaler.transform(y_train.values.reshape(-1, 1)).flatten(), index=y_train.index)
            y_val_scaled = pd.Series(scaler.transform(y_val.values.reshape(-1, 1)).flatten(), index=y_val.index)
            y_test_scaled = pd.Series(scaler.transform(y_test.values.reshape(-1, 1)).flatten(), index=y_test.index)
            
            return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
            
        else:
            raise NotImplementedError(f"scaling for {self.cfg.dataset.standardization} not yet implemented.")
        
    def num_features(self):        
        return self.x_train.shape[1]
    
    def target_size(self):
        return 1 if isinstance(self.y_train, pd.Series) else self.y_train.shape[1]
    
    def get_y_scaler(self):
        return self.y_scaler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.model.batch_size, num_workers=self.cfg.machine.workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.model.batch_size, num_workers=self.cfg.machine.workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.model.batch_size, num_workers=self.cfg.machine.workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.model.batch_size, num_workers=self.cfg.machine.workers)
