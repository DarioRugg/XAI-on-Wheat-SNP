import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TorchDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series, apply_masking=False, apply_noising=False, mask_prob=0.1, noise_prob=0.05):
        self.X = X
        self.y = y
        self.apply_masking = apply_masking
        self.apply_noising = apply_noising
        self.mask_prob = mask_prob
        self.noise_prob = noise_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X.iloc[idx].values
        y = self.y.iloc[idx]

        if self.apply_masking:
            x = self.random_masking(x)

        if self.apply_noising:
            x = self.feature_noising(x)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).unsqueeze(0)

    def random_masking(self, x):
        mask = np.random.rand(x.shape[0]) < self.mask_prob
        x[mask] = 0
        return x

    def feature_noising(self, x):
        noise = np.random.rand(x.shape[0]) < self.noise_prob
        x[noise] = 1 - x[noise]
        return x
    