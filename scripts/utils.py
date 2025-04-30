import os
from os.path import join
import time
import clearml
import joblib
import numpy as np
from omegaconf import DictConfig
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

def get_data(dataset_cfg: DictConfig):
    assert dataset_cfg.name in ["Dataset split", "Dataset Filtered", "Dataset Upload (Data and Targets)"], "Dataset name not recognized"
    
    strat_time = time.time()
    print("Downloading data...")
    strat_time = time.time()
    dataset_path = get_dataset_instance(dataset_cfg=dataset_cfg, alias="input_dataset").get_local_copy()
    print("Time to download data:", time.time() - strat_time)
    print(" - Dataset path:", dataset_path, end="\n\n")
    
    if dataset_cfg.name == "Dataset split":
        print("Reading HDF files...")
        X_train = pd.read_hdf(join(dataset_path, "x_train.h5"), key="data")
        X_test = pd.read_hdf(join(dataset_path, "x_test.h5"), key="data")
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        print("Reading np arrays...")
        strat_time = time.time()
        y_train = np.load(join(dataset_path, "y_train.npy"))
        y_test = np.load(join(dataset_path, "y_test.npy"))        
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        
        print("Loading Scaler...")
        strat_time = time.time()
        y_min_max_scaler = joblib.load(join(dataset_path, "y_min_max_scaler.pkl"))
        print(" - Loaded in:", time.time() - strat_time, end="\n\n")
    
        return X_train, y_train, X_test, y_test, y_min_max_scaler
    elif dataset_cfg.name == "Dataset Filtered":
        print("Reading CSV file...")
        df = pd.read_csv(join(dataset_path, "NAM_dat(dataset).csv"), index_col=0)
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        return df
    elif dataset_cfg.name == "Dataset Upload (Data and Targets)":
        print("Reading HDF file...")
        x_df = pd.read_hdf(join(dataset_path, "x_df.h5"), key="data")
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        print("Reading targets CSV...")
        strat_time = time.time()
        targets_df = pd.read_csv(join(dataset_path, "targets_df.csv"), index_col=0)
        
        print(" - Read in:", time.time() - strat_time, end="\n\n")
        
        return x_df, targets_df


def get_dataset_instance(dataset_cfg: DictConfig, alias=None):
    if "id" in dataset_cfg.keys() and dataset_cfg.id is not None:
        return clearml.Dataset.get(dataset_cfg.id, alias=alias)
    else:
        return clearml.Dataset.get(dataset_name=dataset_cfg.name, dataset_project=dataset_cfg.project, alias=alias)


# Data processing
def data_scaling(dataset,target): 
    # Scaled data
    min_max_scaler = MinMaxScaler(feature_range = (0,1))
    np_scaled = min_max_scaler.fit_transform(dataset)
    X = pd.DataFrame(np_scaled)
    
    target_edit = pd.Series(target).values
    target_edit = target_edit.reshape(-1,1)
    np_scaled = min_max_scaler.fit_transform(target_edit)
    Y = pd.DataFrame(np_scaled)
    
    return X, Y, min_max_scaler

def inverse_scaling(data,scaler=None):
    if scaler is not None:
        if isinstance(data, pd.Series):
            data = data.values
        data = data.reshape(-1,1)
        data = scaler.inverse_transform(data).flatten()
    return data