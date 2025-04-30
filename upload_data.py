import json
import os
from os.path import join
import shutil
from clearml import Dataset, Task
import clearml
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from scripts.clearml_utils import connect_confiuration, set_task_execution, setting_up_task
from scripts.utils import get_data, get_dataset_instance


os.environ["HDF5_USE_FILE_LOCKING"]="FALSE"

def clear_dir_content(dir_path):
    # Remove the directory and its contents, then recreate the directory
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def plot_target_density(vector_a, task: clearml.Task, title, plot_title, vector_b=None, iter=0):
    
    if vector_b is None:
        df = pd.DataFrame({'Value': vector_a, 'Split': 'All'})
    else:
        # Combining the vectors into a DataFrame
        df_a = pd.DataFrame({'Value': vector_a, 'Split': 'Train'})
        df_b = pd.DataFrame({'Value': vector_b, 'Split': 'Test'})
        df = pd.concat([df_a, df_b], ignore_index=True)
        
    # Creating the overlaid density plots
    sns.kdeplot(data=df, x="Value", hue="Split", fill=True, common_norm=False, alpha=0.5)
    plt.title(plot_title)
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    plt.tight_layout()
    
    task.get_logger().report_matplotlib_figure(title, title, figure=plt.gcf(), iteration=iter)
    plt.close()

@hydra.main(config_path="conf", config_name="upload_data_config", version_base=None)
def main(cfg: DictConfig):
    
    task: Task = setting_up_task(cfg.clearml_task)    
    
    cfg = connect_confiuration(clearml_task=task, configuration=cfg)
    print(OmegaConf.to_yaml(cfg))
    
    # original data
    if "comment" in cfg.clearml_task.keys():
        task.set_comment(cfg.clearml_task.comment)
    
    set_task_execution(task_cfg=cfg.clearml_task, machine_cfg=cfg.machine, clearml_task=task)
    
    data_df = get_data(dataset_cfg=cfg.input_dataset)
    
    data_dir_to_upload = join("tmp", "data_to_upload_to_clearml")
    # ==============================
    # SECTION: Data and target
    # ==============================
    
    data_df = data_df.set_index("Row.names")
    
    if "columns_to_drop" in cfg.keys():
        data_df = data_df.drop(columns=cfg.columns_to_drop)
    
    x_df = data_df.drop(columns=cfg.target_columns)
    targets_df = data_df.loc[:, cfg.target_columns]
    
    # Remove the directory and its contents, then recreate the directory
    clear_dir_content(data_dir_to_upload)
    
    x_df.to_hdf(join(data_dir_to_upload, 'x_df.h5'), key="data", mode='w')
    targets_df.to_csv(join(data_dir_to_upload, 'targets_df.csv'))
    
    dataset = Dataset.create(dataset_name=cfg.output_dataset.name, dataset_project=cfg.output_dataset.project, use_current_task=True, 
                             parent_datasets=[get_dataset_instance(dataset_cfg=cfg.input_dataset)])

    removed_files = []
    for file in dataset.list_files():
        removed = dataset.remove_files(file, verbose=True)
        if removed:
            removed_files.append(file)
    print(f"Removed {removed_files} from the dataset")

    dataset.add_files(data_dir_to_upload)
    
    dataset.upload()
    dataset.finalize()
    
    task.get_logger().report_table(title='X Data',series='X Data',iteration=0, table_plot=x_df.iloc[:15, :10])
    
    task.get_logger().report_table(title='Targets Statistics',series='Targets Statistics',iteration=0, table_plot=targets_df.describe())    
    
    for i in range(5):
        plot_target_density(targets_df.iloc[:, i].values, task, "Targets Density Plots", f"{targets_df.columns[i]} Density", iter=i)
        task.get_logger().report_table(title='Targets Data',series=f'{targets_df.columns[i]} Data',iteration=i, table_plot=targets_df.iloc[i, :15].to_frame())
    
    task.close()

    
if __name__ == "__main__":
    main()