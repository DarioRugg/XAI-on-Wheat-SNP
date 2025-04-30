from clearml import Task
import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import shap
from sklearn.model_selection import KFold
from scripts.clearml_utils import connect_confiuration, get_project_path, setting_up_task, set_task_execution
from scripts.training_testing_scripts import explainability_and_shap_values_from_checkpoint
from scripts.utils import get_data
import seaborn as sns


sns.set_theme(style="whitegrid") 


@hydra.main(config_path="conf", config_name="explainability_config", version_base=None)
def main(cfg: DictConfig):

    task: Task = setting_up_task(cfg.clearml_task)
    
    cfg = connect_confiuration(clearml_task=task, configuration=cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if "id" not in cfg.base_task.keys() or cfg.base_task.id is None:
        base_task: Task = Task.get_task(project_name=get_project_path(cfg.base_task), task_name=cfg.base_task.task_name)
    else:
        base_task: Task = Task.get_task(task_id=cfg.base_task.id)
        
    base_task_cfg = OmegaConf.create(base_task.get_parameters_as_dict(True)["hydra_config"])
    
    task.add_tags(base_task_cfg.target_column)
    
    task.connect(OmegaConf.to_object(base_task_cfg), name="base_task_cfg")
    
    set_task_execution(task_cfg=cfg.clearml_task, machine_cfg=cfg.machine, clearml_task=task)
    
    x_df, targets_df = get_data(dataset_cfg=base_task_cfg.dataset)
    
    y_series = targets_df.loc[:, base_task_cfg.target_column]
    
    test_kfold = KFold(n_splits=10, shuffle=True, random_state=base_task_cfg.seed)
    
    shap_values_list = []
    samples_list = []
    col_names = None
    for test_fold_idx, (train_idx, test_idx) in enumerate(test_kfold.split(x_df, y_series)):
        x_train_df, x_test_df = x_df.iloc[train_idx], x_df.iloc[test_idx]
        y_train_series, y_test_series = y_series.iloc[train_idx], y_series.iloc[test_idx]
        
        shap_values, samples, cols = explainability_and_shap_values_from_checkpoint(clearml_task=task, base_task=base_task, base_task_model_cfg=base_task_cfg.model, x_train=x_train_df, y_train=y_train_series, x_test=x_test_df, y_test=y_test_series, test_fold_id=test_fold_idx)

        shap_values_list.append(shap_values)
        samples_list.append(samples)
        
        assert col_names is None or all(col_names == cols), "Columns have to be the same"
        col_names = cols
        
    shap_values = np.concatenate(shap_values_list, axis=0)
    samples = np.concatenate(samples_list, axis=0)
    
    shap.summary_plot(shap_values, samples, feature_names=col_names, show=False)
    plt.xticks(rotation=45)
    plt.title(f"SHAP Summary Plot")
    task.get_logger().report_matplotlib_figure(title='SHAP Summary Plot', series='SHAP Summary Plot', figure=plt, iteration=0)
    plt.clf()
    
    shap.summary_plot(shap_values, samples, feature_names=col_names, plot_type='bar', color='tab:orange', show=False)
    plt.title(f"Feature inportance")
    task.get_logger().report_matplotlib_figure(title='Feature inportance', series='Feature inportance', figure=plt, iteration=0)
    plt.clf()
    
    shap_values_df = pd.DataFrame(shap_values, columns=col_names)
    shap_values_df.abs().sum().hist(log=True, bins=30, color='lightgreen', label="Features distribution")
    plt.title("Features SHAP Values Histogram")
    plt.xlabel("Sum of SHAP Values")
    plt.ylabel("Frequency")
    task.get_logger().report_matplotlib_figure(title="Histogram", series="Histogram", figure=plt, iteration=0)
    plt.clf()

if __name__ == "__main__":
    main()