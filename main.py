import io
from clearml import Task
import hydra
from matplotlib import colors, pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import KFold
from scripts.clearml_utils import connect_confiuration, get_best_params_from_last_hpo, setting_up_task, set_task_execution
from scripts.explainability_utils import Explainer
from scripts.training_testing_scripts import train_and_test_mlp_model
from scripts.utils import get_data
import seaborn as sns

sns.set_theme(style="whitegrid") 


@hydra.main(config_path="conf", config_name="main_config", version_base=None)
def main(cfg: DictConfig):
    
    if "tuned_task" in cfg.keys():
        get_best_params_from_last_hpo(cfg)
    
    task: Task = setting_up_task(cfg.clearml_task)
    
    if "tuned_task" in cfg.keys():
        task.add_tags("Tuned")
    
    cfg = connect_confiuration(clearml_task=task, configuration=cfg)
    print(OmegaConf.to_yaml(cfg))
    
    task.add_tags(cfg.target_column)
    
    assert cfg.cross_validation.validation.evaluate or cfg.cross_validation.test.evaluate, AssertionError("this execution would do nothing")
    
    set_task_execution(task_cfg=cfg.clearml_task, machine_cfg=cfg.machine, clearml_task=task)
    
    x_df, targets_df = get_data(dataset_cfg=cfg.dataset)
    
    y_series = targets_df.loc[:, cfg.target_column]
    
    test_kfold = KFold(n_splits=10, shuffle=True, random_state=cfg.seed)
    
    explainer = Explainer(task) if cfg.cross_validation.test.evaluate else None
    
    test_kfold_results_list = []
    all_kfold_results = pd.DataFrame()
    
    for test_fold_idx, (train_idx, test_idx) in enumerate(test_kfold.split(x_df, y_series)):
        x_train_df, x_test_df = x_df.iloc[train_idx], x_df.iloc[test_idx]
        y_train_series, y_test_series = y_series.iloc[train_idx], y_series.iloc[test_idx]

        if cfg.cross_validation.validation.evaluate:
            
            if "first_only" not in cfg.cross_validation.validation.keys() or not cfg.cross_validation.validation.first_only or (cfg.cross_validation.validation.first_only and test_fold_idx == 0):
            
                validation_kfold = KFold(n_splits=10, shuffle=True, random_state=cfg.seed)
                
                kfold_results_list = []
                # Iterate over each fold
                for validation_fold_index, (kfold_train_idx, kfold_val_idx) in enumerate(validation_kfold.split(x_train_df, y_train_series)):
                    x_kfold_train_df, x_kfold_val_df = x_train_df.iloc[kfold_train_idx], x_train_df.iloc[kfold_val_idx]
                    y_kfold_train_series, y_kfold_val_series = y_train_series.iloc[kfold_train_idx], y_train_series.iloc[kfold_val_idx]
                    
                    test_results_dict = train_and_test_mlp_model(task, x_kfold_train_df, y_kfold_train_series, x_kfold_val_df, y_kfold_val_series, cfg, test_fold_id=test_fold_idx, val_fold_id=validation_fold_index)
                    kfold_results_list.append(pd.DataFrame(test_results_dict|{'fold': validation_fold_index}, index=[validation_fold_index]))
                
                kfold_results = pd.concat(kfold_results_list)
                # report metrics for HPO
                for metric, value in kfold_results.drop(columns="fold").mean().items():
                    task.get_logger().report_scalar(title="Mean Kfold Results", series=str(metric).replace(" ", "_")+f"_test_fold_{test_fold_idx:02d}", value=value, iteration=0)
                
                kfold_results["test_fold_idx"] = test_fold_idx
                all_kfold_results = pd.concat([all_kfold_results, kfold_results])

        if cfg.cross_validation.test.evaluate:
            # evaluation on test
            test_results = train_and_test_mlp_model(task, x_train_df, y_train_series, x_test_df, y_test_series, cfg, explainer=explainer, test_fold_id=test_fold_idx)
            test_kfold_results_list.append(pd.DataFrame(test_results|{'fold': test_fold_idx}, index=[test_fold_idx]))
    
    # report results
    for stage, results_df in zip(["Validation", "Test"], [all_kfold_results, test_kfold_results_list]):
        if len(results_df) > 0:
            
            if isinstance(results_df, list):
                results_df = pd.concat(results_df)
            
            metrics_columns = results_df.drop(columns=["fold", "test_fold_idx"] if stage == "Validation" else "fold").columns
            
            # Upload the kfold results as an artifact
            task.upload_artifact(name=f"{stage}_kfold_results", artifact_object=results_df)
            
            # Report the detailed kfold results
            task.get_logger().report_table(title=f"{stage} - Kfold Results", series="Kfold Results", iteration=0, table_plot=results_df.set_index('fold'))
            
            # Create a summary DataFrame
            summary = pd.DataFrame({'Mean': results_df[metrics_columns].mean().round(5), 'Std Dev': results_df[metrics_columns].std().round(4)})
            # Report the summary table
            task.get_logger().report_table(title=f"{stage} - Kfold Summary", series="Kfold Summary", iteration=0, table_plot=summary)
            
            for i, metric_name in enumerate(metrics_columns):
                if stage == "Validation" and len(results_df["test_fold_idx"].unique()) > 1:
                    sns.histplot(results_df, x=metric_name, hue="test_fold_idx", palette="tab10")
                else:
                    sns.histplot(results_df, x=metric_name)
                plt.title(f'{stage} Kfold Results histograms - {metric_name}')
                # Report the figure to ClearML
                task.get_logger().report_matplotlib_figure(title=f"{stage} - Kfold Results histograms", series=metric_name, iteration=i, figure=plt)
                plt.clf()

    # Explainability
    if explainer is not None:
        explainer.final_plots()
        

if __name__ == "__main__":
    main()