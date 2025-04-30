

from clearml import InputModel, Task
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import shap
from sklearn.preprocessing import MinMaxScaler
import torch

from scripts.lightning_modules import MLPModel


class Explainer:
    def __init__(self, clearml_task: Task) -> None:
        self.task = clearml_task
        self.column_names = None
        
        self.shap_values_list = []
        self.samples_list = []
        
    def explainability_and_shap_values_from_checkpoint(self, model, x_train, x_test, test_fold_id):
        
        assert self.column_names is None or all(self.column_names == x_test.columns), "Columns have to be the same"
        self.column_names = x_test.columns
        
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        model.eval()
        model.to('cpu')  # Move model to CPU for SHAP compatibility

        # SHAP explanation
        background_data = torch.tensor(x_train).float()
        explainer = shap.DeepExplainer(model, background_data)
        test_data_sample = torch.tensor(x_test).float()
        shap_values = np.squeeze(explainer.shap_values(test_data_sample))

        shap.summary_plot(shap_values, test_data_sample.numpy(), feature_names=self.column_names, show=False)
        plt.xticks(rotation=45)
        plt.title(f"SHAP Summary Plot test fold {test_fold_id}")
        self.task.get_logger().report_matplotlib_figure(title='K-Fold SHAP Summary Plot', series='K-Fold SHAP Summary Plot', figure=plt, iteration=test_fold_id)
        plt.clf()
        
        shap.summary_plot(shap_values, test_data_sample.numpy(), feature_names=self.column_names, plot_type='bar', color='lightblue', show=False)
        plt.title(f"Feature inportance test fold {test_fold_id}")
        self.task.get_logger().report_matplotlib_figure(title='K-Fold Feature inportance', series='K-Fold Feature inportance', figure=plt, iteration=test_fold_id)
        plt.clf()
        
        self.shap_values_list.append(shap_values)
        self.samples_list.append(test_data_sample)
    
    def final_plots(self):
        
        shap_values = np.concatenate(self.shap_values_list, axis=0)
        samples = np.concatenate(self.samples_list, axis=0)
        
        shap.summary_plot(shap_values, samples, feature_names=self.column_names, show=False)
        plt.xticks(rotation=45)
        plt.title(f"SHAP Summary Plot")
        self.task.get_logger().report_matplotlib_figure(title='SHAP Summary Plot', series='SHAP Summary Plot', figure=plt, iteration=0)
        plt.clf()
        
        shap.summary_plot(shap_values, samples, feature_names=self.column_names, plot_type='bar', color='tab:orange', show=False)
        plt.title(f"Feature inportance")
        self.task.get_logger().report_matplotlib_figure(title='Feature inportance', series='Feature inportance', figure=plt, iteration=0)
        plt.clf()
        
        shap_values_df = pd.DataFrame(shap_values, columns=self.column_names)
        shap_values_df.abs().sum().hist(log=True, bins=30, color='lightgreen', label="Features distribution")
        plt.title("Features SHAP Values Histogram")
        plt.xlabel("Sum of SHAP Values")
        plt.ylabel("Frequency")
        self.task.get_logger().report_matplotlib_figure(title="Histogram", series="Histogram", figure=plt, iteration=0)
        plt.clf()