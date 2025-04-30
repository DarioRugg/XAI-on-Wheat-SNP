from typing import Union
from clearml import Task, OutputModel, InputModel
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from scripts.explainability_utils import Explainer
from scripts.lightning_modules import MLPModel
from scripts.lightning_datamodules import LightningDataset
import seaborn as sns

from scripts.utils import inverse_scaling


def train_and_test_mlp_model(clearml_task: Task, x_train, y_train, x_test, y_test, cfg, test_fold_id, explainer: Union[Explainer, None]=None, val_fold_id=None):
    
    datamodule = LightningDataset(x_train, y_train, x_test, y_test, cfg)

    # Instantiate the model
    model = MLPModel(num_features=datamodule.num_features(), output_dim=datamodule.target_size(), model_cfg=cfg.model, clearml_logger=clearml_task.get_logger(), val_fold_id=val_fold_id, test_fold_id=test_fold_id)
    
    # Early Stopping
    early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=10, min_delta=0.00005)
    
    # Checkpoint Callback
    model_name= f"best_model_test_fold_{test_fold_id}"
    if val_fold_id is not None:
        model_name += f"val_fold_{val_fold_id}"
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=model_name,
        save_top_k=1,
        verbose=False,
        monitor='val_loss',
        mode='min'
    )
    
    # Trainer
    trainer = Trainer(min_epochs=15, max_epochs=cfg.model.max_epochs, log_every_n_steps=1, 
                      accelerator="gpu", devices=1 if cfg.clearml_task.execution_type != "local" else [cfg.machine.gpu_queues_id], 
                      callbacks=[early_stop_callback, checkpoint_callback], 
                      enable_progress_bar=True if cfg.clearml_task.execution_type == "local" else False,
                      gradient_clip_val=1.)
    
    trainer.fit(model, datamodule=datamodule)
    
    model = MLPModel.load_from_checkpoint(checkpoint_callback.best_model_path, num_features=datamodule.num_features(), output_dim=datamodule.target_size(), model_cfg=cfg.model, clearml_logger=clearml_task.get_logger(), val_fold_id=val_fold_id, test_fold_id=test_fold_id)
    
    if explainer is not None:
        explainer.explainability_and_shap_values_from_checkpoint(model, x_train, x_test, test_fold_id)
    
    output_model = OutputModel(clearml_task, config_dict=OmegaConf.to_object(cfg.model), name=model_name, tags=["validation"] if val_fold_id is not None else ["test"])
    output_model.update_weights(checkpoint_callback.best_model_path)
    
    test_results = trainer.test(model, datamodule=datamodule)[0]

    # Renaming the keys
    test_results = {
        "MSE_loss" if k == "test_loss" else k : v
        for k, v in test_results.items()
    }
    
    # Initialize lists to collect batch outputs
    preds_list, truths_list = [], []

    # Perform prediction
    batch_outputs = trainer.predict(model, datamodule=datamodule)

    # Unpack predictions and ground truths from each batch
    for batch_output in batch_outputs:
        preds, truths = batch_output
        preds_list.append(preds)
        truths_list.append(truths)

    # Concatenate all batch outputs into a single tensor for preds and truths
    y_preds = torch.cat(preds_list, dim=0).numpy()
    y_truths = torch.cat(truths_list, dim=0).numpy()  

    # Scale back predictions and truths
    predictions_rescaled = inverse_scaling(y_preds, datamodule.get_y_scaler())
    ground_truths_rescaled = inverse_scaling(y_truths, datamodule.get_y_scaler())
    
    # add authors correlation
    test_results['Correlation Coefficient'] = np.corrcoef(predictions_rescaled, ground_truths_rescaled)[0, 1]
    
    # add root mean squared error
    test_results['RMSE'] = np.sqrt(np.mean((predictions_rescaled - ground_truths_rescaled) ** 2))

    # Create DataFrame for Seaborn plotting
    data = {
        'Value': np.concatenate([predictions_rescaled, ground_truths_rescaled]),
        'Type': ['Prediction'] * len(predictions_rescaled) + ['Ground Truth'] * len(ground_truths_rescaled),
        'Sample Index': list(range(len(predictions_rescaled))) * 2
    }
    df = pd.DataFrame(data)

    # Plotting using Seaborn
    plt.figure(figsize=(25, 8))
    sns.barplot(x='Sample Index', y='Value', hue='Type', data=df, hue_order=['Ground Truth', 'Prediction'], palette=["tab:blue", "tab:orange"])
    plt.title('Predictions vs Ground Truths Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Values')

    # Log the plot to ClearML
    clearml_task.get_logger().report_matplotlib_figure(title='KFold Predictions vs Ground Truths' if val_fold_id is not None else 'Predictions vs Ground Truths', series='Bar Plot', figure=plt, iteration=test_fold_id if val_fold_id is None else test_fold_id*10+val_fold_id, report_image=True)
    plt.clf()
    
    output_model.wait_for_uploads()
    return test_results


def explainability_and_shap_values_from_checkpoint(clearml_task: Task, base_task: Task, base_task_model_cfg: DictConfig, x_train, y_train, x_test, y_test, test_fold_id):
    
    column_names = x_test.columns
    
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    model_name= f"best_model_test_fold_{test_fold_id}"
    previous_task_model = InputModel(base_task.output_models_id[model_name], name="previous_task_"+model_name)
    
    checkpoint_path = previous_task_model.get_weights()
    
    model = MLPModel.load_from_checkpoint(
        checkpoint_path, 
        num_features=x_train.shape[1], 
        output_dim=y_train.shape[1] if len(y_train.shape) > 1 else 1, 
        model_cfg=base_task_model_cfg, 
        clearml_logger=clearml_task.get_logger(), 
        test_fold_id=test_fold_id
    )
    model.eval()
    model.to('cpu')  # Move model to CPU for SHAP compatibility

    # SHAP explanation
    background_data = torch.tensor(x_train).float()
    explainer = shap.DeepExplainer(model, background_data)
    test_data_sample = torch.tensor(x_test).float()
    shap_values = np.squeeze(explainer.shap_values(test_data_sample))

    shap.summary_plot(shap_values, test_data_sample.numpy(), feature_names=column_names, show=False)
    plt.xticks(rotation=45)
    plt.title(f"SHAP Summary Plot test fold {test_fold_id}")
    clearml_task.get_logger().report_matplotlib_figure(title='K-Fold SHAP Summary Plot', series='K-Fold SHAP Summary Plot', figure=plt, iteration=test_fold_id)
    plt.clf()
    
    shap.summary_plot(shap_values, test_data_sample.numpy(), feature_names=column_names, plot_type='bar', color='lightblue', show=False)
    plt.title(f"Feature inportance test fold {test_fold_id}")
    clearml_task.get_logger().report_matplotlib_figure(title='K-Fold Feature inportance', series='K-Fold Feature inportance', figure=plt, iteration=test_fold_id)
    plt.clf()
    
    return shap_values, test_data_sample.numpy(), column_names