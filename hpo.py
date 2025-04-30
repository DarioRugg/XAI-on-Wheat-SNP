import os
import sys
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange, HyperParameterOptimizer, UniformParameterRange, LogUniformParameterRange,
    UniformIntegerParameterRange, ParameterSet)
import hydra
from omegaconf import DictConfig, OmegaConf
from clearml.automation.hpbandster import OptimizerBOHB
from clearml.automation.optuna import OptimizerOptuna
from clearml.automation.optimization import RandomSearch
import pandas as pd

from scripts.clearml_utils import connect_confiuration, set_task_execution, setting_up_task, get_project_path


def get_hyperparameter_spce(parameter_set: str) -> list:
    if parameter_set == "base_params":
        return [                    
            UniformIntegerParameterRange('hydra_config/model/num_layers', min_value=1, max_value=6),
            UniformIntegerParameterRange('hydra_config/model/first_layer_dim', min_value=64, max_value=512, step_size=32),
            UniformParameterRange('hydra_config/model/dropout_prob', 0., 0.6),
            DiscreteParameterRange('hydra_config/model/batch_norm', [True, False]),
            LogUniformParameterRange('hydra_config/model/weight_decay', 0.00001, 0.001),
            DiscreteParameterRange('hydra_config/model/batch_size', [16, 32, 64]),
            DiscreteParameterRange('hydra_config/model/optimizer', ["sgd", "adam"]),
            LogUniformParameterRange('hydra_config/model/learning_rate', 0.000001, 0.05),
            DiscreteParameterRange('hydra_config/model/lr_scheduler', ["none", "plateau"])
        ]
    elif parameter_set == "augmentations":
        return [
            UniformParameterRange('hydra_config/model/augmentations/mask_prob', 0., 0.5),
            DiscreteParameterRange('hydra_config/model/augmentations/apply_masking', [True]),
            UniformParameterRange('hydra_config/model/augmentations/noise_prob', 0., 0.5),
            DiscreteParameterRange('hydra_config/model/augmentations/apply_noising', [True]),
            LogUniformParameterRange('hydra_config/model/learning_rate', 0.000001, 0.05),
            UniformParameterRange('hydra_config/model/dropout_prob', 0., 0.25)
        ]
    elif parameter_set == "complete":
        return [
            DiscreteParameterRange('hydra_config/model/augmentations/apply_masking', [True, False]),
            DiscreteParameterRange('hydra_config/model/augmentations/apply_noising', [True, False]),
            UniformIntegerParameterRange('hydra_config/model/num_layers', min_value=1, max_value=5),
            UniformIntegerParameterRange('hydra_config/model/first_layer_dim', min_value=64, max_value=512, step_size=32),
            UniformParameterRange('hydra_config/model/dropout_prob', 0., 0.6),
            DiscreteParameterRange('hydra_config/model/batch_norm', [True, False]),
            LogUniformParameterRange('hydra_config/model/weight_decay', 0.00001, 0.001),
            DiscreteParameterRange('hydra_config/model/batch_size', [16, 32, 64]),
            DiscreteParameterRange('hydra_config/model/optimizer', ["sgd", "adam"]),
            LogUniformParameterRange('hydra_config/model/learning_rate', 0.000001, 0.05),
            DiscreteParameterRange('hydra_config/model/lr_scheduler', ["none", "plateau"])
        ]


@hydra.main(config_path="conf", config_name="hpo_config", version_base=None)
def main(cfg : DictConfig) -> None:
    

    task: Task = setting_up_task(cfg.clearml_task, Task.TaskTypes.optimizer)

    cfg = connect_confiuration(clearml_task=task, configuration=cfg)
    print(OmegaConf.to_yaml(cfg))
        
    set_task_execution(task_cfg=cfg.clearml_task, machine_cfg=cfg.machine, clearml_task=task)
    
    
    if cfg.hpo_algo == "random":
        hpo_algo = RandomSearch
    elif cfg.hpo_algo == "bayes":
        # optuna is incompatible with loguniform sampler
        hpo_algo = OptimizerBOHB
    else:
        raise Exception("algorithm not included")
    
    if "id" not in cfg.base_task.keys() or cfg.base_task.id is None:
        base_task: Task = Task.get_task(project_name=get_project_path(cfg.base_task), task_name=cfg.base_task.task_name)
    else:
        base_task: Task = Task.get_task(task_id=cfg.base_task.id)
    
    task.add_tags([base_task.name, cfg.target_column, cfg.hpo_algo, f"param_space-{cfg.param_space}"])
    
    # Example use case:
    optimizer = HyperParameterOptimizer(
        # This is the experiment we want to optimize
        base_task_id=base_task.id,
        
        hyper_parameters=get_hyperparameter_spce(cfg.param_space) + [DiscreteParameterRange('hydra_config/target_column', [cfg.target_column])],
        
        objective_metric_title='Mean Kfold Results',
        objective_metric_series='MSE_loss_test_fold_00',
        objective_metric_sign='min',

        execution_queue=cfg.base_task.queue_name if cfg.base_task.accelerator == "cpu"  
                                      else cfg.base_task.queue_name.format(gpu_id=cfg.machine.gpu_queues_id),
                                      
        # setting optimizer 
        optimizer_class=hpo_algo,

        max_number_of_concurrent_tasks=2,
        
        optimization_time_limit=round(cfg.time_limit_days*24*60),
        total_max_jobs=200,

        # If specified only the top K performing Tasks will be kept, the others will be automatically archived
        save_top_k_tasks_only=5,
        
        max_iteration_per_job=10000,
        min_iteration_per_job=200,

        spawn_project=os.path.join(cfg.clearml_task.project, cfg.target_project_folder)
    )
    
    # optimizer.set_report_period(1)

    # start the optimization process
    print("Starting optimization process")
    optimizer.start()
    
    print(round(cfg.time_limit_days*24*60))
    # wait until process is done (notice we are controlling the optimization process in the background)
    print(optimizer.wait())

    # optimization is completed, print the top performing experiments id
    best_task = optimizer.get_top_experiments(top_k=1)[0]
    
    # make sure background optimization stopped
    optimizer.stop()

    task.connect({"task_id": best_task.id}, "best_task")


if __name__ == "__main__":
    main()
