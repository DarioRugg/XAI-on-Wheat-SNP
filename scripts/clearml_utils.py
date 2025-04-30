import json
import os
import re
from typing import Union
import clearml
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from os.path import join


def connect_confiuration(clearml_task: clearml.Task, configuration: DictConfig) -> DictConfig:
    return OmegaConf.create(str(clearml_task.connect(OmegaConf.to_object(configuration), name="hydra_config")))

def setting_up_task(clearml_cfg: DictConfig, task_type: Union[clearml.Task.TaskTypes, None] = None) -> clearml.Task:
    task: clearml.Task = clearml.Task.init(project_name=get_project_path(clearml_cfg), 
                                           task_name=clearml_cfg.task_name,
                                           task_type=task_type,
                                           reuse_last_task_id=False,
                                           auto_connect_frameworks={'hydra': False})
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image=clearml_cfg.image,
        docker_arguments=clearml_cfg.args.format(**bot_credentials)
    )

    return task

def get_project_path(clearml_cfg):
    if isinstance(clearml_cfg.project_folders, list):
        return join(clearml_cfg.project, *clearml_cfg.project_folders)
    elif isinstance(clearml_cfg.project_folders, str):
        return join(clearml_cfg.project, clearml_cfg.project_folders)
    elif clearml_cfg.project_folders is None:
        return clearml_cfg.project
    else:
        raise ValueError()
    
def set_task_execution(task_cfg: DictConfig, machine_cfg: DictConfig, clearml_task: clearml.Task):
    if task_cfg.execution_type == "draft":
        clearml_task.execute_remotely()
    elif task_cfg.execution_type == "remote":
        clearml_task.execute_remotely(queue_name=task_cfg.queue_name if task_cfg.accelerator == "cpu" 
                                      else task_cfg.queue_name.format(gpu_id=machine_cfg.gpu_queues_id))
        
def create_new_task(clearml_cfg: DictConfig, task_type: Union[clearml.Task.TaskTypes, None] = None) -> clearml.Task:
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
        
    task: clearml.Task = clearml.Task.create(project_name=clearml_cfg.project, 
                                           task_name=clearml_cfg.task_name,
                                           task_type=task_type if task_type is not None else clearml.Task.TaskTypes.training, # type: ignore
                                           docker=clearml_cfg.image,
                                           docker_args=clearml_cfg.args.format(**bot_credentials))

    return task

# getting parameters from previous task
def get_best_params_from_task_id(task_id:str, cfg):
    print(f"loading hyper prameters from previous task (id:{task_id})")
    cfg.model = OmegaConf.create(Task.get_task(task_id=task_id).get_parameters_as_dict(True)["hydra_config"]).model

# finding previous HPO relative to the target and getting best task id
def get_best_params_from_last_hpo(cfg: DictConfig):
    print("getting best task from HPO task")
    task_id = Task.get_task(task_name=cfg.tuned_task.task_name, project_name=get_project_path(cfg.tuned_task), tags=[re.sub(r"201[0-9]_", "2016_", cfg.target_column)], allow_archived=False).get_parameters_as_dict(True)["best_task"]["task_id"]
    get_best_params_from_task_id(task_id=task_id, cfg=cfg)