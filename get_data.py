import json
from clearml import Dataset
import clearml
import pandas as pd


def main():
    task = clearml.Task.init(project_name="e-muse/DL_Wheat_conference/tests", 
                                           task_name="get dataset",
                                           task_type="custom",
                                           reuse_last_task_id=True)
    
    with open("/clearml_conf/clearml_bot_credentials.json", 'r') as file:
        bot_credentials = json.load(file)
    
    task.set_base_docker(
        docker_image="rugg/dlwheatwithclearml:clearml",
        docker_arguments="--env CLEARML_AGENT_SKIP_PIP_VENV_INSTALL=1 \
            --env CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1 \
            --env CLEARML_AGENT_GIT_USER={bot_name} \
            --env CLEARML_AGENT_GIT_PASS={bot_token} \
            --mount type=bind,source=/srv/nfs-data/ruggeri/datasets/DL_Wheat_dataset/,target=/data/ \
            --mount type=bind,source=/srv/nfs-data/ruggeri/clearml_bot_credentials.json,target=/clearml_conf/clearml_bot_credentials.json".format(**bot_credentials)
    )

    dataset = Dataset.get(dataset_project='e-muse/DL_Wheat_conference', dataset_name='Dataset normalized')

    print(dataset.get_local_copy())

    
if __name__ == "__main__":
    main()