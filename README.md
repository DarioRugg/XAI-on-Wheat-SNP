## How to Reproduce or Launch the Code

To run the code in this repository, follow the steps below:

### 1. Set Up ClearML Agent and Docker

- First, set up the **ClearML agent** following the [ClearML setup guide](https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_setup).
- Ensure the **Docker daemon** is running, as Docker will be used to create the required Python and R virtual environments.

### 2. Build Docker Images

- Build the Docker images using the provided Dockerfiles:
  - `Dockerfile` for the **Python** environment.
  - `r.Dockerfile` for the **R** environment (only needed if starting from raw data).

### 3. Prepare the Dataset

- The dataset is located in the `data/` folder. We recommend **excluding this folder from version control** after cloning the repository.
- Inside the folder, you will find two ZIP files:
  - `dataset_raw.zip`: Contains the **original raw data** from [Sandhu et al.](https://github.com/Sandhu-WSU/DL_Wheat/tree/master).  
    To use this:
    1. Preprocess it with the R script `Filtering pipeline/Genotyping and phenotyping filtering pipeline.R` using the R Docker environment.
    2. Upload the filtered data with `upload_data.py`, using `input_dataset=filtered` as an argument.
  - `dataset_processed.zip`: A **preprocessed** version (already passed through the R pipeline and Python processing).  
    Simply run `upload_data.py` (no arguments needed, `input_dataset=preprocessed` is the default).

- This will upload the dataset to ClearML and make it available for subsequent tasks.

### 4. Create a Draft Experiment

- Run `main.py` with the parameter:
  ```bash
  experiment=draft_for_hpo
