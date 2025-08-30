import json
import os
import tempfile

import hydra
import mlflow
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    # "test_regression_model",
]


# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Resolve absolute paths to local steps (Hydra changes CWD for runs)
    project_root = get_original_cwd()
    basic_cleaning_path = os.path.join(project_root, "src", "basic_cleaning")
    data_check_path = os.path.join(project_root, "src", "data_check")
    train_rf_path = os.path.join(project_root, "src", "train_random_forest")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                basic_cleaning_path,  # absolute path
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and nulls cleaned",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                    "apply_nyc_bounds": "true" if config["etl"]["apply_nyc_bounds"] else "false",  # NEW

                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                data_check_path,  # absolute path
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",  # add this alias in W&B UI once
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            # Serialize the RF config to JSON (MLflow step expects a file path)
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            _ = mlflow.run(
                train_rf_path,  # absolute path
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],  # NEW
                    "output_artifact": "random_forest_export",                       # RENAMED (was export_artifact)
                }

            )

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "random_forest_export:prod",  
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
