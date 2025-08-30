# Build an ML Pipeline for Short-Term Rental Prices in NYC

End-to-end, reproducible pipeline to estimate nightly prices for NYC short-term rentals. New data arrives weekly; this pipeline downloads → cleans → checks → splits → trains → (optionally) tests a model, with runs and artifacts tracked in Weights & Biases (W&B) and execution orchestrated by MLflow + Hydra.

https://wandb.ai/wguproject/nyc_airbnb_clean/overview
---

## Quick start

### 0) Environment
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev

1) W&B auth (required)
wandb login    # paste key from https://wandb.ai/authorize
# (or) export WANDB_API_KEY=...

2) Where runs appear in W&B

Use this project & entity:

export WANDB_ENTITY=wguproject
export WANDB_PROJECT=nyc_airbnb_clean

Configuration (Hydra)

Edit config.yaml (read by main.py):

main:
  components_repository: "https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter.git#components"
  project_name: nyc_airbnb_clean
  experiment_name: development
  steps: all

etl:
  sample: "sample1.csv"
  min_price: 10
  max_price: 350
  # Toggleable NYC geofence used in cleaning; see Release Flow below
  apply_nyc_bounds: true

data_check:
  kl_threshold: 0.2

modeling:
  test_size: 0.2
  val_size: 0.2
  random_seed: 42
  stratify_by: "neighbourhood_group"
  max_tfidf_features: 5
  random_forest:
    n_estimators: 100
    max_depth: 15
    min_samples_split: 4
    min_samples_leaf: 3
    n_jobs: -1
    criterion: squared_error
    max_features: 0.5


Any value can be overridden via -P "hydra_options=...".

Running the pipeline
Run everything (from repo root)
mlflow run . -P "hydra_options=main.project_name='${WANDB_PROJECT}'"

Run specific step(s)
# download only
mlflow run . -P steps=download -P "hydra_options=main.project_name='${WANDB_PROJECT}'"

# download + cleaning
mlflow run . -P steps=download,basic_cleaning -P "hydra_options=main.project_name='${WANDB_PROJECT}'"

What each step does

download → logs sample.csv (artifact type: raw_data)

basic_cleaning → filters by price; optional NYC geofence (see etl.apply_nyc_bounds); logs clean_sample.csv (type: clean_sample)

data_check → pytest suite (schema, bounds, KL drift, row count, price range). Requires:

clean_sample.csv:latest

clean_sample.csv:reference (set this alias in the W&B UI on your chosen clean artifact version)

data_split → logs trainval_data.csv and test_data.csv

train_random_forest → trains RF with preprocessing; logs random_forest_export (MLflow sklearn model)

test_regression_model (optional) → evaluates a promoted model on test_data.csv

Hyperparameter multirun (Hydra)

⚠️ Put -m at the end of the hydra_options string.

mlflow run . \
  -P steps=train_random_forest \
  -P "hydra_options=modeling.random_forest.n_estimators=100,300 \
                       modeling.random_forest.max_depth=10,30,60 \
                       main.project_name='${WANDB_PROJECT}' -m"


In W&B Runs → Table, sort by mae (ascending), open the best run → Artifacts → random_forest_export → add alias prod.

Release flow (fail → fix on sample2.csv)

We use the etl.apply_nyc_bounds flag to show a realistic fail-first release, then a fix.

Your repo: https://github.com/LowTideDev/nyc-pipeline-clean.git

vFAIL (bounds off) — v1.0.1
# ensure config has: etl.apply_nyc_bounds: false
git commit -am "v1.0.1: disable NYC bounds"
git tag -a v1.0.1 -m "fail-first: no NYC geofence"
git push origin main --tags

mlflow run https://github.com/LowTideDev/nyc-pipeline-clean.git \
  -v v1.0.1 \
  -P "hydra_options=main.project_name='${WANDB_PROJECT}' etl.sample='sample2.csv'"
# expected: data_check fails on proper_boundaries

vFIX (bounds on) — v1.0.3

(Note: main.py forwards the flag to basic_cleaning.)

# flip config to: etl.apply_nyc_bounds: true
git commit -am "v1.0.3: enable NYC bounds (fix)"
git tag -a v1.0.3 -m "fix: NYC geofence on"
git push origin main --tags

mlflow run https://github.com/LowTideDev/nyc-pipeline-clean.git \
  -v v1.0.3 \
  -P "hydra_options=main.project_name='${WANDB_PROJECT}' etl.sample='sample2.csv'"
# expected: pipeline succeeds end-to-end

(Optional) Test the promoted model

After tagging the best training run’s model as prod:

mlflow run . -P steps=test_regression_model -P "hydra_options=main.project_name='${WANDB_PROJECT}'"

What you should see in W&B

raw_data/sample.csv

clean_sample/clean_sample.csv (+ alias reference)

trainval_data.csv, test_data.csv

Training runs with mae and r2 logged; best model’s random_forest_export tagged prod

Artifacts → Graph shows: download → clean → checks → split → train → model

Troubleshooting

No W&B sync? Make sure you ran wandb login, and you don’t have WANDB_MODE=offline.

Hydra multirun error? Keep -m at the end of the hydra_options string.

Bounds not applied in release? Ensure main.py forwards apply_nyc_bounds to basic_cleaning, and check the printed command includes --apply_nyc_bounds true.

Conda env weirdness? Remove mlflow-* envs:

for e in $(conda info --envs | grep mlflow | awk '{print $1}'); do conda env remove -n "$e" -y; done
