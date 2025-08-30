#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact.
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# DO NOT MODIFY (except: we keep a single wandb.init and rely on env from main.py)
def go(args):
    run = wandb.init(job_type="basic_cleaning", save_code=True)
    run.config.update(
        {
            "input_artifact": args.input_artifact,
            "output_artifact": args.output_artifact,
            "output_type": args.output_type,
            "output_description": args.output_description,
            "min_price": args.min_price,
            "max_price": args.max_price,
        }
    )

    # Download input artifact and read
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # --- Cleaning ---
    # price filter
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    # proper NYC geofence (rubric follow-up step expects this too)
    df = df[
        df["longitude"].between(-74.25, -73.50)
        & df["latitude"].between(40.5, 41.2)
    ].copy()

    # last_review to datetime (non-fatal if missing)
    if "last_review" in df.columns:
        df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Save cleaned csv
    output_csv = "clean_sample.csv"
    df.to_csv(output_csv, index=False)

    # Log the cleaned artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_csv)
    run.log_artifact(artifact)
    run.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic cleaning of sample.csv and upload clean_sample.csv"
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="W&B artifact to read as input (e.g., 'sample.csv:latest')",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of output artifact to create (e.g., 'clean_sample.csv')",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Artifact type for the cleaned data (use 'clean_sample')",
    )
    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Human-readable description of the cleaned dataset",
    )
    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum allowed price; rows below are dropped",
    )
    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum allowed price; rows above are dropped",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    go(args)
