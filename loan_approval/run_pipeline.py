from pipelines.first_pipeline import training_pipeline
import os
import mlflow
from zenml.client import Client


if __name__ == "__main__":
    data = "/Users/maxkucher/preprocessing/mlops/loan_approval/loan_data.csv"

    print("Pipeline started.")

    training_pipeline(data_path=data)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    print("Pipeline completed!")



