from pipelines.first_pipeline import run_pipeline
from zenml.client import Client


if __name__ == "__main__":
    link = "/Users/maxkucher/preprocessing/mlops/astma/synthetic_asthma_dataset.csv"
    trials = 10
    model_name = "random_forest_classifier"
    print("Pipeline started")
    run_pipeline(link, trials, model_name)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    print("Pipeline is over")