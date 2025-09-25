from pipelines.pipeline_1 import run_pipeline
from zenml.client import Client



if __name__ == "__main__":
    data = "car_prices/car_data.csv"
    n_trials = 15
    model_name = "svr"
    print("Pipeline started...")
    run_pipeline(data, n_trials, model_name)
    print("Pipeline ended.")
    print(Client().active_stack.experiment_tracker.get_tracking_uri())


# mlflow ui --backend-store-uri "file:/Users/maxkucher/Library/Application Support/zenml/local_stores/0cf0c7cc-d186-488b-90d0-24105f08400a/mlruns"
