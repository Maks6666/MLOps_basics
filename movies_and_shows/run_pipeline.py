from pipelines.pipeline_1 import run_pipeline
from zenml.client import Client

if __name__ == "__main__":
    data_link = "/Users/maxkucher/preprocessing/mlops/movies_and_shows/data.csv"
    model_name = "Tree"
    n_trials = 20
    run_pipeline(data_link, model_name, n_trials)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

# zenml integration install mlflow -y

# zenml stack set mlflow_stack    or -> 
# zenml stack register mlflow_stack \
#                        -o default \
#                        -a default \
#                        -e mlflow_tracker


# mlflow ui --backend-store-uri 'file:/Users/maxkucher/Library/Application Support/zenml/local_stores/79a9fc9c-2f87-4b7f-8cfb-d44929e4dcfd/mlruns'