from pipelines.pipeline_1 import run_pipeline
from zenml.client import Client


if __name__ == '__main__':
    link = '/Users/maxkucher/preprocessing/mlops/income/data.csv'
    target = 'income'
    model_name = 'Forest'
    n_trials = 20
    run_pipeline(link, target, model_name, n_trials)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())

    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY = YES

    # mlflow ui --backend-store-uri "file:/Users/maxkucher/Library/Application Support/zenml/local_stores/79a9fc9c-2f87-4b7f-8cfb-d44929e4dcfd/mlruns"