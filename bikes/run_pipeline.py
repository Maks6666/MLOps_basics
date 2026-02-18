from pipelines.pipeline_1 import run_pipeline

from zenml.client import Client


if __name__ == '__main__':
    link = '/Users/maxkucher/preprocessing/mlops/bikes/data.csv'
    model_name = 'forest'
    n_trials = 15

    print('Pipeline started...')
    run_pipeline(link=link, model_name=model_name, n_trials=n_trials)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    print('Pipeline ended.')


# !!!! export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# mlflow ui --backend-store-uri 'file:/Users/maxkucher/Library/Application Support/zenml/local_stores/79a9fc9c-2f87-4b7f-8cfb-d44929e4dcfd/mlruns'