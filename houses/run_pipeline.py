from pipelines.pipeline_1 import run
from zenml.client import Client


if __name__ == '__main__':
    link = '/Users/maxkucher/preprocessing/mlops/houses/housing.csv'
    target = 'median_house_value'
    model_name = 'SVR'
    n_trials = 15



    print('Pipeline started...')
    run(link, target, model_name, n_trials)
    print('Pipeline ended')
    print(Client().active_stack.experiment_tracker.get_tracking_uri())


# mlflow ui --backend-store-uri 'file:/Users/maxkucher/Library/Application Support/zenml/local_stores/79a9fc9c-2f87-4b7f-8cfb-d44929e4dcfd/mlruns'