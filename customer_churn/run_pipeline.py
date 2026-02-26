from pipelines.pipeline_1 import run_pipeline
from zenml.client import Client

if __name__ == '__main__':
    link = '/Users/maxkucher/preprocessing/mlops/customer_churn/churn_data.csv'
    target = 'Churn'
    model_name = 'LogRegr'
    n_trials = 15
    print('Pipeline started')
    run_pipeline(link=link, target=target, model_name=model_name, n_trials=n_trials)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    print('Pipeline ended')

# zenml integration install mlflow

# mlflow ui --backend-store-uri 'file:/Users/maxkucher/Library/Application Support/zenml/local_stores/0195adce-55dc-4a6f-97c3-4950b2345b47/mlruns'