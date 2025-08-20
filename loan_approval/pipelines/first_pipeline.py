from zenml import pipeline
from steps.clean_data import clean_data
from steps.injest_data import injest_data
from steps.train_model import train_model
from steps.tune_model import tune_model
from steps.model_eval import model_evaluation
import os

import mlflow


@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    os.environ["USER"] = "Maks Kucher"

    # mlflow.set_tracking_uri("http://localhost:5000")
    # mlflow.set_experiment(name)


    data = injest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(data)       
    params, name = tune_model(x_train, y_train, x_test, y_test, "logistic_regression")

    trained_model = train_model(x_train, y_train, params, name)
    mlflow.sklearn.log_model(trained_model, "model")

    f1, mse, accuracy_score = model_evaluation(trained_model, x_test, y_test)
    mlflow.end_run()

        # mlflow.log_metric("f1_score", f1)
        # mlflow.log_metric("MSE", mse)
        # mlflow.log_metric("accuracy_score", accuracy_score)
