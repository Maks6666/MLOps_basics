from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
from zenml import step 
from sklearn.base import ClassifierMixin
import pandas as pd 
from typing import Tuple
import logging

import mlflow
from zenml.client import Client


experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_eval(trained_model: ClassifierMixin, 
               x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:

    try:
        y_pred = trained_model.predict(x_test)

        acc_score = accuracy_score(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric("accuracy_score", acc_score)

        f_score = f1_score(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric('f1_score', f_score)

        rmse_score = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
        mlflow.log_metric("RMSE", rmse_score)

        return acc_score, f_score, rmse_score

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e