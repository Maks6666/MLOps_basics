from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
from zenml import step
import pandas as pd
from typing import Tuple
import logging

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval(model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
    try:
        preds = model.predict(x_test)

        acc_score = accuracy_score(y_test, preds)
        mlflow.log_metric("acc_score", acc_score)

        f1 = f1_score(y_test, preds)
        mlflow.log_metric("f1", f1)

        rmse = root_mean_squared_error(y_test, preds)
        mlflow.log_metric("rmse", rmse)

        return acc_score, f1, rmse

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
    