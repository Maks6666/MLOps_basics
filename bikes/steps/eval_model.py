from zenml import step

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
from sklearn.base import RegressorMixin

from typing import Tuple
import logging

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_eval(trained_model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float]:
    try:
        y_pred = trained_model.predict(x_test)

        mse_value = mse(y_pred, y_test)
        mlflow.log_metric('mse_value', mse_value)

        mae_value = mae(y_pred, y_test)
        mlflow.log_metric('mae_value', mae_value)

        return mse_value, mae_value

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

