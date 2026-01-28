from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from zenml import step

from sklearn.base import RegressorMixin
from typing import Tuple
import logging
import pandas as pd

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval_model(trained_model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float]:
    try:
        preds = trained_model.predict(x_test)

        mae = mean_absolute_error(preds, y_test)
        mlflow.log_metric("mean_absolute_error", mae)

        mse = mean_squared_error(preds, y_test)
        mlflow.log_metric("mean_squered_error", mse)
        
        r2 = r2_score(preds, y_test)
        mlflow.log_metric("r2_score", r2)

        return mae, mse, r2
    
    except Exception as e: 
        logging.error(f"Error: {e}")
        raise e
