from src.eval import RMSE, MAE, MSE
from sklearn.base import RegressorMixin
import pandas as pd 
from zenml import step

from typing import Tuple
from typing_extensions import Annotated
import logging

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_eval(model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "mse"],
                                                                                         Annotated[float, "mae"],
                                                                                         Annotated[float, "rmse"]]:
    
    try:
        y_preds = model.predict(x_test)

        mse_metric = MSE()
        mse = mse_metric.calculate(y_test, y_preds)
        mlflow.log_metric("mse", mse)

        mae_metric = MAE()
        mae = mae_metric.calculate(y_test, y_preds)
        mlflow.log_metric("mae", mae)

        rmse_metric = RMSE()
        rmse = rmse_metric.calculate(y_test, y_preds)
        mlflow.log_metric("rmse", rmse)

        return mse, mae, rmse

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

        


    
