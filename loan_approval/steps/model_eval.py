from src.eval import F1, MSE, AS
from zenml import step
from sklearn.base import ClassifierMixin

from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
import logging

import mlflow

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_evaluation(model: ClassifierMixin,
                     x_test: pd.DataFrame,
                     y_test: pd.Series) -> Tuple[Annotated[float, "f1_score"],
                                                 Annotated[float, "mean_squared_error"],
                                                   Annotated[float, "accuracy_score"]]:
    
    try:
        preds = model.predict(x_test)

        f1_obj = F1()
        f1 = f1_obj.calculate_metric(preds, y_test)
        mlflow.log_metric("F1", f1)

        mse_obj = MSE()
        mse = mse_obj.calculate_metric(preds, y_test)
        mlflow.log_metric("MSE", mse)

        acc_obj = AS()
        accuracy_score = acc_obj.calculate_metric(preds, y_test)
        mlflow.log_metric("accuracy_score", accuracy_score)

        return f1, mse, accuracy_score
    except Exception as e:
        logging.error(f"Eroor: {e}")
        raise e


    
    

