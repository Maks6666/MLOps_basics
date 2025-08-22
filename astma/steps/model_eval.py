from src.eval import RMSE, F1, AS
from zenml import step
from sklearn.base import ClassifierMixin
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
import logging


import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_eval(x_test: pd.DataFrame, y_test: pd.Series, model: ClassifierMixin) -> Tuple[Annotated[float, "f1_score"],
                                                                                         Annotated[float, "accuracy_score"],
                                                                                         Annotated[float, "rmse"]]:
    
    try:
        y_preds = model.predict(x_test)

        f1_obj = F1()
        f1 = f1_obj.calculate(y_preds, y_test)
        mlflow.log_metric('F1', f1)

        as_object = AS()
        a_s = as_object.calculate(y_preds, y_test)
        mlflow.log_metric('accuracy_score', a_s)


        rmse_obj = RMSE()
        rmse = rmse_obj.calculate(y_preds, y_test)
        mlflow.log_metric('RMSE', rmse)

        return f1, a_s, rmse
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e