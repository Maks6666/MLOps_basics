import logging
import pandas as pd
from zenml import step
from src.model_development import LinRegr, SVR_custom
from sklearn.base import RegressorMixin
# from .config import ModelNameConfig

import mlflow 
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, 
                y_train: pd.Series,
                x_test: pd.DataFrame, 
                y_test: pd.Series,
                model_name: str = "LinearRegression") -> RegressorMixin:
    try:
        model = None
        if model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinRegr()
            trained_model = model.train(x_train, y_train)
            return trained_model
        if model_name == "SVR":
            mlflow.sklearn.autolog()
            model = SVR_custom()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(model_name))
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

    