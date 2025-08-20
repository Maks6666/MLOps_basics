from zenml import step 
from sklearn.base import ClassifierMixin
from src.train_model import RFC_custom, LogReg_custom
import pandas as pd
import logging

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame,
                y_train: pd.Series,
                params: dict,
                model_name: str) -> ClassifierMixin:
    try:
        if model_name == "random_forest":
            mlflow.sklearn.autolog()
            model = RFC_custom(**params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        elif model_name == "logistic_regression":
            mlflow.sklearn.autolog()
            model = LogReg_custom(**params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
        
    

