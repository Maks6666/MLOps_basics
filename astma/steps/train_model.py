from zenml import step
from src.models import RFC_Custom, LogRegr_Custom, DTC_Custom, SVC_Custom
import pandas as pd
from sklearn.base import ClassifierMixin
import logging
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, name: str) -> ClassifierMixin:
    try:
        if name == "logistic_regression":
            # mlflow.sklearn.autolog()
            model = LogRegr_Custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        
        elif name == "support_vector_classifier":
            # mlflow.sklearn.autolog()
            model = SVC_Custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        
        elif name == "random_forest_classifier":
            # mlflow.sklearn.autolog()
            model = RFC_Custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        
        elif name == "descision_tree_classifier":
            # mlflow.sklearn.autolog()
            model = DTC_Custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e
