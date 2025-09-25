from zenml import step
from src.models import LinReg_custom, SVR_custom, RandForest_custom, DT_custom
import pandas as pd
from sklearn.base import RegressorMixin

import logging
from zenml.client import Client
import mlflow
experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, 
                y_train: pd.Series, 
                best_params: dict,
                model_name: str) -> RegressorMixin:
    
    if model_name == "linear_regression":
        try:
            mlflow.sklearn.autolog()
            model = LinReg_custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            
            return trained_model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e

    elif model_name == "svr":
        try:
            mlflow.sklearn.autolog()
            model = SVR_custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        
    elif model_name == "random_forest":
        try:
            mlflow.sklearn.autolog()
            model = RandForest_custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        
    elif model_name == "decision_tree":
        try:
            mlflow.sklearn.autolog()
            model = DT_custom(**best_params)
            trained_model = model.train_model(x_train, y_train)
            return trained_model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e