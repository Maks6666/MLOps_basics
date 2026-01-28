from zenml import step

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import pandas as pd
from sklearn.base import RegressorMixin
import logging

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, best_params: dict) -> RegressorMixin:
    if model_name == "LinReg":
        model = LinearRegression(**best_params)
       
    
    elif model_name == "KNN":
        model = KNeighborsRegressor(**best_params)

    
    elif model_name == "SVR":
        model = SVR(**best_params)
    

    train_model = model.fit(x_train, y_train)

    mlflow.sklearn.log_model(sk_model=train_model, artifact_path='model', registered_model_name=None)

    return train_model
    
