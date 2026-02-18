from zenml import step

import pandas as pd
from sklearn.base import RegressorMixin

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import mlflow 
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, best_params: dict, model_name: str) -> RegressorMixin:
    if model_name == 'tree':
        model = DecisionTreeRegressor(**best_params)
    elif model_name == 'forest':
        model = RandomForestRegressor(**best_params)
    elif model_name == 'svr':
        model = SVR(**best_params)

    train_model = model.fit(x_train, y_train)
    mlflow.sklearn.log_model(sk_model=train_model, artifact_path='model', registered_model_name=None)
    

    return train_model
