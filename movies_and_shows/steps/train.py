from sklearn.base import ClassifierMixin
from zenml import step

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(best_params: dict, model_name: str, x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:

    mlflow.sklearn.autolog(log_models=False)
    if model_name == "LogReg":
        model = LogisticRegression(**best_params)
    
    elif model_name == "Tree":
        model = DecisionTreeClassifier(**best_params)
    
    elif model_name == "Forest":
        model = RandomForestClassifier(**best_params)
        
    trained_model = model.fit(x_train, y_train)

    mlflow.sklearn.log_model(sk_model=trained_model, artifact_path="model", registered_model_name=None,)

    return trained_model