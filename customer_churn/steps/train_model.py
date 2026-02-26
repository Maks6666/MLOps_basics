import pandas as pd
from zenml import step

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.base import ClassifierMixin

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, best_params: dict) -> ClassifierMixin:
    if model_name == "LogRegr":
        penalty = best_params["penalty"]

        if penalty == "elasticnet":
            solver = "saga"
        elif penalty == "l1":
            solver = "liblinear"
        else:
            solver = "lbfgs"

        model = LogisticRegression(
            solver=solver,
            **best_params
        )
    
    elif model_name == 'Tree':
        model = DecisionTreeClassifier(**best_params)

    elif model_name == 'Forest':
        model = RandomForestClassifier(**best_params)

    
    trained_model = model.fit(x_train, y_train)
    mlflow.sklearn.log_model(sk_model=trained_model, artifact_path='model', registered_model_name=None)

    return trained_model


