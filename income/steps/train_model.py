from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from zenml import step
import pandas as pd
from sklearn.base import ClassifierMixin

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, best_params: dict) -> ClassifierMixin:
    if model_name == "LogReg":
        model = LogisticRegression(**best_params)
        
    elif model_name == "Tree":
        model = DecisionTreeClassifier(**best_params)

    elif model_name == "Forest":
        model = RandomForestClassifier(**best_params)

    train_model = model.fit(x_train, y_train)

    mlflow.sklearn.log_model(sk_model=train_model, artifact_path="model", registered_model_name=None)

    return train_model
