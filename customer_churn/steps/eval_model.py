import pandas as pd
from sklearn.base import ClassifierMixin
from zenml import step 
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from typing import Tuple

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def eval_model(trained_model: ClassifierMixin, x_test: pd.DataFrame, y_test: pd.Series) -> Tuple[float, float, float, float, float]:
    try:    
        y_pred = trained_model.predict(x_test)
        y_prob = trained_model.predict_proba(x_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("roc_auc", roc_auc)

        pr_auc = average_precision_score(y_test, y_prob)
        mlflow.log_metric("pr_auc", pr_auc)

        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("f1_score", f1)

        pr = precision_score(y_test, y_pred)
        mlflow.log_metric("precision", pr)

        r = recall_score(y_test, y_pred)
        mlflow.log_metric('recall', r)

        return roc_auc, pr_auc, f1, pr, r


    except Exception as e: 
        logging.error(f'Error: {e}')
        raise e