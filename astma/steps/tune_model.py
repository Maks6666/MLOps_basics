import optuna 
from zenml import step
from src.models import  LogRegr_Custom, SVC_Custom, DTC_Custom, RFC_Custom
import pandas as pd
from typing import Tuple 
from sklearn.metrics import f1_score
import logging

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, 
               y_train: pd.Series,
               x_test: pd.DataFrame, 
               y_test: pd.Series,
               n_trials: int = 15,
               model_name: str = "logistic_regression") -> Tuple[dict, str]:

    if model_name == "logistic_regression":

        def objective(trial):
            try:
                penalty = trial.suggest_categorical("penalty", ["l2", None])
                C = trial.suggest_int("C", 1.0, 5.0)
                fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

                model = LogRegr_Custom(penalty=penalty, C=C, fit_intercept=fit_intercept)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = f1_score(y_test, preds)
                return score
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    
    elif model_name == "support_vector_classifier":
        def objective(trial):
            try:
                C = trial.suggest_int("C", 1.0, 5.0)
                # kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
                gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

                model = SVC_Custom(C=C, gamma=gamma)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = f1_score(y_test, preds)
                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    
    elif model_name == "random_forest_classifier":
        def objective(trial):
            try:
                n_estimators = trial.suggest_int("n_estimators", 100, 500)
                max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
                bootstrap = trial.suggest_categorical("bootstrap", [True, False])
                max_depth = trial.suggest_int("max_depth", 3, 10)

                model = RFC_Custom(n_estimators=n_estimators, max_features=max_features, bootstrap=bootstrap, max_depth=max_depth)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = f1_score(y_test, preds)
                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    
    elif model_name == "descision_tree_classifier":
        def objective(trial):
            try:
                max_depth = trial.suggest_int("max_depth", 3, 8)
                min_samples_split = trial.suggest_int("min_samples_split", 5, 15)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 10)
                criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

                model = DTC_Custom(max_dept=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = f1_score(y_test, preds)
                return score
                
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    for key, value in best_params.items():
        mlflow.log_param(key, value)
    
    mlflow.log_param("model_name", model_name)

    logging.info(f"Best params: {best_params}")
    print(f"Best parameters found: {best_params}")

    return best_params, model_name
    


