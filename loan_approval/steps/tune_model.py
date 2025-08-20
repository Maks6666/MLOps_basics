from zenml import step 
import optuna
from sklearn.metrics import f1_score
from src.train_model import RFC_custom, LogReg_custom
import pandas as pd
import logging
from typing import Tuple

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame,
               y_train: pd.Series,
               x_test: pd.DataFrame,
               y_test: pd.Series,
               model_name: str = "random_forest") -> Tuple[dict, str]:
    
    if model_name == "random_forest":
    
        def objective(trial):
            try:
                n_estimators = trial.suggest_int("n_estimators", 100, 500)

                max_depth = trial.suggest_int("max_depth", 5, 50)
        
                min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

                max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        
                model = RFC_custom(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
                trained_model = model.train_model(x_train, y_train)

                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"[Trial {trial.number}] F1 Score: {score:.4f} | n_estimators: {n_estimators}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, max_features: {max_features}")

                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e 
        
    if model_name == "logistic_regression":
        def objective(trial):
            try:
                penalty = trial.suggest_categorical("penalty", ["l2", None])
                C = trial.suggest_int("C", 1.0, 3.0)
                fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

                model = LogReg_custom(penalty=penalty, C=C, fit_intercept=fit_intercept)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"[Trial {trial.number}] F1 Score: {score:.4f} | penalty: {penalty}, C: {C}, fit_intercept: {fit_intercept}")

                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

        
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params

    for key, value in best_params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("model_name", model_name)

    logging.info(f"Best params: {best_params}")
    print(f"Best parameters found: {best_params}")

    return best_params, model_name