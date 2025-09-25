from zenml import step
import optuna 
import mlflow
from zenml.steps import get_step_context

import pandas as pd
from typing import Tuple
import logging
from sklearn.metrics import root_mean_squared_error as rmse


from src.models import LinReg_custom, SVR_custom, RandForest_custom, DT_custom
from zenml.client import Client


mlflow.autolog(disable=True)


experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)
# @step
def tune_model(x_train: pd.DataFrame, 
               x_test: pd.DataFrame, 
               y_train: pd.Series, 
               y_test: pd.Series,
               n_trials: int = 15,
               model_name: str = "linear_regression") -> dict:
    

    if model_name == "linear_regression":
        def objective(trial):
            try:
                fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
                n_jobs = trial.suggest_int('n_jobs', 1, 3)
                copy_X = trial.suggest_categorical("copy_X", [True, False])

                model = LinReg_custom(fit_intercept=fit_intercept, n_jobs=n_jobs, copy_X=copy_X)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = rmse(preds, y_test)
                return score

            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            
    elif model_name == "svr":
        def objective(trial):
            try:
                degree = trial.suggest_categorical("degree", [1, 2, 3, 4])
                C = trial.suggest_categorical("C", [2.0, 3.0, 4.0])
                gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

                model = SVR_custom(degree=degree, C=C, gamma=gamma)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test)
                score = rmse(preds, y_test)
                return score
            
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            
        
    elif model_name == "random_forest":
        def objective(trial):
            try:
                n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
                criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
                min_samples_split = trial.suggest_int('min_samples_split', [2, 3, 4, 5])

                model = RandForest_custom(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test, y_test)
                score = rmse(preds, y_test)
                return score
            
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e

    elif model_name == "decision_tree":
        def objective(trial):
            try:
                criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
                max_depth = trial.suggest_int("max_depth", [3, 5, 7])
                min_samples_split = trial.suggest_int('min_samples_split', [2, 3, 4, 5])

                model = DT_custom(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
                trained_model = model.train_model(x_train, y_train)
                preds = trained_model.predict(x_test, y_test)
                score = rmse(preds, y_test)
                return score
            
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    for key, value in best_params.items():
        mlflow.log_param(key, value)
    
    mlflow.log_param("model_name", model_name)

    # context = get_step_context()
    # context.add_output_metadata(
    #     metadata={  
    #         "model_name": model_name,
    #         **best_params
    #     }
    # )

    logging.info(f"Best params: {best_params}")
    print(f"Best params: {best_params}")

    return best_params

