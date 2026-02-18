from zenml import step

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error as mae

import optuna 
import pandas as pd
import logging

import mlflow 
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, 
               model_name: str, n_trials: int) -> dict:
    
    if model_name == 'tree':
        def objective(trial):
            try:
                criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
                max_depth = trial.suggest_categorical('max_depth', [5, 10, 15])
                min_samples_split = trial.suggest_categorical('min_samples_split', [2, 4, 6])

                model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)
                mae_value = mae(preds, y_test)

                print(f"[Trial {trial.number}] MAE score: {mae_value:.4f} | criterion: {criterion}, max_depth: {max_depth}, min_samples_split: {min_samples_split}")

                return mae_value


            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    elif model_name == 'forest':
        def objective(trial):
            try:
                n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 150, 200])
                criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
                max_depth = trial.suggest_categorical('max_depth', [10, 15, 20, 25])


                model = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)
                mae_value = mae(preds, y_test)

                print(f"[Trial {trial.number}] MAE score: {mae_value:.4f} | criterion: {criterion}, n_estimators: {n_estimators}, max_depth: {max_depth}")

                return mae_value

            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    elif model_name == 'svr':
        def objective(trial):
            try:    
                kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'sigmoid'])
                degree = trial.suggest_categorical('degree', [2, 3, 4])
                gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

                model = SVR(kernel=kernel, degree=degree, gamma=gamma)

                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)
                mae_value = mae(preds, y_test)

                print(f"[Trial {trial.number}] MAE score: {mae_value:.4f} | kernel: {kernel}, degree: {degree}, gamma: {gamma}")

                return mae_value

            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    mlflow.log_param('model_name', model_name)


    logging.info(f'best_params: {best_params}')
    print(f'Best params: {best_params}')

    return best_params

            

