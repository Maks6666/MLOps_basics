import optuna 
from zenml import step
import pandas as pd
import logging

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error

import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
               model_name: str, n_trials: int) -> dict:
    
    if model_name == "LinReg":
        def objective(trial):
            try:

                fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
                copy_X = trial.suggest_categorical('copy_X', [True, False])
                n_jobs = trial.suggest_categorical('n_jobs', [1, 2, 3, 4, 5])

                model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)
                mae = mean_absolute_error(y_test, preds)

                print(f"[Trial {trial.number}] MAE score: {mae:.4f} | fit_intercept: {fit_intercept}, copy_X: {copy_X}, n_jobs: {n_jobs}")

                return mae

            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    elif model_name == "KNN":
        def objective(trial):
            try:

                algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree'])
                leaf_size = trial.suggest_categorical('leaf_size', [20, 25, 30, 35])
                p = trial.suggest_categorical('p', [1, 2, 3, 4, 5])

                model = KNeighborsRegressor(algorithm=algorithm, leaf_size=leaf_size, p=p)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)
                mae = mean_absolute_error(y_test, preds)

                print(f"[Trial {trial.number}] MAE score: {mae:.4f} | algorithm: {algorithm}, leaf_size: {leaf_size}, p: {p}")

                return mae

            except Exception as e:
                logging.error(f"Error: {e}")
                raise
    
    elif model_name == "SVR":
        def objective(trial):
            try:
                
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])

                if kernel == 'linear':
                    C = trial.suggest_float('C', 0.1, 10)
                    model = SVR(kernel='linear', C=C, max_iter=10000)

                else:
                    C = trial.suggest_float('C', 1, 100)
                    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
                    model = SVR(kernel='rbf', C=C, gamma=gamma, max_iter=10000)
                
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                mae = mean_absolute_error(preds, y_test)



                print(f"[Trial {trial.number}] MAE score: {mae:.4f} | kernel: {kernel}, C: {C}, max_iter: 10000")

                return mae

            except Exception as e:
                logging.error(f"Error: {e}")
                raise

        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    

    best_params = study.best_params
    for key, value in best_params.items():
        mlflow.log_param(key, value)
    
    mlflow.log_param("model_name", model_name)


    logging.info(f"Best params: {best_params}")
    print(f'Best params: {best_params}')

    return best_params