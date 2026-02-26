import optuna 
from zenml import step

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

import pandas as pd
import logging

from zenml.client import Client 
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, 
               model_name: str, n_trials: int) -> dict:
    
    if model_name == 'LogRegr':
        def objective(trial):
            try:
                penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
                C = trial.suggest_categorical('C', [0.1, 1.0, 10.0])
                max_iter = trial.suggest_categorical('max_iter', [100, 500, 1000])

                if penalty == "elasticnet":
                    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
                    solver = "saga"
                    model = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=max_iter,
                        solver=solver,
                        l1_ratio=l1_ratio)
                    
                elif penalty == "l1":
                    solver = "liblinear"
                    model = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=max_iter,
                        solver=solver)
                    
                else:  
                    solver = "lbfgs"
                    model = LogisticRegression(
                        penalty=penalty,
                        C=C,
                        max_iter=max_iter,
                        solver=solver)

                trained_model = model.fit(x_train, y_train)
                y_pred = trained_model.predict(x_test)
                f1 = f1_score(y_test, y_pred)

                print(f"[Trial {trial.number}] F1 score: {f1:.4f} | penalty: {penalty}, C: {C}, max_iter: {max_iter}")

                return f1

            except Exception as e:
                logging.error(f'Error: {e}')
                raise e
            
    elif model_name == 'Tree':
        def objective(trial):
            try:

                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
                max_depth = trial.suggest_categorical('max_depth', [5, 10, 15, 20])
                min_samples_split = trial.suggest_categorical('min_samples_split', [1, 2, 3, 4])
                 
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split)
                trained_model = model.predict(x_train, y_train)
                y_pred = trained_model.predict(x_test)

                f1 = f1_score(y_test, y_pred)
                print(f"[Trial {trial.number}] F1 score: {f1:.4f} | criterion: {criterion}, max_depth: {max_depth}, min_samples_split: {min_samples_split}")

                return f1

            except Exception as e:
                logging.error(f'Error: {e}')
                raise e
            
    elif model_name == 'Forest':
        def objective(trial):
            try:
                n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 150, 200])
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)

                trained_model = model.fit(x_train, y_train)
                y_pred = trained_model.predict(x_test)
                f1 = f1_score(y_test, y_pred)

                print(f"[Trial {trial.number}] F1 score: {f1:.4f} | n_estimators: {n_estimators}, criterion: {criterion}")

                return f1


            except Exception as e:
                logging.error(f'Error: {e}')
                raise e
            
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    mlflow.log_param('model_name', model_name)
    logging.info(f'Best params: {best_params}')
    print(f'Best params: {best_params}')

    return best_params


