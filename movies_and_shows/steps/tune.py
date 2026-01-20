from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

import optuna
from zenml import step
import pandas as pd
import logging

from zenml.client import Client 
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, 
               x_test: pd.DataFrame, 
               y_train: pd.Series, 
               y_test: pd.Series,
               model_name: str,
               n_trials: int) -> dict:
    
    if model_name == "LogRegr":
        def objective(trial):
            try:
                penalty = trial.suggest_categorical("penalty",  [None, 'l1', 'l2'])
                C = trial.suggest_int("C", 1.0, 3.0)
                dualbool = trial.suggest_categorical("dualbool", [True, False])
                fit_intercept = trial.suggest_categorical(fit_intercept, [True, False])

                model = LogisticRegression(penalty=penalty, C=C, dualbool=dualbool, fit_intercept=fit_intercept)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"[Trial {trial.number}] F1 Score: {score:.4f} | penalty: {penalty}, C: {C}, dualbool: {dualbool}, fit_intercept: {fit_intercept}")

                return score


            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    if model_name == "Tree":
        def objective(trial):
            try:
                criterion = trial.suggest_categorical("criterion",  ['gini', 'entropy', 'log_loss'])
                splitter = trial.suggest_categorical("splitter",  ['best', 'random'])
                max_depth = trial.suggest_int("max_depth", 5, 15)

                min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3, 4, 5])
                
                model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"[Trial {trial.number}] F1 Score: {score:.4f} | criterion: {criterion}, splitter: {splitter}, max_depth: {max_depth}, min_samples_split: {min_samples_split}")

                return score


            except Exception as e:
                logging.error(f"Error: {e}")
                raise
    

    if model_name == "Forest":
        def objective(trial):
            try:
                criterion = trial.suggest_categorical("criterion",  ['gini', 'entropy', 'log_loss'])
                n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
                max_depth = trial.suggest_categorical("max_depth", [10, 30, 50])
                

                model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, max_depth=max_depth)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"[Trial {trial.number}] F1 Score: {score:.4f} | criterion: {criterion}, n_estimators: {n_estimators}, max_depth: {max_depth}")

                return score
            
            except Exception as e:
                logging.error(f"Error: {e}")
                raise

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    for key, value in best_params.items():
        mlflow.log_param(key, value)

    mlflow.log_param("model_name", model_name)
    
    logging.info(f"Best params: {best_params}")
    print(f'Best params: {best_params}')

    return best_params