import optuna
from zenml import step
import pandas as pd

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score

from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def tune_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, model_name: str, n_trials: int) -> dict:
    if model_name == "LogReg":
        try:
            def objective(trial):
                penalty = trial.suggest_categorical("penalty", ['l1', 'l2'])
                C = trial.suggest_categotical("C", [1.0, 3.0, 5.0])
                fit_intercept = trial.suggest_categotical("fit_intercept", [True, False])
                max_iter = trial.suggest_categotical('max_iter', [100, 500, 1000, 1500])

                model = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, max_iter=max_iter)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_train)
                score = f1_score(y_test, preds)

                print(f"Trial: {trial.number} | Score: {score} | Penalty: {penalty} | C: {C} | Fit_intercept: {fit_intercept} | Max_iter: {max_iter}")

                return score
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e 
        
    elif model_name == "Tree":
        def objective(trial):
            try:
                criterion = trial.suggest_categorical("criterion", ['gini', "entropy"])
                max_depth = trial.suggest_categorical("max_depth", [1, 3, 5, 7, 9])
                min_samples_split = trial.suggest_categorical("min_samples_split", [1.0, 2.0, 3.0])
                max_features = trial.suggest_categorical("max_features", ['auto', 'sqrt'])

                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(preds, y_test)

                print(f"Trial: {trial.number} | Score: {score} | Criterion: {criterion} | Max_depth: {max_depth} | Min_samples_split: {min_samples_split} | Max_features: {max_features}")

                return score
            
            except Exception as e:
                logging.error(f"Error: {e}")
                raise e
            
    elif model_name == "Forest":
        def objective(trial):
            try:
                n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150, 200])
                criterion = trial.suggest_categorical("criterion", ['gini', "entropy"])
                min_samples_split = trial.suggest_categorical("min_samples_split", [2, 3, 4])

                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split)
                trained_model = model.fit(x_train, y_train)
                preds = trained_model.predict(x_test)

                score = f1_score(y_test, preds)

                print(f"Trial: {trial.number} | Score: {score} | Criterion: {criterion} | N_estimators: {n_estimators} | Min_samples_split: {min_samples_split}")

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
    print(f"Best params: {best_params}")

    return best_params