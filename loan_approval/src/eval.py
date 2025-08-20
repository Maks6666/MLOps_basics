from sklearn.metrics import f1_score, mean_squared_error, accuracy_score
from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin
import numpy as np
import logging

class Eval(ABC):
    @abstractmethod
    def calculate_metric(self, y_pred: np.ndarray, y_true: np.ndarray):
        ...
    

class F1(Eval):
    def calculate_metric(self, y_pred: np.ndarray, y_true: np.ndarray):
        try:
            logging.info("Calculating F1 score: ")
            f1 = f1_score(y_pred, y_true)
            logging.info(f"F1 calculated: {f1}")
            return f1
        except Exception as e:
            logging.error(f'Error: {e}')
            raise e

class MSE(Eval):
    def calculate_metric(self, y_pred: np.ndarray, y_true: np.ndarray):
        try:
            logging.info("Calculating MSE score: ")
            mse = mean_squared_error(y_pred, y_true)
            logging.info(f"MSE calculated: {mse}")
            return mse
        except Exception as e:
            logging.error(f'Error: {e}')
            raise e
        
class AS(Eval):
    def calculate_metric(self, y_pred: np.ndarray, y_true: np.ndarray):
        try:
            logging.info("Calculating accuracy score score: ")
            as_ = accuracy_score(y_pred, y_true)
            logging.info(f"Accuracy score calculated: {as_}")
            return as_
        except Exception as e:
            logging.error(f'Error: {e}')
            raise e


                         