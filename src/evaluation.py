import logging
from abc import ABC, abstractclassmethod
import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    @abstractclassmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        ...


class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE calculated: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e


class R2(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 calculated: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        

class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info(f"RMSE calculated: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        


