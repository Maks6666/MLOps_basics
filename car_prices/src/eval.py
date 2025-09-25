from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse
import numpy as np

import logging

class Metric(ABC):
    @abstractmethod
    def calculate(y_test: np.ndarray, y_preds: np.ndarray):
        ...

class MAE(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating MAE...")
            mae_value = mae(y_test, y_preds)
            logging.info(f"MAE calculated: {mae_value}")
            return mae_value
            
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        

class MSE(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating MSE...")
            mse_value = mse(y_test, y_preds)
            logging.info(f"MSE calculated: {mse_value}")
            return mse_value
            
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
class RMSE(Metric):
    def calculate(self, y_test: np.ndarray, y_preds: np.ndarray):
        try:
            logging.info("Calculating RMSE...")
            rmse_value = rmse(y_test, y_preds)
            logging.info(f"RMSE calculated: {rmse_value}")
            return rmse_value
            
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e