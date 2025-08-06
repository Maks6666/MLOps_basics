from abc import ABC, abstractclassmethod
import logging
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

class Model(ABC):
    @abstractclassmethod
    def train(self, x_train, y_train):
        ...
    


class LinRegr(Model):
    def train(self, x_train, y_train, **kwargs):
        try:
            lin_reg = LinearRegression(**kwargs)
            lin_reg.fit(x_train, y_train)
            logging.info("Model training completed")
            return lin_reg
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        

class SVR_custom(Model):
    def train(self, x_train, y_train, **kwargs):
        try:
            svr = SVR()
            svr.fit(x_train, y_train, **kwargs)
            logging.info("Model training completed")
            return svr
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e


