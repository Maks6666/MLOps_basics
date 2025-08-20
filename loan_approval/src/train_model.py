# from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod
import logging
import pandas as pd
import logging

class Model(ABC):
    @abstractmethod
    def train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        ...


class RFC_custom(Model):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None

    def train_model(self, x_train: pd.DataFrame, y_train: pd.Series):
        try:
            self.model = RandomForestClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            raise e
        
class LogReg_custom(Model):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
    def train_model(self, x_train, y_train):
        try:
            self.model = LogisticRegression(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

            
        

