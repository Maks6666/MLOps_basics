from sklearn.base import RegressorMixin 

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from abc import ABC, abstractmethod
import pandas as pd
import logging

class Model(ABC):
    @abstractmethod
    def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> RegressorMixin:
        ...


class LinReg_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def train_model(self, x_train, y_train):
        try:
            self.model = LinearRegression(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e

class SVR_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def train_model(self, x_train, y_train):
        try:
            self.model = SVR(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        


class RandForest_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def train_model(self, x_train, y_train):
        try:
            self.model = RandomForestRegressor(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e
        


class DT_custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs

    def train_model(self, x_train, y_train):
        try:
            self.model = DecisionTreeRegressor(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e: 
            logging.error(f"Error: {e}")
            raise e




        