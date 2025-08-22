from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import logging

from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        ...


class LogRegr_Custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    def train_model(self, x_train, y_train):
        try:
            self.model = LogisticRegression(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")

class SVC_Custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    def train_model(self, x_train, y_train):
        try:
            self.model = SVC(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")


class DTC_Custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    def train_model(self, x_train, y_train):
        try:
            self.model = DecisionTreeClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")


class RFC_Custom(Model):
    def __init__(self, **kwargs):
        self.model = None
        self.kwargs = kwargs
    def train_model(self, x_train, y_train):
        try:
            self.model = RandomForestClassifier(**self.kwargs)
            self.model.fit(x_train, y_train)
            return self.model
        except Exception as e:
            logging.error(f"Error: {e}")




