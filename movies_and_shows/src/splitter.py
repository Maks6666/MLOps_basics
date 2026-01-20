import pandas as pd 
from typing import Tuple
from sklearn.model_selection import train_test_split
import logging

class Splitter:
    def __init__(self, test_size):
        self.test_size = test_size
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            data = data.copy()
            X = data.drop("type", axis="columns")
            y = data["type"]

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)

            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error: {e}")
            raise