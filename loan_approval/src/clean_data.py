from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import logging
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def hadnle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        ...


class CleanData(DataStrategy):
    def hadnle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data.replace({"person_gender": {"female": 0, "male": 1}}, inplace=True)
            data.replace({"previous_loan_defaults_on_file": {"No": 0, "Yes": 1}}, inplace=True)
            data = pd.get_dummies(data, columns=["person_home_ownership", "loan_intent", "person_education"])
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        

class SplitData(DataStrategy):
    def hadnle_data(self, data: pd.DataFrame) -> pd.Series:
        try:
            x = data.drop("loan_status", axis="columns")
            y = data["loan_status"]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

class DataTool:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.hadnle_data(self.data)
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e