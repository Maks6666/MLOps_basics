import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

class Preprocessor:
    def __init__(self):
        ...

    def drop_values(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            data = data.drop(columns, axis='columns')
            return data
        except Exception as e: 
            logging.error(f'Error: {e}')
            raise


class Scaler:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, data: pd.DataFrame, columns: list):
        self.scaler.fit(data[columns])

    def transform(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        scaled = self.scaler.transform(data[columns])
        data_scaled = data.copy()
        data_scaled[columns] = scaled
        return data_scaled