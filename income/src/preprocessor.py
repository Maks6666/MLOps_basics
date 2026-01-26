import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
import logging 



class Preprocessor:
    def __init__(self):
        ...

    def replace(self, data: pd.DataFrame, column: str, replacer: dict) -> pd.DataFrame:
        try:
            data = data.copy()
            data[column] = data[column].replace(replacer)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
    
    def drop_data(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            data = data.copy()
            data = data.drop(column, axis='columns')
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
    
    def fill_nan(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            data = data.copy()
            mode = data[column].mode()[0]
            data[column] = data[column].fillna(mode)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        
