import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
import numpy as np

class Preprocessor:
    def __init__(self):
       ...

    def fill_empties(self, data: pd.DataFrame, columns: list, fill_value: str) -> pd.DataFrame:
        try:
            data = data.copy()
            for column in columns:
                if column in data.columns:
                    data[column] = data[column].fillna(fill_value)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def drop_nulls(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            data = data.copy()
            for column in columns:
                if column in data.columns:
                    data = data.dropna(subset=[column])

            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def drop_values(self, data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        try:
            data = data.copy()
            data = data.drop(columns, axis="columns")
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise

    
    def replace(self, data: pd.DataFrame, column: str, replacer: dict) -> pd.DataFrame:
        try:
            data = data.copy()
            if column in data.columns:
                data[column] = data[column].replace(replacer)
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def leave_one(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            data = data.copy()
            for column in columns:
                if column in data.columns:
                    data[column] = data[column].str.split(",").str[0].str.strip()
            return data 
        except Exception as e:
            logging.error(f"Error: {e}")
            raise


class Encoder:
    def __init__(self):
        ...
        
    def one_hot(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            data = data.copy()
            ecnoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
            encoded = ecnoder.fit_transform(data[[column]])
            encoded_df = pd.DataFrame(encoded, columns=ecnoder.get_feature_names_out([column]), index=data.index)
            data = pd.concat([data.drop(column, axis="columns"), encoded_df], axis="columns")
            return data
        
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def datetime(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        try:
            data = data.copy()
            if column in data.columns: 
                data[column] = pd.to_datetime(data[column], errors="coerce")
            data["year"] = data[column].dt.year
            data["month"] = data[column].dt.month
            data = data.drop(column, axis="columns")
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def cycle_encoder(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        try: 
            data = data.copy()
            data[f'{column}_sin'] = np.sin(2 * np.pi * data[column] / 12)
            data[f'{column}_cos'] = np.cos(2 * np.pi * data[column] / 12)
            data = data.drop(column, axis="columns")
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    def parse_duration(self, value: str):
        if pd.isna(value):
            return np.nan, np.nan
        
        value = value.lower()

        if "min" in value:
            minutes = int(value.replace("min", "").strip())   
            return minutes, 0
        
        if "season" in value:
            seasons = int(value.replace("seasons", "").replace("season", "").strip())
            return 0, seasons

        return np.nan, np.nan
    
    def ecnode_duration(self, data: pd.DataFrame, columns: list[str], column: str) -> pd.DataFrame:
        try: 
            data = data.copy()
            data[columns] = (data[column].apply(self.parse_duration).apply(pd.Series))
            data = data.drop(column, axis="columns")
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise
        
    
    def process_multi_labels(self, data: pd.DataFrame, column: str, new_column: str) -> pd.DataFrame:
        try:
            data = data.copy()
            data[new_column] = data[column].str.split(",").apply(lambda x: [g.strip() for g in x])
            data = data.drop(column, axis="columns")
            mlb = MultiLabelBinarizer()

            encoded = mlb.fit_transform(data[new_column])
            encoded_df = pd.DataFrame(encoded, columns=[f"column_{c}" for c in mlb.classes_], index=data.index)
            data = pd.concat([data.drop(new_column, axis="columns"), encoded_df], axis="columns")
            return data
        except Exception as e:
            logging.error(f"Error: {e}")
            raise

