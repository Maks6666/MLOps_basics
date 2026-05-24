import pandas as pd

class Cleaner:
    def __init__(self, data) -> None:
        self.data = data.copy()

    def set_target(self, new_name: str, column_name: str) -> pd.DataFrame:
        self.data[new_name] = self.data[column_name] > self.data[column_name].median()
        return self

    def delete_columns(self, columns: list[str]) -> pd.DataFrame:
        self.data = self.data.drop(columns, axis='columns', errors='ignore')
        return self.data
    
    def set_freq(self, column: str) -> pd.DataFrame:
        freq = self.data[column].value_counts()
        self.data[f'freq_{column}'] = self.data[column].map(freq)
        self.data = self.data.drop(column, axis='columns', errors='ignore')
        return self.data
    
    def rename(self, column: str, values: dict) -> pd.DataFrame:
        self.data[column] = self.data[column].replace(values)
        return self.data
    
