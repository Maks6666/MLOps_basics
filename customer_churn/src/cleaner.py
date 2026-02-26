import pandas as pd 


class Cleaner:    
    def clean(self, data: pd.DataFrame, columns: list, replacer: dict) -> pd.DataFrame:
        data = data.copy()
        for col in columns:
            data[col] = data[col].replace(replacer)

        return data
        