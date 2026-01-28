import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler



class Preprocessor:
    def __init__(self):
        ...
    
    def fill_nan(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        data = data.copy()

        for column in columns:
            median = data[column].median()
            data[column] = data[column].fillna(median)

        return data
    
    def add_column(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        data['rooms_per_household'] = (data['total_rooms'] / data['households'])

        return data
    

class Scaler:
    def __init__(self):
        ...

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        columns = data.select_dtypes(include='number').columns

        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[columns])

        scaled_df = pd.DataFrame(scaled, columns=scaler.get_feature_names_out(columns), index=data.index)

        data = pd.concat([data.drop(columns, axis='columns'), scaled_df], axis='columns')

        return data