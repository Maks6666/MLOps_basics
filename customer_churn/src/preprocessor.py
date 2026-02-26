import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Tuple


class Preprocessor:
    def encode(self, x_train: pd.DataFrame, x_test: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_train = x_train.copy()
        x_test = x_test.copy()

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        encoder.fit(x_train[columns])

        train_encoded = encoder.transform(x_train[columns])
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(columns), index=x_train.index)
        x_train = pd.concat([x_train.drop(columns, axis='columns'), train_encoded_df], axis='columns')

        test_encoded = encoder.transform(x_test[columns])
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(columns), index=x_test.index)
        x_test = pd.concat([x_test.drop(columns, axis='columns'), test_encoded_df], axis='columns')

        return x_train, x_test
    
    def scale(self, x_train: pd.DataFrame, x_test: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
        x_train = x_train.copy()
        x_test = x_test.copy()

        scaler = MinMaxScaler()

        scaler.fit(x_train[columns])

        x_train[columns] = scaler.transform(x_train[columns])
        x_test[columns] = scaler.transform(x_test[columns])

        return x_train, x_test
