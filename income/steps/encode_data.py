import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple
from zenml import step


@step
def encode_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = x_train.select_dtypes(include="object").columns

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=40)

    encoded_train = encoder.fit_transform(x_train[categorical_cols])
    encoded_test = encoder.transform(x_test[categorical_cols])

    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols), index=x_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols), index=x_test.index)

    x_train = pd.concat([x_train.drop(categorical_cols, axis='columns'), encoded_train_df], axis='columns')
    x_test = pd.concat([x_test.drop(categorical_cols, axis='columns'), encoded_test_df], axis='columns')

    return x_train, x_test