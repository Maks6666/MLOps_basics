from zenml import step 


from sklearn.preprocessing import OneHotEncoder
import pandas as pd 
from typing import Tuple


@step
def encode_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    columns = x_train.select_dtypes(include='object').columns

    encoded_train = encoder.fit_transform(x_train[columns])
    encoded_test = encoder.transform(x_test[columns])

    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(columns), index=x_train.index)
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(columns), index=x_test.index)

    x_train = pd.concat([x_train.drop(columns, axis='columns'), encoded_train_df], axis='columns')
    x_test = pd.concat([x_test.drop(columns, axis='columns'), encoded_test_df], axis='columns')

    return x_train, x_test