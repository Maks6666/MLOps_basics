import pandas as pd 
from zenml import step
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from typing import Tuple
import logging

@step
def split_data(data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        x = data.drop(target, axis='columns')
        y = data[target]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error: {e}")
        raise e