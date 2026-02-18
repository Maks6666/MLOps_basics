import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from zenml import step

@step
def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data.drop('count', axis='columns')
    y = data['count']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test