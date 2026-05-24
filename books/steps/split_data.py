import pandas as pd
from zenml import step 
from sklearn.model_selection import train_test_split
from typing import Tuple



@step
def split_data(x: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    return  x_train, x_test, y_train, y_test