from src.splitter import Splitter
from zenml import step 
import pandas as pd 
from typing import Tuple


@step
def split(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = Splitter(test_size=0.2)
    x_train, x_test, y_train, y_test = splitter.split_data(data)
    return x_train, x_test, y_train, y_test