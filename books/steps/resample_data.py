import pandas as pd 
from zenml import step 
from typing import Tuple



@step
def resample_data(data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    x = data.drop(target, axis='columns')
    y = data[target]

    return x, y 