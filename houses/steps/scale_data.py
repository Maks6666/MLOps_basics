from zenml import step 
from src.preprocessing import Scaler

import pandas as pd 
from typing import Tuple


@step
def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = Scaler()

    x_train = scaler.scale(x_train)
    x_test = scaler.scale(x_test)
    
    return x_train, x_test