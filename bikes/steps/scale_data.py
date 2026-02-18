from zenml import step 
import pandas as pd
from typing import Tuple

from src.preprocessing import Scaler

@step
def scale_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaler = Scaler()
    columns_to_scale = ['temp', 'atemp', 'humidity', 'windspeed']

    scaler.fit(x_train, columns_to_scale)

    x_train = scaler.transform(x_train, columns_to_scale)
    x_test = scaler.transform(x_test, columns_to_scale)

    return x_train, x_test