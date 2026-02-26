import pandas as pd 
from imblearn.under_sampling import RandomUnderSampler
from zenml import step
from typing import Tuple
@step
def resample_data(data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    x = data.drop(target, axis='columns')
    y = data[target]   

    rus = RandomUnderSampler()
    x, y = rus.fit_resample(x, y)

    return x, y