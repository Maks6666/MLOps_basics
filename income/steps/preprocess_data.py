from zenml import step
import pandas as pd
from typing import Tuple
import numpy as np

from src.preprocessor import Preprocessor



@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    preprocessor = Preprocessor()
    upd_data = preprocessor.replace(data=data, column='income', replacer={'<=50K': 0, '>50K': 1})

    value = 'No-HS'
    upd_data = preprocessor.replace(data=upd_data, column='education', replacer={'7th-8th': value, '10th': value, '1st-4th': value, '5th-6th': value, '9th': value, '11th': value, '12th': value})

    value = 'Married'
    upd_data = preprocessor.replace(data=upd_data, column='marital.status', replacer={'Married-civ-spouse': value, 'Married-spouse-absent': value, 'Married-AF-spouse': value})

    upd_data = preprocessor.replace(data=upd_data, column='occupation', replacer={'?': np.nan})
    upd_data = preprocessor.replace(data=upd_data, column='workclass', replacer={'?': np.nan})

    upd_data = preprocessor.replace(data=upd_data, column='sex', replacer={'Female': 0, 'Male': 1})

    upd_data = preprocessor.drop_data(upd_data, column='fnlwgt')

    upd_data = preprocessor.fill_nan(upd_data, column='occupation')
    upd_data = preprocessor.fill_nan(upd_data, column='workclass')

    return upd_data


