from src.preprocessor import Preprocessor
import pandas as pd 
from zenml import step
from typing import Tuple

@step
def preprocess_data(x_train: pd.DataFrame, x_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = Preprocessor()

    columns_to_encode = ['InternetService', 'Contract', 'PaymentMethod']
    x_train, x_test = preprocessor.encode(x_train, x_test, columns_to_encode)

    columns_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    x_train, x_test = preprocessor.scale(x_train, x_test, columns_to_scale)

    return x_train, x_test