from zenml import step 
from src.preprocessing import Preprocessor
import pandas as pd


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    preprocessor = Preprocessor()
    
    upd_data = preprocessor.fill_nan(data, ['total_bedrooms'])
    upd_data = preprocessor.add_column(upd_data)

    return upd_data