from src.preprocessing import Preprocessor
from src.columns_extractor import Extractor
from zenml import step 
import pandas as pd

@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    extractor = Extractor()
    preprocessor = Preprocessor()

    data = extractor.extract(data)

    columns_to_drop = ['datetime', 'casual', 'registered']
    data = preprocessor.drop_values(data, columns_to_drop)

    return data 

