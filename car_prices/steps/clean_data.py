from zenml import step
from src.clean_data import CleanData, DivideData, DataTool
import pandas as pd
from typing_extensions import Annotated
from typing import Tuple
import logging

@step
def preprocess_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    
    try:
       
        tool = DataTool(data, CleanData)
        cleaned_data = tool.preprocess_data()

        tool = DataTool(cleaned_data, DivideData)
        x_train, x_test, y_train, y_test = tool.preprocess_data()

        return  x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e