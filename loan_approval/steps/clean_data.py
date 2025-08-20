from zenml import step
from src.clean_data import CleanData, SplitData, DataTool
from typing_extensions import Annotated
from typing import Tuple
import pandas as pd
import logging

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]]:
    
    try:
        clean_data_strategy = CleanData()
        split_data_strategy = SplitData()

        cleaner = DataTool(data, clean_data_strategy)
        cleaned_data = cleaner.handle_data()

        splitter = DataTool(cleaned_data, split_data_strategy)
        x_train, x_test, y_train, y_test = splitter.handle_data()
        return x_train, x_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error: {e}")
        raise e

    